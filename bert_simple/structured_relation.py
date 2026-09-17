"""Bounded technical pair features and a shared nonlinear relation scorer.

The pair feature is inherently pairwise, but it must never be materialized for
an unbounded query/key product.  The public ``features`` helper therefore
accepts already-aligned tensors, while ``forward_pairwise`` evaluates the
query/key product in bounded blocks.

The relation-training path is higher-level: ``forward_context_groups`` first
aggregates every member of a complete context group and then scores the group.
Technical pair arithmetic inside that method does not create a semantic
space or a lookup entry for a lexical token pair.
"""

import torch
from torch import nn


class ErrorComponentEraser(nn.Module):
    """Maintains a memory of error/distractor directions and orthogonally subtracts them."""

    def __init__(self, hidden_size: int = 32, num_slots: int = 8):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_slots = int(num_slots)
        self.register_buffer("error_directions", torch.zeros(self.num_slots, self.hidden_size))
        self.register_buffer("slot_active", torch.zeros(self.num_slots, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    def record_diff(self, h_err: torch.Tensor, h_true: torch.Tensor):
        """Record the unit direction from true target to error/distractor."""
        with torch.no_grad():
            diff = (h_err - h_true).detach().view(-1, self.hidden_size)
            if diff.numel() == 0:
                return
            mean_diff = diff.mean(dim=0)
            norm = mean_diff.norm().clamp_min(1e-6)
            unit_dir = mean_diff / norm
            idx = int(self.ptr.item())
            self.error_directions[idx] = unit_dir
            self.slot_active[idx] = True
            self.ptr[0] = (idx + 1) % self.num_slots

    def eliminate(self, h: torch.Tensor) -> torch.Tensor:
        """Orthogonal component subtraction: h_clean = h - sum_k (h . d_k) * d_k."""
        active = self.error_directions[self.slot_active]
        if active.shape[0] == 0:
            return h
        proj_coeffs = torch.matmul(h, active.t())
        proj = torch.matmul(proj_coeffs, active)
        return h - proj


class ExtractiveRelationPointer(nn.Module):
    """Difference-aware pointer network comparing candidate span representations against [MASK]."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.comparator = nn.Sequential(
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        nn.init.normal_(self.comparator[0].weight, std=0.02)
        nn.init.zeros_(self.comparator[0].bias)
        nn.init.zeros_(self.comparator[2].weight)
        nn.init.zeros_(self.comparator[2].bias)

    def forward(self, h_mask: torch.Tensor, cand_reps: torch.Tensor) -> torch.Tensor:
        """
        h_mask: [..., hidden_size]
        cand_reps: [..., num_cands, hidden_size]
        Returns: [..., num_cands]
        """
        if h_mask.ndim == cand_reps.ndim - 1:
            h_mask_expanded = h_mask.unsqueeze(-2).expand_as(cand_reps)
        else:
            h_mask_expanded = h_mask
        diff = (h_mask_expanded - cand_reps).abs()
        prod = h_mask_expanded * cand_reps
        feats = torch.cat([h_mask_expanded, cand_reps, diff, prod], dim=-1)
        return self.comparator(feats).squeeze(-1)


class Structured3DRelationFFN(nn.Module):
    MAX_PAIR_BLOCK_ELEMENTS = 65536

    def __init__(self, hidden_size=32):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.fc1 = nn.Linear(18, hidden_size)
        self.act = nn.GELU()
        self.eraser = ErrorComponentEraser(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def network(self):
        return nn.Sequential(self.fc1, self.act, self.fc2)

    @staticmethod
    def features(q, k):
        if q.shape[-1] != 3 or k.shape[-1] != 3:
            raise ValueError("q and k must have three coordinates")
        if q.shape != k.shape:
            raise ValueError(
                "q and k must have the same shape; use forward_pairwise for "
                "different query/key lengths"
            )
        outer = torch.matmul(q.unsqueeze(-1), k.unsqueeze(-2)).flatten(-2)
        return torch.cat((q, k, (q - k).abs(), outer), dim=-1)

    def forward(self, q, k):
        feat = self.features(q, k)
        h = self.act(self.fc1(feat))
        h_clean = self.eraser.eliminate(h)
        return self.fc2(h_clean).squeeze(-1)

    def forward_pairwise(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_chunk_size: int = 32,
        key_chunk_size: int = 32,
    ) -> torch.Tensor:
        """Score a query/key grid with bounded pair-feature blocks.

        ``query`` and ``key`` have shape ``[..., length, 3]`` and share the
        leading dimensions.  The returned tensor is ``[..., Q, K]``.  The
        temporary 18D feature tensor is at most
        ``[..., query_chunk_size, key_chunk_size, 18]`` instead of
        ``[..., Q, K, 18]``.
        """
        if query.shape[-1] != 3 or key.shape[-1] != 3:
            raise ValueError("query and key must have three coordinates")
        if query.ndim < 2 or key.ndim < 2:
            raise ValueError("query and key must have shape [..., length, 3]")
        if query.shape[:-2] != key.shape[:-2]:
            raise ValueError("query and key leading dimensions must match")
        if query_chunk_size < 1 or key_chunk_size < 1:
            raise ValueError("pair chunk sizes must be positive")

        query_parts = []
        for query_start in range(0, query.shape[-2], query_chunk_size):
            query_end = min(query_start + query_chunk_size, query.shape[-2])
            query_block = query[..., query_start:query_end, :]
            key_parts = []
            for key_start in range(0, key.shape[-2], key_chunk_size):
                key_end = min(key_start + key_chunk_size, key.shape[-2])
                key_block = key[..., key_start:key_end, :]
                if (
                    (query_end - query_start) * (key_end - key_start)
                    > self.MAX_PAIR_BLOCK_ELEMENTS
                ):
                    raise ValueError(
                        "pair block is too large; reduce query/key chunk sizes"
                    )

                # These are views; the only large allocation in this block is
                # the bounded feature tensor created by ``features``.  The
                # outer product itself uses a 3x3 matmul.
                query_view = query_block.unsqueeze(-2)
                key_view = key_block.unsqueeze(-3)
                pair_shape = query_view.shape[:-2] + (
                    key_view.shape[-2],
                    query_view.shape[-1],
                )
                query_aligned = query_view.expand(pair_shape)
                key_aligned = key_view.expand(pair_shape)
                key_parts.append(self(query_aligned, key_aligned))
            query_parts.append(torch.cat(key_parts, dim=-1))
        return torch.cat(query_parts, dim=-2)


class StructuredRelationScores(nn.Module):
    """Hookable FFN-only bias [B,H,T,T], summed over routed spaces.

    Shared across layers/heads/spaces. Pair scoring is blockwise in both query
    and key dimensions. ``forward_pairs`` is the memory-safe training path:
    it computes only selected query/key positions instead of a full ``T x T``
    matrix. ``forward_context_groups`` is the relation-training path: it pools all
    members of each context group before the FFN, so a relation is not learned
    as a collection of independent token-pair facts. Scores are in local-score
    units, before the outer dynamic-QK scale.
    """

    def __init__(self, hidden_size=32, scale=0.1, chunk_size=32, space_chunk_size=32):
        super().__init__()
        if scale <= 0 or chunk_size < 1 or space_chunk_size < 1:
            raise ValueError("scale and chunk sizes must be positive")
        self.ffn = Structured3DRelationFFN(hidden_size)
        self.scale = float(scale)
        self.chunk_size = int(chunk_size)
        self.space_chunk_size = int(space_chunk_size)

    @staticmethod
    def _project(selected: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Apply ``[head, 3, 3]`` matrices to ``[batch, token, 3]`` states."""
        projected = torch.matmul(
            selected.unsqueeze(-2).unsqueeze(-2),
            matrix.unsqueeze(0).unsqueeze(0),
        ).squeeze(-2)
        return projected.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _active_spaces(
        routes: torch.Tensor,
        relation_count: int,
        active_spaces=None,
    ) -> torch.Tensor:
        if active_spaces is None:
            active = (routes.detach().abs().sum((0, 1)) > 0).nonzero(
                as_tuple=False
            ).flatten()
        else:
            active = torch.as_tensor(
                list(active_spaces), dtype=torch.long, device=routes.device
            )
        return active[(active >= 0) & (active < relation_count)]

    def forward_pairs(
        self,
        layer_index,
        hidden,
        routes,
        triples,
        q_bank,
        k_bank,
        query_positions,
        key_positions,
        active_spaces=None,
    ):
        """Return scores only for selected query/key positions.

        This is retained for attention/diagnostic callers that explicitly need
        selected token edges.  The current relation-sidecar loss uses
        ``forward_context_groups`` instead, so it does not turn those edges
        into independent semantic facts.
        """
        del layer_index
        if hidden.ndim != 3 or routes.ndim != 3:
            raise ValueError("hidden and routes must be [batch, sequence, ...]")
        if routes.shape[:2] != hidden.shape[:2]:
            raise ValueError("hidden and routes must share batch/sequence axes")
        query_positions = torch.as_tensor(
            list(query_positions), dtype=torch.long, device=hidden.device
        )
        key_positions = torch.as_tensor(
            list(key_positions), dtype=torch.long, device=hidden.device
        )
        if query_positions.numel() == 0 or key_positions.numel() == 0:
            raise ValueError("query_positions and key_positions must be non-empty")
        if query_positions.min() < 0 or query_positions.max() >= hidden.shape[1]:
            raise ValueError("query position is outside hidden sequence")
        if key_positions.min() < 0 or key_positions.max() >= hidden.shape[1]:
            raise ValueError("key position is outside hidden sequence")

        active = self._active_spaces(routes, len(triples), active_spaces)
        result = hidden.new_zeros(
            hidden.shape[0],
            q_bank.shape[0],
            query_positions.numel(),
            key_positions.numel(),
        )
        query_hidden = hidden.index_select(1, query_positions)
        key_hidden = hidden.index_select(1, key_positions)
        query_routes = routes.index_select(1, query_positions)
        key_routes = routes.index_select(1, key_positions)

        for space in active.tolist():
            dimensions = list(triples[space])
            query = self._project(
                query_hidden[..., dimensions], q_bank[:, space]
            )
            key = self._project(key_hidden[..., dimensions], k_bank[:, space])
            local = self.ffn.forward_pairwise(
                query,
                key,
                query_chunk_size=self.chunk_size,
                key_chunk_size=self.chunk_size,
            )
            pair_weights = torch.matmul(
                query_routes[..., space].unsqueeze(-1),
                key_routes[..., space].unsqueeze(-2),
            ).unsqueeze(1)
            result = result + local * pair_weights
        return result * self.scale

    def forward_context_groups(
        self,
        layer_index,
        hidden,
        routes,
        triples,
        q_bank,
        k_bank,
        query_positions,
        context_groups,
        active_spaces=None,
    ):
        """Score a query against complete context groups.

        ``context_groups`` is a sequence of position sequences.  Every
        position in a group participates in a route-weighted mean before the
        18D FFN is called.  No member is selected by ``topk`` and no token pair
        gets its own dimension triple.  The returned tensor has shape
        ``[batch, heads, query_count, group_count]``.

        This is intentionally separate from ``forward_pairs``.  Attention still
        needs pairwise logits, but the sidecar's supervision must answer the
        higher-level question "does this whole context explain the masked
        token?" rather than rewarding isolated edges such as ``是 || 的``.
        """
        del layer_index
        if hidden.ndim != 3 or routes.ndim != 3:
            raise ValueError("hidden and routes must be [batch, sequence, ...]")
        if routes.shape[:2] != hidden.shape[:2]:
            raise ValueError("hidden and routes must share batch/sequence axes")
        query_positions = torch.as_tensor(
            list(query_positions), dtype=torch.long, device=hidden.device
        )
        if query_positions.numel() == 0:
            raise ValueError("query_positions must be non-empty")
        if query_positions.min() < 0 or query_positions.max() >= hidden.shape[1]:
            raise ValueError("query position is outside hidden sequence")
        groups = [
            torch.as_tensor(list(group), dtype=torch.long, device=hidden.device)
            for group in context_groups
        ]
        if not groups or any(group.numel() == 0 for group in groups):
            raise ValueError("context_groups must contain non-empty groups")
        for group in groups:
            if group.min() < 0 or group.max() >= hidden.shape[1]:
                raise ValueError("context group position is outside hidden sequence")

        active = self._active_spaces(routes, len(triples), active_spaces)
        result = hidden.new_zeros(
            hidden.shape[0],
            q_bank.shape[0],
            query_positions.numel(),
            len(groups),
        )
        group_lengths = torch.tensor(
            [group.numel() for group in groups],
            dtype=hidden.dtype,
            device=hidden.device,
        ).view(1, len(groups), 1)
        max_group_length = max(int(group.numel()) for group in groups)
        group_positions = torch.zeros(
            len(groups), max_group_length, dtype=torch.long, device=hidden.device
        )
        group_mask = torch.zeros(
            len(groups), max_group_length, dtype=torch.bool, device=hidden.device
        )
        for group_index, group in enumerate(groups):
            length = group.numel()
            group_positions[group_index, :length] = group
            group_mask[group_index, :length] = True

        # The old implementation built one autograd subgraph per candidate
        # space.  With 1500 candidates that retained thousands of small q/k
        # activations until backward.  Batch candidates into a bounded block;
        # the only feature expansion is now [B,H,Q,G,C,18] for this block.
        for start in range(0, active.numel(), self.space_chunk_size):
            space_indices = active[start : start + self.space_chunk_size]
            dimensions = torch.as_tensor(
                [triples[int(index)] for index in space_indices.tolist()],
                dtype=torch.long,
                device=hidden.device,
            ).reshape(-1)
            candidate_count = int(space_indices.numel())
            query_hidden = hidden.index_select(1, query_positions)
            query_selected = query_hidden.index_select(-1, dimensions).reshape(
                hidden.shape[0], query_positions.numel(), candidate_count, 3
            )
            q_matrices = q_bank.index_select(1, space_indices)
            query = torch.matmul(
                query_selected.unsqueeze(1).unsqueeze(-2),
                q_matrices.unsqueeze(0).unsqueeze(2),
            ).squeeze(-2)

            flat_positions = group_positions.reshape(-1)
            group_hidden = hidden.index_select(1, flat_positions).reshape(
                hidden.shape[0], len(groups), max_group_length, hidden.shape[-1]
            )
            group_selected = group_hidden.index_select(-1, dimensions).reshape(
                hidden.shape[0], len(groups), max_group_length, candidate_count, 3
            )
            k_matrices = k_bank.index_select(1, space_indices)
            group_key = torch.matmul(
                group_selected.unsqueeze(1).unsqueeze(-2),
                k_matrices.unsqueeze(0).unsqueeze(2).unsqueeze(3),
            ).squeeze(-2)

            group_routes = routes.index_select(1, flat_positions).reshape(
                hidden.shape[0], len(groups), max_group_length, routes.shape[-1]
            ).index_select(-1, space_indices)
            group_routes = group_routes * group_mask.view(
                1, len(groups), max_group_length, 1
            ).to(dtype=group_routes.dtype)
            denominator = group_routes.sum(dim=2)
            weighted = (
                group_key * group_routes.unsqueeze(1).unsqueeze(-1)
            ).sum(dim=3) / denominator.clamp_min(1e-9).unsqueeze(1).unsqueeze(-1)
            valid_members = group_mask.view(
                1, 1, len(groups), max_group_length, 1, 1
            ).to(dtype=group_key.dtype)
            uniform = (
                group_key * valid_members
            ).sum(dim=3) / group_lengths.unsqueeze(1).unsqueeze(-1)
            pooled = torch.where(
                denominator.gt(1e-9).view(
                    hidden.shape[0], 1, len(groups), candidate_count, 1
                ),
                weighted,
                uniform,
            )

            query_aligned = query.unsqueeze(3).expand(
                hidden.shape[0],
                q_bank.shape[0],
                query_positions.numel(),
                len(groups),
                candidate_count,
                3,
            )
            key_aligned = pooled.unsqueeze(2).expand_as(query_aligned)
            local = self.ffn(query_aligned, key_aligned)
            query_routes = routes.index_select(1, query_positions).index_select(
                -1, space_indices
            )
            route_weight = (
                query_routes.unsqueeze(2) * (denominator / group_lengths).unsqueeze(1)
            )
            result = result + (local * route_weight.unsqueeze(1)).sum(dim=-1)
        return result * self.scale

    def forward(
        self,
        layer_index,
        hidden,
        routes,
        triples,
        q_bank,
        k_bank,
        active_positions=None,
    ):
        batch, length, _ = hidden.shape
        result = hidden.new_zeros(batch, q_bank.shape[0], length, length)
        active = self._active_spaces(routes, len(triples))
        if active.numel() == 0:
            return result

        if active_positions is not None:
            active_pos = torch.as_tensor(
                list(active_positions), dtype=torch.long, device=hidden.device
            )
            active_pos = active_pos[(active_pos >= 0) & (active_pos < length)]
            if active_pos.numel() == 0:
                return result

            # Sparse FFN: evaluate 18D features only on candidate entity/slot positions
            q_selected_tokens = hidden.index_select(1, active_pos)
            k_selected_tokens = hidden.index_select(1, active_pos)
            q_routes = routes.index_select(1, active_pos)
            k_routes = routes.index_select(1, active_pos)
            k_len = active_pos.numel()

            sub_accum = hidden.new_zeros(batch, q_bank.shape[0], k_len, k_len)
            for space in active.tolist():
                if space >= len(triples):
                    continue
                dims = list(triples[space])
                q = self._project(q_selected_tokens[..., dims], q_bank[:, space])
                k = self._project(k_selected_tokens[..., dims], k_bank[:, space])
                weights_q = q_routes[..., space]
                weights_k = k_routes[..., space]
                scores = self.ffn.forward_pairwise(
                    q,
                    k,
                    query_chunk_size=self.chunk_size,
                    key_chunk_size=self.chunk_size,
                )
                pair_weights = torch.matmul(
                    weights_q.unsqueeze(-1), weights_k.unsqueeze(-2)
                ).unsqueeze(1)
                sub_accum = sub_accum + scores * pair_weights

            sub_scaled = sub_accum * self.scale
            result = result.index_put(
                (slice(None), slice(None), active_pos[:, None], active_pos[None, :]),
                sub_scaled,
            )
            return result
        else:
            for space in active.tolist():
                if space >= len(triples):
                    continue
                selected = hidden[..., list(triples[space])]
                q = self._project(selected, q_bank[:, space])
                k = self._project(selected, k_bank[:, space])
                weights = routes[..., space]
                scores = self.ffn.forward_pairwise(
                    q,
                    k,
                    query_chunk_size=self.chunk_size,
                    key_chunk_size=self.chunk_size,
                )
                pair_weights = torch.matmul(
                    weights.unsqueeze(-1), weights.unsqueeze(-2)
                ).unsqueeze(1)
                result = result + scores * pair_weights
            return result * self.scale
