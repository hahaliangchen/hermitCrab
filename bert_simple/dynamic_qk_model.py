"""Context-routed dynamic 3D Q/K attention for the fact-memory model.

The older experiment kept a separate dynamic-space BERT implementation in
``examples/bert_mlm_dynamic_word_spaces.py``.  This module ports its important
path into the current fact-memory stack:

* a contextual router chooses relation spaces on every forward pass;
* every attention head owns a relation-specific 3x3 Q/K transform;
* the routed local score is added inside self-attention, before softmax;
* the existing MLM-side relation matrix remains available as a separate
  additive fact-memory component.

The low-level adapter keeps ``relation_triples`` as a compatibility name for
the three-coordinate channel list.  In the current contextual sidecar that
list is a fixed shared bank exposed to every valid position; it is not a
sample-specific fact or token-pair candidate set.  The route within the bank
is computed from the current contextual hidden states.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar_attribute_filter import GrammarAttributeFilter
from .local_relation_adapter_model import _normalize_triple
from .local_relation_adapter_model_v2 import (
    DEFAULT_RELATION_SCORE_SCALE,
    ScaledLocalRelationBilinearAdapter,
    ScaledLocalRelationMarginBertForMaskedLM,
)
from .model import BertConfig, _build_relative_position_bias
from .tokenizer import SimpleBertTokenizer
from .structured_relation import StructuredRelationScores, ExtractiveRelationPointer


Triple = Tuple[int, int, int]
DEFAULT_DYNAMIC_QK_SCORE_SCALE = 0.5
DEFAULT_ROUTE_START_LAYER = 1
DEFAULT_ROUTE_DIM = 32
# Keep the temporary [B,H,T,T,C] Q/K block comfortably bounded when the
# current shared bank uses up to 1500 relation candidates.
DEFAULT_RELATION_SPACE_CHUNK_SIZE = 128


class DimensionFusion(nn.Module):
    """Dense cross-channel mixing for the fused dimensions (>= 1/3 of hidden size)."""
    def __init__(self, hidden_size: int = 64, fusion_dim: int = 24):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.fusion_dim = int(fusion_dim)
        self.comb_dim = max(1, self.hidden_size - self.fusion_dim)
        self.mixer = nn.Sequential(
            nn.Linear(self.hidden_size, self.fusion_dim),
            nn.GELU(),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            return hidden_states
        comb = hidden_states[..., :self.comb_dim]
        raw_fuse = hidden_states[..., self.comb_dim:]
        fused = raw_fuse + self.mixer(hidden_states)
        return torch.cat((comb, fused), dim=-1)


class DynamicQKRelationBank(nn.Module):
    """Per-layer, per-head relation-specific 3D Q/K transforms."""

    def __init__(
        self,
        relation_triples: Sequence[Sequence[int]],
        num_layers: int,
        num_heads: int,
        route_dim: int,
        relation_space_chunk_size: int = DEFAULT_RELATION_SPACE_CHUNK_SIZE,
        hidden_size: int = 64,
        fusion_dim: Optional[int] = None,
    ):
        super().__init__()
        normalized = tuple(_normalize_triple(triple) for triple in relation_triples)
        self.relation_triples: Tuple[Triple, ...] = normalized
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.route_dim = int(route_dim)
        self.space_dim = 3
        self.max_spaces = max(1, len(normalized))
        self.relation_space_chunk_size = int(relation_space_chunk_size)
        actual_fusion_dim = int(fusion_dim) if fusion_dim is not None else max(1, int(round(hidden_size * 0.375)))
        self.fusion_dim = actual_fusion_dim
        self.dimension_fusion = DimensionFusion(hidden_size, actual_fusion_dim) if actual_fusion_dim > 0 else nn.Identity()
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("dynamic Q/K bank needs positive layer and head counts")
        if self.route_dim < 1:
            raise ValueError("route_dim must be positive")
        if self.relation_space_chunk_size < 1:
            raise ValueError("relation_space_chunk_size must be positive")

        self.register_buffer(
            "triple_indices",
            torch.tensor(normalized, dtype=torch.long),
            persistent=False,
        )

        self.q_matrix = nn.Parameter(
            torch.empty(
                self.num_layers,
                self.num_heads,
                self.max_spaces,
                self.space_dim,
                self.space_dim,
            )
        )
        self.k_matrix = nn.Parameter(
            torch.empty(
                self.num_layers,
                self.num_heads,
                self.max_spaces,
                self.space_dim,
                self.space_dim,
            )
        )
        self.space_descriptors = nn.Parameter(
            torch.empty(self.max_spaces, self.route_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            eye = torch.eye(
                self.space_dim,
                device=self.q_matrix.device,
                dtype=self.q_matrix.dtype,
            ).view(1, 1, 1, self.space_dim, self.space_dim)
            self.q_matrix.copy_(eye.expand_as(self.q_matrix))
            self.k_matrix.copy_(eye.expand_as(self.k_matrix))
            self.q_matrix.add_(torch.randn_like(self.q_matrix) * 0.01)
            self.k_matrix.add_(torch.randn_like(self.k_matrix) * 0.01)
            self.space_descriptors.normal_(mean=0.0, std=0.02)

    def local_scores(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        route_weights: torch.Tensor,
        active_positions: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Return routed local attention scores with shape ``[B,H,T,T]``.

        Unlike the MLM-side adapter, the relation dimensions are applied to
        every query/key position.  The context router supplies a separate
        weight for each position, so the same relation can be emphasized for
        one token and suppressed for another in the same sequence.
        """
        if not 0 <= int(layer_index) < self.num_layers:
            raise ValueError("layer_index is outside the dynamic Q/K bank")
        if route_weights.ndim != 3:
            raise ValueError("route_weights must have shape [batch, sequence, space]")
        if route_weights.shape[-1] != self.max_spaces:
            raise ValueError(
                "route_weights width must match the dynamic Q/K space count"
            )

        hidden_states = self.dimension_fusion(hidden_states)
        batch_size, sequence_length, _ = hidden_states.shape
        scores = hidden_states.new_zeros(
            batch_size, self.num_heads, sequence_length, sequence_length
        )
        # Candidate masks are fixed for a forward pass.  Detaching this
        # selection avoids creating a discontinuous autograd path while the
        # actual scores below remain fully differentiable.
        active = (
            route_weights.detach().abs().sum(dim=(0, 1)) > 0.0
        ).nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            return scores

        q_bank = self.q_matrix[int(layer_index)]
        k_bank = self.k_matrix[int(layer_index)]
        for relation_indices in active.split(self.relation_space_chunk_size):
            if relation_indices.numel() == 0:
                continue
            dimensions = self.triple_indices.index_select(0, relation_indices).reshape(-1)
            chunk_size = relation_indices.numel()
            selected_hidden = hidden_states.index_select(-1, dimensions).reshape(
                batch_size, sequence_length, chunk_size, self.space_dim
            )

            # [B, 1, T, C, 1, 3] @ [1, H, 1, C, 3, 3]
            # produces [B, H, T, C, 3].  C is bounded by the relation-space
            # chunk size instead of materializing every active space at once.
            left = selected_hidden.unsqueeze(1).unsqueeze(-2)
            q_matrices = q_bank.index_select(1, relation_indices)
            k_matrices = k_bank.index_select(1, relation_indices)
            q = torch.matmul(
                left,
                q_matrices.unsqueeze(0).unsqueeze(2),
            ).squeeze(-2).permute(0, 1, 3, 2, 4)
            k = torch.matmul(
                left,
                k_matrices.unsqueeze(0).unsqueeze(2),
            ).squeeze(-2).permute(0, 1, 3, 2, 4)
            pair_scores = torch.matmul(q, k.transpose(-1, -2))

            weights = route_weights.index_select(-1, relation_indices).transpose(1, 2)
            pair_weights = torch.matmul(
                weights.unsqueeze(-1), weights.unsqueeze(-2)
            ).unsqueeze(1)
            scores = scores + (pair_scores * pair_weights).sum(dim=2)
        scores = scores / math.sqrt(float(self.space_dim))
        if hasattr(self, "structured_scores"):
            scores = scores + self.structured_scores(
                layer_index,
                hidden_states,
                route_weights,
                self.relation_triples,
                q_bank,
                k_bank,
                active_positions=active_positions,
            )
        return scores


class ContextualRelationRouter(nn.Module):
    """Compute per-position route weights over relation-space descriptors."""

    def __init__(self, hidden_size: int, route_dim: int, initializer_range: float):
        super().__init__()
        self.query = nn.Linear(hidden_size, route_dim)
        self.key = nn.Linear(route_dim, route_dim, bias=False)
        with torch.no_grad():
            self.query.weight.normal_(mean=0.0, std=initializer_range)
            self.query.bias.zero_()
            self.key.weight.normal_(mean=0.0, std=initializer_range)

    def forward(
        self,
        contextual_hidden: torch.Tensor,
        space_descriptors: torch.Tensor,
        candidate_mask: torch.Tensor,
        space_chunk_size: int = DEFAULT_RELATION_SPACE_CHUNK_SIZE,
    ) -> torch.Tensor:
        if candidate_mask.ndim != 3:
            raise ValueError("candidate_mask must have shape [batch, sequence, space]")
        if candidate_mask.shape[-1] != space_descriptors.shape[0]:
            raise ValueError("candidate_mask and descriptors have different widths")
        if space_chunk_size < 1:
            raise ValueError("space_chunk_size must be positive")
        query = F.normalize(self.query(contextual_hidden), dim=-1)
        # The bank may contain up to the current 1500 shared descriptors.
        # Project and match it in bounded chunks; the final logits are still
        # concatenated because
        # softmax must normalize over the complete candidate set, but no large
        # descriptor activation is kept at once.
        logits_parts = []
        for start in range(0, space_descriptors.shape[0], space_chunk_size):
            key = F.normalize(
                self.key(space_descriptors[start:start + space_chunk_size]),
                dim=-1,
            )
            logits_parts.append(torch.matmul(query, key.transpose(0, 1)))
        logits = torch.cat(logits_parts, dim=-1)
        candidate_mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
        has_candidate = candidate_mask.any(dim=-1)
        safe_mask = candidate_mask.clone()
        # A fallback prevents softmax NaNs for padding/empty candidate rows;
        # multiplication by the original mask makes the returned route zero
        # for those rows.
        safe_mask[..., 0] = safe_mask[..., 0] | ~has_candidate
        logits = logits.masked_fill(~safe_mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        return weights * candidate_mask.to(dtype=weights.dtype)


class DynamicQKRelationBilinearAdapter(ScaledLocalRelationBilinearAdapter):
    """MLM relation adapter plus a contextual dynamic-Q/K bank."""

    def __init__(
        self,
        relation_triples: Iterable[Sequence[int]],
        hidden_size: int,
        mask_token_id: int,
        num_layers: int,
        num_heads: int,
        route_dim: int,
        relation_score_scale: float = DEFAULT_RELATION_SCORE_SCALE,
        dynamic_qk_score_scale: float = DEFAULT_DYNAMIC_QK_SCORE_SCALE,
        initializer_range: float = 0.02,
        fusion_dim: Optional[int] = None,
    ):
        super().__init__(
            relation_triples,
            hidden_size,
            mask_token_id,
            relation_score_scale=relation_score_scale,
        )
        triples = self.relation_triples.detach().cpu().tolist()
        self.dynamic_qk = DynamicQKRelationBank(
            triples,
            num_layers=num_layers,
            num_heads=num_heads,
            route_dim=route_dim,
            hidden_size=hidden_size,
            fusion_dim=fusion_dim,
        )
        self.router = ContextualRelationRouter(
            hidden_size,
            route_dim,
            initializer_range,
        )
        self.dynamic_qk_score_scale = float(dynamic_qk_score_scale)
        if self.dynamic_qk_score_scale <= 0.0:
            raise ValueError("dynamic_qk_score_scale must be positive")

    def route_weights(
        self,
        contextual_hidden: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.router(
            contextual_hidden,
            self.dynamic_qk.space_descriptors,
            candidate_mask,
            space_chunk_size=self.dynamic_qk.relation_space_chunk_size,
        )

    def dynamic_local_scores(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        route_weights: torch.Tensor,
        active_positions: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        return self.dynamic_qk.local_scores(
            layer_index,
            hidden_states,
            route_weights,
            active_positions=active_positions,
        )

    def _relation_axis(self, parameter: torch.nn.Parameter) -> Optional[int]:
        if parameter is self.relation_matrix:
            return 0
        if parameter is self.dynamic_qk.space_descriptors:
            return 0
        if parameter is self.dynamic_qk.q_matrix:
            return 2
        if parameter is self.dynamic_qk.k_matrix:
            return 2
        # Router weights are shared across all relation spaces.  They are
        # deliberately left unmasked and receive new+replay gradients.
        return None

    def mask_gradients(
        self,
        gradients: Sequence[torch.Tensor],
        active_indices: Sequence[int],
    ) -> List[torch.Tensor]:
        parameters = list(self.parameters())
        if len(parameters) != len(gradients):
            raise ValueError("adapter parameters and gradients have different lengths")
        active_set = {
            int(index)
            for index in active_indices
            if 0 <= int(index) < self.relation_count
        }
        result: List[torch.Tensor] = []
        for parameter, gradient in zip(parameters, gradients):
            value = gradient.detach().clone()
            axis = self._relation_axis(parameter)
            if axis is None:
                result.append(value)
                continue
            mask = torch.zeros(
                parameter.shape[axis], dtype=value.dtype, device=value.device
            )
            if active_set:
                mask[list(active_set)] = 1.0
            view_shape = [1] * value.ndim
            view_shape[axis] = mask.numel()
            result.append(value * mask.view(view_shape))
        return result

    def relation_gradient_vector(
        self,
        gradients: Sequence[torch.Tensor],
        relation_index: int,
    ) -> torch.Tensor:
        parameters = list(self.parameters())
        pieces = []
        index = int(relation_index)
        for parameter, gradient in zip(parameters, gradients):
            axis = self._relation_axis(parameter)
            if axis is None or index >= parameter.shape[axis]:
                continue
            pieces.append(gradient.select(axis, index).reshape(-1))
        if not pieces:
            return torch.zeros(0, device=gradients[0].device)
        return torch.cat(pieces)

    def scale_relation_gradient(
        self,
        gradients: Sequence[torch.Tensor],
        relation_index: int,
        scale: float,
    ) -> List[torch.Tensor]:
        parameters = list(self.parameters())
        index = int(relation_index)
        result = [gradient.detach().clone() for gradient in gradients]
        for parameter, gradient in zip(parameters, result):
            axis = self._relation_axis(parameter)
            if axis is None or index >= parameter.shape[axis]:
                continue
            selector = [slice(None)] * gradient.ndim
            selector[axis] = index
            gradient[tuple(selector)] *= float(scale)
        return result


class DynamicQKSelfAttention(nn.Module):
    """Standard BERT attention with a routed local Q/K score before softmax."""

    def __init__(
        self,
        original: nn.Module,
        relation_adapter: DynamicQKRelationBilinearAdapter,
        layer_index: int,
    ):
        super().__init__()
        self.qkv = original.qkv
        self.out_proj = original.out_proj
        self.dropout = original.dropout
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.all_head_size = original.all_head_size
        self.relative_position_max_distance = original.relative_position_max_distance
        self.relative_position_bias = original.relative_position_bias
        self.layer_index = int(layer_index)
        object.__setattr__(self, "relation_adapter", relation_adapter)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_weights: Optional[torch.Tensor] = None,
        active_positions: Optional[Sequence[int]] = None,
        sparse_gate_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._transpose_for_scores(query)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        relative_bias = _build_relative_position_bias(
            self.relative_position_bias,
            self.relative_position_max_distance,
            hidden_states.size(1),
            hidden_states.device,
            attn_scores.dtype,
        )
        if relative_bias is not None:
            attn_scores = attn_scores + relative_bias
        if route_weights is not None:
            local_scores = self.relation_adapter.dynamic_local_scores(
                self.layer_index,
                hidden_states,
                route_weights,
                active_positions=active_positions,
            )
            attn_scores = attn_scores + (
                self.relation_adapter.dynamic_qk_score_scale * local_scores
            )
        if sparse_gate_mask is not None:
            attn_scores = attn_scores + sparse_gate_mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:2], self.all_head_size)
        return self.out_proj(context), attn_probs


class DynamicQKLocalRelationMarginBertForMaskedLM(
    ScaledLocalRelationMarginBertForMaskedLM
):
    """Fact-memory MLM with contextual relation routing inside attention."""

    def __init__(
        self,
        config: BertConfig,
        tokenizer: SimpleBertTokenizer,
        relation_triples: Iterable[Sequence[int]],
        frequency_prior: torch.Tensor,
        frequency_class_scales: torch.Tensor,
        attribute_filter: Optional[GrammarAttributeFilter] = None,
        attribute_bias_scale: float = 1.5,
        constrained_frequency_gate: float = 0.25,
        margin_value: float = 2.0,
        relation_score_scale: float = DEFAULT_RELATION_SCORE_SCALE,
        dynamic_qk_score_scale: float = DEFAULT_DYNAMIC_QK_SCORE_SCALE,
        route_start_layer: int = DEFAULT_ROUTE_START_LAYER,
        route_dim: int = DEFAULT_ROUTE_DIM,
        relation_ffn_hidden_size: int = 0,
        relation_ffn_scale: float = 0.1,
        relation_ffn_chunk_size: int = 32,
        fusion_dim: Optional[int] = None,
        enable_grammar_sparse_gate: bool = False,
    ):
        relation_triples = list(relation_triples)
        super().__init__(
            config,
            tokenizer,
            relation_triples=relation_triples,
            frequency_prior=frequency_prior,
            frequency_class_scales=frequency_class_scales,
            attribute_filter=attribute_filter,
            attribute_bias_scale=attribute_bias_scale,
            constrained_frequency_gate=constrained_frequency_gate,
            margin_value=margin_value,
            relation_score_scale=relation_score_scale,
        )
        if route_start_layer < 1:
            raise ValueError("route_start_layer must leave one ordinary BERT layer")
        if config.num_hidden_layers <= route_start_layer:
            raise ValueError("route_start_layer must leave a dynamic Q/K layer")
        self.relation_adapter = DynamicQKRelationBilinearAdapter(
            relation_triples,
            hidden_size=config.hidden_size,
            mask_token_id=config.mask_token_id,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            route_dim=route_dim,
            relation_score_scale=relation_score_scale,
            dynamic_qk_score_scale=dynamic_qk_score_scale,
            initializer_range=config.initializer_range,
            fusion_dim=fusion_dim,
        )
        self.route_start_layer = int(route_start_layer)
        self.route_dim = int(route_dim)
        if relation_ffn_hidden_size < 0:
            raise ValueError("relation_ffn_hidden_size must be nonnegative")
        self.relation_ffn_hidden_size = int(relation_ffn_hidden_size)
        self.relation_ffn_scale = float(relation_ffn_scale)
        self.relation_ffn_chunk_size = int(relation_ffn_chunk_size)
        self.fusion_dim = getattr(self.relation_adapter.dynamic_qk, "fusion_dim", None)
        if self.relation_ffn_hidden_size:
            self.relation_adapter.dynamic_qk.structured_scores = StructuredRelationScores(
                self.relation_ffn_hidden_size, self.relation_ffn_scale,
                self.relation_ffn_chunk_size,
            )
        self.dynamic_qk_score_scale = float(dynamic_qk_score_scale)
        self.enable_grammar_sparse_gate = bool(enable_grammar_sparse_gate)
        self._last_route_weights: Optional[torch.Tensor] = None
        for layer_index in range(route_start_layer, config.num_hidden_layers):
            layer = self.bert.encoder.layer[layer_index]
            layer.attention.self_attn = DynamicQKSelfAttention(
                layer.attention.self_attn,
                self.relation_adapter,
                layer_index,
            )
        self.pointer_head = ExtractiveRelationPointer(config.hidden_size)

    def encode_hidden(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        relation_triples: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        sequence_output, _, _ = self._encode_sequence(
            input_ids,
            token_type_ids,
            attention_mask,
            relation_triples=relation_triples,
        )
        return sequence_output

    def pointer_scores(
        self,
        sequence_output: torch.Tensor,
        mask_pos: int,
        candidate_spans: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        """Score candidate spans against the [MASK] contextual hidden state."""
        h = sequence_output[0] if sequence_output.ndim == 3 else sequence_output
        h_mask = h[mask_pos]
        cand_reps = []
        for start, end in candidate_spans:
            span_h = h[start:end]
            if span_h.shape[0] == 0:
                span_h = h[start:start + 1]
            cand_reps.append(span_h.mean(dim=0))
        cand_reps_tensor = torch.stack(cand_reps, dim=0)
        return self.pointer_head(h_mask, cand_reps_tensor)

    def _candidate_space_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        relation_triples: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        if attention_mask is None:
            valid = input_ids.ne(self.config.pad_token_id)
        else:
            valid = attention_mask.to(device=input_ids.device).bool()
        space_count = self.relation_adapter.dynamic_qk.max_spaces
        candidate_mask = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            space_count,
            dtype=torch.bool,
            device=input_ids.device,
        )
        active = self.relation_adapter.indices_for_triples(relation_triples)
        # No supplied relation candidate means no dynamic relation path.  This
        # keeps the forward candidate set and the gradient row mask identical;
        # a caller that wants the full bank can pass all relation triples.
        if active:
            candidate_mask[..., active] = valid.unsqueeze(-1)
        return candidate_mask

    def _build_sparse_gate(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[List[int], torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        active_pos_set = set()
        special_ids = {
            self.config.pad_token_id,
            self.config.cls_token_id,
            self.config.sep_token_id,
        }
        if getattr(self, "grammar_attribute_filter", None) is not None:
            flat_ids = input_ids.view(-1)
            probs = self.grammar_attribute_filter.word_attributes.probabilities(flat_ids)
            from .grammar_automaton import ATTRIBUTES
            punct_idx = ATTRIBUTES.index("PUNCT")
            func_idx = ATTRIBUTES.index("FUNCTION")
            is_junk = (probs[:, punct_idx] >= 0.5) | (probs[:, func_idx] >= 0.5)
            is_junk = is_junk.view(batch_size, seq_len)

            for b in range(batch_size):
                for idx in range(seq_len):
                    tid = int(input_ids[b, idx].item())
                    if tid == self.config.mask_token_id:
                        active_pos_set.add(idx)
                    elif tid not in special_ids and not bool(is_junk[b, idx].item()):
                        active_pos_set.add(idx)
        else:
            for idx in range(seq_len):
                tid = int(input_ids[0, idx].item())
                if tid not in special_ids:
                    active_pos_set.add(idx)

        active_positions = sorted(active_pos_set)
        if not active_positions:
            active_positions = list(range(seq_len))

        sparse_gate = input_ids.new_full(
            (batch_size, 1, seq_len, seq_len), -10000.0, dtype=torch.float32
        )
        active_pos_tensor = torch.as_tensor(
            active_positions, dtype=torch.long, device=device
        )
        sparse_gate[
            :, :, active_pos_tensor[:, None], active_pos_tensor[None, :]
        ] = 0.0

        return active_positions, sparse_gate

    def _encode_sequence(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relation_triples: Optional[Sequence[Sequence[int]]] = None,
        candidate_space_mask: Optional[torch.Tensor] = None,
    ):
        relation_triples = list(relation_triples or [])
        hidden_states = self.bert.embeddings(input_ids, token_type_ids)
        extended_mask = self.bert._build_attention_mask(input_ids, attention_mask)
        if candidate_space_mask is None:
            candidate_space_mask = self._candidate_space_mask(
                input_ids,
                attention_mask,
                relation_triples,
            )
        expected_shape = (
            input_ids.size(0),
            input_ids.size(1),
            self.relation_adapter.dynamic_qk.max_spaces,
        )
        if tuple(candidate_space_mask.shape) != expected_shape:
            raise ValueError(
                "candidate_space_mask must have shape "
                f"[batch, sequence, {expected_shape[-1]}]"
            )
        candidate_space_mask = candidate_space_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        )

        route_weights: Optional[torch.Tensor] = None
        all_attentions = () if output_attentions else None
        active_positions = None
        sparse_gate_mask = None
        if getattr(self, "enable_grammar_sparse_gate", False):
            active_positions, sparse_gate_mask = self._build_sparse_gate(input_ids)

        for layer_index, layer_module in enumerate(self.bert.encoder.layer):
            if layer_index == self.route_start_layer:
                route_weights = self.relation_adapter.route_weights(
                    hidden_states,
                    candidate_space_mask,
                )
            layer_route = (
                route_weights if layer_index >= self.route_start_layer else None
            )
            hidden_states, layer_attn = layer_module(
                hidden_states,
                extended_mask,
                output_attentions=output_attentions,
                route_weights=layer_route,
                active_positions=active_positions,
                sparse_gate_mask=sparse_gate_mask if (layer_index >= self.route_start_layer) else None,
            )
            if output_attentions:
                all_attentions = all_attentions + (layer_attn,)
        self._last_route_weights = (
            route_weights.detach() if route_weights is not None else None
        )
        return hidden_states, self.bert.pooler(hidden_states), all_attentions

    def save_pretrained(self, save_directory: str):
        super().save_pretrained(save_directory)
        if getattr(self, "grammar_attribute_filter", None) is not None:
            try:
                self.grammar_attribute_filter.save_pretrained(save_directory)
            except Exception:
                pass
        with open(
            os.path.join(save_directory, "dynamic_qk_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "model_class": self.__class__.__name__,
                    "dynamic_qk": True,
                    "route_start_layer": self.route_start_layer,
                    "route_dim": self.route_dim,
                    "dynamic_qk_score_scale": self.dynamic_qk_score_scale,
                    "relation_space_count": self.relation_adapter.relation_count,
                    "route_source": "contextual_hidden_states",
                    "attention_integration": "pre_softmax_local_qk_score",
                    "relation_ffn_hidden_size": self.relation_ffn_hidden_size,
                    "relation_ffn_scale": self.relation_ffn_scale,
                    "relation_ffn_chunk_size": self.relation_ffn_chunk_size,
                    "fusion_dim": self.fusion_dim,
                    "enable_grammar_sparse_gate": self.enable_grammar_sparse_gate,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
    ) -> "DynamicQKLocalRelationMarginBertForMaskedLM":
        config = BertConfig.from_json_file(os.path.join(load_directory, "config.json"))
        tokenizer = SimpleBertTokenizer.from_pretrained(load_directory)
        with open(
            os.path.join(load_directory, "local_relation_adapter_config.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            adapter_config = json.load(handle)
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        prior_config = {}
        prior_config_path = os.path.join(
            load_directory, "prior_controlled_model_config.json"
        )
        if os.path.isfile(prior_config_path):
            with open(prior_config_path, "r", encoding="utf-8") as handle:
                prior_config = json.load(handle)
        v2_config = {}
        v2_config_path = os.path.join(
            load_directory, "local_relation_adapter_v2_config.json"
        )
        if os.path.isfile(v2_config_path):
            with open(v2_config_path, "r", encoding="utf-8") as handle:
                v2_config = json.load(handle)
        dynamic_config = {}
        dynamic_config_path = os.path.join(load_directory, "dynamic_qk_config.json")
        if os.path.isfile(dynamic_config_path):
            with open(dynamic_config_path, "r", encoding="utf-8") as handle:
                dynamic_config = json.load(handle)

        filter_instance = None
        if os.path.isfile(
            os.path.join(load_directory, "grammar_attribute_config.json")
        ) or os.path.isfile(os.path.join(load_directory, "word_attributes_summary.json")):
            try:
                filter_instance = GrammarAttributeFilter.from_pretrained(load_directory)
            except Exception:
                filter_instance = GrammarAttributeFilter(tokenizer)
        else:
            filter_instance = GrammarAttributeFilter(tokenizer)

        model = cls(
            config,
            tokenizer,
            relation_triples=adapter_config["relation_triples"],
            frequency_prior=state_dict["frequency_prior"],
            frequency_class_scales=state_dict["frequency_class_scales"],
            attribute_filter=filter_instance,
            attribute_bias_scale=float(
                prior_config.get("attribute_bias_scale", 1.5)
            ),
            constrained_frequency_gate=float(
                prior_config.get("constrained_frequency_gate", 0.25)
            ),
            margin_value=float(adapter_config.get("margin_value", 2.0)),
            relation_score_scale=float(
                v2_config.get(
                    "relation_score_scale",
                    adapter_config.get(
                        "relation_score_scale", DEFAULT_RELATION_SCORE_SCALE
                    ),
                )
            ),
            dynamic_qk_score_scale=float(
                dynamic_config.get(
                    "dynamic_qk_score_scale", DEFAULT_DYNAMIC_QK_SCORE_SCALE
                )
            ),
            route_start_layer=int(
                dynamic_config.get("route_start_layer", DEFAULT_ROUTE_START_LAYER)
            ),
            route_dim=int(dynamic_config.get("route_dim", DEFAULT_ROUTE_DIM)),
            relation_ffn_hidden_size=int(dynamic_config.get("relation_ffn_hidden_size", 0)),
            relation_ffn_scale=float(dynamic_config.get("relation_ffn_scale", 0.1)),
            relation_ffn_chunk_size=int(dynamic_config.get("relation_ffn_chunk_size", 32)),
            fusion_dim=dynamic_config.get("fusion_dim"),
            enable_grammar_sparse_gate=bool(
                dynamic_config.get("enable_grammar_sparse_gate", False)
            ),
        )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        for key in missing_keys:
            if not key.startswith("pointer_head.") and not key.startswith(
                "grammar_attribute_filter."
            ):
                raise RuntimeError(f"Missing key in state_dict: {key}")
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys in state_dict: {unexpected_keys}")
        model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
        return model

