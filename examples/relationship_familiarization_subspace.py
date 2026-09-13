"""第一轮：基于三维子空间组合的词语关系熟悉训练。

与 relationship_familiarization.py 的全向量点积不同，本实验为每个已经确认的
词对分配一个三维坐标组合。该词对的相似度只在自己的三个坐标上计算，因此：

    relation -> (coordinate_i, coordinate_j, coordinate_k)

从 256 个坐标中选择三个坐标共有 C(256, 3) 种组合。旧坐标保持冻结，新增坐标
作为 V 型冲突的残差通道。该脚本用于验证这种组合方案的维度增长速度。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_simple.tokenizer import SimpleBertTokenizer


DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "shiji_baihua.txt")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "relationship-familiarization-subspace",
    )
)


def _resolve_path(path: str) -> str:
    path = os.path.expandvars(os.path.expanduser(path))
    if len(path) >= 3 and path[0] in ("/", "\\") and path[2] == ":":
        path = path[1:]
    return os.path.abspath(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pair_key(first: int, second: int) -> int:
    low, high = sorted((int(first), int(second)))
    return (low << 32) | high


def choose3(value: int) -> int:
    if value < 3:
        return 0
    return value * (value - 1) * (value - 2) // 6


@dataclass
class FitResult:
    local_embeddings: torch.Tensor
    loss: float
    relation_max_error: float
    separation_max_violation: float

    @property
    def hard_error(self) -> float:
        return max(self.relation_max_error, self.separation_max_violation)


@dataclass
class StateSnapshot:
    active_dim: int
    frozen_dim: int
    expansion_events: int
    frontier_axis: int
    cursors: Dict[int, Tuple[int, int]]
    rows: torch.Tensor
    relation_values: Dict[int, Optional[Tuple[Tuple[int, int, int], float]]]


class SubspaceState:
    """词向量、关系子空间注册表和三维组合分配器。"""

    def __init__(
        self,
        vocab_size: int,
        base_dim: int = 3,
        max_dim: int = 512,
        seed: int = 42,
    ) -> None:
        if base_dim != 3:
            raise ValueError("base_dim is fixed at 3 for this experiment")
        if max_dim < base_dim:
            raise ValueError("max_dim must be at least 3")

        self.embeddings = torch.zeros(vocab_size, max_dim, dtype=torch.float32)
        self.seen_count = torch.zeros(vocab_size, dtype=torch.long)
        self.active_dim = base_dim
        # active_dim - 1 is deliberately kept trainable after an expansion.
        self.frozen_dim = 0
        self.max_dim = max_dim
        self.expansion_events = 0

        # pair key -> coordinate triple and fixed target cosine.
        self.relation_subspaces: Dict[int, Tuple[int, int, int]] = {}
        self.relation_targets: Dict[int, float] = {}
        self.relation_counts: Dict[int, int] = {}

        # For axis k, allocate (i, j, k), 0 <= i < j < k.
        self.frontier_axis = 2
        self.cursors: Dict[int, Tuple[int, int]] = {2: (0, 1)}
        self.expansion_history: List[Dict[str, object]] = []
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def initialize_new_tokens(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        fan_step_degrees: float,
        fan_max_degrees: float,
    ) -> None:
        if len(token_ids) != len(positions):
            raise ValueError("token_ids and positions must have the same length")
        if not token_ids:
            return

        center = (len(token_ids) - 1) / 2.0
        step = math.radians(fan_step_degrees)
        max_angle = math.radians(fan_max_degrees)
        with torch.no_grad():
            for token_id, position in zip(token_ids, positions):
                if self.seen_count[token_id].item() != 0:
                    continue
                angle = (position - center) * step
                angle = max(-max_angle, min(max_angle, angle))
                self.embeddings[token_id, : self.active_dim].zero_()
                self.embeddings[token_id, 0] = math.cos(angle)
                self.embeddings[token_id, 1] = math.sin(angle)
                if self.active_dim > 2:
                    noise = torch.randn(
                        self.active_dim - 2,
                        generator=self._generator,
                    )
                    self.embeddings[token_id, 2 : self.active_dim] += noise * 0.01

    def expand_one_dimension(
        self,
        token_ids: Sequence[int],
        sentence_index: int,
        reason: str,
    ) -> bool:
        if self.active_dim >= self.max_dim:
            return False
        old_active_dim = self.active_dim
        with torch.no_grad():
            self.embeddings[:, old_active_dim].zero_()
            if token_ids:
                unique_ids = list(dict.fromkeys(int(value) for value in token_ids))
                noise = torch.randn(len(unique_ids), generator=self._generator)
                self.embeddings[unique_ids, old_active_dim] = noise * 0.05
        self.active_dim = old_active_dim + 1
        self.frozen_dim = old_active_dim
        self.expansion_events += 1
        self.expansion_history.append(
            {
                "sentence": sentence_index,
                "from_dim": old_active_dim,
                "to_dim": self.active_dim,
                "reason": reason,
            }
        )
        return True

    def snapshot(
        self,
        token_ids: Sequence[int],
        relation_keys: Sequence[int],
    ) -> StateSnapshot:
        relation_values: Dict[int, Optional[Tuple[Tuple[int, int, int], float]]] = {}
        for key in relation_keys:
            if key in self.relation_subspaces:
                relation_values[key] = (
                    self.relation_subspaces[key],
                    self.relation_targets[key],
                )
            else:
                relation_values[key] = None
        return StateSnapshot(
            active_dim=self.active_dim,
            frozen_dim=self.frozen_dim,
            expansion_events=self.expansion_events,
            frontier_axis=self.frontier_axis,
            cursors=dict(self.cursors),
            rows=self.embeddings[list(token_ids)].detach().clone(),
            relation_values=relation_values,
        )

    def restore(
        self,
        token_ids: Sequence[int],
        relation_keys: Sequence[int],
        snapshot: StateSnapshot,
    ) -> None:
        with torch.no_grad():
            self.embeddings[list(token_ids)] = snapshot.rows
        self.active_dim = snapshot.active_dim
        self.frozen_dim = snapshot.frozen_dim
        self.expansion_events = snapshot.expansion_events
        self.frontier_axis = snapshot.frontier_axis
        self.cursors = dict(snapshot.cursors)
        for key in relation_keys:
            value = snapshot.relation_values.get(key)
            if value is None:
                self.relation_subspaces.pop(key, None)
                self.relation_targets.pop(key, None)
                self.relation_counts.pop(key, None)
            else:
                self.relation_subspaces[key] = value[0]
                self.relation_targets[key] = value[1]

    def _take_pair_for_axis(self, axis: int) -> Optional[Tuple[int, int, int]]:
        i, j = self.cursors.get(axis, (0, 1))
        if j >= axis:
            return None
        triple = (i, j, axis)
        if j + 1 < axis:
            self.cursors[axis] = (i, j + 1)
        elif i + 1 < axis - 1:
            self.cursors[axis] = (i + 1, i + 2)
        else:
            self.cursors[axis] = (axis - 1, axis)
        return triple

    def allocate_relation(
        self,
        key: int,
        token_ids: Sequence[int],
        sentence_index: int,
        current_cosine: float,
        relation_step: float,
        relation_floor: float,
        relation_cap: float,
        force_new_axis: bool = False,
    ) -> Tuple[int, int, int]:
        if key in self.relation_subspaces:
            return self.relation_subspaces[key]

        if force_new_axis:
            axis = self.active_dim - 1
            triple = self._take_pair_for_axis(axis)
            while triple is None:
                if not self.expand_one_dimension(
                    token_ids,
                    sentence_index,
                    reason="subspace_capacity_trial",
                ):
                    raise RuntimeError("maximum dimension reached")
                axis = self.active_dim - 1
                triple = self._take_pair_for_axis(axis)
        else:
            while True:
                axis = self.frontier_axis
                if axis >= self.active_dim:
                    if not self.expand_one_dimension(
                        token_ids,
                        sentence_index,
                        reason="subspace_capacity",
                    ):
                        raise RuntimeError("maximum dimension reached")
                triple = self._take_pair_for_axis(axis)
                if triple is not None:
                    break
                self.frontier_axis += 1
                self.cursors.setdefault(self.frontier_axis, (0, 1))

        self.relation_subspaces[key] = triple
        target = max(
            float(current_cosine),
            min(
                relation_cap,
                max(relation_floor, float(current_cosine)) + relation_step,
            ),
        )
        self.relation_targets[key] = target
        self.relation_counts[key] = self.relation_counts.get(key, 0)
        return triple

    def relation_cosine(
        self,
        first_token: int,
        second_token: int,
        triple: Sequence[int],
    ) -> float:
        first = F.normalize(
            self.embeddings[first_token, list(triple)].unsqueeze(0), dim=-1
        )
        second = F.normalize(
            self.embeddings[second_token, list(triple)].unsqueeze(0), dim=-1
        )
        return float((first * second).sum().item())

    def apply_local(
        self,
        token_ids: Sequence[int],
        local_embeddings: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.embeddings[list(token_ids), : local_embeddings.size(1)] = local_embeddings

    def mark_seen(self, token_ids: Sequence[int]) -> None:
        with torch.no_grad():
            for token_id in set(int(value) for value in token_ids):
                self.seen_count[token_id] += 1


def _first_positions(token_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    position_by_token: Dict[int, int] = {}
    for position, token_id in enumerate(token_ids):
        position_by_token.setdefault(int(token_id), position)
    ordered = sorted(position_by_token.items(), key=lambda item: item[1])
    return [item[0] for item in ordered], [item[1] for item in ordered]


@dataclass
class SentenceBatch:
    token_ids: List[int]
    relation_first: torch.Tensor
    relation_second: torch.Tensor
    relation_triples: torch.Tensor
    relation_targets: torch.Tensor
    separation_first: torch.Tensor
    separation_second: torch.Tensor
    separation_middle: torch.Tensor
    separation_triples: torch.Tensor

    @property
    def relation_count(self) -> int:
        return int(self.relation_first.numel())


def _long_tensor(rows: List[List[int]], width: int) -> torch.Tensor:
    if rows:
        return torch.tensor(rows, dtype=torch.long)
    return torch.empty((0, width), dtype=torch.long)


def build_batch(
    state: SubspaceState,
    token_ids: Sequence[int],
    positions: Sequence[int],
    known_before: Sequence[bool],
) -> SentenceBatch:
    relation_first: List[int] = []
    relation_second: List[int] = []
    relation_triples: List[List[int]] = []
    relation_targets: List[float] = []
    relation_keys: set[int] = set()

    count = len(token_ids)
    for first in range(count):
        for second in range(first + 1, count):
            if not (known_before[first] and known_before[second]):
                continue
            key = pair_key(token_ids[first], token_ids[second])
            triple = state.relation_subspaces.get(key)
            if triple is None:
                continue
            relation_first.append(first)
            relation_second.append(second)
            relation_triples.append(list(triple))
            relation_targets.append(state.relation_targets[key])
            relation_keys.add(key)

    separation_first: List[int] = []
    separation_second: List[int] = []
    separation_middle: List[int] = []
    separation_triples: List[List[int]] = []
    for index in range(len(relation_first)):
        first = relation_first[index]
        second = relation_second[index]
        left = min(positions[first], positions[second])
        right = max(positions[first], positions[second])
        relation_similarity_triple = relation_triples[index]
        for middle in range(count):
            if middle in (first, second):
                continue
            if not (left < positions[middle] < right):
                continue
            middle_key = pair_key(token_ids[second], token_ids[middle])
            if middle_key in relation_keys:
                continue
            separation_first.append(first)
            separation_second.append(second)
            separation_middle.append(middle)
            separation_triples.append(list(relation_similarity_triple))

    return SentenceBatch(
        token_ids=list(token_ids),
        relation_first=torch.tensor(relation_first, dtype=torch.long),
        relation_second=torch.tensor(relation_second, dtype=torch.long),
        relation_triples=_long_tensor(relation_triples, 3),
        relation_targets=torch.tensor(relation_targets, dtype=torch.float32),
        separation_first=torch.tensor(separation_first, dtype=torch.long),
        separation_second=torch.tensor(separation_second, dtype=torch.long),
        separation_middle=torch.tensor(separation_middle, dtype=torch.long),
        separation_triples=_long_tensor(separation_triples, 3),
    )


def _projected_similarities(
    normalized: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    triples: torch.Tensor,
) -> torch.Tensor:
    if first.numel() == 0:
        return torch.empty(0, dtype=normalized.dtype)
    first_vectors = normalized[first.unsqueeze(1), triples]
    second_vectors = normalized[second.unsqueeze(1), triples]
    return (first_vectors * second_vectors).sum(-1)


def evaluate_local(
    local_embeddings: torch.Tensor,
    batch: SentenceBatch,
    separation_margin: float,
    separation_weight: float,
) -> FitResult:
    normalized = F.normalize(local_embeddings, dim=-1, eps=1e-8)
    relation_error = torch.empty(0, dtype=normalized.dtype)
    weighted_loss = torch.tensor(0.0, dtype=normalized.dtype)
    if batch.relation_count:
        similarities = _projected_similarities(
            normalized,
            batch.relation_first,
            batch.relation_second,
            batch.relation_triples,
        )
        relation_error = (similarities - batch.relation_targets).abs()
        weighted_loss = (similarities - batch.relation_targets).square().mean()
    relation_max_error = float(relation_error.max().item()) if relation_error.numel() else 0.0

    separation_max_violation = 0.0
    if batch.separation_first.numel():
        relation_similarity = _projected_similarities(
            normalized,
            batch.separation_first,
            batch.separation_second,
            batch.separation_triples,
        )
        middle_similarity = _projected_similarities(
            normalized,
            batch.separation_second,
            batch.separation_middle,
            batch.separation_triples,
        )
        violation = F.relu(
            middle_similarity - relation_similarity + separation_margin
        )
        weighted_loss = weighted_loss + separation_weight * violation.square().mean()
        separation_max_violation = float(violation.max().item())

    return FitResult(
        local_embeddings=local_embeddings.detach().clone(),
        loss=float(weighted_loss.item()),
        relation_max_error=relation_max_error,
        separation_max_violation=separation_max_violation,
    )


def fit_local(
    state: SubspaceState,
    batch: SentenceBatch,
    active_dim: int,
    frozen_dim: int,
    steps: int,
    learning_rate: float,
    norm_weight: float,
    separation_margin: float,
    separation_weight: float,
) -> FitResult:
    base = state.embeddings[batch.token_ids, :active_dim].detach().clone()
    local = base.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([local], lr=learning_rate)
    best = evaluate_local(local.detach(), batch, separation_margin, separation_weight)
    best_score = best.hard_error + 0.05 * math.sqrt(max(best.loss, 0.0))

    for _ in range(max(int(steps), 0)):
        normalized = F.normalize(local, dim=-1, eps=1e-8)
        similarities = _projected_similarities(
            normalized,
            batch.relation_first,
            batch.relation_second,
            batch.relation_triples,
        )
        if similarities.numel():
            loss = (similarities - batch.relation_targets).square().mean()
        else:
            loss = local.sum() * 0.0

        if batch.separation_first.numel():
            relation_similarity = _projected_similarities(
                normalized,
                batch.separation_first,
                batch.separation_second,
                batch.separation_triples,
            )
            middle_similarity = _projected_similarities(
                normalized,
                batch.separation_second,
                batch.separation_middle,
                batch.separation_triples,
            )
            violation = F.relu(
                middle_similarity - relation_similarity + separation_margin
            )
            loss = loss + separation_weight * violation.square().mean()

        loss = loss + norm_weight * (local.norm(dim=-1) - 1.0).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if frozen_dim > 0 and local.grad is not None:
            local.grad[:, : min(frozen_dim, active_dim)] = 0.0
        optimizer.step()
        with torch.no_grad():
            if frozen_dim > 0:
                local[:, : min(frozen_dim, active_dim)] = base[
                    :, : min(frozen_dim, active_dim)
                ]

        candidate = evaluate_local(
            local.detach(), batch, separation_margin, separation_weight
        )
        candidate_score = candidate.hard_error + 0.05 * math.sqrt(
            max(candidate.loss, 0.0)
        )
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def _new_relation_pairs(
    token_ids: Sequence[int],
    known_before: Sequence[bool],
    state: SubspaceState,
) -> List[Tuple[int, int, int]]:
    result: List[Tuple[int, int, int]] = []
    for first in range(len(token_ids)):
        for second in range(first + 1, len(token_ids)):
            if not (known_before[first] and known_before[second]):
                continue
            key = pair_key(token_ids[first], token_ids[second])
            if key not in state.relation_subspaces:
                result.append((key, token_ids[first], token_ids[second]))
    return result


def _allocate_relations(
    state: SubspaceState,
    new_relations: Sequence[Tuple[int, int, int]],
    token_ids: Sequence[int],
    sentence_index: int,
    relation_step: float,
    relation_floor: float,
    relation_cap: float,
    force_new_axis: bool,
) -> None:
    for key, first_token, second_token in new_relations:
        if key in state.relation_subspaces:
            continue
        # The triple is allocated before measuring its initial similarity.
        # The target is fixed once; repeated occurrences do not keep increasing it.
        triple = state.allocate_relation(
            key=key,
            token_ids=token_ids,
            sentence_index=sentence_index,
            current_cosine=0.0,
            relation_step=relation_step,
            relation_floor=relation_floor,
            relation_cap=relation_cap,
            force_new_axis=force_new_axis,
        )
        current = state.relation_cosine(first_token, second_token, triple)
        target = max(
            current,
            min(relation_cap, max(relation_floor, current) + relation_step),
        )
        state.relation_targets[key] = target


def process_sentence(
    state: SubspaceState,
    tokenizer: SimpleBertTokenizer,
    text: str,
    sentence_index: int,
    max_tokens: int,
    fit_steps: int,
    learning_rate: float,
    norm_weight: float,
    fan_step_degrees: float,
    fan_max_degrees: float,
    relation_step: float,
    relation_floor: float,
    relation_cap: float,
    fit_tolerance: float,
    min_expansion_improvement: float,
    separation_margin: float,
    separation_weight: float,
) -> Dict[str, object]:
    tokens = tokenizer.tokenize(text)
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    token_ids = [
        tokenizer.token_to_id.get(token, tokenizer.unk_token_id) for token in tokens
    ]
    token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    unique_ids, positions = _first_positions(token_ids)

    if len(unique_ids) < 2:
        state.initialize_new_tokens(
            unique_ids,
            positions,
            fan_step_degrees,
            fan_max_degrees,
        )
        state.mark_seen(unique_ids)
        return {
            "has_relationship": False,
            "new_relations": 0,
            "active_dim": state.active_dim,
            "capacity_total": choose3(state.active_dim),
            "capacity_used": len(state.relation_subspaces),
            "hard_error": 0.0,
            "expanded": False,
            "dimension_added": 0,
        }

    known_before = [state.seen_count[token_id].item() > 0 for token_id in unique_ids]
    state.initialize_new_tokens(
        unique_ids,
        positions,
        fan_step_degrees,
        fan_max_degrees,
    )

    new_relations = _new_relation_pairs(unique_ids, known_before, state)
    relation_keys = [item[0] for item in new_relations]
    before_snapshot = state.snapshot(unique_ids, relation_keys)
    before_active_dim = state.active_dim

    _allocate_relations(
        state,
        new_relations,
        unique_ids,
        sentence_index,
        relation_step,
        relation_floor,
        relation_cap,
        force_new_axis=False,
    )
    initial_snapshot = state.snapshot(unique_ids, relation_keys)
    initial_active_dim = state.active_dim
    initial_frozen_dim = (
        before_active_dim if initial_active_dim > before_active_dim else state.frozen_dim
    )
    batch = build_batch(state, unique_ids, positions, known_before)
    current = fit_local(
        state,
        batch,
        active_dim=state.active_dim,
        frozen_dim=initial_frozen_dim,
        steps=fit_steps,
        learning_rate=learning_rate,
        norm_weight=norm_weight,
        separation_margin=separation_margin,
        separation_weight=separation_weight,
    )

    accepted = current
    expanded = initial_active_dim > before_active_dim
    trial_expanded = False
    if (
        new_relations
        and current.hard_error > fit_tolerance
        and state.active_dim < state.max_dim
    ):
        # The new relations were not yet part of history, so they may be moved to
        # a fresh subspace whose newest coordinate can bypass the old frozen plane.
        state.restore(unique_ids, relation_keys, before_snapshot)
        state.expand_one_dimension(
            unique_ids,
            sentence_index,
            reason="v_conflict_trial",
        )
        _allocate_relations(
            state,
            new_relations,
            unique_ids,
            sentence_index,
            relation_step,
            relation_floor,
            relation_cap,
            force_new_axis=True,
        )
        trial_batch = build_batch(state, unique_ids, positions, known_before)
        trial = fit_local(
            state,
            trial_batch,
            active_dim=state.active_dim,
            frozen_dim=before_active_dim,
            steps=fit_steps,
            learning_rate=learning_rate,
            norm_weight=norm_weight,
            separation_margin=separation_margin,
            separation_weight=separation_weight,
        )
        if trial.hard_error <= fit_tolerance or trial.hard_error <= (
            current.hard_error - min_expansion_improvement
        ):
            accepted = trial
            expanded = True
            trial_expanded = True
        else:
            state.restore(unique_ids, relation_keys, initial_snapshot)
            accepted = current

    if state.active_dim > before_active_dim:
        # All but the newest coordinate become part of the immutable old space.
        state.frozen_dim = max(state.frozen_dim, state.active_dim - 1)

    state.apply_local(unique_ids, accepted.local_embeddings)
    state.mark_seen(unique_ids)
    relation_keys_in_sentence: set[int] = set()
    for first in range(len(unique_ids)):
        for second in range(first + 1, len(unique_ids)):
            if known_before[first] and known_before[second]:
                key = pair_key(unique_ids[first], unique_ids[second])
                if key in state.relation_subspaces:
                    relation_keys_in_sentence.add(key)
    for key in relation_keys_in_sentence:
        state.relation_counts[key] = state.relation_counts.get(key, 0) + 1

    return {
        "has_relationship": bool(relation_keys_in_sentence),
        "new_relations": len(new_relations),
        "active_dim": state.active_dim,
        "capacity_total": choose3(state.active_dim),
        "capacity_used": len(state.relation_subspaces),
        "hard_error": accepted.hard_error,
        "relation_max_error": accepted.relation_max_error,
        "separation_max_violation": accepted.separation_max_violation,
        "expanded": expanded,
        "trial_expanded": trial_expanded,
        "dimension_added": state.active_dim - before_active_dim,
    }


def load_texts(path: str) -> List[str]:
    resolved = _resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Training file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]
    if not texts:
        raise ValueError(f"Training file is empty: {resolved}")
    return texts


def train_relationships(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_dim: int = 512,
    max_tokens: int = 256,
    fit_steps: int = 6,
    learning_rate: float = 0.04,
    norm_weight: float = 0.02,
    fan_step_degrees: float = 12.0,
    fan_max_degrees: float = 150.0,
    relation_step: float = 0.12,
    relation_floor: float = 0.30,
    relation_cap: float = 0.88,
    fit_tolerance: float = 0.18,
    min_expansion_improvement: float = 0.01,
    separation_margin: float = 0.08,
    separation_weight: float = 0.20,
    max_sentences: Optional[int] = None,
    log_every: int = 250,
    seed: int = 42,
) -> str:
    if max_dim < 3:
        raise ValueError("max_dim must be at least 3")
    if max_tokens == 0 or max_tokens < -1:
        raise ValueError("max_tokens must be positive or -1")
    if fit_steps < 0:
        raise ValueError("fit_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= relation_floor <= relation_cap <= 1.0:
        raise ValueError("relation_floor and relation_cap must be ordered in [0, 1]")

    set_seed(seed)
    resolved_training_file = _resolve_path(training_file)
    texts = load_texts(resolved_training_file)
    if max_sentences is not None:
        texts = texts[:max_sentences]

    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    state = SubspaceState(
        vocab_size=len(tokenizer),
        base_dim=3,
        max_dim=max_dim,
        seed=seed,
    )
    resolved_output_dir = _resolve_path(output_dir)
    os.makedirs(resolved_output_dir, exist_ok=True)
    log_path = os.path.join(resolved_output_dir, "training_log.jsonl")
    stats = {
        "sentences": 0,
        "relationship_sentences": 0,
        "new_vocab_tokens": 0,
        "unresolved_sentences": 0,
        "new_relations": 0,
    }

    with open(log_path, "w", encoding="utf-8") as log_handle:
        for sentence_index, text in enumerate(texts, start=1):
            before_seen = int((state.seen_count > 0).sum().item())
            result = process_sentence(
                state=state,
                tokenizer=tokenizer,
                text=text,
                sentence_index=sentence_index,
                max_tokens=max_tokens,
                fit_steps=fit_steps,
                learning_rate=learning_rate,
                norm_weight=norm_weight,
                fan_step_degrees=fan_step_degrees,
                fan_max_degrees=fan_max_degrees,
                relation_step=relation_step,
                relation_floor=relation_floor,
                relation_cap=relation_cap,
                fit_tolerance=fit_tolerance,
                min_expansion_improvement=min_expansion_improvement,
                separation_margin=separation_margin,
                separation_weight=separation_weight,
            )
            after_seen = int((state.seen_count > 0).sum().item())
            stats["sentences"] += 1
            stats["new_vocab_tokens"] += max(0, after_seen - before_seen)
            stats["new_relations"] += int(result["new_relations"])
            if result["has_relationship"]:
                stats["relationship_sentences"] += 1
            if result["has_relationship"] and result["hard_error"] > fit_tolerance:
                stats["unresolved_sentences"] += 1

            record = {
                "sentence": sentence_index,
                **result,
                "expansion_events": state.expansion_events,
            }
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()
            if sentence_index == 1 or sentence_index % log_every == 0:
                print(
                    f"sentence {sentence_index}/{len(texts)}: "
                    f"active_dim={state.active_dim}, "
                    f"relations={len(state.relation_subspaces)}, "
                    f"capacity={choose3(state.active_dim)}, "
                    f"hard_error={result['hard_error']:.4f}, "
                    f"added={result['dimension_added']}"
                )

    tokenizer.save_pretrained(resolved_output_dir)
    torch.save(
        {
            "embeddings": state.embeddings,
            "active_dim": state.active_dim,
            "frozen_dim": state.frozen_dim,
            "base_dim": 3,
            "max_dim": max_dim,
            "seen_count": state.seen_count,
        },
        os.path.join(resolved_output_dir, "relationship_embeddings.pt"),
    )

    relation_output = {
        str(key): [
            *state.relation_subspaces[key],
            float(state.relation_targets[key]),
            int(state.relation_counts.get(key, 0)),
        ]
        for key in state.relation_subspaces
    }
    with open(
        os.path.join(resolved_output_dir, "relation_subspaces.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(relation_output, handle, ensure_ascii=False)
    with open(
        os.path.join(resolved_output_dir, "dimension_growth.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(state.expansion_history, handle, ensure_ascii=False, indent=2)

    config = {
        "stage": "relationship_familiarization_subspace",
        "description": "Each confirmed word pair owns one coordinate triple; cosine is computed only in that triple.",
        "training_file": resolved_training_file,
        "base_dim": 3,
        "max_dim": max_dim,
        "active_dim": state.active_dim,
        "frozen_dim": state.frozen_dim,
        "relation_subspaces": len(state.relation_subspaces),
        "available_coordinate_triples": choose3(max_dim),
        "used_coordinate_triples": len(state.relation_subspaces),
        "expansion_events": state.expansion_events,
        "stats": stats,
        "hyperparameters": {
            "max_tokens": max_tokens,
            "fit_steps": fit_steps,
            "learning_rate": learning_rate,
            "relation_step": relation_step,
            "relation_floor": relation_floor,
            "relation_cap": relation_cap,
            "fit_tolerance": fit_tolerance,
            "separation_margin": separation_margin,
            "separation_weight": separation_weight,
        },
    }
    with open(
        os.path.join(resolved_output_dir, "relationship_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print(f"Saved subspace embeddings to {resolved_output_dir}")
    print(
        f"Final active_dim={state.active_dim}, "
        f"frozen_dim={state.frozen_dim}, "
        f"relations={len(state.relation_subspaces)}, "
        f"coordinate_triples={choose3(state.active_dim)}"
    )
    return resolved_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relationship familiarization using coordinate-triple subspaces."
    )
    parser.add_argument("data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-dim", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--fit-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--relation-step", type=float, default=0.12)
    parser.add_argument("--relation-floor", type=float, default=0.30)
    parser.add_argument("--relation-cap", type=float, default=0.88)
    parser.add_argument("--fit-tolerance", type=float, default=0.18)
    parser.add_argument("--separation-margin", type=float, default=0.08)
    parser.add_argument("--separation-weight", type=float, default=0.20)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_relationships(
        training_file=args.data_path,
        output_dir=args.output_dir,
        max_dim=args.max_dim,
        max_tokens=args.max_tokens,
        fit_steps=args.fit_steps,
        learning_rate=args.learning_rate,
        relation_step=args.relation_step,
        relation_floor=args.relation_floor,
        relation_cap=args.relation_cap,
        fit_tolerance=args.fit_tolerance,
        separation_margin=args.separation_margin,
        separation_weight=args.separation_weight,
        max_sentences=args.max_sentences,
        log_every=args.log_every,
        seed=args.seed,
    )
