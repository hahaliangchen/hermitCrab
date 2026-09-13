"""第一轮：词语关系熟悉训练（纯几何，不经过 QKV/FFN/MLM）。

规则：

1. 活动空间从 3 维开始，最多扩展到 256 维；
2. 词第一次出现时，只按句内顺序放入扇形骨架，不主动进行语义拉近；
3. 如果词对双方在本句开始前都已经有稳定向量，本次共同出现会让它们逐步靠近；
4. 先在当前活动维度中拟合；只有当前空间确实没有满足余地，且扩维试验能降低
   冲突时才增加一维；
5. 扩维后，扩维前的全部坐标冻结，只学习新增坐标；
6. Gram 矩阵只对局部关系子集做秩/正定性诊断，避免完整语料上的 O(n^3) 检查
   成为 CPU 瓶颈。句子约束是部分约束，所以实际拟合结果才是最终扩维依据。

输出的 relationship_embeddings.pt 可以在后续 BERT 阶段作为词嵌入初始化。
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

# Allow direct execution: python examples/relationship_familiarization.py
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
        "relationship-familiarization",
    )
)


def _resolve_path(path: str) -> str:
    """兼容普通 Windows 路径以及被复制成 /D:/... 的路径。"""
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


@dataclass
class FitResult:
    local_embeddings: torch.Tensor
    loss: float
    max_error: float
    fan_max_error: float
    relation_max_deficit: float
    separation_max_violation: float

    @property
    def hard_error(self) -> float:
        return max(self.relation_max_deficit, self.separation_max_violation)


@dataclass
class SentenceConstraints:
    token_ids: List[int]
    positions: List[int]
    pair_i: torch.Tensor
    pair_j: torch.Tensor
    pair_targets: torch.Tensor
    pair_weights: torch.Tensor
    relation_mask: torch.Tensor
    target_gram: torch.Tensor
    target_rank: int
    target_is_psd: bool
    negative_eigenvalue: float
    separation_triplets: torch.Tensor

    @property
    def has_relationship(self) -> bool:
        return bool(self.relation_mask.any().item())


class GeometryState:
    """全词表几何状态，以及词出现次数和词对关系记忆。"""

    def __init__(
        self,
        vocab_size: int,
        base_dim: int = 3,
        max_dim: int = 256,
        seed: int = 42,
        max_pair_memory: int = 1_000_000,
    ) -> None:
        if base_dim != 3:
            raise ValueError("The first-stage base dimension is fixed at 3")
        if max_dim < base_dim:
            raise ValueError("max_dim must be greater than or equal to 3")
        if max_pair_memory <= 0:
            raise ValueError("max_pair_memory must be greater than zero")

        self.embeddings = torch.zeros(vocab_size, max_dim, dtype=torch.float32)
        self.seen_count = torch.zeros(vocab_size, dtype=torch.long)
        self.active_dim = base_dim
        self.frozen_dim = 0
        self.max_dim = max_dim
        self.max_pair_memory = max_pair_memory
        # key -> [sentence-level co-occurrence count, highest remembered cosine]
        self.pair_memory: Dict[int, List[float]] = {}
        self.expansion_events = 0
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def initialize_new_tokens(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        fan_step_degrees: float,
        fan_max_degrees: float,
    ) -> None:
        """给本句第一次出现的词建立旧空间扇形投影。"""
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

                # 新词在最新的可训练坐标上保留微小扰动，避免新增坐标全为 0。
                start = max(2, self.frozen_dim)
                if start < self.active_dim:
                    noise = torch.randn(
                        self.active_dim - start,
                        generator=self._generator,
                    )
                    self.embeddings[token_id, start : self.active_dim] += noise * 0.01

    def expand_one_dimension(self, token_ids: Sequence[int]) -> bool:
        """增加一维；扩维前的全部活动坐标从此冻结。"""
        if self.active_dim >= self.max_dim:
            return False
        old_active_dim = self.active_dim
        with torch.no_grad():
            self.embeddings[:, old_active_dim].zero_()
            if token_ids:
                unique_ids = list(dict.fromkeys(int(token_id) for token_id in token_ids))
                noise = torch.randn(len(unique_ids), generator=self._generator)
                self.embeddings[unique_ids, old_active_dim] = noise * 0.05
        self.frozen_dim = old_active_dim
        self.active_dim = old_active_dim + 1
        self.expansion_events += 1
        return True

    def apply_local(self, token_ids: Sequence[int], local_embeddings: torch.Tensor) -> None:
        with torch.no_grad():
            self.embeddings[list(token_ids), : local_embeddings.size(1)] = local_embeddings

    def mark_seen(self, token_ids: Sequence[int]) -> None:
        with torch.no_grad():
            for token_id in set(int(value) for value in token_ids):
                self.seen_count[token_id] += 1

    def record_pairs(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        max_pair_distance: int,
    ) -> None:
        """记住本句词对，保存它们的最新熟悉度。"""
        if len(token_ids) < 2:
            return
        vectors = F.normalize(self.embeddings[list(token_ids), : self.active_dim], dim=-1)
        similarities = vectors @ vectors.T
        for first in range(len(token_ids)):
            for second in range(first + 1, len(token_ids)):
                if abs(positions[second] - positions[first]) > max_pair_distance:
                    continue
                key = pair_key(token_ids[first], token_ids[second])
                current_cosine = float(similarities[first, second].item())
                memory = self.pair_memory.get(key)
                if memory is None:
                    if len(self.pair_memory) >= self.max_pair_memory:
                        continue
                    self.pair_memory[key] = [1.0, current_cosine]
                else:
                    memory[0] += 1.0
                    memory[1] = max(memory[1], current_cosine)


def _first_positions(token_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    """把重复词折叠成一个全局词向量，并保留其第一次出现的位置。"""
    position_by_token: Dict[int, int] = {}
    for position, token_id in enumerate(token_ids):
        position_by_token.setdefault(int(token_id), position)
    ordered = sorted(position_by_token.items(), key=lambda item: item[1])
    return [item[0] for item in ordered], [item[1] for item in ordered]


def _gram_diagnostics(
    gram: torch.Tensor,
    rank_relative_tolerance: float = 1e-5,
) -> Tuple[int, bool, float]:
    """返回有效正秩、是否半正定、最大负特征值绝对值。"""
    if gram.numel() == 0:
        return 0, True, 0.0
    symmetric = (gram + gram.T) / 2.0
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    scale = max(float(eigenvalues.abs().max().item()), 1.0)
    tolerance = scale * rank_relative_tolerance
    rank = int((eigenvalues > tolerance).sum().item())
    negative = eigenvalues[eigenvalues < -tolerance]
    negative_magnitude = float((-negative).max().item()) if negative.numel() else 0.0
    return rank, negative_magnitude <= tolerance, negative_magnitude


def _select_gram_indices(
    count: int,
    positions: Sequence[int],
    relation_pairs: Sequence[Tuple[int, int]],
    limit: int,
) -> List[int]:
    """只挑关系边和它们之间的少量词用于 Gram 诊断。"""
    if not relation_pairs or limit <= 0:
        return []
    selected = set()
    for first, second in relation_pairs:
        selected.add(first)
        selected.add(second)
        if len(selected) >= limit:
            break
        left = min(positions[first], positions[second])
        right = max(positions[first], positions[second])
        for index in range(count):
            if left < positions[index] < right:
                selected.add(index)
                if len(selected) >= limit:
                    break
        if len(selected) >= limit:
            break
    return sorted(selected)[:limit]


def build_constraints(
    state: GeometryState,
    token_ids: Sequence[int],
    positions: Sequence[int],
    known_before: Sequence[bool],
    fan_step_degrees: float,
    fan_max_degrees: float,
    fan_weight: float,
    relation_weight: float,
    relation_step: float,
    relation_cap: float,
    separation_margin: float,
    gram_token_limit: int,
) -> SentenceConstraints:
    """建立新词扇形、旧词接近，以及中间词分离约束。"""
    count = len(token_ids)
    if count != len(positions) or count != len(known_before):
        raise ValueError("sentence constraint inputs must have the same length")

    vectors = F.normalize(
        state.embeddings[list(token_ids), : state.active_dim],
        dim=-1,
        eps=1e-8,
    )
    current_gram = vectors @ vectors.T
    target_overrides: Dict[Tuple[int, int], float] = {}
    pair_i: List[int] = []
    pair_j: List[int] = []
    pair_targets: List[float] = []
    pair_weights: List[float] = []
    relation_flags: List[bool] = []
    relation_local_pairs: List[Tuple[int, int]] = []

    step = math.radians(fan_step_degrees)
    max_angle = math.radians(fan_max_degrees)
    for first in range(count):
        for second in range(first + 1, count):
            distance = abs(positions[second] - positions[first])
            fan_target = math.cos(min(distance * step, max_angle))
            key = pair_key(token_ids[first], token_ids[second])
            memory = state.pair_memory.get(key)
            has_relationship = known_before[first] and known_before[second]
            has_new_token = not known_before[first] or not known_before[second]
            current_cosine = float(current_gram[first, second].item())
            target = current_cosine
            weight = 0.0
            is_relation = False

            if has_relationship:
                remembered_target = float(memory[1]) if memory is not None else current_cosine
                # 只提高一点目标相似度，不把已有词对强行改成扇形角度。
                target = max(
                    current_cosine,
                    min(
                        relation_cap,
                        max(remembered_target, current_cosine) + relation_step,
                    ),
                )
                weight = relation_weight
                is_relation = True
                relation_local_pairs.append((first, second))
            elif has_new_token:
                # 新词只接受句子顺序的扇形先验，不产生语义接近约束。
                target = fan_target
                weight = fan_weight

            target_overrides[(first, second)] = target
            if weight > 0.0:
                pair_i.append(first)
                pair_j.append(second)
                pair_targets.append(target)
                pair_weights.append(weight)
                relation_flags.append(is_relation)

    relation_keys = {
        pair_key(token_ids[first], token_ids[second])
        for first, second in relation_local_pairs
    }
    separation_rows: List[Tuple[int, int, int]] = []
    for first, second in relation_local_pairs:
        left = min(positions[first], positions[second])
        right = max(positions[first], positions[second])
        for middle in range(count):
            if middle in (first, second):
                continue
            if not (left < positions[middle] < right):
                continue
            middle_key = pair_key(token_ids[second], token_ids[middle])
            if middle_key in relation_keys:
                continue
            separation_rows.append((first, second, middle))

    if pair_i:
        pair_i_tensor = torch.tensor(pair_i, dtype=torch.long)
        pair_j_tensor = torch.tensor(pair_j, dtype=torch.long)
        target_tensor = torch.tensor(pair_targets, dtype=torch.float32)
        weight_tensor = torch.tensor(pair_weights, dtype=torch.float32)
        relation_mask = torch.tensor(relation_flags, dtype=torch.bool)
    else:
        pair_i_tensor = torch.empty(0, dtype=torch.long)
        pair_j_tensor = torch.empty(0, dtype=torch.long)
        target_tensor = torch.empty(0, dtype=torch.float32)
        weight_tensor = torch.empty(0, dtype=torch.float32)
        relation_mask = torch.empty(0, dtype=torch.bool)

    separation_tensor = (
        torch.tensor(separation_rows, dtype=torch.long)
        if separation_rows
        else torch.empty((0, 3), dtype=torch.long)
    )

    # 只在关系边存在时构造小 Gram 子矩阵；无关系的新句子直接跳过特征分解。
    selected = _select_gram_indices(
        count,
        positions,
        relation_local_pairs,
        gram_token_limit,
    )
    if selected:
        target_gram = current_gram[selected][:, selected].detach().clone()
        selected_set = set(selected)
        local_index = {global_index: local for local, global_index in enumerate(selected)}
        for (first, second), target in target_overrides.items():
            if first in selected_set and second in selected_set:
                first_local = local_index[first]
                second_local = local_index[second]
                target_gram[first_local, second_local] = target
                target_gram[second_local, first_local] = target
        target_gram.fill_diagonal_(1.0)
    else:
        target_gram = torch.empty((0, 0), dtype=torch.float32)
    target_rank, target_is_psd, negative_eigenvalue = _gram_diagnostics(target_gram)

    return SentenceConstraints(
        token_ids=list(token_ids),
        positions=list(positions),
        pair_i=pair_i_tensor,
        pair_j=pair_j_tensor,
        pair_targets=target_tensor,
        pair_weights=weight_tensor,
        relation_mask=relation_mask,
        target_gram=target_gram,
        target_rank=target_rank,
        target_is_psd=target_is_psd,
        negative_eigenvalue=negative_eigenvalue,
        separation_triplets=separation_tensor,
    )


def _evaluate_local(
    local_embeddings: torch.Tensor,
    constraints: SentenceConstraints,
    separation_margin: float,
) -> FitResult:
    normalized = F.normalize(local_embeddings, dim=-1, eps=1e-8)
    errors = torch.empty(0, dtype=normalized.dtype)
    weighted_loss = torch.tensor(0.0, dtype=normalized.dtype)
    relation_deficits = torch.empty(0, dtype=normalized.dtype)
    fan_errors = torch.empty(0, dtype=normalized.dtype)

    if constraints.pair_i.numel():
        similarities = (normalized[constraints.pair_i] * normalized[constraints.pair_j]).sum(-1)
        errors = (similarities - constraints.pair_targets).abs()
        relation_deficits = F.relu(
            constraints.pair_targets[constraints.relation_mask]
            - similarities[constraints.relation_mask]
        )
        fan_errors = errors[~constraints.relation_mask]
        relation_loss = torch.tensor(0.0, dtype=normalized.dtype)
        if constraints.relation_mask.any():
            relation_loss = (
                constraints.pair_weights[constraints.relation_mask]
                * relation_deficits.square()
            ).sum()
        if (~constraints.relation_mask).any():
            fan_loss = (
                constraints.pair_weights[~constraints.relation_mask]
                * fan_errors.square()
            ).sum()
        else:
            fan_loss = torch.tensor(0.0, dtype=normalized.dtype)
        weighted_loss = (relation_loss + fan_loss) / constraints.pair_weights.sum().clamp_min(1e-8)

    relation_max_deficit = (
        float(relation_deficits.max().item()) if relation_deficits.numel() else 0.0
    )
    fan_max_error = float(fan_errors.max().item()) if fan_errors.numel() else 0.0
    max_error = float(errors.max().item()) if errors.numel() else 0.0

    if constraints.separation_triplets.numel():
        first = constraints.separation_triplets[:, 0]
        second = constraints.separation_triplets[:, 1]
        middle = constraints.separation_triplets[:, 2]
        relation_similarity = (normalized[first] * normalized[second]).sum(-1)
        middle_similarity = (normalized[second] * normalized[middle]).sum(-1)
        separation_violation = F.relu(
            middle_similarity - relation_similarity + separation_margin
        )
        weighted_loss = weighted_loss + separation_violation.square().mean()
        separation_max_violation = float(separation_violation.max().item())
    else:
        separation_max_violation = 0.0

    return FitResult(
        local_embeddings=local_embeddings.detach().clone(),
        loss=float(weighted_loss.item()),
        max_error=max_error,
        fan_max_error=fan_max_error,
        relation_max_deficit=relation_max_deficit,
        separation_max_violation=separation_max_violation,
    )


def fit_local_geometry(
    state: GeometryState,
    constraints: SentenceConstraints,
    active_dim: int,
    frozen_dim: int,
    steps: int,
    learning_rate: float,
    norm_weight: float,
    separation_margin: float,
) -> FitResult:
    """在当前活动空间优化；frozen_dim 左侧坐标完全不更新。"""
    token_ids = constraints.token_ids
    base = state.embeddings[token_ids, :active_dim].detach().clone()
    local = base.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([local], lr=learning_rate)
    best = _evaluate_local(local.detach(), constraints, separation_margin)
    best_score = best.hard_error + 0.05 * math.sqrt(max(best.loss, 0.0))

    for _ in range(max(int(steps), 0)):
        normalized = F.normalize(local, dim=-1, eps=1e-8)
        if constraints.pair_i.numel():
            similarities = (normalized[constraints.pair_i] * normalized[constraints.pair_j]).sum(-1)
            relation_mask = constraints.relation_mask
            if relation_mask.any():
                relation_deficit = F.relu(
                    constraints.pair_targets[relation_mask]
                    - similarities[relation_mask]
                )
                relation_loss = (
                    constraints.pair_weights[relation_mask]
                    * relation_deficit.square()
                ).sum()
            else:
                relation_loss = local.sum() * 0.0
            if (~relation_mask).any():
                fan_error = similarities[~relation_mask] - constraints.pair_targets[~relation_mask]
                fan_loss = (
                    constraints.pair_weights[~relation_mask] * fan_error.square()
                ).sum()
            else:
                fan_loss = local.sum() * 0.0
            pair_loss = (relation_loss + fan_loss) / constraints.pair_weights.sum().clamp_min(1e-8)
        else:
            pair_loss = local.sum() * 0.0

        loss = pair_loss + norm_weight * (local.norm(dim=-1) - 1.0).square().mean()
        if constraints.separation_triplets.numel():
            first = constraints.separation_triplets[:, 0]
            second = constraints.separation_triplets[:, 1]
            middle = constraints.separation_triplets[:, 2]
            relation_similarity = (normalized[first] * normalized[second]).sum(-1)
            middle_similarity = (normalized[second] * normalized[middle]).sum(-1)
            separation_loss = F.relu(
                middle_similarity - relation_similarity + separation_margin
            ).square().mean()
            loss = loss + separation_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if frozen_dim > 0 and local.grad is not None:
            local.grad[:, : min(frozen_dim, active_dim)] = 0.0
        optimizer.step()
        with torch.no_grad():
            if frozen_dim > 0:
                local[:, : min(frozen_dim, active_dim)] = base[:, : min(frozen_dim, active_dim)]

        candidate = _evaluate_local(local.detach(), constraints, separation_margin)
        candidate_score = candidate.hard_error + 0.05 * math.sqrt(max(candidate.loss, 0.0))
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def _is_feasible(constraints: SentenceConstraints, result: FitResult, tolerance: float) -> bool:
    # 新词第一次出现时没有历史关系，不因扇形初始化误差而升维。
    if not constraints.has_relationship:
        return True
    return result.hard_error <= tolerance


def process_sentence(
    state: GeometryState,
    tokenizer: SimpleBertTokenizer,
    text: str,
    max_tokens: int,
    fit_steps: int,
    learning_rate: float,
    norm_weight: float,
    fan_step_degrees: float,
    fan_max_degrees: float,
    fan_weight: float,
    relation_weight: float,
    relation_step: float,
    relation_cap: float,
    separation_margin: float,
    fit_tolerance: float,
    min_expansion_improvement: float,
    max_pair_distance: int,
    gram_token_limit: int,
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
        tokenizer.token_to_id.get(token, tokenizer.unk_token_id)
        for token in tokens
    ]
    token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    unique_ids, positions = _first_positions(token_ids)

    if len(unique_ids) < 2:
        state.initialize_new_tokens(
            unique_ids,
            positions,
            fan_step_degrees=fan_step_degrees,
            fan_max_degrees=fan_max_degrees,
        )
        state.mark_seen(unique_ids)
        return {
            "has_relationship": False,
            "active_dim": state.active_dim,
            "target_rank": 0,
            "target_is_psd": True,
            "negative_eigenvalue": 0.0,
            "hard_error": 0.0,
            "expanded": False,
        }

    known_before = [state.seen_count[token_id].item() > 0 for token_id in unique_ids]
    state.initialize_new_tokens(
        unique_ids,
        positions,
        fan_step_degrees=fan_step_degrees,
        fan_max_degrees=fan_max_degrees,
    )
    constraints = build_constraints(
        state=state,
        token_ids=unique_ids,
        positions=positions,
        known_before=known_before,
        fan_step_degrees=fan_step_degrees,
        fan_max_degrees=fan_max_degrees,
        fan_weight=fan_weight,
        relation_weight=relation_weight,
        relation_step=relation_step,
        relation_cap=relation_cap,
        separation_margin=separation_margin,
        gram_token_limit=gram_token_limit,
    )

    old_active_dim = state.active_dim
    old_frozen_dim = state.frozen_dim
    before = state.embeddings[unique_ids].detach().clone()
    current = fit_local_geometry(
        state,
        constraints,
        active_dim=old_active_dim,
        frozen_dim=old_frozen_dim,
        steps=fit_steps,
        learning_rate=learning_rate,
        norm_weight=norm_weight,
        separation_margin=separation_margin,
    )

    accepted = current
    expanded = False
    if not _is_feasible(constraints, current, fit_tolerance) and constraints.has_relationship:
        # 失败试验不污染旧空间；扩维时冻结的是扩维前的完整活动空间。
        state.embeddings[unique_ids] = before
        state.active_dim = old_active_dim
        state.frozen_dim = old_frozen_dim

        if state.expand_one_dimension(unique_ids):
            trial = fit_local_geometry(
                state,
                constraints,
                active_dim=state.active_dim,
                frozen_dim=old_active_dim,
                steps=fit_steps,
                learning_rate=learning_rate,
                norm_weight=norm_weight,
                separation_margin=separation_margin,
            )
            improved = trial.hard_error <= current.hard_error - min_expansion_improvement
            if _is_feasible(constraints, trial, fit_tolerance) or improved:
                accepted = trial
                expanded = True
            else:
                state.embeddings[unique_ids] = before
                state.active_dim = old_active_dim
                state.frozen_dim = old_frozen_dim
                accepted = current

    state.apply_local(unique_ids, accepted.local_embeddings)
    state.mark_seen(unique_ids)
    state.record_pairs(
        unique_ids,
        positions,
        max_pair_distance=max_pair_distance,
    )

    return {
        "has_relationship": constraints.has_relationship,
        "active_dim": state.active_dim,
        "target_rank": constraints.target_rank,
        "target_is_psd": constraints.target_is_psd,
        "negative_eigenvalue": constraints.negative_eigenvalue,
        "hard_error": accepted.hard_error,
        "max_error": accepted.max_error,
        "fan_max_error": accepted.fan_max_error,
        "expanded": expanded,
    }


def load_texts(path: str) -> List[str]:
    resolved = _resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Training file not found: {resolved}")
    texts: List[str] = []
    with open(resolved, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                texts.append(line)
    if not texts:
        raise ValueError(f"Training file is empty: {resolved}")
    return texts


def _json_pair_memory(pair_memory: Dict[int, List[float]]) -> Dict[str, List[float]]:
    return {
        str(key): [float(value[0]), float(value[1])]
        for key, value in pair_memory.items()
    }


def train_relationships(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_dim: int = 256,
    max_tokens: int = 256,
    fit_steps: int = 16,
    learning_rate: float = 0.04,
    norm_weight: float = 0.05,
    fan_step_degrees: float = 12.0,
    fan_max_degrees: float = 150.0,
    fan_weight: float = 0.2,
    relation_weight: float = 1.0,
    relation_step: float = 0.08,
    relation_cap: float = 0.96,
    separation_margin: float = 0.08,
    fit_tolerance: float = 0.12,
    min_expansion_improvement: float = 0.01,
    max_pair_distance: int = 128,
    max_pair_memory: int = 1_000_000,
    gram_token_limit: int = 32,
    max_sentences: Optional[int] = None,
    log_every: int = 100,
    seed: int = 42,
) -> str:
    if max_dim < 3:
        raise ValueError("max_dim must be at least 3")
    if max_tokens == 0 or max_tokens < -1:
        raise ValueError("max_tokens must be positive or -1 for no truncation")
    if fit_steps < 0:
        raise ValueError("fit_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if relation_cap < -1.0 or relation_cap > 1.0:
        raise ValueError("relation_cap must be in [-1, 1]")
    if relation_step < 0.0:
        raise ValueError("relation_step must be non-negative")
    if max_pair_distance <= 0:
        raise ValueError("max_pair_distance must be greater than zero")
    if gram_token_limit <= 0:
        raise ValueError("gram_token_limit must be greater than zero")
    if log_every <= 0:
        raise ValueError("log_every must be greater than zero")
    if max_sentences is not None and max_sentences <= 0:
        raise ValueError("max_sentences must be greater than zero when provided")

    set_seed(seed)
    resolved_training_file = _resolve_path(training_file)
    texts = load_texts(resolved_training_file)
    if max_sentences is not None:
        texts = texts[:max_sentences]

    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    state = GeometryState(
        vocab_size=len(tokenizer),
        base_dim=3,
        max_dim=max_dim,
        seed=seed,
        max_pair_memory=max_pair_memory,
    )

    resolved_output_dir = _resolve_path(output_dir)
    os.makedirs(resolved_output_dir, exist_ok=True)
    log_path = os.path.join(resolved_output_dir, "training_log.jsonl")
    stats = {
        "sentences": 0,
        "relationship_sentences": 0,
        "expanded_sentences": 0,
        "new_vocab_tokens": 0,
        "unresolved_sentences": 0,
    }

    with open(log_path, "w", encoding="utf-8") as log_handle:
        for sentence_index, text in enumerate(texts, start=1):
            before_seen = int((state.seen_count > 0).sum().item())
            result = process_sentence(
                state=state,
                tokenizer=tokenizer,
                text=text,
                max_tokens=max_tokens,
                fit_steps=fit_steps,
                learning_rate=learning_rate,
                norm_weight=norm_weight,
                fan_step_degrees=fan_step_degrees,
                fan_max_degrees=fan_max_degrees,
                fan_weight=fan_weight,
                relation_weight=relation_weight,
                relation_step=relation_step,
                relation_cap=relation_cap,
                separation_margin=separation_margin,
                fit_tolerance=fit_tolerance,
                min_expansion_improvement=min_expansion_improvement,
                max_pair_distance=max_pair_distance,
                gram_token_limit=gram_token_limit,
            )
            after_seen = int((state.seen_count > 0).sum().item())
            stats["sentences"] += 1
            stats["new_vocab_tokens"] += max(0, after_seen - before_seen)
            if result["has_relationship"]:
                stats["relationship_sentences"] += 1
            if result["expanded"]:
                stats["expanded_sentences"] += 1
            if result["has_relationship"] and result["hard_error"] > fit_tolerance:
                stats["unresolved_sentences"] += 1

            log_record = {"sentence": sentence_index, **result}
            log_handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")

            if sentence_index == 1 or sentence_index % log_every == 0:
                print(
                    f"sentence {sentence_index}/{len(texts)}: "
                    f"active_dim={state.active_dim}, "
                    f"pair_memory={len(state.pair_memory)}, "
                    f"hard_error={result['hard_error']:.4f}, "
                    f"expanded={result['expanded']}"
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
    with open(
        os.path.join(resolved_output_dir, "relation_memory.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(_json_pair_memory(state.pair_memory), handle, ensure_ascii=False)

    config = {
        "stage": "relationship_familiarization",
        "description": "Pure geometric familiarization; no QKV, FFN, or MLM loss.",
        "training_file": resolved_training_file,
        "base_dim": 3,
        "max_dim": max_dim,
        "active_dim": state.active_dim,
        "frozen_dim": state.frozen_dim,
        "pair_memory_size": len(state.pair_memory),
        "expansion_events": state.expansion_events,
        "stats": stats,
        "hyperparameters": {
            "max_tokens": max_tokens,
            "fit_steps": fit_steps,
            "learning_rate": learning_rate,
            "fan_step_degrees": fan_step_degrees,
            "fan_max_degrees": fan_max_degrees,
            "relation_step": relation_step,
            "relation_cap": relation_cap,
            "separation_margin": separation_margin,
            "fit_tolerance": fit_tolerance,
            "gram_token_limit": gram_token_limit,
        },
    }
    with open(
        os.path.join(resolved_output_dir, "relationship_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print(f"Saved relationship embeddings to {resolved_output_dir}")
    print(
        f"Final active_dim={state.active_dim}, frozen_dim={state.frozen_dim}, "
        f"remembered_pairs={len(state.pair_memory)}"
    )
    return resolved_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-stage pure geometric relationship familiarization training."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=DEFAULT_DATA_PATH,
        help=f"UTF-8 text file, one sentence per line (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-dim", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--fit-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--relation-step", type=float, default=0.08)
    parser.add_argument("--relation-cap", type=float, default=0.96)
    parser.add_argument("--fit-tolerance", type=float, default=0.12)
    parser.add_argument("--separation-margin", type=float, default=0.08)
    parser.add_argument("--max-pair-distance", type=int, default=128)
    parser.add_argument("--max-pair-memory", type=int, default=1_000_000)
    parser.add_argument("--gram-token-limit", type=int, default=32)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
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
        relation_cap=args.relation_cap,
        fit_tolerance=args.fit_tolerance,
        separation_margin=args.separation_margin,
        max_pair_distance=args.max_pair_distance,
        max_pair_memory=args.max_pair_memory,
        gram_token_limit=args.gram_token_limit,
        max_sentences=args.max_sentences,
        log_every=args.log_every,
        seed=args.seed,
    )
