"""使用较干净的小样本运行完整空间 V2 几何测试。

在现有 V2 流程上补上两个实验约定：

1. 关系投影先截取三维，再分别归一化，计算真正的三维余弦；
2. 新句子的初始扇形相邻角使用 20 度，并将扇形放到当前活动空间中的
   随机二维平面，避免不同句子的相同位置复用同一条全局方向。

这仍然不是 QKV/FFN 训练，只用于观察词关系和维度增长。
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
import relationship_familiarization_fullspace_v2_test as probe


geometry = probe.geometry
base = geometry.base


def _true_projected_similarities(
    normalized: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    triples: torch.Tensor,
) -> torch.Tensor:
    """先取三坐标，再对每个词的三维切片单独归一化。"""
    if first.numel() == 0:
        return torch.empty(0, dtype=normalized.dtype)
    first_vectors = F.normalize(
        normalized[first.unsqueeze(1), triples], dim=-1, eps=1e-8
    )
    second_vectors = F.normalize(
        normalized[second.unsqueeze(1), triples], dim=-1, eps=1e-8
    )
    return (first_vectors * second_vectors).sum(-1)


base._projected_similarities = _true_projected_similarities

_original_initialize = base.SubspaceState.initialize_new_tokens


def _initialize_with_random_plane(
    state,
    token_ids,
    positions,
    fan_step_degrees,
    fan_max_degrees,
) -> None:
    """在随机二维平面中放置新词，保留相邻词的 20 度初始夹角。"""
    if not token_ids:
        return

    _original_initialize(
        state,
        token_ids,
        positions,
        20.0,
        fan_max_degrees,
    )

    new_ids = [
        int(token_id)
        for token_id in token_ids
        if state.seen_count[int(token_id)].item() == 0
    ]
    if not new_ids:
        return

    active_dim = state.active_dim
    center = (len(token_ids) - 1) / 2.0
    step = math.radians(20.0)
    max_angle = math.radians(fan_max_degrees)
    first_basis = torch.randn(active_dim, generator=state._generator)
    first_basis = F.normalize(first_basis, dim=0, eps=1e-8)
    second_basis = torch.randn(active_dim, generator=state._generator)
    second_basis = second_basis - (second_basis * first_basis).sum() * first_basis
    second_basis = F.normalize(second_basis, dim=0, eps=1e-8)

    with torch.no_grad():
        for token_id, position in zip(token_ids, positions):
            if state.seen_count[int(token_id)].item() != 0:
                continue
            angle = (position - center) * step
            angle = max(-max_angle, min(max_angle, angle))
            state.embeddings[int(token_id), :active_dim] = (
                math.cos(angle) * first_basis + math.sin(angle) * second_basis
            )


base.SubspaceState.initialize_new_tokens = _initialize_with_random_plane


if __name__ == "__main__":
    args = geometry.parse_args()
    geometry.train(
        training_file=args.data_path,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
        max_dim=args.max_dim,
        max_tokens=args.max_tokens,
        relation_fit_steps=args.relation_fit_steps,
        v_check_steps=args.v_check_steps,
        v_fit_steps=args.v_fit_steps,
        max_v_tests=args.max_v_tests,
        max_extra_dims=args.max_extra_dims,
        close_target=args.close_target,
        far_target=args.far_target,
        v_tolerance=args.v_tolerance,
        relation_step=args.relation_step,
        relation_floor=args.relation_floor,
        relation_cap=args.relation_cap,
        log_every=args.log_every,
        seed=args.seed,
    )
