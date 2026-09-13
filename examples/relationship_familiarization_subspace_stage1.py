"""三维子空间组合实验的正式入口。

核心实现位于 relationship_familiarization_subspace.py。这里补上失败扩维试验
的历史回滚，使 dimension_growth.json 只记录最终接受的维度增长。
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(__file__))
import relationship_familiarization_subspace as geometry


_original_snapshot = geometry.SubspaceState.snapshot
_original_restore = geometry.SubspaceState.restore


def _snapshot_with_history(*args, **kwargs):
    snapshot = _original_snapshot(*args, **kwargs)
    state = args[0] if args else kwargs["self"]
    snapshot.expansion_history_length = len(state.expansion_history)
    return snapshot


def _restore_with_history(*args, **kwargs):
    _original_restore(*args, **kwargs)
    state = args[0] if args else kwargs["self"]
    snapshot = args[3] if len(args) > 3 else kwargs["snapshot"]
    history_length = getattr(snapshot, "expansion_history_length", len(state.expansion_history))
    del state.expansion_history[history_length:]


geometry.SubspaceState.snapshot = _snapshot_with_history
geometry.SubspaceState.restore = _restore_with_history


if __name__ == "__main__":
    args = geometry.parse_args()
    geometry.train_relationships(
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
