"""第一轮几何熟悉训练的正式入口。

核心实现位于 relationship_familiarization.py。这里仅修正一个运行统计细节：
失败的扩维试验会恢复坐标，但底层原型的事件计数不会自动回滚；本入口按最终
活动维度重新同步事件计数，保证输出配置和实际冻结空间一致。
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(__file__))
import relationship_familiarization as geometry


_original_process_sentence = geometry.process_sentence


def _process_sentence_with_synced_stats(*args, **kwargs):
    state = kwargs.get("state")
    if state is None and args:
        state = args[0]
    result = _original_process_sentence(*args, **kwargs)
    if state is not None:
        state.expansion_events = max(0, state.active_dim - 3)
    return result


geometry.process_sentence = _process_sentence_with_synced_stats


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
