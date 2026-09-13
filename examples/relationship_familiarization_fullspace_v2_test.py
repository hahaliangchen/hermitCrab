"""V2 的 100 句测试入口：保留共现的 B 作为 V 型远离候选。

正式训练阶段可以根据语义标签决定哪些词是负向 B；本入口只是为了在当前
只有纯文本的语料上实际触发并观察完整空间合力判定。
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(__file__))
import relationship_familiarization_fullspace_v2 as geometry


def _find_v_constraints_with_all_middle_tokens(
    state,
    token_ids,
    positions,
    known_before,
    max_tests,
):
    result = []
    seen = set()
    count = len(token_ids)
    for d in range(count):
        if not known_before[d]:
            continue
        partners = [
            index
            for index in range(count)
            if index != d
            and known_before[index]
            and geometry._pair_key(token_ids[d], token_ids[index])
            in state.relation_subspaces
        ]
        if len(partners) < 2:
            continue
        partners.sort(key=lambda index: positions[index])
        for left_index in range(len(partners)):
            for right_index in range(left_index + 1, len(partners)):
                a = partners[left_index]
                c = partners[right_index]
                if positions[a] > positions[c]:
                    a, c = c, a
                for b in range(count):
                    if b in (a, c, d):
                        continue
                    if not (positions[a] < positions[b] < positions[c]):
                        continue
                    if not known_before[b]:
                        continue
                    item = (a, b, c, d)
                    if item in seen:
                        continue
                    seen.add(item)
                    result.append(geometry.VConstraint(a=a, b=b, c=c, d=d))
                    if len(result) >= max_tests:
                        return result
    return result


geometry._find_v_constraints = _find_v_constraints_with_all_middle_tokens


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
