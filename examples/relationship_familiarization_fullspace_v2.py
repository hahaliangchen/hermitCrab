"""V2 小实验：完整空间 V 型检测 + 三维关系子空间分配。

每个已确认词对仍然拥有一个三坐标关系子空间，但扩维判定不再读取单个三维
投影，而是使用完整活动向量检查：D 是否可以同时接近 A、C 并远离 B。
该脚本主要用于观察前若干句的维度增长速度。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
import relationship_familiarization_subspace as base


DEFAULT_DATA_PATH = base.DEFAULT_DATA_PATH
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "relationship-familiarization-fullspace-v2-100",
    )
)


@dataclass(frozen=True)
class VConstraint:
    """D 要接近 A、C，同时远离 B；字段是当前句子的局部位置。"""

    a: int
    b: int
    c: int
    d: int


def _pair_key(first: int, second: int) -> int:
    return base.pair_key(first, second)


def _first_positions(token_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
    return base._first_positions(token_ids)


def _normalized_rows(state: base.SubspaceState, token_ids: Sequence[int]) -> torch.Tensor:
    return F.normalize(
        state.embeddings[list(token_ids), : state.active_dim],
        dim=-1,
        eps=1e-8,
    )


def _find_new_relations(
    state: base.SubspaceState,
    token_ids: Sequence[int],
    known_before: Sequence[bool],
) -> List[Tuple[int, int, int]]:
    result: List[Tuple[int, int, int]] = []
    for first in range(len(token_ids)):
        for second in range(first + 1, len(token_ids)):
            if not (known_before[first] and known_before[second]):
                continue
            key = _pair_key(token_ids[first], token_ids[second])
            if key not in state.relation_subspaces:
                result.append((key, token_ids[first], token_ids[second]))
    return result


def _allocate_relations(
    state: base.SubspaceState,
    new_relations: Sequence[Tuple[int, int, int]],
    sentence_index: int,
    relation_step: float,
    relation_floor: float,
    relation_cap: float,
    token_ids: Sequence[int],
) -> None:
    for key, first_token, second_token in new_relations:
        triple = state.allocate_relation(
            key=key,
            token_ids=token_ids,
            sentence_index=sentence_index,
            current_cosine=0.0,
            relation_step=relation_step,
            relation_floor=relation_floor,
            relation_cap=relation_cap,
            force_new_axis=False,
        )
        current = state.relation_cosine(first_token, second_token, triple)
        state.relation_targets[key] = max(
            current,
            min(relation_cap, max(relation_floor, current) + relation_step),
        )


def _find_v_constraints(
    state: base.SubspaceState,
    token_ids: Sequence[int],
    positions: Sequence[int],
    known_before: Sequence[bool],
    max_tests: int,
) -> List[VConstraint]:
    """从句内已知关系中提取有限数量的 A-B-C-D V 型候选。"""
    result: List[VConstraint] = []
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
            and _pair_key(token_ids[d], token_ids[index])
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
                    # 如果 D-B 已经是关系边，就不把它当作“远离 B”的候选。
                    if _pair_key(token_ids[d], token_ids[b]) in state.relation_subspaces:
                        continue
                    item = (a, b, c, d)
                    if item in seen:
                        continue
                    seen.add(item)
                    result.append(VConstraint(a=a, b=b, c=c, d=d))
                    if len(result) >= max_tests:
                        return result
    return result


def _v_errors(
    query: torch.Tensor,
    anchors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    close_target: float,
    far_target: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_query = F.normalize(query, dim=-1, eps=1e-8)
    a, b, c = anchors
    d_a = (normalized_query * a).sum()
    d_c = (normalized_query * c).sum()
    d_b = (normalized_query * b).sum()
    errors = torch.stack(
        [
            F.relu(torch.as_tensor(close_target, dtype=query.dtype) - d_a),
            F.relu(torch.as_tensor(close_target, dtype=query.dtype) - d_c),
            F.relu(d_b - torch.as_tensor(far_target, dtype=query.dtype)),
        ]
    )
    return errors, d_a, d_b, d_c


def _full_space_v_check(
    state: base.SubspaceState,
    token_ids: Sequence[int],
    constraint: VConstraint,
    close_target: float,
    far_target: float,
    steps: int,
) -> Dict[str, float]:
    """在当前完整活动空间中寻找 D；不改变全局词向量。"""
    ids = [token_ids[constraint.a], token_ids[constraint.b], token_ids[constraint.c]]
    anchors = _normalized_rows(state, ids)
    a, b, c = anchors[0].detach(), anchors[1].detach(), anchors[2].detach()
    resultant = a + c
    resultant_norm = float(resultant.norm().item())
    if resultant_norm > 1e-8:
        initial = F.normalize(resultant, dim=0, eps=1e-8)
        resultant_alignment = float((initial * b).sum().item())
    else:
        initial = _normalized_rows(
            state,
            [token_ids[constraint.d]],
        )[0].detach().clone()
        resultant_alignment = 1.0

    query = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([query], lr=0.12)
    best_error = float("inf")
    best_close_a = 0.0
    best_close_c = 0.0
    best_far_b = 0.0
    for _ in range(max(int(steps), 0)):
        errors, d_a, d_b, d_c = _v_errors(
            query,
            (a, b, c),
            close_target,
            far_target,
        )
        loss = errors.square().mean() + 0.01 * (query.norm() - 1.0).square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            candidate_errors, candidate_a, candidate_b, candidate_c = _v_errors(
                query,
                (a, b, c),
                close_target,
                far_target,
            )
            candidate_error = float(candidate_errors.max().item())
            if candidate_error < best_error:
                best_error = candidate_error
                best_close_a = float(candidate_a.item())
                best_close_c = float(candidate_c.item())
                best_far_b = float(candidate_b.item())
    if steps == 0:
        errors, d_a, d_b, d_c = _v_errors(
            query,
            (a, b, c),
            close_target,
            far_target,
        )
        best_error = float(errors.max().item())
        best_close_a = float(d_a.item())
        best_close_c = float(d_c.item())
        best_far_b = float(d_b.item())
    return {
        "best_error": best_error,
        "resultant_alignment": resultant_alignment,
        "best_d_a": best_close_a,
        "best_d_c": best_close_c,
        "best_d_b": best_far_b,
    }


def _fit_v_after_expansion(
    state: base.SubspaceState,
    token_ids: Sequence[int],
    constraints: Sequence[VConstraint],
    sentence_index: int,
    close_target: float,
    far_target: float,
    fit_steps: int,
    tolerance: float,
    max_extra_dims: int,
) -> Dict[str, object]:
    if not constraints or state.active_dim >= state.max_dim:
        return {"expanded": False, "dimension_added": 0, "best_error": 0.0}

    involved = sorted(
        {
            token_ids[index]
            for constraint in constraints
            for index in (constraint.a, constraint.b, constraint.c, constraint.d)
        }
    )
    old_active_dim = state.active_dim
    dimension_added = 0
    best_error = float("inf")
    best_rows: Optional[torch.Tensor] = None

    for _ in range(max_extra_dims):
        if not state.expand_one_dimension(
            involved,
            sentence_index,
            reason="full_space_v_conflict",
        ):
            break
        dimension_added += 1
        active_dim = state.active_dim
        rows = state.embeddings[involved, :active_dim].detach().clone()
        local = rows.clone().requires_grad_(True)
        row_index = {token_id: index for index, token_id in enumerate(involved)}
        optimizer = torch.optim.Adam([local], lr=0.10)
        local_best_error = float("inf")
        local_best_rows = rows.clone()

        for _ in range(max(int(fit_steps), 0)):
            normalized = F.normalize(local, dim=-1, eps=1e-8)
            losses: List[torch.Tensor] = []
            errors: List[torch.Tensor] = []
            for constraint in constraints:
                a_id = token_ids[constraint.a]
                b_id = token_ids[constraint.b]
                c_id = token_ids[constraint.c]
                d_id = token_ids[constraint.d]
                a = normalized[row_index[a_id]]
                b = normalized[row_index[b_id]]
                c = normalized[row_index[c_id]]
                d = normalized[row_index[d_id]]
                pair_errors = torch.stack(
                    [
                        F.relu(
                            torch.as_tensor(close_target, dtype=local.dtype)
                            - (d * a).sum()
                        ),
                        F.relu(
                            torch.as_tensor(close_target, dtype=local.dtype)
                            - (d * c).sum()
                        ),
                        F.relu(
                            (d * b).sum()
                            - torch.as_tensor(far_target, dtype=local.dtype)
                        ),
                    ]
                )
                errors.append(pair_errors)
                losses.append(pair_errors.square().mean())
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if local.grad is not None:
                local.grad[:, :old_active_dim] = 0.0
            optimizer.step()
            with torch.no_grad():
                local[:, :old_active_dim] = rows[:, :old_active_dim]
                current_error = float(torch.cat(errors).max().item())
                if current_error < local_best_error:
                    local_best_error = current_error
                    local_best_rows = local.detach().clone()

        if local_best_error < best_error:
            best_error = local_best_error
            best_rows = local_best_rows
        if local_best_error <= tolerance:
            break

    if best_rows is not None:
        state.apply_local(involved, best_rows)
        state.frozen_dim = max(state.frozen_dim, state.active_dim - 1)
    return {
        "expanded": dimension_added > 0,
        "dimension_added": dimension_added,
        "best_error": 0.0 if best_error == float("inf") else best_error,
    }


def _process_sentence(
    state: base.SubspaceState,
    tokenizer: base.SimpleBertTokenizer,
    text: str,
    sentence_index: int,
    max_tokens: int,
    relation_step: float,
    relation_floor: float,
    relation_cap: float,
    close_target: float,
    far_target: float,
    v_tolerance: float,
    v_check_steps: int,
    v_fit_steps: int,
    max_v_tests: int,
    max_extra_dims: int,
    relation_fit_steps: int,
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
        state.initialize_new_tokens(unique_ids, positions, 12.0, 150.0)
        state.mark_seen(unique_ids)
        return {
            "has_relationship": False,
            "new_relations": 0,
            "v_tests": 0,
            "v_conflicts": 0,
            "v_expanded": False,
            "active_dim": state.active_dim,
            "capacity_total": base.choose3(state.active_dim),
            "capacity_used": len(state.relation_subspaces),
            "dimension_added": 0,
            "max_v_error_before": 0.0,
            "max_v_error_after": 0.0,
        }

    known_before = [state.seen_count[token_id].item() > 0 for token_id in unique_ids]
    state.initialize_new_tokens(unique_ids, positions, 12.0, 150.0)
    new_relations = _find_new_relations(state, unique_ids, known_before)
    before_active_dim = state.active_dim
    _allocate_relations(
        state,
        new_relations,
        sentence_index,
        relation_step,
        relation_floor,
        relation_cap,
        unique_ids,
    )
    constraints = _find_v_constraints(
        state,
        unique_ids,
        positions,
        known_before,
        max_v_tests,
    )

    checks = [
        _full_space_v_check(
            state,
            unique_ids,
            constraint,
            close_target,
            far_target,
            v_check_steps,
        )
        for constraint in constraints
    ]
    conflicts = [
        constraint
        for constraint, check in zip(constraints, checks)
        if check["best_error"] > v_tolerance
    ]
    max_before = max((check["best_error"] for check in checks), default=0.0)
    expansion_result = _fit_v_after_expansion(
        state,
        unique_ids,
        conflicts,
        sentence_index,
        close_target,
        far_target,
        v_fit_steps,
        v_tolerance,
        max_extra_dims,
    )

    relation_fit_error = 0.0
    batch = base.build_batch(state, unique_ids, positions, known_before)
    if batch.relation_count and relation_fit_steps > 0:
        relation_fit = base.fit_local(
            state,
            batch,
            active_dim=state.active_dim,
            frozen_dim=state.frozen_dim,
            steps=relation_fit_steps,
            learning_rate=0.04,
            norm_weight=0.02,
            separation_margin=0.08,
            separation_weight=0.20,
        )
        relation_fit_error = relation_fit.hard_error
        state.apply_local(unique_ids, relation_fit.local_embeddings)

    state.mark_seen(unique_ids)
    relation_keys_in_sentence: set[int] = set()
    for first in range(len(unique_ids)):
        for second in range(first + 1, len(unique_ids)):
            if known_before[first] and known_before[second]:
                key = _pair_key(unique_ids[first], unique_ids[second])
                if key in state.relation_subspaces:
                    relation_keys_in_sentence.add(key)
    for key in relation_keys_in_sentence:
        state.relation_counts[key] = state.relation_counts.get(key, 0) + 1

    return {
        "has_relationship": bool(relation_keys_in_sentence),
        "new_relations": len(new_relations),
        "v_tests": len(constraints),
        "v_conflicts": len(conflicts),
        "v_expanded": expansion_result["expanded"],
        "active_dim": state.active_dim,
        "capacity_total": base.choose3(state.active_dim),
        "capacity_used": len(state.relation_subspaces),
        "dimension_added": state.active_dim - before_active_dim,
        "max_v_error_before": max_before,
        "max_v_error_after": expansion_result["best_error"],
        "relation_fit_error": relation_fit_error,
    }


def train(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_sentences: int = 100,
    max_dim: int = 512,
    max_tokens: int = 256,
    relation_fit_steps: int = 2,
    v_check_steps: int = 8,
    v_fit_steps: int = 12,
    max_v_tests: int = 8,
    max_extra_dims: int = 2,
    close_target: float = 0.65,
    far_target: float = 0.15,
    v_tolerance: float = 0.08,
    relation_step: float = 0.12,
    relation_floor: float = 0.30,
    relation_cap: float = 0.88,
    log_every: int = 25,
    seed: int = 42,
) -> str:
    base.set_seed(seed)
    texts = base.load_texts(training_file)[:max_sentences]
    tokenizer = base.SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    state = base.SubspaceState(
        vocab_size=len(tokenizer),
        base_dim=3,
        max_dim=max_dim,
        seed=seed,
    )
    resolved_output_dir = base._resolve_path(output_dir)
    os.makedirs(resolved_output_dir, exist_ok=True)
    logs: List[Dict[str, object]] = []
    stats = {
        "sentences": 0,
        "relationship_sentences": 0,
        "new_vocab_tokens": 0,
        "new_relations": 0,
        "v_tests": 0,
        "v_conflicts": 0,
        "v_expanded_sentences": 0,
    }
    log_path = os.path.join(resolved_output_dir, "training_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as log_handle:
        for sentence_index, text in enumerate(texts, start=1):
            before_seen = int((state.seen_count > 0).sum().item())
            result = _process_sentence(
                state,
                tokenizer,
                text,
                sentence_index,
                max_tokens,
                relation_step,
                relation_floor,
                relation_cap,
                close_target,
                far_target,
                v_tolerance,
                v_check_steps,
                v_fit_steps,
                max_v_tests,
                max_extra_dims,
                relation_fit_steps,
            )
            after_seen = int((state.seen_count > 0).sum().item())
            stats["sentences"] += 1
            stats["new_vocab_tokens"] += max(0, after_seen - before_seen)
            stats["new_relations"] += int(result["new_relations"])
            stats["v_tests"] += int(result["v_tests"])
            stats["v_conflicts"] += int(result["v_conflicts"])
            if result["has_relationship"]:
                stats["relationship_sentences"] += 1
            if result["v_expanded"]:
                stats["v_expanded_sentences"] += 1
            record = {
                "sentence": sentence_index,
                **result,
                "expansion_events": state.expansion_events,
            }
            logs.append(record)
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()
            if sentence_index == 1 or sentence_index % log_every == 0:
                print(
                    f"sentence {sentence_index}/{len(texts)}: "
                    f"active_dim={state.active_dim}, "
                    f"relations={len(state.relation_subspaces)}, "
                    f"capacity={base.choose3(state.active_dim)}, "
                    f"v_tests={result['v_tests']}, "
                    f"v_conflicts={result['v_conflicts']}, "
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
        "stage": "relationship_familiarization_fullspace_v2",
        "description": "Full active-space resultant feasibility check before dimension expansion; relation scores use assigned coordinate triples.",
        "training_file": base._resolve_path(training_file),
        "base_dim": 3,
        "max_dim": max_dim,
        "active_dim": state.active_dim,
        "frozen_dim": state.frozen_dim,
        "relation_subspaces": len(state.relation_subspaces),
        "available_coordinate_triples": base.choose3(max_dim),
        "used_coordinate_triples": len(state.relation_subspaces),
        "expansion_events": state.expansion_events,
        "stats": stats,
        "hyperparameters": {
            "max_sentences": max_sentences,
            "relation_fit_steps": relation_fit_steps,
            "v_check_steps": v_check_steps,
            "v_fit_steps": v_fit_steps,
            "max_v_tests": max_v_tests,
            "max_extra_dims": max_extra_dims,
            "close_target": close_target,
            "far_target": far_target,
            "v_tolerance": v_tolerance,
        },
    }
    with open(
        os.path.join(resolved_output_dir, "relationship_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print(f"Saved V2 outputs to {resolved_output_dir}")
    print(
        f"Final active_dim={state.active_dim}, "
        f"relations={len(state.relation_subspaces)}, "
        f"coordinate_triples={base.choose3(state.active_dim)}"
    )
    return resolved_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sentences", type=int, default=100)
    parser.add_argument("--max-dim", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--relation-fit-steps", type=int, default=2)
    parser.add_argument("--v-check-steps", type=int, default=8)
    parser.add_argument("--v-fit-steps", type=int, default=12)
    parser.add_argument("--max-v-tests", type=int, default=8)
    parser.add_argument("--max-extra-dims", type=int, default=2)
    parser.add_argument("--close-target", type=float, default=0.65)
    parser.add_argument("--far-target", type=float, default=0.15)
    parser.add_argument("--v-tolerance", type=float, default=0.08)
    parser.add_argument("--relation-step", type=float, default=0.12)
    parser.add_argument("--relation-floor", type=float, default=0.30)
    parser.add_argument("--relation-cap", type=float, default=0.88)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
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
