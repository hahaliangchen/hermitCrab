"""Build a v2 relation-pair JSONL draft from the existing fact manifest.

The fact manifest is a conservative, project-provided annotation of the
segmented Shiji material.  This converter keeps only two-fact source groups so
each emitted group has a real contrast pair, isolates every emitted long
context to one split, expands source variants at complete sentence boundaries,
and preserves the existing distractors as auditable hard negatives.  The
output is a draft for review, not a claim that every heuristic relation or
distractor is true.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata

from validate_relation_training_data import (
    DEFAULT_MIN_RELATION_TOKENS,
    DEFAULT_RELATION_MAX_LENGTH,
    validate_groups,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "shiji" / "manifests" / "fact_memory_dataset.json"
DEFAULT_SOURCE_TEXT = ROOT / "data" / "shiji" / "sentences" / "shiji_sentences.txt"
DEFAULT_OUTPUT = ROOT / "data" / "shiji" / "manifests" / "relation_training_v2_draft.jsonl"
DEFAULT_REPORT = ROOT / "data" / "shiji" / "manifests" / "relation_training_v2_draft_report.json"

SENTENCE_BOUNDARIES = frozenset(("。", "！", "？", "；", "!", "?", ";", "…", "……"))
CLOSING_TOKENS = frozenset(("”", "’", "\"", "'", "」", "』", "）", ")", "】", "]", "》"))
MIN_SOURCE_LINE_CHARS = 6

PUNCTUATION_VARIANTS = (
    ("，", ","),
    ("；", ";"),
    ("：", ":"),
    ("、", "/"),
    ("！", "!"),
    ("？", "?"),
)


def _load_rows(path: Path) -> List[Dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"fact manifest must be a non-empty JSON list: {path}")
    return rows


def _load_source_lines(path: Path) -> Dict[int, str]:
    """Load the original one-sentence-per-line corpus for split isolation."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line_number: line.strip() for line_number, line in enumerate(lines, 1) if line.strip()}


def _normalized(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split())


def _context_coverage(rows: Sequence[Dict[str, object]], source_lines: Dict[int, str]):
    """Map each fact to source sentence lines visible in its long context.

    The manifest split is attached to source lines, not to arbitrary windows.
    A long context that contains a line assigned to another split is therefore
    unsafe even when its fact ID and complete masked input are unique.
    """
    line_texts = {
        line_number: _normalized(text)
        for line_number, text in source_lines.items()
        if len(_normalized(text)) >= MIN_SOURCE_LINE_CHARS
    }
    line_splits = defaultdict(set)
    context_lines = defaultdict(set)
    for row in rows:
        source_line = int(row["source_line"])
        split = str(row["split"])
        line_splits[source_line].add(split)
        context_key = _normalized(str(row.get("context", "")))
        if len(context_key) >= MIN_SOURCE_LINE_CHARS:
            context_lines[context_key].add(source_line)

    coverage = {}
    for row in rows:
        fact_id = str(row["fact_id"])
        source_line = int(row["source_line"])
        long_context = _normalized(str(row.get("long_context", "")))
        covered = {source_line}
        for line_number, line_text in line_texts.items():
            if line_text in long_context:
                covered.add(line_number)
        for context_key, source_numbers in context_lines.items():
            if context_key in long_context:
                covered.update(source_numbers)
        coverage[fact_id] = covered
    return line_splits, coverage


def _context_conflict_lines(
    source_rows: Sequence[Dict[str, object]], line_splits, coverage
) -> List[int]:
    split = str(source_rows[0]["split"])
    covered = set().union(*(coverage[str(row["fact_id"])] for row in source_rows))
    return sorted(
        line_number
        for line_number in covered
        if any(other_split != split for other_split in line_splits.get(line_number, set()))
    )


def _find_subsequence(tokens: Sequence[str], pattern: Sequence[str]) -> List[Tuple[int, int]]:
    if not pattern or len(pattern) > len(tokens):
        return []
    result = []
    width = len(pattern)
    for start in range(len(tokens) - width + 1):
        if list(tokens[start : start + width]) == list(pattern):
            result.append((start, start + width))
    return result


def _role_spans(
    tokens: Sequence[str], roles: Dict[str, str], target_slot: str
) -> Dict[str, List[int]]:
    mask = tokens.index("[MASK]")
    occupied = set()
    spans: Dict[str, List[int]] = {}
    for role, value in roles.items():
        if role == target_slot:
            continue
        pattern = str(value).split()
        matches = [
            (start, end)
            for start, end in _find_subsequence(tokens, pattern)
            if not (start <= mask < end)
        ]
        # Ambiguous mentions are intentionally left unlabelled.  The v2
        # protocol allows partial role spans and this avoids asserting that a
        # background mention is the event anchor.
        if len(matches) != 1:
            continue
        start, end = matches[0]
        positions = set(range(start, end))
        if occupied & positions:
            continue
        spans[role] = [start, end]
        occupied |= positions
    return spans


def _hard_negatives(row: Dict[str, object], answer: str, target_slot: str):
    values = []
    for value in row.get("distractors", []) or []:
        value = str(value).strip()
        if value and value != answer and not any(c.isspace() for c in value) and value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{row['fact_id']}: no usable distractors")
    reasons = {
        value: (
            "原事实清单将该词列为同类型干扰项；当前事实的 "
            f"{target_slot} 标注为 {answer}。本批为自动转换草稿，需人工复核。"
        )
        for value in values
    }
    return values, reasons


def _fits_max_length(
    tokens: Sequence[str], max_length: int, min_tokens: int = 0
) -> bool:
    """Check content bounds and the length consumed by CLS/SEP."""
    return min_tokens <= len(tokens) and len(tokens) + 2 <= max_length


def _sentence_segments(tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Return contiguous sentence/clause segments including closing quotes."""
    segments = []
    start = 0
    index = 0
    while index < len(tokens):
        if tokens[index] in SENTENCE_BOUNDARIES:
            index += 1
            while index < len(tokens) and tokens[index] in CLOSING_TOKENS:
                index += 1
            segments.append((start, index))
            start = index
        else:
            index += 1
    if start < len(tokens):
        segments.append((start, len(tokens)))
    return segments


def _core_span(candidate_tokens: Sequence[str], base_tokens: Sequence[str]):
    matches = _find_subsequence(candidate_tokens, base_tokens)
    return matches[0] if len(matches) == 1 else None


def _background_spans(
    candidate_tokens: Sequence[str], base_tokens: Sequence[str]
) -> List[List[int]]:
    core = _core_span(candidate_tokens, base_tokens)
    if core is None:
        return []
    start, end = core
    spans = []
    if start:
        spans.append([0, start])
    if end < len(candidate_tokens):
        spans.append([end, len(candidate_tokens)])
    return spans


def _meaningful_background(
    candidate_tokens: Sequence[str], base_tokens: Sequence[str]
) -> bool:
    """Require an added sentence/clause, not a lone punctuation token."""
    core = _core_span(candidate_tokens, base_tokens)
    if core is None:
        return False
    start, end = core
    added = list(candidate_tokens[:start]) + list(candidate_tokens[end:])
    return (
        any(token in SENTENCE_BOUNDARIES for token in added)
        and any(token not in SENTENCE_BOUNDARIES | CLOSING_TOKENS for token in added)
    )


def _source_context_variant(
    text: str,
    row: Dict[str, object],
    max_length: int,
    min_tokens: int = 0,
    excluded_texts: Sequence[str] = (),
):
    """Find the largest fitting window made of complete source segments."""
    base_tokens = text.split()
    mask_index = base_tokens.index("[MASK]")
    restored = list(base_tokens)
    restored[mask_index] = str(row["answer"])
    long_tokens = str(row["long_context"]).split()
    segments = _sentence_segments(long_tokens)
    excluded = {str(value) for value in excluded_texts}
    if not segments:
        return None

    candidates = []
    for start, end in _find_subsequence(long_tokens, restored):
        containing = [
            segment_index
            for segment_index, (segment_start, segment_end) in enumerate(segments)
            if segment_start <= start and end <= segment_end
        ]
        if not containing:
            continue
        base_segment = containing[0]
        for left_segment in range(base_segment + 1):
            for right_segment in range(base_segment, len(segments)):
                if left_segment == base_segment and right_segment == base_segment:
                    continue
                candidate_start = segments[left_segment][0]
                candidate_end = segments[right_segment][1]
                candidate = long_tokens[candidate_start:candidate_end]
                if not _fits_max_length(candidate, max_length, min_tokens):
                    continue
                local_mask = start - candidate_start + mask_index
                if local_mask < 0 or local_mask >= len(candidate):
                    continue
                if any(
                    token == str(row["answer"])
                    for index, token in enumerate(candidate)
                    if index != local_mask
                ):
                    continue
                candidate[local_mask] = "[MASK]"
                candidate_text = " ".join(candidate)
                if candidate_text in excluded:
                    continue
                if not _meaningful_background(candidate, base_tokens):
                    continue
                candidates.append((len(candidate), right_segment - left_segment, candidate))

    if not candidates:
        return None
    # Prefer the strongest complete context that fits the configured limit.
    _, _, candidate = max(candidates, key=lambda item: (item[0], item[1]))
    return " ".join(candidate), _background_spans(candidate, base_tokens)


def _surface_variant(
    text: str,
    row: Dict[str, object],
    max_length: int,
    min_tokens: int = 0,
    excluded_texts: Sequence[str] = (),
):
    base_tokens = text.split()
    excluded = {str(value) for value in excluded_texts}
    for variant in row.get("variants", []) or []:
        candidate = str(variant.get("masked_text", ""))
        candidate_tokens = candidate.split()
        if (
            candidate
            and candidate != text
            and candidate_tokens.count("[MASK]") == 1
            and candidate not in excluded
            and _fits_max_length(candidate_tokens, max_length, min_tokens)
            and _meaningful_background(candidate_tokens, base_tokens)
        ):
            return candidate, "source", "existing_variant", _background_spans(candidate_tokens, base_tokens)

    source_variant = _source_context_variant(
        text, row, max_length, min_tokens, excluded_texts=excluded
    )
    if source_variant is not None:
        candidate, background_spans = source_variant
        return candidate, "source", "context_window", background_spans

    for old, new in PUNCTUATION_VARIANTS:
        if old in text:
            candidate = text.replace(old, new, 1)
            if (
                candidate not in excluded
                and _fits_max_length(candidate.split(), max_length, min_tokens)
            ):
                return candidate, "paraphrase", "punctuation_variant", []
    raise ValueError(f"{row['fact_id']}: could not make a distinct source variant")


def _canonical_rows(rows: Sequence[Dict[str, object]]):
    """Return (source row, fact metadata) pairs with stable v2 role keys."""
    if len(rows) != 2:
        return None

    if any(str(row["relation_type"]) == "APPOINT" for row in rows):
        if not all(str(row["relation_type"]) == "APPOINT" for row in rows):
            return None
        by_target = {str(row["target_role"]): row for row in rows}
        if set(by_target) != {"object", "role"}:
            return None
        recipient_row = by_target["object"]
        title_row = by_target["role"]
        shared = ("subject", "object", "relation_surface")
        if any(recipient_row[key] != title_row[key] for key in shared):
            return None
        title = str(title_row["answer"])
        result = []
        for row in rows:
            result.append(
                (
                    row,
                    {
                        "relation_type": "APPOINT",
                        "roles": {
                            "actor": str(row["subject"]),
                            "predicate": str(row["relation_surface"]),
                            "recipient": str(row["object"]),
                            "title": title,
                        },
                        "target_slot": "title"
                        if str(row["target_role"]) == "role"
                        else "recipient",
                        "relation_key": {
                            "relation_type": "APPOINT",
                            "source_slot": "actor",
                            "target_slot": "recipient",
                            "direction": "actor_to_recipient",
                        },
                    },
                )
            )
        return result

    if any(str(row["target_role"]) not in {"subject", "object"} for row in rows):
        return None
    result = []
    for row in rows:
        relation_type = str(row["relation_type"])
        result.append(
            (
                row,
                {
                    "relation_type": relation_type,
                    "roles": {
                        "subject": str(row["subject"]),
                        "predicate": str(row["relation_surface"]),
                        "object": str(row["object"]),
                    },
                    "target_slot": str(row["target_role"]),
                    "relation_key": {
                        "relation_type": relation_type,
                        "source_slot": "subject",
                        "target_slot": "object",
                        "direction": "subject_to_object",
                    },
                },
            )
        )
    return result


def _fact_entry(row: Dict[str, object], metadata: Dict[str, object]) -> Dict[str, object]:
    source_id = f"data/shiji/sentences/shiji_sentences.txt#line={int(row['source_line'])}"
    return {
        "fact_id": str(row["fact_id"]),
        "relation_type": metadata["relation_type"],
        "roles": metadata["roles"],
        "qualifiers": {},
        "relation_key": metadata["relation_key"],
        "evidence": {
            "kind": "provided_source",
            "source_id": source_id,
            # long_context contains both the short source sentence and the
            # existing long-context variants, while remaining auditable.
            "quote": str(row["long_context"]),
        },
    }


def _sample(
    row: Dict[str, object],
    metadata: Dict[str, object],
    text: str,
    surface_kind: str,
    sample_id: str,
    background_spans: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, object]:
    tokens = text.split()
    answer = str(row["answer"])
    target_slot = str(metadata["target_slot"])
    negatives, reasons = _hard_negatives(row, answer, target_slot)
    role_spans = _role_spans(tokens, metadata["roles"], target_slot)
    occupied = {index for start, end in role_spans.values() for index in range(start, end)}
    clean_background = []
    for start, end in background_spans or []:
        cursor = start
        for index in sorted(position for position in occupied if start <= position < end):
            if cursor < index:
                clean_background.append([cursor, index])
            cursor = index + 1
        if cursor < end:
            clean_background.append([cursor, end])
    return {
        "sample_id": sample_id,
        "fact_id": str(row["fact_id"]),
        "tokens": tokens,
        "answer": answer,
        "target_slot": target_slot,
        "surface_kind": surface_kind,
        "role_spans": role_spans,
        "background_spans": clean_background,
        "hard_negatives": negatives,
        "negative_reasons": reasons,
    }


def _changed_slots(first, second, first_sample, second_sample):
    changed = []
    if first["relation_type"] != second["relation_type"]:
        changed.append("relation_type")
    for role in sorted(set(first["roles"]) | set(second["roles"])):
        if first["roles"].get(role) != second["roles"].get(role):
            changed.append(f"roles.{role}")
    for qualifier in sorted(set(first.get("qualifiers", {})) | set(second.get("qualifiers", {}))):
        if first.get("qualifiers", {}).get(qualifier) != second.get("qualifiers", {}).get(qualifier):
            changed.append(f"qualifiers.{qualifier}")
    if first_sample["target_slot"] != second_sample["target_slot"]:
        changed.append("target_slot")
    if not changed:
        raise ValueError("contrast facts have different answers but no changed field")
    return changed


def build(
    rows: Sequence[Dict[str, object]],
    max_length: int,
    source_lines: Optional[Dict[int, str]] = None,
    min_tokens: int = DEFAULT_MIN_RELATION_TOKENS,
):
    if min_tokens < 1:
        raise ValueError("min_tokens must be positive")
    if max_length < min_tokens + 2:
        raise ValueError("max_length must leave room for min_tokens plus CLS/SEP")
    if source_lines is None:
        source_lines = _load_source_lines(DEFAULT_SOURCE_TEXT)
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["fact_group_id"])].append(row)

    groups = []
    skipped = Counter()
    variant_counts = Counter()
    line_splits, context_coverage = _context_coverage(rows, source_lines)
    # The source manifest contains a small number of duplicate semantic facts
    # on different source lines.  Remove every group containing such a fact
    # before validation; otherwise the v2 split-isolation rule would be
    # violated even though source_group itself is correctly assigned.
    semantic_splits = defaultdict(set)
    canonical_by_group = {}
    for fact_group_id, source_rows in grouped.items():
        canonical = _canonical_rows(source_rows) if len(source_rows) == 2 else None
        canonical_by_group[fact_group_id] = canonical
        if canonical is None:
            continue
        for _, metadata in canonical:
            key = (
                metadata["relation_type"],
                tuple(sorted((role, _normalized(value)) for role, value in metadata["roles"].items())),
            )
            semantic_splits[key].add(str(source_rows[0]["split"]))

    for fact_group_id, source_rows in sorted(grouped.items()):
        if len(source_rows) != 2:
            skipped["not_two_fact_group"] += 1
            continue
        if len({str(row["split"]) for row in source_rows}) != 1:
            skipped["mixed_split"] += 1
            continue
        canonical = canonical_by_group[fact_group_id]
        if canonical is None:
            skipped["unsupported_or_incomplete_group"] += 1
            continue
        if any(
            len(
                semantic_splits[
                    (
                        metadata["relation_type"],
                        tuple(sorted((role, _normalized(value)) for role, value in metadata["roles"].items())),
                    )
                ]
            )
            > 1
            for _, metadata in canonical
        ):
            skipped["semantic_fact_cross_split"] += 1
            continue
        conflict_lines = _context_conflict_lines(source_rows, line_splits, context_coverage)
        if conflict_lines:
            skipped["context_cross_split"] += 1
            continue
        if any(
            not _fits_max_length(str(row["masked_text"]).split(), max_length)
            for row in source_rows
        ):
            skipped["max_length_exceeded"] += 1
            continue

        facts = []
        samples = []
        variants = []
        variant_failed = False
        for row, metadata in canonical:
            facts.append(_fact_entry(row, metadata))
            base_id = f"{row['fact_id']}:base"
            variant_id = f"{row['fact_id']}:variant"
            original_text = str(row["masked_text"])
            base_text = original_text
            base_background_spans = []
            expanded = _source_context_variant(
                original_text, row, max_length, min_tokens
            )
            if expanded is not None and len(expanded[0].split()) > len(original_text.split()):
                base_text, base_background_spans = expanded
            if not _fits_max_length(base_text.split(), max_length, min_tokens):
                skipped["min_length_exceeded"] += 1
                variant_failed = True
                break
            base_sample = _sample(
                row, metadata, base_text, "source", base_id, base_background_spans
            )
            try:
                variant_text, variant_kind, variant_origin, background_spans = _surface_variant(
                    original_text,
                    row,
                    max_length,
                    min_tokens,
                    excluded_texts=(base_text,),
                )
            except ValueError:
                skipped["no_valid_variant"] += 1
                variant_failed = True
                break
            if not _fits_max_length(variant_text.split(), max_length, min_tokens):
                skipped["min_length_exceeded"] += 1
                variant_failed = True
                break
            variant_sample = _sample(
                row, metadata, variant_text, variant_kind, variant_id, background_spans
            )
            samples.extend((base_sample, variant_sample))
            variants.append((base_sample, variant_sample))
            variant_counts[variant_origin] += 1

        if variant_failed:
            continue

        first_fact, second_fact = canonical[0][1], canonical[1][1]
        first_base, first_variant = variants[0]
        second_base, second_variant = variants[1]
        source_line = int(source_rows[0]["source_line"])
        groups.append(
            {
                "schema_version": 2,
                "group_id": f"shiji-rel-v2-{fact_group_id}",
                "source_group": f"shiji-source-line-{source_line}",
                "split": str(source_rows[0]["split"]),
                "facts": facts,
                "samples": samples,
                "pairs": [
                    {
                        "a": first_base["sample_id"],
                        "b": first_variant["sample_id"],
                        "kind": "invariance",
                        "changed_slots": [],
                    },
                    {
                        "a": second_base["sample_id"],
                        "b": second_variant["sample_id"],
                        "kind": "invariance",
                        "changed_slots": [],
                    },
                    {
                        "a": first_base["sample_id"],
                        "b": second_base["sample_id"],
                        "kind": "contrast",
                        "changed_slots": _changed_slots(
                            first_fact, second_fact, first_base, second_base
                        ),
                    },
                ],
            }
        )
    return groups, skipped, variant_counts


def _report(
    groups,
    skipped,
    variant_counts,
    source_path: Path,
    source_text_path: Path,
    max_length: int,
    min_tokens: int,
    validation,
):
    relation_counts = Counter()
    split_counts = Counter()
    surface_counts = Counter()
    for group in groups:
        split_counts[group["split"]] += 1
        for fact in group["facts"]:
            relation_counts[fact["relation_type"]] += 1
        for sample in group["samples"]:
            surface_counts[sample["surface_kind"]] += 1
    return {
        "schema_version": 2,
        "status": "draft_from_existing_fact_manifest_requires_manual_audit",
        "source_manifest": str(source_path),
        "source_text": str(source_text_path),
        "max_length": max_length,
        "min_tokens": min_tokens,
        "groups": len(groups),
        "facts": sum(len(group["facts"]) for group in groups),
        "samples": sum(len(group["samples"]) for group in groups),
        "split_groups": dict(sorted(split_counts.items())),
        "relation_type_fact_counts": dict(sorted(relation_counts.items())),
        "surface_kind_sample_counts": dict(sorted(surface_counts.items())),
        "variant_origin_counts": dict(sorted(variant_counts.items())),
        "skipped_groups": dict(sorted(skipped.items())),
        "validator": validation,
        "notes": [
            "Facts and source text are inherited from the existing conservative manifest.",
            "Distractors are inherited and receive an audit-required reason; they are not independently proven here.",
            "qualifiers are empty because the source manifest has no structured qualifier fields.",
            "This draft is suitable for protocol/smoke training after review, not a final complete Shiji relation annotation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-text", type=Path, default=DEFAULT_SOURCE_TEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-length", type=int, default=DEFAULT_RELATION_MAX_LENGTH)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_RELATION_TOKENS)
    args = parser.parse_args()
    if args.min_tokens < 1:
        raise ValueError("min-tokens must be positive")
    if args.max_length < args.min_tokens + 2:
        raise ValueError("max-length must leave room for min-tokens plus CLS/SEP")

    rows = _load_rows(args.input)
    source_lines = _load_source_lines(args.source_text)
    groups, skipped, variant_counts = build(
        rows, args.max_length, source_lines, args.min_tokens
    )
    validation = validate_groups(groups, min_tokens=args.min_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(group, ensure_ascii=False, separators=(",", ":")) + "\n" for group in groups),
        encoding="utf-8",
    )
    report = _report(
        groups,
        skipped,
        variant_counts,
        args.input,
        args.source_text,
        args.max_length,
        args.min_tokens,
        validation,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
