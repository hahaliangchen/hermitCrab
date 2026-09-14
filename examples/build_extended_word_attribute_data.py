"""Build verified second-generation grammar slots from the Shiji corpus.

The original weak-label builder supplies lexical labels and the first local
slots.  This script adds SUBJECT, PASSIVE, NEGATION, CAUSATIVE, and CONDITION
examples, but keeps only samples whose intended structure is the structure
actually returned by the fixed automaton.  Ambiguous forms such as ``叫`` and
``被`` therefore do not silently become mislabeled training examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bert_simple.grammar_automaton import (  # noqa: E402
    ATTRIBUTES,
    CONDITION_OPENERS,
    FixedGrammarAutomaton,
    STRUCTURES,
    SUBJECT_PREDICATES,
)
from build_word_attribute_data import (  # noqa: E402
    FUNCTION_WORDS,
    PERSON_WORDS,
    PLACE_WORDS,
    PRONOUNS,
    TITLE_WORDS,
    _is_number,
    _is_place,
    _is_punctuation,
    mine_slots,
)


ACTION_WORDS = frozenset(
    (
        "攻打", "进攻", "攻下", "击败", "击溃", "杀死", "处死", "疏通", "治理",
        "平定", "收集", "召集", "召来", "围困", "俘获", "迁移", "逃走", "逃入",
        "逃到", "出发", "停留", "回到", "来到", "前往", "救援", "听取", "完成",
        "施行", "取得", "修炼", "歧视", "居住", "任命", "封", "立", "率领", "劝说",
        "答应", "回答", "认为", "决定", "成功", "失败", "做到", "改变",
    )
)
CAUSATIVE_TRIGGERS = frozenset(("派", "让", "使", "令", "命令", "叫"))
NEGATION_TRIGGERS = frozenset(("不", "未", "没有", "无", "未曾", "莫"))
CAUSATIVE_SERIAL_FOLLOWERS = frozenset(
    ("去", "来", "攻打", "进攻", "前往", "救援", "出发", "完成", "做")
)


def _is_known_entity(token: str) -> bool:
    return (
        token in PERSON_WORDS
        or token in PLACE_WORDS
        or token in TITLE_WORDS
        or token in PRONOUNS
        or _is_place(token)
    )


def _target_attributes(token: str) -> Tuple[str, ...]:
    if token in PERSON_WORDS or token in PRONOUNS:
        return ("PERSON", "ENTITY")
    if token in TITLE_WORDS:
        return ("PERSON", "TITLE")
    if token in PLACE_WORDS or _is_place(token):
        return ("PLACE", "ENTITY")
    if token in ACTION_WORDS:
        return ("ACTION",)
    if _is_number(token):
        return ("NUMBER",)
    if token in FUNCTION_WORDS:
        return ("FUNCTION",)
    return ("ENTITY",)


def _candidate(tokens: Sequence[str], index: int) -> bool:
    return (
        0 <= index < len(tokens)
        and tokens[index] != "[MASK]"
        and not _is_punctuation(tokens[index])
    )


def _append_verified(
    slots: List[Dict[str, object]],
    seen: Set[Tuple[object, ...]],
    automaton: FixedGrammarAutomaton,
    tokens: Sequence[str],
    target_index: int,
    structure: str,
    attributes: Iterable[str],
) -> bool:
    if not _candidate(tokens, target_index):
        return False
    masked = list(tokens)
    target = tokens[target_index]
    masked[target_index] = "[MASK]"
    state = automaton.analyze_text(" ".join(masked))
    if state.structure != structure:
        return False
    attrs = tuple(sorted(set(attributes), key=ATTRIBUTES.index))
    if not attrs:
        return False
    item = {
        "text": " ".join(masked),
        "target_token": target,
        "structure": structure,
        "attributes": list(attrs),
    }
    key = (item["text"], item["target_token"], item["structure"], tuple(attrs))
    if key in seen:
        return False
    seen.add(key)
    slots.append(item)
    return True


def _mine_extensions(
    lines: Sequence[str], labels: Dict[str, Set[str]], slots: List[Dict[str, object]]
) -> Counter:
    seen = {
        (item["text"], item["target_token"], item["structure"], tuple(item["attributes"]))
        for item in slots
    }
    counts: Counter = Counter()
    automaton = FixedGrammarAutomaton()

    for line in lines:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        for index, token in enumerate(tokens):
            if (
                _candidate(tokens, index)
                and index + 1 < len(tokens)
                and tokens[index + 1] in SUBJECT_PREDICATES
                and _is_known_entity(token)
                and _append_verified(
                    slots,
                    seen,
                    automaton,
                    tokens,
                    index,
                    "SUBJECT",
                    _target_attributes(token),
                )
            ):
                counts["SUBJECT"] += 1
                labels.setdefault(token, set()).update(_target_attributes(token))

            if token in NEGATION_TRIGGERS and _candidate(tokens, index + 1):
                target = tokens[index + 1]
                attrs = ("ENTITY",) if _is_known_entity(target) else ("ACTION",)
                if _append_verified(
                    slots, seen, automaton, tokens, index + 1, "NEGATION", attrs
                ):
                    counts["NEGATION"] += 1

            if token in CAUSATIVE_TRIGGERS and _candidate(tokens, index + 1):
                target = tokens[index + 1]
                if token == "叫":
                    if (
                        index + 2 >= len(tokens)
                        or tokens[index + 2] not in CAUSATIVE_SERIAL_FOLLOWERS
                    ):
                        continue
                if target not in FUNCTION_WORDS and target not in {"了", "着", "过"}:
                    attrs = _target_attributes(target)
                    if _append_verified(
                        slots, seen, automaton, tokens, index + 1, "CAUSATIVE", attrs
                    ):
                        counts["CAUSATIVE"] += 1
                        if _is_known_entity(target):
                            labels.setdefault(target, set()).update(attrs)

            if token == "被" and _candidate(tokens, index + 1):
                target = tokens[index + 1]
                attrs = _target_attributes(target)
                if _append_verified(
                    slots, seen, automaton, tokens, index + 1, "PASSIVE", attrs
                ):
                    counts["PASSIVE"] += 1
                    if _is_known_entity(target):
                        labels.setdefault(target, set()).update(attrs)

            if token in CONDITION_OPENERS and _candidate(tokens, index + 1):
                if "，" in tokens[index + 1 :]:
                    if _append_verified(
                        slots,
                        seen,
                        automaton,
                        tokens,
                        index + 1,
                        "CONDITION",
                        ("CLAUSE",),
                    ):
                        counts["CONDITION"] += 1

    return counts


def build(input_path: Path, output_path: Path, max_sentences: int = 0) -> Dict[str, object]:
    lines = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_sentences:
        lines = lines[:max_sentences]
    labels, slots = mine_slots(lines)
    extension_counts = _mine_extensions(lines, labels, slots)
    payload = {
        "metadata": {
            "source": str(input_path),
            "sentences": len(lines),
            "word_types_with_labels": len(labels),
            "slot_examples": len(slots),
            "attributes": list(ATTRIBUTES),
            "structures": list(STRUCTURES),
            "supervision": "weighted_lexicon_plus_verified_context_slots_v2",
            "lexicon_weight": 8,
            "slot_weight": 1,
            "extension_slot_counts": dict(extension_counts),
        },
        "word_labels": {
            token: sorted(values, key=ATTRIBUTES.index)
            for token, values in sorted(labels.items())
        },
        "slot_examples": slots,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/shiji/segmented/shiji_segmented.txt"))
    parser.add_argument("--output", default=str(ROOT / "data/shiji/manifests/word_attribute_dataset.json"))
    parser.add_argument("--max-sentences", type=int, default=0)
    args = parser.parse_args()
    payload = build(Path(args.input), Path(args.output), args.max_sentences)
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
