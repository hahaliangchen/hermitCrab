"""Build the verified Shiji fact-memory dataset.

The extraction implementation is kept in build_shiji_fact_dataset_impl.py.
This entry point installs conservative target/origin guards, normalizes
legacy attribute labels, and makes the report JSON-safe before invoking it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import build_shiji_fact_dataset_impl as implementation


implementation.TITLE_WORDS.update(
    {"将", "上卿", "廷尉", "典客", "中大夫", "郎中令", "侍中", "庶长"}
)


def _strict_token_attributes(
    token: str, default: Optional[str] = None
) -> Sequence[str]:
    """Prefer current lexical evidence over stale weak-label attributes."""

    old = set(implementation.ATTRIBUTE_LEXICON.get(token, set()))
    if token in implementation.GROUP_WORDS:
        labels = {"GROUP", "ENTITY"}
    elif token in implementation.TITLE_WORDS:
        labels = (old & {"PERSON", "ORG"}) | {"TITLE", "ENTITY"}
    elif token in implementation.PERSON_NAMES:
        labels = (old & {"TITLE", "ORG"}) | {"PERSON", "ENTITY"}
    elif token in implementation.PLACE_NAMES or token.endswith(
        ("国", "县", "郡", "城", "关")
    ):
        labels = (old & {"ORG"}) | {"PLACE", "ENTITY"}
    elif token in implementation.ACTION_WORDS:
        labels = {"ACTION", "ENTITY"}
    elif default:
        labels = {default}
        if default not in {"ACTION", "NUMBER"}:
            labels.add("ENTITY")
    else:
        labels = old or {"ENTITY"}
    return sorted(labels)


implementation.token_attributes = _strict_token_attributes


_BOUNDARY_AFTER_COMMAND = {
    "去", "来", "到", "往", "向", "在", "送", "报告", "寻找", "制造", "率领",
    "携带", "召集", "出使", "游说", "行刺", "打听", "迎接", "收集", "联系",
    "侍奉", "做", "喝", "治理", "攻打", "进攻", "夺取", "平定", "追击",
    "用", "拿", "带", "率", "随", "从", "把", "给", "和", "与", "并",
}


def _strict_find_causative_target(
    tokens: Sequence[str], verb_index: int, tag_map: Dict[str, str]
) -> Optional[int]:
    """Find only the command object, never a later action/object."""

    fallback: Optional[int] = None
    end = min(len(tokens), verb_index + 5)
    for index in range(verb_index + 1, end):
        token = tokens[index]
        if token in implementation.PUNCTUATION or token in _BOUNDARY_AFTER_COMMAND:
            break
        if token in {"自己", "他们", "大家", "有人", "人", "们"}:
            if token == "人" and fallback is None:
                return None
            continue
        attrs = set(implementation.token_attributes(token))
        if token in implementation.PERSON_NAMES or (
            "PERSON" in attrs and "GROUP" not in attrs
        ):
            return index
        if "TITLE" in attrs or "GROUP" in attrs:
            if fallback is None:
                fallback = index
            continue
        # A one-character unknown token after a role is often the person's
        # name (e.g. 太子 申).  The current one-token MLM schema cannot mask
        # the complete multi-token entity safely, so reject the candidate.
        if fallback is not None and len(token) == 1:
            return None
        return fallback
    return fallback


def _valid_origin_shape(fact: Dict) -> bool:
    tokens = fact["context"].split()
    try:
        person_index = tokens.index(fact["subject"])
    except ValueError:
        return False
    place_index = person_index + 2
    if (
        person_index + 1 >= len(tokens)
        or tokens[person_index + 1] not in implementation.ORIGIN_WORDS
        or place_index >= len(tokens)
        or tokens[place_index] != fact["object"]
    ):
        return False
    tail = []
    for token in tokens[place_index + 1 :]:
        if token in implementation.SENTENCE_END:
            break
        tail.append(token)
    return any(token in {"人", "名士", "门客", "之后"} for token in tail)


def _valid_fact_target_type(fact: Dict) -> bool:
    attrs = set(fact["answer_attributes"])
    if fact["grammar"] == "APPOINTMENT":
        if fact["target_role"] == "object":
            return "PERSON" in attrs and "PLACE" not in attrs
        return "TITLE" in attrs
    if fact["grammar"] == "ORIGIN":
        if fact["target_role"] == "subject":
            return "PERSON" in attrs
        return "PLACE" in attrs
    if fact["grammar"] == "LOCATION_EVENT":
        return "PLACE" in attrs
    return bool(attrs & {"PERSON", "PLACE", "GROUP", "TITLE"})


implementation.find_causative_target = _strict_find_causative_target
_extract_all_facts = implementation.extract_all_facts


def _strict_extract_all_facts(sentences):
    facts = _extract_all_facts(sentences)
    return [
        fact
        for fact in facts
        if (fact["grammar"] != "ORIGIN" or _valid_origin_shape(fact))
        and _valid_fact_target_type(fact)
    ]


implementation.extract_all_facts = _strict_extract_all_facts
_build_report = implementation.build_report


def _json_safe_report(*args, **kwargs):
    report = _build_report(*args, **kwargs)
    report["attribute_distribution"] = {
        str(key): value for key, value in report["attribute_distribution"].items()
    }
    return report


implementation.build_report = _json_safe_report


if __name__ == "__main__":
    implementation.main()
