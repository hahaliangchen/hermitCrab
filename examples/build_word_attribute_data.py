"""Build weighted weak supervision for learnable word attributes.

The original prototype unioned every weak label.  That made a place name
occasionally become PERSON or TITLE merely because a noisy trigger matched.
This version keeps evidence weights: curated lexical evidence is strong, and
slot-mining evidence is weaker.  Conflicting weak labels are therefore not
blindly merged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

ATTRIBUTES = (
    "PLACE", "TIME", "PERSON", "ORG", "TITLE", "NUMBER", "ACTION",
    "ENTITY", "FUNCTION", "PUNCT", "INTERROGATIVE", "CLAUSE", "UNKNOWN",
)
STRUCTURES = (
    "NAME", "LOCATION", "TEMPORAL", "PREDICATE", "OBJECT", "IDENTITY",
    "QUANTITY", "CONDITION", "COORDINATION", "PARTICLE", "BOUNDARY", "OPEN",
)

PERSON_WORDS = {
    "司马迁", "张耳", "陈余", "刘邦", "项羽", "范增", "韩信", "陈胜", "吴广",
    "蒯通", "萧何", "曹参", "樊哙", "张良", "吕不韦", "嬴政", "蒙恬", "李斯",
    "扁鹊", "屈原", "贾谊", "廉颇", "蔺相如", "赵括", "白起", "孙武", "孙膑",
    "庞涓", "伍子胥", "勾践", "夫差", "齐桓公", "晋文公", "楚庄王", "秦穆公",
    "宋襄公", "商汤", "夏桀", "纣王", "周武王", "周文王", "姜尚", "姜子牙",
    "伯夷", "叔齐", "卫青", "霍去病", "晁错", "袁盎", "魏公子", "信陵君",
    "平原君", "孟尝君", "春申君", "高祖", "武信君", "武臣", "秦始皇", "秦二世",
    "胡亥", "扶苏", "子婴", "陈涉", "章邯", "英布", "彭越", "周勃", "灌婴",
    "夏侯婴", "陆贾", "郦食其", "随何", "大禹", "启", "舜", "尧", "鲧",
    "葛伯", "伊尹", "商鞅", "郑国",
}
PLACE_WORDS = {
    "沛县", "外黄", "外黄县", "苦陉", "苦陉县", "大梁", "咸阳", "邯郸", "临淄",
    "郢都", "姑苏", "鸿门", "乌江", "垓下", "巨鹿", "荥阳", "彭城", "函谷关",
    "白马津", "渔阳", "大泽乡", "范阳", "范阳城", "楚国", "赵国", "魏国", "韩国",
    "燕国", "齐国", "秦国", "蜀地", "汉中", "陈县", "蕲县", "河北", "河南",
    "黄河", "长江", "淮水", "昆吾", "豕韦", "葛国",
}
TITLE_WORDS = {
    "门客", "县令", "校尉", "相国", "丞相", "将军", "太尉", "楚王", "汉王", "诸侯",
    "豪杰", "侍从", "宾客", "臣子", "国君", "军师", "刺客", "谋士", "士卒", "部下",
    "屯长", "大夫", "太史令", "使者", "父老", "接班人", "富家女", "大王",
}
FUNCTION_WORDS = {
    "在", "于", "从", "向", "到", "去", "来", "回", "的", "之", "者", "所",
    "和", "与", "及", "同", "而", "但", "却", "是", "为", "如果", "若", "因为",
    "所以", "只要", "没有", "不", "未", "还", "已经", "正在", "呢", "吗",
}
TIME_WORDS = {"时", "时候", "期间", "年", "月", "日", "夜间", "晚上"}
PRONOUNS = {"我", "你", "他", "她", "它", "谁", "人"}
PUNCTUATION = set("，。！？；：、‘’“”「」『』（）()《》〈〉")
NUMBER_RE = re.compile(r"^[零一二三四五六七八九十百千万亿两数几多约\d]+$")


def _is_punctuation(token: str) -> bool:
    return bool(token) and all(char in PUNCTUATION for char in token)


def _is_number(token: str) -> bool:
    return bool(NUMBER_RE.fullmatch(token)) or token in {
        "三十万", "七十多万", "好几万", "数万", "数十", "几十", "十四", "十三",
    }


def _is_place(token: str) -> bool:
    return token in PLACE_WORDS or token.endswith(("县", "郡", "城", "国", "地", "关", "津", "乡"))


def _add_evidence(
    evidence: Dict[str, Counter], token: str, attributes: Iterable[str], weight: int
) -> None:
    for attribute in attributes:
        evidence[token][attribute] += weight


def _initial_evidence(token: str, evidence: Dict[str, Counter]) -> None:
    if _is_punctuation(token):
        _add_evidence(evidence, token, ("PUNCT",), 8)
    if token in PERSON_WORDS:
        _add_evidence(evidence, token, ("PERSON", "ENTITY"), 8)
    if _is_place(token):
        _add_evidence(evidence, token, ("PLACE", "ENTITY"), 8)
    if token in TITLE_WORDS:
        _add_evidence(evidence, token, ("TITLE", "PERSON"), 8)
    if token in FUNCTION_WORDS:
        _add_evidence(evidence, token, ("FUNCTION",), 8)
    if token in TIME_WORDS:
        _add_evidence(evidence, token, ("TIME",), 8)
    if _is_number(token):
        _add_evidence(evidence, token, ("NUMBER",), 8)


def _candidate_token(tokens: Sequence[str], index: int) -> bool:
    return 0 <= index < len(tokens) and not _is_punctuation(tokens[index])


def _add_slot(
    slots: List[Dict[str, object]],
    evidence: Dict[str, Counter],
    tokens: Sequence[str],
    target_index: int,
    structure: str,
    attributes: Iterable[str],
) -> None:
    if not _candidate_token(tokens, target_index):
        return
    target = tokens[target_index]
    attrs = tuple(sorted(set(attributes), key=ATTRIBUTES.index))
    if not attrs:
        return
    masked = list(tokens)
    masked[target_index] = "[MASK]"
    slots.append({
        "text": " ".join(masked),
        "target_token": target,
        "structure": structure,
        "attributes": list(attrs),
    })
    # A mined context slot is useful supervision, but weaker than a curated
    # entity/function lexicon.  The weighted resolver below can reject it
    # when it conflicts with strong lexical evidence.
    _add_evidence(evidence, target, attrs, 1)


def mine_slots(lines: Sequence[str]) -> Tuple[Dict[str, Set[str]], List[Dict[str, object]]]:
    evidence: Dict[str, Counter] = defaultdict(Counter)
    slots: List[Dict[str, object]] = []
    name_triggers = {"叫", "名叫", "名为", "姓名"}
    location_triggers = {"在", "住在", "前往", "到达", "去"}
    predicate_triggers = {"正在", "已经", "开始", "还在", "企图", "决定"}
    object_triggers = {"叫", "让", "请", "派", "命令", "协助", "任用"}
    identity_triggers = {"是", "为", "担任", "成为", "出任", "立为", "封为", "任命为"}
    quantity_triggers = {"有", "带了", "共有", "拥有", "率领", "斩首", "俘获"}
    time_markers = TIME_WORDS | {"正月", "秋天", "春天", "冬天", "夏天"}

    for line in lines:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        for token in tokens:
            _initial_evidence(token, evidence)
        for index, token in enumerate(tokens):
            if token in name_triggers and _candidate_token(tokens, index + 1):
                target = tokens[index + 1]
                if target in PERSON_WORDS:
                    _add_slot(slots, evidence, tokens, index + 1, "NAME", ("PERSON", "ENTITY"))

            if token in location_triggers and _candidate_token(tokens, index + 1):
                target = tokens[index + 1]
                if index + 2 < len(tokens) and tokens[index + 2] in time_markers:
                    _add_slot(slots, evidence, tokens, index + 1, "TEMPORAL", ("TIME",))
                elif _is_place(target):
                    _add_slot(slots, evidence, tokens, index + 1, "LOCATION", ("PLACE", "ENTITY"))

            if token in predicate_triggers and _candidate_token(tokens, index + 1):
                target = tokens[index + 1]
                if target not in FUNCTION_WORDS and target not in PERSON_WORDS and not _is_place(target):
                    _add_slot(slots, evidence, tokens, index + 1, "PREDICATE", ("ACTION",))

            if token in object_triggers and _candidate_token(tokens, index + 1):
                target = tokens[index + 1]
                if target in PERSON_WORDS or target in PRONOUNS:
                    _add_slot(slots, evidence, tokens, index + 1, "OBJECT", ("PERSON", "ENTITY"))

            if token in identity_triggers:
                target_index = index + 1
                if target_index < len(tokens) and tokens[target_index] in {"一位", "一名", "一个"}:
                    target_index += 1
                if _candidate_token(tokens, target_index):
                    target = tokens[target_index]
                    if target in TITLE_WORDS:
                        _add_slot(slots, evidence, tokens, target_index, "IDENTITY", ("TITLE", "ENTITY"))

            if token in quantity_triggers and _candidate_token(tokens, index + 1):
                if _is_number(tokens[index + 1]):
                    _add_slot(slots, evidence, tokens, index + 1, "QUANTITY", ("NUMBER",))

            if token in {"和", "与", "及", "同"} and index + 1 < len(tokens):
                if tokens[index + 1] in {"一起", "一同", "共同", "结伴"}:
                    _add_slot(slots, evidence, tokens, index, "COORDINATION", ("FUNCTION",))

            if token in {"哪", "哪里", "谁", "什么", "何处", "何时"}:
                _add_slot(slots, evidence, tokens, index, "OPEN", ("INTERROGATIVE",))

        if tokens[-1] in PUNCTUATION:
            _add_slot(slots, evidence, tokens, len(tokens) - 1, "BOUNDARY", ("PUNCT",))

    unique_slots = {}
    for slot in slots:
        key = (slot["text"], slot["target_token"], slot["structure"], tuple(slot["attributes"]))
        unique_slots[key] = slot

    labels: Dict[str, Set[str]] = {}
    for token, scores in evidence.items():
        if not scores:
            continue
        maximum = max(scores.values())
        strong = any(score >= 8 for score in scores.values())
        threshold = max(2, maximum // 4) if strong else maximum
        selected = {attribute for attribute, score in scores.items() if score >= threshold}
        if selected:
            labels[token] = selected
    return labels, list(unique_slots.values())


def build(input_path: Path, output_path: Path, max_sentences: int = 0) -> Dict[str, object]:
    lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_sentences:
        lines = lines[:max_sentences]
    labels, slots = mine_slots(lines)
    payload = {
        "metadata": {
            "source": str(input_path),
            "sentences": len(lines),
            "word_types_with_labels": len(labels),
            "slot_examples": len(slots),
            "attributes": list(ATTRIBUTES),
            "structures": list(STRUCTURES),
            "supervision": "weighted_lexicon_plus_context_slot_weak_labels",
            "lexicon_weight": 8,
            "slot_weight": 1,
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

