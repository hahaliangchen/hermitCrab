"""Bootstrap grammar attributes, roles, and sparse active pairs from 173 Shiji groups.

Zero manual labeling:
1. High-confidence gold roles (subject, predicate, object) from sample facts.
2. High-frequency ancient/modern Chinese grammatical function words & punctuation.
3. Named entities, titles, and locations bootstrapped directly from unit facts.
4. Active candidate pair mask: cuts 99%+ junk pairs (e.g. 是-的, 为-者, 以-，)
   so the 18D relational FFN computes ONLY on valid entity candidate pairs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from bert_simple.tokenizer import SimpleBertTokenizer
from examples.validate_relation_training_data import load_groups

ATTRIBUTES = (
    "PLACE",
    "TIME",
    "PERSON",
    "ORG",
    "TITLE",
    "NUMBER",
    "ACTION",
    "ENTITY",
    "FUNCTION",
    "PUNCT",
    "INTERROGATIVE",
    "CLAUSE",
    "UNKNOWN",
)

STRUCTURES = (
    "SUBJECT",
    "PREDICATE",
    "OBJECT",
    "IDENTITY",
    "LOCATION",
    "TEMPORAL",
    "QUANTITY",
    "FUNCTION",
    "BOUNDARY",
    "OTHER",
)

PUNCTUATION = set("，。！？；：、‘’“”「」『』（）()《》〈〉—…-·\"'")

FUNCTION_WORDS = {
    "在", "于", "从", "向", "到", "去", "来", "回", "的", "之", "者", "所",
    "和", "与", "及", "同", "而", "但", "却", "是", "为", "如果", "若", "因为",
    "所以", "只要", "没有", "不", "未", "还", "已经", "正在", "呢", "吗", "以",
    "其", "也", "则", "且", "乃", "乎", "虽", "然", "故", "因", "即", "遂",
    "夫", "盖", "矣", "焉", "兮", "哉", "欤", "耶", "纵", "倘", "既", "便",
    "犹", "尚", "曾", "方", "正", "适", "俱", "皆", "咸", "悉", "莫", "毋",
    "勿", "弗", "非", "无", "否", "岂", "宁", "庸", "孰", "何", "曷", "奚",
    "盍", "安", "恶", "乌", "胡", "由", "沿", "自", "按", "据", "照", "把",
    "将", "被", "蒙", "让", "给", "跟", "同", "共", "借", "凭", "任",
    "随", "除", "趁", "乘", "关于", "对于", "至于", "连", "乃至", "从而",
    "后", "中", "前", "上", "下", "内", "外", "间", "等", "又", "就", "才",
    "最", "更", "很", "大", "已", "并", "或", "各", "自", "互", "皆",
}

TIME_WORDS = {
    "时", "时候", "期间", "年", "月", "日", "夜间", "晚上", "少年时期", "当初",
    "最后", "一时", "此后", "最后", "十一月", "十二月", "三年", "八年", "九年",
    "春天", "夏天", "秋天", "冬天", "正月", "秦末", "灭秦后", "时年",
}

PRONOUNS = {"我", "你", "他", "她", "它", "他们", "她们", "它们", "谁", "人", "自己", "其"}

BASE_PERSON_WORDS = {
    "司马迁", "张耳", "陈余", "刘邦", "项羽", "范增", "韩信", "陈胜", "吴广",
    "蒯通", "萧何", "曹参", "樊哙", "张良", "吕不韦", "嬴政", "蒙恬", "李斯",
    "扁鹊", "屈原", "贾谊", "廉颇", "蔺相如", "赵括", "白起", "孙武", "孙膑",
    "庞涓", "伍子胥", "勾践", "夫差", "齐桓公", "晋文公", "楚庄王", "秦穆公",
    "宋襄公", "商汤", "夏桀", "纣王", "周武王", "周文王", "姜尚", "姜子牙",
    "伯夷", "叔齐", "卫青", "霍去病", "晁错", "袁盎", "魏公子", "信陵君",
    "平原君", "孟尝君", "春申君", "高祖", "武信君", "武臣", "秦始皇", "秦二世",
    "胡亥", "扶苏", "子婴", "陈涉", "章邯", "英布", "彭越", "周勃", "灌婴",
    "夏侯婴", "陆贾", "郦食其", "随何", "大禹", "启", "舜", "尧", "鲧",
    "葛伯", "伊尹", "商鞅", "郑国", "项籍", "周亚夫", "魏子", "周及", "缪贤",
    "吕禄", "吕太后", "王子婴", "许负", "刘濞", "雍齿", "齐王刘襄", "刘襄",
    "孝文皇帝", "文帝", "景帝", "孝惠帝", "陈平", "周公旦", "夫差", "武王",
    "怀王", "楚怀王", "义帝", "汉王", "楚王", "赵悼襄王", "楚隐王",
}

BASE_TITLE_WORDS = {
    "门客", "县令", "校尉", "相国", "丞相", "将军", "太尉", "楚王", "汉王", "诸侯",
    "豪杰", "侍从", "宾客", "臣子", "国君", "军师", "刺客", "谋士", "士卒", "部下",
    "屯长", "大夫", "太史令", "使者", "父老", "接班人", "富家女", "大王", "太仆",
    "陛下", "诸侯王", "中山王", "上将军", "大将", "舍人", "宦者令", "条侯", "汉高祖",
    "西楚霸王", "吴王", "魏王", "赵王", "齐王", "沛侯", "弓高侯", "将相", "郡守",
    "领袖", "名将", "叔父", "哥哥", "父亲", "儿子", "列侯", "祖先", "皇帝", "皇上",
}

BASE_PLACE_WORDS = {
    "沛县", "外黄", "外黄县", "苦陉", "苦陉县", "大梁", "咸阳", "邯郸", "临淄",
    "郢都", "姑苏", "鸿门", "乌江", "垓下", "巨鹿", "荥阳", "彭城", "函谷关",
    "白马津", "渔阳", "大泽乡", "范阳", "范阳城", "楚国", "赵国", "魏国", "韩国",
    "燕国", "齐国", "秦国", "蜀地", "汉中", "陈县", "蕲县", "河北", "河南",
    "黄河", "长江", "淮水", "昆吾", "豕韦", "葛国", "项城县", "丰邑", "昌邑",
    "会稽", "江淮", "关中", "汉城", "江苏", "长安", "梁国", "颖阴县",
}

NUMBER_RE = re.compile(r"^[零一二三四五六七八九十百千万亿两数几多约\d]+$")


def is_punctuation(token: str) -> bool:
    return bool(token) and all(char in PUNCTUATION for char in token)


def is_number(token: str) -> bool:
    return bool(NUMBER_RE.fullmatch(token)) or token in {
        "三十一岁", "十二年", "十户", "二十户", "五户", "二千五百户", "三年", "八年", "九年",
    }


def extract_vocab_from_groups(groups: Sequence[dict]) -> SimpleBertTokenizer:
    tokenizer = SimpleBertTokenizer()
    tokenizer.add_tokens(["候选", "：", "正文", "、", "[ENT]"])
    for group in groups:
        for sample in group["samples"]:
            tokenizer.add_tokens(sample["tokens"] + [sample["answer"]] + sample["hard_negatives"])
    return tokenizer


def bootstrap_lexicons(groups: Sequence[dict]) -> Tuple[Set[str], Set[str], Set[str]]:
    persons = set(BASE_PERSON_WORDS)
    titles = set(BASE_TITLE_WORDS)
    places = set(BASE_PLACE_WORDS)

    for group in groups:
        for fact in group.get("facts", []):
            rel_type = fact.get("relation_type", "")
            roles = fact.get("roles", {})
            sub = roles.get("subject", "")
            obj = roles.get("object", "")

            if rel_type in {"HAS_TITLE", "TITLE_OF"}:
                if rel_type == "HAS_TITLE":
                    if sub: persons.add(sub)
                    if obj: titles.add(obj)
                else:
                    if sub: titles.add(sub)
                    if obj: persons.add(obj)
            elif rel_type in {"BORN_IN", "LOCATED_IN", "CAPITAL_OF"}:
                if sub: persons.add(sub)
                if obj: places.add(obj)
            else:
                if sub: persons.add(sub)
                if obj: persons.add(obj)

        for sample in group.get("samples", []):
            ans = sample.get("answer", "")
            target_slot = sample.get("target_slot", "")
            if target_slot in {"subject", "object"}:
                if any(t in ans for t in ["侯", "王", "公", "将军", "尉", "相", "令", "皇", "帝"]):
                    titles.add(ans)
                else:
                    persons.add(ans)
            for neg in sample.get("hard_negatives", []):
                if any(t in neg for t in ["侯", "王", "公", "将军", "尉", "相", "令", "皇", "帝"]):
                    titles.add(neg)
                else:
                    persons.add(neg)

    return persons, titles, places


def annotate_token_attributes(
    token: str,
    persons: Set[str],
    titles: Set[str],
    places: Set[str],
) -> List[str]:
    attrs = []
    if is_punctuation(token):
        attrs.append("PUNCT")
        return attrs
    if token in FUNCTION_WORDS:
        attrs.append("FUNCTION")
    if token in persons:
        attrs.extend(["PERSON", "ENTITY"])
    if token in titles:
        attrs.extend(["TITLE", "PERSON"])
    if token in places or token.endswith(("县", "郡", "城", "国", "地", "关", "津", "乡")):
        attrs.extend(["PLACE", "ENTITY"])
    if is_number(token):
        attrs.append("NUMBER")
    if token in TIME_WORDS:
        attrs.append("TIME")
    if token == "[MASK]":
        attrs.extend(["ENTITY", "PERSON"])

    if not attrs:
        attrs.append("UNKNOWN")
    return sorted(set(attrs), key=ATTRIBUTES.index)


def annotate_sample(
    sample: dict,
    persons: Set[str],
    titles: Set[str],
    places: Set[str],
) -> dict:
    tokens = sample["tokens"]
    n = len(tokens)
    role_spans = sample.get("role_spans", {})

    # 1. Per-token attributes and structure role
    token_attributes: List[List[str]] = []
    token_structures: List[str] = ["OTHER"] * n

    for i, token in enumerate(tokens):
        attrs = annotate_token_attributes(token, persons, titles, places)
        token_attributes.append(attrs)
        if "PUNCT" in attrs:
            token_structures[i] = "BOUNDARY"
        elif "FUNCTION" in attrs:
            token_structures[i] = "FUNCTION"

    # Map role spans to structures
    for role, span in role_spans.items():
        if len(span) == 2:
            start, end = span
            struct = "OBJECT" if role == "object" else ("SUBJECT" if role == "subject" else "PREDICATE")
            for idx in range(start, min(end, n)):
                token_structures[idx] = struct

    # Mark [MASK] target slot
    if "[MASK]" in tokens:
        mask_idx = tokens.index("[MASK]")
        target_slot = sample.get("target_slot", "object")
        token_structures[mask_idx] = "SUBJECT" if target_slot == "subject" else "OBJECT"

    # 2. Active Candidate Pairs for 18D Relational FFN (If-Else Mask Gate)
    # A pair (i, j) is allowed into 18D FFN if and only if:
    # - i != j
    # - neither i nor j is PUNCT or FUNCTION
    # - BOTH i and j are Candidate Entity/Title/Role slots or [MASK]
    active_pairs: List[Tuple[int, int]] = []
    mask_pairs: List[Tuple[int, int]] = []
    entity_structures = {"SUBJECT", "OBJECT", "IDENTITY"}

    valid_indices = []
    for idx, (tok, attr, st) in enumerate(zip(tokens, token_attributes, token_structures)):
        attrs_set = set(attr)
        if "PUNCT" in attrs_set or "FUNCTION" in attrs_set:
            continue
        if tok == "[MASK]" or bool(attrs_set & {"PERSON", "TITLE", "ENTITY", "PLACE", "ORG"}) or st in entity_structures:
            valid_indices.append(idx)

    for i in valid_indices:
        for j in valid_indices:
            if i != j:
                active_pairs.append((i, j))
                if tokens[i] == "[MASK]" or tokens[j] == "[MASK]":
                    mask_pairs.append((i, j))

    total_pairs = n * n
    active_count = len(active_pairs)
    sparsity = 1.0 - (active_count / max(1, total_pairs))

    return {
        "sample_id": sample.get("sample_id", ""),
        "fact_id": sample.get("fact_id", ""),
        "answer": sample.get("answer", ""),
        "target_slot": sample.get("target_slot", ""),
        "tokens": tokens,
        "token_attributes": token_attributes,
        "token_structures": token_structures,
        "active_pairs": active_pairs,
        "mask_pairs": mask_pairs,
        "stats": {
            "token_length": n,
            "total_pairs": total_pairs,
            "active_pairs": active_count,
            "mask_pairs": len(mask_pairs),
            "sparsity": sparsity,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "data/shiji/units/shiji_units_merged_173.jsonl"),
        help="Path to merged 173-group JSONL dataset",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/shiji/manifests/shiji_173_grammar_attributes.json"),
        help="Path to output grammar attribute dataset",
    )
    args = parser.parse_args()

    print(f"Loading groups from: {args.dataset}")
    groups = list(load_groups(args.dataset))
    print(f"Loaded {len(groups)} groups.")

    tokenizer = extract_vocab_from_groups(groups)
    print(f"Vocabulary size across all groups: {len(tokenizer)}")

    persons, titles, places = bootstrap_lexicons(groups)
    print(f"Bootstrapped Lexicons:")
    print(f"  PERSON words: {len(persons)}")
    print(f"  TITLE words:  {len(titles)}")
    print(f"  PLACE words:  {len(places)}")
    print(f"  FUNCTION words: {len(FUNCTION_WORDS)}")
    print(f"  PUNCTUATION chars: {len(PUNCTUATION)}")

    # Ensure all lexicon words are in the tokenizer
    tokenizer.add_tokens(sorted(persons | titles | places | FUNCTION_WORDS))
    print(f"Total vocabulary size with lexicons: {len(tokenizer)}")

    # Annotate vocabulary tokens
    word_labels: Dict[str, List[str]] = {}
    for token in tokenizer.id_to_token:
        attrs = annotate_token_attributes(token, persons, titles, places)
        word_labels[token] = attrs

    # Count attribute frequencies
    attr_counts = Counter()
    for attrs in word_labels.values():
        for a in attrs:
            attr_counts[a] += 1
    print("\nVocabulary Attribute Distribution:")
    for a in ATTRIBUTES:
        print(f"  {a:14s}: {attr_counts[a]:5d} tokens")

    # Annotate sentences
    annotated_samples = []
    total_raw_pairs = 0
    total_active_pairs = 0

    for group in groups:
        for sample in group["samples"]:
            ann = annotate_sample(sample, persons, titles, places)
            annotated_samples.append(ann)
            total_raw_pairs += ann["stats"]["total_pairs"]
            total_active_pairs += ann["stats"]["active_pairs"]

    overall_sparsity = 1.0 - (total_active_pairs / max(1, total_raw_pairs))
    print("\nPair Sparsity Analysis across all samples:")
    print(f"  Total samples processed : {len(annotated_samples)}")
    print(f"  Total raw pairwise pairs: {total_raw_pairs:,}")
    print(f"  Active entity pairs kept: {total_active_pairs:,}")
    print(f"  Blocked junk pairs      : {total_raw_pairs - total_active_pairs:,}")
    print(f"  Overall Sparsity Rate   : {overall_sparsity * 100:.2f}% (Junk Blocked)")
    print(f"  Average pairs/sentence  : {total_raw_pairs / len(annotated_samples):.1f} -> {total_active_pairs / len(annotated_samples):.1f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "dataset": str(args.dataset),
            "total_groups": len(groups),
            "total_samples": len(annotated_samples),
            "vocab_size": len(tokenizer),
            "attributes": list(ATTRIBUTES),
            "structures": list(STRUCTURES),
            "overall_sparsity": overall_sparsity,
            "average_active_pairs": total_active_pairs / len(annotated_samples),
        },
        "word_labels": word_labels,
        "annotated_samples": annotated_samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully saved grammar dataset to: {output_path}")


if __name__ == "__main__":
    main()
