"""Build a source-verified Shiji fact-memory dataset.

This builder deliberately keeps the data conservative:

* every answer is an exact one-token span in the source line;
* the mask is applied at that span, never at the first textual occurrence;
* every variant carries its own answer and target role;
* relation names are canonical labels, while the original verb is retained as
  evidence;
* naming expressions such as “项羽 又 叫 项籍” are not treated as commands;
* source lines, rather than individual variants, determine train/dev/test split.

The extractor is still heuristic Chinese pattern matching.  The output should
therefore be treated as a verified candidate set, not as a complete semantic
graph.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import jieba.posseg as pseg
except ImportError as exc:  # pragma: no cover - exercised only without jieba
    raise RuntimeError(
        "jieba is required to build the Shiji fact dataset. "
        "Install it in the project Python environment first."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "shiji"
SEGMENTED_PATH = DATA_DIR / "segmented" / "shiji_segmented.txt"
MANIFESTS_DIR = DATA_DIR / "manifests"
OUTPUT_DIR = ROOT / "outputs" / "fact-memory"
ATTRIBUTE_MANIFEST = MANIFESTS_DIR / "word_attribute_dataset.json"

PUNCTUATION = {
    "，", "。", "；", "：", "！", "？", "、", "“", "”", "‘", "’",
    "(", ")", "（", "）", "《", "》", "—", "——",
}
SENTENCE_END = {"。", "！", "？", "；", "!", "?"}
COMMA_MARKS = {"，", ",", "；", ";", "：", ":"}

PERSON_DISTRACTORS = [
    "刘邦", "项羽", "张耳", "陈余", "韩信", "范增", "陈胜", "吴广", "蒯通",
    "萧何", "曹参", "樊哙", "张良", "吕不韦", "嬴政", "蒙恬", "李斯", "扁鹊",
    "屈原", "贾谊", "廉颇", "蔺相如", "赵括", "白起", "孙武", "孙膑", "庞涓",
    "伍子胥", "勾践", "夫差", "齐桓公", "晋文公", "楚庄王", "商汤", "夏桀",
    "周武王", "姜尚", "卫青", "霍去病", "魏公子", "信陵君", "平原君",
    "孟尝君", "武臣", "邵骚", "秦始皇", "章邯", "英布", "彭越", "周勃",
    "大禹", "商鞅", "郑国", "赵王歇", "张敖", "宋义", "周公旦", "微子",
    "箕子", "荆轲",
]

MORE_PERSON_NAMES = {
    "寒浞", "羿", "少康", "靡", "孔甲", "刘累", "桀", "纣王", "妲己", "武丁",
    "姬昌", "吕望", "周公", "召公", "秦王政", "赵高", "项梁", "项籍", "项伯",
    "灌婴", "周兰", "龙且", "李良", "陈平", "周亚夫", "李广", "李息", "李陵",
    "张骞", "陆贾", "陆生", "赵佗", "蒙骜", "蒙武", "蒙毅", "卫鞅", "吴起",
    "田单", "田广", "子玉", "重耳", "晋惠公", "晋献公", "楚昭王", "楚怀王",
    "楚成王", "秦穆公", "秦惠王", "秦昭王", "秦孝公", "秦襄公", "秦庄襄王",
    "文帝", "景帝", "孝文帝", "孝景帝", "汉武帝", "汉高祖", "高祖", "高帝",
}
PERSON_NAMES = set(PERSON_DISTRACTORS) | MORE_PERSON_NAMES

PLACE_NAMES = {
    "沛县", "大梁", "外黄", "苦陉", "咸阳", "邯郸", "临淄", "郢都", "姑苏",
    "鸿门", "乌江", "垓下", "巨鹿", "钜鹿", "荥阳", "彭城", "函谷关", "白马津",
    "渔阳", "大泽乡", "范阳", "楚国", "赵国", "魏国", "韩国", "燕国", "齐国",
    "秦国", "蜀地", "汉中", "陈县", "蕲县", "河北", "河南", "信都", "襄国",
    "废丘", "井陉", "会稽", "朝歌", "傅岩", "项城县", "狄", "岐山", "洛邑",
    "宛城", "关中", "三秦", "常山", "太原", "长平", "阏与", "南越", "西域",
    "蓬莱", "阳周", "句注山", "雁门", "定襄", "云中", "陇西", "月氏", "柏人城",
    "成武", "阳城", "户牖乡",
}
POLITICAL_PLACES = {
    "楚国", "赵国", "魏国", "韩国", "燕国", "齐国", "秦国", "南越", "匈奴",
    "汉", "楚", "赵", "魏", "齐", "秦", "周", "晋",
}

TITLE_WORDS = {
    "汉王", "项王", "沛公", "高祖", "高帝", "天子", "皇帝", "皇上", "大王",
    "楚王", "赵王", "魏王", "齐王", "秦王", "怀王", "文帝", "景帝", "孝文帝",
    "孝景帝", "武帝", "汉武帝", "始皇帝", "县令", "太子", "国君", "王", "将军",
    "大将", "大将军", "上将军", "次将", "末将", "丞相", "右丞相", "左丞相",
    "相国", "太尉", "太傅", "太师", "郡守", "国尉", "中尉", "郎中", "中郎将",
    "校尉", "司马", "大夫", "舍人", "侯", "列侯", "王后", "诸侯王", "世子",
    "安平君", "常山王", "南越王", "右贤王", "左贤王", "单于", "使者", "使臣",
    "御史", "侍臣", "门客", "谋士", "辩士", "小臣", "执法官",
}

GROUP_WORDS = {
    "军队", "大军", "部队", "部众", "楚军", "汉军", "秦军", "赵军", "燕军",
    "齐军", "骑兵", "骑士", "士兵", "兵卒", "百官", "诸侯", "民众", "百姓",
    "官员", "将领", "大臣", "贵人", "侍卫", "工匠", "姬妾", "士人", "九夷",
    "楚人", "秦人", "汉人", "赵人", "燕人", "军营",
}

ACTION_WORDS = {
    "攻打", "攻下", "攻占", "进攻", "进击", "讨伐", "征讨", "围困", "夺取",
    "平定", "收复", "打败", "击败", "杀死", "杀害", "处死", "斩杀", "起兵",
    "兴兵", "发兵", "率兵", "进兵", "出兵", "进军", "行刺", "寻找", "任命",
    "封", "拜", "立", "派", "派遣", "命令", "委托",
}

PRONOUNS_AND_FUNCTIONS = {
    "他", "她", "它", "他们", "她们", "自己", "有人", "大家", "我们", "你们",
    "这", "这个", "那", "那些", "有个", "一个", "一次", "后来", "于是", "然后",
    "同时", "随后", "现在", "当时", "不久", "可能", "准备", "打算", "愿意", "必须",
    "如果", "虽然", "即使", "因为", "由于", "而且", "并且", "以及", "共同", "一起",
    "前去", "向东", "向西", "向南", "向北", "往北", "进发", "出来", "打开",
    "再次", "一举", "终于", "正在", "已经", "还是", "就此", "不再", "永无休止",
    "十分高兴", "很大", "如何", "为了", "确实", "真的", "强行", "当众", "私下",
    "恰好", "立即", "首先", "一部分", "之中", "以东", "以北", "地区", "土地",
    "天下", "城市", "城池", "各县", "一带", "什么",
}
CONTINUATION_MARKERS = {
    "并", "并且", "又", "还", "便", "就", "随后", "同时", "接着", "于是", "再", "仍",
    "然后",
}
DIRECTION_MARKERS = {
    "向东", "向西", "向南", "向北", "往东", "往西", "往南", "往北", "进军", "进兵",
    "出兵", "率兵",
}

CAUSATIVE_VERBS = {"派", "命令", "派遣", "令", "委托"}
APPOINTMENT_VERBS = {"立", "封", "拜", "任命", "任用", "出任", "聘为"}
ORIGIN_WORDS = {"是", "本是", "原为", "原是"}
CAMPAIGN_RELATIONS = {
    "攻打": "ATTACK",
    "进攻": "ATTACK",
    "进击": "ATTACK",
    "攻下": "CONQUER",
    "攻占": "CONQUER",
    "夺取": "CONQUER",
    "平定": "CONQUER",
    "收复": "CONQUER",
    "讨伐": "PUNISH",
    "征讨": "PUNISH",
    "围困": "SIEGE",
}
COMBAT_RELATIONS = {
    "打败": "DEFEAT",
    "击败": "DEFEAT",
    "杀死": "KILL",
    "杀害": "KILL",
    "处死": "KILL",
    "斩杀": "KILL",
}
LOCATION_EVENTS = {
    "起义", "即位", "称王", "建都", "驻扎", "大败", "自杀", "病逝", "起兵", "诛杀",
}

DISTRACTOR_POOLS = {
    "PERSON": PERSON_DISTRACTORS,
    "PLACE": sorted(PLACE_NAMES),
    "TITLE": sorted(TITLE_WORDS),
    "GROUP": sorted(GROUP_WORDS),
}


def load_attribute_lexicon() -> Dict[str, Set[str]]:
    """Load previously learned lexical labels as an optional extra signal."""

    labels: Dict[str, Set[str]] = {}
    if not ATTRIBUTE_MANIFEST.exists():
        return labels
    try:
        payload = json.loads(ATTRIBUTE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return labels
    for word, values in payload.get("word_labels", {}).items():
        labels[str(word)] = {str(value) for value in values}
    return labels


ATTRIBUTE_LEXICON = load_attribute_lexicon()


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def split_for_source(source_line: int) -> str:
    bucket = stable_seed("split", source_line) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def pos_tags(line: str) -> Dict[str, str]:
    clean_line = line.replace(" ", "")
    return {word: flag for word, flag in pseg.cut(clean_line)}


def token_attributes(token: str, default: Optional[str] = None) -> List[str]:
    """Return coarse multi-label attributes for one target token."""

    labels = set(ATTRIBUTE_LEXICON.get(token, set()))
    if token in GROUP_WORDS:
        labels.discard("PERSON")
        labels.discard("TITLE")
        labels.add("GROUP")
    elif token in TITLE_WORDS:
        labels.add("TITLE")
    elif token in PERSON_NAMES:
        labels.add("PERSON")
    elif token in PLACE_NAMES or token.endswith(("国", "县", "郡", "城", "关")):
        labels.add("PLACE")
    elif token in ACTION_WORDS:
        labels.add("ACTION")

    if re.fullmatch(r"[0-9０-９]+", token):
        labels.add("NUMBER")

    semantic_labels = {
        "PERSON", "PLACE", "TITLE", "GROUP", "ORG", "ACTION", "NUMBER",
    }
    if labels & semantic_labels and "PUNCT" not in labels:
        labels.add("ENTITY")
    if not labels and default:
        labels.add(default)
        if default not in {"ACTION", "NUMBER"}:
            labels.add("ENTITY")
    if not labels:
        labels.add("ENTITY")

    if "GROUP" in labels:
        labels.discard("PERSON")
        labels.discard("TITLE")
        labels.add("ENTITY")
    return sorted(labels)


def is_place_like(token: str, tag_map: Dict[str, str]) -> bool:
    attrs = set(token_attributes(token))
    return "PLACE" in attrs or tag_map.get(token, "").startswith("ns")


def is_person_like(token: str, tag_map: Dict[str, str]) -> bool:
    if token in PRONOUNS_AND_FUNCTIONS or token in GROUP_WORDS:
        return False
    attrs = set(token_attributes(token))
    if "PERSON" in attrs or "TITLE" in attrs:
        return True
    flag = tag_map.get(token, "")
    return flag.startswith(("nr", "nrf")) and "PLACE" not in attrs


def is_actor_candidate(token: str, tag_map: Dict[str, str]) -> bool:
    if token in PUNCTUATION or token in PRONOUNS_AND_FUNCTIONS:
        return False
    if token in ACTION_WORDS or not token:
        return False
    attrs = set(token_attributes(token))
    if attrs & {"PERSON", "TITLE", "GROUP"}:
        return True
    if token in POLITICAL_PLACES:
        return True
    flag = tag_map.get(token, "")
    return flag.startswith(("nr", "nrf")) and "PLACE" not in attrs


def candidate_span(
    tokens: Sequence[str], index: int, tag_map: Dict[str, str]
) -> Tuple[int, int, str]:
    """Expand an obvious title-name span such as 太子 丹."""

    start = index
    end = index + 1
    if (
        index > 0
        and tokens[index - 1] in TITLE_WORDS
        and tokens[index - 1] not in GROUP_WORDS
    ):
        start = index - 1
    elif (
        index + 1 < len(tokens)
        and tokens[index] in TITLE_WORDS
        and is_person_like(tokens[index + 1], tag_map)
    ):
        end = index + 2
    return start, end, " ".join(tokens[start:end])


def find_subject(
    tokens: Sequence[str], verb_index: int, tag_map: Dict[str, str]
) -> Optional[Dict[str, object]]:
    """Find a conservative local agent; pronouns are never resolved."""

    sentence_start = 0
    for index in range(verb_index - 1, -1, -1):
        if tokens[index] in SENTENCE_END:
            sentence_start = index + 1
            break

    comma_start = sentence_start
    for index in range(verb_index - 1, sentence_start - 1, -1):
        if tokens[index] in COMMA_MARKS:
            comma_start = index + 1
            break

    local_candidates = [
        index
        for index in range(comma_start, verb_index)
        if is_actor_candidate(tokens[index], tag_map)
    ]
    local_prefix = tokens[comma_start:verb_index]
    marker_continuation = bool(local_prefix) and (
        local_prefix[0] in CONTINUATION_MARKERS
        or local_prefix[0] in DIRECTION_MARKERS
    )

    if local_candidates and not marker_continuation:
        index = local_candidates[-1]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "explicit_nearest",
            "attributes": token_attributes(tokens[index]),
        }

    prior_candidates = [
        index
        for index in range(sentence_start, comma_start)
        if is_actor_candidate(tokens[index], tag_map)
    ]
    if marker_continuation and prior_candidates:
        index = prior_candidates[0]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "continued_sentence_actor",
            "attributes": token_attributes(tokens[index]),
        }

    if local_candidates:
        index = local_candidates[-1]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "explicit_after_marker",
            "attributes": token_attributes(tokens[index]),
        }

    if local_prefix and local_prefix[0] in DIRECTION_MARKERS and prior_candidates:
        index = prior_candidates[0]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "continued_direction_actor",
            "attributes": token_attributes(tokens[index]),
        }
    return None


def find_causative_target(
    tokens: Sequence[str], verb_index: int, tag_map: Dict[str, str]
) -> Optional[int]:
    """Find a person before a generic role, e.g. 命令 骑兵 将领 灌婴."""

    end = min(len(tokens), verb_index + 7)
    person_candidates: List[int] = []
    fallback_candidates: List[int] = []
    for index in range(verb_index + 1, end):
        token = tokens[index]
        if token in PUNCTUATION:
            break
        if token in {"自己", "他们", "大家", "有人", "人", "们"}:
            continue
        if is_person_like(token, tag_map):
            person_candidates.append(index)
        elif set(token_attributes(token)) & {"TITLE", "GROUP"}:
            fallback_candidates.append(index)

    if person_candidates:
        return person_candidates[0]
    if fallback_candidates:
        return fallback_candidates[0]
    return None


def find_appointment_parts(
    tokens: Sequence[str], verb_index: int, tag_map: Dict[str, str]
) -> Optional[Tuple[int, int]]:
    """Return person and single-token role positions for an appointment."""

    for_index = None
    for index in range(verb_index + 2, min(len(tokens), verb_index + 7)):
        if tokens[index] in SENTENCE_END or tokens[index] in {"，", "；", "："}:
            break
        if tokens[index] == "为":
            for_index = index
            break
    if for_index is None:
        return None

    person_candidates = [
        index
        for index in range(verb_index + 1, for_index)
        if is_person_like(tokens[index], tag_map)
    ]
    if person_candidates:
        person_index = person_candidates[-1]
    else:
        fallback = [
            index
            for index in range(verb_index + 1, for_index)
            if tokens[index] not in TITLE_WORDS
            and tokens[index] not in PRONOUNS_AND_FUNCTIONS
            and tokens[index] not in PUNCTUATION
        ]
        if not fallback:
            return None
        person_index = fallback[-1]

    role_index = for_index + 1
    if role_index >= len(tokens) or tokens[role_index] in PUNCTUATION:
        return person_index, -1
    # Do not truncate multi-token roles such as 常山 王 or 上 将军.
    if (
        role_index + 1 < len(tokens)
        and tokens[role_index + 1] not in PUNCTUATION
        and tokens[role_index] in {"上", "下", "左", "右", "中"}
    ):
        return person_index, -1
    return person_index, role_index


def mask_at(tokens: Sequence[str], start: int, end: int) -> List[str]:
    return list(tokens[:start]) + ["[MASK]"] + list(tokens[end:])


def make_long_context(
    sentences: Sequence[str],
    line_index: int,
    target_index: int,
    window: int = 2,
) -> Tuple[str, str, int]:
    start_line = max(0, line_index - window)
    end_line = min(len(sentences), line_index + window + 1)
    long_tokens: List[str] = []
    absolute_target = -1
    for index in range(start_line, end_line):
        line_tokens = sentences[index].split()
        if index == line_index:
            absolute_target = len(long_tokens) + target_index
        long_tokens.extend(line_tokens)
    if absolute_target < 0:
        raise ValueError("target line was not included in long context")
    return (
        " ".join(long_tokens),
        " ".join(mask_at(long_tokens, absolute_target, absolute_target + 1)),
        absolute_target,
    )


def answer_is_visible(answer: str, masked_text: str) -> bool:
    return answer in masked_text.split()


def choose_distractors(
    answer: str,
    attributes: Sequence[str],
    exclude: Iterable[str],
    source_line: int,
    target_index: int,
    long_context: str,
    k: int = 5,
) -> List[str]:
    primary = next(
        (
            label
            for label in ("PERSON", "PLACE", "TITLE", "GROUP")
            if label in attributes
        ),
        "PERSON",
    )
    pool = DISTRACTOR_POOLS.get(primary, PERSON_DISTRACTORS)
    excluded = set(exclude)
    excluded.add(answer)
    evidence_tokens = set(long_context.split())
    candidates = [
        value
        for value in pool
        if value not in excluded and value not in evidence_tokens
    ]
    rng = random.Random(stable_seed("distractors", source_line, target_index, answer))
    rng.shuffle(candidates)
    return candidates[:k]


def make_fact(
    *,
    sentences: Sequence[str],
    line_index: int,
    tokens: Sequence[str],
    target_index: int,
    target_role: str,
    subject: Dict[str, object],
    relation_type: str,
    grammar: str,
    relation_surface: str,
    relation_evidence: str,
    object_value: str,
    object_attributes: Sequence[str],
    default_answer_attribute: Optional[str] = None,
    fact_group_id: Optional[str] = None,
) -> Optional[Dict]:
    if target_index < 0 or target_index >= len(tokens):
        return None
    answer = tokens[target_index]
    if answer in PUNCTUATION or answer in PRONOUNS_AND_FUNCTIONS:
        return None
    answer_attributes = token_attributes(answer, default_answer_attribute)
    source_masked = " ".join(mask_at(tokens, target_index, target_index + 1))
    if answer_is_visible(answer, source_masked):
        # Repeated answers are intentionally excluded from one-token MLM data.
        return None

    long_context, long_masked, long_target = make_long_context(
        sentences, line_index, target_index
    )
    variants: List[Dict] = []
    if not answer_is_visible(answer, long_masked):
        variants.append(
            {
                "type": "long_context_masked",
                "context": long_context,
                "masked_text": long_masked,
                "answer": answer,
                "target_role": target_role,
                "answer_attributes": list(answer_attributes),
                "target_span": [long_target, long_target + 1],
            }
        )

    source_line = line_index + 1
    subject_text = str(subject["text"])
    subject_attrs = list(subject.get("attributes", []))
    fact_group_id = fact_group_id or (
        f"line_{source_line}_{relation_type}_{target_index}"
    )
    distractors = choose_distractors(
        answer,
        answer_attributes,
        {answer, subject_text, object_value, *tokens},
        source_line,
        target_index,
        long_context,
    )
    return {
        "fact_group_id": fact_group_id,
        "source_line": source_line,
        "context": " ".join(tokens),
        "long_context": long_context,
        "masked_text": source_masked,
        "answer": answer,
        "target_role": target_role,
        "target_span": [target_index, target_index + 1],
        "subject": subject_text,
        "subject_attributes": subject_attrs,
        "relation": relation_type,
        "relation_type": relation_type,
        "relation_surface": relation_surface,
        "relation_evidence": relation_evidence,
        "object": object_value,
        "object_attributes": list(object_attributes),
        "grammar": grammar,
        "answer_attributes": list(answer_attributes),
        "confidence": 0.96 if subject.get("mode") == "explicit_nearest" else 0.91,
        "confidence_basis": "source_span_verified+pattern_match",
        "subject_resolution": subject.get("mode"),
        "distractors": distractors,
        "variants": variants,
        "split": split_for_source(source_line),
    }


def source_evidence(tokens: Sequence[str], start: int, end: int) -> str:
    return " ".join(tokens[max(0, start - 2) : min(len(tokens), end + 6)])


def extract_all_facts(sentences: Sequence[str]) -> List[Dict]:
    facts: List[Dict] = []
    for line_index, line in enumerate(sentences):
        tokens = line.split()
        if len(tokens) < 4 or len(tokens) > 120:
            continue
        tag_map = pos_tags(line)

        # 1. Assignment/command.  Naming expressions using 叫 are excluded.
        for verb_index, verb in enumerate(tokens):
            if verb not in CAUSATIVE_VERBS:
                continue
            target_index = find_causative_target(tokens, verb_index, tag_map)
            subject = find_subject(tokens, verb_index, tag_map)
            if target_index is None or subject is None:
                continue
            target = tokens[target_index]
            attrs = token_attributes(target, "PERSON")
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type="ASSIGN",
                grammar="CAUSATIVE",
                relation_surface=verb,
                relation_evidence=source_evidence(tokens, verb_index, target_index + 1),
                object_value=target,
                object_attributes=attrs,
                default_answer_attribute="PERSON",
                fact_group_id=f"line_{line_index + 1}_assign_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

        # 2. Appointment/ennoblement.  Person and single-token title targets
        # are separate records with separate answers.
        for verb_index, verb in enumerate(tokens):
            if verb not in APPOINTMENT_VERBS:
                continue
            parts = find_appointment_parts(tokens, verb_index, tag_map)
            subject = find_subject(tokens, verb_index, tag_map)
            if parts is None or subject is None:
                continue
            person_index, role_index = parts
            person = tokens[person_index]
            role = tokens[role_index] if role_index >= 0 else ""
            role_attributes = token_attributes(role, "TITLE") if role else ["TITLE", "ENTITY"]
            evidence = source_evidence(
                tokens, verb_index, max(person_index + 1, role_index + 1)
            )
            group_id = f"line_{line_index + 1}_appoint_{person_index}_{role_index}"
            person_fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=person_index,
                target_role="object",
                subject=subject,
                relation_type="APPOINT",
                grammar="APPOINTMENT",
                relation_surface=verb,
                relation_evidence=evidence,
                object_value=person,
                object_attributes=token_attributes(person, "PERSON"),
                default_answer_attribute="PERSON",
                fact_group_id=group_id,
            )
            if person_fact is not None:
                person_fact["role"] = role
                person_fact["role_attributes"] = role_attributes
                facts.append(person_fact)
            if role_index >= 0:
                role_fact = make_fact(
                    sentences=sentences,
                    line_index=line_index,
                    tokens=tokens,
                    target_index=role_index,
                    target_role="role",
                    subject=subject,
                    relation_type="APPOINT",
                    grammar="APPOINTMENT",
                    relation_surface=verb,
                    relation_evidence=evidence,
                    object_value=person,
                    object_attributes=token_attributes(person, "PERSON"),
                    default_answer_attribute="TITLE",
                    fact_group_id=group_id,
                )
                if role_fact is not None:
                    role_fact["role"] = role
                    role_fact["role_attributes"] = role_attributes
                    facts.append(role_fact)

        # 3. Origin/place.  Person and place are separate prediction targets.
        for person_index in range(len(tokens) - 2):
            if not is_person_like(tokens[person_index], tag_map):
                continue
            if tokens[person_index + 1] not in ORIGIN_WORDS:
                continue
            place_index = person_index + 2
            if not is_place_like(tokens[place_index], tag_map):
                continue
            person = tokens[person_index]
            place = tokens[place_index]
            subject = {
                "text": person,
                "attributes": token_attributes(person, "PERSON"),
                "mode": "explicit_origin_subject",
            }
            evidence = source_evidence(tokens, person_index, place_index + 1)
            group_id = f"line_{line_index + 1}_origin_{person_index}_{place_index}"
            person_fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=person_index,
                target_role="subject",
                subject=subject,
                relation_type="BORN_IN",
                grammar="ORIGIN",
                relation_surface=tokens[person_index + 1],
                relation_evidence=evidence,
                object_value=place,
                object_attributes=token_attributes(place, "PLACE"),
                default_answer_attribute="PERSON",
                fact_group_id=group_id,
            )
            if person_fact is not None:
                facts.append(person_fact)
            place_fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=place_index,
                target_role="object",
                subject=subject,
                relation_type="BORN_IN",
                grammar="ORIGIN",
                relation_surface=tokens[person_index + 1],
                relation_evidence=evidence,
                object_value=place,
                object_attributes=token_attributes(place, "PLACE"),
                default_answer_attribute="PLACE",
                fact_group_id=group_id,
            )
            if place_fact is not None:
                facts.append(place_fact)

        # 4. Campaign/attack facts with canonical relation types.
        for verb_index, verb in enumerate(tokens):
            relation_type = CAMPAIGN_RELATIONS.get(verb)
            if relation_type is None or verb_index + 1 >= len(tokens):
                continue
            target_index = verb_index + 1
            target = tokens[target_index]
            target_attrs = token_attributes(target)
            if target in PRONOUNS_AND_FUNCTIONS or target in PUNCTUATION:
                continue
            if not (set(target_attrs) & {"PERSON", "PLACE", "GROUP", "TITLE"}):
                continue
            subject = find_subject(tokens, verb_index, tag_map)
            if subject is None:
                continue
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type=relation_type,
                grammar="CAMPAIGN",
                relation_surface=verb,
                relation_evidence=source_evidence(tokens, verb_index, target_index + 1),
                object_value=target,
                object_attributes=target_attrs,
                default_answer_attribute="PERSON",
                fact_group_id=f"line_{line_index + 1}_{relation_type}_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

        # 5. Combat outcomes.
        for verb_index, verb in enumerate(tokens):
            relation_type = COMBAT_RELATIONS.get(verb)
            if relation_type is None or verb_index + 1 >= len(tokens):
                continue
            target_index = verb_index + 1
            target = tokens[target_index]
            target_attrs = token_attributes(target, "PERSON")
            if target in PRONOUNS_AND_FUNCTIONS or target in PUNCTUATION:
                continue
            if not (set(target_attrs) & {"PERSON", "PLACE", "GROUP", "TITLE"}):
                continue
            subject = find_subject(tokens, verb_index, tag_map)
            if subject is None:
                continue
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type=relation_type,
                grammar="COMBAT_OUTCOME",
                relation_surface=verb,
                relation_evidence=source_evidence(tokens, verb_index, target_index + 1),
                object_value=target,
                object_attributes=target_attrs,
                default_answer_attribute="PERSON",
                fact_group_id=f"line_{line_index + 1}_{relation_type}_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

        # 6. Place attached to a compact historical event.
        for place_index in range(len(tokens) - 2):
            if tokens[place_index] != "在":
                continue
            target_index = place_index + 1
            event_index = place_index + 2
            if not is_place_like(tokens[target_index], tag_map):
                continue
            if tokens[event_index] not in LOCATION_EVENTS:
                continue
            subject = find_subject(tokens, place_index, tag_map)
            if subject is None:
                continue
            place = tokens[target_index]
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type="EVENT_AT",
                grammar="LOCATION_EVENT",
                relation_surface=tokens[event_index],
                relation_evidence=source_evidence(tokens, place_index, event_index + 1),
                object_value=place,
                object_attributes=token_attributes(place, "PLACE"),
                default_answer_attribute="PLACE",
                fact_group_id=f"line_{line_index + 1}_event_at_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

    return facts


def validate_record(record: Dict, original_sentences: Sequence[str]) -> List[str]:
    errors: List[str] = []
    source_line = record.get("source_line")
    if not isinstance(source_line, int) or not (1 <= source_line <= len(original_sentences)):
        return ["invalid_source_line"]
    context = record.get("context")
    ground_truth = original_sentences[source_line - 1]
    if context != ground_truth:
        errors.append("context_mismatch")

    tokens = str(context).split()
    answer = record.get("answer")
    span = record.get("target_span")
    if not isinstance(span, list) or len(span) != 2:
        errors.append("invalid_target_span")
        return errors
    start, end = span
    if not (isinstance(start, int) and isinstance(end, int) and end == start + 1):
        errors.append("non_single_token_target")
    elif not (0 <= start < len(tokens)) or tokens[start] != answer:
        errors.append("target_span_answer_mismatch")

    expected_mask = " ".join(mask_at(tokens, start, end)) if 0 <= start < len(tokens) else ""
    if record.get("masked_text") != expected_mask:
        errors.append("masked_text_not_exact")
    if str(record.get("masked_text", "")).split().count("[MASK]") != 1:
        errors.append("base_mask_count")
    if answer_is_visible(str(answer), str(record.get("masked_text", ""))):
        errors.append("base_answer_leakage")

    if not record.get("relation_type") or record.get("relation") != record.get("relation_type"):
        errors.append("relation_not_canonical")
    if not record.get("answer_attributes"):
        errors.append("missing_answer_attributes")
    if record.get("split") not in {"train", "dev", "test"}:
        errors.append("invalid_split")

    for variant in record.get("variants", []):
        if variant.get("answer") != answer:
            errors.append("variant_answer_mismatch")
        masked = str(variant.get("masked_text", ""))
        if masked.split().count("[MASK]") != 1:
            errors.append("variant_mask_count")
        if answer_is_visible(str(answer), masked):
            errors.append("variant_answer_leakage")
        if not variant.get("answer_attributes"):
            errors.append("variant_missing_attributes")
    return errors


def validate_and_deduplicate(
    facts: Sequence[Dict], original_sentences: Sequence[str]
) -> Tuple[List[Dict], Counter]:
    valid: List[Dict] = []
    rejected: Counter = Counter()
    seen: Set[Tuple[object, ...]] = set()
    for fact in facts:
        errors = validate_record(fact, original_sentences)
        if errors:
            rejected.update(errors)
            continue
        key = (
            fact["source_line"],
            fact["target_span"][0],
            fact["relation_type"],
            fact["target_role"],
            fact["answer"],
        )
        if key in seen:
            rejected["duplicate_fact"] += 1
            continue
        seen.add(key)
        valid.append(fact)
    valid.sort(
        key=lambda item: (
            item["source_line"],
            item["target_span"][0],
            item["relation_type"],
            item["target_role"],
        )
    )
    return valid, rejected


def build_report(
    facts: Sequence[Dict],
    rejected: Counter,
    source_sentence_count: int,
) -> Dict:
    quality = Counter()
    for fact in facts:
        quality["exact_one_base_mask"] += int(
            fact["masked_text"].split().count("[MASK]") == 1
        )
        quality["base_answer_leakage"] += int(
            answer_is_visible(fact["answer"], fact["masked_text"])
        )
        for variant in fact.get("variants", []):
            quality["variant_count"] += 1
            quality["variant_answer_mismatch"] += int(
                variant.get("answer") != fact["answer"]
            )
            quality["variant_answer_leakage"] += int(
                answer_is_visible(fact["answer"], variant["masked_text"])
            )
    return {
        "schema_version": 2,
        "total_facts": len(facts),
        "source_sentences_count": source_sentence_count,
        "source_lines_used": len({fact["source_line"] for fact in facts}),
        "grammar_distribution": dict(Counter(fact["grammar"] for fact in facts)),
        "relation_type_distribution": dict(
            Counter(fact["relation_type"] for fact in facts)
        ),
        "target_role_distribution": dict(
            Counter(fact["target_role"] for fact in facts)
        ),
        "split_distribution": dict(Counter(fact["split"] for fact in facts)),
        "attribute_distribution": dict(
            Counter(tuple(fact["answer_attributes"]) for fact in facts)
        ),
        "variant_distribution": dict(
            Counter(len(fact.get("variants", [])) for fact in facts)
        ),
        "distractor_distribution": dict(
            Counter(len(fact.get("distractors", [])) for fact in facts)
        ),
        "average_heuristic_confidence": (
            sum(float(fact["confidence"]) for fact in facts) / len(facts)
            if facts
            else 0.0
        ),
        "quality_checks": dict(quality),
        "rejected_candidates": dict(rejected),
        "notes": [
            "confidence is a rule-based score, not a measured accuracy",
            "naming/alias expressions using 叫 are excluded from CAUSATIVE",
            "all variants carry an independent answer label",
            "source_line determines the split to prevent context leakage",
        ],
        "sample_preview": list(facts[:3]),
    }


def main() -> None:
    print(f"=== Loading segmented sentences from {SEGMENTED_PATH} ===")
    sentences = SEGMENTED_PATH.read_text(encoding="utf-8").splitlines()
    print(f"Loaded {len(sentences)} sentences.")

    print("=== Mining source-verified fact candidates ===")
    raw_facts = extract_all_facts(sentences)
    print(f"Mined {len(raw_facts)} raw candidates.")

    print("=== Validating exact spans, masks, variants, and splits ===")
    facts, rejected = validate_and_deduplicate(raw_facts, sentences)
    print(f"Validated {len(facts)} facts.")
    if rejected:
        print("Rejected:", dict(rejected))

    for index, fact in enumerate(facts, start=1):
        fact["fact_id"] = f"shiji_fact_{index:04d}"

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = MANIFESTS_DIR / "fact_memory_dataset.json"
    mirror_path = OUTPUT_DIR / "facts_dataset.json"
    report_path = MANIFESTS_DIR / "fact_memory_report.json"
    serialized = json.dumps(facts, ensure_ascii=False, indent=2)
    dataset_path.write_text(serialized, encoding="utf-8")
    mirror_path.write_text(serialized, encoding="utf-8")
    report_path.write_text(
        json.dumps(
            build_report(facts, rejected, len(sentences)),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved dataset to {dataset_path}")
    print(f"Mirrored dataset to {mirror_path}")
    print(f"Saved report to {report_path}")
    print(json.dumps(build_report(facts, rejected, len(sentences)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
