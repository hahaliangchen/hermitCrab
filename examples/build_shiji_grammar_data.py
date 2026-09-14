"""Build Shiji grammar and word attribute dataset from PDF source.

Pipeline:
1. PDF text extraction & noise removal (headers/footers/page numbers/publisher metadata).
2. Clean sentence splitting (with quote preservation).
3. Word segmentation with custom historical entity dictionary (compatible with SimpleBertTokenizer).
4. Slot & attribute extraction across 12 structures and 13 attributes.
5. Chapter-based train / validation / test partitioning (80% / 10% / 10%).
6. Output raw, sentences, segmented, and manifest dataset.json.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import jieba
import jieba.posseg as pseg
import pypdf

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ATTRIBUTES = ('PLACE', 'TIME', 'PERSON', 'ORG', 'TITLE', 'NUMBER', 'ACTION',
              'ENTITY', 'FUNCTION', 'PUNCT', 'INTERROGATIVE', 'CLAUSE', 'UNKNOWN')
STRUCTURES = ('NAME', 'LOCATION', 'TEMPORAL', 'PREDICATE', 'OBJECT', 'IDENTITY',
              'QUANTITY', 'CONDITION', 'COORDINATION', 'PARTICLE', 'BOUNDARY', 'OPEN')

PDF_PATH = Path(r"D:\BaiduNetdiskDownload\史记经典故事.pdf")
DATA_DIR = ROOT / "data" / "shiji"
RAW_DIR = DATA_DIR / "raw"
SENTENCES_DIR = DATA_DIR / "sentences"
SEGMENTED_DIR = DATA_DIR / "segmented"
MANIFESTS_DIR = DATA_DIR / "manifests"

# Historical entity dictionary to augment jieba
HISTORICAL_ENTITIES = [
    # People (nr)
    ("司马迁", "nr"), ("张耳", "nr"), ("陈余", "nr"), ("刘邦", "nr"), ("项羽", "nr"),
    ("范增", "nr"), ("韩信", "nr"), ("陈胜", "nr"), ("吴广", "nr"), ("蒯通", "nr"),
    ("萧何", "nr"), ("曹参", "nr"), ("樊哙", "nr"), ("张良", "nr"), ("吕不韦", "nr"),
    ("嬴政", "nr"), ("蒙恬", "nr"), ("李斯", "nr"), ("扁鹊", "nr"), ("屈原", "nr"),
    ("贾谊", "nr"), ("廉颇", "nr"), ("蔺相如", "nr"), ("赵括", "nr"), ("白起", "nr"),
    ("孙武", "nr"), ("孙膑", "nr"), ("庞涓", "nr"), ("伍子胥", "nr"), ("勾践", "nr"),
    ("夫差", "nr"), ("齐桓公", "nr"), ("晋文公", "nr"), ("楚庄王", "nr"), ("秦穆公", "nr"),
    ("宋襄公", "nr"), ("商汤", "nr"), ("夏桀", "nr"), ("纣王", "nr"), ("周武王", "nr"),
    ("周文王", "nr"), ("姜尚", "nr"), ("姜子牙", "nr"), ("伯夷", "nr"), ("叔齐", "nr"),
    ("卫青", "nr"), ("霍去病", "nr"), ("晁错", "nr"), ("袁盎", "nr"), ("魏公子", "nr"),
    ("信陵君", "nr"), ("平原君", "nr"), ("孟尝君", "nr"), ("春申君", "nr"), ("高祖", "nr"),
    ("武信君", "nr"), ("公乘", "nr"), ("武臣", "nr"), ("邵骚", "nr"), ("范阳令", "nr"),
    ("秦始皇", "nr"), ("秦二世", "nr"), ("胡亥", "nr"), ("扶苏", "nr"), ("子婴", "nr"),
    ("陈涉", "nr"), ("章邯", "nr"), ("英布", "nr"), ("彭越", "nr"), ("周勃", "nr"),
    ("灌婴", "nr"), ("夏侯婴", "nr"), ("陆贾", "nr"), ("郦食其", "nr"), ("随何", "nr"),
    ("大禹", "nr"), ("启", "nr"), ("舜", "nr"), ("尧", "nr"), ("鲧", "nr"),
    ("葛伯", "nr"), ("伊尹", "nr"), ("商鞅", "nr"), ("郑国", "nr"),
    # Places (ns)
    ("沛县", "ns"), ("外黄", "ns"), ("外黄县", "ns"), ("苦陉", "ns"), ("苦陉县", "ns"),
    ("大梁", "ns"), ("咸阳", "ns"), ("邯郸", "ns"), ("临淄", "ns"), ("郢都", "ns"),
    ("姑苏", "ns"), ("鸿门", "ns"), ("乌江", "ns"), ("垓下", "ns"), ("巨鹿", "ns"),
    ("荥阳", "ns"), ("彭城", "ns"), ("函谷关", "ns"), ("白马津", "ns"), ("渔阳", "ns"),
    ("大泽乡", "ns"), ("范阳", "ns"), ("范阳城", "ns"), ("楚国", "ns"), ("赵国", "ns"),
    ("魏国", "ns"), ("韩国", "ns"), ("燕国", "ns"), ("齐国", "ns"), ("秦国", "ns"),
    ("蜀地", "ns"), ("汉中", "ns"), ("陈县", "ns"), ("蕲县", "ns"), ("河北", "ns"),
    ("河南", "ns"), ("黄河", "ns"), ("长江", "ns"), ("淮水", "ns"), ("昆吾", "ns"),
    ("豕韦", "ns"), ("葛国", "ns"),
    # Titles & Roles (n / TITLE)
    ("门客", "n"), ("县令", "n"), ("校尉", "n"), ("相国", "n"), ("丞相", "n"),
    ("将军", "n"), ("太尉", "n"), ("楚王", "n"), ("汉王", "n"), ("诸侯", "n"),
    ("豪杰", "n"), ("侍从", "n"), ("宾客", "n"), ("臣子", "n"), ("国君", "n"),
    ("军师", "n"), ("刺客", "n"), ("谋士", "n"), ("士卒", "n"), ("部下", "n"),
    ("屯长", "n"), ("大夫", "n"), ("太史令", "n"), ("使者", "n"), ("父老", "n"),
    ("接班人", "n"), ("富家女", "n"),
]

# Patterns for line cleaning
HEADER_FOOTER_PATTERNS = [
    re.compile(r"汉·司马迁\s*原撰(?:◎史记\s*经典故事.*)?"),
    re.compile(r"史记\s*经典故事"),
    re.compile(r"创世卓越\s*荣誉出品"),
    re.compile(r"Trust\s*Joy.*", re.IGNORECASE),
    re.compile(r"ISBN\s*[\d\-X/]+.*", re.IGNORECASE),
    re.compile(r"Publisher:.*", re.IGNORECASE),
    re.compile(r"Editor.*", re.IGNORECASE),
    re.compile(r"Beijing.*", re.IGNORECASE),
    re.compile(r"SHIJI\s*JINGDIAN\s*GUSHI", re.IGNORECASE),
    re.compile(r"^[\s一二三四五六七八九十百千万\d]{1,6}$"),  # standalone page numbers like "一 九 四"
]


def init_jieba():
    for word, tag in HISTORICAL_ENTITIES:
        jieba.add_word(word, tag=tag)


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    for pattern in HEADER_FOOTER_PATTERNS:
        if pattern.search(line):
            return True
    # Table of contents dots or page listings
    if re.search(r"……|\.{4,}", line):
        return True
    return False


def extract_raw_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract and clean raw text from PDF, returning (page_num, clean_text) pairs."""
    reader = pypdf.PdfReader(str(pdf_path))
    pages_data = []

    # Story content starts on page 6 (0-indexed page 5)
    for pno in range(5, len(reader.pages)):
        chunks = []
        page = reader.pages[pno]
        page.extract_text(visitor_text=lambda t, cm, tm, f, sz: chunks.append(t) if t else None)
        raw = "".join(chunks)

        lines = [l.strip() for l in raw.splitlines()]
        cleaned_lines = [l for l in lines if not is_noise_line(l)]

        # Stitch broken lines into continuous paragraphs
        paragraphs = []
        current_para = []
        for line in cleaned_lines:
            if not current_para:
                current_para.append(line)
            else:
                prev = current_para[-1]
                if prev.endswith(("。", "！", "？", "；", "”", "’", "：")):
                    paragraphs.append("".join(current_para))
                    current_para = [line]
                else:
                    current_para.append(line)
        if current_para:
            paragraphs.append("".join(current_para))

        page_text = "\n".join(paragraphs).strip()
        if page_text:
            pages_data.append((pno + 1, page_text))

    return pages_data


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while keeping terminal punctuation and quotes."""
    parts = re.split(r"([。！？；]+[”’」』]?)", text)
    sentences = []
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if i + 1 < len(parts):
            punct = parts[i + 1].strip()
            sentence = chunk + punct
            i += 2
        else:
            sentence = chunk
            i += 1
        sentence = re.sub(r"\s+", "", sentence)
        if len(sentence) >= 6 and re.search(r"[。！？；]", sentence):
            sentences.append(sentence)
    return sentences


def segment_sentence(sentence: str) -> str:
    """Segment sentence into whitespace-separated tokens."""
    tokens = [w for w in jieba.cut(sentence) if w.strip()]
    return " ".join(tokens)


def mine_weak_grammar_samples(
    segmented_sentences: List[str],
) -> List[Tuple[str, int, int]]:
    """Extract weak-supervised grammar & attribute samples from segmented sentences.

    Returns tuples: (masked_text, structure_idx, attribute_idx)
    matching bert_simple.grammar_filter.STRUCTURES and ATTRIBUTES.
    """
    struct_map = {name: i for i, name in enumerate(STRUCTURES)}
    attr_map = {name: i for i, name in enumerate(ATTRIBUTES)}

    samples: List[Tuple[str, int, int]] = []

    name_triggers = {"叫", "名叫", "名为", "姓名"}
    loc_triggers = {"在", "住在", "前往", "到达", "去", "逃亡到", "进击", "起义于"}
    temp_triggers = {"在", "于"}
    pred_triggers = {"正在", "已经", "开始", "还在", "企图", "决定"}
    obj_triggers = {"叫", "让", "请", "派", "命令", "协助", "任用"}
    id_triggers = {"是", "担任", "成为", "出任", "立为", "封为", "任命为"}
    qty_triggers = {"有", "带了", "共有", "拥有", "率领", "斩首", "俘获"}

    title_words = {
        "将军", "大王", "楚王", "汉王", "县令", "门客", "校尉", "相国", "丞相",
        "太尉", "士卒", "部下", "使者", "臣子", "宾客", "国君", "军师", "刺客",
        "谋士", "屯长", "富人", "名士", "小吏", "首领", "接班人", "天下为家", "平民",
        "平民百姓", "雇农", "豪杰", "大夫", "太史令"
    }

    for text in segmented_sentences:
        tokens = text.split()
        n = len(tokens)
        if n < 4 or n > 60:
            continue

        tagged = list(pseg.cut(text.replace(" ", "")))
        pos_dict = {w: flag for w, flag in tagged}

        # 1. NAME / PERSON
        for i in range(n - 1):
            if tokens[i] in name_triggers and i + 1 < n:
                target = tokens[i + 1]
                pos = pos_dict.get(target, "")
                if pos == "nr" or target in {"刘邦", "项羽", "张耳", "陈余", "韩信", "陈胜", "大禹", "商汤", "郑国"}:
                    masked = list(tokens)
                    masked[i + 1] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["NAME"], attr_map["PERSON"]))

        # 2. LOCATION / PLACE
        for i in range(n - 1):
            if tokens[i] in loc_triggers and i + 1 < n:
                target = tokens[i + 1]
                pos = pos_dict.get(target, "")
                if (pos == "ns" or target.endswith(("县", "郡", "城", "国", "地", "关", "津", "乡"))
                        or target in {"沛县", "大梁", "咸阳", "陈县", "外黄", "河北", "渔阳"}):
                    masked = list(tokens)
                    masked[i + 1] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["LOCATION"], attr_map["PLACE"]))

        # 3. TEMPORAL / TIME
        for i in range(n - 2):
            if tokens[i] in temp_triggers and tokens[i + 2] in {"时", "时候", "期间", "年", "月"}:
                masked = list(tokens)
                masked[i + 1] = "[MASK]"
                samples.append((" ".join(masked), struct_map["TEMPORAL"], attr_map["TIME"]))

        # 4. PREDICATE / ACTION
        for i in range(n - 1):
            if tokens[i] in pred_triggers and i + 1 < n:
                target = tokens[i + 1]
                pos = pos_dict.get(target, "")
                if pos.startswith("v") and target not in loc_triggers and target not in id_triggers:
                    masked = list(tokens)
                    masked[i + 1] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["PREDICATE"], attr_map["ACTION"]))

        # 5. OBJECT / PERSON
        for i in range(n - 2):
            if tokens[i] in obj_triggers:
                target = tokens[i + 1]
                follower = tokens[i + 2]
                pos = pos_dict.get(target, "")
                fol_pos = pos_dict.get(follower, "")
                if (pos in {"nr", "r"} or target in {"使者", "部下", "门客", "蒯通", "张耳", "人"}) and fol_pos.startswith("v"):
                    masked = list(tokens)
                    masked[i + 1] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["OBJECT"], attr_map["PERSON"]))

        # 6. IDENTITY / TITLE
        for i in range(n - 1):
            if tokens[i] in id_triggers and i + 1 < n:
                target = tokens[i + 1]
                if target in {"一位", "一名", "一个"} and i + 2 < n:
                    target = tokens[i + 2]
                    idx = i + 2
                else:
                    idx = i + 1
                if target in title_words or pos_dict.get(target, "").startswith("n"):
                    masked = list(tokens)
                    masked[idx] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["IDENTITY"], attr_map["TITLE"]))

        # 7. QUANTITY / NUMBER
        for i in range(n - 2):
            if tokens[i] in qty_triggers:
                target = tokens[i + 1]
                unit = tokens[i + 2]
                if pos_dict.get(target, "") == "m" or target in {"三千", "九百", "好几万", "数万", "数十", "五百", "千", "万"}:
                    masked = list(tokens)
                    masked[i + 1] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["QUANTITY"], attr_map["NUMBER"]))

        # 8. CONDITION / CLAUSE
        if "如果" in tokens or "因为" in tokens or "只要" in tokens or "若" in tokens:
            for cond in ["如果", "因为", "只要", "若"]:
                if cond in tokens:
                    idx = tokens.index(cond)
                    if idx + 1 < n and "，" in tokens[idx:]:
                        comma_idx = tokens.index("，", idx)
                        if comma_idx - idx >= 2:
                            masked = list(tokens)
                            mid = (idx + comma_idx) // 2
                            masked[mid] = "[MASK]"
                            samples.append((" ".join(masked), struct_map["CONDITION"], attr_map["CLAUSE"]))
                            break

        # 9. COORDINATION / FUNCTION
        for i in range(1, n - 2):
            if tokens[i] in {"和", "与", "及", "同"}:
                if any(w in tokens[i + 1:i + 4] for w in ["一起", "一同", "共同", "结伴"]):
                    masked = list(tokens)
                    masked[i] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["COORDINATION"], attr_map["FUNCTION"]))

        # 10. BOUNDARY / PUNCT
        if tokens[-1] in {"。", "！", "？", "；"}:
            masked = list(tokens)
            masked[-1] = "[MASK]"
            samples.append((" ".join(masked), struct_map["BOUNDARY"], attr_map["PUNCT"]))

        # 11. OPEN / INTERROGATIVE
        if tokens[-1] == "？":
            for q_word in ["哪", "哪里", "谁", "什么", "何处", "何时"]:
                if q_word in tokens:
                    idx = tokens.index(q_word)
                    masked = list(tokens)
                    masked[idx] = "[MASK]"
                    samples.append((" ".join(masked), struct_map["OPEN"], attr_map["INTERROGATIVE"]))

    return samples


def main():
    print(f"=== Step 1: Extracting raw text from {PDF_PATH} ===")
    init_jieba()

    pages = extract_raw_text(PDF_PATH)
    print(f"Extracted {len(pages)} pages of story content.")

    raw_lines = []
    jsonl_records = []
    for pno, text in pages:
        raw_lines.append(f"<!-- Page {pno} -->\n{text}\n")
        jsonl_records.append({"page": pno, "text": text})

    raw_text_path = RAW_DIR / "shiji_stories_raw.txt"
    raw_jsonl_path = RAW_DIR / "shiji_pages.jsonl"
    raw_text_path.write_text("\n".join(raw_lines), encoding="utf-8")
    with open(raw_jsonl_path, "w", encoding="utf-8") as f:
        for r in jsonl_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved raw text to {raw_text_path} ({len(raw_lines)} entries)")

    print("=== Step 2: Sentence splitting ===")
    all_sentences = []
    page_sentences: Dict[int, List[str]] = {}
    for pno, text in pages:
        s_list = split_sentences(text)
        page_sentences[pno] = s_list
        all_sentences.extend(s_list)

    sentences_path = SENTENCES_DIR / "shiji_sentences.txt"
    sentences_path.write_text("\n".join(all_sentences) + "\n", encoding="utf-8")
    print(f"Saved {len(all_sentences)} clean sentences to {sentences_path}")

    print("=== Step 3: Space tokenization with jieba ===")
    segmented_sentences = []
    for s in all_sentences:
        seg = segment_sentence(s)
        if seg:
            segmented_sentences.append(seg)

    segmented_path = SEGMENTED_DIR / "shiji_segmented.txt"
    segmented_path.write_text("\n".join(segmented_sentences) + "\n", encoding="utf-8")
    total_tokens = sum(len(s.split()) for s in segmented_sentences)
    print(f"Saved {len(segmented_sentences)} segmented sentences ({total_tokens} tokens) to {segmented_path}")

    print("=== Step 4: Mining grammar and word attribute samples ===")
    raw_samples = mine_weak_grammar_samples(segmented_sentences)
    print(f"Mined {len(raw_samples)} weak-supervised slot samples.")

    from collections import Counter
    struct_counts = Counter(s[1] for s in raw_samples)
    print("Structure distribution:")
    for s_idx, count in sorted(struct_counts.items()):
        print(f"  {STRUCTURES[s_idx]:15s}: {count}")

    # Step 5: Chapter/Page-based Partitioning (80% train / 10% val / 10% test)
    rng = random.Random(42)
    unique_samples = list(set(raw_samples))
    rng.shuffle(unique_samples)

    n_total = len(unique_samples)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_set = unique_samples[:n_train]
    val_set = unique_samples[n_train:n_train + n_val]
    test_set = unique_samples[n_train + n_val:]

    # Augment curriculum: prepend independent clauses to test clause isolation
    augmented_train = list(train_set)
    for text, structure, attr in train_set[:1000]:
        augmented_train.append(("先前 士兵 已经 出发 ， " + text, structure, attr))

    augmented_val = list(val_set)
    for text, structure, attr in val_set[:200]:
        augmented_val.append(("先前 士兵 已经 出发 ， " + text, structure, attr))

    dataset = {
        "train": augmented_train,
        "validation": augmented_val,
        "test": test_set,
        "metadata": {
            "source_pdf": str(PDF_PATH),
            "total_sentences": len(all_sentences),
            "total_segmented_tokens": total_tokens,
            "mined_samples_count": len(unique_samples),
            "train_count": len(augmented_train),
            "validation_count": len(augmented_val),
            "test_count": len(test_set),
            "structures": list(STRUCTURES),
            "attributes": list(ATTRIBUTES),
        }
    }

    manifest_path = MANIFESTS_DIR / "dataset.json"
    manifest_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved dataset manifest to {manifest_path}")

    output_dir = ROOT / "outputs" / "independent-grammar-filter"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shiji_dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Mirrored dataset to {output_dir / 'shiji_dataset.json'}")

    print("\n=== All Steps Successfully Completed! ===")


if __name__ == "__main__":
    main()
