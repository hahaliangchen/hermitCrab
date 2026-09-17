"""Build thematic Shiji unit datasets for localized curriculum training and evaluation.

Creates 6 thematic historical units spanning 100% of the annotated Shiji groups:
1. unit1_chuhan_founding: 楚汉争霸与开国功臣
2. unit2_warring_states_retainers: 战国风云与将相门客
3. unit3_han_generals_frontier: 大汉边疆与帝国名将
4. unit4_qin_empire_reform: 大秦帝国与法家变法
5. unit5_spring_autumn_hegemony: 春秋诸侯与先秦霸业
6. unit6_han_society_and_economy: 汉代儒林与货殖风云

Strictly guarantees data integrity via connected component clustering:
- No evidence sentence (source_id) crosses splits.
- No semantic triple (semantic_key) crosses splits.
- Generates both combined JSONL and explicit train/dev/test split files.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))
from validate_relation_training_data import validate_groups

OUTPUT_DIR = ROOT / "data" / "shiji" / "units"
SPLITS_DIR = OUTPUT_DIR / "splits"

UNITS_DEF = {
    "unit1_chuhan_founding": {
        "unit_id": 1,
        "name": "楚汉争霸与开国功臣",
        "description": "围绕汉高祖刘邦、项羽、韩信、张耳、陈余、周勃、萧何等人的封王拜将、诛杀背叛与楚汉争霸历史。",
        "chapters": [
            "高祖本纪", "项羽本纪", "淮阴侯列传", "张耳陈余列传", 
            "绛侯周勃世家", "汉初勋戚列传(刘敬/叔孙通/季布)", "陈涉世家"
        ]
    },
    "unit2_warring_states_retainers": {
        "unit_id": 2,
        "name": "战国风云与将相门客",
        "description": "围绕孟尝君养士收租、廉颇蔺相如完璧归赵与将相和、田单火牛阵、孙子吴起兵学等战国风云。",
        "chapters": [
            "孟尝君列传", "廉颇蔺相如列传", "田单/屈原/刺客列传", 
            "兵学将相列传(伯夷/孙子/吴起)", "孟子荀卿列传"
        ]
    },
    "unit3_han_generals_frontier": {
        "unit_id": 3,
        "name": "大汉边疆与帝国名将",
        "description": "围绕汉武帝开疆拓土、卫青霍去病出击匈奴、飞将军李广抗匈、张骞出使大宛西域等边疆战功。",
        "chapters": [
            "卫将军骠骑列传", "李将军列传", "大宛列传", 
            "司马相如列传", "张释之冯唐列传"
        ]
    },
    "unit4_qin_empire_reform": {
        "unit_id": 4,
        "name": "大秦帝国与法家变法",
        "description": "围绕商鞅变法、秦始皇一统天下、李斯蒙恬辅政与沙丘权谋、酷吏执法与早周先秦世系。",
        "chapters": [
            "商君/苏秦列传", "秦始皇本纪", "李斯/蒙恬列传", 
            "酷吏列传", "周本纪", "夏本纪", "殷本纪"
        ]
    },
    "unit5_spring_autumn_hegemony": {
        "unit_id": 5,
        "name": "春秋诸侯与先秦霸业",
        "description": "围绕春秋五霸齐桓晋文争霸、勾践卧薪尝胆灭吴、孔子周游列国与儒家先贤修齐治平。",
        "chapters": [
            "春秋诸侯世家(齐/晋/楚)", "越王勾践世家", "孔子世家"
        ]
    },
    "unit6_han_society_and_economy": {
        "unit_id": 6,
        "name": "汉代儒林与货殖风云",
        "description": "围绕汉代儒学兴起独尊儒术、货殖列传天下商贾富商大贾、滑稽讽谏与文景之治社会百态。",
        "chapters": [
            "货殖/滑稽列传", "儒林列传", "孝文本纪/武帝"
        ]
    }
}

CHAPTER_RANGES = [
    (1, 335, "夏本纪"),
    (336, 503, "殷本纪"),
    (504, 823, "周本纪"),
    (824, 1184, "秦始皇本纪"),
    (1185, 1376, "项羽本纪"),
    (1377, 1662, "高祖本纪"),
    (1663, 1769, "孝文本纪/武帝"),
    (1770, 2364, "春秋诸侯世家(齐/晋/楚)"),
    (2365, 2561, "越王勾践世家"),
    (2562, 2908, "孔子世家"),
    (2909, 3237, "陈涉世家"),
    (3238, 3451, "绛侯周勃世家"),
    (3452, 3862, "兵学将相列传(伯夷/孙子/吴起)"),
    (3863, 4347, "商君/苏秦列传"),
    (4348, 4540, "孟子荀卿列传"),
    (4541, 4767, "孟尝君列传"),
    (4768, 5042, "廉颇蔺相如列传"),
    (5043, 5696, "田单/屈原/刺客列传"),
    (5697, 6041, "李斯/蒙恬列传"),
    (6042, 6262, "张耳陈余列传"),
    (6263, 7131, "淮阴侯列传"),
    (7132, 7678, "汉初勋戚列传(刘敬/叔孙通/季布)"),
    (7679, 8092, "张释之冯唐列传"),
    (8093, 8523, "李将军列传"),
    (8524, 9276, "卫将军骠骑列传"),
    (9277, 9637, "司马相如列传"),
    (9638, 9832, "儒林列传"),
    (9833, 10038, "酷吏列传"),
    (10039, 10203, "大宛列传"),
    (10204, 10631, "货殖/滑稽列传"),
]


def find_chapter(line_no: int) -> str:
    for start, end, name in CHAPTER_RANGES:
        if start <= line_no <= end:
            return name
    return "其他篇章"


def _normalized(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split())


QUALITY_EXCLUSIONS = {
    # These groups are unambiguous extraction failures in the current draft.
    "shiji-rel-v2-line_5760_appoint_15_17": "拜官吏为老师 is a learning relationship, not an appointment; 人/官吏/老师 are generic roles.",
    "shiji-rel-v2-line_6472_appoint_5_7": "the appointment extractor used 前往 as the actor in 派张良前往立韩信为齐王.",
    "shiji-rel-v2-line_9997_appoint_9_14": "义 is a truncated mention in 义的弟弟义纵; the one-token schema cannot safely recover 义纵.",
    "shiji-rel-v2-line_9307_KINSHIP_2_6": "王孙 is a truncated parent in 卓王孙的女儿卓文君; the exact parent is not represented.",
    "shiji-rel-v2-line_5469_KINSHIP_16_19": "国君 is a generic title, not an identified person, in 侠累是韩国国君的叔父.",
    # Keep the gate reproducible if somebody points this builder at the older
    # pre-regeneration draft again.
    "shiji-rel-v2-line_6948_TITLE_LINK_14_13": "晋见 is a verb, not a person/name endpoint.",
    "shiji-rel-v2-line_7287_KINSHIP_30_33": "姬 is a truncated mention in 骊姬的儿子奚齐.",
    "shiji-rel-v2-line_7048_KINSHIP_0_3": "郦 is a truncated mention in 郦食其的儿子郦疥.",
    "shiji-rel-v2-line_3372_KINSHIP_0_4": "栗姬生 is a glued phrase, not an atomic parent entity.",
    "shiji-rel-v2-line_4215_KINSHIP_2_6": "the source names 燕文侯的夫人 as the mother, not 燕文侯.",
    "shiji-rel-v2-line_5505_ALIAS_4_7": "燕国人 is a generic group and is not the person whose alias is荆卿.",
    "shiji-rel-v2-line_9564_ALIAS_22_26": "凤/雄 are bird gender terms, not historical person aliases.",
}

# High-precision lexical guard for old draft rows.  These values are only
# rejected in identity/title contexts; ordinary historical names such as 吴
# or 梁 are not globally blacklisted.
TITLE_LINK_NON_ENTITY_ENDPOINTS = {
    "晋见", "侍奉", "敬劳", "爱抚", "慎重考虑", "任符玺", "张苍任",
    "渊博", "阳陵", "嘉许", "师事荀", "闻言", "任邺", "太初", "骆越",
    "南越武", "汉廷", "元鼎", "恩宠", "官居", "府", "赵胡派", "封张良",
    "任太中", "汉孝文", "秦孝", "光禄", "鲁军",
}
ALIAS_NON_ENTITY_ENDPOINTS = {"阿房宫", "宫殿", "燕国人", "雄", "雌"}
GENERIC_ORIGIN_ENDPOINTS = {"祖先", "先祖", "后代", "国人"}


def _quality_issue(group: dict) -> str | None:
    """Return a high-confidence quality reason, or None for reviewable data."""

    group_id = str(group.get("group_id", ""))
    if group_id in QUALITY_EXCLUSIONS:
        return QUALITY_EXCLUSIONS[group_id]

    relation_types = {str(f.get("relation_type", "")) for f in group.get("facts", [])}
    role_values = [
        str(value)
        for fact in group.get("facts", [])
        for role, value in fact.get("roles", {}).items()
        if role != "predicate"
    ]

    if relation_types <= {"TITLE_OF", "HAS_TITLE"}:
        bad = sorted(set(role_values) & TITLE_LINK_NON_ENTITY_ENDPOINTS)
        if bad:
            return f"title/identity endpoint is a verb, place, time expression, or glued phrase: {', '.join(bad)}"

    if "ALIAS_OF" in relation_types:
        bad = sorted(set(role_values) & ALIAS_NON_ENTITY_ENDPOINTS)
        if bad:
            return f"alias endpoint is a generic/non-person expression: {', '.join(bad)}"

    if "BORN_IN" in relation_types:
        bad = sorted(set(role_values) & GENERIC_ORIGIN_ENDPOINTS)
        if bad:
            return f"origin endpoint is a generic description rather than an identified entity: {', '.join(bad)}"

    if "APPOINT" in relation_types:
        appointment_roles = [
            f.get("roles", {}) for f in group.get("facts", [])
            if f.get("relation_type") == "APPOINT"
        ]
        if appointment_roles:
            roles = appointment_roles[0]
            if roles.get("actor") in {"人", "有人", "前往"}:
                return f"appointment actor is a generic/function expression: {roles.get('actor')}"
            if roles.get("recipient") in {"官吏", "义"}:
                return f"appointment recipient is generic or truncated: {roles.get('recipient')}"
            if roles.get("title") in {"老师"}:
                return "appointment title is a learning-role noun, not an official title"

    kinship_types = {
        "CHILD_OF", "PARENT_OF", "GRANDCHILD_OF", "GRANDPARENT_OF",
        "SIBLING_OF", "SPOUSE_OF", "UNCLE_OR_AUNT_OF", "NEPHEW_OR_NIECE_OF",
        "ANCESTOR_OF", "DESCENDANT_OF",
    }
    if relation_types & kinship_types:
        bad = sorted(set(role_values) & {"国君"})
        if bad:
            return "kinship endpoint is a generic title without a named person: 国君"

    return None


def _semantic_key(fact: dict):
    return (
        str(fact["relation_type"]),
        tuple(sorted((str(k), _normalized(str(v))) for k, v in fact.get("roles", {}).items())),
        tuple(sorted((str(k), _normalized(str(v))) for k, v in fact.get("qualifiers", {}).items())),
    )


def _connected_components(groups):
    """Cluster groups globally, including groups from different thematic units."""

    parent = list(range(len(groups)))

    def find(index: int) -> int:
        if parent[index] != index:
            parent[index] = find(parent[index])
        return parent[index]

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners = {}
    for index, group in enumerate(groups):
        keys = set()
        for fact in group.get("facts", []):
            keys.add(("fact", str(fact.get("fact_id", ""))))
            evidence = fact.get("evidence", {})
            keys.add(("source", str(evidence.get("source_id", ""))))
            keys.add(("semantic", _semantic_key(fact)))
        for key in keys:
            previous = owners.get(key)
            if previous is None:
                owners[key] = index
            else:
                union(index, previous)

    clusters = defaultdict(list)
    for index, group in enumerate(groups):
        clusters[find(index)].append(group)
    return list(clusters.values())


def _holdout_targets(unit_counts):
    """Set stable group-level targets without pretending tiny units are large."""

    targets = {}
    for unit_id, total in unit_counts.items():
        if total < 3:
            targets[unit_id] = {"train": total, "dev": 0, "test": 0}
            continue
        minimum = 2 if total >= 8 else 1
        dev = max(minimum, int(round(total * 0.15)))
        test = max(minimum, int(round(total * 0.15)))
        while dev + test >= total:
            if dev >= test and dev > 1:
                dev -= 1
            elif test > 1:
                test -= 1
            else:
                break
        targets[unit_id] = {"train": total - dev - test, "dev": dev, "test": test}
    return targets


def _assign_global_splits(groups, unit_counts):
    """Assign whole global components while staying close to each unit target."""

    targets = _holdout_targets(unit_counts)
    current = {
        unit_id: {"train": 0, "dev": 0, "test": 0}
        for unit_id in unit_counts
    }
    clusters = _connected_components(groups)
    # Place cross-unit and larger components first so their shared split is
    # decided before small independent groups fill the targets.
    clusters.sort(
        key=lambda component: (
            -len({str(g["_unit_uid"]) for g in component}),
            -len(component),
            min(str(g.get("group_id", "")) for g in component),
        )
    )

    split_order = {"dev": 0, "test": 1, "train": 2}
    for component in clusters:
        component_counts = Counter(str(g["_unit_uid"]) for g in component)
        costs = {}
        for split in ("dev", "test", "train"):
            cost = 0.0
            for unit_id, amount in component_counts.items():
                before = abs(current[unit_id][split] - targets[unit_id][split])
                after = abs(current[unit_id][split] + amount - targets[unit_id][split])
                cost += after - before
            # Avoid needlessly overshooting holdouts once train is still
            # available.  This is a soft tie-break, not a hard class filter.
            if split != "train":
                cost += sum(
                    max(0, current[unit_id][split] + amount - targets[unit_id][split]) * 0.25
                    for unit_id, amount in component_counts.items()
                )
            costs[split] = (cost, split_order[split])
        chosen = min(costs, key=lambda split: costs[split])
        for group in component:
            group["split"] = chosen
        for unit_id, amount in component_counts.items():
            current[unit_id][chosen] += amount

    return clusters, targets, current


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    unit_map = {}
    for uid, info in UNITS_DEF.items():
        for ch in info["chapters"]:
            unit_map[ch] = uid

    raw_path = ROOT / "data" / "shiji" / "manifests" / "relation_training_v2_draft.jsonl"
    all_groups = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    unit_groups = {uid: [] for uid in UNITS_DEF}
    filtered_groups = []
    excluded_groups = []
    unmapped = []

    for item in all_groups:
        item = deepcopy(item)
        gid = item.get("group_id", "")
        m = re.search(r"line_(\d+)", gid)
        if m:
            l_no = int(m.group(1))
            ch = find_chapter(l_no)
            uid = unit_map.get(ch)
            item["chapter_tag"] = ch
            if uid:
                item["_unit_uid"] = uid
                issue = _quality_issue(item)
                if issue:
                    excluded_groups.append({
                        "group_id": gid,
                        "unit_id": uid,
                        "chapter_tag": ch,
                        "reason": issue,
                    })
                else:
                    unit_groups[uid].append(item)
                    filtered_groups.append(item)
            else:
                unmapped.append((ch, item))
        else:
            unmapped.append(("未知", item))

    if unmapped:
        print(f"Warning: {len(unmapped)} groups were not mapped into units: {[x[0] for x in unmapped]}")

    unit_counts = {uid: len(groups) for uid, groups in unit_groups.items() if groups}
    clusters, targets, current = _assign_global_splits(filtered_groups, unit_counts)

    all_unit_items = []
    for item in filtered_groups:
        uid = item.pop("_unit_uid")
        # The serialized unit is copied from the precomputed mapping; it is
        # never inferred from the newly assigned split.
        item["unit_id"] = uid
        all_unit_items.append(item)

    # This is deliberately a global validation.  Per-unit validation alone
    # cannot detect a semantic duplicate split between two different units.
    global_validation = validate_groups(all_unit_items)

    summary_report = {}
    for uid, info in UNITS_DEF.items():
        unit_items = sorted(
            [g for g in all_unit_items if g["unit_id"] == uid],
            key=lambda g: (g.get("split", ""), g.get("group_id", "")),
        )
        if not unit_items:
            continue

        # 1. Write combined unit file

        # 1. Write combined unit file
        out_file = OUTPUT_DIR / f"{uid}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in unit_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 2. Write dedicated split files (group-level) and flattened sample-level files
        unit_split_dir = SPLITS_DIR / uid
        unit_split_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ("train", "dev", "test"):
            split_groups = [g for g in unit_items if g["split"] == split_name]
            split_group_file = unit_split_dir / f"{split_name}.jsonl"
            with open(split_group_file, "w", encoding="utf-8") as f:
                for g in split_groups:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")

            # Flattened sample file for direct MLM/inference testing
            split_samples_file = unit_split_dir / f"samples_{split_name}.jsonl"
            with open(split_samples_file, "w", encoding="utf-8") as f:
                for g in split_groups:
                    for s in g["samples"]:
                        sample_row = {
                            "group_id": g["group_id"],
                            "sample_id": s["sample_id"],
                            "fact_id": s["fact_id"],
                            "split": split_name,
                            "chapter_tag": g.get("chapter_tag", ""),
                            "tokens": s["tokens"],
                            "text": " ".join(s["tokens"]),
                        "answer": s["answer"],
                        "target_slot": s["target_slot"],
                        "surface_kind": s.get("surface_kind", ""),
                        "role_spans": s.get("role_spans", {}),
                        "background_spans": s.get("background_spans", []),
                        "hard_negatives": s.get("hard_negatives", []),
                        }
                        f.write(json.dumps(sample_row, ensure_ascii=False) + "\n")

        # Collect entities and stats
        entity_counter = Counter()
        for g in unit_items:
            for f in g.get("facts", []):
                roles = f.get("roles", {})
                if "subject" in roles:
                    entity_counter[roles["subject"]] += 1
                if "object" in roles:
                    entity_counter[roles["object"]] += 1

        n_train = len([g for g in unit_items if g["split"] == "train"])
        n_dev = len([g for g in unit_items if g["split"] == "dev"])
        n_test = len([g for g in unit_items if g["split"] == "test"])

        train_samples = sum(len(g["samples"]) for g in unit_items if g["split"] == "train")
        dev_samples = sum(len(g["samples"]) for g in unit_items if g["split"] == "dev")
        test_samples = sum(len(g["samples"]) for g in unit_items if g["split"] == "test")

        summary_report[uid] = {
            "name": info["name"],
            "description": info["description"],
            "total_groups": len(unit_items),
            "total_samples": len(unit_items) * 4,
            "train_groups": n_train,
            "train_samples": train_samples,
            "dev_groups": n_dev,
            "dev_samples": dev_samples,
            "test_groups": n_test,
            "test_samples": test_samples,
            "chapters_covered": sorted(list(set(g["chapter_tag"] for g in unit_items))),
            "core_entities": [item[0] for item in entity_counter.most_common(15)],
            "combined_file": str(out_file.relative_to(ROOT)),
            "split_directory": str(unit_split_dir.relative_to(ROOT)),
        }

    quality_report = {
        "schema_version": 1,
        "status": "cleaned_draft_requires_manual_audit",
        "source_manifest": str(raw_path.relative_to(ROOT)),
        "input_groups": len(all_groups),
        "unmapped_groups": len(unmapped),
        "excluded_groups": len(excluded_groups),
        "kept_groups": len(filtered_groups),
        "global_components": len(clusters),
        "cross_unit_components": sum(
            1 for component in clusters
            if len({str(g["unit_id"]) if "unit_id" in g else str(unit_map.get(g["chapter_tag"], "")) for g in component}) > 1
        ),
        "excluded": excluded_groups,
        "unmapped": [
            {"chapter_tag": chapter, "group_id": item.get("group_id", "")}
            for chapter, item in unmapped
        ],
        "split_targets": targets,
        "split_actual": current,
        "global_validation": global_validation,
        "notes": [
            "Quality exclusions are high-confidence rejects, not a claim that every remaining heuristic relation is correct.",
            "Source, fact ID, and semantic-key connected components are assigned globally across all six units.",
            "Units with fewer than eight groups cannot have a statistically stable independent holdout; their scores should be reported as exploratory.",
            "Inherited hard negatives still require manual review.",
        ],
    }
    (OUTPUT_DIR / "shiji_units_quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = OUTPUT_DIR / "shiji_units_summary.json"
    report_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate comprehensive README.md in data/shiji/units/
    readme_content = generate_readme(summary_report, quality_report)
    (OUTPUT_DIR / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"Successfully generated {len(summary_report)} Shiji unit datasets in {OUTPUT_DIR}")
    print(f"Total groups kept: {sum(r['total_groups'] for r in summary_report.values())} / {len(all_groups)}")
    print(f"High-confidence quality exclusions: {len(excluded_groups)}")
    print(f"Global connected components: {len(clusters)}; cross-unit components: {quality_report['cross_unit_components']}")
    for uid, r in summary_report.items():
        print(f"\n【{r['name']}】 ({uid}):")
        print(f"  合并文件: {r['combined_file']}")
        print(f"  组数: 总计 {r['total_groups']} (训练={r['train_groups']}, 验证={r['dev_groups']}, 测试={r['test_groups']})")
        print(f"  样本条数: 总计 {r['total_samples']} (训练={r['train_samples']}, 验证={r['dev_samples']}, 测试={r['test_samples']})")
        print(f"  核心角色: {', '.join(r['core_entities'][:8])}")


def generate_readme(summary: dict, quality_report: dict) -> str:
    lines = [
        "# 《史记》历史单元关系抽取训练数据集清单 (Shiji Unit Datasets)",
        "",
        "本数据集将《史记》10631 句语料与标注事实，依据太史公本纪、世家、列传的脉络，划分为 **6 个主题历史单元**。",
        "每个单元构成一个高内聚、低跨度的局部历史语义网络，杜绝跨朝代/跨篇章的名人光环先验偏置（如蔺相如吞噬魏子、刘邦吞噬楚怀王等问题）。",
        "",
        "## 数据集完整性与无泄漏保证",
        f"1. **全局原子连通分量划分**：输入 {quality_report['input_groups']} 组，保留 {quality_report['kept_groups']} 组；凡共享同一证据句（`source_id`）、事实 ID 或语义事实（`semantic_key`）的组，先在 6 个单元之间统一聚类，再分配到同一个 train/dev/test。",
        f"2. **高置信度清洗**：剔除 {quality_report['excluded_groups']} 组明确的伪实体、截断实体或错误关系，并在 `shiji_units_quality_report.json` 中保留逐组原因；剩余规则抽取仍需人工抽查。",
        "3. **双层文件组织**：既提供标准组级合并文件（供 `train_relation_pairs.py` 训练），也提供子目录拆分文件及扁平化样本文件（`samples_*.jsonl`，供常规 MLM/评估使用）。",
        "",
        "## 单元概览表",
        "",
        "| 单元 ID | 单元名称 | 总组数 | 样本总数 | Train 组(条) | Dev 组(条) | Test 组(条) | 核心实体 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for uid, r in summary.items():
        lines.append(
            f"| `{uid}` | **{r['name']}** | {r['total_groups']} | {r['total_samples']} | "
            f"{r['train_groups']} ({r['train_samples']}) | {r['dev_groups']} ({r['dev_samples']}) | "
            f"{r['test_groups']} ({r['test_samples']}) | {', '.join(r['core_entities'][:6])} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 清洗与评估注意事项",
        "",
        f"- 全局连通分量数：{quality_report['global_components']}；跨单元连通分量：{quality_report['cross_unit_components']}。",
        "- 每个单元的 split 以 group 为单位分配；8 组以上的单元至少保留 2 组 dev 和 2 组 test。少于 8 组的单元只能保留单组 holdout，结果仅作探索性参考。",
        "- 本次高置信度剔除：",
    ])
    for item in quality_report["excluded"]:
        lines.append(f"  - `{item['group_id']}`：{item['reason']}")
    lines.extend([
        "- hard negatives 继承自原事实清单，仍带有人工复核标记，不能视为已完成的负例标注。",
        "",
        "## 各单元详细说明",
        "",
    ])

    for uid, r in summary.items():
        lines.extend([
            f"### 【{r['name']}】 (`{uid}`)",
            f"- **历史背景**：{r['description']}",
            f"- **涵盖篇章**：{', '.join(r['chapters_covered'])}",
            f"- **主文件**：`{r['combined_file']}`",
            f"- **拆分目录**：`{r['split_directory']}/`",
            f"  - `train.jsonl` / `samples_train.jsonl`：{r['train_groups']} 组 ({r['train_samples']} 条样本)",
            f"  - `dev.jsonl` / `samples_dev.jsonl`：{r['dev_groups']} 组 ({r['dev_samples']} 条样本)",
            f"  - `test.jsonl` / `samples_test.jsonl`：{r['test_groups']} 组 ({r['test_samples']} 条样本)",
            f"- **高频角色/实体候选**（仅作统计，不等同于知识库实体）：{', '.join(r['core_entities'])}",
            "",
        ])

    lines.extend([
        "## 使用方法与训练示例",
        "",
        "### 1. 使用 relation_pair 模型进行单元训练",
        "```bash",
        "# 训练战国风云与将相门客单元 (Unit 2)",
        "python examples/train_relation_pairs.py data/shiji/units/unit2_warring_states_retainers.jsonl \\",
        "    --output-dir outputs/unit2-256d-4l \\",
        "    --hidden-size 256 \\",
        "    --num-layers 4 \\",
        "    --epochs 20",
        "```",
        "",
        "### 2. 使用 evaluate_by_unit 进行多单元多维评测与探针诊断",
        "```bash",
        "# 评估模型在各单元上的准确率、负例击败率及核心探针 (如 魏子 vs 蔺相如)",
        "python examples/evaluate_by_unit.py outputs/unit2-256d-4l",
        "```",
        "",
        "### 3. 直接读取扁平化样本 (samples_*.jsonl)",
        "每条样本为一行独立 JSON：",
        "```json",
        "{",
        '  "group_id": "shiji-rel-v2-line_4781_TITLE_LINK_7_8",',
        '  "sample_id": "shiji_fact_0370:base",',
        '  "split": "dev",',
        '  "tokens": ["宦者", "令缪贤", "说", "：", "“", "我", "的", "[MASK]", "蔺相如", "可以", "充任", "使者", "。", "”"],',
        '  "answer": "舍人",',
        '  "target_slot": "object",',
        '  "hard_negatives": ["御史大夫", "郎中令", "中山王", "汉武帝", "太子"]',
        "}",
        "```",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    build()
