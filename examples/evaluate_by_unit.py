"""Evaluate relation extraction checkpoints across Shiji biographical units.

Computes accuracy, candidate selection, hard negative defeat rate across all 4 units,
and specifically probes key relation diagnostic targets (e.g. 魏子 vs 蔺相如, 皇帝 vs 怀王).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

import torch
from bert_simple.dynamic_qk_model import DynamicQKLocalRelationMarginBertForMaskedLM
from bert_simple.tokenizer import SimpleBertTokenizer
from bert_simple.relation_pair_training import (
    CANDIDATE_POLICY,
    evaluate,
    visible_inputs,
    sample_logits,
)
from validate_relation_training_data import load_groups


UNITS = [
    ("unit1_chuhan_founding", "楚汉争霸与开国功臣", ROOT / "data" / "shiji" / "units" / "unit1_chuhan_founding.jsonl"),
    ("unit2_warring_states_retainers", "战国风云与将相门客", ROOT / "data" / "shiji" / "units" / "unit2_warring_states_retainers.jsonl"),
    ("unit3_han_generals_frontier", "大汉边疆与帝国名将", ROOT / "data" / "shiji" / "units" / "unit3_han_generals_frontier.jsonl"),
    ("unit4_qin_empire_reform", "大秦帝国与法家变法", ROOT / "data" / "shiji" / "units" / "unit4_qin_empire_reform.jsonl"),
    ("unit5_spring_autumn_hegemony", "春秋诸侯与先秦霸业", ROOT / "data" / "shiji" / "units" / "unit5_spring_autumn_hegemony.jsonl"),
    ("unit6_han_society_and_economy", "汉代儒林与货殖风云", ROOT / "data" / "shiji" / "units" / "unit6_han_society_and_economy.jsonl"),
]

PROBES = [
    {
        "name": "魏子 vs 蔺相如 (孟尝君舍人收租)",
        "unit": "unit2_warring_states_retainers",
        "text": "孟尝君 任齐 相时 ， 他 的 舍人 [MASK] 为 他 收取 封邑 的 租税 ， 往返 三次 而 没有 交 来 一笔 收入 。",
        "gold": "魏子",
        "competitor": "蔺相如",
        "candidates": ["魏子", "蔺相如", "孙膑", "范增", "郑国"],
    },
    {
        "name": "皇帝 vs 楚怀王 (高祖本纪/弑君尊号)",
        "unit": "unit1_chuhan_founding",
        "text": "汉高祖 既定 天下 ， 诸侯 尊 [MASK] 为 皇帝 。",
        "gold": "皇帝",
        "competitor": "怀王",
        "candidates": ["皇帝", "怀王", "沛公", "汉王", "项羽"],
    }
]


def run_probe(model, tokenizer, probe: dict, max_length: int = 256):
    tokens = probe["text"].split()
    inputs = visible_inputs(model, tokenizer, tokens, max_length=max_length)
    mask_idx = tokens.index("[MASK]") + 1
    scores = model(**inputs)[0][0, mask_idx]
    
    probs = scores.softmax(dim=-1)
    gold_tok = probe["gold"]
    comp_tok = probe["competitor"]
    
    gold_id = tokenizer.token_to_id.get(gold_tok)
    comp_id = tokenizer.token_to_id.get(comp_tok)
    
    gold_prob = float(probs[gold_id]) if gold_id is not None else 0.0
    comp_prob = float(probs[comp_id]) if comp_id is not None else 0.0
    
    cand_probs = []
    for c in probe["candidates"]:
        cid = tokenizer.token_to_id.get(c)
        p = float(probs[cid]) if cid is not None else 0.0
        cand_probs.append((c, p))
    cand_probs.sort(key=lambda x: x[1], reverse=True)
    
    top5 = probs.topk(min(5, len(tokenizer)))
    top5_tokens = [(tokenizer.id_to_token[int(i)], float(p)) for i, p in zip(top5.indices, top5.values)]
    
    return {
        "gold": gold_tok,
        "gold_prob": gold_prob,
        "competitor": comp_tok,
        "comp_prob": comp_prob,
        "gold_wins": gold_prob > comp_prob,
        "candidate_ranking": cand_probs,
        "top5_predictions": top5_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to trained model directory")
    parser.add_argument("--unit", default="all", help="Unit ID to evaluate, or 'all'")
    parser.add_argument("--device", default=None, help="cpu or cuda")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config_file = checkpoint / "relation_pair_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Cannot find relation_pair_config.json in {checkpoint}")
    settings = json.loads(config_file.read_text(encoding="utf-8"))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint from: {checkpoint} (device={device})")

    model = DynamicQKLocalRelationMarginBertForMaskedLM.from_pretrained(str(checkpoint))
    tokenizer = SimpleBertTokenizer.from_pretrained(str(checkpoint))
    model.to(device)
    model.eval()

    max_length = settings.get("max_length", 256)
    min_tokens = settings.get("min_tokens", 12)
    entity_masking = settings.get("entity_masking", True)
    pointer_extraction = settings.get("pointer_extraction", True)

    print("\n" + "=" * 70)
    print(" 史记单元化评测成绩单 (Shiji Unit Scorecard) ")
    print("=" * 70)

    scorecard = {}
    for uid, name, path in UNITS:
        if args.unit != "all" and args.unit != uid:
            continue
        if not path.exists():
            print(f"[-] 跳过 {uid} (文件不存在: {path})")
            continue

        groups = list(load_groups(path))
        metrics = evaluate(
            model,
            tokenizer,
            groups,
            max_length=max_length,
            min_tokens=min_tokens,
            entity_masking=entity_masking,
            pointer_extraction=pointer_extraction,
        )
        scorecard[uid] = {"name": name, "metrics": metrics}

        print(f"\n【{name}】 ({uid}) - 共 {len(groups)} 组:")
        for split in ("train", "dev", "test"):
            m = metrics.get(split, {})
            acc = m.get("accuracy", 0.0) * 100
            cand_acc = m.get("candidate_accuracy", 0.0) * 100
            neg_def = m.get("negatives_defeated_rate", 0.0) * 100
            inv = m.get("invariance_agreement", 0.0) * 100
            print(
                f"  [{split:5s}] 准确率: {acc:5.1f}% | 候选集准确率: {cand_acc:5.1f}% | "
                f"负例击败率: {neg_def:5.1f}% | 不变量一致性: {inv:5.1f}%"
            )

    print("\n" + "=" * 70)
    print(" 核心实体偏置与探针测试 (Diagnostic Probes) ")
    print("=" * 70)

    for probe in PROBES:
        res = run_probe(model, tokenizer, probe, max_length=max_length)
        win_str = "SUCCESS (正确)" if res["gold_wins"] else "FAILED (偏置吸附)"
        print(f"\n探针: {probe['name']}")
        print(f"  测试输入: {probe['text']}")
        print(f"  正确答案 [{res['gold']}] 概率: {res['gold_prob']:.4f} | 竞争高频词 [{res['competitor']}] 概率: {res['comp_prob']:.4f}")
        print(f"  判定结果: {win_str}")
        print("  候选词排序:")
        for c, p in res["candidate_ranking"]:
            print(f"    - {c:6s}: {p:.4f}")
        print("  Top-5 预测词:")
        for tok, p in res["top5_predictions"]:
            print(f"    - {tok:6s}: {p:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
