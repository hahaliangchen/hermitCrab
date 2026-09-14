"""Train the learned word-attribute layer beside the fixed grammar automaton.

Grammar is not optimized here: the automaton is a fixed, inspectable rule
system.  The trainable part learns multi-label properties for vocabulary words
from dictionary labels and masked-slot observations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bert_simple.grammar_attribute_filter import GrammarAttributeFilter
from bert_simple.grammar_automaton import ATTRIBUTES
from bert_simple.tokenizer import SimpleBertTokenizer


DEFAULT_TOKENIZER = ROOT / "outputs/bert-mlm-fact-memory-zhangchen-gaozu-20x-256"
DEFAULT_DATA = ROOT / "data/shiji/manifests/word_attribute_dataset.json"
DEFAULT_OUTPUT = ROOT / "outputs/grammar-attribute-filter"


def load_tokenizer(path: Path, data: Dict[str, object]) -> SimpleBertTokenizer:
    if (path / "vocab.json").exists():
        return SimpleBertTokenizer.from_pretrained(str(path))
    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(
        [example["text"] for example in data.get("slot_examples", [])]
        + list(data.get("word_labels", {}).keys())
    )
    return tokenizer


def make_targets(
    tokenizer: SimpleBertTokenizer,
    word_labels: Dict[str, Sequence[str]],
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    targets = torch.zeros(len(tokenizer), len(ATTRIBUTES), dtype=torch.float32)
    valid = torch.zeros_like(targets, dtype=torch.bool)
    ids: List[int] = []
    for token, names in word_labels.items():
        token_id = tokenizer.token_to_id.get(token)
        if token_id is None:
            continue
        ids.append(token_id)
        valid[token_id] = True
        for name in names:
            if name in ATTRIBUTES:
                targets[token_id, ATTRIBUTES.index(name)] = 1.0
    return targets, valid, sorted(set(ids))


def multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ids: Sequence[int],
) -> Dict[str, object]:
    if not ids:
        return {"words": 0, "micro_f1": 0.0, "exact_match": 0.0, "per_attribute": {}}
    rows = logits[list(ids)].sigmoid() >= 0.5
    gold = targets[list(ids)].bool()
    tp = (rows & gold).sum().item()
    fp = (rows & ~gold).sum().item()
    fn = (~rows & gold).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    exact = rows.eq(gold).all(-1).float().mean().item()
    per_attribute: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(ATTRIBUTES):
        pred = rows[:, index]
        truth = gold[:, index]
        atp = (pred & truth).sum().item()
        afp = (pred & ~truth).sum().item()
        afn = (~pred & truth).sum().item()
        ap = atp / max(atp + afp, 1)
        ar = atp / max(atp + afn, 1)
        per_attribute[name] = {
            "precision": ap,
            "recall": ar,
            "f1": 2 * ap * ar / max(ap + ar, 1e-12),
            "support": float(truth.sum().item()),
        }
    return {
        "words": len(ids),
        "micro_f1": f1,
        "exact_match": exact,
        "per_attribute": per_attribute,
    }


def grammar_metrics(
    filter_model: GrammarAttributeFilter,
    slots: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    matched = 0
    non_fallback = 0
    records = []
    for example in slots:
        state = filter_model.analyze(str(example["text"]))
        expected = set(example["attributes"])
        allowed = set(state.allowed_attributes)
        hit = bool(expected & allowed)
        matched += int(hit)
        non_fallback += int(not state.fallback)
        if len(records) < 12:
            records.append(
                {
                    "text": example["text"],
                    "target_token": example["target_token"],
                    "expected": sorted(expected),
                    "allowed": list(state.allowed_attributes),
                    "structure": state.structure,
                    "rule": state.matches[0].rule if state.matches else None,
                    "matched": hit,
                }
            )
    total = max(len(slots), 1)
    return {
        "slot_examples": len(slots),
        "attribute_match_rate": matched / total,
        "non_fallback_rate": non_fallback / total,
        "examples": records,
    }


def train(
    data_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    epochs: int,
    learning_rate: float,
    attribute_dim: int,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(2)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(tokenizer_path, data)
    model = GrammarAttributeFilter(tokenizer, attribute_dim).to(torch.device("cpu"))
    targets, valid, ids = make_targets(tokenizer, data["word_labels"])
    model.word_attributes.mark_known(ids)
    id_tensor = torch.tensor(ids, dtype=torch.long)
    target_rows = targets[id_tensor]
    valid_rows = valid[id_tensor]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logs: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        logits = model(id_tensor)
        raw_loss = F.binary_cross_entropy_with_logits(
            logits, target_rows, reduction="none"
        )
        loss = raw_loss.masked_select(valid_rows).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        record = {"epoch": epoch + 1, "loss": float(loss.detach().item())}
        logs.append(record)
        if epoch == 0 or (epoch + 1) % max(epochs // 10, 1) == 0 or epoch + 1 == epochs:
            print(json.dumps(record, ensure_ascii=False), flush=True)

    model.eval()
    with torch.no_grad():
        final_logits = model(id_tensor)
    metrics = multilabel_metrics(final_logits, target_rows, list(range(len(ids))))
    slot_report = grammar_metrics(model, data.get("slot_examples", []))
    probes = {}
    for token in ("魏国", "刘邦", "张耳", "大梁", "沛县", "将军", "年"):
        token_id = tokenizer.token_to_id.get(token)
        if token_id is None:
            continue
        probabilities = model.word_attributes.probabilities(
            torch.tensor([token_id], dtype=torch.long)
        )[0]
        probes[token] = {
            name: float(probabilities[index].detach().item())
            for index, name in enumerate(ATTRIBUTES)
            if float(probabilities[index].detach().item()) >= 0.5
        }
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    (output_path / "training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "stage": "fixed_grammar_automaton_plus_learned_word_attributes",
        "data": str(data_path),
        "tokenizer": str(tokenizer_path),
        "vocab_size": len(tokenizer),
        "attribute_dim": len(ATTRIBUTES),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "labeled_word_types": len(ids),
        "word_attribute_metrics": metrics,
        "grammar_metrics": slot_report,
        "probes": probes,
        "unknown_words_are_neutral": True,
    }
    (output_path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"grammar_metrics", "word_attribute_metrics"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--attribute-dim", type=int, default=len(ATTRIBUTES))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train(
        Path(args.data),
        Path(args.tokenizer),
        Path(args.output),
        args.epochs,
        args.learning_rate,
        args.attribute_dim,
        args.seed,
    )


if __name__ == "__main__":
    main()
