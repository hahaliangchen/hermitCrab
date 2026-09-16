"""[历史对照] Train word-level BERT MLM from the structured Shiji fact manifest.

This module uses the retired token-pair allocator to reproduce earlier
fact-memory runs.  The active relation design uses the fixed shared context
bank and complete context groups in ``train_full_relation_filter_stages.py``.

The manifest is deliberately kept separate from the model.  This entry point
turns each exact ``masked_text -> answer`` record into an MLM sample, uses the
source contexts for broad language learning, and then runs a targeted fact
memory stage.  The existing relation-specific 3D gradient routing and soft
replay guard are reused; grammar and word attributes remain metadata for the
next model-integration step rather than silently pretending to affect BERT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as base
from bert_mlm_dimension_combinations_epochs import (
    _make_epoch_sample,
    _save_relation_map,
    _make_stable_examples,
)
from bert_mlm_fact_memory import _fact_accuracy, _run_update


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FACT_DATASET = os.path.join(
    ROOT, "data", "shiji", "manifests", "fact_memory_dataset.json"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    ROOT, "outputs", "bert-mlm-fact-memory-shiji-structured-256"
)

Triple = Tuple[int, int, int]


def _unique_preserve(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _load_manifest(path: str) -> List[Dict[str, object]]:
    resolved = base._resolve_path(path)
    with open(resolved, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Fact manifest must be a non-empty JSON list: {resolved}")
    required = {"fact_id", "context", "masked_text", "answer", "split"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Fact row is missing {sorted(missing)}: {row}")
    return rows


def _build_tokenizer(rows: Sequence[Dict[str, object]]) -> base.SimpleBertTokenizer:
    # Build the vocabulary before the split so held-out facts are not turned
    # into [UNK].  The vectors are still randomly initialized at this point;
    # only the token inventory is shared across train/dev/test.
    texts: List[str] = []
    for row in rows:
        texts.append(str(row["context"]))
        texts.append(str(row.get("long_context", "")))
        texts.append(str(row["masked_text"]))
        for variant in row.get("variants", []) or []:
            texts.append(str(variant.get("masked_text", "")))
    tokenizer = base.SimpleBertTokenizer()
    tokenizer.train_from_texts(_unique_preserve(texts), min_freq=1)
    return tokenizer


def _encode_fact_sample(
    text: str,
    answer: str,
    fact_id: str,
    split: str,
    tokenizer: base.SimpleBertTokenizer,
    max_length: int,
    variant_index: int = 0,
) -> Optional[Dict[str, object]]:
    encoded = tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_special_tokens_mask=True,
    )
    mask_token_id = tokenizer.mask_token_id
    positions = [
        index for index, token_id in enumerate(encoded["input_ids"])
        if int(token_id) == mask_token_id
    ]
    answer_id = tokenizer.token_to_id.get(answer)
    if len(positions) != 1 or answer_id is None:
        return None

    position = positions[0]
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    original_input_ids = input_ids.clone()
    original_input_ids[position] = int(answer_id)
    labels = torch.full_like(input_ids, -100)
    labels[position] = int(answer_id)
    mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
    mask_positions[position] = True
    return {
        "input_ids": input_ids.unsqueeze(0),
        "token_type_ids": torch.tensor(
            encoded["token_type_ids"], dtype=torch.long
        ).unsqueeze(0),
        "attention_mask": torch.tensor(
            encoded["attention_mask"], dtype=torch.long
        ).unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "original_input_ids": original_input_ids,
        "mask_positions": mask_positions,
        "fact_token": answer,
        "fact_token_id": int(answer_id),
        "fact_id": fact_id,
        "split": split,
        "variant_index": variant_index,
        "masked_text": text,
    }


def _fact_relation_keys(
    row: Dict[str, object],
    sample: Dict[str, object],
    tokenizer: base.SimpleBertTokenizer,
    max_context_tokens: int,
) -> List[base.RelationKey]:
    """Prioritize semantic subject/object anchors, then local context words."""
    target_id = int(sample["fact_token_id"])
    original = sample["original_input_ids"].tolist()
    mask_positions = sample["mask_positions"].tolist()
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    keys: List[base.RelationKey] = []

    anchor_tokens: List[str] = []
    if str(row.get("target_role", "object")) == "subject":
        anchor_tokens.append(str(row.get("object", "")))
    else:
        anchor_tokens.append(str(row.get("subject", "")))
        anchor_tokens.append(str(row.get("object", "")))
    anchor_tokens.append(str(row.get("relation_surface", "")))

    def add_token(token: str) -> None:
        token = token.strip()
        if not token or " " in token:
            return
        token_id = tokenizer.token_to_id.get(token)
        if token_id is None or token_id in special_ids or token_id == target_id:
            return
        key = tuple(sorted((target_id, int(token_id))))
        if key[0] != key[1] and key not in keys:
            keys.append(key)

    for token in anchor_tokens:
        add_token(token)

    local_ids = []
    for position, token_id in enumerate(original):
        token_id = int(token_id)
        if mask_positions[position] or token_id in special_ids or token_id == target_id:
            continue
        if token_id not in local_ids:
            local_ids.append(token_id)
    for token_id in local_ids:
        if len(keys) >= max_context_tokens:
            break
        add_token(tokenizer.id_to_token[token_id])
    return keys


def _attach_relations(
    sample: Dict[str, object],
    row: Dict[str, object],
    tokenizer: base.SimpleBertTokenizer,
    allocator: base.DimensionCombinationAllocator,
    max_context_tokens: int,
) -> Dict[str, object]:
    relation_keys = _fact_relation_keys(row, sample, tokenizer, max_context_tokens)
    sample["relation_keys"] = relation_keys
    sample["relation_triples"] = [allocator.allocate(key) for key in relation_keys]
    return sample


def _build_fact_samples(
    rows: Sequence[Dict[str, object]],
    split: str,
    tokenizer: base.SimpleBertTokenizer,
    allocator: base.DimensionCombinationAllocator,
    max_length: int,
    max_context_tokens: int,
    include_variants: bool,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for row in rows:
        if str(row["split"]) != split:
            continue
        fact_id = str(row["fact_id"])
        answer = str(row["answer"])
        base_sample = _encode_fact_sample(
            str(row["masked_text"]),
            answer,
            fact_id,
            split,
            tokenizer,
            max_length,
        )
        if base_sample is not None:
            samples.append(
                _attach_relations(
                    base_sample,
                    row,
                    tokenizer,
                    allocator,
                    max_context_tokens,
                )
            )
        if not include_variants:
            continue
        for variant_index, variant in enumerate(row.get("variants", []) or [], start=1):
            variant_sample = _encode_fact_sample(
                str(variant["masked_text"]),
                answer,
                fact_id,
                split,
                tokenizer,
                max_length,
                variant_index=variant_index,
            )
            if variant_sample is not None:
                samples.append(
                    _attach_relations(
                        variant_sample,
                        row,
                        tokenizer,
                        allocator,
                        max_context_tokens,
                    )
                )
    return samples


def _save_fact_manifest(
    output_dir: str, samples: Sequence[Dict[str, object]]
) -> None:
    manifest = []
    for sample in samples:
        manifest.append(
            {
                "fact_id": sample["fact_id"],
                "split": sample["split"],
                "variant_index": int(sample["variant_index"]),
                "fact_token": sample["fact_token"],
                "fact_token_id": int(sample["fact_token_id"]),
                "masked_text": sample["masked_text"],
                "relation_triples": [
                    list(triple) for triple in sample["relation_triples"]
                ],
            }
        )
    with open(
        os.path.join(output_dir, "fact_training_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _evaluate_memory_and_restore(
    model: base.BertForMaskedLM,
    memory: Sequence[Dict[str, object]],
) -> Tuple[float, float]:
    value = base._evaluate_memory(model, memory)
    return value


def train(
    fact_dataset: str = DEFAULT_FACT_DATASET,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_length: int = 128,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: Optional[int] = None,
    general_epochs: int = 3,
    fact_epochs: int = 1,
    fact_repeats: int = 2,
    learning_rate: float = 2e-4,
    fact_learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    replay_weight: float = 0.5,
    background_weight: float = 0.35,
    max_context_tokens: int = 8,
    soft_conflict_threshold: float = -0.35,
    soft_conflict_scale: float = 0.85,
    mlm_probability: float = 0.15,
    seed: int = 42,
    log_every: int = 100,
) -> str:
    if hidden_size < 3 or hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be >= 3 and divisible by num_attention_heads")
    if general_epochs < 1 or fact_epochs < 1 or fact_repeats < 1:
        raise ValueError("all epoch/repeat values must be positive")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    base.set_seed(seed)
    resolved_dataset = base._resolve_path(fact_dataset)
    rows = _load_manifest(resolved_dataset)
    tokenizer = _build_tokenizer(rows)
    train_rows = [row for row in rows if str(row["split"]) == "train"]
    dev_rows = [row for row in rows if str(row["split"]) == "dev"]
    test_rows = [row for row in rows if str(row["split"]) == "test"]
    general_texts = _unique_preserve(str(row["context"]) for row in train_rows)
    if not general_texts:
        raise ValueError("The fact manifest has no train contexts")

    frequencies = Counter()
    for text in general_texts:
        frequencies.update(
            int(tokenizer.token_to_id[token])
            for token in tokenizer.tokenize(text)
            if token in tokenizer.token_to_id
        )
    print(f"Using fact dataset: {resolved_dataset}")
    print(
        f"Rows: train={len(train_rows)}, dev={len(dev_rows)}, test={len(test_rows)}; "
        f"general_contexts={len(general_texts)}; vocab={len(tokenizer)}"
    )

    config = base.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_length + 10,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = base.BertForMaskedLM(config).to(torch.device("cpu"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    layout = base.FeatureIndexLayout(parameters, hidden_size)
    allocator = base.DimensionCombinationAllocator(hidden_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    probes, encodings = _make_stable_examples(
        general_texts,
        tokenizer,
        frequencies,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
    )
    fact_train = _build_fact_samples(
        rows,
        "train",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=True,
    )
    fact_dev = _build_fact_samples(
        rows,
        "dev",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=False,
    )
    fact_test = _build_fact_samples(
        rows,
        "test",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=False,
    )
    if not fact_train or not fact_dev or not fact_test:
        raise RuntimeError(
            f"Could not encode all splits: train={len(fact_train)}, "
            f"dev={len(fact_dev)}, test={len(fact_test)}"
        )
    print(
        f"Fact samples: train={len(fact_train)} (including long variants), "
        f"dev={len(fact_dev)}, test={len(fact_test)}"
    )

    output_path = base._resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    memory: List[Dict[str, object]] = []
    global_step = 0
    logs: List[Dict[str, object]] = []
    total_soft_conflicts = 0
    total_checked = 0
    total_overlap_replays = 0
    total_replay_overlap = 0
    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        model.train()
        for epoch in range(general_epochs):
            epoch_seed = seed + epoch * 1_000_003
            for index, encoded in enumerate(encodings):
                probe = probes[index]
                sample = _make_epoch_sample(
                    encoded,
                    tokenizer,
                    probe["relation_keys"],
                    probe["relation_triples"],
                    seed=epoch_seed + index * 1009,
                    mlm_probability=mlm_probability,
                )
                global_step += 1
                stats = _run_update(
                    model,
                    parameters,
                    layout,
                    hidden_size,
                    optimizer,
                    sample,
                    memory,
                    global_step,
                    replay_weight,
                    background_weight,
                    soft_conflict_threshold,
                    soft_conflict_scale,
                )
                if epoch == 0:
                    memory.append(
                        {
                            "sample": probe,
                            "reference_loss": base._evaluate_memory(model, [{
                                "sample": probe,
                                "reference_loss": 0.0,
                            }])[0],
                            "relation_triples": probe["relation_triples"],
                        }
                    )
                total_soft_conflicts += int(stats["soft_conflict_triples"])
                total_checked += int(stats["checked_relation_triples"])
                if stats["replay_overlap"] > 0:
                    total_overlap_replays += 1
                    total_replay_overlap += int(stats["replay_overlap"])
                if (
                    global_step % log_every == 0
                    or index + 1 == len(encodings)
                ):
                    memory_loss, forgetting = _evaluate_memory_and_restore(model, memory)
                    record = {
                        "stage": "general",
                        "epoch": epoch + 1,
                        "step_in_epoch": index + 1,
                        "global_step": global_step,
                        **stats,
                        "memory_size": len(memory),
                        "memory_loss": memory_loss,
                        "forgetting": forgetting,
                    }
                    logs.append(record)
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_handle.flush()
                    print(
                        f"general {epoch + 1}/{general_epochs}: "
                        f"step={index + 1}/{len(encodings)}, "
                        f"loss={stats['new_loss']:.4f}, "
                        f"memory_loss={memory_loss:.4f}, "
                        f"forgetting={forgetting:.6f}"
                    )

        general_memory_loss, general_forgetting = _evaluate_memory_and_restore(
            model, memory
        )
        fact_before = _fact_accuracy(model, fact_test)
        model.save_pretrained(os.path.join(output_path, "before_fact"))
        tokenizer.save_pretrained(os.path.join(output_path, "before_fact"))

        for group in optimizer.param_groups:
            group["lr"] = fact_learning_rate
        for fact_epoch in range(fact_epochs):
            for repeat in range(fact_repeats):
                for fact_index, sample in enumerate(fact_train):
                    global_step += 1
                    stats = _run_update(
                        model,
                        parameters,
                        layout,
                        hidden_size,
                        optimizer,
                        sample,
                        memory,
                        global_step,
                        replay_weight,
                        background_weight,
                        soft_conflict_threshold,
                        soft_conflict_scale,
                    )
                    total_soft_conflicts += int(stats["soft_conflict_triples"])
                    total_checked += int(stats["checked_relation_triples"])
                    if stats["replay_overlap"] > 0:
                        total_overlap_replays += 1
                        total_replay_overlap += int(stats["replay_overlap"])
                    local_step = (
                        fact_epoch * fact_repeats * len(fact_train)
                        + repeat * len(fact_train)
                        + fact_index
                        + 1
                    )
                    total_fact_steps = fact_epochs * fact_repeats * len(fact_train)
                    if local_step % log_every == 0 or local_step == total_fact_steps:
                        print(
                            f"fact {fact_epoch + 1}/{fact_epochs}, "
                            f"repeat {repeat + 1}/{fact_repeats}: "
                            f"step={local_step}/{total_fact_steps}, "
                            f"token={sample['fact_token']}, "
                            f"loss={stats['new_loss']:.4f}"
                        )
                    record = {
                        "stage": "fact_memory",
                        "fact_epoch": fact_epoch + 1,
                        "fact_repeat": repeat + 1,
                        "fact_step": local_step,
                        "global_step": global_step,
                        "fact_id": sample["fact_id"],
                        "fact_token": sample["fact_token"],
                        **stats,
                    }
                    logs.append(record)
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_handle.flush()

        final_memory_loss, final_forgetting = _evaluate_memory_and_restore(model, memory)
        fact_after = _fact_accuracy(model, fact_test)
        fact_dev_after = _fact_accuracy(model, fact_dev)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    _save_relation_map(output_path, tokenizer, allocator.relation_triples)
    _save_fact_manifest(output_path, fact_train + fact_dev + fact_test)
    used_dimensions = set()
    for triple in allocator.relation_triples.values():
        used_dimensions.update(triple)
    summary = {
        "stage": "bert_mlm_structured_shiji_fact_memory",
        "fact_dataset": resolved_dataset,
        "device": "cpu",
        "general_contexts": len(general_texts),
        "general_epochs": general_epochs,
        "fact_epochs": fact_epochs,
        "fact_repeats": fact_repeats,
        "optimizer_steps": global_step,
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "train_fact_samples": len(fact_train),
        "dev_fact_samples": len(fact_dev),
        "test_fact_samples": len(fact_test),
        "relation_mapping_fixed": True,
        "combination_capacity": allocator.capacity,
        "allocated_relation_combinations": len(allocator.relation_triples),
        "used_hidden_dimensions": len(used_dimensions),
        "learning_rate": learning_rate,
        "fact_learning_rate": fact_learning_rate,
        "weight_decay": weight_decay,
        "replay_weight": replay_weight,
        "background_weight": background_weight,
        "soft_conflict_threshold": soft_conflict_threshold,
        "soft_conflict_scale": soft_conflict_scale,
        "mlm_probability": mlm_probability,
        "total_soft_conflict_triples": total_soft_conflicts,
        "total_checked_relation_triples": total_checked,
        "overlap_replay_steps": total_overlap_replays,
        "total_replay_overlap": total_replay_overlap,
        "fact_test_before": fact_before,
        "fact_dev_after": fact_dev_after,
        "fact_test_after": fact_after,
        "general_memory_loss_before_fact": general_memory_loss,
        "general_forgetting_before_fact": general_forgetting,
        "final_memory_loss": final_memory_loss,
        "final_forgetting": final_forgetting,
    }
    with open(
        os.path.join(output_path, "structured_fact_memory_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved structured fact-memory outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train word-level BERT MLM on the structured Shiji fact manifest."
    )
    parser.add_argument("fact_dataset", nargs="?", default=DEFAULT_FACT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--general-epochs", type=int, default=3)
    parser.add_argument("--fact-epochs", type=int, default=1)
    parser.add_argument("--fact-repeats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--fact-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.35)
    parser.add_argument("--max-context-tokens", type=int, default=8)
    parser.add_argument("--soft-conflict-threshold", type=float, default=-0.35)
    parser.add_argument("--soft-conflict-scale", type=float, default=0.85)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        fact_dataset=args.fact_dataset,
        output_dir=args.output_dir,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        general_epochs=args.general_epochs,
        fact_epochs=args.fact_epochs,
        fact_repeats=args.fact_repeats,
        learning_rate=args.learning_rate,
        fact_learning_rate=args.fact_learning_rate,
        weight_decay=args.weight_decay,
        replay_weight=args.replay_weight,
        background_weight=args.background_weight,
        max_context_tokens=args.max_context_tokens,
        soft_conflict_threshold=args.soft_conflict_threshold,
        soft_conflict_scale=args.soft_conflict_scale,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
