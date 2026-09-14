"""Train structured Shiji facts with grammar/attribute-aware MLM logits.

This is the next experiment after ``train_shiji_fact_memory_dataset.py``.  The
fixed grammar automaton and frozen word-attribute table add a soft bias at
actual ``[MASK]`` positions; the BERT backbone and its relation-specific 3D
gradient replay still do the learning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as base
import train_shiji_fact_memory_dataset_impl as dataset_impl
from bert_mlm_fact_memory import _run_update
from bert_simple.grammar_attribute_filter import GrammarAttributeFilter
from bert_simple.grammar_attribute_model import GrammarAttributeBertForMaskedLM
from bert_simple.grammar_automaton import ATTRIBUTES


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FACT_DATASET = os.path.join(
    ROOT, "data", "shiji", "manifests", "fact_memory_dataset.json"
)
DEFAULT_ATTRIBUTE_DATASET = os.path.join(
    ROOT, "data", "shiji", "manifests", "word_attribute_dataset.json"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    ROOT, "outputs", "bert-mlm-fact-memory-shiji-grammar-attribute-256"
)

ATTRIBUTE_ALIASES = {"GROUP": "ORG"}


def _eval_loss(model: GrammarAttributeBertForMaskedLM, sample: Dict[str, object]) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        value = float(base._loss_for_sample(model, sample).item())
    if was_training:
        model.train()
    return value


@torch.no_grad()
def _fact_accuracy(
    model: GrammarAttributeBertForMaskedLM,
    samples: Sequence[Dict[str, object]],
    apply_grammar_attributes: bool = True,
) -> Dict[str, float]:
    if not samples:
        return {"samples": 0.0, "top1": 0.0, "top5": 0.0, "loss": 0.0}
    was_training = model.training
    model.eval()
    top1 = 0
    top5 = 0
    losses: List[float] = []
    for sample in samples:
        loss, logits, _ = model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            labels=sample["labels"],
            apply_grammar_attributes=apply_grammar_attributes,
        )
        position = int(sample["fact_position"])
        target = int(sample["fact_token_id"])
        scores = logits[0, position]
        order = torch.argsort(scores, descending=True)
        rank = int((order == target).nonzero(as_tuple=False)[0].item()) + 1
        top1 += int(rank <= 1)
        top5 += int(rank <= 5)
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    count = len(samples)
    return {
        "samples": float(count),
        "top1": top1 / count,
        "top5": top5 / count,
        "loss": sum(losses) / count,
    }


def _add_labels(
    labels: Dict[str, Set[str]], token: object, attributes: Iterable[object]
) -> None:
    parts = str(token).strip().split()
    if len(parts) != 1 or not parts[0]:
        return
    normalized = set()
    for attribute in attributes:
        name = ATTRIBUTE_ALIASES.get(str(attribute), str(attribute))
        if name in ATTRIBUTES:
            normalized.add(name)
    if normalized:
        labels[parts[0]].update(normalized)


def _build_attribute_filter(
    tokenizer: base.SimpleBertTokenizer,
    train_rows: Sequence[Dict[str, object]],
    attribute_dataset: str,
) -> tuple[GrammarAttributeFilter, int]:
    labels: Dict[str, Set[str]] = defaultdict(set)
    resolved_attribute_data = base._resolve_path(attribute_dataset)
    if os.path.isfile(resolved_attribute_data):
        with open(resolved_attribute_data, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for token, attributes in data.get("word_labels", {}).items():
            _add_labels(labels, token, attributes)

    # Fact labels from the training split add properties for the larger
    # structured vocabulary without leaking held-out answer labels.
    for row in train_rows:
        _add_labels(labels, row.get("subject", ""), row.get("subject_attributes", ()))
        _add_labels(labels, row.get("object", ""), row.get("object_attributes", ()))
        _add_labels(labels, row.get("answer", ""), row.get("answer_attributes", ()))

    attribute_filter = GrammarAttributeFilter(tokenizer)
    known_ids: List[int] = []
    with torch.no_grad():
        logits = attribute_filter.word_attributes.word_attribute_logits
        logits.zero_()
        for token, names in sorted(labels.items()):
            token_id = tokenizer.token_to_id.get(token)
            if token_id is None:
                continue
            logits[token_id].fill_(-4.0)
            for name in names:
                logits[token_id, ATTRIBUTES.index(name)] = 4.0
            known_ids.append(int(token_id))
    attribute_filter.mark_known(known_ids)
    return attribute_filter, len(set(known_ids))


def _ensure_fact_positions(samples: Sequence[Dict[str, object]]) -> None:
    for sample in samples:
        positions = sample["mask_positions"].nonzero(as_tuple=False).flatten()
        if positions.numel() != 1:
            raise ValueError("every fact sample must contain exactly one mask")
        sample["fact_position"] = int(positions[0].item())


def _save_model_metadata(
    output_dir: str,
    attribute_dataset: str,
    attribute_bias_scale: float,
    known_attribute_tokens: int,
) -> None:
    metadata = {
        "model_class": "GrammarAttributeBertForMaskedLM",
        "attribute_dataset": base._resolve_path(attribute_dataset),
        "attribute_bias_scale": attribute_bias_scale,
        "attribute_table_frozen_during_mlm": True,
        "known_attribute_tokens": known_attribute_tokens,
        "bias_is_soft": True,
        "bias_applies_only_to_actual_mask_tokens": True,
    }
    with open(
        os.path.join(output_dir, "grammar_attribute_model_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def train(
    fact_dataset: str = DEFAULT_FACT_DATASET,
    attribute_dataset: str = DEFAULT_ATTRIBUTE_DATASET,
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
    attribute_bias_scale: float = 1.5,
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
    rows = dataset_impl._load_manifest(resolved_dataset)
    tokenizer = dataset_impl._build_tokenizer(rows)
    train_rows = [row for row in rows if str(row["split"]) == "train"]
    dev_rows = [row for row in rows if str(row["split"]) == "dev"]
    test_rows = [row for row in rows if str(row["split"]) == "test"]
    general_texts = dataset_impl._unique_preserve(
        str(row["context"]) for row in train_rows
    )
    frequencies = Counter()
    for text in general_texts:
        frequencies.update(
            int(tokenizer.token_to_id[token])
            for token in tokenizer.tokenize(text)
            if token in tokenizer.token_to_id
        )

    attribute_filter, known_attribute_tokens = _build_attribute_filter(
        tokenizer, train_rows, attribute_dataset
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
    model = GrammarAttributeBertForMaskedLM(
        config,
        tokenizer,
        attribute_filter=attribute_filter,
        attribute_bias_scale=attribute_bias_scale,
    ).to(torch.device("cpu"))
    # The main MLM is allowed to change BERT, but not the lexical attribute
    # table.  This keeps the grammar/attribute ground stable for later use by
    # a generative decoder.
    for parameter in model.grammar_attribute_filter.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    layout = base.FeatureIndexLayout(parameters, hidden_size)
    allocator = base.DimensionCombinationAllocator(hidden_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    probes, encodings = dataset_impl._make_stable_examples(
        general_texts,
        tokenizer,
        frequencies,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
    )
    fact_train = dataset_impl._build_fact_samples(
        rows,
        "train",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=True,
    )
    fact_dev = dataset_impl._build_fact_samples(
        rows,
        "dev",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=False,
    )
    fact_test = dataset_impl._build_fact_samples(
        rows,
        "test",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=False,
    )
    _ensure_fact_positions(fact_train + fact_dev + fact_test)
    if not fact_train or not fact_dev or not fact_test:
        raise RuntimeError(
            f"Could not encode all splits: train={len(fact_train)}, "
            f"dev={len(fact_dev)}, test={len(fact_test)}"
        )

    output_path = base._resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    memory: List[Dict[str, object]] = []
    global_step = 0
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
                sample = dataset_impl._make_epoch_sample(
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
                            "reference_loss": _eval_loss(model, probe),
                            "relation_triples": probe["relation_triples"],
                        }
                    )
                total_soft_conflicts += int(stats["soft_conflict_triples"])
                total_checked += int(stats["checked_relation_triples"])
                if stats["replay_overlap"] > 0:
                    total_overlap_replays += 1
                    total_replay_overlap += int(stats["replay_overlap"])
                if global_step % log_every == 0 or index + 1 == len(encodings):
                    memory_loss, forgetting = base._evaluate_memory(model, memory)
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
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_handle.flush()
                    print(
                        f"general {epoch + 1}/{general_epochs}: "
                        f"step={index + 1}/{len(encodings)}, "
                        f"loss={stats['new_loss']:.4f}, "
                        f"memory_loss={memory_loss:.4f}, "
                        f"forgetting={forgetting:.6f}"
                    )

        general_memory_loss, general_forgetting = base._evaluate_memory(model, memory)
        fact_test_before_raw = _fact_accuracy(model, fact_test, False)
        fact_test_before = _fact_accuracy(model, fact_test, True)
        model.save_pretrained(os.path.join(output_path, "before_fact"))
        tokenizer.save_pretrained(os.path.join(output_path, "before_fact"))

        for group in optimizer.param_groups:
            group["lr"] = fact_learning_rate
        total_fact_steps = fact_epochs * fact_repeats * len(fact_train)
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
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_handle.flush()

        final_memory_loss, final_forgetting = base._evaluate_memory(model, memory)
        fact_dev_after_raw = _fact_accuracy(model, fact_dev, False)
        fact_dev_after = _fact_accuracy(model, fact_dev, True)
        fact_test_after_raw = _fact_accuracy(model, fact_test, False)
        fact_test_after = _fact_accuracy(model, fact_test, True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    dataset_impl._save_fact_manifest(
        output_path, fact_train + fact_dev + fact_test
    )
    # The helper writes a map based on the same allocator used by both stages.
    from bert_mlm_dimension_combinations_epochs import _save_relation_map

    _save_relation_map(output_path, tokenizer, allocator.relation_triples)
    _save_model_metadata(
        output_path,
        attribute_dataset,
        attribute_bias_scale,
        known_attribute_tokens,
    )
    used_dimensions = set()
    for triple in allocator.relation_triples.values():
        used_dimensions.update(triple)
    summary = {
        "stage": "bert_mlm_structured_shiji_fact_memory_with_grammar_attributes",
        "fact_dataset": resolved_dataset,
        "attribute_dataset": base._resolve_path(attribute_dataset),
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
        "replay_weight": replay_weight,
        "background_weight": background_weight,
        "soft_conflict_threshold": soft_conflict_threshold,
        "soft_conflict_scale": soft_conflict_scale,
        "mlm_probability": mlm_probability,
        "grammar_attribute_bias_enabled": True,
        "grammar_attribute_bias_scale": attribute_bias_scale,
        "attribute_table_frozen_during_mlm": True,
        "known_attribute_tokens": known_attribute_tokens,
        "total_soft_conflict_triples": total_soft_conflicts,
        "total_checked_relation_triples": total_checked,
        "overlap_replay_steps": total_overlap_replays,
        "total_replay_overlap": total_replay_overlap,
        "fact_test_before_raw": fact_test_before_raw,
        "fact_test_before_gated": fact_test_before,
        "fact_dev_after_raw": fact_dev_after_raw,
        "fact_dev_after_gated": fact_dev_after,
        "fact_test_after_raw": fact_test_after_raw,
        "fact_test_after_gated": fact_test_after,
        "general_memory_loss_before_fact": general_memory_loss,
        "general_forgetting_before_fact": general_forgetting,
        "final_memory_loss": final_memory_loss,
        "final_forgetting": final_forgetting,
    }
    with open(
        os.path.join(output_path, "grammar_attribute_fact_memory_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved grammar/attribute fact-memory outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train structured Shiji facts with grammar/attribute logit bias."
    )
    parser.add_argument("fact_dataset", nargs="?", default=DEFAULT_FACT_DATASET)
    parser.add_argument("--attribute-dataset", default=DEFAULT_ATTRIBUTE_DATASET)
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
    parser.add_argument("--attribute-bias-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    values = vars(args)
    values["num_hidden_layers"] = values.pop("layers")
    values["num_attention_heads"] = values.pop("heads")
    train(**values)
