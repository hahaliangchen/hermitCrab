"""Train Shiji facts from the grammar checkpoint with a weak frequency fallback.

The grammar checkpoint is kept intact.  This experiment starts from its
``before_fact`` snapshot, trains only the fact-memory stage, and logs how much
of each answer score comes from the initial context model, the fact update,
the grammar/attribute bias, and the frequency prior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as base
import train_shiji_fact_memory_dataset_impl as dataset_impl
from bert_mlm_fact_memory import _run_update
from train_shiji_fact_memory_grammar_impl import _ensure_fact_positions

from bert_simple.grammar_attribute_filter import (
    SPECIAL_TOKENS,
    _surface_attribute,
)
from bert_simple.grammar_attribute_model import GrammarAttributeBertForMaskedLM
from bert_simple.prior_controlled_model import (
    PriorControlledGrammarAttributeBertForMaskedLM,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FACT_DATASET = os.path.join(
    ROOT, "data", "shiji", "manifests", "fact_memory_dataset.json"
)
DEFAULT_ATTRIBUTE_DATASET = os.path.join(
    ROOT, "data", "shiji", "manifests", "word_attribute_dataset.json"
)
DEFAULT_INIT_CHECKPOINT = os.path.join(
    ROOT,
    "outputs",
    "bert-mlm-fact-memory-shiji-grammar-attribute-256",
    "before_fact",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "bert-mlm-fact-memory-shiji-prior-controlled-256",
)


def _loss_for_sample(model, sample: Dict[str, object]) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        value = float(base._loss_for_sample(model, sample).item())
    if was_training:
        model.train()
    return value


@torch.no_grad()
def _fact_accuracy(
    model,
    samples: Sequence[Dict[str, object]],
    apply_grammar_attributes: bool,
    apply_frequency_prior: bool,
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
            apply_frequency_prior=apply_frequency_prior,
        )
        position = int(sample["fact_position"])
        target = int(sample["fact_token_id"])
        order = torch.argsort(logits[0, position], descending=True)
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


def _rank(scores: torch.Tensor, target: int) -> int:
    order = torch.argsort(scores, descending=True)
    positions = (order == target).nonzero(as_tuple=False)
    return int(positions[0].item()) + 1 if positions.numel() else -1


def _build_frequency_vectors(
    tokenizer,
    train_rows: Sequence[Dict[str, object]],
    content_scale: float,
    function_scale: float,
    punctuation_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, Counter[str]]:
    frequencies: Counter[str] = Counter()
    for row in train_rows:
        frequencies.update(tokenizer.tokenize(str(row.get("context", ""))))

    log_counts = torch.tensor(
        [math.log1p(frequencies.get(token, 0)) for token in tokenizer.id_to_token],
        dtype=torch.float32,
    )
    usable_ids = [
        index
        for index, token in enumerate(tokenizer.id_to_token)
        if token not in SPECIAL_TOKENS
    ]
    if usable_ids:
        usable = log_counts[usable_ids]
        mean = usable.mean()
        std = usable.std(unbiased=False).clamp_min(1e-6)
        prior = ((log_counts - mean) / std).clamp(-2.5, 2.5)
    else:
        prior = torch.zeros_like(log_counts)

    class_scales = torch.full_like(prior, float(content_scale))
    for index, token in enumerate(tokenizer.id_to_token):
        if token in SPECIAL_TOKENS:
            class_scales[index] = 0.0
        elif _surface_attribute(token) == "PUNCT":
            class_scales[index] = float(punctuation_scale)
        elif _surface_attribute(token) == "FUNCTION":
            class_scales[index] = float(function_scale)
    return prior, class_scales, frequencies


def _build_memory(
    general_texts: Sequence[str],
    tokenizer,
    frequencies: Counter[str],
    allocator,
    max_length: int,
    max_context_tokens: int,
    seed: int,
    mlm_probability: float,
    model,
) -> List[Dict[str, object]]:
    probes, _ = dataset_impl._make_stable_examples(
        general_texts,
        tokenizer,
        frequencies,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
    )
    memory: List[Dict[str, object]] = []
    for probe in probes:
        memory.append(
            {
                "sample": probe,
                "reference_loss": _loss_for_sample(model, probe),
                "relation_triples": probe["relation_triples"],
            }
        )
    return memory


@torch.no_grad()
def _contribution_report(
    model: PriorControlledGrammarAttributeBertForMaskedLM,
    initial_model: GrammarAttributeBertForMaskedLM,
    samples: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    initial_model.eval()
    records: List[Dict[str, object]] = []
    dominant = Counter[str]()
    tokenizer = model.grammar_attribute_filter.tokenizer

    for sample in samples:
        result = model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            return_component_info=True,
        )
        _, _, components = result
        initial_logits, _ = initial_model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            apply_grammar_attributes=False,
        )
        position = int(sample["fact_position"])
        target = int(sample["fact_token_id"])
        initial_scores = initial_logits[0, position].detach().cpu()
        raw_scores = components["raw_logits"][0, position].detach().cpu()
        grammar_scores = components["grammar_bias"][0, position].detach().cpu()
        frequency_scores = components["frequency_bias"][0, position].detach().cpu()
        total_scores = components["logits"][0, position].detach().cpu()
        fact_delta = raw_scores - initial_scores

        additive = {
            "fact_update": float(fact_delta[target]),
            "grammar_attribute": float(grammar_scores[target]),
            "frequency_prior": float(frequency_scores[target]),
        }
        source = max(additive, key=lambda name: abs(additive[name]))
        dominant[source] += 1

        top_ids = torch.topk(total_scores, min(5, total_scores.numel())).indices.tolist()
        if target not in top_ids:
            top_ids.append(target)
        candidates = []
        for candidate_id in top_ids:
            candidate_id = int(candidate_id)
            candidates.append(
                {
                    "token": tokenizer.id_to_token[candidate_id],
                    "context": float(initial_scores[candidate_id]),
                    "fact_update": float(fact_delta[candidate_id]),
                    "grammar_attribute": float(grammar_scores[candidate_id]),
                    "frequency_prior": float(frequency_scores[candidate_id]),
                    "total": float(total_scores[candidate_id]),
                }
            )

        records.append(
            {
                "fact_id": sample["fact_id"],
                "fact_token": sample["fact_token"],
                "position": position,
                "relation_triples": [
                    list(triple) for triple in sample["relation_triples"]
                ],
                "target_scores": additive,
                "dominant_additive_source": source,
                "ranks": {
                    "initial_context": _rank(initial_scores, target),
                    "after_fact_raw": _rank(raw_scores, target),
                    "after_grammar": _rank(raw_scores + grammar_scores, target),
                    "after_frequency_only": _rank(
                        raw_scores + frequency_scores, target
                    ),
                    "final": _rank(total_scores, target),
                },
                "candidates": candidates,
            }
        )

    if was_training:
        model.train()
    return {
        "samples": len(records),
        "dominant_additive_source_counts": dict(dominant),
        "records": records,
    }


def train(
    fact_dataset: str = DEFAULT_FACT_DATASET,
    init_checkpoint: str = DEFAULT_INIT_CHECKPOINT,
    attribute_dataset: str = DEFAULT_ATTRIBUTE_DATASET,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_length: int = 128,
    max_context_tokens: int = 8,
    fact_epochs: int = 1,
    fact_repeats: int = 2,
    fact_learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    replay_weight: float = 0.5,
    background_weight: float = 0.35,
    soft_conflict_threshold: float = -0.35,
    soft_conflict_scale: float = 0.85,
    mlm_probability: float = 0.15,
    attribute_bias_scale: float = 1.5,
    frequency_content_scale: float = 0.12,
    frequency_function_scale: float = 0.025,
    frequency_punctuation_scale: float = 0.01,
    constrained_frequency_gate: float = 0.25,
    seed: int = 42,
    log_every: int = 100,
) -> str:
    if fact_epochs < 1 or fact_repeats < 1:
        raise ValueError("fact_epochs and fact_repeats must be positive")
    base.set_seed(seed)
    resolved_dataset = base._resolve_path(fact_dataset)
    resolved_init = base._resolve_path(init_checkpoint)
    rows = dataset_impl._load_manifest(resolved_dataset)
    if not os.path.isfile(os.path.join(resolved_init, "pytorch_model.bin")):
        raise FileNotFoundError(f"init checkpoint not found: {resolved_init}")

    initial_model = GrammarAttributeBertForMaskedLM.from_pretrained(resolved_init)
    tokenizer = initial_model.grammar_attribute_filter.tokenizer
    expected_tokenizer = dataset_impl._build_tokenizer(rows)
    if expected_tokenizer.token_to_id != tokenizer.token_to_id:
        raise ValueError("init checkpoint tokenizer does not match fact dataset")

    train_rows = [row for row in rows if str(row["split"]) == "train"]
    dev_rows = [row for row in rows if str(row["split"]) == "dev"]
    test_rows = [row for row in rows if str(row["split"]) == "test"]
    general_texts = dataset_impl._unique_preserve(
        str(row["context"]) for row in train_rows
    )
    frequency_prior, frequency_class_scales, frequencies = _build_frequency_vectors(
        tokenizer,
        train_rows,
        frequency_content_scale,
        frequency_function_scale,
        frequency_punctuation_scale,
    )

    model = PriorControlledGrammarAttributeBertForMaskedLM(
        initial_model.config,
        tokenizer,
        frequency_prior=frequency_prior,
        frequency_class_scales=frequency_class_scales,
        attribute_filter=initial_model.grammar_attribute_filter,
        attribute_bias_scale=attribute_bias_scale,
        constrained_frequency_gate=constrained_frequency_gate,
    ).to(torch.device("cpu"))
    model.bert.load_state_dict(initial_model.bert.state_dict(), strict=True)
    model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
    for parameter in model.grammar_attribute_filter.parameters():
        parameter.requires_grad_(False)

    hidden_size = int(model.config.hidden_size)
    allocator = base.DimensionCombinationAllocator(hidden_size)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    layout = base.FeatureIndexLayout(parameters, hidden_size)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=fact_learning_rate,
        weight_decay=weight_decay,
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
    if os.path.isfile(os.path.join(output_path, "pytorch_model.bin")):
        raise FileExistsError(
            f"refusing to overwrite existing output checkpoint: {output_path}"
        )
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "before_fact"), exist_ok=True)

    memory = _build_memory(
        general_texts,
        tokenizer,
        frequencies,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
        model,
    )
    initial_model.eval()
    model.eval()
    fact_test_before_context = _fact_accuracy(
        model, fact_test, False, False
    )
    fact_test_before_grammar = _fact_accuracy(
        model, fact_test, True, False
    )
    fact_test_before_full = _fact_accuracy(model, fact_test, True, True)
    model.save_pretrained(os.path.join(output_path, "before_fact"))
    tokenizer.save_pretrained(os.path.join(output_path, "before_fact"))

    global_step = 0
    total_soft_conflicts = 0
    total_checked = 0
    total_overlap_replays = 0
    total_replay_overlap = 0
    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        model.train()
        total_steps = fact_epochs * fact_repeats * len(fact_train)
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
                    record = {
                        "stage": "fact_memory_prior_controlled",
                        "fact_epoch": fact_epoch + 1,
                        "fact_repeat": repeat + 1,
                        "fact_step": local_step,
                        "global_step": global_step,
                        "fact_id": sample["fact_id"],
                        "fact_token": sample["fact_token"],
                        **stats,
                    }
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if local_step % log_every == 0 or local_step == total_steps:
                        log_handle.flush()
                        print(
                            f"fact {fact_epoch + 1}/{fact_epochs}, "
                            f"repeat {repeat + 1}/{fact_repeats}: "
                            f"step={local_step}/{total_steps}, "
                            f"token={sample['fact_token']}, "
                            f"loss={stats['new_loss']:.4f}"
                        )

    model.eval()
    final_memory_loss, final_forgetting = base._evaluate_memory(model, memory)
    fact_dev_after_context = _fact_accuracy(model, fact_dev, False, False)
    fact_dev_after_grammar = _fact_accuracy(model, fact_dev, True, False)
    fact_dev_after_full = _fact_accuracy(model, fact_dev, True, True)
    fact_test_after_context = _fact_accuracy(model, fact_test, False, False)
    fact_test_after_grammar = _fact_accuracy(model, fact_test, True, False)
    fact_test_after_full = _fact_accuracy(model, fact_test, True, True)

    contribution = _contribution_report(model, initial_model, fact_test)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    dataset_impl._save_fact_manifest(
        output_path, fact_train + fact_dev + fact_test
    )
    from bert_mlm_dimension_combinations_epochs import _save_relation_map

    _save_relation_map(output_path, tokenizer, allocator.relation_triples)

    with open(
        os.path.join(output_path, "frequency_prior.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "role": "low_weight_fallback",
                "prior_is_frozen": True,
                "content_scale": frequency_content_scale,
                "function_scale": frequency_function_scale,
                "punctuation_scale": frequency_punctuation_scale,
                "constrained_frequency_gate": constrained_frequency_gate,
                "top_training_tokens": frequencies.most_common(100),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    used_dimensions = set()
    for triple in allocator.relation_triples.values():
        used_dimensions.update(triple)
    summary = {
        "stage": "bert_mlm_shiji_fact_memory_prior_controlled",
        "fact_dataset": resolved_dataset,
        "init_checkpoint": resolved_init,
        "attribute_dataset": base._resolve_path(attribute_dataset),
        "device": "cpu",
        "general_contexts_reused": len(general_texts),
        "fact_epochs": fact_epochs,
        "fact_repeats": fact_repeats,
        "optimizer_steps": global_step,
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "intermediate_size": int(model.config.intermediate_size),
        "max_length": max_length,
        "train_fact_samples": len(fact_train),
        "dev_fact_samples": len(fact_dev),
        "test_fact_samples": len(fact_test),
        "relation_mapping_fixed": True,
        "combination_capacity": allocator.capacity,
        "allocated_relation_combinations": len(allocator.relation_triples),
        "used_hidden_dimensions": len(used_dimensions),
        "fact_learning_rate": fact_learning_rate,
        "frequency_prior_frozen": True,
        "frequency_prior_learning_rate": 0.0,
        "frequency_content_scale": frequency_content_scale,
        "frequency_function_scale": frequency_function_scale,
        "frequency_punctuation_scale": frequency_punctuation_scale,
        "constrained_frequency_gate": constrained_frequency_gate,
        "attribute_bias_scale": attribute_bias_scale,
        "total_soft_conflict_triples": total_soft_conflicts,
        "total_checked_relation_triples": total_checked,
        "overlap_replay_steps": total_overlap_replays,
        "total_replay_overlap": total_replay_overlap,
        "fact_test_before_context": fact_test_before_context,
        "fact_test_before_grammar": fact_test_before_grammar,
        "fact_test_before_full": fact_test_before_full,
        "fact_dev_after_context": fact_dev_after_context,
        "fact_dev_after_grammar": fact_dev_after_grammar,
        "fact_dev_after_full": fact_dev_after_full,
        "fact_test_after_context": fact_test_after_context,
        "fact_test_after_grammar": fact_test_after_grammar,
        "fact_test_after_full": fact_test_after_full,
        "final_memory_loss": final_memory_loss,
        "final_forgetting": final_forgetting,
        "contribution_report": "contribution_report.json",
        "dominant_additive_source_counts": contribution[
            "dominant_additive_source_counts"
        ],
    }
    with open(
        os.path.join(output_path, "prior_controlled_fact_memory_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(output_path, "contribution_report.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(contribution, handle, ensure_ascii=False, indent=2)
    print(f"Saved prior-controlled fact-memory outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Shiji facts with grammar-first, low-frequency-prior MLM."
    )
    parser.add_argument("fact_dataset", nargs="?", default=DEFAULT_FACT_DATASET)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--attribute-dataset", default=DEFAULT_ATTRIBUTE_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=8)
    parser.add_argument("--fact-epochs", type=int, default=1)
    parser.add_argument("--fact-repeats", type=int, default=2)
    parser.add_argument("--fact-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.35)
    parser.add_argument("--soft-conflict-threshold", type=float, default=-0.35)
    parser.add_argument("--soft-conflict-scale", type=float, default=0.85)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--attribute-bias-scale", type=float, default=1.5)
    parser.add_argument("--frequency-content-scale", type=float, default=0.12)
    parser.add_argument("--frequency-function-scale", type=float, default=0.025)
    parser.add_argument("--frequency-punctuation-scale", type=float, default=0.01)
    parser.add_argument("--constrained-frequency-gate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
