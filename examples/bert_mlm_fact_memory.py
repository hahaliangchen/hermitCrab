"""Two-stage MLM training with a targeted rare-fact memory phase.

Stage 1 learns general context with stable relation-specific 3D combinations.
Stage 2 repeatedly masks low-frequency content words at their real source
positions, while replaying stage-1 samples.  The relation-to-triple map is
never reallocated in either stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as base
from bert_mlm_dimension_combinations_epochs import (
    _eval_loss,
    _make_epoch_sample,
    _make_stable_examples,
)
from bert_mlm_dimension_combinations_overlap import _select_replay_index


DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "shiji_baihua_zhangchen_gaozu_long_context.txt"
    )
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "bert-mlm-fact-memory-100",
    )
)

Triple = Tuple[int, int, int]


def _is_chinese_content_token(token: str) -> bool:
    return len(token) >= 2 and any("\u4e00" <= char <= "\u9fff" for char in token)


def _make_fact_example(
    encoded: Dict[str, List[int]],
    tokenizer: base.SimpleBertTokenizer,
    position: int,
    relation_keys: Sequence[base.RelationKey],
    relation_triples: Sequence[Triple],
    source_index: int,
) -> Dict[str, object]:
    original = torch.tensor(encoded["input_ids"], dtype=torch.long)
    input_ids = original.clone()
    labels = torch.full_like(original, -100)
    target_id = int(original[position].item())
    input_ids[position] = tokenizer.mask_token_id
    labels[position] = target_id
    mask_positions = torch.zeros_like(original, dtype=torch.bool)
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
        "original_input_ids": original,
        "mask_positions": mask_positions,
        "relation_keys": list(relation_keys),
        "relation_triples": list(relation_triples),
        "fact_position": position,
        "fact_token_id": target_id,
        "source_index": source_index,
    }


def _build_fact_samples(
    encodings: Sequence[Dict[str, List[int]]],
    probes: Sequence[Dict[str, object]],
    tokenizer: base.SimpleBertTokenizer,
    frequencies: Counter,
    min_frequency: int,
    max_frequency: int,
    max_occurrences: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    candidates = []
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    for source_index, encoded in enumerate(encodings):
        special_mask = encoded.get(
            "special_tokens_mask", [0] * len(encoded["input_ids"])
        )
        for position, token_id in enumerate(encoded["input_ids"]):
            token_id = int(token_id)
            if special_mask[position] or token_id in special_ids:
                continue
            token = tokenizer.id_to_token[token_id]
            frequency = int(frequencies[token])
            if (
                min_frequency <= frequency <= max_frequency
                and _is_chinese_content_token(token)
            ):
                candidates.append((frequency, token, source_index, position))

    # Prefer tokens with a repeatable but still rare signal.  The second and
    # third occurrences are more useful for this experiment than singletons.
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    if max_occurrences > 0:
        candidates = candidates[:max_occurrences]

    samples: List[Dict[str, object]] = []
    token_names = set()
    for _, token, source_index, position in candidates:
        probe = probes[source_index]
        sample = _make_fact_example(
            encodings[source_index],
            tokenizer,
            position,
            probe["relation_keys"],
            probe["relation_triples"],
            source_index,
        )
        sample["fact_token"] = token
        samples.append(sample)
        token_names.add(token)
    return samples, sorted(token_names)


@torch.no_grad()
def _fact_accuracy(
    model: base.BertForMaskedLM, samples: Sequence[Dict[str, object]]
) -> Dict[str, float]:
    if not samples:
        return {"samples": 0.0, "top1": 0.0, "top5": 0.0, "loss": 0.0}
    was_training = model.training
    model.eval()
    top1 = 0
    top5 = 0
    losses = []
    for sample in samples:
        loss, logits, _ = model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            labels=sample["labels"],
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


def _run_update(
    model: base.BertForMaskedLM,
    parameters: Sequence[torch.nn.Parameter],
    layout: base.FeatureIndexLayout,
    hidden_size: int,
    optimizer: torch.optim.Optimizer,
    sample: Dict[str, object],
    memory: Sequence[Dict[str, object]],
    global_step: int,
    replay_weight: float,
    background_weight: float,
    soft_conflict_threshold: float,
    soft_conflict_scale: float,
) -> Dict[str, float]:
    replay_index, replay_overlap = _select_replay_index(
        memory, sample["relation_triples"], global_step
    )
    optimizer.zero_grad(set_to_none=True)
    new_loss = base._loss_for_sample(model, sample)
    new_gradients = base._gradient_tuple(new_loss, parameters)
    old_gradients = [torch.zeros_like(gradient) for gradient in new_gradients]
    old_triples: List[Triple] = []
    old_loss_value = 0.0
    if replay_index is not None:
        old_item = memory[replay_index]
        old_sample = old_item["sample"]
        old_loss = base._loss_for_sample(model, old_sample)
        old_loss_value = float(old_loss.item())
        old_gradients = base._gradient_tuple(old_loss, parameters)
        old_triples = old_item["relation_triples"]

    accepted_gradients, guard_stats = base._combine_gradients(
        new_gradients,
        old_gradients,
        sample["relation_triples"],
        old_triples,
        layout,
        hidden_size,
        replay_weight=replay_weight if memory else 0.0,
        background_weight=background_weight,
        soft_conflict_threshold=soft_conflict_threshold,
        soft_conflict_scale=soft_conflict_scale,
    )
    for parameter, gradient in zip(parameters, accepted_gradients):
        parameter.grad = gradient
    optimizer.step()
    return {
        "new_loss": float(new_loss.item()),
        "replay_loss": old_loss_value,
        "replay_overlap": float(replay_overlap),
        "replay_index": float(replay_index if replay_index is not None else -1),
        **guard_stats,
    }


def _save_fact_manifest(
    output_path: str, samples: Sequence[Dict[str, object]], tokenizer: base.SimpleBertTokenizer
) -> None:
    manifest = []
    for sample in samples:
        manifest.append(
            {
                "fact_token": sample["fact_token"],
                "fact_token_id": int(sample["fact_token_id"]),
                "source_index": int(sample["source_index"]),
                "position": int(sample["fact_position"]),
                "relation_triples": [list(triple) for triple in sample["relation_triples"]],
            }
        )
    with open(
        os.path.join(output_path, "fact_memory_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def train(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_sentences: int = 100,
    max_length: int = 128,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: Optional[int] = None,
    general_epochs: int = 20,
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
    rare_min_frequency: int = 2,
    rare_max_frequency: int = 3,
    max_fact_occurrences: int = 0,
    seed: int = 42,
    log_every: int = 500,
) -> str:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if general_epochs < 1 or fact_epochs < 1 or fact_repeats < 1:
        raise ValueError("general_epochs, fact_epochs, and fact_repeats must be positive")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    base.set_seed(seed)
    resolved_training_file = base._resolve_path(training_file)
    texts = base.load_texts(resolved_training_file, max_sentences)
    tokenizer = base.SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    frequencies = Counter()
    for text in texts:
        frequencies.update(tokenizer.tokenize(text))
    print(f"Using training file: {resolved_training_file}")
    print(
        f"Loaded {len(texts)} samples; vocab={len(tokenizer)}; "
        f"general_epochs={general_epochs}; fact_epochs={fact_epochs}"
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

    probe_samples, encodings = _make_stable_examples(
        texts,
        tokenizer,
        frequencies,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
    )
    fact_samples, fact_token_names = _build_fact_samples(
        encodings,
        probe_samples,
        tokenizer,
        frequencies,
        rare_min_frequency,
        rare_max_frequency,
        max_fact_occurrences,
    )
    if not fact_samples:
        raise RuntimeError("no rare fact samples matched the configured frequency range")
    print(
        f"Fact candidates: {len(fact_samples)} occurrences, "
        f"{len(fact_token_names)} token types"
    )

    memory: List[Dict[str, object]] = []
    logs: List[Dict[str, object]] = []
    total_soft_conflicts = 0
    total_checked = 0
    total_overlap_replays = 0
    total_replay_overlap = 0
    output_path = base._resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    global_step = 0
    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        # Stage 1: broad context learning.  The relation map is created once;
        # only the random mask changes from epoch to epoch.
        model.train()
        for epoch in range(general_epochs):
            epoch_seed = seed + epoch * 1_000_003
            for sample_index, encoded in enumerate(encodings):
                probe = probe_samples[sample_index]
                sample = _make_epoch_sample(
                    encoded,
                    tokenizer,
                    probe["relation_keys"],
                    probe["relation_triples"],
                    seed=epoch_seed + sample_index * 1009,
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
                if (
                    global_step % log_every == 0
                    or sample_index + 1 == len(encodings)
                ):
                    memory_loss, forgetting = base._evaluate_memory(model, memory)
                    record = {
                        "stage": "general",
                        "epoch": epoch + 1,
                        "step_in_epoch": sample_index + 1,
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
                        f"general epoch {epoch + 1}/{general_epochs}: "
                        f"step={sample_index + 1}/{len(encodings)}, "
                        f"loss={stats['new_loss']:.4f}, memory_loss={memory_loss:.4f}, "
                        f"forgetting={forgetting:.6f}"
                    )

        general_memory_loss, general_forgetting = base._evaluate_memory(model, memory)
        fact_before = _fact_accuracy(model, fact_samples)
        before_fact_dir = os.path.join(output_path, "before_fact")
        model.save_pretrained(before_fact_dir)
        tokenizer.save_pretrained(before_fact_dir)

        # Stage 2: targeted rare-fact memory.  The base memory is always
        # replayed so this stage cannot freely overwrite general context.
        for group in optimizer.param_groups:
            group["lr"] = fact_learning_rate
        for fact_epoch in range(fact_epochs):
            for repeat in range(fact_repeats):
                for fact_index, sample in enumerate(fact_samples):
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
                    fact_step = (
                        fact_epoch * fact_repeats * len(fact_samples)
                        + repeat * len(fact_samples)
                        + fact_index
                        + 1
                    )
                    if fact_step % log_every == 0 or fact_step == fact_repeats * len(fact_samples):
                        record = {
                            "stage": "fact_memory",
                            "fact_epoch": fact_epoch + 1,
                            "fact_repeat": repeat + 1,
                            "fact_step": fact_step,
                            "global_step": global_step,
                            "fact_token": sample["fact_token"],
                            **stats,
                        }
                        logs.append(record)
                        log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log_handle.flush()
                        print(
                            f"fact epoch {fact_epoch + 1}/{fact_epochs}, "
                            f"repeat {repeat + 1}/{fact_repeats}: "
                            f"step={fact_step}/{fact_epochs * fact_repeats * len(fact_samples)}, "
                            f"token={sample['fact_token']}, loss={stats['new_loss']:.4f}"
                        )

        final_memory_loss, final_forgetting = base._evaluate_memory(model, memory)
        fact_after = _fact_accuracy(model, fact_samples)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    _save_fact_manifest(output_path, fact_samples, tokenizer)
    used_dimensions = set()
    for triple in allocator.relation_triples.values():
        used_dimensions.update(triple)
    summary = {
        "stage": "bert_mlm_fact_memory_two_stage",
        "training_file": resolved_training_file,
        "device": "cpu",
        "sentences_per_epoch": len(texts),
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
        "relation_mapping_fixed": True,
        "masks_regenerated_in_general_stage": True,
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
        "rare_min_frequency": rare_min_frequency,
        "rare_max_frequency": rare_max_frequency,
        "fact_occurrences": len(fact_samples),
        "fact_token_types": len(fact_token_names),
        "fact_before": fact_before,
        "fact_after": fact_after,
        "general_memory_loss_before_fact": general_memory_loss,
        "general_forgetting_before_fact": general_forgetting,
        "final_memory_loss": final_memory_loss,
        "final_forgetting": final_forgetting,
        "total_soft_conflict_triples": total_soft_conflicts,
        "total_checked_relation_triples": total_checked,
        "overlap_replay_steps": total_overlap_replays,
        "total_replay_overlap": total_replay_overlap,
    }
    with open(
        os.path.join(output_path, "fact_memory_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved fact-memory outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT MLM with a targeted rare-fact memory stage."
    )
    parser.add_argument("data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sentences", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--general-epochs", type=int, default=20)
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
    parser.add_argument("--rare-min-frequency", type=int, default=2)
    parser.add_argument("--rare-max-frequency", type=int, default=3)
    parser.add_argument("--max-fact-occurrences", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        training_file=args.data_path,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
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
        rare_min_frequency=args.rare_min_frequency,
        rare_max_frequency=args.rare_max_frequency,
        max_fact_occurrences=args.max_fact_occurrences,
        seed=args.seed,
        log_every=args.log_every,
    )
