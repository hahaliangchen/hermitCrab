"""Multi-epoch BERT MLM training with stable relation-specific 3D masks.

This is the longer-training counterpart to
``bert_mlm_dimension_combinations_overlap.py``.  Relation-to-triple
assignments are created once and then reused for every epoch.  The MLM masks
are regenerated each epoch so a small corpus exposes different target
positions over time.  Shared triples are replayed, while only strong reverse
gradients receive a mild soft reduction.
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
        "bert-mlm-dimension-combinations-epochs-100",
    )
)

Triple = Tuple[int, int, int]


def _eval_loss(model: base.BertForMaskedLM, sample: Dict[str, object]) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        value = float(base._loss_for_sample(model, sample).item())
    if was_training:
        model.train()
    return value


def _make_stable_examples(
    texts: Sequence[str],
    tokenizer: base.SimpleBertTokenizer,
    frequencies: Counter,
    allocator: base.DimensionCombinationAllocator,
    max_length: int,
    max_context_tokens: int,
    seed: int,
    mlm_probability: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, List[int]]]]:
    """Create fixed relation metadata and one deterministic probe mask."""
    probes: List[Dict[str, object]] = []
    encodings: List[Dict[str, List[int]]] = []
    for index, text in enumerate(texts):
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )
        probe = base._make_masked_example(
            encoded,
            tokenizer,
            seed=seed + index * 1009,
            mlm_probability=mlm_probability,
        )
        relations = base._relation_keys(
            probe,
            tokenizer,
            frequencies,
            max_context_tokens=max_context_tokens,
        )
        triples = [allocator.allocate(relation) for relation in relations]
        probe["relation_keys"] = relations
        probe["relation_triples"] = triples
        probes.append(probe)
        encodings.append(encoded)
    return probes, encodings


def _make_epoch_sample(
    encoded: Dict[str, List[int]],
    tokenizer: base.SimpleBertTokenizer,
    relation_keys: Sequence[base.RelationKey],
    relation_triples: Sequence[Triple],
    seed: int,
    mlm_probability: float,
) -> Dict[str, object]:
    sample = base._make_masked_example(
        encoded,
        tokenizer,
        seed=seed,
        mlm_probability=mlm_probability,
    )
    sample["relation_keys"] = list(relation_keys)
    sample["relation_triples"] = list(relation_triples)
    return sample


def _save_relation_map(
    output_path: str,
    tokenizer: base.SimpleBertTokenizer,
    relation_triples: Dict[base.RelationKey, Triple],
) -> None:
    mapping = {}
    for (left_id, right_id), triple in sorted(relation_triples.items()):
        left = tokenizer.id_to_token[left_id]
        right = tokenizer.id_to_token[right_id]
        mapping[f"{left} || {right}"] = list(triple)
    with open(
        os.path.join(output_path, "dimension_combination_map.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)


def train(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_sentences: int = 100,
    max_length: int = 128,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: Optional[int] = None,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.0,
    epochs: int = 20,
    replay_weight: float = 0.5,
    background_weight: float = 0.35,
    max_context_tokens: int = 8,
    soft_conflict_threshold: float = -0.35,
    soft_conflict_scale: float = 0.85,
    mlm_probability: float = 0.15,
    seed: int = 42,
    log_every: int = 100,
) -> str:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError("background_weight must be between 0 and 1")
    if not 0.0 < soft_conflict_scale <= 1.0:
        raise ValueError("soft_conflict_scale must be in (0, 1]")
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
        f"Loaded {len(texts)} samples; word-level vocab={len(tokenizer)}; "
        f"epochs={epochs}"
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
    memory: List[Dict[str, object]] = []
    logs: List[Dict[str, object]] = []
    total_soft_conflicts = 0
    total_checked = 0
    total_overlap_replays = 0
    total_replay_overlap = 0
    output_path = base._resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        model.train()
        for epoch in range(epochs):
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
                global_step = epoch * len(encodings) + sample_index + 1
                replay_index, replay_overlap = _select_replay_index(
                    memory,
                    sample["relation_triples"],
                    global_step,
                )
                if replay_overlap > 0:
                    total_overlap_replays += 1
                    total_replay_overlap += replay_overlap

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

                total_soft_conflicts += int(guard_stats["soft_conflict_triples"])
                total_checked += int(guard_stats["checked_relation_triples"])
                current_new_loss = _eval_loss(model, sample)

                if epoch == 0:
                    memory.append(
                        {
                            "sample": probe,
                            "reference_loss": _eval_loss(model, probe),
                            "relation_triples": probe["relation_triples"],
                        }
                    )

                should_log = (
                    global_step == 1
                    or global_step % log_every == 0
                    or sample_index + 1 == len(encodings)
                )
                memory_loss = 0.0
                forgetting = 0.0
                if should_log:
                    memory_loss, forgetting = base._evaluate_memory(model, memory)
                    print(
                        f"epoch {epoch + 1}/{epochs}, "
                        f"step {sample_index + 1}/{len(encodings)}: "
                        f"new_loss={float(new_loss.item()):.4f}, "
                        f"replay_loss={old_loss_value:.4f}, "
                        f"relations={len(sample['relation_triples'])}, "
                        f"replay_overlap={replay_overlap}, "
                        f"soft_conflicts={int(guard_stats['soft_conflict_triples'])}, "
                        f"memory_loss={memory_loss:.4f}, "
                        f"forgetting={forgetting:.6f}"
                    )

                record = {
                    "epoch": epoch + 1,
                    "step_in_epoch": sample_index + 1,
                    "global_step": global_step,
                    "new_loss": float(new_loss.item()),
                    "replay_loss": old_loss_value,
                    "replay_index": replay_index,
                    "current_new_loss": current_new_loss,
                    "memory_size": len(memory),
                    "memory_loss": memory_loss,
                    "forgetting": forgetting,
                    "replay_overlap": replay_overlap,
                    **guard_stats,
                }
                logs.append(record)
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    _save_relation_map(output_path, tokenizer, allocator.relation_triples)
    memory_loss, forgetting = base._evaluate_memory(model, memory)
    summary = {
        "stage": "bert_mlm_dimension_combination_multi_epoch_overlap_replay",
        "training_file": resolved_training_file,
        "device": "cpu",
        "sentences_per_epoch": len(texts),
        "epochs": epochs,
        "optimizer_steps": len(texts) * epochs,
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "masks_regenerated_each_epoch": True,
        "relation_mapping_fixed_across_epochs": True,
        "combination_capacity": allocator.capacity,
        "allocated_relation_combinations": len(allocator.relation_triples),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "replay_weight": replay_weight,
        "background_weight": background_weight,
        "max_context_tokens": max_context_tokens,
        "soft_conflict_threshold": soft_conflict_threshold,
        "soft_conflict_scale": soft_conflict_scale,
        "mlm_probability": mlm_probability,
        "overlap_replay_steps": total_overlap_replays,
        "total_replay_overlap": total_replay_overlap,
        "total_soft_conflict_triples": total_soft_conflicts,
        "total_checked_relation_triples": total_checked,
        "final_memory_loss": memory_loss,
        "final_forgetting": forgetting,
    }
    with open(
        os.path.join(output_path, "dimension_combination_epochs_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved multi-epoch outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT MLM for multiple epochs with stable 3D combinations."
    )
    parser.add_argument("data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sentences", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
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
        training_file=args.data_path,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        replay_weight=args.replay_weight,
        background_weight=args.background_weight,
        max_context_tokens=args.max_context_tokens,
        soft_conflict_threshold=args.soft_conflict_threshold,
        soft_conflict_scale=args.soft_conflict_scale,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
