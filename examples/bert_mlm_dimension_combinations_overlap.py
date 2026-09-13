"""BERT MLM with relation-specific 3D combinations and overlap-aware replay.

This is the follow-up experiment for the softer V-conflict idea.  It reuses
the implementation in bert_mlm_dimension_combinations.py, but chooses a
replay sentence with the largest relation overlap with the current sentence.
That makes the combination reuse and soft-conflict measurements observable
on a small sequential test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as base


DEFAULT_DATA_PATH = base.DEFAULT_DATA_PATH
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "bert-mlm-dimension-combinations-overlap-100",
    )
)

Triple = Tuple[int, int, int]


def _select_replay_index(
    memory: Sequence[Dict[str, object]],
    current_triples: Sequence[Triple],
    step: int,
) -> Tuple[Optional[int], int]:
    """优先选择与当前句共享三维关系最多的旧句。"""
    if not memory:
        return None, 0
    current = set(current_triples)
    overlap_scores = [
        len(current & set(item["relation_triples"])) for item in memory
    ]
    max_overlap = max(overlap_scores)
    if max_overlap > 0:
        candidates = [
            index
            for index, score in enumerate(overlap_scores)
            if score == max_overlap
        ]
        return candidates[(step - 1) % len(candidates)], max_overlap
    return (step - 1) % len(memory), 0


def train(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_sentences: int = 100,
    max_length: int = 128,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: Optional[int] = None,
    learning_rate: float = 5e-4,
    replay_weight: float = 0.5,
    background_weight: float = 0.35,
    max_context_tokens: int = 8,
    soft_conflict_threshold: float = -0.35,
    soft_conflict_scale: float = 0.85,
    mlm_probability: float = 0.15,
    seed: int = 42,
    log_every: int = 10,
) -> str:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
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
    print(f"Loaded {len(texts)} sentences; word-level vocab={len(tokenizer)}")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    examples: List[Dict[str, object]] = []
    for index, text in enumerate(texts):
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_special_tokens_mask=True,
        )
        sample = base._make_masked_example(
            encoded,
            tokenizer,
            seed=seed + index * 1009,
            mlm_probability=mlm_probability,
        )
        relations = base._relation_keys(
            sample,
            tokenizer,
            frequencies,
            max_context_tokens=max_context_tokens,
        )
        triples = [allocator.allocate(relation) for relation in relations]
        sample["relation_keys"] = relations
        sample["relation_triples"] = triples
        examples.append(sample)

    memory: List[Dict[str, object]] = []
    logs: List[Dict[str, object]] = []
    total_soft_conflicts = 0
    total_checked = 0
    overlap_replay_steps = 0
    total_replay_overlap = 0
    max_replay_overlap = 0
    output_path = base._resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        model.train()
        for step, new_sample in enumerate(examples, start=1):
            optimizer.zero_grad(set_to_none=True)
            new_loss = base._loss_for_sample(model, new_sample)
            new_gradients = base._gradient_tuple(new_loss, parameters)
            old_loss_value = 0.0
            old_gradients = [torch.zeros_like(gradient) for gradient in new_gradients]
            old_triples: List[Triple] = []
            replay_index, replay_overlap = _select_replay_index(
                memory, new_sample["relation_triples"], step
            )
            if replay_index is not None:
                if replay_overlap > 0:
                    overlap_replay_steps += 1
                    total_replay_overlap += replay_overlap
                    max_replay_overlap = max(max_replay_overlap, replay_overlap)
                old_item = memory[replay_index]
                old_sample = old_item["sample"]
                old_loss = base._loss_for_sample(model, old_sample)
                old_loss_value = float(old_loss.item())
                old_gradients = base._gradient_tuple(old_loss, parameters)
                old_triples = old_item["relation_triples"]

            accepted_gradients, guard_stats = base._combine_gradients(
                new_gradients,
                old_gradients,
                new_sample["relation_triples"],
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
            current_new_loss = float(base._loss_for_sample(model, new_sample).item())
            memory.append(
                {
                    "sample": new_sample,
                    "reference_loss": current_new_loss,
                    "relation_triples": new_sample["relation_triples"],
                }
            )

            memory_loss = 0.0
            forgetting = 0.0
            if step == 1 or step % log_every == 0 or step == len(examples):
                memory_loss, forgetting = base._evaluate_memory(model, memory)
                model.train()
                print(
                    f"step {step}/{len(examples)}: "
                    f"new_loss={float(new_loss.item()):.4f}, "
                    f"replay_loss={old_loss_value:.4f}, "
                    f"relations={len(new_sample['relation_triples'])}, "
                    f"replay_overlap={replay_overlap}, "
                    f"shared={int(guard_stats['shared_relation_triples'])}, "
                    f"soft_conflicts={int(guard_stats['soft_conflict_triples'])}, "
                    f"memory_loss={memory_loss:.4f}, "
                    f"forgetting={forgetting:.4f}"
                )

            record = {
                "step": step,
                "new_loss": float(new_loss.item()),
                "replay_loss": old_loss_value,
                "replay_index": replay_index,
                "replay_overlap": replay_overlap,
                "current_new_loss": current_new_loss,
                "memory_size": len(memory),
                "memory_loss": memory_loss,
                "forgetting": forgetting,
                **guard_stats,
            }
            logs.append(record)
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    used_dimensions: Set[int] = set()
    for triple in allocator.relation_triples.values():
        used_dimensions.update(triple)
    summary = {
        "stage": "bert_mlm_dimension_combination_overlap_replay",
        "training_file": resolved_training_file,
        "device": "cpu",
        "sentences": len(texts),
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "combination_capacity": allocator.capacity,
        "allocated_relation_combinations": len(allocator.relation_triples),
        "used_hidden_dimensions": len(used_dimensions),
        "learning_rate": learning_rate,
        "replay_weight": replay_weight,
        "background_weight": background_weight,
        "max_context_tokens": max_context_tokens,
        "soft_conflict_threshold": soft_conflict_threshold,
        "soft_conflict_scale": soft_conflict_scale,
        "mlm_probability": mlm_probability,
        "overlap_replay_steps": overlap_replay_steps,
        "total_replay_overlap": total_replay_overlap,
        "max_replay_overlap": max_replay_overlap,
        "total_soft_conflict_triples": total_soft_conflicts,
        "total_checked_relation_triples": total_checked,
        "final_memory_loss": logs[-1]["memory_loss"],
        "final_forgetting": logs[-1]["forgetting"],
    }
    with open(
        os.path.join(output_path, "dimension_combination_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved overlap-aware outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train word-level BERT MLM with overlap-aware 3D combinations."
    )
    parser.add_argument("data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sentences", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.35)
    parser.add_argument("--max-context-tokens", type=int, default=8)
    parser.add_argument("--soft-conflict-threshold", type=float, default=-0.35)
    parser.add_argument("--soft-conflict-scale", type=float, default=0.85)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
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
        replay_weight=args.replay_weight,
        background_weight=args.background_weight,
        max_context_tokens=args.max_context_tokens,
        soft_conflict_threshold=args.soft_conflict_threshold,
        soft_conflict_scale=args.soft_conflict_scale,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
