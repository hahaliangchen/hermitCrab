"""Larger train/validation ablation for the routed dynamic 3D V path."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as combination_base
import bert_mlm_dynamic_word_spaces as base
import ablate_dynamic_value as small_ablation


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "shiji_baihua_zhangchen_gaozu_long_context.txt"
)
MAX_LENGTH = 96
HIDDEN_SIZE = 64
NUM_LAYERS = 4
NUM_HEADS = 4
EPOCHS = 8


def _build_model(
    tokenizer: combination_base.SimpleBertTokenizer,
    dynamic_value: bool,
) -> base.DynamicWordSpaceBertForMaskedLM:
    config = combination_base.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=HIDDEN_SIZE * 4,
        max_position_embeddings=MAX_LENGTH + 10,
        position_embedding_type="relative",
        relative_position_max_distance=MAX_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    if dynamic_value:
        bank = small_ablation.DynamicValueWordSpaceBank(
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            max_spaces=8,
            hidden_size=HIDDEN_SIZE,
            route_dim=16,
        )
        return small_ablation.DynamicValueWordSpaceBertForMaskedLM(
            config,
            bank=bank,
            route_start_layer=1,
            route_dim=16,
        )
    bank = base.DynamicWordSpaceBank(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_spaces=8,
        hidden_size=HIDDEN_SIZE,
        route_dim=16,
    )
    return base.DynamicWordSpaceBertForMaskedLM(
        config,
        bank=bank,
        route_start_layer=1,
        route_dim=16,
    )


def _make_samples(
    texts: Sequence[str],
    tokenizer: combination_base.SimpleBertTokenizer,
    registry: base.WordSpaceRegistry,
    attribute_registry: base.WordAttributeRegistry,
    seed_offset: int,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for index, text in enumerate(texts):
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )
        sample = combination_base._make_masked_example(
            encoded,
            tokenizer,
            seed=seed_offset + index,
            mlm_probability=0.15,
        )
        sample["touched_token_ids"] = sorted(registry.touched_token_ids(sample))
        sample["candidate_space_mask"] = registry.candidate_mask(sample)
        sample["grammar_allowed_mask"] = attribute_registry.allowed_mask(
            sample["input_ids"]
        )
        samples.append(sample)
    return samples


@torch.no_grad()
def _evaluate(
    model: base.DynamicWordSpaceBertForMaskedLM,
    samples: Sequence[Dict[str, object]],
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    losses: List[float] = []
    correct = 0
    total = 0
    for sample in samples:
        result = model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            labels=sample["labels"],
            candidate_space_mask=sample["candidate_space_mask"],
            grammar_allowed_mask=sample["grammar_allowed_mask"],
        )
        losses.append(float(result[0].item()))
        labels = sample["labels"]
        valid = labels.ge(0)
        predictions = result[1].argmax(dim=-1)
        correct += int((predictions[valid] == labels[valid]).sum().item())
        total += int(valid.sum().item())
    if was_training:
        model.train()
    return sum(losses) / max(len(losses), 1), correct / max(total, 1)


def _run(
    model: base.DynamicWordSpaceBertForMaskedLM,
    train_samples: Sequence[Dict[str, object]],
    valid_samples: Sequence[Dict[str, object]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    total_steps = EPOCHS * len(train_samples)
    print(f"  training {total_steps} steps", flush=True)
    for step in range(1, total_steps + 1):
        sample = train_samples[(step - 1) % len(train_samples)]
        optimizer.zero_grad(set_to_none=True)
        result = model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            labels=sample["labels"],
            candidate_space_mask=sample["candidate_space_mask"],
            grammar_allowed_mask=sample["grammar_allowed_mask"],
        )
        result[0].backward()
        optimizer.step()
        if step == len(train_samples) or step == total_steps:
            print(f"  finished step {step}/{total_steps}", flush=True)
    return _evaluate(model, train_samples), _evaluate(model, valid_samples)


def main() -> None:
    combination_base.set_seed(42)
    texts = combination_base.load_texts(DATA_PATH, max_sentences=100)
    train_texts = [text for index, text in enumerate(texts) if index % 5 != 0]
    valid_texts = [text for index, text in enumerate(texts) if index % 5 == 0]

    tokenizer = combination_base.SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    registry = base.WordSpaceRegistry(tokenizer, max_spaces=8, initial_spaces=4)
    attribute_registry = base.WordAttributeRegistry(tokenizer)
    train_samples = _make_samples(
        train_texts,
        tokenizer,
        registry,
        attribute_registry,
        seed_offset=1000,
    )
    valid_samples = _make_samples(
        valid_texts,
        tokenizer,
        registry,
        attribute_registry,
        seed_offset=2000,
    )

    baseline = _build_model(tokenizer, dynamic_value=False)
    dynamic_v = _build_model(tokenizer, dynamic_value=True)
    small_ablation._copy_common_parameters(baseline, dynamic_v)

    print(
        f"sentences={len(texts)} train={len(train_samples)} valid={len(valid_samples)} "
        f"vocab={len(tokenizer)} max_length={MAX_LENGTH} hidden={HIDDEN_SIZE} "
        f"layers={NUM_LAYERS} heads={NUM_HEADS} epochs={EPOCHS}"
    )
    initial_train = _evaluate(baseline, train_samples)
    initial_valid = _evaluate(baseline, valid_samples)
    print(
        f"initial loss train={initial_train[0]:.4f} top1={initial_train[1]:.1%} "
        f"valid={initial_valid[0]:.4f} top1={initial_valid[1]:.1%}"
    )

    print("ordinary dynamic Q/K")
    baseline_result = _run(baseline, train_samples, valid_samples)
    print(
        f"  final train_loss={baseline_result[0][0]:.4f} "
        f"train_top1={baseline_result[0][1]:.1%} "
        f"valid_loss={baseline_result[1][0]:.4f} "
        f"valid_top1={baseline_result[1][1]:.1%}"
    )

    print("dynamic Q/K + dynamic V")
    dynamic_v_result = _run(dynamic_v, train_samples, valid_samples)
    print(
        f"  final train_loss={dynamic_v_result[0][0]:.4f} "
        f"train_top1={dynamic_v_result[0][1]:.1%} "
        f"valid_loss={dynamic_v_result[1][0]:.4f} "
        f"valid_top1={dynamic_v_result[1][1]:.1%}"
    )
    bank = dynamic_v.space_bank
    print(
        f"dynamic_v v_out_norm={float(bank.v_out.norm().item()):.4f} "
        f"params ordinary={sum(p.numel() for p in baseline.parameters())} "
        f"dynamic_v={sum(p.numel() for p in dynamic_v.parameters())}"
    )


if __name__ == "__main__":
    main()
