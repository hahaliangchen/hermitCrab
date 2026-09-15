"""Compare 4-layer and 8-layer residual BERTs on the same tiny MLM task."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bert_simple.model import BertConfig, BertForMaskedLM
from examples.ablate_residuals import (
    BATCH_SIZE,
    SEED,
    STEPS,
    VALID_ROWS,
    TRAIN_ROWS,
    evaluate,
    grad_norm,
    make_examples,
    make_tokenizer,
    make_batch,
)


def run_depth(
    layers: int,
    tokenizer,
    train_examples,
    valid_examples,
    max_length: int,
) -> None:
    torch.manual_seed(SEED)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=max_length + 2,
        position_embedding_type="relative",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    model = BertForMaskedLM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    initial_train = evaluate(model, train_examples, max_length)
    initial_valid = evaluate(model, valid_examples, max_length)
    print(
        f"layers={layers:<2d} params={parameter_count:<7d} step=0   "
        f"train_loss={initial_train[0]:.4f} train_top1={initial_train[1]:.2%} "
        f"valid_loss={initial_valid[0]:.4f} valid_top1={initial_valid[1]:.2%}"
    )

    checkpoints = {1, 10, 50, 100, 200, STEPS}
    for step in range(1, STEPS + 1):
        model.train()
        input_ids, attention_mask, labels, _ = make_batch(
            train_examples,
            step - 1,
            BATCH_SIZE,
            max_length,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if not bool(torch.isfinite(loss)):
            print(f"layers={layers:<2d} step={step:<3d} loss=NON_FINITE")
            return
        loss.backward()
        current_grad_norm = grad_norm(model)
        optimizer.step()
        if step in checkpoints:
            train_metrics = evaluate(model, train_examples, max_length)
            valid_metrics = evaluate(model, valid_examples, max_length)
            print(
                f"layers={layers:<2d} params={parameter_count:<7d} step={step:<3d} "
                f"train_loss={train_metrics[0]:.4f} train_top1={train_metrics[1]:.2%} "
                f"valid_loss={valid_metrics[0]:.4f} valid_top1={valid_metrics[1]:.2%} "
                f"grad_norm={current_grad_norm:.4f}"
            )


def main() -> None:
    torch.set_num_threads(1)
    tokenizer = make_tokenizer()
    train_examples = make_examples(tokenizer, TRAIN_ROWS)
    valid_examples = make_examples(tokenizer, VALID_ROWS)
    max_length = max(
        len(example["input_ids"])
        for example in train_examples + valid_examples
    )
    print(
        f"vocab={len(tokenizer)} max_length={max_length} hidden=64 heads=4 "
        f"steps={STEPS}"
    )
    for layers in (4, 8):
        run_depth(
            layers,
            tokenizer,
            train_examples,
            valid_examples,
            max_length,
        )


if __name__ == "__main__":
    main()
