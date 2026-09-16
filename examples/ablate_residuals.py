"""Controlled MLM experiment for removing Transformer residual connections.

The three compared modes use the same initial state and the same batches:

* baseline: attention and FFN residuals enabled;
* no_attention_residual: remove only Attention(x) + x;
* no_all_residuals: remove both Attention(x) + x and FFN(x) + x.

This is intentionally a small, deterministic experiment.  Its numbers are
not a benchmark; they are meant to make the role of the skip connections
visible on the project's MLM implementation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

import torch

from bert_simple.model import BertAttention, BertConfig, BertForMaskedLM, BertOutput
from bert_simple.tokenizer import SimpleBertTokenizer


SEED = 20260915
STEPS = 240
BATCH_SIZE = 4

TRAIN_ROWS: Sequence[Tuple[str, str]] = (
    ("汉王 立 张耳 为 赵王 。", "赵王"),
    ("汉王 立 韩信 为 齐王 。", "齐王"),
    ("汉王 立 彭越 为 梁王 。", "梁王"),
    ("项羽 封 英布 为 九江王 。", "九江王"),
    ("刘邦 封 萧何 为 相国 。", "相国"),
    ("高祖 立 刘肥 为 齐王 。", "齐王"),
    ("秦王 封 蒙恬 为 将军 。", "将军"),
    ("赵王 任 李牧 为 将军 。", "将军"),
)

VALID_ROWS: Sequence[Tuple[str, str]] = (
    ("高祖 立 张耳 为 赵王 。", "赵王"),
    ("项羽 立 韩信 为 齐王 。", "齐王"),
    ("刘邦 立 蒙恬 为 将军 。", "将军"),
    ("汉王 封 萧何 为 相国 。", "相国"),
)


class NoAttentionResidual(BertAttention):
    """BertAttention with the attention skip connection removed."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        route_weights: torch.Tensor | None = None,
    ):
        self_out, attn_probs = self.self_attn(
            hidden_states,
            attention_mask,
            route_weights=route_weights,
        )
        hidden_states = self.LayerNorm(self.dropout(self_out))
        return hidden_states, attn_probs if output_attentions else None


class NoFFNResidual(BertOutput):
    """BertOutput with the feed-forward skip connection removed."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(self.dropout(hidden_states))
        return hidden_states


def seed_everything() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def make_tokenizer() -> SimpleBertTokenizer:
    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(
        [text for text, _ in TRAIN_ROWS + VALID_ROWS],
        min_freq=1,
    )
    return tokenizer


def make_examples(
    tokenizer: SimpleBertTokenizer,
    rows: Sequence[Tuple[str, str]],
) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for text, target in rows:
        tokens = tokenizer.tokenize(text)
        if target not in tokens:
            raise ValueError(f"target {target!r} is not a token in {text!r}")
        target_position = tokens.index(target) + 1  # account for [CLS]
        input_ids = [
            tokenizer.cls_token_id,
            *[tokenizer.token_to_id[token] for token in tokens],
            tokenizer.sep_token_id,
        ]
        labels = [-100] * len(input_ids)
        labels[target_position] = input_ids[target_position]
        masked_input_ids = list(input_ids)
        masked_input_ids[target_position] = tokenizer.mask_token_id
        examples.append(
            {
                "input_ids": masked_input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
                "target_position": target_position,
            }
        )
    return examples


def make_batch(
    examples: Sequence[Dict[str, object]],
    step: int,
    batch_size: int,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [examples[(step * batch_size + index) % len(examples)] for index in range(batch_size)]
    input_rows = []
    label_rows = []
    attention_rows = []
    target_positions = []
    for example in selected:
        input_ids = list(example["input_ids"])
        labels = list(example["labels"])
        attention_mask = list(example["attention_mask"])
        pad_length = max_length - len(input_ids)
        input_rows.append(input_ids + [0] * pad_length)
        label_rows.append(labels + [-100] * pad_length)
        attention_rows.append(attention_mask + [0] * pad_length)
        target_positions.append(int(example["target_position"]))
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(attention_rows, dtype=torch.long),
        torch.tensor(label_rows, dtype=torch.long),
        torch.tensor(target_positions, dtype=torch.long),
    )


def replace_residuals(model: BertForMaskedLM, mode: str) -> None:
    if mode not in {"baseline", "no_attention_residual", "no_all_residuals"}:
        raise ValueError(f"unknown mode: {mode}")
    for layer in model.bert.encoder.layer:
        if mode in {"no_attention_residual", "no_all_residuals"}:
            old_attention = layer.attention
            new_attention = NoAttentionResidual(model.config)
            new_attention.load_state_dict(old_attention.state_dict())
            layer.attention = new_attention
        if mode == "no_all_residuals":
            old_output = layer.output
            new_output = NoFFNResidual(model.config)
            new_output.load_state_dict(old_output.state_dict())
            layer.output = new_output


def build_model(
    config: BertConfig,
    initial_state: Dict[str, torch.Tensor],
    mode: str,
) -> BertForMaskedLM:
    model = BertForMaskedLM(config)
    model.load_state_dict(deepcopy(initial_state), strict=True)
    replace_residuals(model, mode)
    return model


@torch.no_grad()
def evaluate(
    model: BertForMaskedLM,
    examples: Sequence[Dict[str, object]],
    max_length: int,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for step in range(math.ceil(len(examples) / BATCH_SIZE)):
        batch = make_batch(examples, step, min(BATCH_SIZE, len(examples)), max_length)
        input_ids, attention_mask, labels, target_positions = batch
        loss, logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        losses.append(float(loss.item()))
        rows = torch.arange(input_ids.size(0))
        predictions = logits[rows, target_positions].argmax(dim=-1)
        targets = labels[rows, target_positions]
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
    return sum(losses) / len(losses), correct / max(total, 1)


def grad_norm(model: BertForMaskedLM) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(squared)


def run_mode(
    mode: str,
    config: BertConfig,
    initial_state: Dict[str, torch.Tensor],
    train_examples: Sequence[Dict[str, object]],
    valid_examples: Sequence[Dict[str, object]],
    max_length: int,
) -> None:
    model = build_model(config, initial_state, mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    initial_train = evaluate(model, train_examples, max_length)
    initial_valid = evaluate(model, valid_examples, max_length)
    print(
        f"{mode:22s} step=0   train_loss={initial_train[0]:.4f} "
        f"train_top1={initial_train[1]:.2%} valid_loss={initial_valid[0]:.4f} "
        f"valid_top1={initial_valid[1]:.2%}"
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
            print(f"{mode:22s} step={step:<3d} loss=NON_FINITE")
            return
        loss.backward()
        current_grad_norm = grad_norm(model)
        optimizer.step()

        if step in checkpoints:
            train_metrics = evaluate(model, train_examples, max_length)
            valid_metrics = evaluate(model, valid_examples, max_length)
            print(
                f"{mode:22s} step={step:<3d} train_loss={train_metrics[0]:.4f} "
                f"train_top1={train_metrics[1]:.2%} valid_loss={valid_metrics[0]:.4f} "
                f"valid_top1={valid_metrics[1]:.2%} grad_norm={current_grad_norm:.4f}"
            )


def main() -> None:
    seed_everything()
    tokenizer = make_tokenizer()
    train_examples = make_examples(tokenizer, TRAIN_ROWS)
    valid_examples = make_examples(tokenizer, VALID_ROWS)
    max_length = max(
        len(example["input_ids"])
        for example in train_examples + valid_examples
    )
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=max_length + 2,
        position_embedding_type="relative",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    reference = BertForMaskedLM(config)
    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in reference.state_dict().items()
    }
    print(
        f"vocab={len(tokenizer)} max_length={max_length} "
        f"hidden={config.hidden_size} layers={config.num_hidden_layers} "
        f"heads={config.num_attention_heads} steps={STEPS}"
    )
    for mode in ("baseline", "no_attention_residual", "no_all_residuals"):
        run_mode(
            mode,
            config,
            initial_state,
            train_examples,
            valid_examples,
            max_length,
        )


if __name__ == "__main__":
    main()
