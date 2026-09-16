"""Compare FFN variants while keeping dynamic Q/K and data fixed."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as combination_base
import bert_mlm_dynamic_word_spaces as base
import ablate_dynamic_value_large as large_ablation


VARIANTS = ("standard", "narrow", "linear", "parallel")
NARROW_SIZE = 128


class NarrowIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class NarrowOutput(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.0)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states) + input_tensor
        return self.LayerNorm(hidden_states)


class ParallelIntermediate(nn.Module):
    """Two independent half-width branches concatenated to the old width."""

    def __init__(self, hidden_size: int, branch_size: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Linear(hidden_size, branch_size) for _ in range(2)]
        )
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.act(branch(hidden_states)) for branch in self.branches],
            dim=-1,
        )


def _build_model(
    tokenizer: combination_base.SimpleBertTokenizer,
) -> base.DynamicWordSpaceBertForMaskedLM:
    config = combination_base.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=large_ablation.HIDDEN_SIZE,
        num_hidden_layers=large_ablation.NUM_LAYERS,
        num_attention_heads=large_ablation.NUM_HEADS,
        intermediate_size=large_ablation.HIDDEN_SIZE * 4,
        max_position_embeddings=large_ablation.MAX_LENGTH + 10,
        position_embedding_type="relative",
        relative_position_max_distance=large_ablation.MAX_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    bank = base.DynamicWordSpaceBank(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        max_spaces=8,
        hidden_size=config.hidden_size,
        route_dim=16,
    )
    return base.DynamicWordSpaceBertForMaskedLM(
        config,
        bank=bank,
        route_start_layer=1,
        route_dim=16,
    )


def _copy_all_parameters(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)
    target.load_state_dict(target_state, strict=True)


def _copy_non_ffn_parameters(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        is_ffn_weight = ".intermediate." in name or ".output.dense." in name
        if (
            not is_ffn_weight
            and name in target_state
            and target_state[name].shape == value.shape
        ):
            target_state[name].copy_(value)
    target.load_state_dict(target_state, strict=True)


def _make_variant(
    source: base.DynamicWordSpaceBertForMaskedLM,
    tokenizer: combination_base.SimpleBertTokenizer,
    variant: str,
) -> base.DynamicWordSpaceBertForMaskedLM:
    target = _build_model(tokenizer)
    if variant == "standard":
        _copy_all_parameters(source, target)
        return target
    if variant == "linear":
        _copy_all_parameters(source, target)
        for layer in target.bert.encoder.layer:
            layer.intermediate.act = nn.Identity()
        return target

    _copy_non_ffn_parameters(source, target)
    hidden_size = large_ablation.HIDDEN_SIZE
    full_intermediate = hidden_size * 4
    for source_layer, target_layer in zip(
        source.bert.encoder.layer,
        target.bert.encoder.layer,
    ):
        if variant == "narrow":
            target_layer.intermediate = NarrowIntermediate(
                hidden_size,
                NARROW_SIZE,
            )
            target_layer.output = NarrowOutput(hidden_size, NARROW_SIZE)
            with torch.no_grad():
                target_layer.intermediate.dense.weight.copy_(
                    source_layer.intermediate.dense.weight[:NARROW_SIZE]
                )
                target_layer.intermediate.dense.bias.copy_(
                    source_layer.intermediate.dense.bias[:NARROW_SIZE]
                )
                target_layer.output.dense.weight.copy_(
                    source_layer.output.dense.weight[:, :NARROW_SIZE]
                )
                target_layer.output.dense.bias.copy_(
                    source_layer.output.dense.bias
                )
        elif variant == "parallel":
            target_layer.intermediate = ParallelIntermediate(
                hidden_size,
                NARROW_SIZE,
            )
            with torch.no_grad():
                for branch_index, branch in enumerate(
                    target_layer.intermediate.branches
                ):
                    start = branch_index * NARROW_SIZE
                    end = start + NARROW_SIZE
                    branch.weight.copy_(
                        source_layer.intermediate.dense.weight[start:end]
                    )
                    branch.bias.copy_(
                        source_layer.intermediate.dense.bias[start:end]
                    )
                target_layer.output.dense.weight.copy_(
                    source_layer.output.dense.weight
                )
                target_layer.output.dense.bias.copy_(
                    source_layer.output.dense.bias
                )
        else:
            raise ValueError(f"unknown FFN variant: {variant}")
    return target


def _make_samples(
    texts: Sequence[str],
    tokenizer: combination_base.SimpleBertTokenizer,
    registry: base.WordSpaceRegistry,
    attribute_registry: base.WordAttributeRegistry,
    seed_offset: int,
) -> List[Dict[str, object]]:
    return large_ablation._make_samples(
        texts,
        tokenizer,
        registry,
        attribute_registry,
        seed_offset,
    )


def main() -> None:
    combination_base.set_seed(42)
    texts = combination_base.load_texts(
        large_ablation.DATA_PATH,
        max_sentences=100,
    )
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

    source = _build_model(tokenizer)
    models = {
        variant: source
        if variant == "standard"
        else _make_variant(source, tokenizer, variant)
        for variant in VARIANTS
    }
    initial = large_ablation._evaluate(models["standard"], valid_samples)
    print(
        f"sentences={len(texts)} train={len(train_samples)} "
        f"valid={len(valid_samples)} vocab={len(tokenizer)} "
        f"hidden={large_ablation.HIDDEN_SIZE} layers={large_ablation.NUM_LAYERS} "
        f"heads={large_ablation.NUM_HEADS} epochs={large_ablation.EPOCHS}"
    )
    print(
        f"initial valid_loss={initial[0]:.4f} "
        f"valid_top1={initial[1]:.1%}"
    )

    for variant, model in models.items():
        print(
            f"{variant} params={sum(parameter.numel() for parameter in model.parameters())}"
        )
        train_result, valid_result = large_ablation._run(
            model,
            train_samples,
            valid_samples,
        )
        print(
            f"  train_loss={train_result[0]:.4f} "
            f"train_top1={train_result[1]:.1%} "
            f"valid_loss={valid_result[0]:.4f} "
            f"valid_top1={valid_result[1]:.1%}"
        )


if __name__ == "__main__":
    main()
