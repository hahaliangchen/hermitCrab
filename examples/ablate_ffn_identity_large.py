"""Directly measure the contribution of FFN under dynamic Q/K routing."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as combination_base
import bert_mlm_dynamic_word_spaces as base
import ablate_ffn_variants_large as ffn_variants
import ablate_dynamic_value_large as large_ablation


class ResidualOnlyOutput(nn.Module):
    """The FFN branch is removed; retain only its residual normalization."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        del hidden_states
        return self.LayerNorm(input_tensor)


def _make_identity_model(
    source: base.DynamicWordSpaceBertForMaskedLM,
    tokenizer: combination_base.SimpleBertTokenizer,
) -> base.DynamicWordSpaceBertForMaskedLM:
    target = ffn_variants._build_model(tokenizer)
    ffn_variants._copy_all_parameters(source, target)
    for layer in target.bert.encoder.layer:
        layer.intermediate = nn.Identity()
        layer.output = ResidualOnlyOutput(large_ablation.HIDDEN_SIZE)
    return target


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
    train_samples = large_ablation._make_samples(
        train_texts,
        tokenizer,
        registry,
        attribute_registry,
        seed_offset=1000,
    )
    valid_samples = large_ablation._make_samples(
        valid_texts,
        tokenizer,
        registry,
        attribute_registry,
        seed_offset=2000,
    )

    standard = ffn_variants._build_model(tokenizer)
    identity = _make_identity_model(standard, tokenizer)
    initial = large_ablation._evaluate(standard, valid_samples)
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
    print("standard FFN")
    standard_result = large_ablation._run(
        standard,
        train_samples,
        valid_samples,
    )
    print(
        f"  params={sum(parameter.numel() for parameter in standard.parameters())} "
        f"train_loss={standard_result[0][0]:.4f} "
        f"train_top1={standard_result[0][1]:.1%} "
        f"valid_loss={standard_result[1][0]:.4f} "
        f"valid_top1={standard_result[1][1]:.1%}"
    )
    print("FFN removed, residual + LayerNorm only")
    identity_result = large_ablation._run(
        identity,
        train_samples,
        valid_samples,
    )
    print(
        f"  params={sum(parameter.numel() for parameter in identity.parameters())} "
        f"train_loss={identity_result[0][0]:.4f} "
        f"train_top1={identity_result[0][1]:.1%} "
        f"valid_loss={identity_result[1][0]:.4f} "
        f"valid_top1={identity_result[1][1]:.1%}"
    )


if __name__ == "__main__":
    main()
