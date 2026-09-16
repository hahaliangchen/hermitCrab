"""Compare dynamic 3D Q/K routing with an additional dynamic 3D V path.

The existing experiment uses routed 3D Q/K vectors to add a local attention
score.  This file keeps that path unchanged and adds an optional routed 3D V
payload: each source token mixes its space-specific 3D values, projects the
result back to the ordinary attention-head width, and adds it to standard V.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as combination_base
import bert_mlm_dynamic_word_spaces as base


TEXTS = [
    "汉王 立 张耳 为 赵王",
    "汉王 立 韩信 为 齐王",
    "高祖 封 萧何 为 相国",
    "高祖 封 曹参 为 平阳侯",
    "项王 拜 英布 为 九江王",
    "赵王 任 陈馀 为 将军",
    "汉王 召 陈平 为 都尉",
    "高祖 使 樊哙 守 关中",
]


class DynamicValueWordSpaceBank(base.DynamicWordSpaceBank):
    """The original Q/K bank plus a routed 3D V payload."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_spaces: int,
        hidden_size: int,
        route_dim: int,
        space_dim: int = 3,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            num_heads=num_heads,
            max_spaces=max_spaces,
            hidden_size=hidden_size,
            route_dim=route_dim,
            space_dim=space_dim,
        )
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.dynamic_value = True
        self.head_dim = hidden_size // num_heads
        self.p_v = nn.Parameter(
            torch.empty(num_layers, num_heads, max_spaces, hidden_size, space_dim)
        )
        self.a_v = nn.Parameter(
            torch.empty(num_layers, num_heads, max_spaces, space_dim, space_dim)
        )
        self.v_out = nn.Parameter(
            torch.empty(num_layers, num_heads, space_dim, self.head_dim)
        )
        self._reset_value_parameters()

    def _reset_value_parameters(self) -> None:
        with torch.no_grad():
            self.p_v.normal_(mean=0.0, std=0.02)
            eye = torch.eye(
                self.space_dim,
                device=self.a_v.device,
                dtype=self.a_v.dtype,
            ).view(1, 1, 1, self.space_dim, self.space_dim)
            self.a_v.copy_(eye.expand_as(self.a_v))
            self.a_v.add_(torch.randn_like(self.a_v) * 0.01)
            # This makes the two models identical at step zero.  The V path
            # becomes active as v_out learns from the MLM loss.
            self.v_out.zero_()

    def initialize_slot(self, slot_id: int, source_slot: int = 0) -> None:
        super().initialize_slot(slot_id, source_slot=source_slot)
        with torch.no_grad():
            self.p_v[:, :, slot_id].copy_(
                self.p_v[:, :, source_slot]
                + torch.randn_like(self.p_v[:, :, source_slot]) * 0.005
            )
            self.a_v[:, :, slot_id].copy_(
                self.a_v[:, :, source_slot]
                + torch.randn_like(self.a_v[:, :, source_slot]) * 0.005
            )

    def local_values(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return routed values with shape ``[B,T,heads,head_dim]``."""

        p_v = self.p_v[layer_index]
        u_v = torch.einsum("btd,hsdk->bthsk", hidden_states, p_v)
        a_v = self.a_v[layer_index]
        values = torch.einsum("bthsk,hskm->bthsm", u_v, a_v)
        weights = route_weights.to(dtype=values.dtype)
        mixed_values = torch.einsum("bthsd,bts->bthd", values, weights)
        return torch.einsum(
            "bthd,hde->bthe",
            mixed_values,
            self.v_out[layer_index],
        )


class DynamicValueWordSpaceSelfAttention(base.WordSpaceSelfAttention):
    """Existing dynamic Q/K attention with a routed dynamic V addition."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._transpose_for_scores(query)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / self.head_dim**0.5
        relative_bias = base._build_relative_position_bias(
            self.relative_position_bias,
            self.relative_position_max_distance,
            hidden_states.size(1),
            hidden_states.device,
            attn_scores.dtype,
        )
        if relative_bias is not None:
            attn_scores = attn_scores + relative_bias
        if route_weights is not None:
            local_scores = self.bank.local_scores(
                self.layer_index,
                hidden_states,
                route_weights,
            )
            attn_scores = attn_scores + self.local_score_scale * local_scores
            dynamic_value = self.bank.local_values(
                self.layer_index,
                hidden_states,
                route_weights,
            )
            value = value + dynamic_value.permute(0, 2, 1, 3).contiguous()
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:2], self.all_head_size)
        return self.out_proj(context), attn_probs


class DynamicValueWordSpaceBertForMaskedLM(
    base.DynamicWordSpaceBertForMaskedLM
):
    """Model wrapper that replaces only the dynamic attention implementation."""

    def __init__(
        self,
        config: combination_base.BertConfig,
        bank: DynamicValueWordSpaceBank,
        local_score_scale: float = 0.5,
        route_start_layer: int = 1,
        route_dim: int = 32,
    ) -> None:
        super().__init__(
            config,
            bank=bank,
            local_score_scale=local_score_scale,
            route_start_layer=route_start_layer,
            route_dim=route_dim,
        )
        for layer_index in range(route_start_layer, config.num_hidden_layers):
            layer = self.bert.encoder.layer[layer_index]
            layer.attention.self_attn = DynamicValueWordSpaceSelfAttention(
                layer.attention.self_attn,
                bank,
                layer_index,
                local_score_scale,
            )


def _make_samples(
    tokenizer: combination_base.SimpleBertTokenizer,
    registry: base.WordSpaceRegistry,
    attribute_registry: base.WordAttributeRegistry,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for index, text in enumerate(TEXTS):
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=10,
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )
        sample = combination_base._make_masked_example(
            encoded,
            tokenizer,
            seed=1000 + index,
            mlm_probability=0.35,
        )
        sample["touched_token_ids"] = sorted(registry.touched_token_ids(sample))
        sample["candidate_space_mask"] = registry.candidate_mask(sample)
        sample["grammar_allowed_mask"] = attribute_registry.allowed_mask(
            sample["input_ids"]
        )
        samples.append(sample)
    return samples


def _build_model(
    tokenizer: combination_base.SimpleBertTokenizer,
    dynamic_value: bool,
) -> base.DynamicWordSpaceBertForMaskedLM:
    config = combination_base.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=20,
        position_embedding_type="relative",
        relative_position_max_distance=10,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    if dynamic_value:
        bank: base.DynamicWordSpaceBank = DynamicValueWordSpaceBank(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            max_spaces=4,
            hidden_size=config.hidden_size,
            route_dim=16,
        )
        return DynamicValueWordSpaceBertForMaskedLM(config, bank=bank)
    bank = base.DynamicWordSpaceBank(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        max_spaces=4,
        hidden_size=config.hidden_size,
        route_dim=16,
    )
    return base.DynamicWordSpaceBertForMaskedLM(config, bank=bank)


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
    return sum(losses) / len(losses), correct / max(total, 1)


def _copy_common_parameters(
    source: nn.Module,
    target: nn.Module,
) -> None:
    """Make the V model share every ordinary parameter with the baseline."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)
    target.load_state_dict(target_state, strict=True)


def _run(
    model: base.DynamicWordSpaceBertForMaskedLM,
    samples: Sequence[Dict[str, object]],
    steps: int = 240,
) -> Tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for step in range(1, steps + 1):
        sample = samples[(step - 1) % len(samples)]
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
    return _evaluate(model, samples)


def main() -> None:
    combination_base.set_seed(42)
    tokenizer = combination_base.SimpleBertTokenizer()
    tokenizer.train_from_texts(TEXTS, min_freq=1)
    registry = base.WordSpaceRegistry(tokenizer, max_spaces=4, initial_spaces=4)
    attribute_registry = base.WordAttributeRegistry(tokenizer)
    samples = _make_samples(tokenizer, registry, attribute_registry)

    baseline = _build_model(tokenizer, dynamic_value=False)
    dynamic_v = _build_model(tokenizer, dynamic_value=True)
    _copy_common_parameters(baseline, dynamic_v)

    initial_base = _evaluate(baseline, samples)
    initial_v = _evaluate(dynamic_v, samples)
    baseline_result = _run(baseline, samples)
    dynamic_v_result = _run(dynamic_v, samples)

    print(
        f"vocab={len(tokenizer)} samples={len(samples)} hidden=64 "
        "layers=4 heads=4 steps=240"
    )
    print(
        f"initial ordinary_qk loss={initial_base[0]:.4f} "
        f"top1={initial_base[1]:.1%}"
    )
    print(
        f"initial dynamic_v loss={initial_v[0]:.4f} "
        f"top1={initial_v[1]:.1%}"
    )
    print(
        f"ordinary_qk params={sum(p.numel() for p in baseline.parameters())} "
        f"final_loss={baseline_result[0]:.4f} top1={baseline_result[1]:.1%}"
    )
    print(
        f"dynamic_v params={sum(p.numel() for p in dynamic_v.parameters())} "
        f"final_loss={dynamic_v_result[0]:.4f} top1={dynamic_v_result[1]:.1%}"
    )
    bank = dynamic_v.space_bank
    print(
        f"dynamic_v p_v_norm={float(bank.p_v.norm().item()):.4f} "
        f"a_v_norm={float(bank.a_v.norm().item()):.4f} "
        f"v_out_norm={float(bank.v_out.norm().item()):.4f}"
    )


if __name__ == "__main__":
    main()
