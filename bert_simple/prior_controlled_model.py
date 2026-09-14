"""Grammar/fact MLM with an explicitly low-weight frequency fallback."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .grammar_attribute_filter import GrammarAttributeFilter
from .grammar_attribute_model import (
    GrammarAttributeBertForMaskedLM,
    grammar_attribute_bias,
)
from .model import BertConfig
from .tokenizer import SimpleBertTokenizer


class PriorControlledGrammarAttributeBertForMaskedLM(
    GrammarAttributeBertForMaskedLM
):
    """Add a small, explicit frequency prior after grammar/attribute logits.

    The frequency prior is a frozen fallback vector derived from the training
    corpus.  Keeping it separate makes its contribution measurable and avoids
    confusing a token's input embedding with its frequency preference.  The
    prior is weaker in recognized grammar states and weakest for punctuation
    and clear function words.
    """

    def __init__(
        self,
        config: BertConfig,
        tokenizer: SimpleBertTokenizer,
        frequency_prior: torch.Tensor,
        frequency_class_scales: torch.Tensor,
        attribute_filter: Optional[GrammarAttributeFilter] = None,
        attribute_bias_scale: float = 1.5,
        constrained_frequency_gate: float = 0.25,
    ):
        super().__init__(
            config,
            tokenizer,
            attribute_filter=attribute_filter,
            attribute_bias_scale=attribute_bias_scale,
        )
        if frequency_prior.ndim != 1 or frequency_prior.numel() != len(tokenizer):
            raise ValueError("frequency_prior must have shape [vocab_size]")
        if (
            frequency_class_scales.ndim != 1
            or frequency_class_scales.numel() != len(tokenizer)
        ):
            raise ValueError("frequency_class_scales must have shape [vocab_size]")
        self.register_buffer("frequency_prior", frequency_prior.detach().float())
        self.register_buffer(
            "frequency_class_scales",
            frequency_class_scales.detach().float(),
        )
        self.constrained_frequency_gate = float(constrained_frequency_gate)

    def _fallback_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return ``True`` for mask positions where grammar abstains."""
        tokenizer = self.grammar_attribute_filter.tokenizer
        vocab_size = len(tokenizer)
        fallback = torch.zeros_like(input_ids, dtype=torch.bool)
        for batch_index, row in enumerate(input_ids.detach().cpu().tolist()):
            tokens = [
                tokenizer.id_to_token[int(token_id)]
                if 0 <= int(token_id) < vocab_size
                else tokenizer.unk_token
                for token_id in row
            ]
            for position, token_id in enumerate(row):
                if int(token_id) != tokenizer.mask_token_id:
                    continue
                state = self.grammar_attribute_filter.automaton.analyze(
                    tokens, position
                )
                fallback[batch_index, position] = bool(state.fallback)
        return fallback.to(input_ids.device)

    def frequency_bias(
        self,
        input_ids: torch.Tensor,
        fallback_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the low-weight frequency fallback at actual mask slots."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        base = self.frequency_prior * self.frequency_class_scales
        values = base.view(1, 1, -1).expand(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        if fallback_mask is not None:
            gate = torch.where(
                fallback_mask,
                torch.ones_like(fallback_mask, dtype=values.dtype),
                torch.full_like(
                    fallback_mask,
                    self.constrained_frequency_gate,
                    dtype=values.dtype,
                ),
            )
            values = values * gate.unsqueeze(-1)
        mask = input_ids.eq(self.config.mask_token_id).unsqueeze(-1)
        return torch.where(mask, values, torch.zeros_like(values))

    def logits_components(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        apply_grammar_attributes: bool = True,
        apply_frequency_prior: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, ...]],
        Dict[str, object],
    ]:
        sequence_output, _, all_attentions = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_attentions=output_attentions,
        )
        raw_logits = self.lm_head(sequence_output)
        grammar_bias, grammar_info = grammar_attribute_bias(
            self.grammar_attribute_filter,
            input_ids,
            scale=self.attribute_bias_scale,
        )
        fallback_mask = self._fallback_mask(input_ids)
        frequency_bias = self.frequency_bias(input_ids, fallback_mask)

        logits = raw_logits
        if apply_grammar_attributes:
            logits = logits + grammar_bias.to(dtype=logits.dtype)
        if apply_frequency_prior:
            logits = logits + frequency_bias.to(dtype=logits.dtype)
        grammar_info = dict(grammar_info)
        grammar_info["fallback_mask"] = fallback_mask
        return (
            raw_logits,
            grammar_bias,
            frequency_bias,
            logits,
            all_attentions,
            grammar_info,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        apply_grammar_attributes: bool = True,
        apply_frequency_prior: bool = True,
        return_grammar_info: bool = False,
        return_component_info: bool = False,
        **kwargs,
    ):
        (
            raw_logits,
            grammar_bias,
            frequency_bias,
            logits,
            all_attentions,
            grammar_info,
        ) = self.logits_components(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            apply_grammar_attributes=apply_grammar_attributes,
            apply_frequency_prior=apply_frequency_prior,
        )
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            result = (loss, logits, all_attentions)
        else:
            result = (logits, all_attentions)
        if return_grammar_info:
            result = result + (grammar_info,)
        if return_component_info:
            result = result + (
                {
                    "raw_logits": raw_logits,
                    "grammar_bias": grammar_bias,
                    "frequency_bias": frequency_bias,
                    "logits": logits,
                },
            )
        return result

    def save_pretrained(self, save_directory: str):
        super().save_pretrained(save_directory)
        with open(
            os.path.join(save_directory, "prior_controlled_model_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "model_class": self.__class__.__name__,
                    "attribute_bias_scale": self.attribute_bias_scale,
                    "constrained_frequency_gate": self.constrained_frequency_gate,
                    "frequency_prior_frozen": True,
                    "frequency_prior_role": "low_weight_fallback",
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        attribute_bias_scale: Optional[float] = None,
    ) -> "PriorControlledGrammarAttributeBertForMaskedLM":
        config = BertConfig.from_json_file(f"{load_directory}/config.json")
        tokenizer = SimpleBertTokenizer.from_pretrained(load_directory)
        state_dict = torch.load(
            f"{load_directory}/pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        frequency_prior = state_dict.get(
            "frequency_prior", torch.zeros(len(tokenizer))
        )
        frequency_class_scales = state_dict.get(
            "frequency_class_scales", torch.zeros(len(tokenizer))
        )
        model_config_path = os.path.join(
            load_directory, "prior_controlled_model_config.json"
        )
        metadata = {}
        if os.path.isfile(model_config_path):
            with open(model_config_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        model = cls(
            config,
            tokenizer,
            frequency_prior=frequency_prior,
            frequency_class_scales=frequency_class_scales,
            attribute_filter=GrammarAttributeFilter(tokenizer),
            attribute_bias_scale=float(
                attribute_bias_scale
                if attribute_bias_scale is not None
                else metadata.get("attribute_bias_scale", 1.5)
            ),
            constrained_frequency_gate=float(
                metadata.get("constrained_frequency_gate", 0.25)
            ),
        )
        model.load_state_dict(state_dict, strict=True)
        model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
        return model
