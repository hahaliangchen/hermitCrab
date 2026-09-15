"""Scaled local relation adapters for a second, measurable margin test."""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar_attribute_filter import GrammarAttributeFilter
from .local_relation_adapter_model import (
    LocalRelationBilinearAdapter,
    LocalRelationMarginBertForMaskedLM,
)
from .model import BertConfig
from .tokenizer import SimpleBertTokenizer


DEFAULT_RELATION_SCORE_SCALE = 24.0


class ScaledLocalRelationBilinearAdapter(LocalRelationBilinearAdapter):
    """Normalize local candidate keys and expose a useful logit scale."""

    def __init__(
        self,
        relation_triples: Iterable[Sequence[int]],
        hidden_size: int,
        mask_token_id: int,
        relation_score_scale: float = DEFAULT_RELATION_SCORE_SCALE,
    ):
        super().__init__(relation_triples, hidden_size, mask_token_id)
        self.relation_score_scale = float(relation_score_scale)
        if self.relation_score_scale <= 0.0:
            raise ValueError("relation_score_scale must be positive")

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
        relation_triples: Optional[Iterable[Sequence[int]]],
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        vocab_size = int(embedding_weight.shape[0])
        result = hidden_states.new_zeros(
            batch_size, sequence_length, vocab_size
        )
        active_indices = self.indices_for_triples(relation_triples or [])
        if not active_indices:
            return result

        mask = input_ids.eq(self.mask_token_id).reshape(-1)
        if not bool(mask.any()):
            return result
        flat_hidden = hidden_states.reshape(-1, hidden_size)[mask]
        local_scores = flat_hidden.new_zeros(flat_hidden.shape[0], vocab_size)
        candidate_embeddings = embedding_weight.detach()
        route_scale = math.sqrt(float(len(active_indices)))
        for relation_index in active_indices:
            dimensions = self.relation_triples[relation_index].tolist()
            query = flat_hidden[:, dimensions] @ self.relation_matrix[relation_index]
            keys = F.normalize(
                candidate_embeddings[:, dimensions], dim=-1, eps=1e-6
            )
            local_scores = local_scores + (
                query @ keys.transpose(0, 1)
            ) * (self.relation_score_scale / math.sqrt(3.0) / route_scale)

        flat_result = result.reshape(-1, vocab_size)
        flat_result[mask] = local_scores
        return result


class ScaledLocalRelationMarginBertForMaskedLM(
    LocalRelationMarginBertForMaskedLM
):
    """Local relation model with normalized keys and explicit score scale."""

    def __init__(
        self,
        config: BertConfig,
        tokenizer: SimpleBertTokenizer,
        relation_triples: Iterable[Sequence[int]],
        frequency_prior: torch.Tensor,
        frequency_class_scales: torch.Tensor,
        attribute_filter: Optional[GrammarAttributeFilter] = None,
        attribute_bias_scale: float = 1.5,
        constrained_frequency_gate: float = 0.25,
        margin_value: float = 2.0,
        relation_score_scale: float = DEFAULT_RELATION_SCORE_SCALE,
    ):
        super().__init__(
            config,
            tokenizer,
            relation_triples=relation_triples,
            frequency_prior=frequency_prior,
            frequency_class_scales=frequency_class_scales,
            attribute_filter=attribute_filter,
            attribute_bias_scale=attribute_bias_scale,
            constrained_frequency_gate=constrained_frequency_gate,
            margin_value=margin_value,
        )
        triples = self.relation_adapter.relation_triples.detach().cpu().tolist()
        self.relation_adapter = ScaledLocalRelationBilinearAdapter(
            triples,
            hidden_size=config.hidden_size,
            mask_token_id=config.mask_token_id,
            relation_score_scale=relation_score_scale,
        )
        self.relation_score_scale = float(relation_score_scale)

    def save_pretrained(self, save_directory: str):
        super().save_pretrained(save_directory)
        with open(
            os.path.join(save_directory, "local_relation_adapter_v2_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "model_class": self.__class__.__name__,
                    "candidate_key_normalization": "l2",
                    "relation_score_scale": self.relation_score_scale,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
    ) -> "ScaledLocalRelationMarginBertForMaskedLM":
        config = BertConfig.from_json_file(f"{load_directory}/config.json")
        tokenizer = SimpleBertTokenizer.from_pretrained(load_directory)
        with open(
            f"{load_directory}/local_relation_adapter_config.json",
            "r",
            encoding="utf-8",
        ) as handle:
            adapter_config = json.load(handle)
        v2_config_path = os.path.join(
            load_directory, "local_relation_adapter_v2_config.json"
        )
        v2_config = {}
        if os.path.isfile(v2_config_path):
            with open(v2_config_path, "r", encoding="utf-8") as handle:
                v2_config = json.load(handle)
        state_dict = torch.load(
            f"{load_directory}/pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        prior_config_path = os.path.join(
            load_directory, "prior_controlled_model_config.json"
        )
        prior_config = {}
        if os.path.isfile(prior_config_path):
            with open(prior_config_path, "r", encoding="utf-8") as handle:
                prior_config = json.load(handle)
        model = cls(
            config,
            tokenizer,
            relation_triples=adapter_config["relation_triples"],
            frequency_prior=state_dict["frequency_prior"],
            frequency_class_scales=state_dict["frequency_class_scales"],
            attribute_filter=GrammarAttributeFilter(tokenizer),
            attribute_bias_scale=float(
                prior_config.get("attribute_bias_scale", 1.5)
            ),
            constrained_frequency_gate=float(
                prior_config.get("constrained_frequency_gate", 0.25)
            ),
            margin_value=float(adapter_config.get("margin_value", 2.0)),
            relation_score_scale=float(
                v2_config.get(
                    "relation_score_scale",
                    adapter_config.get(
                        "relation_score_scale", DEFAULT_RELATION_SCORE_SCALE
                    ),
                )
            ),
        )
        model.load_state_dict(state_dict, strict=True)
        model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
        return model
