"""Context-routed local relation adapters for fact-memory experiments.

The base BERT remains the shared language model.  Each observed relation
triple owns a small 3x3 bilinear matrix.  A sentence may activate several of
those matrices; their local scores are added only at mask positions.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar_attribute_filter import GrammarAttributeFilter
from .grammar_attribute_model import grammar_attribute_bias
from .model import BertConfig
from .prior_controlled_model import (
    PriorControlledGrammarAttributeBertForMaskedLM,
)
from .tokenizer import SimpleBertTokenizer


Triple = Tuple[int, int, int]


def _normalize_triple(triple: Sequence[int]) -> Triple:
    values = tuple(sorted(int(value) for value in triple))
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError(f"relation triple must contain three distinct dimensions: {triple}")
    return values  # type: ignore[return-value]


class LocalRelationBilinearAdapter(nn.Module):
    """A bank of lazily-routed 3x3 local relation matrices."""

    def __init__(
        self,
        relation_triples: Iterable[Sequence[int]],
        hidden_size: int,
        mask_token_id: int,
    ):
        super().__init__()
        normalized = [_normalize_triple(triple) for triple in relation_triples]
        if len(set(normalized)) != len(normalized):
            raise ValueError("relation_triples must be unique")
        if any(
            dimension < 0 or dimension >= hidden_size
            for triple in normalized
            for dimension in triple
        ):
            raise ValueError("relation triple dimension is outside hidden_size")
        triple_tensor = torch.tensor(normalized, dtype=torch.long)
        if not normalized:
            triple_tensor = triple_tensor.reshape(0, 3)
        self.register_buffer("relation_triples", triple_tensor)
        self.relation_matrix = nn.Parameter(
            torch.zeros(len(normalized), 3, 3, dtype=torch.float32)
        )
        self.hidden_size = int(hidden_size)
        self.mask_token_id = int(mask_token_id)
        self._relation_to_index = {
            triple: index for index, triple in enumerate(normalized)
        }

    @property
    def relation_count(self) -> int:
        return int(self.relation_triples.shape[0])

    def indices_for_triples(self, triples: Iterable[Sequence[int]]) -> List[int]:
        indices = set()
        for triple in triples:
            normalized = _normalize_triple(triple)
            if normalized in self._relation_to_index:
                indices.add(self._relation_to_index[normalized])
        return sorted(indices)

    def mask_gradients(
        self,
        gradients: Sequence[torch.Tensor],
        active_indices: Sequence[int],
    ) -> List[torch.Tensor]:
        """Keep only the relation rows active for this update.

        The base adapter has one relation-scoped parameter.  Dynamic Q/K
        adapters override this hook because their layer/head matrices use a
        different relation axis and also contain shared router parameters.
        """
        if len(gradients) != 1:
            raise ValueError("base relation adapter expects one gradient tensor")
        gradient = gradients[0].detach().clone()
        mask = torch.zeros(
            gradient.shape[0], dtype=gradient.dtype, device=gradient.device
        )
        if active_indices:
            mask[list(active_indices)] = 1.0
        return [gradient * mask.view(-1, 1, 1)]

    def relation_gradient_vector(
        self,
        gradients: Sequence[torch.Tensor],
        relation_index: int,
    ) -> torch.Tensor:
        if len(gradients) != 1:
            raise ValueError("base relation adapter expects one gradient tensor")
        return gradients[0][int(relation_index)].reshape(-1)

    def scale_relation_gradient(
        self,
        gradients: Sequence[torch.Tensor],
        relation_index: int,
        scale: float,
    ) -> List[torch.Tensor]:
        result = [gradient.detach().clone() for gradient in gradients]
        result[0][int(relation_index)] *= float(scale)
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
        relation_triples: Optional[Iterable[Sequence[int]]],
    ) -> torch.Tensor:
        """Return local vocabulary scores at the input's mask positions.

        For relation ``r`` with selected dimensions ``I_r``, the local score
        is ``(h[I_r] A_r) · e_token[I_r]``.  The candidate embedding is
        detached so this branch does not directly rewrite every vocabulary
        row; the relation matrix and the contextual route learn instead.
        """
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
            keys = candidate_embeddings[:, dimensions]
            local_scores = local_scores + (
                query @ keys.transpose(0, 1)
            ) / math.sqrt(3.0) / route_scale

        flat_result = result.reshape(-1, vocab_size)
        flat_result[mask] = local_scores
        return result


class LocalRelationMarginBertForMaskedLM(
    PriorControlledGrammarAttributeBertForMaskedLM
):
    """Prior-controlled MLM with context-routed local relation scoring."""

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
    ):
        super().__init__(
            config,
            tokenizer,
            frequency_prior=frequency_prior,
            frequency_class_scales=frequency_class_scales,
            attribute_filter=attribute_filter,
            attribute_bias_scale=attribute_bias_scale,
            constrained_frequency_gate=constrained_frequency_gate,
        )
        self.relation_adapter = LocalRelationBilinearAdapter(
            relation_triples,
            hidden_size=config.hidden_size,
            mask_token_id=config.mask_token_id,
        )
        self.margin_value = float(margin_value)

    def logits_components(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        apply_grammar_attributes: bool = True,
        apply_frequency_prior: bool = True,
        relation_triples: Optional[Iterable[Sequence[int]]] = None,
        candidate_space_mask: Optional[torch.Tensor] = None,
    ):
        relation_triples = list(relation_triples or [])
        sequence_output, _, all_attentions = self._encode_sequence(
            input_ids,
            token_type_ids,
            attention_mask,
            output_attentions=output_attentions,
            relation_triples=relation_triples,
            candidate_space_mask=candidate_space_mask,
        )
        base_logits = self.lm_head(sequence_output)
        local_relation_bias = self.relation_adapter(
            sequence_output,
            input_ids,
            self.bert.embeddings.word_embeddings.weight,
            relation_triples,
        )
        raw_logits = base_logits + local_relation_bias

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
        grammar_info["active_relation_triples"] = [
            list(triple)
            for triple in (relation_triples or [])
        ]
        return (
            base_logits,
            local_relation_bias,
            grammar_bias,
            frequency_bias,
            logits,
            all_attentions,
            grammar_info,
        )

    def _encode_sequence(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relation_triples: Optional[Sequence[Sequence[int]]] = None,
        candidate_space_mask: Optional[torch.Tensor] = None,
    ):
        """Encode the sequence; dynamic attention subclasses override this."""
        del relation_triples, candidate_space_mask
        return self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_attentions=output_attentions,
        )

    @staticmethod
    def _margin_loss(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_id: Optional[int],
        hard_negative_ids: Optional[Sequence[int]],
        mask_token_id: int,
        margin_value: float,
    ) -> torch.Tensor:
        if target_id is None:
            return logits.sum() * 0.0
        positions = (input_ids == mask_token_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            return logits.sum() * 0.0
        batch_index, position = positions[0].tolist()
        scores = logits[batch_index, position]
        target_id = int(target_id)
        if target_id < 0 or target_id >= scores.numel():
            return logits.sum() * 0.0

        candidates = {
            int(candidate)
            for candidate in (hard_negative_ids or [])
            if 0 <= int(candidate) < scores.numel() and int(candidate) != target_id
        }
        top_k = min(8, scores.numel())
        candidates.update(
            int(candidate)
            for candidate in torch.topk(scores.detach(), top_k).indices.tolist()
            if int(candidate) != target_id
        )
        if not candidates:
            return logits.sum() * 0.0
        negative_scores = scores[sorted(candidates)]
        return F.softplus(
            float(margin_value) - scores[target_id] + negative_scores
        ).mean()

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        apply_grammar_attributes: bool = True,
        apply_frequency_prior: bool = True,
        relation_triples: Optional[Iterable[Sequence[int]]] = None,
        candidate_space_mask: Optional[torch.Tensor] = None,
        margin_target_id: Optional[int] = None,
        hard_negative_ids: Optional[Sequence[int]] = None,
        margin_weight: float = 0.0,
        margin_value: Optional[float] = None,
        return_grammar_info: bool = False,
        return_component_info: bool = False,
        return_route_weights: bool = False,
        **kwargs,
    ):
        (
            base_logits,
            local_relation_bias,
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
            relation_triples=relation_triples,
            candidate_space_mask=candidate_space_mask,
        )

        margin_loss = logits.sum() * 0.0
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            if margin_target_id is None:
                valid_labels = labels[labels != -100]
                if valid_labels.numel() == 1:
                    margin_target_id = int(valid_labels[0].item())
            margin_loss = self._margin_loss(
                logits,
                input_ids,
                margin_target_id,
                hard_negative_ids,
                self.config.mask_token_id,
                self.margin_value if margin_value is None else margin_value,
            )
            loss = ce_loss + float(margin_weight) * margin_loss
            result = (loss, logits, all_attentions)
        else:
            ce_loss = logits.sum() * 0.0
            result = (logits, all_attentions)

        grammar_info = dict(grammar_info)
        grammar_info["ce_loss"] = ce_loss.detach()
        grammar_info["margin_loss"] = margin_loss.detach()
        if return_grammar_info:
            result = result + (grammar_info,)
        if return_route_weights:
            result = result + (getattr(self, "_last_route_weights", None),)
        if return_component_info:
            result = result + (
                {
                    "base_logits": base_logits,
                    "local_relation_bias": local_relation_bias,
                    "raw_logits": base_logits + local_relation_bias,
                    "grammar_bias": grammar_bias,
                    "frequency_bias": frequency_bias,
                    "logits": logits,
                    "margin_loss": margin_loss,
                },
            )
        return result

    def save_pretrained(self, save_directory: str):
        super().save_pretrained(save_directory)
        with open(
            os.path.join(save_directory, "local_relation_adapter_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            adapter_config = {
                "model_class": self.__class__.__name__,
                "adapter_type": "3x3_bilinear_relation_matrix",
                "hidden_size": self.config.hidden_size,
                "relation_count": self.relation_adapter.relation_count,
                "relation_triples": [
                    triple.tolist()
                    for triple in self.relation_adapter.relation_triples
                ],
                "margin_value": self.margin_value,
            }
            if hasattr(self, "relation_score_scale"):
                adapter_config["relation_score_scale"] = float(
                    self.relation_score_scale
                )
            json.dump(
                adapter_config,
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
    ) -> "LocalRelationMarginBertForMaskedLM":
        config = BertConfig.from_json_file(f"{load_directory}/config.json")
        tokenizer = SimpleBertTokenizer.from_pretrained(load_directory)
        with open(
            f"{load_directory}/local_relation_adapter_config.json",
            "r",
            encoding="utf-8",
        ) as handle:
            adapter_config = json.load(handle)
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
        frequency_prior = state_dict["frequency_prior"]
        frequency_class_scales = state_dict["frequency_class_scales"]
        model = cls(
            config,
            tokenizer,
            relation_triples=adapter_config["relation_triples"],
            frequency_prior=frequency_prior,
            frequency_class_scales=frequency_class_scales,
            attribute_filter=GrammarAttributeFilter(tokenizer),
            attribute_bias_scale=float(
                prior_config.get("attribute_bias_scale", 1.5)
            ),
            constrained_frequency_gate=float(
                prior_config.get("constrained_frequency_gate", 0.25)
            ),
            margin_value=float(adapter_config.get("margin_value", 2.0)),
        )
        model.load_state_dict(state_dict, strict=True)
        model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
        return model
