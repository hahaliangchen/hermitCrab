"""BERT MLM integration for fixed grammar and word attributes."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .grammar_attribute_filter import GrammarAttributeFilter
from .model import BertConfig, BertForMaskedLM
from .tokenizer import SimpleBertTokenizer


@torch.no_grad()
def grammar_attribute_bias(
    attribute_filter: GrammarAttributeFilter,
    input_ids: torch.Tensor,
    scale: float = 1.5,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return signed soft biases for real mask slots."""
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")

    batch_size, seq_len = input_ids.shape
    tokenizer = attribute_filter.tokenizer
    vocab_size = len(tokenizer)
    parameter = next(attribute_filter.parameters())
    parameter_device = parameter.device
    biases = torch.zeros(
        batch_size,
        seq_len,
        vocab_size,
        dtype=parameter.dtype,
        device=parameter_device,
    )
    constrained_positions = 0
    fallback_positions = 0
    structures: List[str] = []

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
            state = attribute_filter.automaton.analyze(tokens, position)
            structures.append(state.structure)
            if state.fallback:
                fallback_positions += 1
            else:
                constrained_positions += 1
            biases[batch_index, position] = attribute_filter.bias_for_state(
                state,
                tokenizer.id_to_token,
                scale=scale,
            )

    if biases.device != input_ids.device:
        biases = biases.to(input_ids.device)
    return biases, {
        "constrained_positions": constrained_positions,
        "fallback_positions": fallback_positions,
        "structures": structures,
    }


class GrammarAttributeBertForMaskedLM(BertForMaskedLM):
    """BERT MLM whose logits receive grammar/attribute soft bias."""

    def __init__(
        self,
        config: BertConfig,
        tokenizer: SimpleBertTokenizer,
        attribute_filter: Optional[GrammarAttributeFilter] = None,
        attribute_bias_scale: float = 1.5,
    ):
        super().__init__(config)
        if len(tokenizer) != config.vocab_size:
            raise ValueError("tokenizer size must equal config.vocab_size")
        self.grammar_attribute_filter = attribute_filter or GrammarAttributeFilter(
            tokenizer
        )
        self.attribute_bias_scale = float(attribute_bias_scale)

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        apply_grammar_attributes: bool = True,
        return_grammar_info: bool = False,
        **kwargs,
    ):
        sequence_output, _, all_attentions = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_attentions=output_attentions,
        )
        logits = self.lm_head(sequence_output)
        grammar_info: Dict[str, object] = {
            "constrained_positions": 0,
            "fallback_positions": 0,
            "structures": [],
        }
        if apply_grammar_attributes:
            bias, grammar_info = grammar_attribute_bias(
                self.grammar_attribute_filter,
                input_ids,
                scale=self.attribute_bias_scale,
            )
            logits = logits + bias.to(dtype=logits.dtype)
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
            return result + (grammar_info,)
        return result

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        attribute_bias_scale: float = 1.5,
    ) -> "GrammarAttributeBertForMaskedLM":
        config = BertConfig.from_json_file(f"{load_directory}/config.json")
        tokenizer = SimpleBertTokenizer.from_pretrained(load_directory)
        model = cls(
            config,
            tokenizer,
            attribute_filter=GrammarAttributeFilter(tokenizer),
            attribute_bias_scale=attribute_bias_scale,
        )
        state_dict = torch.load(
            f"{load_directory}/pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
        return model
