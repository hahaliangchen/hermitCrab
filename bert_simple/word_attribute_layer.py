"""Directly learnable multi-label attributes for vocabulary words."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn

from .grammar_automaton import ATTRIBUTES


class WordAttributeLayer(nn.Module):
    """A transparent ``word x attribute`` logit table.

    Each row is a word's learned property vector.  A sigmoid is applied per
    attribute, so a word can have several properties at once.  For candidate
    matching, only attributes whose probability crosses 0.5 count as active;
    small probabilities are uncertainty rather than extra word properties.
    """

    def __init__(self, vocab_size: int, attribute_dim: int = 16):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = int(vocab_size)
        self.attribute_dim = len(ATTRIBUTES)
        self.word_attribute_logits = nn.Parameter(
            torch.zeros(self.vocab_size, self.attribute_dim)
        )
        self.register_buffer(
            "known_token_mask", torch.zeros(self.vocab_size, dtype=torch.bool)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.word_attribute_logits[token_ids]

    def mark_known(self, token_ids: Iterable[int]) -> None:
        ids = sorted({int(token_id) for token_id in token_ids})
        if ids:
            self.known_token_mask[ids] = True

    def probabilities(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self(token_ids).sigmoid()

    def candidate_compatibility(
        self,
        candidate_ids: torch.Tensor,
        allowed_attributes: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_ids.ndim != 1:
            raise ValueError("candidate_ids must be one-dimensional")
        allowed = torch.zeros(
            len(ATTRIBUTES), dtype=torch.bool, device=candidate_ids.device
        )
        indices = [ATTRIBUTES.index(name) for name in allowed_attributes]
        if indices:
            allowed[indices] = True
        active = self.probabilities(candidate_ids) >= 0.5
        known = self.known_token_mask[candidate_ids]
        active_count = active.sum(-1).clamp_min(1)
        compatibility = (active & allowed).sum(-1).float() / active_count.float()
        # An all-negative known row is uncertain and should not be crushed.
        uncertain = active.sum(-1).eq(0)
        compatibility = torch.where(
            uncertain, torch.full_like(compatibility, 0.5), compatibility
        )
        compatibility = torch.where(
            known, compatibility.clamp(1e-4, 1.0), torch.ones_like(compatibility)
        )
        return compatibility, known

