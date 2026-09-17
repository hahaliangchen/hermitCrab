"""Fixed grammar automaton plus learned word-attribute filtering.

Grammar is a fixed transition system; word properties are a trainable
multi-label table.  The filter contributes a signed, soft candidate-logit
bias: compatible attributes are rewarded, known conflicts are penalized, and
high-confidence punctuation/function words get a context-sensitive surface
class adjustment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from .grammar_automaton import ATTRIBUTES, FixedGrammarAutomaton, GrammarState
from .tokenizer import SimpleBertTokenizer
from .word_attribute_layer import WordAttributeLayer


SPECIAL_TOKENS = frozenset(("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"))

# High-confidence surface classes are used even when the attribute dataset
# has no row for a token.  A word such as "去" is intentionally not included:
# it can be a legitimate ACTION answer in some grammar states.
PUNCTUATION_CHARS = frozenset(
    "。，、；：？！……——‘’“”（）《》「」『』【】.,;:?!()[]{}<>\"'_-—–"
)
FUNCTION_WORDS = frozenset(
    (
        "的",
        "地",
        "得",
        "了",
        "着",
        "过",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "也",
        "都",
        "之",
        "者",
        "所",
        "其",
        "而",
        "则",
        "乃",
        "且",
        "即",
        "因",
        "故",
        "于",
        "以",
        "为",
        "是",
        "在",
        "将",
        "被",
        "把",
        "和",
        "与",
        "及",
        "或",
        "但",
        "矣",
        "焉",
    )
)


def _surface_attribute(token: str) -> str | None:
    """Return a high-confidence surface class for a vocabulary token."""
    if token and all(char in PUNCTUATION_CHARS for char in token):
        return "PUNCT"
    if token in FUNCTION_WORDS:
        return "FUNCTION"
    return None


class GrammarAttributeFilter(nn.Module):
    """Learn word properties and apply fixed grammar to candidate logits."""

    SCHEMA_VERSION = 1

    def __init__(self, tokenizer: SimpleBertTokenizer, attribute_dim: int = 16):
        super().__init__()
        self.tokenizer = tokenizer
        # ``attribute_dim`` is retained for checkpoint/CLI compatibility.  The
        # attribute vocabulary is fixed by grammar_automaton and has 13 rows;
        # it must not silently become a second, incompatible label space.
        self.attribute_dim = len(ATTRIBUTES)
        self.word_attributes = WordAttributeLayer(len(tokenizer), self.attribute_dim)
        self.automaton = FixedGrammarAutomaton()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.word_attributes(token_ids)

    def mark_known_tokens(self, tokens: Iterable[str]) -> None:
        ids = [
            self.tokenizer.token_to_id[token]
            for token in tokens
            if token in self.tokenizer.token_to_id
        ]
        self.word_attributes.mark_known(ids)

    def mark_known(self, token_ids: Iterable[int]) -> None:
        """Compatibility alias used by the training entrypoints."""
        self.word_attributes.mark_known(token_ids)

    def attribute_probabilities(self, tokens: Sequence[str]) -> torch.Tensor:
        ids = [
            self.tokenizer.token_to_id.get(token, self.tokenizer.unk_token_id)
            for token in tokens
        ]
        device = next(self.parameters()).device
        return self.word_attributes.probabilities(
            torch.tensor(ids, dtype=torch.long, device=device)
        )

    def analyze(self, text: str) -> GrammarState:
        tokens = self.tokenizer.tokenize(text)
        mask_token = self.tokenizer.mask_token
        if tokens.count(mask_token) != 1:
            raise ValueError("text must contain exactly one [MASK] token")
        return self.automaton.analyze(tokens, tokens.index(mask_token))

    @torch.no_grad()
    def candidate_bias(
        self,
        text: str,
        candidate_tokens: Sequence[str],
        scale: float = 3.0,
    ) -> tuple[torch.Tensor, GrammarState]:
        """Return one signed logit bias per candidate and the grammar state."""
        state = self.analyze(text)
        return self.bias_for_state(state, candidate_tokens, scale=scale), state

    @torch.no_grad()
    def bias_for_state(
        self,
        state: GrammarState,
        candidate_tokens: Sequence[str],
        scale: float = 3.0,
        unknown_bias_scale: float = 0.08,
    ) -> torch.Tensor:
        """Build a soft, signed grammar/attribute logit bias.

        Compatible known words receive a positive gain.  Known incompatible
        words receive a stronger negative penalty.  Unknown words get only a
        small adjustment, while punctuation and clear function words are
        adjusted from their surface class when the grammar state is known.
        A fallback state stays conservative and only suppresses specials.
        """
        device = next(self.parameters()).device
        bias = torch.zeros(len(candidate_tokens), device=device)

        if not state.fallback:
            ids = torch.tensor(
                [
                    self.tokenizer.token_to_id.get(
                        token, self.tokenizer.unk_token_id
                    )
                    for token in candidate_tokens
                ],
                dtype=torch.long,
                device=device,
            )
            candidate_probs = self.word_attributes.probabilities(ids)
            allowed = torch.zeros(
                len(ATTRIBUTES), dtype=candidate_probs.dtype, device=device
            )
            for name in state.allowed_attributes:
                if name in ATTRIBUTES:
                    allowed[ATTRIBUTES.index(name)] = 1.0

            total_mass = candidate_probs.sum(-1).clamp_min(1e-6)
            compatible_mass = (candidate_probs * allowed).sum(-1) / total_mass
            compatibility = 0.25 + 0.75 * compatible_mass
            centered = ((compatibility - 0.5) / 0.5).clamp(-1.0, 1.0)
            safe_scale = max(float(scale), 0.0)

            # Positive and negative gains are intentionally asymmetric:
            # semantic conflicts are more informative than unknown words.
            positive = safe_scale * centered
            negative = (2.0 * safe_scale) * centered
            values = torch.where(centered >= 0.0, positive, negative)
            known = self.word_attributes.known_token_mask[ids]
            values = torch.where(
                known,
                values,
                torch.full_like(values, -safe_scale * float(unknown_bias_scale)),
            )

            allowed_names = set(state.allowed_attributes)
            punctuation_mask = torch.tensor(
                [_surface_attribute(token) == "PUNCT" for token in candidate_tokens],
                dtype=torch.bool,
                device=device,
            )
            function_mask = torch.tensor(
                [
                    _surface_attribute(token) == "FUNCTION"
                    for token in candidate_tokens
                ],
                dtype=torch.bool,
                device=device,
            )
            if "PUNCT" in allowed_names:
                values = torch.where(
                    punctuation_mask,
                    torch.maximum(
                        values,
                        torch.full_like(values, 0.35 * safe_scale),
                    ),
                    values,
                )
            else:
                values = torch.where(
                    punctuation_mask,
                    torch.minimum(
                        values,
                        torch.full_like(values, -1.5 * safe_scale),
                    ),
                    values,
                )
            if "FUNCTION" in allowed_names:
                values = torch.where(
                    function_mask,
                    torch.maximum(
                        values,
                        torch.full_like(values, 0.25 * safe_scale),
                    ),
                    values,
                )
            else:
                values = torch.where(
                    function_mask,
                    torch.minimum(
                        values,
                        torch.full_like(values, -0.5 * safe_scale),
                    ),
                    values,
                )
            bias = values

        special_mask = torch.tensor(
            [token in SPECIAL_TOKENS for token in candidate_tokens],
            dtype=torch.bool,
            device=device,
        )
        return torch.where(
            special_mask,
            torch.full_like(bias, -10000.0),
            bias,
        )

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(path))
        (path / "grammar_attribute_config.json").write_text(
            json.dumps(
                {
                    "version": self.SCHEMA_VERSION,
                    "attribute_dim": len(ATTRIBUTES),
                    "attributes": list(ATTRIBUTES),
                    "grammar": "fixed_local_automaton_v2",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        torch.save(self.state_dict(), path / "grammar_attribute.pt")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "GrammarAttributeFilter":
        path = Path(directory)
        config = json.loads(
            (path / "grammar_attribute_config.json").read_text(encoding="utf-8")
        )
        if config.get("version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported grammar attribute filter schema")
        if tuple(config.get("attributes", ())) != ATTRIBUTES:
            raise ValueError("attribute vocabulary does not match this filter")
        tokenizer = SimpleBertTokenizer.from_pretrained(str(path))
        model = cls(tokenizer, int(config.get("attribute_dim", len(ATTRIBUTES))))
        weights_path = path / "grammar_attribute.pt"
        summary_path = path / "word_attributes_summary.json"
        if weights_path.exists():
            model.load_state_dict(
                torch.load(
                    weights_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )
        elif summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with torch.no_grad():
                model.word_attributes.word_attribute_logits.fill_(-5.0)
                known_ids = []
                for token, attrs in summary.items():
                    if token in tokenizer.token_to_id:
                        tid = tokenizer.token_to_id[token]
                        known_ids.append(tid)
                        for attr in attrs:
                            if attr in ATTRIBUTES:
                                aidx = ATTRIBUTES.index(attr)
                                model.word_attributes.word_attribute_logits[tid, aidx] = 5.0
                model.mark_known(known_ids)
        return model

