"""Public entry point for grammar/attribute-aware fact-memory training."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(__file__))

from bert_simple.grammar_attribute_filter import GrammarAttributeFilter


def _mark_known_compat(self, token_ids):
    self.word_attributes.mark_known(token_ids)


# Keep the public filter API backward-compatible with the training entrypoint.
if not hasattr(GrammarAttributeFilter, "mark_known"):
    GrammarAttributeFilter.mark_known = _mark_known_compat

import train_shiji_fact_memory_grammar_impl as implementation

train = implementation.train
parse_args = implementation.parse_args


if __name__ == "__main__":
    args = parse_args()
    values = vars(args)
    values["num_hidden_layers"] = values.pop("layers")
    values["num_attention_heads"] = values.pop("heads")
    train(**values)
