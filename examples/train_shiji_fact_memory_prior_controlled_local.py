"""Run the prior-controlled fact stage with local-strong gradient routing.

The relation-selected dimensions receive the full update.  Inactive hidden
dimensions and parameters without a hidden-size axis receive only the small
background update, while the existing replay gradient protects old facts.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_shiji_fact_memory_prior_controlled_impl as implementation
from bert_simple.local_gradient_layout import LocalFeatureIndexLayout


# The original layout left non-feature parameters unmasked.  Patch only this
# experiment's implementation so older checkpoints and experiments are kept
# reproducible.
implementation.base.FeatureIndexLayout = LocalFeatureIndexLayout


_original_build_memory = implementation._build_memory


def _build_memory_with_id_frequencies(
    general_texts,
    tokenizer,
    frequencies,
    allocator,
    max_length,
    max_context_tokens,
    seed,
    mlm_probability,
    model,
):
    # The frequency vector uses token strings, while the existing relation
    # allocator expects a Counter keyed by token ids.
    frequency_ids = Counter(
        {
            int(tokenizer.token_to_id[token]): int(count)
            for token, count in frequencies.items()
            if token in tokenizer.token_to_id
        }
    )
    return _original_build_memory(
        general_texts,
        tokenizer,
        frequency_ids,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
        model,
    )


implementation._build_memory = _build_memory_with_id_frequencies


if __name__ == "__main__":
    args = implementation.parse_args()
    output_path = implementation.train(**vars(args))
    Path(output_path, "gradient_routing_config.json").write_text(
        json.dumps(
            {
                "layout": "LocalFeatureIndexLayout",
                "active_relation_weight": 1.0,
                "background_weight": float(args.background_weight),
                "non_feature_parameter_weight": float(args.background_weight),
                "replay_weight": float(args.replay_weight),
                "description": (
                    "Relation-selected dimensions receive full gradient; "
                    "inactive dimensions and non-feature tensors receive only "
                    "the small background gradient, while replay protects "
                    "previously learned relations."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
