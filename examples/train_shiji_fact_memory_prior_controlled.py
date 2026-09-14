"""Public entrypoint for the grammar-first, low-frequency-prior experiment."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_shiji_fact_memory_prior_controlled_impl as implementation


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
    implementation.train(**vars(args))
