"""Public entry point for structured Shiji fact-memory training."""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(__file__))
import train_shiji_fact_memory_dataset_impl as implementation


_original_encode_fact_sample = implementation._encode_fact_sample


def _encode_fact_sample_with_position(*args, **kwargs):
    sample = _original_encode_fact_sample(*args, **kwargs)
    if sample is not None and "fact_position" not in sample:
        positions = sample["mask_positions"].nonzero(as_tuple=False)
        if positions.numel() != 1:
            raise ValueError("A fact sample must contain exactly one mask position")
        sample["fact_position"] = int(positions[0].item())
    return sample


implementation._encode_fact_sample = _encode_fact_sample_with_position
train = implementation.train
parse_args = implementation.parse_args


if __name__ == "__main__":
    args = parse_args()
    values = vars(args)
    values["num_hidden_layers"] = values.pop("layers")
    values["num_attention_heads"] = values.pop("heads")
    train(**values)
