"""Run the local 3x3 relation-adapter fact-memory experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The implementation imports this helper for a validation step.  Bootstrap
# the symbol here so the experiment remains isolated from older modules while
# reusing the existing dataset code.
import train_shiji_fact_memory_dataset_impl as dataset_impl
from train_shiji_fact_memory_grammar_impl import _ensure_fact_positions

dataset_impl._ensure_fact_positions = _ensure_fact_positions

import train_shiji_fact_memory_local_relation_margin_impl as implementation


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    output_dir = (
        root
        / "outputs"
        / "bert-mlm-fact-memory-shiji-local-relation-margin-256"
    )
    output_path = implementation.train(
        output_dir=str(output_dir),
        global_background_weight=0.02,
        relation_learning_rate=5e-4,
        relation_margin_value=2.0,
        relation_margin_weight=0.25,
        log_every=100,
    )
    Path(output_path, "local_relation_training_config.json").write_text(
        json.dumps(
            {
                "global_base_update_weight": 0.02,
                "relation_adapter_update_weight": 1.0,
                "relation_adapter_type": "3x3_bilinear_relation_matrix",
                "relation_learning_rate": 0.0005,
                "margin_value": 2.0,
                "margin_weight": 0.25,
                "hard_negative_sources": [
                    "context_tokens",
                    "current_top_8_candidates",
                ],
                "replay_weight": 0.5,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
