"""Run the local-strong gradient experiment with a 0.05 background update."""

from __future__ import annotations

import json
from pathlib import Path

import train_shiji_fact_memory_prior_controlled_impl as implementation
from bert_simple.local_gradient_layout import LocalFeatureIndexLayout


implementation.base.FeatureIndexLayout = LocalFeatureIndexLayout


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "outputs" / "bert-mlm-fact-memory-shiji-prior-controlled-local05-256"
    output_path = implementation.train(
        output_dir=str(output_dir),
        background_weight=0.05,
        log_every=100,
    )
    Path(output_path, "gradient_routing_config.json").write_text(
        json.dumps(
            {
                "layout": "LocalFeatureIndexLayout",
                "active_relation_weight": 1.0,
                "background_weight": 0.05,
                "non_feature_parameter_weight": 0.05,
                "replay_weight": 0.5,
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
