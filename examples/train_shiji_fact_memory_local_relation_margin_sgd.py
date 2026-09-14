"""Run local 3x3 fact-memory training with routed SGD updates.

The global BERT path is still softly gated by ``global_background_weight`` in
the implementation, while the relation adapters use the stronger learning
rate. Unlike AdamW, SGD keeps that gradient/update ratio visible instead of
normalizing it away with per-parameter moments.

The legacy implementation currently constructs AdamW internally. This entry
point replaces that one optimizer factory for this process only, so the old
experiment remains reproducible and AdamW can be restored later by using the
original entry point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_shiji_fact_memory_dataset_impl as dataset_impl
from train_shiji_fact_memory_grammar_impl import _ensure_fact_positions

dataset_impl._ensure_fact_positions = _ensure_fact_positions

import train_shiji_fact_memory_local_relation_margin_impl as implementation
from bert_simple.local_relation_adapter_model_v2 import (
    ScaledLocalRelationMarginBertForMaskedLM,
)


implementation.LocalRelationMarginBertForMaskedLM = (
    ScaledLocalRelationMarginBertForMaskedLM
)


def _run_with_sgd() -> str:
    original_adamw = torch.optim.AdamW

    def sgd_factory(parameter_groups):
        return torch.optim.SGD(
            parameter_groups,
            momentum=0.9,
            nesterov=True,
        )

    # Keep the change local to this process. The core implementation applies
    # the 0.02 global gradient gate; matching raw learning rates therefore
    # makes the effective global update approximately 2% of the local update.
    torch.optim.AdamW = sgd_factory
    try:
        root = Path(__file__).resolve().parent.parent
        output_dir = (
            root
            / "outputs"
            / "bert-mlm-fact-memory-shiji-local-relation-margin-sgd-256"
        )
        return implementation.train(
            output_dir=str(output_dir),
            fact_learning_rate=5e-4,
            relation_learning_rate=5e-4,
            global_background_weight=0.02,
            relation_margin_value=2.0,
            relation_margin_weight=0.25,
        )
    finally:
        torch.optim.AdamW = original_adamw


if __name__ == "__main__":
    output_path = _run_with_sgd()
    Path(output_path, "local_relation_training_config.json").write_text(
        json.dumps(
            {
                "optimizer": "sgd",
                "optimizer_momentum": 0.9,
                "optimizer_nesterov": True,
                "global_base_gradient_gate": 0.02,
                "global_base_optimizer_learning_rate": 0.0005,
                "relation_adapter_update_weight": 1.0,
                "relation_adapter_type": "3x3_bilinear_relation_matrix",
                "candidate_key_normalization": "l2",
                "relation_score_scale": 24.0,
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
