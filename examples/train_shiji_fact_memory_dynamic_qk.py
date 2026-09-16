"""[历史对照] Run the old fact-memory experiment with contextual dynamic Q/K.

The imported legacy implementation still allocates token-pair relation
triples.  The current context-group sidecar is
``train_full_relation_filter_stages.py``; this entry point is kept only for
old-result comparison and must not be used to produce the new relation bank.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import train_shiji_fact_memory_dataset_impl as dataset_impl
from train_shiji_fact_memory_grammar_impl import _ensure_fact_positions

dataset_impl._ensure_fact_positions = _ensure_fact_positions

import train_shiji_fact_memory_local_relation_margin_impl as implementation
from bert_simple.dynamic_qk_model import (
    DEFAULT_DYNAMIC_QK_SCORE_SCALE,
    DEFAULT_ROUTE_DIM,
    DEFAULT_ROUTE_START_LAYER,
    DynamicQKLocalRelationMarginBertForMaskedLM,
)
from bert_simple.local_relation_adapter_model_v2 import (
    DEFAULT_RELATION_SCORE_SCALE,
)


implementation.LocalRelationMarginBertForMaskedLM = (
    DynamicQKLocalRelationMarginBertForMaskedLM
)


if __name__ == "__main__":
    output_dir = ROOT / "outputs" / "bert-mlm-fact-memory-shiji-dynamic-qk-256"
    output_path = implementation.train(
        output_dir=str(output_dir),
        fact_learning_rate=1e-5,
        global_background_weight=0.02,
        relation_learning_rate=5e-4,
        relation_margin_value=2.0,
        relation_margin_weight=0.25,
        relation_score_scale=DEFAULT_RELATION_SCORE_SCALE,
        dynamic_qk_score_scale=DEFAULT_DYNAMIC_QK_SCORE_SCALE,
        route_start_layer=DEFAULT_ROUTE_START_LAYER,
        route_dim=DEFAULT_ROUTE_DIM,
    )
    Path(output_path, "dynamic_qk_training_config.json").write_text(
        json.dumps(
            {
                "dynamic_qk": True,
                "route_source": "contextual_hidden_states",
                "attention_integration": "pre_softmax_local_qk_score",
                "route_start_layer": DEFAULT_ROUTE_START_LAYER,
                "route_dim": DEFAULT_ROUTE_DIM,
                "dynamic_qk_score_scale": DEFAULT_DYNAMIC_QK_SCORE_SCALE,
                "global_base_update_weight": 0.02,
                "global_base_optimizer_learning_rate": 1e-5,
                "relation_adapter_update_weight": 1.0,
                "relation_adapter_type": "3x3_bilinear_relation_matrix",
                "candidate_key_normalization": "l2",
                "relation_score_scale": DEFAULT_RELATION_SCORE_SCALE,
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
