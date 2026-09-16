"""Legacy answer-conditioned manifest experiment; use train_relation_pairs.py for new JSONL."""

import argparse
import warnings
from train_shiji_fact_memory_dynamic_qk import implementation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--fact-dataset", required=True)
    parser.add_argument("--fact-epochs", type=int, default=1)
    parser.add_argument("--correction-probe-every", type=int, default=100)
    args = parser.parse_args()
    warnings.warn(
        "Legacy manifest candidates depend on gold answers; do not interpret its evaluation as "
        "unknown-answer generalization. Use train_relation_pairs.py for label-independent training.",
        stacklevel=1,
    )
    implementation.train(
        **vars(args), relation_ffn_hidden_size=32, relation_ffn_scale=0.1,
        relation_ffn_chunk_size=32, relation_score_scale=24.0,
        dynamic_qk_score_scale=0.5, route_start_layer=1, route_dim=32,
    )
