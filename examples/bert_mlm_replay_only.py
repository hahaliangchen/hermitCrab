"""与三维 V 投影实验配套的“仅回放”对照入口。"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(__file__))
import bert_mlm_v_conflict_fixed as fixed


if __name__ == "__main__":
    args = fixed.experiment.parse_args()
    fixed.experiment.train(
        training_file=args.data_path,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        learning_rate=args.learning_rate,
        replay_weight=0.5,
        conflict_cosine=-1.0,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
