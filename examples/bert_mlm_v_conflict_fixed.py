"""可直接运行的 V 型冲突 MLM 入口。

复用 bert_mlm_v_conflict 的训练实现，并修正单句 MLM 掩码索引的边界情况。
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import torch

sys.path.append(os.path.dirname(__file__))
import bert_mlm_v_conflict as experiment


def _make_masked_example_fixed(
    encoded: Dict[str, list],
    tokenizer,
    seed: int,
    mlm_probability: float,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    token_type_ids = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
    special_tokens_mask = torch.tensor(
        encoded.get("special_tokens_mask", [0] * len(encoded["input_ids"])),
        dtype=torch.bool,
    )

    labels = input_ids.clone()
    maskable = (~special_tokens_mask) & (attention_mask == 1)
    mask_positions = maskable & (
        torch.rand(input_ids.shape, generator=generator) < mlm_probability
    )
    if not bool(mask_positions.any()):
        candidates = maskable.nonzero(as_tuple=False)
        if candidates.numel() == 0:
            raise ValueError("Sentence contains no maskable token")
        mask_positions[int(candidates[0].item())] = True
    labels[~mask_positions] = -100

    masked_input_ids = input_ids.clone()
    replacement_probs = torch.rand(input_ids.shape, generator=generator)
    mask80 = mask_positions & (replacement_probs < 0.8)
    random10 = mask_positions & (replacement_probs >= 0.8) & (
        replacement_probs < 0.9
    )
    masked_input_ids[mask80] = tokenizer.mask_token_id

    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    valid_ids = torch.tensor(
        [idx for idx in range(len(tokenizer)) if idx not in special_ids],
        dtype=torch.long,
    )
    if valid_ids.numel() == 0:
        raise ValueError("Tokenizer has no regular tokens")
    random_indices = torch.randint(
        low=0,
        high=valid_ids.numel(),
        size=input_ids.shape,
        generator=generator,
    )
    random_words = valid_ids[random_indices]
    masked_input_ids[random10] = random_words[random10]

    return {
        "input_ids": masked_input_ids.unsqueeze(0),
        "token_type_ids": token_type_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "labels": labels.unsqueeze(0),
    }


experiment._make_masked_example = _make_masked_example_fixed


if __name__ == "__main__":
    args = experiment.parse_args()
    experiment.train(
        training_file=args.data_path,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        learning_rate=args.learning_rate,
        replay_weight=args.replay_weight,
        conflict_cosine=args.conflict_cosine,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
