"""[历史对照] 标准 BERT MLM + token-pair 三维组合更新实验。

本文件及其调用方保留用于复现实验结果，不属于当前关系架构。它把 token pair
映射到三维坐标，正是当前方案已经废弃的知识记忆式分配；新的训练入口必须使用
``bert_simple.context_spaces.build_context_space_triples`` 和上下文组评分。

与 bert_mlm_v_conflict.py 的区别：

* 不把 256 维硬切成 85 个固定三维块；
* 使用 C(256, 3) 个候选组合，为词对关系分配并复用三维坐标；
* 新句子和旧句子的梯度分别写入各自关系组合，允许其它维度少量泄漏；
* 不要求所有负余弦都正交，只对很强的反向更新做轻微衰减。

Transformer 仍然是标准的上下文 MLM，三维组合只控制参数更新的特征通道。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

sys.path.append(os.path.dirname(__file__))
from bert_mlm_v_conflict import (
    BertConfig,
    BertForMaskedLM,
    SimpleBertTokenizer,
    _loss_for_sample,
    _resolve_path,
    load_texts,
    set_seed,
)


DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "shiji_baihua_100_clean.txt")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "bert-mlm-dimension-combinations-100",
    )
)


RelationKey = Tuple[int, int]
Triple = Tuple[int, int, int]


class DimensionCombinationAllocator:
    """历史 token-pair allocator；当前训练禁止使用。"""

    def __init__(self, hidden_size: int):
        if hidden_size < 3:
            raise ValueError("hidden_size must be at least 3")
        self.hidden_size = hidden_size
        self.capacity = hidden_size * (hidden_size - 1) * (hidden_size - 2) // 6
        self.relation_triples: Dict[RelationKey, Triple] = {}
        self._next_i = 0
        self._next_j = 1
        self._next_k = 2

    def _take_next(self) -> Triple:
        if len(self.relation_triples) >= self.capacity:
            raise RuntimeError("three-dimensional combination capacity exhausted")
        triple = (self._next_i, self._next_j, self._next_k)
        self._next_k += 1
        if self._next_k >= self.hidden_size:
            self._next_j += 1
            self._next_k = self._next_j + 1
            if self._next_k >= self.hidden_size:
                self._next_i += 1
                self._next_j = self._next_i + 1
                self._next_k = self._next_j + 1
        return triple

    def allocate(self, relation: RelationKey) -> Triple:
        relation = tuple(sorted((int(relation[0]), int(relation[1]))))
        if relation[0] == relation[1]:
            raise ValueError("a relation needs two different token ids")
        if relation in self.relation_triples:
            return self.relation_triples[relation]
        triple = self._take_next()
        self.relation_triples[relation] = triple
        return triple


def _make_masked_example(
    encoded: Dict[str, List[int]],
    tokenizer: SimpleBertTokenizer,
    seed: int,
    mlm_probability: float,
) -> Dict[str, torch.Tensor]:
    """生成固定的单句 MLM 样本，避免回放时目标发生变化。"""
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
    masked_input_ids[random10] = valid_ids[random_indices][random10]
    return {
        "input_ids": masked_input_ids.unsqueeze(0),
        "token_type_ids": token_type_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "original_input_ids": input_ids,
        "mask_positions": mask_positions,
    }


class FeatureIndexLayout:
    """把参数中的 hidden_size 轴映射到一个扁平索引空间。"""

    def __init__(self, parameters: Sequence[torch.nn.Parameter], hidden_size: int):
        self.parameters = list(parameters)
        self.hidden_size = hidden_size
        self.indices_by_dim: List[torch.Tensor] = self._build_indices()

    def _build_indices(self) -> List[torch.Tensor]:
        buckets: List[List[int]] = [[] for _ in range(self.hidden_size)]
        offset = 0
        for parameter in self.parameters:
            shape = tuple(parameter.shape)
            if parameter.ndim == 1 and parameter.numel() == self.hidden_size:
                for dimension in range(self.hidden_size):
                    buckets[dimension].append(offset + dimension)
            elif parameter.ndim >= 2 and shape[-1] == self.hidden_size:
                rows = parameter.numel() // self.hidden_size
                row_offsets = torch.arange(rows, dtype=torch.long) * self.hidden_size
                for dimension in range(self.hidden_size):
                    buckets[dimension].extend(
                        (offset + row_offsets + dimension).tolist()
                    )
            elif parameter.ndim >= 2 and shape[0] == self.hidden_size:
                cols = parameter.numel() // self.hidden_size
                col_offsets = torch.arange(cols, dtype=torch.long)
                for dimension in range(self.hidden_size):
                    buckets[dimension].extend(
                        (offset + dimension * cols + col_offsets).tolist()
                    )
            offset += parameter.numel()
        return [torch.tensor(values, dtype=torch.long) for values in buckets]

    def mask_gradients(
        self,
        gradients: Sequence[torch.Tensor],
        feature_mask: torch.Tensor,
        background_weight: float,
    ) -> List[torch.Tensor]:
        if feature_mask.numel() != self.hidden_size:
            raise ValueError("feature mask width does not match hidden size")
        weights = background_weight + (1.0 - background_weight) * feature_mask.float()
        result: List[torch.Tensor] = []
        for parameter, gradient in zip(self.parameters, gradients):
            if parameter.ndim == 1 and parameter.numel() == self.hidden_size:
                result.append(gradient * weights)
            elif parameter.ndim >= 2 and parameter.shape[-1] == self.hidden_size:
                view_shape = [1] * gradient.ndim
                view_shape[-1] = self.hidden_size
                result.append(gradient * weights.view(view_shape))
            elif (
                parameter.ndim >= 2
                and parameter.shape[0] == self.hidden_size
            ):
                view_shape = [self.hidden_size] + [1] * (gradient.ndim - 1)
                result.append(gradient * weights.view(view_shape))
            else:
                result.append(gradient)
        return result

    def flat(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([gradient.reshape(-1) for gradient in gradients])

    def indices_for_triples(self, triples: Iterable[Triple]) -> torch.Tensor:
        dimensions: Set[int] = set()
        for triple in triples:
            dimensions.update(int(value) for value in triple)
        if not dimensions:
            return torch.empty(0, dtype=torch.long)
        return torch.cat([self.indices_by_dim[dimension] for dimension in sorted(dimensions)])


def _gradient_tuple(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> List[torch.Tensor]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        allow_unused=True,
        retain_graph=False,
    )
    return [
        gradient.detach().clone()
        if gradient is not None
        else torch.zeros_like(parameter)
        for parameter, gradient in zip(parameters, gradients)
    ]


def _union_mask(triples: Sequence[Triple], hidden_size: int) -> torch.Tensor:
    mask = torch.zeros(hidden_size, dtype=torch.bool)
    for triple in triples:
        mask[list(triple)] = True
    return mask


def _relation_keys(
    sample: Dict[str, torch.Tensor],
    tokenizer: SimpleBertTokenizer,
    frequencies: Counter,
    max_context_tokens: int,
) -> List[RelationKey]:
    original = sample["original_input_ids"].tolist()
    mask_positions = sample["mask_positions"].tolist()
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    content_ids = [
        int(token_id)
        for position, token_id in enumerate(original)
        if mask_positions[position] is False and token_id not in special_ids
    ]
    unique_context = sorted(
        set(content_ids),
        key=lambda token_id: (-frequencies[token_id], token_id),
    )[:max_context_tokens]
    targets = sorted(
        {
            int(original[position])
            for position, is_masked in enumerate(mask_positions)
            if is_masked and original[position] not in special_ids
        }
    )
    result: List[RelationKey] = []
    for target in targets:
        for context in unique_context:
            if target != context:
                result.append(tuple(sorted((target, context))))
    return list(dict.fromkeys(result))


def _combine_gradients(
    new_gradients: Sequence[torch.Tensor],
    old_gradients: Sequence[torch.Tensor],
    new_triples: Sequence[Triple],
    old_triples: Sequence[Triple],
    layout: FeatureIndexLayout,
    hidden_size: int,
    replay_weight: float,
    background_weight: float,
    soft_conflict_threshold: float,
    soft_conflict_scale: float,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    new_mask = _union_mask(new_triples, hidden_size)
    old_mask = _union_mask(old_triples, hidden_size)
    masked_new = layout.mask_gradients(new_gradients, new_mask, background_weight)
    masked_old = layout.mask_gradients(old_gradients, old_mask, background_weight)
    new_flat = layout.flat(masked_new)
    old_flat = layout.flat(masked_old)

    shared_triples = set(new_triples).intersection(old_triples)
    checked = 0
    conflicts = 0
    conflict_cosines: List[float] = []
    for triple in sorted(shared_triples):
        indices = layout.indices_for_triples([triple])
        new_block = new_flat.index_select(0, indices)
        old_block = old_flat.index_select(0, indices)
        new_norm = float(new_block.norm().item())
        old_norm = float(old_block.norm().item())
        if new_norm <= 1e-12 or old_norm <= 1e-12:
            continue
        checked += 1
        cosine = float(torch.dot(new_block, old_block).item()) / max(
            new_norm * old_norm, 1e-12
        )
        if cosine < soft_conflict_threshold:
            # 只减弱强冲突的新更新，不做严格正交投影。
            new_flat.index_copy_(0, indices, new_block * soft_conflict_scale)
            conflicts += 1
            conflict_cosines.append(cosine)

    combined = new_flat + replay_weight * old_flat
    result: List[torch.Tensor] = []
    offset = 0
    for parameter in layout.parameters:
        size = parameter.numel()
        result.append(combined[offset : offset + size].view_as(parameter))
        offset += size
    return result, {
        "shared_relation_triples": float(len(shared_triples)),
        "checked_relation_triples": float(checked),
        "soft_conflict_triples": float(conflicts),
        "worst_soft_conflict_cosine": min(conflict_cosines, default=0.0),
        "new_active_dimensions": float(new_mask.sum().item()),
        "old_active_dimensions": float(old_mask.sum().item()),
    }


@torch.no_grad()
def _evaluate_memory(
    model: BertForMaskedLM,
    memory: Sequence[Dict[str, object]],
) -> Tuple[float, float]:
    if not memory:
        return 0.0, 0.0
    was_training = model.training
    model.eval()
    losses: List[float] = []
    forgetting: List[float] = []
    for item in memory:
        sample = item["sample"]
        loss = float(_loss_for_sample(model, sample).item())
        losses.append(loss)
        forgetting.append(max(0.0, loss - float(item["reference_loss"])))
    if was_training:
        model.train()
    return sum(losses) / len(losses), sum(forgetting) / len(forgetting)


def train(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_sentences: int = 100,
    max_length: int = 128,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: Optional[int] = None,
    learning_rate: float = 5e-4,
    replay_weight: float = 0.5,
    background_weight: float = 0.35,
    max_context_tokens: int = 8,
    soft_conflict_threshold: float = -0.35,
    soft_conflict_scale: float = 0.85,
    mlm_probability: float = 0.15,
    seed: int = 42,
    log_every: int = 10,
) -> str:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if not 0.0 <= background_weight <= 1.0:
        raise ValueError("background_weight must be between 0 and 1")
    if not 0.0 < soft_conflict_scale <= 1.0:
        raise ValueError("soft_conflict_scale must be in (0, 1]")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    set_seed(seed)
    resolved_training_file = _resolve_path(training_file)
    texts = load_texts(resolved_training_file, max_sentences)
    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    frequencies = Counter()
    for text in texts:
        frequencies.update(tokenizer.tokenize(text))
    print(f"Using training file: {resolved_training_file}")
    print(f"Loaded {len(texts)} sentences; word-level vocab={len(tokenizer)}")

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_length + 10,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = BertForMaskedLM(config).to(torch.device("cpu"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    layout = FeatureIndexLayout(parameters, hidden_size)
    allocator = DimensionCombinationAllocator(hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    examples: List[Dict[str, object]] = []
    for index, text in enumerate(texts):
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_special_tokens_mask=True,
        )
        sample = _make_masked_example(
            encoded,
            tokenizer,
            seed=seed + index * 1009,
            mlm_probability=mlm_probability,
        )
        relations = _relation_keys(
            sample,
            tokenizer,
            frequencies,
            max_context_tokens=max_context_tokens,
        )
        triples = [allocator.allocate(relation) for relation in relations]
        sample["relation_keys"] = relations
        sample["relation_triples"] = triples
        # 关系元数据不需要进入模型 forward，但保留在回放内存中。
        examples.append(sample)

    memory: List[Dict[str, object]] = []
    logs: List[Dict[str, object]] = []
    total_soft_conflicts = 0
    total_checked = 0
    output_path = _resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8") as log_handle:
        model.train()
        for step, new_sample in enumerate(examples, start=1):
            optimizer.zero_grad(set_to_none=True)
            new_loss = _loss_for_sample(model, new_sample)
            new_gradients = _gradient_tuple(new_loss, parameters)
            old_loss_value = 0.0
            old_gradients = [torch.zeros_like(gradient) for gradient in new_gradients]
            old_triples: List[Triple] = []
            replay_index = None
            if memory:
                replay_index = (step - 1) % len(memory)
                old_item = memory[replay_index]
                old_sample = old_item["sample"]
                old_loss = _loss_for_sample(model, old_sample)
                old_loss_value = float(old_loss.item())
                old_gradients = _gradient_tuple(old_loss, parameters)
                old_triples = old_item["relation_triples"]

            accepted_gradients, guard_stats = _combine_gradients(
                new_gradients,
                old_gradients,
                new_sample["relation_triples"],
                old_triples,
                layout,
                hidden_size,
                replay_weight=replay_weight if memory else 0.0,
                background_weight=background_weight,
                soft_conflict_threshold=soft_conflict_threshold,
                soft_conflict_scale=soft_conflict_scale,
            )
            for parameter, gradient in zip(parameters, accepted_gradients):
                parameter.grad = gradient
            optimizer.step()

            total_soft_conflicts += int(guard_stats["soft_conflict_triples"])
            total_checked += int(guard_stats["checked_relation_triples"])
            current_new_loss = float(_loss_for_sample(model, new_sample).item())
            memory.append(
                {
                    "sample": new_sample,
                    "reference_loss": current_new_loss,
                    "relation_triples": new_sample["relation_triples"],
                }
            )

            memory_loss = 0.0
            forgetting = 0.0
            if step == 1 or step % log_every == 0 or step == len(examples):
                memory_loss, forgetting = _evaluate_memory(model, memory)
                model.train()
                print(
                    f"step {step}/{len(examples)}: "
                    f"new_loss={float(new_loss.item()):.4f}, "
                    f"replay_loss={old_loss_value:.4f}, "
                    f"relations={len(new_sample['relation_triples'])}, "
                    f"shared={int(guard_stats['shared_relation_triples'])}, "
                    f"soft_conflicts={int(guard_stats['soft_conflict_triples'])}, "
                    f"memory_loss={memory_loss:.4f}, "
                    f"forgetting={forgetting:.4f}"
                )

            record = {
                "step": step,
                "new_loss": float(new_loss.item()),
                "replay_loss": old_loss_value,
                "replay_index": replay_index,
                "current_new_loss": current_new_loss,
                "memory_size": len(memory),
                "memory_loss": memory_loss,
                "forgetting": forgetting,
                **guard_stats,
            }
            logs.append(record)
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    used_dimensions: Set[int] = set()
    for triple in allocator.relation_triples.values():
        used_dimensions.update(triple)
    summary = {
        "stage": "bert_mlm_dimension_combination_updates",
        "training_file": resolved_training_file,
        "device": "cpu",
        "sentences": len(texts),
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "combination_capacity": allocator.capacity,
        "allocated_relation_combinations": len(allocator.relation_triples),
        "used_hidden_dimensions": len(used_dimensions),
        "learning_rate": learning_rate,
        "replay_weight": replay_weight,
        "background_weight": background_weight,
        "max_context_tokens": max_context_tokens,
        "soft_conflict_threshold": soft_conflict_threshold,
        "soft_conflict_scale": soft_conflict_scale,
        "mlm_probability": mlm_probability,
        "total_soft_conflict_triples": total_soft_conflicts,
        "total_checked_relation_triples": total_checked,
        "final_memory_loss": logs[-1]["memory_loss"],
        "final_forgetting": logs[-1]["forgetting"],
    }
    with open(
        os.path.join(output_path, "dimension_combination_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved BERT dimension-combination outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train word-level BERT MLM with relation-specific 3D combinations."
    )
    parser.add_argument("data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sentences", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--background-weight", type=float, default=0.35)
    parser.add_argument("--max-context-tokens", type=int, default=8)
    parser.add_argument("--soft-conflict-threshold", type=float, default=-0.35)
    parser.add_argument("--soft-conflict-scale", type=float, default=0.85)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
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
        background_weight=args.background_weight,
        max_context_tokens=args.max_context_tokens,
        soft_conflict_threshold=args.soft_conflict_threshold,
        soft_conflict_scale=args.soft_conflict_scale,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
