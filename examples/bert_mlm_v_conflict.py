"""在标准 BERT MLM 更新中加入三维子空间 V 型冲突保护。

这个实验不是把词按句中位置摆成扇形，而是：

* Transformer 仍然通过上下文预测被遮住的词；
* 每次新句子的梯度都和一个旧句子回放梯度比较；
* 将隐藏维度按三个一组切成 3D 子空间；
* 如果新梯度在某个三维子空间里和旧梯度方向相反，就把新梯度
  投影到旧梯度的正交残差上，再执行 AdamW 更新。

这相当于在调参时给旧关系留出“绕行空间”。它是一个固定 256 维的
控制实验，先使用互不重叠的三维块，避免把组合数量和参数更新混在一起。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_simple import BertConfig, BertForMaskedLM
from bert_simple.tokenizer import SimpleBertTokenizer


DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "shiji_baihua_100_clean_final.txt")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "bert-mlm-v-conflict-100",
    )
)


def _resolve_path(path: str) -> str:
    """Resolve regular Windows paths and paths written as /D:/... ."""
    path = os.path.expandvars(os.path.expanduser(path))
    if len(path) >= 3 and path[0] in ("/", "\\") and path[2] == ":":
        path = path[1:]
    return os.path.abspath(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_texts(path: str, max_sentences: int) -> List[str]:
    resolved = _resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Training file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]
    if max_sentences > 0:
        texts = texts[:max_sentences]
    if not texts:
        raise ValueError(f"Training file is empty: {resolved}")
    return texts


def _make_masked_example(
    encoded: Dict[str, List[int]],
    tokenizer: SimpleBertTokenizer,
    seed: int,
    mlm_probability: float,
) -> Dict[str, torch.Tensor]:
    """为一个句子生成固定 MLM 掩码，后续回放时保持答案不变。"""
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
        row, col = candidates[0].tolist()
        mask_positions[row, col] = True
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


def _batch(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.cat([sample[key] for sample in samples], dim=0)
        for key in samples[0]
    }


def _loss_for_sample(
    model: BertForMaskedLM,
    sample: Dict[str, torch.Tensor],
) -> torch.Tensor:
    loss, _, _ = model(
        input_ids=sample["input_ids"],
        token_type_ids=sample["token_type_ids"],
        attention_mask=sample["attention_mask"],
        labels=sample["labels"],
    )
    return loss


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


class ThreeDimensionalVGuard:
    """按隐藏维度三元组投影冲突性梯度。"""

    def __init__(self, parameters: Sequence[torch.nn.Parameter], hidden_size: int):
        self.parameters = list(parameters)
        self.hidden_size = hidden_size
        self.triple_count = hidden_size // 3
        self.group_indices = self._build_group_indices()

    def _build_group_indices(self) -> List[torch.Tensor]:
        """建立扁平参数向量中每个三维子空间的索引。

        对输入特征维度为 hidden_size 的矩阵按列分组；对输出特征维度为
        hidden_size 且没有对应输入轴的矩阵按行分组；LayerNorm 向量按元素分组。
        这样检查的是 Transformer 参数真正接收/产生的隐藏特征，而不是词在句中的位置。
        """
        buckets: List[List[int]] = [[] for _ in range(self.triple_count)]
        offset = 0
        for parameter in self.parameters:
            shape = tuple(parameter.shape)
            if parameter.ndim == 1 and parameter.numel() == self.hidden_size:
                for group in range(self.triple_count):
                    for dimension in range(group * 3, group * 3 + 3):
                        buckets[group].append(offset + dimension)
            elif parameter.ndim >= 2 and shape[-1] == self.hidden_size:
                rows = parameter.numel() // self.hidden_size
                row_offsets = torch.arange(rows, dtype=torch.long) * self.hidden_size
                for group in range(self.triple_count):
                    for dimension in range(group * 3, group * 3 + 3):
                        indices = offset + row_offsets + dimension
                        buckets[group].extend(indices.tolist())
            elif parameter.ndim >= 2 and shape[0] == self.hidden_size:
                cols = parameter.numel() // self.hidden_size
                col_offsets = torch.arange(cols, dtype=torch.long)
                for group in range(self.triple_count):
                    for dimension in range(group * 3, group * 3 + 3):
                        indices = offset + dimension * cols + col_offsets
                        buckets[group].extend(indices.tolist())
            offset += parameter.numel()
        return [torch.tensor(values, dtype=torch.long) for values in buckets]

    def apply(
        self,
        new_gradients: Sequence[torch.Tensor],
        old_gradients: Sequence[torch.Tensor],
        replay_weight: float,
        conflict_cosine: float,
    ) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        new_flat = torch.cat([gradient.reshape(-1) for gradient in new_gradients])
        old_flat = torch.cat([gradient.reshape(-1) for gradient in old_gradients])
        conflict_count = 0
        checked_count = 0
        projected_squared = 0.0
        new_squared = float(new_flat.square().sum().item())
        conflict_cosines: List[float] = []

        for indices in self.group_indices:
            if indices.numel() == 0:
                continue
            new_block = new_flat.index_select(0, indices)
            old_block = old_flat.index_select(0, indices)
            new_norm = float(new_block.norm().item())
            old_norm = float(old_block.norm().item())
            if new_norm <= 1e-12 or old_norm <= 1e-12:
                continue
            checked_count += 1
            dot = float(torch.dot(new_block, old_block).item())
            cosine = dot / max(new_norm * old_norm, 1e-12)
            if cosine >= conflict_cosine:
                continue

            # 新梯度在这个三维块中反向指向旧梯度，去掉旧方向的分量，
            # 保留正交残差；这就是“在旧关系旁边绕过去”。
            coefficient = dot / max(old_norm * old_norm, 1e-12)
            residual = new_block - coefficient * old_block
            projected_squared += float((new_block - residual).square().sum().item())
            new_flat.index_copy_(0, indices, residual)
            conflict_count += 1
            conflict_cosines.append(cosine)

        combined_flat = new_flat + replay_weight * old_flat
        result: List[torch.Tensor] = []
        offset = 0
        for parameter in self.parameters:
            size = parameter.numel()
            result.append(combined_flat[offset : offset + size].view_as(parameter))
            offset += size
        return result, {
            "checked_triples": float(checked_count),
            "v_conflict_triples": float(conflict_count),
            "projected_gradient_norm_fraction": math.sqrt(projected_squared)
            / max(math.sqrt(new_squared), 1e-12),
            "worst_conflict_cosine": min(conflict_cosines, default=0.0),
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
    conflict_cosine: float = 0.0,
    mlm_probability: float = 0.15,
    seed: int = 42,
    log_every: int = 10,
) -> str:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if hidden_size < 3:
        raise ValueError("hidden_size must be at least 3")
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    set_seed(seed)
    device = torch.device("cpu")
    resolved_training_file = _resolve_path(training_file)
    texts = load_texts(resolved_training_file, max_sentences)
    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
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
    model = BertForMaskedLM(config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    v_guard = ThreeDimensionalVGuard(parameters, hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    masked_examples: List[Dict[str, torch.Tensor]] = []
    for index, text in enumerate(texts):
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_special_tokens_mask=True,
        )
        masked_examples.append(
            _make_masked_example(
                encoded,
                tokenizer,
                seed=seed + index * 1009,
                mlm_probability=mlm_probability,
            )
        )

    memory: List[Dict[str, object]] = []
    logs: List[Dict[str, object]] = []
    total_conflicts = 0
    total_checked = 0
    output_path = _resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, "training_log.jsonl")

    with open(log_path, "w", encoding="utf-8") as log_handle:
        model.train()
        for step, new_sample in enumerate(masked_examples, start=1):
            optimizer.zero_grad(set_to_none=True)
            new_loss = _loss_for_sample(model, new_sample)
            new_gradients = _gradient_tuple(new_loss, parameters)

            old_loss_value = 0.0
            old_gradients = [torch.zeros_like(gradient) for gradient in new_gradients]
            replay_index = None
            if memory:
                replay_index = (step - 1) % len(memory)
                old_sample = memory[replay_index]["sample"]
                old_loss = _loss_for_sample(model, old_sample)
                old_loss_value = float(old_loss.item())
                old_gradients = _gradient_tuple(old_loss, parameters)

            accepted_gradients, guard_stats = v_guard.apply(
                new_gradients,
                old_gradients,
                replay_weight=replay_weight if memory else 0.0,
                conflict_cosine=conflict_cosine,
            )
            for parameter, gradient in zip(parameters, accepted_gradients):
                parameter.grad = gradient
            optimizer.step()

            total_conflicts += int(guard_stats["v_conflict_triples"])
            total_checked += int(guard_stats["checked_triples"])
            current_new_loss = float(_loss_for_sample(model, new_sample).item())
            memory.append(
                {
                    "sample": new_sample,
                    "reference_loss": current_new_loss,
                }
            )

            memory_loss = 0.0
            forgetting = 0.0
            if step == 1 or step % log_every == 0 or step == len(masked_examples):
                memory_loss, forgetting = _evaluate_memory(model, memory)
                model.train()
                print(
                    f"step {step}/{len(masked_examples)}: "
                    f"new_loss={float(new_loss.item()):.4f}, "
                    f"replay_loss={old_loss_value:.4f}, "
                    f"v_conflicts={int(guard_stats['v_conflict_triples'])}, "
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
    summary = {
        "stage": "bert_mlm_with_3d_v_conflict_guard",
        "training_file": resolved_training_file,
        "device": "cpu",
        "sentences": len(texts),
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "three_dimensional_triples": v_guard.triple_count,
        "unused_hidden_tail_dimensions": hidden_size % 3,
        "learning_rate": learning_rate,
        "replay_weight": replay_weight,
        "conflict_cosine": conflict_cosine,
        "mlm_probability": mlm_probability,
        "total_v_conflict_triples": total_conflicts,
        "total_checked_triples": total_checked,
        "final_memory_loss": logs[-1]["memory_loss"],
        "final_forgetting": logs[-1]["forgetting"],
    }
    with open(os.path.join(output_path, "v_conflict_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved BERT MLM V-conflict outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 256D word-level BERT MLM with a 3D gradient V-conflict guard."
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
    parser.add_argument("--conflict-cosine", type=float, default=0.0)
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
        conflict_cosine=args.conflict_cosine,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
