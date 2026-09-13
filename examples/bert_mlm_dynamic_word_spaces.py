"""Dynamic word-to-local-space routing for a small BERT MLM experiment.

Each active local space owns a 256->3 Q projection, a 256->3 K projection,
and a 3x3 transform for each side.  A sparse registry records which spaces a
word has participated in.  Context selects among those spaces; a new space is
activated when the current local gradient conflicts with the space history.

The bank is preallocated only to keep the optimizer stable.  Inactive slots
are not used by the forward pass and are not part of the effective model.
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
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as combination_base


DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "shiji_baihua_100_clean.txt")
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "bert-mlm-dynamic-word-spaces-100",
    )
)


class WordSpaceRegistry:
    """稀疏记录词曾经参与过的局部空间，并按上下文选择空间。"""

    def __init__(self, tokenizer: combination_base.SimpleBertTokenizer, max_spaces: int):
        self.max_spaces = max_spaces
        self.active_spaces = 1
        self.word_to_spaces: Dict[int, Set[int]] = {}
        self.space_contexts: List[Counter] = [Counter() for _ in range(max_spaces)]
        self.space_usage: List[int] = [0 for _ in range(max_spaces)]
        self.special_ids = {
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        }
        self.mask_token_id = tokenizer.mask_token_id
        self.tokenizer = tokenizer

    def allocate(self) -> int:
        if self.active_spaces >= self.max_spaces:
            raise RuntimeError("dynamic local-space capacity exhausted")
        space_id = self.active_spaces
        self.active_spaces += 1
        return space_id

    def candidates(self, token_id: int) -> List[int]:
        candidates = self.word_to_spaces.get(int(token_id))
        if not candidates:
            return [0]
        return sorted(space for space in candidates if space < self.active_spaces)

    def _context_score(self, space_id: int, context_ids: Set[int]) -> float:
        if not context_ids:
            return -0.001 * self.space_usage[space_id]
        known = set(self.space_contexts[space_id])
        if not known:
            return -0.001 * self.space_usage[space_id]
        overlap = len(context_ids & known)
        union = len(context_ids | known)
        return overlap / max(union, 1) + 0.0001 * math.log1p(
            self.space_usage[space_id]
        )

    def choose_space(self, token_id: int, context_ids: Iterable[int]) -> int:
        context = set(context_ids)
        choices = self.candidates(token_id)
        return max(
            choices,
            key=lambda space_id: (self._context_score(space_id, context), -space_id),
        )

    def choose_context_space(self, context_ids: Iterable[int]) -> int:
        context = set(context_ids)
        choices = list(range(self.active_spaces))
        return max(
            choices,
            key=lambda space_id: (self._context_score(space_id, context), -space_id),
        )

    def context_ids(self, sample: Dict[str, object]) -> List[int]:
        input_ids = sample["input_ids"][0].tolist()
        attention_mask = sample["attention_mask"][0].tolist()
        return [
            int(token_id)
            for token_id, is_valid in zip(input_ids, attention_mask)
            if is_valid
            and token_id not in self.special_ids
            and token_id != self.mask_token_id
        ]

    def route_sample(self, sample: Dict[str, object]) -> torch.Tensor:
        input_ids = sample["input_ids"][0].tolist()
        attention_mask = sample["attention_mask"][0].tolist()
        all_context = set(self.context_ids(sample))
        route = torch.full(
            (1, len(input_ids)), -1, dtype=torch.long
        )
        for position, (token_id, is_valid) in enumerate(
            zip(input_ids, attention_mask)
        ):
            if not is_valid or token_id in self.special_ids:
                continue
            if token_id == self.mask_token_id:
                route[0, position] = self.choose_context_space(all_context)
            else:
                local_context = all_context - {int(token_id)}
                route[0, position] = self.choose_space(token_id, local_context)
        return route

    def words_using_space(
        self, sample: Dict[str, object], route: torch.Tensor, space_id: int
    ) -> Set[int]:
        input_ids = sample["input_ids"][0].tolist()
        attention_mask = sample["attention_mask"][0].tolist()
        return {
            int(token_id)
            for token_id, is_valid, selected_space in zip(
                input_ids, attention_mask, route[0].tolist()
            )
            if is_valid
            and selected_space == space_id
            and token_id not in self.special_ids
            and token_id != self.mask_token_id
        }

    def attach_words(self, word_ids: Iterable[int], space_id: int) -> None:
        for token_id in set(int(word_id) for word_id in word_ids):
            self.word_to_spaces.setdefault(token_id, {0}).add(space_id)

    def note_usage(self, sample: Dict[str, object], route: torch.Tensor) -> None:
        input_ids = sample["input_ids"][0].tolist()
        attention_mask = sample["attention_mask"][0].tolist()
        context = self.context_ids(sample)
        for token_id, is_valid, space_id in zip(
            input_ids, attention_mask, route[0].tolist()
        ):
            if (
                not is_valid
                or space_id < 0
                or token_id in self.special_ids
                or token_id == self.mask_token_id
            ):
                continue
            self.word_to_spaces.setdefault(int(token_id), {0}).add(int(space_id))
            self.space_contexts[space_id].update(
                other for other in context if other != token_id
            )
            self.space_usage[space_id] += 1

    def to_json(self) -> Dict[str, object]:
        mapping = {}
        for token_id, spaces in sorted(self.word_to_spaces.items()):
            token = self.tokenizer.id_to_token[token_id]
            mapping[token] = sorted(spaces)
        space_summary = []
        for space_id in range(self.active_spaces):
            context_tokens = [
                self.tokenizer.id_to_token[token_id]
                for token_id, _ in self.space_contexts[space_id].most_common(12)
                if 0 <= token_id < len(self.tokenizer.id_to_token)
            ]
            space_summary.append(
                {
                    "space_id": space_id,
                    "usage": self.space_usage[space_id],
                    "context_tokens": context_tokens,
                }
            )
        return {
            "active_spaces": self.active_spaces,
            "word_to_spaces": mapping,
            "spaces": space_summary,
        }


class DynamicWordSpaceBank(nn.Module):
    """预留少量槽位，每个槽位包含局部 Q/K 投影及 3x3 变换。"""

    def __init__(self, num_layers: int, max_spaces: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.max_spaces = max_spaces
        self.hidden_size = hidden_size
        self.p_q = nn.Parameter(torch.empty(num_layers, max_spaces, hidden_size, 3))
        self.p_k = nn.Parameter(torch.empty(num_layers, max_spaces, hidden_size, 3))
        self.a_q = nn.Parameter(torch.empty(num_layers, max_spaces, 3, 3))
        self.a_k = nn.Parameter(torch.empty(num_layers, max_spaces, 3, 3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.p_q.normal_(mean=0.0, std=0.02)
            self.p_k.normal_(mean=0.0, std=0.02)
            eye = torch.eye(3, device=self.a_q.device, dtype=self.a_q.dtype)
            self.a_q.copy_(eye)
            self.a_k.copy_(eye)
            self.a_q.add_(torch.randn_like(self.a_q) * 0.01)
            self.a_k.add_(torch.randn_like(self.a_k) * 0.01)

    def initialize_slot(self, slot_id: int, source_slot: int = 0) -> None:
        if not 0 <= slot_id < self.max_spaces:
            raise ValueError("invalid local-space slot")
        with torch.no_grad():
            self.p_q[:, slot_id].copy_(
                self.p_q[:, source_slot] + torch.randn_like(self.p_q[:, source_slot]) * 0.005
            )
            self.p_k[:, slot_id].copy_(
                self.p_k[:, source_slot] + torch.randn_like(self.p_k[:, source_slot]) * 0.005
            )
            self.a_q[:, slot_id].copy_(
                self.a_q[:, source_slot] + torch.randn_like(self.a_q[:, source_slot]) * 0.005
            )
            self.a_k[:, slot_id].copy_(
                self.a_k[:, source_slot] + torch.randn_like(self.a_k[:, source_slot]) * 0.005
            )

    def local_qk(
        self, layer_index: int, hidden_states: torch.Tensor, route_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        safe_ids = route_ids.clamp_min(0).clamp_max(self.max_spaces - 1)
        valid = route_ids.ge(0).unsqueeze(-1)
        batch_size, seq_len, _ = hidden_states.shape
        p_q = self.p_q[layer_index].index_select(
            0, safe_ids.reshape(-1)
        ).view(batch_size, seq_len, self.hidden_size, 3)
        p_k = self.p_k[layer_index].index_select(
            0, safe_ids.reshape(-1)
        ).view(batch_size, seq_len, self.hidden_size, 3)
        u_q = torch.einsum("btd,btdk->btk", hidden_states, p_q)
        u_k = torch.einsum("btd,btdk->btk", hidden_states, p_k)
        a_q = self.a_q[layer_index][safe_ids]
        a_k = self.a_k[layer_index][safe_ids]
        q = torch.einsum("btk,btkm->btm", u_q, a_q)
        k = torch.einsum("btk,btkm->btm", u_k, a_k)
        return q * valid, k * valid

    def local_scores(
        self, layer_index: int, hidden_states: torch.Tensor, route_ids: torch.Tensor
    ) -> torch.Tensor:
        q, k = self.local_qk(layer_index, hidden_states, route_ids)
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(3.0)

    def slot_gradient_vector(self, slot_id: int) -> torch.Tensor:
        pieces = []
        for parameter in (self.p_q, self.p_k, self.a_q, self.a_k):
            if parameter.grad is None:
                pieces.append(torch.zeros_like(parameter[:, slot_id]).reshape(-1))
            else:
                pieces.append(parameter.grad[:, slot_id].detach().reshape(-1))
        return torch.cat(pieces)


class WordSpaceSelfAttention(nn.Module):
    """保留标准注意力，并额外加入动态局部三维 QK 分数。"""

    def __init__(
        self,
        original: nn.Module,
        bank: DynamicWordSpaceBank,
        layer_index: int,
        local_score_scale: float,
    ):
        super().__init__()
        self.qkv = original.qkv
        self.out_proj = original.out_proj
        self.dropout = original.dropout
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.all_head_size = original.all_head_size
        self.layer_index = layer_index
        self.local_score_scale = local_score_scale
        object.__setattr__(self, "bank", bank)
        self.route_ids: Optional[torch.Tensor] = None

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def set_route_ids(self, route_ids: torch.Tensor) -> None:
        self.route_ids = route_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._transpose_for_scores(query)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        if self.route_ids is not None:
            local_scores = self.bank.local_scores(
                self.layer_index, hidden_states, self.route_ids
            )
            attn_scores = attn_scores + self.local_score_scale * local_scores.unsqueeze(1)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:2], self.all_head_size)
        return self.out_proj(context), attn_probs


class DynamicWordSpaceBertForMaskedLM(combination_base.BertForMaskedLM):
    def __init__(
        self,
        config: combination_base.BertConfig,
        bank: DynamicWordSpaceBank,
        local_score_scale: float,
    ):
        super().__init__(config)
        self.space_bank = bank
        self._local_attentions: List[WordSpaceSelfAttention] = []
        for layer_index, layer in enumerate(self.bert.encoder.layer):
            replacement = WordSpaceSelfAttention(
                layer.attention.self_attn,
                bank,
                layer_index,
                local_score_scale,
            )
            layer.attention.self_attn = replacement
            self._local_attentions.append(replacement)

    def set_route_ids(self, route_ids: torch.Tensor) -> None:
        for attention in self._local_attentions:
            attention.set_route_ids(route_ids)

    def forward(self, *args, route_ids: Optional[torch.Tensor] = None, **kwargs):
        if route_ids is not None:
            self.set_route_ids(route_ids)
        return super().forward(*args, **kwargs)


def _loss_for_sample(
    model: DynamicWordSpaceBertForMaskedLM, sample: Dict[str, object]
) -> torch.Tensor:
    loss, _, _ = model(
        input_ids=sample["input_ids"],
        token_type_ids=sample["token_type_ids"],
        attention_mask=sample["attention_mask"],
        labels=sample["labels"],
        route_ids=sample["route_ids"],
    )
    return loss


@torch.no_grad()
def _evaluate_memory(
    model: DynamicWordSpaceBertForMaskedLM, memory: Sequence[Dict[str, object]]
) -> Tuple[float, float]:
    if not memory:
        return 0.0, 0.0
    was_training = model.training
    model.eval()
    losses: List[float] = []
    forgetting: List[float] = []
    for item in memory:
        loss = float(_loss_for_sample(model, item["sample"]).item())
        losses.append(loss)
        forgetting.append(max(0.0, loss - float(item["reference_loss"])))
    if was_training:
        model.train()
    return sum(losses) / len(losses), sum(forgetting) / len(forgetting)


def _assign_gradients(
    parameters: Sequence[torch.nn.Parameter], gradients: Sequence[torch.Tensor]
) -> None:
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient


def _select_replay_index(
    memory: Sequence[Dict[str, object]], current_route: torch.Tensor, step: int
) -> Tuple[Optional[int], int]:
    if not memory:
        return None, 0
    current_spaces = set(int(x) for x in current_route.flatten().tolist() if x >= 0)
    overlap_scores = [
        len(current_spaces & set(item["active_spaces"])) for item in memory
    ]
    max_overlap = max(overlap_scores)
    if max_overlap > 0:
        candidates = [
            index for index, score in enumerate(overlap_scores) if score == max_overlap
        ]
        return candidates[(step - 1) % len(candidates)], max_overlap
    return ((step - 1) * 9973) % len(memory), 0


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
    max_spaces: int = 32,
    local_score_scale: float = 0.5,
    conflict_threshold: float = -0.02,
    mlm_probability: float = 0.15,
    seed: int = 42,
    log_every: int = 10,
) -> str:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if max_spaces < 1:
        raise ValueError("max_spaces must be positive")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    combination_base.set_seed(seed)
    resolved_training_file = combination_base._resolve_path(training_file)
    texts = combination_base.load_texts(resolved_training_file, max_sentences)
    tokenizer = combination_base.SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    print(f"Using training file: {resolved_training_file}")
    print(f"Loaded {len(texts)} sentences; word-level vocab={len(tokenizer)}")

    config = combination_base.BertConfig(
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
    bank = DynamicWordSpaceBank(num_hidden_layers, max_spaces, hidden_size)
    registry = WordSpaceRegistry(tokenizer, max_spaces)
    model = DynamicWordSpaceBertForMaskedLM(config, bank, local_score_scale).to(
        torch.device("cpu")
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    examples: List[Dict[str, object]] = []
    for index, text in enumerate(texts):
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_special_tokens_mask=True,
        )
        examples.append(
            combination_base._make_masked_example(
                encoded,
                tokenizer,
                seed=seed + index * 1009,
                mlm_probability=mlm_probability,
            )
        )

    memory: List[Dict[str, object]] = []
    slot_history: Dict[int, torch.Tensor] = {}
    logs: List[Dict[str, object]] = []
    total_added_spaces = 0
    total_conflict_slots = 0
    total_overlap_replays = 0
    total_overlap_count = 0
    output_path = combination_base._resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        model.train()
        for step, new_sample in enumerate(examples, start=1):
            new_sample["route_ids"] = registry.route_sample(new_sample)
            replay_index, replay_overlap = _select_replay_index(
                memory, new_sample["route_ids"], step
            )
            if replay_overlap > 0:
                total_overlap_replays += 1
                total_overlap_count += replay_overlap

            optimizer.zero_grad(set_to_none=True)
            new_loss_probe = _loss_for_sample(model, new_sample)
            new_gradients_probe = combination_base._gradient_tuple(
                new_loss_probe, parameters
            )
            _assign_gradients(parameters, new_gradients_probe)
            active_current = set(
                int(x) for x in new_sample["route_ids"].flatten().tolist() if x >= 0
            )
            conflicted = []
            for space_id in sorted(active_current):
                current_gradient = bank.slot_gradient_vector(space_id)
                previous_gradient = slot_history.get(space_id)
                if previous_gradient is None:
                    continue
                current_norm = float(current_gradient.norm().item())
                previous_norm = float(previous_gradient.norm().item())
                if current_norm <= 1e-12 or previous_norm <= 1e-12:
                    continue
                cosine = float(
                    torch.dot(current_gradient, previous_gradient).item()
                    / max(current_norm * previous_norm, 1e-12)
                )
                if cosine < conflict_threshold:
                    conflicted.append((space_id, cosine))

            route = new_sample["route_ids"].clone()
            added_this_step = 0
            for old_space, _ in conflicted:
                words = registry.words_using_space(new_sample, route, old_space)
                if not words:
                    continue
                new_space = registry.allocate()
                bank.initialize_slot(new_space, source_slot=old_space)
                registry.attach_words(words, new_space)
                for position, token_id in enumerate(
                    new_sample["input_ids"][0].tolist()
                ):
                    if route[0, position].item() == old_space and (
                        token_id in words or token_id == registry.mask_token_id
                    ):
                        route[0, position] = new_space
                added_this_step += 1
            if added_this_step:
                total_added_spaces += added_this_step
                total_conflict_slots += len(conflicted)
                new_sample["route_ids"] = route

            optimizer.zero_grad(set_to_none=True)
            new_loss = _loss_for_sample(model, new_sample)
            old_loss_value = 0.0
            total_loss = new_loss
            if replay_index is not None:
                old_loss = _loss_for_sample(model, memory[replay_index]["sample"])
                old_loss_value = float(old_loss.item())
                total_loss = total_loss + replay_weight * old_loss
            total_gradients = combination_base._gradient_tuple(total_loss, parameters)
            _assign_gradients(parameters, total_gradients)
            optimizer.step()

            route = new_sample["route_ids"]
            active_after = set(int(x) for x in route.flatten().tolist() if x >= 0)
            for space_id in active_after:
                current_gradient = bank.slot_gradient_vector(space_id)
                if float(current_gradient.norm().item()) > 1e-12:
                    previous_gradient = slot_history.get(space_id)
                    if previous_gradient is None:
                        slot_history[space_id] = current_gradient.clone()
                    else:
                        slot_history[space_id] = (
                            0.8 * previous_gradient + 0.2 * current_gradient
                        )
            registry.note_usage(new_sample, route)
            model.eval()
            with torch.no_grad():
                current_new_loss = float(_loss_for_sample(model, new_sample).item())
            model.train()
            memory.append(
                {
                    "sample": new_sample,
                    "reference_loss": current_new_loss,
                    "active_spaces": sorted(active_after),
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
                    f"active_spaces={registry.active_spaces}, "
                    f"added={added_this_step}, "
                    f"replay_overlap={replay_overlap}, "
                    f"memory_loss={memory_loss:.4f}, "
                    f"forgetting={forgetting:.4f}"
                )

            record = {
                "step": step,
                "new_loss": float(new_loss.item()),
                "replay_loss": old_loss_value,
                "replay_index": replay_index,
                "replay_overlap": replay_overlap,
                "current_new_loss": current_new_loss,
                "memory_size": len(memory),
                "active_spaces": registry.active_spaces,
                "added_spaces": added_this_step,
                "conflicted_slots": [
                    {"space_id": space_id, "cosine": cosine}
                    for space_id, cosine in conflicted
                ],
                "memory_loss": memory_loss,
                "forgetting": forgetting,
            }
            logs.append(record)
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    with open(
        os.path.join(output_path, "word_space_registry.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(registry.to_json(), handle, ensure_ascii=False, indent=2)
    summary = {
        "stage": "bert_mlm_dynamic_word_space_routing",
        "training_file": resolved_training_file,
        "device": "cpu",
        "sentences": len(texts),
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "local_space_capacity": max_spaces,
        "active_local_spaces": registry.active_spaces,
        "total_added_spaces": total_added_spaces,
        "total_conflict_slots": total_conflict_slots,
        "local_score_scale": local_score_scale,
        "conflict_threshold": conflict_threshold,
        "replay_weight": replay_weight,
        "overlap_replay_steps": total_overlap_replays,
        "total_replay_overlap": total_overlap_count,
        "word_mapping_links": sum(
            len(spaces) for spaces in registry.word_to_spaces.values()
        ),
        "words_with_multiple_spaces": sum(
            1 for spaces in registry.word_to_spaces.values() if len(spaces) > 1
        ),
        "final_memory_loss": logs[-1]["memory_loss"],
        "final_forgetting": logs[-1]["forgetting"],
    }
    with open(
        os.path.join(output_path, "dynamic_word_space_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved dynamic word-space outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT MLM with dynamic word-to-local-space routing."
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
    parser.add_argument("--max-spaces", type=int, default=32)
    parser.add_argument("--local-score-scale", type=float, default=0.5)
    parser.add_argument("--conflict-threshold", type=float, default=-0.02)
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
        max_spaces=args.max_spaces,
        local_score_scale=args.local_score_scale,
        conflict_threshold=args.conflict_threshold,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
