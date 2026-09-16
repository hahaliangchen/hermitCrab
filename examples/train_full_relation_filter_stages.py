"""Train a context-conditioned relation sidecar in isolated stages.

Stages:

1. freeze contextual routing and dynamic Q/K; train only the structured FFN;
2. freeze the FFN and dynamic Q/K; train the contextual router;
3. freeze the FFN and router; train the full relation-specific Q/K bank.

The base checkpoint is never updated.  The complete 256-dimensional hidden
space is covered by a small, fixed shared bank of three-dimensional blocks.
The bank is not keyed by token pairs: ``是 || 的`` and ``在 || 了`` never
allocate spaces.  Every token occurrence gets its own contextual route from
the ordinary relative-position BERT hidden state, and the structured FFN
scores the masked position against its visible role context.

The previous 14k token-pair ``dimension_combination_map.json`` is deliberately
not read by this trainer.  It belongs to the retired candidate-allocation
experiment and must not be used as a semantic relation bank.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from bert_simple.dynamic_qk_model import DynamicQKRelationBilinearAdapter  # noqa: E402
from bert_simple.grammar_attribute_model import (  # noqa: E402
    GrammarAttributeBertForMaskedLM,
)
from bert_simple.context_spaces import (  # noqa: E402
    DEFAULT_RELATION_CANDIDATE_COUNT,
    build_context_space_triples,
)
from bert_simple.structured_relation import StructuredRelationScores  # noqa: E402
from bert_simple.tokenizer import SimpleBertTokenizer  # noqa: E402
from train_frozen_relation_filter import _sample_positions  # noqa: E402
from validate_relation_training_data import (  # noqa: E402
    DEFAULT_MIN_RELATION_TOKENS,
    DEFAULT_RELATION_MAX_LENGTH,
    load_groups,
    validate_groups,
)


DEFAULT_BASE_CHECKPOINT = (
    ROOT / "outputs/bert-mlm-fact-memory-shiji-grammar-attribute-256/before_fact"
)
DEFAULT_DATASET = ROOT / "data/shiji/manifests/relation_training_v2_draft.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/context-relation-filter-full256-stage1"
DEFAULT_CONTEXT_SPACE_COUNT = DEFAULT_RELATION_CANDIDATE_COUNT

Triple = Tuple[int, int, int]


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _current_rss_bytes() -> int:
    """Return process resident memory on Windows and Linux without psutil."""
    if os.name == "nt":
        try:
            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.restype = ctypes.c_void_p
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                ctypes.c_ulong,
            ]
            get_process_memory_info.restype = ctypes.c_int
            process = get_current_process()
            if get_process_memory_info(
                process, ctypes.byref(counters), counters.cb
            ):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError, TypeError, ValueError):
            return 0
        return 0

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


class MemoryBudget:
    """Hard RSS guard for training experiments.

    The limit is total process resident memory: Windows WorkingSetSize or
    Linux VmRSS, not an informal estimate of tensor sizes.  It does not
    measure CUDA VRAM.  This turns a dangerous host-memory run into an
    actionable error before the host is allowed to enter global OOM recovery.
    """

    def __init__(self, limit_mb: int) -> None:
        if limit_mb < 256:
            raise ValueError("memory budget must be at least 256 MiB")
        self.limit_bytes = int(limit_mb) * 1024 * 1024
        self.peak_bytes = 0

    def check(self, label: str) -> None:
        current = _current_rss_bytes()
        self.peak_bytes = max(self.peak_bytes, current)
        if current and current > self.limit_bytes:
            raise RuntimeError(
                f"memory budget exceeded at {label}: "
                f"rss={current / 1024**2:.1f} MiB, "
                f"limit={self.limit_bytes / 1024**2:.1f} MiB"
            )

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1024**2


class ContextualRelationFilter(nn.Module):
    """Structured relation sidecar over fixed shared context spaces."""

    def __init__(
        self,
        relation_triples: Sequence[Sequence[int]],
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mask_token_id: int,
        route_dim: int,
        route_start_layer: int,
        initializer_range: float,
        relation_ffn_hidden_size: int,
        relation_ffn_scale: float,
        relation_ffn_chunk_size: int,
    ) -> None:
        super().__init__()
        if hidden_size != 256:
            raise ValueError("context relation training requires hidden_size=256")
        if not 1 <= route_start_layer < num_layers:
            raise ValueError("route_start_layer must leave a dynamic layer")
        self.triples: Tuple[Triple, ...] = tuple(
            tuple(sorted(int(value) for value in triple))
            for triple in relation_triples
        )  # type: ignore[assignment]
        if not self.triples or max(max(triple) for triple in self.triples) >= hidden_size:
            raise ValueError("relation bank does not cover the full hidden space")
        self.route_start_layer = int(route_start_layer)
        self.adapter = DynamicQKRelationBilinearAdapter(
            relation_triples=self.triples,
            hidden_size=hidden_size,
            mask_token_id=mask_token_id,
            num_layers=num_layers,
            num_heads=num_heads,
            route_dim=route_dim,
            initializer_range=initializer_range,
        )
        self.adapter.dynamic_qk.structured_scores = StructuredRelationScores(
            hidden_size=relation_ffn_hidden_size,
            scale=relation_ffn_scale,
            chunk_size=relation_ffn_chunk_size,
        )

    @property
    def relation_count(self) -> int:
        return len(self.triples)

    def _candidate_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        active_spaces: Sequence[int],
    ) -> torch.Tensor:
        valid = attention_mask.bool()
        candidate_mask = torch.zeros(
            hidden_states.size(0),
            hidden_states.size(1),
            self.relation_count,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        active = torch.as_tensor(
            list(active_spaces), dtype=torch.long, device=hidden_states.device
        )
        # Assignment broadcasts the one-token validity column.  No expanded
        # tensor is created; the destination owns the active-space storage.
        candidate_mask[..., active] = valid.unsqueeze(-1)
        return candidate_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        active_spaces: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not active_spaces:
            raise ValueError("each sample needs at least one visible relation space")
        candidate_mask = self._candidate_mask(
            hidden_states, attention_mask, active_spaces
        )
        routes = self.adapter.route_weights(hidden_states, candidate_mask)
        bank = self.adapter.dynamic_qk
        layer_scores = []
        for layer_index in range(self.route_start_layer, bank.num_layers):
            layer_scores.append(
                bank.structured_scores(
                    layer_index,
                    hidden_states,
                    routes,
                    self.triples,
                    bank.q_matrix[layer_index],
                    bank.k_matrix[layer_index],
                )
            )
        # Attention compatibility path: [B, H, T, T] pair logits.  These
        # logits are not token-pair space allocation and are not the
        # supervised relation-group objective used by the current trainer.
        scores = torch.stack(layer_scores, dim=0).mean(dim=(0, 2))
        return scores, routes

    def forward_pairs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        active_spaces: Sequence[int],
        query_positions: Sequence[int],
        key_positions: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute only the supervised query/key pairs for sidecar training."""
        if not active_spaces:
            raise ValueError("each sample needs at least one visible relation space")
        candidate_mask = self._candidate_mask(
            hidden_states, attention_mask, active_spaces
        )
        routes = self.adapter.route_weights(hidden_states, candidate_mask)
        bank = self.adapter.dynamic_qk
        layer_scores = []
        for layer_index in range(self.route_start_layer, bank.num_layers):
            layer_scores.append(
                bank.structured_scores.forward_pairs(
                    layer_index,
                    hidden_states,
                    routes,
                    self.triples,
                    bank.q_matrix[layer_index],
                    bank.k_matrix[layer_index],
                    query_positions,
                    key_positions,
                    active_spaces=active_spaces,
                )
            )
        # [layers, B, H, query, key] -> [B, query, key].
        scores = torch.stack(layer_scores, dim=0).mean(dim=(0, 2))
        return scores, routes

    def forward_groups(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        active_spaces: Sequence[int],
        query_positions: Sequence[int],
        context_groups: Sequence[Sequence[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score complete relation-context groups, not isolated token pairs."""
        if not active_spaces:
            raise ValueError("each sample needs at least one visible relation space")
        candidate_mask = self._candidate_mask(
            hidden_states, attention_mask, active_spaces
        )
        routes = self.adapter.route_weights(hidden_states, candidate_mask)
        bank = self.adapter.dynamic_qk
        layer_scores = []
        for layer_index in range(self.route_start_layer, bank.num_layers):
            layer_scores.append(
                bank.structured_scores.forward_context_groups(
                    layer_index,
                    hidden_states,
                    routes,
                    self.triples,
                    bank.q_matrix[layer_index],
                    bank.k_matrix[layer_index],
                    query_positions,
                    context_groups,
                    active_spaces=active_spaces,
                )
            )
        # [layers, B, H, query, group] -> [B, query, group].
        scores = torch.stack(layer_scores, dim=0).mean(dim=(0, 2))
        return scores, routes


# Keep the old import name usable for local probes; it now means the fixed
# shared contextual relation sidecar, never a token-pair bank.
FullRelationFilter = ContextualRelationFilter


def _encode_sample(
    tokenizer: SimpleBertTokenizer,
    sample: Dict[str, object],
    max_length: int,
    relation_space_count: int,
    min_tokens: int,
) -> Dict[str, object]:
    tokens = list(sample["tokens"])
    if len(tokens) < min_tokens:
        raise ValueError(
            f"sample has fewer than {min_tokens} content tokens: {sample['sample_id']}"
        )
    if len(tokens) + 2 > max_length:
        raise ValueError(f"sample exceeds max_length: {sample['sample_id']}")
    required = set(tokens) | {str(sample["answer"])}
    missing = sorted(token for token in required if token not in tokenizer.token_to_id)
    if missing:
        raise ValueError(f"base tokenizer OOV in {sample['sample_id']}: {missing}")
    encoded = tokenizer.encode(
        " ".join(tokens),
        max_length=max_length,
        truncation=False,
        return_tensors="pt",
    )
    mask_position, positive, negative = _sample_positions(sample)
    return {
        "sample_id": str(sample["sample_id"]),
        "group_id": "",
        "input_ids": encoded["input_ids"][0],
        "attention_mask": encoded["attention_mask"][0],
        "mask_position": mask_position,
        "positive": positive,
        "negative": negative,
        # Every position routes over the same fixed context bank.  This is
        # intentionally independent of token identity and token pairs.
        "active_spaces": list(range(relation_space_count)),
    }


@torch.no_grad()
def _cache_hidden(
    base_model: GrammarAttributeBertForMaskedLM,
    examples: Sequence[Dict[str, object]],
    batch_size: int,
    pad_token_id: int,
) -> None:
    base_model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        length = max(int(row["input_ids"].numel()) for row in batch)
        input_ids = torch.full(
            (len(batch), length), pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(batch):
            ids = row["input_ids"]
            input_ids[index, : ids.numel()] = ids
            attention_mask[index, : ids.numel()] = 1
        hidden = base_model.bert(
            input_ids,
            torch.zeros_like(input_ids),
            attention_mask,
            output_attentions=False,
        )[0]
        for index, row in enumerate(batch):
            length = int(row["input_ids"].numel())
            row["hidden"] = hidden[index, :length].detach().clone()


def _loss_for_example(
    sidecar: FullRelationFilter,
    example: Dict[str, object],
    margin: float,
) -> Tuple[torch.Tensor, float, float]:
    hidden = example["hidden"].unsqueeze(0)
    attention = torch.ones(1, hidden.size(1), dtype=torch.long)
    positive_positions = sorted(set(int(value) for value in example["positive"]))
    negative_positions = sorted(set(int(value) for value in example["negative"]))
    scores, routes = sidecar.forward_groups(
        hidden,
        attention,
        example["active_spaces"],
        [int(example["mask_position"])],
        [positive_positions, negative_positions],
    )
    # The scorer has already pooled every member of each group.  There is no
    # pairwise top-k negative mining here: the positive group is the complete
    # visible relation context and the negative group is the complete marked
    # background context.
    positive = scores[0, 0, 0]
    negative = scores[0, 0, 1]
    loss = F.softplus(float(margin) - positive + negative)
    entropy = -(
        routes.clamp_min(1e-9) * routes.clamp_min(1e-9).log()
    ).sum(-1).mean()
    return loss, float((positive - negative).detach()), float(entropy.detach())


@torch.no_grad()
def _evaluate(
    sidecar: FullRelationFilter,
    examples: Sequence[Dict[str, object]],
    margin: float,
) -> Dict[str, float]:
    sidecar.eval()
    correct = 0
    margins = []
    entropies = []
    for example in examples:
        _, value, entropy = _loss_for_example(sidecar, example, margin)
        correct += int(value > 0.0)
        margins.append(value)
        entropies.append(entropy)
    count = max(len(examples), 1)
    return {
        "examples": float(len(examples)),
        "positive_over_hard_negative_rate": correct / count,
        "mean_margin": sum(margins) / count,
        "mean_route_entropy": sum(entropies) / count,
    }


def _select_stage_parameters(
    sidecar: FullRelationFilter,
    stage: str,
) -> List[nn.Parameter]:
    for parameter in sidecar.parameters():
        parameter.requires_grad_(False)
    bank = sidecar.adapter.dynamic_qk
    if stage == "ffn":
        parameters = list(bank.structured_scores.parameters())
    elif stage == "route":
        parameters = list(sidecar.adapter.router.parameters()) + [
            bank.space_descriptors
        ]
    elif stage == "qk":
        parameters = [bank.q_matrix, bank.k_matrix]
    else:
        raise ValueError(f"unknown stage: {stage}")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _stage_train(
    sidecar: FullRelationFilter,
    stage: str,
    examples: Sequence[Dict[str, object]],
    dev_examples: Sequence[Dict[str, object]],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    margin: float,
    max_grad_norm: float,
    seed: int,
    memory_budget: MemoryBudget,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    parameters = _select_stage_parameters(sidecar, stage)
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    best_state = {
        name: value.detach().clone() for name, value in sidecar.state_dict().items()
    }
    best_score = (float("-inf"), float("-inf"))
    best_epoch = 0
    logs = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(examples)
        sidecar.train()
        losses, margins, entropies = [], [], []
        for example_index, example in enumerate(examples):
            memory_budget.check(f"{stage}/epoch-{epoch + 1}/sample-{example_index}")
            optimizer.zero_grad(set_to_none=True)
            loss, value, entropy = _loss_for_example(sidecar, example, margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            memory_budget.check(
                f"{stage}/epoch-{epoch + 1}/sample-{example_index}-after-step"
            )
            losses.append(float(loss.detach()))
            margins.append(value)
            entropies.append(entropy)
        dev = _evaluate(sidecar, dev_examples, margin)
        record = {
            "stage": stage,
            "epoch": epoch + 1,
            "loss": sum(losses) / max(len(losses), 1),
            "train_mean_margin": sum(margins) / max(len(margins), 1),
            "train_route_entropy": sum(entropies) / max(len(entropies), 1),
            "dev_positive_over_hard_negative_rate": dev[
                "positive_over_hard_negative_rate"
            ],
            "dev_mean_margin": dev["mean_margin"],
        }
        logs.append(record)
        score = (
            float(dev["positive_over_hard_negative_rate"]),
            float(dev["mean_margin"]),
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().clone()
                for name, value in sidecar.state_dict().items()
            }
            memory_budget.check(f"{stage}/epoch-{epoch + 1}-best-state")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    sidecar.load_state_dict(best_state)
    return {
        "stage": stage,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_dev_score": list(best_score),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "logs": logs,
    }, _evaluate(sidecar, dev_examples, margin)


def train(args: argparse.Namespace) -> Dict[str, object]:
    if min(args.ffn_epochs, args.route_epochs, args.qk_epochs) < 1:
        raise ValueError("all stage epoch counts must be positive")
    if args.min_tokens < 1 or args.max_length < args.min_tokens + 2:
        raise ValueError("max-length must leave room for min-tokens plus CLS/SEP")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    memory_budget = MemoryBudget(args.memory_budget_mb)
    memory_budget.check("startup")

    base_checkpoint = Path(args.base_checkpoint)
    base_model = GrammarAttributeBertForMaskedLM.from_pretrained(
        str(base_checkpoint)
    ).to(torch.device("cpu"))
    base_model.eval()
    base_model.requires_grad_(False)
    memory_budget.check("base-model-loaded")
    tokenizer = SimpleBertTokenizer.from_pretrained(str(base_checkpoint))
    config = base_model.config
    if config.hidden_size != 256 or config.intermediate_size != 1024:
        raise ValueError("base checkpoint is not the required 256 -> 1024 -> 256 model")
    relation_triples = build_context_space_triples(
        config.hidden_size,
        candidate_count=DEFAULT_CONTEXT_SPACE_COUNT,
    )
    if config.hidden_size == 256 and len(relation_triples) != DEFAULT_CONTEXT_SPACE_COUNT:
        raise RuntimeError("unexpected fixed context-space count for hidden_size=256")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    groups = list(load_groups(args.dataset))
    validation_report = validate_groups(groups, min_tokens=args.min_tokens)
    examples_by_split = {"train": [], "dev": [], "test": []}
    skipped = {"train": 0, "dev": 0, "test": 0}
    for group in groups:
        split = str(group["split"])
        for sample in group["samples"]:
            try:
                row = _encode_sample(
                    tokenizer,
                    sample,
                    args.max_length,
                    len(relation_triples),
                    args.min_tokens,
                )
            except ValueError as error:
                if "no usable positive/negative positions" not in str(error):
                    raise
                skipped[split] += 1
                continue
            row["group_id"] = str(group["group_id"])
            examples_by_split[split].append(row)
    all_examples = [row for rows in examples_by_split.values() for row in rows]
    _cache_hidden(
        base_model,
        all_examples,
        args.cache_batch_size,
        tokenizer.pad_token_id,
    )
    memory_budget.check("hidden-cache")

    sidecar = FullRelationFilter(
        relation_triples=relation_triples,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        mask_token_id=tokenizer.mask_token_id,
        route_dim=args.route_dim,
        route_start_layer=args.route_start_layer,
        initializer_range=config.initializer_range,
        relation_ffn_hidden_size=args.relation_ffn_hidden_size,
        relation_ffn_scale=args.relation_ffn_scale,
        relation_ffn_chunk_size=args.relation_ffn_chunk_size,
    )
    # The frozen BERT is no longer needed after hidden-state caching and bank
    # initialization.  The context bank is intentionally freshly initialized;
    # keeping the retired token-pair bank alive during sidecar training needlessly
    # consumes the user's memory budget.
    del base_model
    gc.collect()
    memory_budget.check("sidecar-initialized")

    stage_specs = [
        ("ffn", args.ffn_epochs, args.ffn_learning_rate),
        ("route", args.route_epochs, args.route_learning_rate),
        ("qk", args.qk_epochs, args.qk_learning_rate),
    ]
    stage_reports = {}
    for stage, epochs, learning_rate in stage_specs:
        report, _ = _stage_train(
            sidecar,
            stage,
            examples_by_split["train"],
            examples_by_split["dev"],
            epochs,
            learning_rate,
            args.weight_decay,
            args.margin,
            args.max_grad_norm,
            args.seed + len(stage),
            memory_budget,
        )
        stage_reports[stage] = report
        torch.save(sidecar.state_dict(), output / f"relation_filter_{stage}.pt")

    torch.save(sidecar.state_dict(), output / "relation_filter.pt")
    tokenizer.save_pretrained(str(output))
    base_hash = hashlib.sha256(
        (base_checkpoint / "pytorch_model.bin").read_bytes()
    ).hexdigest()
    final_metrics = {
        "train": _evaluate(sidecar, examples_by_split["train"], args.margin),
        "dev": _evaluate(sidecar, examples_by_split["dev"], args.margin),
        "test": _evaluate(sidecar, examples_by_split["test"], args.margin),
    }
    metadata = {
        "model_class": "ContextualRelationFilter",
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_hash,
        "base_hidden_size": int(config.hidden_size),
        "base_intermediate_size": int(config.intermediate_size),
        "base_num_hidden_layers": int(config.num_hidden_layers),
        "base_num_attention_heads": int(config.num_attention_heads),
        "general_ffn_shape": [
            int(config.hidden_size),
            int(config.intermediate_size),
            int(config.hidden_size),
        ],
        "space_allocation": "fixed_shared_context_blocks",
        "context_space_triples": [list(triple) for triple in relation_triples],
        "relation_space_count": len(relation_triples),
        "relation_candidate_cap": DEFAULT_CONTEXT_SPACE_COUNT,
        "base_coordinate_coverage_count": (config.hidden_size + 2) // 3,
        "max_relation_dimension": max(max(triple) for triple in relation_triples),
        "candidate_selection": "all_fixed_spaces_contextual_route",
        "route_dim": args.route_dim,
        "route_start_layer": args.route_start_layer,
        "relation_ffn_hidden_size": args.relation_ffn_hidden_size,
        "relation_ffn_scale": args.relation_ffn_scale,
        "relation_ffn_chunk_size": args.relation_ffn_chunk_size,
        "stages": ["ffn", "route", "qk"],
        "frozen_main_bert": True,
        "max_length": args.max_length,
        "min_tokens": args.min_tokens,
        "validation": validation_report,
        "skipped_no_negative": skipped,
        "counts": {split: len(rows) for split, rows in examples_by_split.items()},
        "stage_reports": stage_reports,
        "final_metrics": final_metrics,
        "memory_budget_mb": args.memory_budget_mb,
        "peak_rss_mb": memory_budget.peak_mb,
    }
    (output / "relation_filter_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **final_metrics}, ensure_ascii=False, indent=2))
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-length", type=int, default=DEFAULT_RELATION_MAX_LENGTH)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_RELATION_TOKENS)
    parser.add_argument("--route-dim", type=int, default=32)
    parser.add_argument("--route-start-layer", type=int, default=1)
    parser.add_argument("--relation-ffn-hidden-size", type=int, default=32)
    parser.add_argument("--relation-ffn-scale", type=float, default=0.1)
    parser.add_argument("--relation-ffn-chunk-size", type=int, default=32)
    parser.add_argument("--ffn-epochs", type=int, default=4)
    parser.add_argument("--route-epochs", type=int, default=3)
    parser.add_argument("--qk-epochs", type=int, default=3)
    parser.add_argument("--ffn-learning-rate", type=float, default=5e-4)
    parser.add_argument("--route-learning-rate", type=float, default=2e-4)
    parser.add_argument("--qk-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-budget-mb", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
