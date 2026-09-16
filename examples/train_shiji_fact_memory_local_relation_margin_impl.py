"""Train fact memory with routed 3x3 adapters and margin loss.

The default adapter is the MLM-side relation matrix.  Dynamic-Q/K entry
points can replace the model class and reuse the same update loop; all
relation-scoped adapter parameters then share the replay/conflict guard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as base
import train_shiji_fact_memory_dataset_impl as dataset_impl
from bert_mlm_dimension_combinations_epochs import _save_relation_map
from bert_simple.grammar_attribute_model import GrammarAttributeBertForMaskedLM
from bert_simple.local_relation_adapter_model import (
    LocalRelationMarginBertForMaskedLM,
)
from train_shiji_fact_memory_prior_controlled_impl import (
    _build_frequency_vectors,
)
from train_shiji_fact_memory_dataset_impl import _ensure_fact_positions


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FACT_DATASET = os.path.join(
    ROOT, "data", "shiji", "manifests", "fact_memory_dataset.json"
)
DEFAULT_INIT_CHECKPOINT = os.path.join(
    ROOT,
    "outputs",
    "bert-mlm-fact-memory-shiji-grammar-attribute-256",
    "before_fact",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "bert-mlm-fact-memory-shiji-local-relation-margin-256",
)

Triple = Tuple[int, int, int]


def _frequency_ids(
    frequencies: Counter[str], tokenizer
) -> Counter[int]:
    return Counter(
        {
            int(tokenizer.token_to_id[token]): int(count)
            for token, count in frequencies.items()
            if token in tokenizer.token_to_id
        }
    )


def _attach_hard_negatives(sample: Dict[str, object], tokenizer) -> None:
    """Add context tokens that are useful hard negatives for the fact mask."""
    target_id = int(sample["fact_token_id"])
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    candidates: List[int] = []
    for token_id in sample["original_input_ids"].tolist():
        token_id = int(token_id)
        if token_id in special_ids or token_id == target_id:
            continue
        if token_id not in candidates:
            candidates.append(token_id)
    sample["hard_negative_ids"] = candidates


def _forward_loss(
    model: LocalRelationMarginBertForMaskedLM,
    sample: Dict[str, object],
    margin_weight: float,
    margin_value: float,
    return_route_weights: bool = False,
):
    return model(
        input_ids=sample["input_ids"],
        token_type_ids=sample["token_type_ids"],
        attention_mask=sample["attention_mask"],
        labels=sample["labels"],
        relation_triples=sample.get("relation_triples", []),
        margin_target_id=(
            int(sample["fact_token_id"])
            if "fact_token_id" in sample
            else None
        ),
        hard_negative_ids=sample.get("hard_negative_ids", []),
        margin_weight=margin_weight,
        margin_value=margin_value,
        return_grammar_info=True,
        return_route_weights=return_route_weights,
    )


@torch.no_grad()
def _loss_for_sample(
    model: LocalRelationMarginBertForMaskedLM,
    sample: Dict[str, object],
) -> float:
    was_training = model.training
    model.eval()
    result = _forward_loss(model, sample, margin_weight=0.0, margin_value=0.0)
    value = float(result[0].item())
    if was_training:
        model.train()
    return value


def _active_adapter_indices(
    model: LocalRelationMarginBertForMaskedLM,
    triples: Iterable[Sequence[int]],
) -> List[int]:
    return model.relation_adapter.indices_for_triples(triples)


def _gradient_norm(gradients: Sequence[torch.Tensor]) -> float:
    squared = sum(
        float(gradient.detach().float().pow(2).sum().item())
        for gradient in gradients
    )
    return squared ** 0.5


def _route_diagnostics(
    route_weights: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    if route_weights is None:
        return {
            "route_entropy": 0.0,
            "route_active_spaces": 0.0,
            "route_candidate_count": 0.0,
        }
    valid = attention_mask.bool().to(device=route_weights.device)
    probabilities = route_weights[valid]
    if probabilities.numel() == 0:
        return {
            "route_entropy": 0.0,
            "route_active_spaces": 0.0,
            "route_candidate_count": 0.0,
        }
    entropy = (
        -probabilities.clamp_min(1e-9)
        * probabilities.clamp_min(1e-9).log()
    ).sum(-1).mean()
    return {
        "route_entropy": float(entropy.item()),
        "route_active_spaces": float(
            (probabilities.sum(0) > 0.0).sum().item()
        ),
        "route_candidate_count": float(
            probabilities.gt(0.0).sum(-1).float().mean().item()
        ),
    }


def _combine_adapter_gradients(
    model: LocalRelationMarginBertForMaskedLM,
    new_gradients: Sequence[torch.Tensor],
    old_gradients: Sequence[torch.Tensor],
    new_triples: Sequence[Triple],
    old_triples: Sequence[Triple],
    replay_weight: float,
    soft_conflict_threshold: float,
    soft_conflict_scale: float,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Keep strong active adapter updates and replay old adapter rows.

    The original adapter has one ``[relation, 3, 3]`` parameter.  The dynamic
    Q/K adapter adds relation-indexed layer/head matrices and shared router
    parameters, so the adapter owns the masking/block-vector hooks used here.
    """
    if len(new_gradients) != len(old_gradients):
        raise ValueError("new and replay adapter gradients have different lengths")
    new_indices = _active_adapter_indices(model, new_triples)
    old_indices = _active_adapter_indices(model, old_triples)
    adapter = model.relation_adapter
    new_gradient = adapter.mask_gradients(new_gradients, new_indices)
    old_gradient = adapter.mask_gradients(old_gradients, old_indices)

    shared = sorted(set(new_indices).intersection(old_indices))
    checked = 0
    conflicts = 0
    cosines: List[float] = []
    for index in shared:
        new_block = adapter.relation_gradient_vector(new_gradient, index)
        old_block = adapter.relation_gradient_vector(old_gradient, index)
        new_norm = float(new_block.norm().item())
        old_norm = float(old_block.norm().item())
        if new_norm <= 1e-12 or old_norm <= 1e-12:
            continue
        checked += 1
        cosine = float(torch.dot(new_block, old_block).item()) / max(
            new_norm * old_norm, 1e-12
        )
        if cosine < soft_conflict_threshold:
            new_gradient = adapter.scale_relation_gradient(
                new_gradient,
                index,
                soft_conflict_scale,
            )
            conflicts += 1
            cosines.append(cosine)

    accepted = [
        new_gradient_value + float(replay_weight) * old_gradient_value
        for new_gradient_value, old_gradient_value in zip(
            new_gradient, old_gradient
        )
    ]
    return accepted, {
        "active_relation_adapters": float(len(new_indices)),
        "replay_relation_adapters": float(len(old_indices)),
        "shared_relation_adapters": float(len(shared)),
        "checked_relation_adapters": float(checked),
        "soft_conflict_adapters": float(conflicts),
        "worst_adapter_conflict_cosine": min(cosines, default=0.0),
        "new_adapter_gradient_norm": _gradient_norm(new_gradient),
        "replay_adapter_gradient_norm": _gradient_norm(old_gradient),
        "accepted_adapter_gradient_norm": _gradient_norm(accepted),
    }


def _combine_base_gradients(
    new_gradients: Sequence[torch.Tensor],
    old_gradients: Sequence[torch.Tensor],
    background_weight: float,
    replay_weight: float,
) -> List[torch.Tensor]:
    """Keep the shared BERT base as a weak glue path in this experiment."""
    weight = float(background_weight)
    return [
        weight * (new_gradient + float(replay_weight) * old_gradient)
        for new_gradient, old_gradient in zip(new_gradients, old_gradients)
    ]


def _target_margin(
    scores: torch.Tensor,
    target_id: int,
    hard_negative_ids: Sequence[int],
) -> Tuple[float, float]:
    target_score = float(scores[target_id].item())
    without_target = scores.detach().clone()
    without_target[target_id] = float("-inf")
    strongest_wrong = float(without_target.max().item())
    all_margin = target_score - strongest_wrong
    hard_ids = [
        int(candidate)
        for candidate in hard_negative_ids
        if 0 <= int(candidate) < scores.numel() and int(candidate) != target_id
    ]
    hard_wrong = (
        float(scores[hard_ids].max().item()) if hard_ids else strongest_wrong
    )
    return all_margin, target_score - hard_wrong


def _run_update(
    model: LocalRelationMarginBertForMaskedLM,
    all_parameters: Sequence[torch.nn.Parameter],
    global_parameters: Sequence[torch.nn.Parameter],
    adapter_parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    sample: Dict[str, object],
    memory: Sequence[Dict[str, object]],
    global_step: int,
    replay_weight: float,
    global_background_weight: float,
    relation_margin_weight: float,
    relation_margin_value: float,
    soft_conflict_threshold: float,
    soft_conflict_scale: float,
) -> Dict[str, float]:
    replay_index = (global_step - 1) % len(memory) if memory else None
    optimizer.zero_grad(set_to_none=True)
    new_result = _forward_loss(
        model,
        sample,
        margin_weight=relation_margin_weight,
        margin_value=relation_margin_value,
        return_route_weights=True,
    )
    new_loss = new_result[0]
    new_info = new_result[3]
    route_stats = _route_diagnostics(
        new_result[4] if len(new_result) > 4 else None,
        sample["attention_mask"],
    )
    new_gradients = base._gradient_tuple(new_loss, all_parameters)

    old_loss_value = 0.0
    old_gradients = [torch.zeros_like(parameter) for parameter in all_parameters]
    old_triples: Sequence[Triple] = []
    if replay_index is not None:
        old_sample = memory[replay_index]["sample"]
        old_result = _forward_loss(
            model,
            old_sample,
            margin_weight=0.0,
            margin_value=relation_margin_value,
        )
        old_loss = old_result[0]
        old_loss_value = float(old_loss.item())
        old_gradients = base._gradient_tuple(old_loss, all_parameters)
        old_triples = memory[replay_index]["relation_triples"]

    global_count = len(global_parameters)
    new_global = new_gradients[:global_count]
    old_global = old_gradients[:global_count]
    new_adapter = new_gradients[global_count:]
    old_adapter = old_gradients[global_count:]
    accepted_global = _combine_base_gradients(
        new_global,
        old_global,
        global_background_weight,
        replay_weight if memory else 0.0,
    )
    accepted_adapter, adapter_stats = _combine_adapter_gradients(
        model,
        new_adapter,
        old_adapter,
        sample.get("relation_triples", []),
        old_triples,
        replay_weight if memory else 0.0,
        soft_conflict_threshold,
        soft_conflict_scale,
    )
    adapter_before = [parameter.detach().clone() for parameter in adapter_parameters]
    relation_matrix_before = model.relation_adapter.relation_matrix.detach().clone()
    dynamic_qk = getattr(model.relation_adapter, "dynamic_qk", None)
    dynamic_qk_before = []
    if dynamic_qk is not None:
        dynamic_qk_before = [
            dynamic_qk.q_matrix.detach().clone(),
            dynamic_qk.k_matrix.detach().clone(),
        ]
    accepted_gradients = accepted_global + accepted_adapter
    for parameter, gradient in zip(all_parameters, accepted_gradients):
        parameter.grad = gradient
    optimizer.step()

    adapter_update_norm = _gradient_norm(
        [parameter.detach() - before for parameter, before in zip(adapter_parameters, adapter_before)]
    )
    matrix_update_norm = float(
        (model.relation_adapter.relation_matrix.detach() - relation_matrix_before)
        .float()
        .norm()
        .item()
    )
    relation_matrix_index = next(
        index
        for index, parameter in enumerate(adapter_parameters)
        if parameter is model.relation_adapter.relation_matrix
    )
    adapter_stats.update(
        {
            "adapter_parameter_norm": _gradient_norm(
                [parameter.detach() for parameter in adapter_parameters]
            ),
            "adapter_update_norm": adapter_update_norm,
            "relation_matrix_gradient_norm": float(
                accepted_adapter[relation_matrix_index]
                .detach()
                .float()
                .norm()
                .item()
            ),
            "relation_matrix_norm": float(
                model.relation_adapter.relation_matrix.detach().float().norm().item()
            ),
            "relation_matrix_nonzero_rows": float(
                model.relation_adapter.relation_matrix.detach()
                .float()
                .norm(dim=(1, 2))
                .gt(1e-12)
                .sum()
                .item()
            ),
            "relation_matrix_update_norm": matrix_update_norm,
        }
    )
    if dynamic_qk is not None:
        structured_scores = getattr(dynamic_qk, "structured_scores", None)
        if structured_scores is not None:
            ffn_ids = {id(parameter) for parameter in structured_scores.parameters()}
            adapter_stats["relation_ffn_gradient_norm"] = _gradient_norm(
                [gradient for parameter, gradient in zip(adapter_parameters, accepted_adapter)
                 if id(parameter) in ffn_ids]
            )
            adapter_stats["relation_ffn_update_norm"] = _gradient_norm(
                [parameter.detach() - old for parameter, old in zip(adapter_parameters, adapter_before)
                 if id(parameter) in ffn_ids]
            )
        adapter_stats["dynamic_qk_gradient_norm"] = _gradient_norm(
            [
                accepted_adapter[index]
                for index, parameter in enumerate(adapter_parameters)
                if parameter is dynamic_qk.q_matrix or parameter is dynamic_qk.k_matrix
            ]
        )
        adapter_stats["dynamic_qk_update_norm"] = _gradient_norm(
            [
                parameter.detach() - before
                for parameter, before in zip(
                    (dynamic_qk.q_matrix, dynamic_qk.k_matrix),
                    dynamic_qk_before,
                )
            ]
        )

    position = int(sample["fact_position"]) if "fact_position" in sample else -1
    target_margin = 0.0
    hard_margin = 0.0
    if position >= 0:
        target_margin, hard_margin = _target_margin(
            new_result[1][0, position],
            int(sample["fact_token_id"]),
            sample.get("hard_negative_ids", []),
        )
    return {
        "new_loss": float(new_loss.item()),
        "new_ce_loss": float(new_info["ce_loss"].item()),
        "new_margin_loss": float(new_info["margin_loss"].item()),
        "target_margin_before_update": target_margin,
        "hard_negative_margin_before_update": hard_margin,
        "replay_loss": old_loss_value,
        "replay_index": float(replay_index if replay_index is not None else -1),
        **route_stats,
        **adapter_stats,
    }


@torch.no_grad()
def _evaluate_memory(
    model: LocalRelationMarginBertForMaskedLM,
    memory: Sequence[Dict[str, object]],
) -> Tuple[float, float]:
    if not memory:
        return 0.0, 0.0
    was_training = model.training
    model.eval()
    losses: List[float] = []
    forgetting: List[float] = []
    for item in memory:
        loss = _loss_for_sample(model, item["sample"])
        losses.append(loss)
        forgetting.append(max(0.0, loss - float(item["reference_loss"])))
    if was_training:
        model.train()
    return sum(losses) / len(losses), sum(forgetting) / len(forgetting)


@torch.no_grad()
def _fact_metrics(
    model: LocalRelationMarginBertForMaskedLM,
    samples: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    if not samples:
        return {
            "samples": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "loss": 0.0,
            "mean_margin": 0.0,
            "mean_hard_negative_margin": 0.0,
        }
    was_training = model.training
    model.eval()
    top1 = 0
    top5 = 0
    losses: List[float] = []
    margins: List[float] = []
    hard_margins: List[float] = []
    for sample in samples:
        result = _forward_loss(model, sample, margin_weight=0.0, margin_value=0.0)
        loss, logits = result[:2]
        position = int(sample["fact_position"])
        target = int(sample["fact_token_id"])
        scores = logits[0, position]
        order = torch.argsort(scores, descending=True)
        rank = int((order == target).nonzero(as_tuple=False)[0].item()) + 1
        top1 += int(rank <= 1)
        top5 += int(rank <= 5)
        losses.append(float(loss.item()))
        margin, hard_margin = _target_margin(
            scores, target, sample.get("hard_negative_ids", [])
        )
        margins.append(margin)
        hard_margins.append(hard_margin)
    if was_training:
        model.train()
    count = len(samples)
    return {
        "samples": float(count),
        "top1": top1 / count,
        "top5": top5 / count,
        "loss": sum(losses) / count,
        "mean_margin": sum(margins) / count,
        "mean_hard_negative_margin": sum(hard_margins) / count,
    }


def _build_memory(
    general_texts: Sequence[str],
    tokenizer,
    frequencies: Counter[int],
    allocator,
    max_length: int,
    max_context_tokens: int,
    seed: int,
    mlm_probability: float,
    model: LocalRelationMarginBertForMaskedLM,
) -> List[Dict[str, object]]:
    probes, _ = dataset_impl._make_stable_examples(
        general_texts,
        tokenizer,
        frequencies,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
    )
    memory: List[Dict[str, object]] = []
    for probe in probes:
        memory.append(
            {
                "sample": probe,
                "reference_loss": _loss_for_sample(model, probe),
                "relation_triples": probe["relation_triples"],
            }
        )
    return memory


def _rank(scores: torch.Tensor, target: int) -> int:
    order = torch.argsort(scores, descending=True)
    positions = (order == target).nonzero(as_tuple=False)
    return int(positions[0].item()) + 1 if positions.numel() else -1


@torch.no_grad()
def _contribution_report(
    model: LocalRelationMarginBertForMaskedLM,
    initial_model: GrammarAttributeBertForMaskedLM,
    samples: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    initial_model.eval()
    tokenizer = model.grammar_attribute_filter.tokenizer
    records: List[Dict[str, object]] = []
    dominant = Counter[str]()
    for sample in samples:
        result = model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            relation_triples=sample["relation_triples"],
            return_component_info=True,
        )
        _, _, components = result
        initial_logits, _ = initial_model(
            input_ids=sample["input_ids"],
            token_type_ids=sample["token_type_ids"],
            attention_mask=sample["attention_mask"],
            apply_grammar_attributes=False,
        )
        position = int(sample["fact_position"])
        target = int(sample["fact_token_id"])
        initial_scores = initial_logits[0, position].detach().cpu()
        base_scores = components["base_logits"][0, position].detach().cpu()
        local_scores = components["local_relation_bias"][0, position].detach().cpu()
        grammar_scores = components["grammar_bias"][0, position].detach().cpu()
        frequency_scores = components["frequency_bias"][0, position].detach().cpu()
        total_scores = components["logits"][0, position].detach().cpu()
        additive = {
            "global_base_update": float(base_scores[target] - initial_scores[target]),
            "local_relation": float(local_scores[target]),
            "grammar_attribute": float(grammar_scores[target]),
            "frequency_prior": float(frequency_scores[target]),
        }
        dominant[max(additive, key=lambda name: abs(additive[name]))] += 1
        records.append(
            {
                "fact_id": sample["fact_id"],
                "fact_token": sample["fact_token"],
                "relation_triples": [
                    list(triple) for triple in sample["relation_triples"]
                ],
                "target_scores": additive,
                "ranks": {
                    "initial_context": _rank(initial_scores, target),
                    "after_global_base": _rank(base_scores, target),
                    "after_local_relation": _rank(base_scores + local_scores, target),
                    "after_grammar": _rank(
                        base_scores + local_scores + grammar_scores, target
                    ),
                    "final": _rank(total_scores, target),
                },
                "margin": _target_margin(
                    total_scores,
                    target,
                    sample.get("hard_negative_ids", []),
                ),
                "top_candidates": [
                    tokenizer.id_to_token[int(candidate)]
                    for candidate in torch.topk(
                        total_scores, min(5, total_scores.numel())
                    ).indices.tolist()
                ],
            }
        )
    if was_training:
        model.train()
    return {
        "samples": len(records),
        "dominant_additive_source_counts": dict(dominant),
        "records": records,
    }


def train(
    fact_dataset: str = DEFAULT_FACT_DATASET,
    init_checkpoint: str = DEFAULT_INIT_CHECKPOINT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_length: int = 128,
    max_context_tokens: int = 8,
    fact_epochs: int = 1,
    fact_repeats: int = 2,
    fact_learning_rate: float = 5e-5,
    relation_learning_rate: float = 5e-4,
    weight_decay: float = 0.0,
    replay_weight: float = 0.5,
    global_background_weight: float = 0.02,
    relation_margin_value: float = 2.0,
    relation_margin_weight: float = 0.25,
    relation_score_scale: Optional[float] = None,
    dynamic_qk_score_scale: Optional[float] = None,
    route_start_layer: Optional[int] = None,
    route_dim: Optional[int] = None,
    relation_ffn_hidden_size: Optional[int] = None,
    relation_ffn_scale: Optional[float] = None,
    relation_ffn_chunk_size: Optional[int] = None,
    correction_probe_every: int = 0,
    soft_conflict_threshold: float = -0.35,
    soft_conflict_scale: float = 0.85,
    mlm_probability: float = 0.15,
    attribute_bias_scale: float = 1.5,
    frequency_content_scale: float = 0.12,
    frequency_function_scale: float = 0.025,
    frequency_punctuation_scale: float = 0.01,
    constrained_frequency_gate: float = 0.25,
    seed: int = 42,
    log_every: int = 100,
) -> str:
    if fact_epochs < 1 or fact_repeats < 1:
        raise ValueError("fact_epochs and fact_repeats must be positive")
    if correction_probe_every < 0 or (correction_probe_every and not relation_ffn_hidden_size):
        raise ValueError("correction probes require an enabled relation FFN and a nonnegative interval")
    if not 0.0 <= global_background_weight <= 1.0:
        raise ValueError("global_background_weight must be between 0 and 1")
    base.set_seed(seed)
    resolved_dataset = base._resolve_path(fact_dataset)
    resolved_init = base._resolve_path(init_checkpoint)
    rows = dataset_impl._load_manifest(resolved_dataset)
    if not os.path.isfile(os.path.join(resolved_init, "pytorch_model.bin")):
        raise FileNotFoundError(f"init checkpoint not found: {resolved_init}")

    initial_model = GrammarAttributeBertForMaskedLM.from_pretrained(resolved_init)
    tokenizer = initial_model.grammar_attribute_filter.tokenizer
    expected_tokenizer = dataset_impl._build_tokenizer(rows)
    if expected_tokenizer.token_to_id != tokenizer.token_to_id:
        raise ValueError("init checkpoint tokenizer does not match fact dataset")

    train_rows = [row for row in rows if str(row["split"]) == "train"]
    dev_rows = [row for row in rows if str(row["split"]) == "dev"]
    test_rows = [row for row in rows if str(row["split"]) == "test"]
    general_texts = dataset_impl._unique_preserve(
        str(row["context"]) for row in train_rows
    )
    frequency_prior, frequency_class_scales, frequencies = _build_frequency_vectors(
        tokenizer,
        train_rows,
        frequency_content_scale,
        frequency_function_scale,
        frequency_punctuation_scale,
    )

    allocator = base.DimensionCombinationAllocator(256)
    frequency_ids = _frequency_ids(frequencies, tokenizer)
    probes, _ = dataset_impl._make_stable_examples(
        general_texts,
        tokenizer,
        frequency_ids,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
    )
    fact_train = dataset_impl._build_fact_samples(
        rows,
        "train",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=True,
    )
    fact_dev = dataset_impl._build_fact_samples(
        rows,
        "dev",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=False,
    )
    fact_test = dataset_impl._build_fact_samples(
        rows,
        "test",
        tokenizer,
        allocator,
        max_length,
        max_context_tokens,
        include_variants=False,
    )
    _ensure_fact_positions(fact_train + fact_dev + fact_test)
    for sample in fact_train + fact_dev + fact_test:
        _attach_hard_negatives(sample, tokenizer)
    if not fact_train or not fact_dev or not fact_test:
        raise RuntimeError(
            f"Could not encode all splits: train={len(fact_train)}, "
            f"dev={len(fact_dev)}, test={len(fact_test)}"
        )

    relation_triples = list(allocator.relation_triples.values())
    model_kwargs = {
        "relation_triples": relation_triples,
        "frequency_prior": frequency_prior,
        "frequency_class_scales": frequency_class_scales,
        "attribute_filter": initial_model.grammar_attribute_filter,
        "attribute_bias_scale": attribute_bias_scale,
        "constrained_frequency_gate": constrained_frequency_gate,
        "margin_value": relation_margin_value,
    }
    optional_model_kwargs = {
        "relation_score_scale": relation_score_scale,
        "dynamic_qk_score_scale": dynamic_qk_score_scale,
        "route_start_layer": route_start_layer,
        "route_dim": route_dim,
        "relation_ffn_hidden_size": relation_ffn_hidden_size,
        "relation_ffn_scale": relation_ffn_scale,
        "relation_ffn_chunk_size": relation_ffn_chunk_size,
    }
    model_kwargs.update(
        {
            key: value
            for key, value in optional_model_kwargs.items()
            if value is not None
        }
    )
    model = LocalRelationMarginBertForMaskedLM(
        initial_model.config,
        tokenizer,
        **model_kwargs,
    ).to(torch.device("cpu"))
    if relation_score_scale is not None:
        actual_relation_score_scale = getattr(model, "relation_score_scale", None)
        if actual_relation_score_scale is None or abs(
            float(actual_relation_score_scale) - float(relation_score_scale)
        ) > 1e-12:
            raise RuntimeError(
                "relation_score_scale configuration did not reach the model: "
                f"requested={relation_score_scale}, "
                f"actual={actual_relation_score_scale}"
            )
    model.bert.load_state_dict(initial_model.bert.state_dict(), strict=True)
    model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
    for parameter in model.grammar_attribute_filter.parameters():
        parameter.requires_grad_(False)

    global_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("relation_adapter.")
    ]
    adapter_parameters = [
        parameter
        for parameter in model.relation_adapter.parameters()
        if parameter.requires_grad
    ]
    all_parameters = global_parameters + adapter_parameters
    optimizer = torch.optim.AdamW(
        [
            {
                "params": global_parameters,
                "lr": fact_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": adapter_parameters,
                "lr": relation_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )

    output_path = base._resolve_path(output_dir)
    if os.path.isfile(os.path.join(output_path, "pytorch_model.bin")):
        raise FileExistsError(
            f"refusing to overwrite existing output checkpoint: {output_path}"
        )
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "before_fact"), exist_ok=True)

    memory = _build_memory(
        general_texts,
        tokenizer,
        frequency_ids,
        allocator,
        max_length,
        max_context_tokens,
        seed,
        mlm_probability,
        model,
    )
    fact_test_before = _fact_metrics(model, fact_test)
    model.save_pretrained(os.path.join(output_path, "before_fact"))
    tokenizer.save_pretrained(os.path.join(output_path, "before_fact"))

    global_step = 0
    total_adapter_conflicts = 0
    total_checked_adapters = 0
    total_replay_overlap = 0
    total_overlap_replays = 0
    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        model.train()
        total_steps = fact_epochs * fact_repeats * len(fact_train)
        for fact_epoch in range(fact_epochs):
            for repeat in range(fact_repeats):
                for fact_index, sample in enumerate(fact_train):
                    global_step += 1
                    before_probe = None
                    if correction_probe_every and global_step % correction_probe_every == 0:
                        from bert_simple.relation_correction import (
                            capture_relation_snapshot, analyze_relation_correction,
                        )
                        probe_inputs = {
                            key: sample[key] for key in
                            ("input_ids", "token_type_ids", "attention_mask")
                        }
                        probe_inputs["relation_triples"] = sample.get("relation_triples", [])
                        before_probe = capture_relation_snapshot(model, **probe_inputs)
                    stats = _run_update(
                        model,
                        all_parameters,
                        global_parameters,
                        adapter_parameters,
                        optimizer,
                        sample,
                        memory,
                        global_step,
                        replay_weight,
                        global_background_weight,
                        relation_margin_weight,
                        relation_margin_value,
                        soft_conflict_threshold,
                        soft_conflict_scale,
                    )
                    total_adapter_conflicts += int(stats["soft_conflict_adapters"])
                    total_checked_adapters += int(stats["checked_relation_adapters"])
                    if stats["replay_relation_adapters"] > 0:
                        total_overlap_replays += 1
                        total_replay_overlap += int(stats["replay_relation_adapters"])
                    local_step = (
                        fact_epoch * fact_repeats * len(fact_train)
                        + repeat * len(fact_train)
                        + fact_index
                        + 1
                    )
                    record = {
                        "stage": "local_relation_fact_memory",
                        "fact_epoch": fact_epoch + 1,
                        "fact_repeat": repeat + 1,
                        "fact_step": local_step,
                        "global_step": global_step,
                        "fact_id": sample["fact_id"],
                        "fact_token": sample["fact_token"],
                        **stats,
                    }
                    if before_probe is not None:
                        record["correction_probe"] = analyze_relation_correction(
                            model, before_probe, int(sample["fact_token_id"]),
                            only_corrected=True, **probe_inputs,
                        )
                        record["correction_probe"]["tokens"] = [
                            tokenizer.id_to_token[int(i)] for i in sample["input_ids"][0]
                        ]
                        before_probe = None
                    log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if local_step % log_every == 0 or local_step == total_steps:
                        log_handle.flush()
                        print(
                            f"fact {fact_epoch + 1}/{fact_epochs}, "
                            f"repeat {repeat + 1}/{fact_repeats}: "
                            f"step={local_step}/{total_steps}, "
                            f"token={sample['fact_token']}, "
                            f"loss={stats['new_loss']:.4f}, "
                            f"margin={stats['target_margin_before_update']:.4f}"
                        )

    model.eval()
    final_memory_loss, final_forgetting = _evaluate_memory(model, memory)
    fact_dev_after = _fact_metrics(model, fact_dev)
    fact_test_after = _fact_metrics(model, fact_test)
    contribution = _contribution_report(model, initial_model, fact_test)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    dataset_impl._save_fact_manifest(
        output_path, fact_train + fact_dev + fact_test
    )
    _save_relation_map(output_path, tokenizer, allocator.relation_triples)

    with open(
        os.path.join(output_path, "frequency_prior.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "role": "low_weight_fallback",
                "prior_is_frozen": True,
                "content_scale": frequency_content_scale,
                "function_scale": frequency_function_scale,
                "punctuation_scale": frequency_punctuation_scale,
                "constrained_frequency_gate": constrained_frequency_gate,
                "top_training_tokens": frequencies.most_common(100),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    used_dimensions = set()
    for triple in allocator.relation_triples.values():
        used_dimensions.update(triple)
    summary = {
        "stage": "bert_mlm_shiji_local_relation_margin_fact_memory",
        "fact_dataset": resolved_dataset,
        "init_checkpoint": resolved_init,
        "device": "cpu",
        "general_contexts_reused": len(general_texts),
        "fact_epochs": fact_epochs,
        "fact_repeats": fact_repeats,
        "optimizer_steps": global_step,
        "vocab_size": len(tokenizer),
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "intermediate_size": int(model.config.intermediate_size),
        "max_length": max_length,
        "train_fact_samples": len(fact_train),
        "dev_fact_samples": len(fact_dev),
        "test_fact_samples": len(fact_test),
        "local_relation_adapter": True,
        "local_adapter_type": "3x3_bilinear_relation_matrix",
        "relation_adapter_count": model.relation_adapter.relation_count,
        "relation_mapping_fixed": True,
        "combination_capacity": allocator.capacity,
        "allocated_relation_combinations": len(allocator.relation_triples),
        "used_hidden_dimensions": len(used_dimensions),
        "global_base_update_weight": global_background_weight,
        "fact_learning_rate": fact_learning_rate,
        "relation_learning_rate": relation_learning_rate,
        "replay_weight": replay_weight,
        "relation_margin_value": relation_margin_value,
        "relation_margin_weight": relation_margin_weight,
        "relation_score_scale": getattr(model, "relation_score_scale", None),
        "dynamic_qk_enabled": bool(
            hasattr(model.relation_adapter, "dynamic_qk")
        ),
        "dynamic_qk_score_scale": getattr(
            model.relation_adapter, "dynamic_qk_score_scale", None
        ),
        "route_start_layer": getattr(model, "route_start_layer", None),
        "route_dim": getattr(model, "route_dim", None),
        "relation_ffn_hidden_size": getattr(model, "relation_ffn_hidden_size", 0),
        "relation_ffn_scale": getattr(model, "relation_ffn_scale", None),
        "correction_probe_every": correction_probe_every,
        "frequency_prior_frozen": True,
        "frequency_prior_learning_rate": 0.0,
        "frequency_content_scale": frequency_content_scale,
        "frequency_function_scale": frequency_function_scale,
        "frequency_punctuation_scale": frequency_punctuation_scale,
        "constrained_frequency_gate": constrained_frequency_gate,
        "attribute_bias_scale": attribute_bias_scale,
        "total_soft_conflict_adapters": total_adapter_conflicts,
        "total_checked_adapters": total_checked_adapters,
        "overlap_replay_steps": total_overlap_replays,
        "total_replay_overlap": total_replay_overlap,
        "fact_test_before": fact_test_before,
        "fact_dev_after": fact_dev_after,
        "fact_test_after": fact_test_after,
        "final_memory_loss": final_memory_loss,
        "final_forgetting": final_forgetting,
        "contribution_report": "contribution_report.json",
        "dominant_additive_source_counts": contribution[
            "dominant_additive_source_counts"
        ],
    }
    with open(
        os.path.join(output_path, "local_relation_margin_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(output_path, "contribution_report.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(contribution, handle, ensure_ascii=False, indent=2)
    print(f"Saved local-relation fact-memory outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Shiji fact memory with local 3x3 relation adapters."
    )
    parser.add_argument("fact_dataset", nargs="?", default=DEFAULT_FACT_DATASET)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=8)
    parser.add_argument("--fact-epochs", type=int, default=1)
    parser.add_argument("--fact-repeats", type=int, default=2)
    parser.add_argument("--fact-learning-rate", type=float, default=5e-5)
    parser.add_argument("--relation-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--global-background-weight", type=float, default=0.02)
    parser.add_argument("--relation-margin-value", type=float, default=2.0)
    parser.add_argument("--relation-margin-weight", type=float, default=0.25)
    parser.add_argument("--relation-score-scale", type=float, default=None)
    parser.add_argument("--dynamic-qk-score-scale", type=float, default=None)
    parser.add_argument("--route-start-layer", type=int, default=None)
    parser.add_argument("--route-dim", type=int, default=None)
    parser.add_argument("--soft-conflict-threshold", type=float, default=-0.35)
    parser.add_argument("--soft-conflict-scale", type=float, default=0.85)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--attribute-bias-scale", type=float, default=1.5)
    parser.add_argument("--frequency-content-scale", type=float, default=0.12)
    parser.add_argument("--frequency-function-scale", type=float, default=0.025)
    parser.add_argument("--frequency-punctuation-scale", type=float, default=0.01)
    parser.add_argument("--constrained-frequency-gate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
