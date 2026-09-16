"""Same-input before/after traces and scoped activation rollback experiments.

No optimizer changes or persistent model hooks. Rollback replaces only the
selected FFN bias entries; all downstream activations are recomputed.
This measures a model-internal intervention, not a historical causal fact.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import itertools

import torch


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_copy(item) for item in value)
    return value


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


@contextmanager
def _evaluation(model):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in states:
            module.training = training


def _module(model):
    module = getattr(model.relation_adapter.dynamic_qk, "structured_scores", None)
    if module is None:
        raise ValueError("enable relation_ffn_hidden_size before tracing")
    return module


@dataclass
class RelationSnapshot:
    inputs: dict
    mask_position: int
    logits: torch.Tensor
    scores: dict
    attentions: dict
    scale: tuple


def _inputs(inputs):
    allowed = {"input_ids", "attention_mask", "token_type_ids", "relation_triples",
               "candidate_space_mask", "apply_grammar_attributes", "apply_frequency_prior"}
    if set(inputs) - allowed:
        raise ValueError("trace accepts inference inputs only; remove labels/return flags")
    return inputs


def _run(model, inputs, old=None, edges=()):
    inputs = _inputs(inputs)
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("correction probes currently require batch size 1")
    masks = (ids[0] == model.config.mask_token_id).nonzero().flatten()
    if masks.numel() != 1:
        raise ValueError("correction probes require exactly one MASK")
    position = int(masks.item())
    module = _module(model)
    scores = {}

    def hook(_module, args, output):
        layer = int(args[0])
        selected = [edge for edge in edges if edge[0] == layer]
        if selected:
            output = output.clone()
            for _, head, key in selected:
                output[0, head, position, key] = old.scores[layer][0, head, position, key].to(output)
        scores[layer] = output.detach().cpu().clone()
        return output

    handle = module.register_forward_hook(hook)
    try:
        with _evaluation(model):
            logits, attentions = model(**inputs, output_attentions=True)[:2]
    finally:
        handle.remove()
    return RelationSnapshot(
        _copy(inputs), position, logits[0, position].detach().cpu().clone(), scores,
        {i: value.detach().cpu().clone() for i, value in enumerate(attentions)},
        (module.scale, model.dynamic_qk_score_scale),
    )


def capture_relation_snapshot(model, **inputs):
    """Call before optimizer.step, on a fixed probe; eval disables dropout."""
    return _run(model, inputs)


def analyze_relation_correction(model, before, target_id, top_k=2, only_corrected=False, **inputs):
    """After optimizer.step: capture, select, and test singles + pairs.

    Candidates are MASK->key edges (layer, head, key), with distinct key
    positions. Both increased and decreased signals enter the ranking.
    This bounded heuristic can miss indirect paths and higher-order effects.
    """
    if top_k not in (2, 3):
        raise ValueError("top_k must be 2 or 3")
    if not _equal(before.inputs, _copy(_inputs(inputs))):
        raise ValueError("before/after probes must have identical inputs and candidates")
    after = _run(model, inputs)
    if before.scale != after.scale or before.scores.keys() != after.scores.keys():
        raise ValueError("trace scales and active layers must match")
    if not 0 <= target_id < after.logits.numel():
        raise ValueError("target_id outside vocabulary")
    negatives = before.logits.clone()
    negatives[target_id] = -torch.inf
    negative_id = int(negatives.argmax())

    def margin(snapshot):
        return float(snapshot.logits[target_id] - snapshot.logits[negative_id])

    report = {
        "target_id": int(target_id), "negative_id": negative_id,
        "before_prediction": int(before.logits.argmax()),
        "after_prediction": int(after.logits.argmax()),
        "corrected": int(before.logits.argmax()) != target_id and int(after.logits.argmax()) == target_id,
        "before_margin": margin(before), "after_margin": margin(after),
        "target_logit_before": float(before.logits[target_id]),
        "target_logit_after": float(after.logits[target_id]),
        "ffn_scale": after.scale[0], "dynamic_qk_scale": after.scale[1],
        "singles": [], "pairs": [],
        "intervention": "restore_ffn_bias_before_common_dynamic_scale",
        "scope": "sample_specific_internal_evidence_not_semantic_causality",
    }
    if only_corrected and not report["corrected"]:
        report["skip_reason"] = "not_a_wrong_to_right_transition"
        return report

    ids = inputs["input_ids"][0].detach().cpu()
    valid = inputs.get("attention_mask", ids.ne(model.config.pad_token_id).unsqueeze(0))[0].detach().cpu().bool()
    tokenizer = model.grammar_attribute_filter.tokenizer
    excluded = {getattr(tokenizer, name, None) for name in
                ("pad_token_id", "mask_token_id", "cls_token_id", "sep_token_id", "unk_token_id")}
    valid &= torch.tensor([int(token) not in excluded for token in ids])
    candidates = []
    pos = after.mask_position
    for layer, new_scores in after.scores.items():
        delta = new_scores[0, :, pos] - before.scores[layer][0, :, pos]
        da = after.attentions[layer][0, :, pos] - before.attentions[layer][0, :, pos]
        # Independently normalize heuristic signals, never label them causal.
        priority = delta.abs() / delta.abs().max().clamp_min(1e-8)
        priority += da.abs() / da.abs().max().clamp_min(1e-8)
        for head, key in itertools.product(range(delta.shape[0]), range(delta.shape[1])):
            if valid[key] and abs(float(delta[head, key])) > 1e-12:
                candidates.append((float(priority[head, key]), (layer, head, key),
                                   float(delta[head, key]), float(da[head, key])))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected, seen = [], set()
    for item in candidates:
        if item[1][2] not in seen:
            selected.append(item)
            seen.add(item[1][2])
        if len(selected) == top_k:
            break
    full = margin(after)
    singles, pairs = [], []
    for _, edge, delta, da in selected:
        rolled = _run(model, inputs, before, [edge])
        layer, head, key = edge
        singles.append({"edge": list(edge), "token_id": int(ids[edge[2]]),
                        "relation_score_before": float(before.scores[layer][0, head, pos, key]),
                        "relation_score_after": float(after.scores[layer][0, head, pos, key]),
                        "attention_before": float(before.attentions[layer][0, head, pos, key]),
                        "attention_after": float(after.attentions[layer][0, head, pos, key]),
                        "delta_relation_score": delta, "delta_attention": da,
                        "margin_drop": full - margin(rolled)})
    for i, j in itertools.combinations(range(len(selected)), 2):
        rolled = _run(model, inputs, before, [selected[i][1], selected[j][1]])
        drop = full - margin(rolled)
        pairs.append({"members": [i, j], "margin_drop": drop,
                      "nonadditivity": drop - singles[i]["margin_drop"] - singles[j]["margin_drop"]})
    report.update(singles=singles, pairs=pairs)
    return report
