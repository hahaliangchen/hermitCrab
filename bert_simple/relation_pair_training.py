"""Label-independent inputs and losses for grouped relation examples."""

import torch
from torch.nn import functional as F

from .dynamic_qk_model import DynamicQKLocalRelationMarginBertForMaskedLM
from .model import BertConfig
from .tokenizer import SimpleBertTokenizer


CANDIDATE_POLICY = "shared_bank_all_spaces_v1"
DEFAULT_RELATION_MAX_LENGTH = 256


def training_tokenizer(groups, all_splits=False):
    tokenizer = SimpleBertTokenizer()
    tokenizer.add_tokens(["候选", "：", "正文", "、", "[ENT]"])
    for group in groups:
        if not all_splits and group["split"] != "train":
            continue
        for sample in group["samples"]:
            tokenizer.add_tokens(sample["tokens"] + [sample["answer"]] + sample["hard_negatives"])
    return tokenizer


def apply_entity_masking(sample):
    """Replace specific role entities with [ENT] placeholder to prevent memorizing names."""
    tokens = list(sample["tokens"])
    role_spans = sample.get("role_spans", {})
    mask_pos = tokens.index("[MASK]") if "[MASK]" in tokens else -1
    for role, span in role_spans.items():
        start, end = span
        if mask_pos != -1 and start <= mask_pos < end:
            continue
        for idx in range(start, end):
            if idx < len(tokens):
                tokens[idx] = "[ENT]"
    new_sample = dict(sample)
    new_sample["tokens"] = tokens
    return new_sample


def format_candidate_prefix(sample, shuffle=True, rng=None, order=None):
    """Format sample with candidate prefix for extractive pointer scoring."""
    candidates = [sample["answer"]] + list(sample.get("hard_negatives", []))
    if order is not None:
        candidates = [candidates[i] for i in order]
        target_idx = order.index(0)
    elif shuffle:
        order = list(range(len(candidates)))
        if rng is not None:
            rng.shuffle(order)
        else:
            import random
            random.shuffle(order)
        candidates = [candidates[i] for i in order]
        target_idx = order.index(0)
    else:
        target_idx = 0

    prefix_tokens = ["候选", "："]
    candidate_spans = []
    for i, cand in enumerate(candidates):
        start = len(prefix_tokens)
        prefix_tokens.append(cand)
        end = len(prefix_tokens)
        candidate_spans.append((start, end))
        if i < len(candidates) - 1:
            prefix_tokens.append("、")
    prefix_tokens.extend(["正文", "："])

    orig_tokens = list(sample["tokens"])
    full_tokens = prefix_tokens + orig_tokens

    prefix_len = len(prefix_tokens)
    new_role_spans = {}
    for role, span in sample.get("role_spans", {}).items():
        new_role_spans[role] = [span[0] + prefix_len, span[1] + prefix_len]

    new_sample = dict(sample)
    new_sample["tokens"] = full_tokens
    new_sample["candidate_spans"] = candidate_spans
    new_sample["candidate_tokens"] = candidates
    new_sample["target_candidate_index"] = target_idx
    new_sample["role_spans"] = new_role_spans
    return new_sample


def build_model(
    tokenizer,
    hidden_size=256,
    num_layers=4,
    num_heads=4,
    spaces=8,
    max_length=256,
    relation_ffn_chunk_size=256,
    dynamic_qk_score_scale=1.2,
    fusion_dim=None,
    enable_grammar_sparse_gate=False,
    attribute_filter=None,
):
    if (
        spaces < 2
        or num_heads < 1
        or hidden_size < 3
        or hidden_size % num_heads
        or num_layers < 2
        or max_length < 3
    ):
        raise ValueError("need >=2 spaces/layers, hidden_size >= 3 and divisible by heads")
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=hidden_size, num_hidden_layers=num_layers,
        num_attention_heads=num_heads, intermediate_size=4 * hidden_size,
        max_position_embeddings=max_length + 10,
        relative_position_max_distance=max_length,
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    # 1/3 Dimension Fusion + 3D Combinations
    actual_fusion_dim = int(fusion_dim) if fusion_dim is not None else max(1, int(round(hidden_size * 0.375)))
    actual_fusion_dim = max(1, min(actual_fusion_dim, hidden_size - 2))
    comb_dim = max(2, hidden_size - actual_fusion_dim)
    triples = []
    for i in range(spaces):
        d3 = comb_dim + (i % actual_fusion_dim)
        d1 = (2 * i) % comb_dim
        d2 = (d1 + 1 + (i % max(1, comb_dim - 1))) % comb_dim
        triple_set = {d1, d2, d3}
        cand = 0
        while len(triple_set) < 3 and cand < hidden_size:
            triple_set.add(cand)
            cand += 1
        t_sorted = sorted(list(triple_set))
        triples.append((t_sorted[0], t_sorted[1], t_sorted[2]))

    return DynamicQKLocalRelationMarginBertForMaskedLM(
        config, tokenizer, triples, torch.zeros(len(tokenizer)), torch.zeros(len(tokenizer)),
        attribute_filter=attribute_filter,
        relation_ffn_hidden_size=32, route_dim=16,
        relation_ffn_chunk_size=relation_ffn_chunk_size,
        dynamic_qk_score_scale=dynamic_qk_score_scale,
        fusion_dim=actual_fusion_dim,
        enable_grammar_sparse_gate=enable_grammar_sparse_gate,
    )



def visible_inputs(
    model, tokenizer, tokens, max_length=DEFAULT_RELATION_MAX_LENGTH, min_tokens=0
):
    """The only inference constructor; accepts no answer or gold annotations.

    Every visible position has the same small candidate bank. The existing
    contextual router learns the weights. No registry is updated at evaluation.
    """
    if not isinstance(tokens, list) or tokens.count("[MASK]") != 1:
        raise ValueError("expected token list with exactly one MASK")
    if any(not isinstance(t, str) or not t or any(c.isspace() for c in t) for t in tokens):
        raise ValueError("tokens must be nonempty and whitespace-free")
    if set(tokens) & {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}:
        raise ValueError("do not supply special tokens other than MASK")
    if len(tokens) < min_tokens:
        raise ValueError(f"expected at least {min_tokens} content tokens")
    if len(tokens) + 2 > max_length:
        raise ValueError("sample exceeds max_length; refusing to truncate roles or MASK")
    device = next(model.parameters()).device
    ids = [tokenizer.cls_token_id] + [tokenizer.token_to_id.get(t, tokenizer.unk_token_id) for t in tokens] + [tokenizer.sep_token_id]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    return {
        "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "token_type_ids": torch.zeros_like(ids),
        "relation_triples": model.relation_adapter.relation_triples.detach().cpu().tolist(),
        "apply_grammar_attributes": False, "apply_frequency_prior": False,
    }


def sample_logits(model, tokenizer, sample, max_length, min_tokens=0):
    inputs = visible_inputs(
        model, tokenizer, sample["tokens"], max_length, min_tokens
    )
    position = sample["tokens"].index("[MASK]") + 1
    return model(**inputs)[0][0, position]


def sample_pointer_logits(
    model, tokenizer, sample, max_length=DEFAULT_RELATION_MAX_LENGTH, min_tokens=0
):
    """Extractive pointer logits over candidate spans in contextual hidden states."""
    inputs = visible_inputs(
        model, tokenizer, sample["tokens"], max_length, min_tokens
    )
    position = sample["tokens"].index("[MASK]") + 1
    hidden = model.encode_hidden(
        input_ids=inputs["input_ids"],
        token_type_ids=inputs["token_type_ids"],
        attention_mask=inputs["attention_mask"],
        relation_triples=inputs["relation_triples"],
    )
    seq_spans = [(s + 1, e + 1) for s, e in sample["candidate_spans"]]
    return model.pointer_scores(hidden, position, seq_spans)


def paired_loss(
    logits,
    samples,
    pairs,
    tokenizer,
    margin=2.0,
    margin_weight=0.25,
    consistency_weight=0.1,
    candidate_weight=1.0,
    target_weights=None,
    distractor_weights=None,
    pointer_logits=None,
    pointer_weight=1.0,
):
    """Full-vocabulary CE + candidate-CE + explicit-negative softplus margin + symmetric KL.

    Optionally also computes extractive pointer loss if pointer_logits is supplied.
    """
    if min(margin, margin_weight, consistency_weight, candidate_weight, pointer_weight) < 0:
        raise ValueError("loss weights/margin must be nonnegative")
    ces, margins, consistencies, cand_ces = [], [], [], []
    for sample in samples:
        sid = sample["sample_id"]
        scores = logits[sid]
        required = [sample["answer"]] + sample["hard_negatives"]
        missing = set(required) - tokenizer.token_to_id.keys()
        if missing:
            raise ValueError(f"{sid}: training labels/negatives outside vocabulary: {sorted(missing)}")
        target = tokenizer.token_to_id[sample["answer"]]
        negatives = [tokenizer.token_to_id[t] for t in sample["hard_negatives"]]
        ce = F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=scores.device))
        if target_weights is not None:
            ce = ce * target_weights[target].to(device=scores.device)
        ces.append(ce)

        cand_indices = torch.tensor([target] + negatives, device=scores.device)
        cand_scores = scores[cand_indices]
        cand_ce = F.cross_entropy(cand_scores.unsqueeze(0), torch.tensor([0], device=scores.device))
        cand_ces.append(cand_ce)

        diff = margin - scores[target] + scores[negatives]
        margin_loss = F.softplus(diff)
        if distractor_weights is not None:
            margin_loss = margin_loss * distractor_weights[negatives].to(device=scores.device)
        margins.append(margin_loss.mean())
    for pair in pairs:
        if pair["kind"] != "invariance":
            continue
        logp = F.log_softmax(logits[pair["a"]], dim=-1)
        logq = F.log_softmax(logits[pair["b"]], dim=-1)
        consistencies.append(0.5 * ((logp.exp() - logq.exp()) * (logp - logq)).sum())
    ce, negative_margin = torch.stack(ces).mean(), torch.stack(margins).mean()
    cand_ce_loss = torch.stack(cand_ces).mean() if cand_ces else ce * 0.0
    consistency = torch.stack(consistencies).mean() if consistencies else ce * 0.0
    base_loss = ce + candidate_weight * cand_ce_loss + margin_weight * negative_margin + consistency_weight * consistency

    p_ce_loss = ce * 0.0
    p_margin_loss = ce * 0.0
    p_cons_loss = ce * 0.0
    if pointer_logits is not None:
        p_ces, p_margins, p_consistencies = [], [], []
        sample_map = {s["sample_id"]: s for s in samples}
        for sid, p_scores in pointer_logits.items():
            s = sample_map[sid]
            t_idx = s.get("target_candidate_index", 0)
            p_ces.append(F.cross_entropy(p_scores.unsqueeze(0), torch.tensor([t_idx], device=p_scores.device)))
            p_target = p_scores[t_idx]
            if len(p_scores) > 1:
                p_negs = torch.cat([p_scores[:t_idx], p_scores[t_idx + 1:]])
                p_margins.append(F.softplus(margin - p_target + p_negs).mean())
        for pair in pairs:
            if pair["kind"] != "invariance":
                continue
            if pair["a"] in pointer_logits and pair["b"] in pointer_logits:
                plogp = F.log_softmax(pointer_logits[pair["a"]], dim=-1)
                plogq = F.log_softmax(pointer_logits[pair["b"]], dim=-1)
                p_consistencies.append(0.5 * ((plogp.exp() - plogq.exp()) * (plogp - plogq)).sum())
        if p_ces:
            p_ce_loss = torch.stack(p_ces).mean()
        if p_margins:
            p_margin_loss = torch.stack(p_margins).mean()
        if p_consistencies:
            p_cons_loss = torch.stack(p_consistencies).mean()

    pointer_total = p_ce_loss + margin_weight * p_margin_loss + consistency_weight * p_cons_loss
    loss = base_loss + pointer_weight * pointer_total

    return loss, {
        "ce": float(ce.detach()),
        "cand_ce": float(cand_ce_loss.detach()),
        "negative_margin": float(negative_margin.detach()),
        "consistency": float(consistency.detach()),
        "pointer_ce": float(p_ce_loss.detach()),
        "pointer_loss": float(pointer_total.detach()),
        "loss": float(loss.detach()),
    }


@torch.no_grad()
def evaluate(
    model, tokenizer, groups, max_length=DEFAULT_RELATION_MAX_LENGTH, min_tokens=0,
    entity_masking=False, pointer_extraction=False,
):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        result = {}
        for split in ("train", "dev", "test"):
            rows = [g for g in groups if g["split"] == split]
            count = correct = known = oov_inputs = group_correct = 0
            cand_correct = total_negs_tested = negs_defeated_count = 0
            pair_counts = {kind: [0, 0] for kind in ("invariance", "contrast")}
            for group in rows:
                correctness = {}
                for raw_sample in group["samples"]:
                    sample = apply_entity_masking(raw_sample) if entity_masking else raw_sample
                    scores = sample_logits(
                        model, tokenizer, sample, max_length, min_tokens
                    )
                    prediction = tokenizer.id_to_token[int(scores.argmax())]
                    ok = prediction == sample["answer"]
                    correctness[sample["sample_id"]] = ok
                    count += 1
                    correct += int(ok)
                    known += int(sample["answer"] in tokenizer.token_to_id)
                    oov_inputs += int(any(t not in tokenizer.token_to_id for t in sample["tokens"]))

                    if pointer_extraction and hasattr(model, "pointer_scores"):
                        prefixed = format_candidate_prefix(sample, shuffle=False)
                        p_scores = sample_pointer_logits(model, tokenizer, prefixed, max_length, min_tokens)
                        tgt_idx = prefixed["target_candidate_index"]
                        best_cand_idx = int(p_scores.argmax())
                        cand_correct += int(best_cand_idx == tgt_idx)
                        target_s = p_scores[tgt_idx]
                        if len(p_scores) > 1:
                            neg_scores = torch.cat([p_scores[:tgt_idx], p_scores[tgt_idx + 1:]])
                            defeated = sum(1 for ns in neg_scores if target_s > ns)
                            total_negs_tested += len(neg_scores)
                            negs_defeated_count += int(defeated)
                    else:
                        cand_tokens = [sample["answer"]] + sample.get("hard_negatives", [])
                        cand_ids = [tokenizer.token_to_id[c] for c in cand_tokens if c in tokenizer.token_to_id]
                        if cand_ids and (sample["answer"] in tokenizer.token_to_id):
                            target_tid = tokenizer.token_to_id[sample["answer"]]
                            cand_scores = scores[cand_ids]
                            cand_best_idx = int(cand_scores.argmax())
                            cand_correct += int(cand_ids[cand_best_idx] == target_tid)
                            target_s = scores[target_tid]
                            neg_cids = [cid for cid in cand_ids if cid != target_tid]
                            defeated = sum(1 for cid in neg_cids if target_s > scores[cid])
                            total_negs_tested += len(neg_cids)
                            negs_defeated_count += defeated

                group_correct += int(all(correctness.values()))
                for pair in group["pairs"]:
                    pair_counts[pair["kind"]][0] += 1
                    pair_counts[pair["kind"]][1] += int(correctness[pair["a"]] and correctness[pair["b"]])
            result[split] = {
                "samples": count, "correct": correct, "top1": correct / count if count else None,
                "candidate_top1": cand_correct / count if count else None,
                "candidate_pairwise_accuracy": negs_defeated_count / total_negs_tested if total_negs_tested else None,
                "known_answer_count": known, "oov_answer_count": count - known,
                "input_oov_samples": oov_inputs, "groups": len(rows),
                "all_correct_groups": group_correct,
                "pairs": {kind: {"count": n, "both_correct": ok, "accuracy": ok/n if n else None}
                          for kind, (n, ok) in pair_counts.items()},
            }
        return result
    finally:
        for module, mode in modes:
            module.training = mode
