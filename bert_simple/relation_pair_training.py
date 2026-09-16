"""Label-independent inputs and losses for grouped relation examples."""

import torch
from torch.nn import functional as F

from .dynamic_qk_model import DynamicQKLocalRelationMarginBertForMaskedLM
from .model import BertConfig
from .tokenizer import SimpleBertTokenizer


CANDIDATE_POLICY = "shared_bank_all_spaces_v1"


def training_tokenizer(groups):
    tokenizer = SimpleBertTokenizer()
    for group in groups:
        if group["split"] != "train":
            continue
        for sample in group["samples"]:
            tokenizer.add_tokens(sample["tokens"] + [sample["answer"]] + sample["hard_negatives"])
    return tokenizer


def build_model(tokenizer, hidden_size=64, num_layers=2, num_heads=4, spaces=4):
    if spaces < 2 or num_heads < 1 or hidden_size < 3 * spaces or hidden_size % num_heads or num_layers < 2:
        raise ValueError("need >=2 spaces/layers, hidden_size >= 3*spaces and divisible by heads")
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=hidden_size, num_hidden_layers=num_layers,
        num_attention_heads=num_heads, intermediate_size=4 * hidden_size,
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    # Fixed latent spaces, not labels, token pairs, or per-fact IDs.
    triples = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(spaces)]
    return DynamicQKLocalRelationMarginBertForMaskedLM(
        config, tokenizer, triples, torch.zeros(len(tokenizer)), torch.zeros(len(tokenizer)),
        relation_ffn_hidden_size=32, route_dim=16,
    )


def visible_inputs(model, tokenizer, tokens, max_length=128):
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


def sample_logits(model, tokenizer, sample, max_length):
    inputs = visible_inputs(model, tokenizer, sample["tokens"], max_length)
    position = sample["tokens"].index("[MASK]") + 1
    return model(**inputs)[0][0, position]


def paired_loss(logits, samples, pairs, tokenizer, margin=2.0,
                margin_weight=0.25, consistency_weight=0.1):
    """Full-vocabulary CE + explicit-negative softplus margin + symmetric KL.

    Only invariance pairs receive KL. Contrast pairs retain their own labels;
    no ground-truth role/target-slot fields enter a model forward pass.
    """
    if min(margin, margin_weight, consistency_weight) < 0:
        raise ValueError("loss weights/margin must be nonnegative")
    ces, margins, consistencies = [], [], []
    for sample in samples:
        sid = sample["sample_id"]
        scores = logits[sid]
        required = [sample["answer"]] + sample["hard_negatives"]
        missing = set(required) - tokenizer.token_to_id.keys()
        if missing:
            raise ValueError(f"{sid}: training labels/negatives outside vocabulary: {sorted(missing)}")
        target = tokenizer.token_to_id[sample["answer"]]
        negatives = [tokenizer.token_to_id[t] for t in sample["hard_negatives"]]
        ces.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=scores.device)))
        margins.append(F.softplus(margin - scores[target] + scores[negatives]).mean())
    for pair in pairs:
        if pair["kind"] != "invariance":
            continue
        logp = F.log_softmax(logits[pair["a"]], dim=-1)
        logq = F.log_softmax(logits[pair["b"]], dim=-1)
        consistencies.append(0.5 * ((logp.exp() - logq.exp()) * (logp - logq)).sum())
    ce, negative_margin = torch.stack(ces).mean(), torch.stack(margins).mean()
    consistency = torch.stack(consistencies).mean() if consistencies else ce * 0.0
    loss = ce + margin_weight * negative_margin + consistency_weight * consistency
    return loss, {"ce": float(ce.detach()), "negative_margin": float(negative_margin.detach()),
                  "consistency": float(consistency.detach()), "loss": float(loss.detach())}


@torch.no_grad()
def evaluate(model, tokenizer, groups, max_length=128):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        result = {}
        for split in ("train", "dev", "test"):
            rows = [g for g in groups if g["split"] == split]
            count = correct = known = oov_inputs = group_correct = 0
            pair_counts = {kind: [0, 0] for kind in ("invariance", "contrast")}
            for group in rows:
                correctness = {}
                for sample in group["samples"]:
                    scores = sample_logits(model, tokenizer, sample, max_length)
                    prediction = tokenizer.id_to_token[int(scores.argmax())]
                    ok = prediction == sample["answer"]
                    correctness[sample["sample_id"]] = ok
                    count += 1
                    correct += int(ok)
                    known += int(sample["answer"] in tokenizer.token_to_id)
                    oov_inputs += int(any(t not in tokenizer.token_to_id for t in sample["tokens"]))
                group_correct += int(all(correctness.values()))
                for pair in group["pairs"]:
                    pair_counts[pair["kind"]][0] += 1
                    pair_counts[pair["kind"]][1] += int(correctness[pair["a"]] and correctness[pair["b"]])
            result[split] = {
                "samples": count, "correct": correct, "top1": correct / count if count else None,
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
