"""Train validated JSONL pairs without answer-derived relation candidates."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

"""Train validated JSONL pairs without answer-derived relation candidates."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from bert_simple.dynamic_qk_model import DynamicQKLocalRelationMarginBertForMaskedLM
from bert_simple.tokenizer import SimpleBertTokenizer
from bert_simple.relation_pair_training import (
    CANDIDATE_POLICY, build_model, training_tokenizer, visible_inputs,
    sample_logits, sample_pointer_logits, paired_loss, evaluate,
    apply_entity_masking, format_candidate_prefix,
)
from bert_simple.relation_correction import capture_relation_snapshot, analyze_relation_correction
from validate_relation_training_data import (
    DEFAULT_MIN_RELATION_TOKENS,
    DEFAULT_RELATION_MAX_LENGTH,
    load_groups,
    validate_groups,
)



def train(args):
    if args.epochs < 1 or args.learning_rate <= 0 or args.max_length < 3 or args.probe_every < 0:
        raise ValueError("invalid epochs/lr/max_length/probe interval")
    if args.min_tokens < 1 or args.max_length < args.min_tokens + 2:
        raise ValueError("max-length must leave room for min-tokens plus CLS/SEP")
    if min(args.margin, args.margin_weight, args.consistency_weight) < 0:
        raise ValueError("loss weights/margin must be nonnegative")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(args.threads)
    groups = list(load_groups(args.dataset))
    validation = validate_groups(groups, min_tokens=args.min_tokens)
    training_groups = [g for g in groups if g["split"] == "train"]
    if not training_groups:
        raise ValueError("dataset has no training groups")
    if args.init_checkpoint:
        checkpoint = Path(args.init_checkpoint)
        config = json.loads((checkpoint / "relation_pair_config.json").read_text())
        if config.get("candidate_policy") != CANDIDATE_POLICY:
            raise ValueError("warm start requires a label-independent pair-training checkpoint")
        model = DynamicQKLocalRelationMarginBertForMaskedLM.from_pretrained(str(checkpoint))
        tokenizer = SimpleBertTokenizer.from_pretrained(str(checkpoint))
        if not model.relation_ffn_hidden_size:
            raise ValueError("checkpoint has no relation FFN")
    else:
        tokenizer = training_tokenizer(groups, all_splits=(args.vocab_scope == "all_splits"))
        model = build_model(
            tokenizer,
            args.hidden_size,
            2,
            4,
            args.spaces,
            args.max_length,
            relation_ffn_chunk_size=args.relation_ffn_chunk_size,
            dynamic_qk_score_scale=args.dynamic_qk_score_scale,
            fusion_dim=getattr(args, "fusion_dim", None),
        )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    if args.max_length > int(model.config.max_position_embeddings):
        raise ValueError(
            "max-length exceeds the checkpoint positional capacity; rebuild the model "
            "with the requested length"
        )
    if (
        model.config.position_embedding_type == "relative"
        and args.max_length > int(model.config.relative_position_max_distance)
    ):
        raise ValueError(
            "max-length exceeds the checkpoint relative-position capacity; "
            "rebuild the model with the requested length"
        )
    # Reject train OOV rather than silently supervising UNK. Dev/test OOV is
    # counted in evaluation, never used to expand the training vocabulary.
    validate_groups(training_groups, tokenizer.token_to_id, args.min_tokens)
    validate_groups(groups, min_tokens=args.min_tokens)
    for group in groups:
        for sample in group["samples"]:
            visible_inputs(
                model,
                tokenizer,
                sample["tokens"],
                args.max_length,
                args.min_tokens,
            )
    for parameter in model.grammar_attribute_filter.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), candidate_policy=CANDIDATE_POLICY,
                  dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                  vocabulary_source="checkpoint" if args.init_checkpoint else "train_samples_only",
                  validation=validation, torch_version=str(torch.__version__))
    (output / "relation_pair_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    target_weights = None
    distractor_weights = None
    if getattr(args, "damping", True):
        from collections import Counter
        import math
        ans_counts = Counter()
        for g in training_groups:
            for s in g["samples"]:
                ans_counts[s["answer"]] += 1
        
        raw_weights = [1.0 / math.log(math.e + ans_counts[s["answer"]]) for g in training_groups for s in g["samples"]]
        avg_weight = sum(raw_weights) / len(raw_weights) if raw_weights else 1.0

        target_weights = torch.ones(len(tokenizer), dtype=torch.float32)
        distractor_weights = torch.ones(len(tokenizer), dtype=torch.float32)
        for token_str, freq in ans_counts.items():
            tid = tokenizer.token_to_id.get(token_str)
            if tid is not None:
                target_weights[tid] = (1.0 / math.log(math.e + freq)) / avg_weight
                distractor_weights[tid] = math.log(math.e + freq)
        target_weights = target_weights.to(device)
        distractor_weights = distractor_weights.to(device)

    step = 0
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            order = list(training_groups)
            rng.shuffle(order)
            for group in order:
                step += 1
                model.train()
                probe = group["samples"][0]
                probe_inputs = visible_inputs(
                    model,
                    tokenizer,
                    probe["tokens"],
                    args.max_length,
                    args.min_tokens,
                )
                before = capture_relation_snapshot(model, **probe_inputs) if args.probe_every and step % args.probe_every == 0 else None
                optimizer.zero_grad(set_to_none=True)
                use_entity_masking = getattr(args, "entity_masking", True)
                use_pointer = getattr(args, "pointer_extraction", True)
                input_samples = [apply_entity_masking(s) if use_entity_masking else s for s in group["samples"]]
                logits = {
                    s["sample_id"]: sample_logits(
                        model, tokenizer, s, args.max_length, args.min_tokens
                    )
                    for s in input_samples
                }

                pointer_logits = None
                prefixed_samples = []
                if use_pointer and hasattr(model, "pointer_scores"):
                    pointer_logits = {}
                    first_sample = group["samples"][0]
                    cands = [first_sample["answer"]] + list(first_sample.get("hard_negatives", []))
                    order_indices = list(range(len(cands)))
                    rng.shuffle(order_indices)
                    for s in input_samples:
                        prefixed = format_candidate_prefix(s, order=order_indices)
                        prefixed_samples.append(prefixed)
                        pointer_logits[s["sample_id"]] = sample_pointer_logits(
                            model, tokenizer, prefixed, args.max_length, args.min_tokens
                        )

                loss, stats = paired_loss(
                    logits,
                    prefixed_samples if pointer_logits is not None else group["samples"],
                    group["pairs"],
                    tokenizer,
                    args.margin,
                    args.margin_weight,
                    args.consistency_weight,
                    candidate_weight=getattr(args, "candidate_weight", 1.0),
                    target_weights=target_weights,
                    distractor_weights=distractor_weights,
                    pointer_logits=pointer_logits,
                    pointer_weight=getattr(args, "pointer_weight", 2.0),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Orthogonal error subtraction: update eraser on wrong predictions
                if getattr(args, "error_subtraction", True) and hasattr(model, "relation_adapter") and hasattr(model.relation_adapter.dynamic_qk, "structured_scores"):
                    eraser = model.relation_adapter.dynamic_qk.structured_scores.ffn.eraser
                    for s in group["samples"]:
                        pred_id = int(logits[s["sample_id"]].argmax())
                        target_id = tokenizer.token_to_id[s["answer"]]
                        if pred_id != target_id:
                            with torch.no_grad():
                                dec_w = model.lm_head.weight
                                diff = dec_w[pred_id] - dec_w[target_id]
                                if diff.shape[-1] < eraser.hidden_size:
                                    diff = torch.nn.functional.pad(diff, (0, eraser.hidden_size - diff.shape[-1]))
                                else:
                                    diff = diff[:eraser.hidden_size]
                                eraser.record_diff(diff.unsqueeze(0), torch.zeros_like(diff).unsqueeze(0))

                record = dict(step=step, epoch=epoch + 1, group_id=group["group_id"], **stats)
                if before is not None:
                    record["correction_probe"] = analyze_relation_correction(
                        model, before, tokenizer.token_to_id[probe["answer"]], only_corrected=True, **probe_inputs)
                    record["correction_probe"].update(sample_id=probe["sample_id"], tokens=probe["tokens"])
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            p_ce_disp = stats.get('pointer_ce', stats.get('cand_ce', 0.0))
            print(f"epoch={epoch+1} steps={step} loss={stats['loss']:.5f} ptr_ce={p_ce_disp:.5f}", flush=True)
    model.eval()
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    metrics = evaluate(
        model, tokenizer, groups, args.max_length, args.min_tokens,
        entity_masking=getattr(args, "entity_masking", True),
        pointer_extraction=getattr(args, "pointer_extraction", True),
    )
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("dataset")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--init-checkpoint", help="warm start a NEW pair-training checkpoint; optimizer restarts")
    result.add_argument("--epochs", type=int, default=25)
    result.add_argument("--hidden-size", type=int, default=64)
    result.add_argument("--spaces", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--margin", type=float, default=3.5)
    result.add_argument("--margin-weight", type=float, default=0.5)
    result.add_argument("--consistency-weight", type=float, default=0.1)
    result.add_argument("--candidate-weight", type=float, default=1.0)
    result.add_argument("--dynamic-qk-score-scale", type=float, default=1.2)
    result.add_argument("--damping", action="store_true", default=True)
    result.add_argument("--no-damping", dest="damping", action="store_false")
    result.add_argument("--entity-masking", action="store_true", default=True)
    result.add_argument("--no-entity-masking", dest="entity_masking", action="store_false")
    result.add_argument("--error-subtraction", action="store_true", default=True)
    result.add_argument("--no-error-subtraction", dest="error_subtraction", action="store_false")
    result.add_argument("--pointer-extraction", action="store_true", default=True)
    result.add_argument("--no-pointer-extraction", dest="pointer_extraction", action="store_false")
    result.add_argument("--pointer-weight", type=float, default=2.0)
    result.add_argument("--fusion-dim", type=int, default=None)
    result.add_argument("--max-length", type=int, default=DEFAULT_RELATION_MAX_LENGTH)
    result.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_RELATION_TOKENS)
    result.add_argument("--probe-every", type=int, default=100)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--vocab-scope", choices=["train_only", "all_splits"], default="all_splits")
    result.add_argument("--relation-ffn-chunk-size", type=int, default=256)
    result.add_argument("--device", default=None, help="Device to use ('cuda' or 'cpu'). Defaults to cuda if available.")
    return result


if __name__ == "__main__":
    train(parser().parse_args())
