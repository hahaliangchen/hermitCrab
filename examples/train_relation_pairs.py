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
    sample_logits, paired_loss, evaluate,
)
from bert_simple.relation_correction import capture_relation_snapshot, analyze_relation_correction
from validate_relation_training_data import load_groups, validate_groups


def train(args):
    if args.epochs < 1 or args.learning_rate <= 0 or args.max_length < 3 or args.probe_every < 0:
        raise ValueError("invalid epochs/lr/max_length/probe interval")
    if min(args.margin, args.margin_weight, args.consistency_weight) < 0:
        raise ValueError("loss weights/margin must be nonnegative")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(args.threads)
    groups = list(load_groups(args.dataset))
    validation = validate_groups(groups)
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
        tokenizer = training_tokenizer(groups)
        model = build_model(tokenizer, args.hidden_size, 2, 4, args.spaces)
    # Reject train OOV rather than silently supervising UNK. Dev/test OOV is
    # counted in evaluation, never used to expand the training vocabulary.
    validate_groups(training_groups, tokenizer.token_to_id)
    for group in groups:
        for sample in group["samples"]:
            visible_inputs(model, tokenizer, sample["tokens"], args.max_length)
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
    step = 0
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            order = list(training_groups)
            rng.shuffle(order)
            for group in order:
                step += 1
                model.train()
                probe = group["samples"][0]
                probe_inputs = visible_inputs(model, tokenizer, probe["tokens"], args.max_length)
                before = capture_relation_snapshot(model, **probe_inputs) if args.probe_every and step % args.probe_every == 0 else None
                optimizer.zero_grad(set_to_none=True)
                logits = {s["sample_id"]: sample_logits(model, tokenizer, s, args.max_length) for s in group["samples"]}
                loss, stats = paired_loss(logits, group["samples"], group["pairs"], tokenizer,
                                          args.margin, args.margin_weight, args.consistency_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                record = dict(step=step, epoch=epoch+1, group_id=group["group_id"], **stats)
                if before is not None:
                    record["correction_probe"] = analyze_relation_correction(
                        model, before, tokenizer.token_to_id[probe["answer"]], only_corrected=True, **probe_inputs)
                    record["correction_probe"].update(sample_id=probe["sample_id"], tokens=probe["tokens"])
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print(f"epoch={epoch+1} steps={step} loss={stats['loss']:.5f}", flush=True)
    model.eval()
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    metrics = evaluate(model, tokenizer, groups, args.max_length)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("dataset")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--init-checkpoint", help="warm start a NEW pair-training checkpoint; optimizer restarts")
    result.add_argument("--epochs", type=int, default=10)
    result.add_argument("--hidden-size", type=int, default=64)
    result.add_argument("--spaces", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--margin", type=float, default=2.0)
    result.add_argument("--margin-weight", type=float, default=0.25)
    result.add_argument("--consistency-weight", type=float, default=0.1)
    result.add_argument("--max-length", type=int, default=128)
    result.add_argument("--probe-every", type=int, default=100)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--threads", type=int, default=1)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
