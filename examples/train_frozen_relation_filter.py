"""Train the structured relation FFN while keeping the main BERT frozen.

The base checkpoint supplies contextual hidden states.  Only the contextual
route, the dynamic 3x3 Q/K bank, and ``Structured3DRelationFFN`` are updated.
Role spans in the relation-pair manifest provide weak supervision: the
masked position should score annotated context roles above background tokens.

This is a sidecar pretraining stage.  It deliberately does not update the
main BERT, its embeddings, its general 256 -> 1024 -> 256 FFN, or its MLM
head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from bert_simple.dynamic_qk_model import (  # noqa: E402
    DynamicQKRelationBilinearAdapter,
)
from bert_simple.context_spaces import (  # noqa: E402
    DEFAULT_RELATION_CANDIDATE_COUNT,
    build_context_space_triples,
)
from bert_simple.grammar_attribute_model import (  # noqa: E402
    GrammarAttributeBertForMaskedLM,
)
from bert_simple.structured_relation import StructuredRelationScores  # noqa: E402
from bert_simple.tokenizer import SimpleBertTokenizer  # noqa: E402
from validate_relation_training_data import load_groups, validate_groups  # noqa: E402


DEFAULT_BASE_CHECKPOINT = (
    ROOT / "outputs/bert-mlm-fact-memory-shiji-grammar-attribute-256/before_fact"
)
DEFAULT_DATASET = ROOT / "data/shiji/manifests/relation_training_v2_draft.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/frozen-relation-filter-stage1"

Triple = Tuple[int, int, int]


class FrozenStructuredRelationFilter(nn.Module):
    """A trainable relation sidecar over frozen BERT hidden states."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mask_token_id: int,
        initializer_range: float,
        spaces: int,
        route_dim: int,
        route_start_layer: int,
        relation_ffn_hidden_size: int,
        relation_ffn_scale: float,
        relation_ffn_chunk_size: int,
    ) -> None:
        super().__init__()
        fixed_triples = build_context_space_triples(
            hidden_size,
            candidate_count=int(spaces),
        )
        if int(spaces) != len(fixed_triples):
            raise ValueError(
                "spaces must equal the full fixed context bank size "
                f"({len(fixed_triples)} for hidden_size={hidden_size})"
            )
        if not 1 <= route_start_layer < num_layers:
            raise ValueError("route_start_layer must leave a dynamic layer")
        self.triples: Tuple[Triple, ...] = tuple(fixed_triples)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        valid = attention_mask.bool()
        candidate_mask = torch.zeros(
            hidden_states.size(0), hidden_states.size(1), len(self.triples),
            dtype=torch.bool, device=hidden_states.device,
        )
        candidate_mask[...] = valid.unsqueeze(-1)
        routes = self.adapter.route_weights(hidden_states, candidate_mask)
        layer_scores = []
        bank = self.adapter.dynamic_qk
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
        # Average heads and dynamic layers: this trains the shared structured
        # FFN without allowing one head/layer to dominate the weak labels.
        scores = torch.stack(layer_scores, dim=0).mean(dim=(0, 1))
        return scores, routes

    def forward_groups(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_positions: Sequence[int],
        context_groups: Sequence[Sequence[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score complete context groups instead of individual token edges."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        valid = attention_mask.bool()
        candidate_mask = torch.zeros(
            hidden_states.size(0), hidden_states.size(1), len(self.triples),
            dtype=torch.bool, device=hidden_states.device,
        )
        candidate_mask[...] = valid.unsqueeze(-1)
        routes = self.adapter.route_weights(hidden_states, candidate_mask)
        layer_scores = []
        bank = self.adapter.dynamic_qk
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
                )
            )
        scores = torch.stack(layer_scores, dim=0).mean(dim=(0, 1))
        return scores, routes


def _sample_positions(sample: Dict[str, object]) -> Tuple[int, List[int], List[int]]:
    """Build complete relation/background groups in positions including [CLS].

    A role span is the high-confidence core.  We also retain every
    non-punctuation token between that core and the MASK, so connectors such as
    ``封`` and ``为`` remain part of the relation context.  If no role span is
    available, the whole local clause is used as a weak context group rather
    than an arbitrary eight-token candidate list.
    """
    tokens = list(sample["tokens"])
    mask_index = int(tokens.index("[MASK]"))
    mask = mask_index + 1
    punctuation = {"，", "。", "？", ",", ".", "?", "！", "!", "；", ";"}
    positive_data = set()
    for span in sample.get("role_spans", {}).values():
        start, end = int(span[0]), int(span[1])
        positive_data.update(
            index
            for index in range(start, end)
            if index != mask_index and tokens[index] not in punctuation
        )

    if positive_data:
        # Include the complete connective span from the nearest role member to
        # MASK.  This is still local and auditable, but does not throw away
        # relation members merely because they are not separately annotated.
        lower = min(min(positive_data), mask_index)
        upper = max(max(positive_data), mask_index)
        positive_data.update(
            index
            for index in range(lower, upper + 1)
            if index != mask_index and tokens[index] not in punctuation
        )
    else:
        # No role metadata is available for some long-context variants.  Use
        # the complete clause containing MASK as weak context; the distant
        # clauses then form the background group.
        clause_start = mask_index
        while clause_start > 0 and tokens[clause_start - 1] not in punctuation:
            clause_start -= 1
        clause_end = mask_index + 1
        while clause_end < len(tokens) and tokens[clause_end] not in punctuation:
            clause_end += 1
        positive_data.update(
            index
            for index in range(clause_start, clause_end)
            if index != mask_index and tokens[index] not in punctuation
        )
    positive = {index + 1 for index in positive_data}

    background = set()
    for span in sample.get("background_spans", []):
        start, end = int(span[0]), int(span[1])
        background.update(range(start + 1, end + 1))

    # When no explicit background was added, every non-positive visible token
    # is a weak negative.  For long variants this makes the relation group
    # compete against the remaining context instead of selecting one token.
    negative = {
        index + 1
        for index, token in enumerate(tokens)
        if index + 1 != mask
        and index + 1 not in positive
        and token not in {"[MASK]", *punctuation}
    }
    negative.update(background - positive)
    positive = {index for index in positive if index != mask}
    negative = {index for index in negative if index != mask and index not in positive}
    if not positive or not negative:
        raise ValueError(f"sample has no usable positive/negative positions: {sample['sample_id']}")
    return mask, sorted(positive), sorted(negative)


def _encode_sample(
    tokenizer: SimpleBertTokenizer,
    sample: Dict[str, object],
    max_length: int,
) -> Dict[str, object]:
    tokens = list(sample["tokens"])
    if len(tokens) + 2 > max_length:
        raise ValueError(f"sample exceeds max_length: {sample['sample_id']}")
    missing = [token for token in tokens if token not in tokenizer.token_to_id]
    if missing:
        raise ValueError(f"base tokenizer OOV in {sample['sample_id']}: {sorted(set(missing))}")
    encoded = tokenizer.encode(
        " ".join(tokens),
        max_length=max_length,
        truncation=False,
        return_tensors="pt",
    )
    mask_position, positive, negative = _sample_positions(sample)
    return {
        "sample_id": str(sample["sample_id"]),
        "fact_id": str(sample["fact_id"]),
        "split": str(sample.get("split", "")),
        "input_ids": encoded["input_ids"][0],
        "attention_mask": encoded["attention_mask"][0],
        "mask_position": mask_position,
        "positive": positive,
        "negative": negative,
    }


@torch.no_grad()
def cache_hidden_states(
    base_model: GrammarAttributeBertForMaskedLM,
    examples: Sequence[Dict[str, object]],
    batch_size: int,
    pad_token_id: int,
) -> None:
    """Run the frozen main BERT once and cache detached hidden states."""
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


def _example_loss(
    sidecar: FrozenStructuredRelationFilter,
    example: Dict[str, object],
    margin: float,
) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
    hidden = example["hidden"].unsqueeze(0)
    attention = torch.ones(1, hidden.size(1), dtype=torch.long)
    scores, routes = sidecar.forward_groups(
        hidden,
        attention,
        [int(example["mask_position"])],
        [list(example["positive"]), list(example["negative"])],
    )
    # The FFN receives one pooled representation for the complete relation
    # context and one for the complete background context.  It no longer
    # learns from isolated positive/negative token pairs or top-k fragments.
    positive = scores[0, 0, 0]
    negative = scores[0, 0, 1]
    loss = F.softplus(float(margin) - positive + negative)
    route_entropy = -(
        routes.clamp_min(1e-9) * routes.clamp_min(1e-9).log()
    ).sum(-1).mean()
    return loss, float((positive - negative).detach()), float(route_entropy.detach()), routes


@torch.no_grad()
def evaluate(
    sidecar: FrozenStructuredRelationFilter,
    examples: Sequence[Dict[str, object]],
    margin: float,
) -> Dict[str, float]:
    sidecar.eval()
    correct = 0
    margins: List[float] = []
    entropies: List[float] = []
    for example in examples:
        _, value, entropy, _ = _example_loss(sidecar, example, margin)
        margins.append(value)
        entropies.append(entropy)
        correct += int(value > 0.0)
    count = max(len(examples), 1)
    return {
        "examples": float(len(examples)),
        "positive_over_hard_negative_rate": correct / count,
        "mean_margin": sum(margins) / count,
        "mean_route_entropy": sum(entropies) / count,
    }


def _group_examples(
    groups: Sequence[Dict[str, object]],
    tokenizer: SimpleBertTokenizer,
    max_length: int,
) -> Dict[str, List[Dict[str, object]]]:
    result = {"train": [], "dev": [], "test": []}
    for group in groups:
        split = str(group["split"])
        for sample in group["samples"]:
            try:
                row = _encode_sample(tokenizer, sample, max_length)
            except ValueError as error:
                # A few very short facts contain no genuine background token
                # after all role spans are marked.  They cannot provide a
                # meaningful positive-vs-negative ranking signal.
                if "no usable positive/negative positions" not in str(error):
                    raise
                continue
            row["group_id"] = str(group["group_id"])
            result[split].append(row)
    return result


def train(args: argparse.Namespace) -> Dict[str, object]:
    if args.epochs < 1 or args.learning_rate <= 0 or args.max_length < 3:
        raise ValueError("invalid epochs/learning-rate/max-length")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)

    base_checkpoint = Path(args.base_checkpoint)
    base_model = GrammarAttributeBertForMaskedLM.from_pretrained(
        str(base_checkpoint)
    ).to(torch.device("cpu"))
    base_model.eval()
    base_model.requires_grad_(False)
    tokenizer = base_model.grammar_attribute_filter.tokenizer
    if args.max_length > base_model.config.max_position_embeddings:
        raise ValueError("max_length exceeds the base checkpoint position capacity")

    groups = list(load_groups(args.dataset))
    # The sidecar consumes visible context and the masked answer only.  The
    # manifest's hard negatives are candidate-list metadata and may contain a
    # word absent from the frozen checkpoint vocabulary; they are not fed to
    # this stage.
    validation_report = validate_groups(groups)
    for group in groups:
        for sample in group["samples"]:
            required = list(sample["tokens"]) + [str(sample["answer"])]
            missing = sorted(
                {token for token in required if token not in tokenizer.token_to_id}
            )
            if missing:
                raise ValueError(
                    f"base tokenizer OOV in {sample['sample_id']}: {missing}"
                )
    examples_by_split = _group_examples(groups, tokenizer, args.max_length)
    all_examples = [row for rows in examples_by_split.values() for row in rows]
    cache_hidden_states(
        base_model,
        all_examples,
        args.cache_batch_size,
        tokenizer.pad_token_id,
    )

    sidecar = FrozenStructuredRelationFilter(
        hidden_size=base_model.config.hidden_size,
        num_layers=base_model.config.num_hidden_layers,
        num_heads=base_model.config.num_attention_heads,
        mask_token_id=tokenizer.mask_token_id,
        initializer_range=base_model.config.initializer_range,
        spaces=args.spaces,
        route_dim=args.route_dim,
        route_start_layer=args.route_start_layer,
        relation_ffn_hidden_size=args.relation_ffn_hidden_size,
        relation_ffn_scale=args.relation_ffn_scale,
        relation_ffn_chunk_size=args.relation_ffn_chunk_size,
    )
    trainable = list(sidecar.adapter.dynamic_qk.parameters()) + list(
        sidecar.adapter.router.parameters()
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_examples = examples_by_split["train"]
    logs = []
    best_state = {
        name: value.detach().clone() for name, value in sidecar.state_dict().items()
    }
    best_score = (float("-inf"), float("-inf"))
    best_epoch = 0
    for epoch in range(args.epochs):
        sidecar.train()
        order = list(train_examples)
        random.shuffle(order)
        losses = []
        margins = []
        entropies = []
        for example in order:
            optimizer.zero_grad(set_to_none=True)
            loss, value, entropy, _ = _example_loss(sidecar, example, args.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            margins.append(value)
            entropies.append(entropy)
        dev = evaluate(sidecar, examples_by_split["dev"], args.margin)
        record = {
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
        if epoch == 0 or (epoch + 1) % max(args.log_every, 1) == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record, ensure_ascii=False), flush=True)

    sidecar.load_state_dict(best_state)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(sidecar.state_dict(), output / "relation_filter.pt")
    tokenizer.save_pretrained(str(output))
    config = {
        "model_class": "FrozenStructuredRelationFilter",
        "base_checkpoint": str(base_checkpoint),
        "base_hidden_size": int(base_model.config.hidden_size),
        "base_intermediate_size": int(base_model.config.intermediate_size),
        "base_num_hidden_layers": int(base_model.config.num_hidden_layers),
        "base_num_attention_heads": int(base_model.config.num_attention_heads),
        "spaces": len(sidecar.triples),
        "space_allocation": "fixed_shared_context_blocks",
        "context_space_triples": [list(triple) for triple in sidecar.triples],
        "relation_space_count": len(sidecar.triples),
        "relation_candidate_cap": len(sidecar.triples),
        "base_coordinate_coverage_count": (base_model.config.hidden_size + 2) // 3,
        "route_dim": args.route_dim,
        "route_start_layer": args.route_start_layer,
        "relation_ffn_hidden_size": args.relation_ffn_hidden_size,
        "relation_ffn_scale": args.relation_ffn_scale,
        "relation_ffn_chunk_size": args.relation_ffn_chunk_size,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "best_epoch": best_epoch,
        "best_dev_score": list(best_score),
        "validation": validation_report,
        "train_examples": len(examples_by_split["train"]),
        "dev_examples": len(examples_by_split["dev"]),
        "test_examples": len(examples_by_split["test"]),
        "frozen_main_bert": True,
        "general_ffn_shape": [
            int(base_model.config.hidden_size),
            int(base_model.config.intermediate_size),
            int(base_model.config.hidden_size),
        ],
    }
    (output / "relation_filter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "train": evaluate(sidecar, examples_by_split["train"], args.margin),
        "dev": evaluate(sidecar, examples_by_split["dev"], args.margin),
        "test": evaluate(sidecar, examples_by_split["test"], args.margin),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metrics}, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--spaces", type=int, default=DEFAULT_RELATION_CANDIDATE_COUNT
    )
    parser.add_argument("--route-dim", type=int, default=32)
    parser.add_argument("--route-start-layer", type=int, default=1)
    parser.add_argument("--relation-ffn-hidden-size", type=int, default=32)
    parser.add_argument("--relation-ffn-scale", type=float, default=0.1)
    parser.add_argument("--relation-ffn-chunk-size", type=int, default=32)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
