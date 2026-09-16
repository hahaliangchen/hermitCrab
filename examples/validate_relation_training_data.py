"""Validate relation contrast JSONL v1/v2; no model or network dependency."""

import argparse
from collections import Counter
import json
import unicodedata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token(value):
    return isinstance(value, str) and bool(value) and not any(c.isspace() for c in value)


def normalized(value):
    """Surface deduplication only; not alias resolution or semantic proof."""
    return "".join(unicodedata.normalize("NFKC", value).split()).translate(
        str.maketrans({"。": ".", "“": '"', "”": '"', "‘": "'", "’": "'"})
    )


def changed_value(path, sample, fact, version):
    if path == "target_slot":
        return sample["target_slot"]
    if path == "relation_type":
        return fact["relation_type"]
    if version == 1 and "." not in path:
        path = "roles." + path
    prefix, separator, key = path.partition(".")
    require(separator and prefix in {"roles", "qualifiers"} and key and "." not in key,
            f"invalid changed_slots path: {path}")
    return fact.get(prefix, {}).get(key)


def validate_groups(groups, vocabulary=None):
    seen_groups, seen_samples = set(), set()
    source_splits, fact_records, text_splits, semantic_splits, evidence_splits = {}, {}, {}, {}, {}
    counts = Counter()
    role_templates = {}
    surface_counts = Counter()
    for group in groups:
        gid = group["group_id"]
        require(isinstance(gid, str) and gid and gid not in seen_groups, "duplicate/empty group_id")
        seen_groups.add(gid)
        version = group["schema_version"]
        require(type(version) is int and version in {1, 2}, f"{gid}: schema_version must be 1 or 2")
        counts[f"schema_v{version}_groups"] += 1
        split, source = group["split"], group["source_group"]
        require(split in {"train", "dev", "test"}, f"{gid}: invalid split")
        require(isinstance(source, str) and source, f"{gid}: source_group required")
        require(source_splits.setdefault(source, split) == split, f"{gid}: source crosses splits")
        facts = {}
        for fact in group["facts"]:
            fid = fact["fact_id"]
            require(isinstance(fid, str) and fid and fid not in facts, f"{gid}: invalid fact_id")
            require(isinstance(fact["relation_type"], str) and fact["relation_type"], f"{fid}: relation type required")
            require(isinstance(fact["roles"], dict) and fact["roles"], f"{fid}: roles required")
            require(all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in fact["roles"].items()), f"{fid}: invalid roles")
            qualifiers = fact.get("qualifiers", {})
            if version == 2:
                require("qualifiers" in fact, f"{fid}: qualifiers required (may be empty)")
                template = tuple(sorted(fact["roles"]))
                require(role_templates.setdefault(fact["relation_type"], template) == template,
                        f"{fid}: inconsistent role template")
            require(isinstance(qualifiers, dict) and all(
                k in {"time", "place", "condition", "stage"} and isinstance(v, str) and v.strip()
                for k, v in qualifiers.items()), f"{fid}: invalid qualifiers")
            if "relation_key" in fact:
                key = fact["relation_key"]
                require(isinstance(key, dict) and key.get("relation_type") == fact["relation_type"], f"{fid}: invalid relation_key")
                source_slot, target_slot = key.get("source_slot"), key.get("target_slot")
                require(source_slot in fact["roles"] and target_slot in fact["roles"] and source_slot != target_slot,
                        f"{fid}: invalid relation_key slots")
                require(key.get("direction") == f"{source_slot}_to_{target_slot}", f"{fid}: invalid direction")
            evidence = fact["evidence"]
            require(evidence["kind"] in {"provided_source", "synthetic"}, f"{fid}: invalid evidence kind")
            require(all(isinstance(evidence[k], str) and evidence[k].strip() for k in ("source_id", "quote")), f"{fid}: evidence required")
            require(evidence_splits.setdefault(evidence["source_id"], split) == split, f"{fid}: evidence source crosses splits")
            semantic_key = (fact["relation_type"], tuple(sorted((k, normalized(v)) for k, v in fact["roles"].items())),
                            tuple(sorted((k, normalized(v)) for k, v in qualifiers.items())))
            require(semantic_splits.setdefault(semantic_key, split) == split, f"{fid}: same fact under another ID crosses splits")
            signature = (split, fact)
            require(fact_records.setdefault(fid, signature) == signature, f"{fid}: conflicting fact or cross-split reuse")
            facts[fid] = fact
            if not all(normalized(value) in normalized(evidence["quote"]) for value in fact["roles"].values()):
                counts["lexical_evidence_warnings"] += 1
        require(2 <= len(group["samples"]) <= 8, f"{gid}: expected 2..8 samples")
        samples = {}
        for sample in group["samples"]:
            sid = sample["sample_id"]
            require(isinstance(sid, str) and sid and sid not in seen_samples, f"{gid}: duplicate sample_id")
            seen_samples.add(sid)
            require(sample["fact_id"] in facts, f"{sid}: missing fact")
            roles = facts[sample["fact_id"]]["roles"]
            tokens, answer = sample["tokens"], sample["answer"]
            require(isinstance(tokens, list) and all(token(t) for t in tokens), f"{sid}: invalid tokens")
            require(tokens.count("[MASK]") == 1, f"{sid}: exactly one MASK required")
            require(not set(tokens) & {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}, f"{sid}: special tokens supplied")
            require(token(answer) and answer not in tokens and not answer.startswith("["), f"{sid}: answer invalid/leaked")
            require(sample["target_slot"] in roles and roles[sample["target_slot"]] == answer, f"{sid}: answer does not match fact slot")
            if version == 2:
                require(sample.get("surface_kind") in {"source", "paraphrase", "synthetic", "adversarial"}, f"{sid}: invalid surface_kind")
                if sample["surface_kind"] == "source":
                    evidence = facts[sample["fact_id"]]["evidence"]
                    restored = "".join(answer if t == "[MASK]" else t for t in tokens)
                    require(evidence["kind"] == "provided_source" and normalized(restored) in normalized(evidence["quote"]),
                            f"{sid}: source surface must be a contiguous quoted excerpt")
            negatives = sample["hard_negatives"]
            require(isinstance(negatives, list) and negatives and all(token(n) and n != answer and not n.startswith("[") for n in negatives), f"{sid}: invalid hard negatives")
            require(len(set(negatives)) == len(negatives), f"{sid}: duplicate hard negatives")
            if version == 2:
                reasons = sample.get("negative_reasons")
                require(isinstance(reasons, dict) and set(reasons) == set(negatives) and all(
                    isinstance(value, str) and value.strip() for value in reasons.values()), f"{sid}: negative_reasons must cover all negatives")
            if vocabulary is not None:
                missing = set(tokens + [answer] + negatives) - set(vocabulary)
                require(not missing, f"{sid}: tokens outside vocabulary: {sorted(missing)}")
            mask = tokens.index("[MASK]")

            def span_indices(span):
                require(isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span), f"{sid}: invalid span")
                start, end = span
                require(0 <= start < end <= len(tokens) and not start <= mask < end, f"{sid}: span out of range/covers MASK")
                return set(range(start, end))

            occupied = set()
            require(isinstance(sample["role_spans"], dict), f"{sid}: role_spans must be object")
            for role, span in sample["role_spans"].items():
                require(role in roles and role != sample["target_slot"], f"{sid}: unknown/masked role")
                positions = span_indices(span)
                require(not occupied & positions, f"{sid}: overlapping role spans")
                occupied |= positions
            require(isinstance(sample["background_spans"], list), f"{sid}: background_spans must be array")
            for span in sample["background_spans"]:
                positions = span_indices(span)
                require(not occupied & positions, f"{sid}: overlapping background/role spans")
                occupied |= positions
            text_key = normalized("".join(tokens))
            require(text_splits.setdefault(text_key, (split, answer)) == (split, answer), f"{sid}: duplicate text across split or conflicting answer")
            samples[sid] = sample
            surface_counts[sample.get("surface_kind", "unspecified_v1")] += 1
            counts[f"{split}_samples"] += 1
        require(group["pairs"], f"{gid}: pairs required")
        invariance = 0
        seen_pairs = set()
        for pair in group["pairs"]:
            require(pair["a"] in samples and pair["b"] in samples and pair["a"] != pair["b"], f"{gid}: invalid pair reference")
            a, b = samples[pair["a"]], samples[pair["b"]]
            pair_key = tuple(sorted((pair["a"], pair["b"])))
            require(pair_key not in seen_pairs, f"{gid}: duplicate pair")
            seen_pairs.add(pair_key)
            require(a["tokens"] != b["tokens"], f"{gid}: identical pair texts")
            require(isinstance(pair["changed_slots"], list), f"{gid}: changed_slots must be array")
            if pair["kind"] == "invariance":
                require((a["fact_id"], a["target_slot"], a["answer"]) == (b["fact_id"], b["target_slot"], b["answer"]) and not pair["changed_slots"], f"{gid}: invalid invariance pair")
                invariance += 1
            else:
                require(pair["kind"] == "contrast" and a["answer"] != b["answer"] and pair["changed_slots"], f"{gid}: invalid contrast pair")
                for slot in pair["changed_slots"]:
                    require(isinstance(slot, str), f"{gid}: changed_slots must contain strings")
                    changed = changed_value(slot, a, facts[a["fact_id"]], version) != changed_value(slot, b, facts[b["fact_id"]], version)
                    require(changed, f"{gid}: claimed slot did not change: {slot}")
            counts[pair["kind"]] += 1
        require(invariance > 0, f"{gid}: needs an invariance pair")
        counts[f"{split}_groups"] += 1
    require(seen_groups, "empty dataset")
    require(counts["contrast"] > 0, "dataset needs contrast pairs")
    counts["unique_facts"] = len(fact_records)
    counts["source_groups"] = len(source_splits)
    counts["relation_types"] = len({record[1]["relation_type"] for record in fact_records.values()})
    report = dict(counts)
    report["surface_kind_counts"] = dict(surface_counts)
    report["evidence_kind_counts"] = dict(Counter(record[1]["evidence"]["kind"] for record in fact_records.values()))
    return report


def load_groups(path):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"line {line_number}: {error}") from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--vocab", help="optional tokenizer vocab.json; reject OOV tokens/labels/negatives")
    args = parser.parse_args()
    try:
        vocabulary = None
        if args.vocab:
            with open(args.vocab, encoding="utf-8") as stream:
                vocabulary = json.load(stream)
        print(json.dumps(validate_groups(load_groups(args.path), vocabulary), ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"INVALID: {error}\n")
