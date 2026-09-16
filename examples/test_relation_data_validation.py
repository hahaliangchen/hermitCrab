"""Dependency-free regression tests for the relation-training data protocol."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from build_relation_training_v2 import (
    CLOSING_TOKENS,
    SENTENCE_BOUNDARIES,
    _context_conflict_lines,
    _context_coverage,
    _load_rows,
    _load_source_lines,
    build,
)
from validate_relation_training_data import load_groups, validate_groups


DATASET = ROOT / "data" / "shiji" / "manifests" / "relation_training_v2_draft.jsonl"
REPORT = ROOT / "data" / "shiji" / "manifests" / "relation_training_v2_draft_report.json"
SOURCE_TEXT = ROOT / "data" / "shiji" / "sentences" / "shiji_sentences.txt"
FACT_MANIFEST = ROOT / "data" / "shiji" / "manifests" / "fact_memory_dataset.json"


class RelationDataValidationTests(unittest.TestCase):
    def test_generated_draft_matches_its_report(self):
        groups = list(load_groups(str(DATASET)))
        report = validate_groups(groups)
        saved_report = json.loads(REPORT.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_v2_groups"], saved_report["validator"]["schema_v2_groups"])
        self.assertEqual(report["unique_facts"], saved_report["validator"]["unique_facts"])
        self.assertEqual(report["train_groups"], saved_report["validator"]["train_groups"])
        self.assertEqual(report["dev_groups"], saved_report["validator"]["dev_groups"])
        self.assertEqual(report["test_groups"], saved_report["validator"]["test_groups"])
        self.assertEqual(report["train_samples"], saved_report["validator"]["train_samples"])
        self.assertEqual(report["dev_samples"], saved_report["validator"]["dev_samples"])
        self.assertEqual(report["test_samples"], saved_report["validator"]["test_samples"])

    def test_every_group_is_counted_in_its_split(self):
        groups = list(load_groups(str(DATASET)))
        report = validate_groups(groups)
        actual = {split: sum(group["split"] == split for group in groups)
                  for split in ("train", "dev", "test")}
        counted = {split: report[f"{split}_groups"] for split in actual}
        self.assertEqual(counted, actual)
        self.assertGreater(len(set(actual.values())), 1)

    def test_generated_contexts_do_not_cross_splits(self):
        rows = _load_rows(FACT_MANIFEST)
        source_lines = _load_source_lines(SOURCE_TEXT)
        line_splits, coverage = _context_coverage(rows, source_lines)
        row_by_id = {row["fact_id"]: row for row in rows}
        groups = list(load_groups(str(DATASET)))

        for group in groups:
            source_rows = [row_by_id[fact["fact_id"]] for fact in group["facts"]]
            self.assertEqual(_context_conflict_lines(source_rows, line_splits, coverage), [])

        train_rows = [row for row in rows if row.get("source_line") == 4949]
        self.assertIn(4950, _context_conflict_lines(train_rows, line_splits, coverage))

    def test_builder_honors_small_max_lengths(self):
        rows = _load_rows(FACT_MANIFEST)
        source_lines = _load_source_lines(SOURCE_TEXT)
        for max_length in (64, 32):
            groups, skipped, _ = build(rows, max_length, source_lines)
            self.assertGreater(skipped["max_length_exceeded"], 0)
            self.assertTrue(groups)
            for group in groups:
                for sample in group["samples"]:
                    self.assertLessEqual(len(sample["tokens"]) + 2, max_length)
            validate_groups(groups)

    def test_source_variants_use_complete_background_segments(self):
        groups = list(load_groups(str(DATASET)))
        for group in groups:
            samples = {sample["sample_id"]: sample for sample in group["samples"]}
            for pair in group["pairs"]:
                if pair["kind"] != "invariance":
                    continue
                variant = samples[pair["b"]]
                if variant["surface_kind"] != "source":
                    continue
                self.assertTrue(variant["background_spans"], variant["sample_id"])
                background = [
                    token
                    for start, end in variant["background_spans"]
                    for token in variant["tokens"][start:end]
                ]
                self.assertTrue(any(token in SENTENCE_BOUNDARIES for token in background))
                self.assertTrue(any(token not in SENTENCE_BOUNDARIES | CLOSING_TOKENS for token in background))


if __name__ == "__main__":
    unittest.main()
