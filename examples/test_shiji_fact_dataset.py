"""Regression tests for the source-verified Shiji fact dataset."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "shiji" / "manifests" / "fact_memory_dataset.json"
MIRROR = ROOT / "outputs" / "fact-memory" / "facts_dataset.json"
REPORT = ROOT / "data" / "shiji" / "manifests" / "fact_memory_report.json"


class ShijiFactDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = json.loads(DATASET.read_text(encoding="utf-8"))
        cls.mirror = json.loads(MIRROR.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_mirror_and_report_are_current(self) -> None:
        self.assertEqual(self.rows, self.mirror)
        self.assertEqual(self.report["total_facts"], len(self.rows))

    def test_every_base_sample_has_an_exact_mask(self) -> None:
        for row in self.rows:
            tokens = row["context"].split()
            start, end = row["target_span"]
            expected = tokens[:start] + ["[MASK]"] + tokens[end:]
            self.assertEqual(row["masked_text"].split(), expected)
            self.assertEqual(row["masked_text"].split().count("[MASK]"), 1)
            self.assertNotIn(row["answer"], row["masked_text"].split())

    def test_variants_have_their_own_consistent_answer(self) -> None:
        for row in self.rows:
            for variant in row["variants"]:
                self.assertEqual(variant["answer"], row["answer"])
                self.assertEqual(
                    variant["masked_text"].split().count("[MASK]"), 1
                )
                self.assertNotIn(row["answer"], variant["masked_text"].split())

    def test_source_line_controls_the_split(self) -> None:
        line_splits = {}
        for row in self.rows:
            previous = line_splits.setdefault(row["source_line"], row["split"])
            self.assertEqual(previous, row["split"])

    def test_known_zhanger_facts_survive(self) -> None:
        rows = [
            row
            for row in self.rows
            if row["source_line"] == 6233 and row["answer"] == "张耳"
        ]
        self.assertTrue(rows)
        self.assertEqual(rows[0]["subject"], "汉王")
        self.assertEqual(rows[0]["relation_type"], "ASSIGN")

    def test_naming_is_not_mislabelled_as_causative(self) -> None:
        forbidden_lines = {258, 1190, 6009, 6243}
        self.assertFalse(
            any(row["source_line"] in forbidden_lines for row in self.rows)
        )


if __name__ == "__main__":
    unittest.main()
