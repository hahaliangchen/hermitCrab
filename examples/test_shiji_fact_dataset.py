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
            any(
                row["source_line"] in forbidden_lines
                and row["grammar"] == "CAUSATIVE"
                for row in self.rows
            )
        )

    def test_relation_rows_have_explicit_link_metadata(self) -> None:
        relation_grammars = {"TITLE_LINK", "ALIAS", "KINSHIP"}
        relation_rows = [
            row for row in self.rows if row["grammar"] in relation_grammars
        ]
        self.assertTrue(relation_rows)
        for row in relation_rows:
            self.assertEqual(row["relation"], row["relation_type"])
            self.assertIn(row["relation_direction"], {
                "title_to_person", "person_to_title", "alias",
                "child_to_parent", "parent_to_child",
                "grandparent_to_grandchild", "grandchild_to_grandparent",
                "sibling_to_sibling", "spouse_to_spouse",
                "uncle_to_nephew", "nephew_to_uncle",
                "ancestor_to_descendant", "descendant_to_ancestor",
            })
            self.assertEqual(set(row["linked_entity"]), {"left", "right"})
            self.assertTrue(row["linked_entity"]["left"])
            self.assertTrue(row["linked_entity"]["right"])

    def test_identity_kinship_and_spouse_edges_survive_noise_guards(self) -> None:
        pairs = {
            (
                row["grammar"],
                row["relation_type"],
                row["linked_entity"]["left"],
                row["linked_entity"]["right"],
            )
            for row in self.rows
            if row["grammar"] in {"TITLE_LINK", "ALIAS", "KINSHIP"}
        }
        self.assertIn(("TITLE_LINK", "TITLE_OF", "高祖", "刘邦"), pairs)
        self.assertIn(("ALIAS", "ALIAS_OF", "项羽", "项籍"), pairs)
        self.assertIn(("KINSHIP", "SIBLING_OF", "蒙恬", "蒙毅"), pairs)
        self.assertIn(("KINSHIP", "SPOUSE_OF", "樊哙", "吕须"), pairs)
        self.assertNotIn(("TITLE_LINK", "TITLE_OF", "乘机", "王"), pairs)
        self.assertNotIn(("TITLE_LINK", "TITLE_OF", "王", "三个"), pairs)


if __name__ == "__main__":
    unittest.main()
