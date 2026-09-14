"""Tests for the fixed grammar automaton and learned word attributes."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bert_simple.grammar_attribute_filter import GrammarAttributeFilter
from bert_simple.grammar_automaton import FixedGrammarAutomaton
from bert_simple.tokenizer import SimpleBertTokenizer


class GrammarAttributeLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        tokenizer = SimpleBertTokenizer()
        tokenizer.add_tokens(
            [
                "高祖",
                "刘邦",
                "魏国",
                "将军",
                "年",
                "新词",
                "的",
                "姓名",
                "是",
                "在",
                "时",
                "。",
                "[MASK]",
            ]
        )
        self.model = GrammarAttributeFilter(tokenizer).eval()

    def test_automaton_returns_fixed_states(self):
        automaton = FixedGrammarAutomaton()
        name = automaton.analyze_text("高祖 的 姓名 是 [MASK] 。")
        temporal = automaton.analyze_text("高祖 在 [MASK] 时 出发 。")
        self.assertEqual(name.structure, "NAME")
        self.assertIn("PERSON", name.allowed_attributes)
        self.assertEqual(temporal.structure, "TEMPORAL")
        self.assertEqual(temporal.allowed_attributes, ("TIME",))

    def test_each_word_has_independent_multilabel_state(self):
        tokenizer = self.model.tokenizer
        ids = tokenizer.token_to_id
        with torch.no_grad():
            self.model.word_attributes.word_attribute_logits[ids["刘邦"]].fill_(-6.0)
            self.model.word_attributes.word_attribute_logits[ids["魏国"]].fill_(-6.0)
            self.model.word_attributes.word_attribute_logits[ids["刘邦"], 2] = 6.0
            self.model.word_attributes.word_attribute_logits[ids["刘邦"], 7] = 6.0
            self.model.word_attributes.word_attribute_logits[ids["魏国"], 0] = 6.0
            self.model.word_attributes.word_attribute_logits[ids["魏国"], 7] = 6.0
        self.model.mark_known_tokens(["刘邦", "魏国"])
        person_bias, _ = self.model.candidate_bias(
            "高祖 的 姓名 是 [MASK] 。", ["刘邦", "魏国", "新词"]
        )
        self.assertGreater(float(person_bias[0]), float(person_bias[1]))
        self.assertLess(float(person_bias[2]), 0.0)

    def test_boundary_state_and_mask_candidate(self):
        bias, state = self.model.candidate_bias(
            "高祖 已经 出发 [MASK]", ["。", "刘邦", "[MASK]"]
        )
        self.assertEqual(state.structure, "BOUNDARY")
        self.assertEqual(float(bias[-1]), -10000.0)


if __name__ == "__main__":
    unittest.main()
