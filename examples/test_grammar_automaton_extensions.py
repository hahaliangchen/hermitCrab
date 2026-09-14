"""Regression tests for the second fixed-grammar rule set."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bert_simple.grammar_automaton import FixedGrammarAutomaton


class GrammarAutomatonExtensionTests(unittest.TestCase):
    def setUp(self):
        self.automaton = FixedGrammarAutomaton()

    def test_subject_slot(self):
        state = self.automaton.analyze_text("[MASK] 是 魏国 大梁 人 。")
        self.assertEqual(state.structure, "SUBJECT")
        self.assertIn("PERSON", state.allowed_attributes)
        self.assertIn("ENTITY", state.allowed_attributes)

    def test_passive_agent_and_subject(self):
        agent = self.automaton.analyze_text("汉王 被 [MASK] 击败 。")
        subject = self.automaton.analyze_text("[MASK] 被 项羽 击败 。")
        self.assertEqual(agent.structure, "PASSIVE")
        self.assertEqual(subject.structure, "PASSIVE")
        self.assertIn("PERSON", agent.allowed_attributes)
        self.assertIn("ENTITY", subject.allowed_attributes)

    def test_negation_slot(self):
        state = self.automaton.analyze_text("汉王 未 [MASK] 成功 。")
        self.assertEqual(state.structure, "NEGATION")
        self.assertIn("ACTION", state.allowed_attributes)

    def test_causative_slot(self):
        state = self.automaton.analyze_text("汉王 派 [MASK] 去 攻赵 。")
        self.assertEqual(state.structure, "CAUSATIVE")
        self.assertIn("PERSON", state.allowed_attributes)

    def test_nested_condition_stack(self):
        inside = self.automaton.analyze_text("如果 [MASK] 失败 ， 汉王 进攻 。")
        outside = self.automaton.analyze_text("如果 张耳 失败 ， [MASK] 进攻 。")
        nested = self.automaton.analyze_text(
            "如果 张耳 失败 ， 若 [MASK] 继续 ， 汉王 进攻 。"
        )
        self.assertEqual(inside.structure, "CONDITION")
        self.assertEqual(inside.stack, ("CONDITION",))
        self.assertNotEqual(outside.stack, ("CONDITION",))
        self.assertEqual(nested.structure, "CONDITION")
        self.assertEqual(nested.stack, ("CONDITION",))


if __name__ == "__main__":
    unittest.main()
