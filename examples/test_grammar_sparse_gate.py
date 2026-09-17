"""Unit tests for System 1 vs System 2 Grammar Sparse Mask Gating and Submatrix 18D FFN."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from bert_simple.model import BertConfig
    from bert_simple.tokenizer import SimpleBertTokenizer
    from bert_simple.dynamic_qk_model import DynamicQKLocalRelationMarginBertForMaskedLM
    from bert_simple.structured_relation import StructuredRelationScores
    from bert_simple.grammar_attribute_filter import GrammarAttributeFilter


@unittest.skipUnless(HAS_TORCH, "PyTorch required for neural test cases")
class TestGrammarSparseGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = SimpleBertTokenizer()
        self.sample_text = "周亚夫 是 太尉 ， 魏子 是 舍人 [MASK] 的 。"
        self.tokenizer.train_from_texts([self.sample_text])

        self.config = BertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=12,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=24,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.triples = [(0, 1, 2), (3, 4, 5)]

    def test_build_sparse_gate_masks_junk(self):
        filter_inst = GrammarAttributeFilter(self.tokenizer)
        with torch.no_grad():
            filter_inst.word_attributes.word_attribute_logits.fill_(-5.0)
            from bert_simple.grammar_automaton import ATTRIBUTES
            func_idx = ATTRIBUTES.index("FUNCTION")
            punct_idx = ATTRIBUTES.index("PUNCT")
            person_idx = ATTRIBUTES.index("PERSON")

            for func_w in ["是", "的"]:
                if func_w in self.tokenizer.token_to_id:
                    filter_inst.word_attributes.word_attribute_logits[self.tokenizer.token_to_id[func_w], func_idx] = 5.0
            for punct_w in ["，", "。"]:
                if punct_w in self.tokenizer.token_to_id:
                    filter_inst.word_attributes.word_attribute_logits[self.tokenizer.token_to_id[punct_w], punct_idx] = 5.0
            for ent_w in ["周亚夫", "魏子", "太尉", "舍人"]:
                if ent_w in self.tokenizer.token_to_id:
                    filter_inst.word_attributes.word_attribute_logits[self.tokenizer.token_to_id[ent_w], person_idx] = 5.0

        model = DynamicQKLocalRelationMarginBertForMaskedLM(
            self.config,
            self.tokenizer,
            self.triples,
            torch.zeros(len(self.tokenizer)),
            torch.zeros(len(self.tokenizer)),
            attribute_filter=filter_inst,
            route_start_layer=2,
            route_dim=4,
            relation_ffn_hidden_size=16,
            enable_grammar_sparse_gate=True,
        )

        tokens = ["[CLS]", "周亚夫", "是", "太尉", "，", "[MASK]", "的", "。", "[SEP]"]
        input_ids = torch.tensor([[self.tokenizer.token_to_id[t] for t in tokens]])

        active_positions, sparse_gate = model._build_sparse_gate(input_ids)

        # Special tokens [CLS], [SEP] must NOT be in active positions
        self.assertNotIn(0, active_positions)  # [CLS]
        self.assertNotIn(len(tokens) - 1, active_positions)  # [SEP]

        # Function words ("是", "的") and punctuation ("，", "。") must NOT be in active positions
        shi_idx = tokens.index("是")
        de_idx = tokens.index("的")
        comma_idx = tokens.index("，")
        period_idx = tokens.index("。")
        self.assertNotIn(shi_idx, active_positions)
        self.assertNotIn(de_idx, active_positions)
        self.assertNotIn(comma_idx, active_positions)
        self.assertNotIn(period_idx, active_positions)

        # Entity "周亚夫", "太尉", and [MASK] MUST be in active positions
        zhou_idx = tokens.index("周亚夫")
        taiwei_idx = tokens.index("太尉")
        mask_idx = tokens.index("[MASK]")
        self.assertIn(zhou_idx, active_positions)
        self.assertIn(taiwei_idx, active_positions)
        self.assertIn(mask_idx, active_positions)

        # Check gate mask values
        # Between Zhou and Mask: 0.0 (unmasked)
        self.assertEqual(sparse_gate[0, 0, zhou_idx, mask_idx].item(), 0.0)
        # Between Zhou and "是": -10000.0 (masked out)
        self.assertEqual(sparse_gate[0, 0, zhou_idx, shi_idx].item(), -10000.0)
        # Between "是" and "的": -10000.0 (masked out)
        self.assertEqual(sparse_gate[0, 0, shi_idx, de_idx].item(), -10000.0)

    def test_structured_relation_scores_sparse_submatrix(self):
        rel_scores = StructuredRelationScores(hidden_size=16, scale=0.1, chunk_size=8)
        batch, length, hidden_dim = 1, 10, 12
        heads = 2
        hidden = torch.randn(batch, length, hidden_dim)
        routes = torch.rand(batch, length, len(self.triples))
        q_bank = torch.randn(heads, len(self.triples), 3, 3)
        k_bank = torch.randn(heads, len(self.triples), 3, 3)

        active_positions = [1, 3, 5]

        # Call with active_positions
        scores_sparse = rel_scores.forward(
            layer_index=2,
            hidden=hidden,
            routes=routes,
            triples=self.triples,
            q_bank=q_bank,
            k_bank=k_bank,
            active_positions=active_positions,
        )

        self.assertEqual(scores_sparse.shape, (batch, heads, length, length))

        # Inactive positions should be strictly 0.0 in scores_sparse
        for i in range(length):
            for j in range(length):
                if i not in active_positions or j not in active_positions:
                    self.assertEqual(scores_sparse[0, :, i, j].abs().sum().item(), 0.0)

        # Active submatrix should be non-zero
        self.assertGreater(scores_sparse[0, :, 1, 3].abs().sum().item(), 0.0)

    def test_save_and_load_preserves_sparse_gate(self):
        model = DynamicQKLocalRelationMarginBertForMaskedLM(
            self.config,
            self.tokenizer,
            self.triples,
            torch.zeros(len(self.tokenizer)),
            torch.zeros(len(self.tokenizer)),
            route_start_layer=2,
            route_dim=4,
            relation_ffn_hidden_size=16,
            enable_grammar_sparse_gate=True,
        )
        self.assertTrue(model.enable_grammar_sparse_gate)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            self.tokenizer.save_pretrained(tmpdir)

            loaded = DynamicQKLocalRelationMarginBertForMaskedLM.from_pretrained(tmpdir)
            self.assertTrue(loaded.enable_grammar_sparse_gate)
            self.assertEqual(loaded.route_start_layer, 2)


if __name__ == "__main__":
    unittest.main()
