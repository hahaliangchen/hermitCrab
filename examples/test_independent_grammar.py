"""Behavioral regression tests for independence and token/position alignment."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from bert_simple.grammar_filter import IndependentGrammarFilter


class IndependentGrammarTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)
        self.model = IndependentGrammarFilter(['我', '在', '呢', '刘邦', '。'],
                                              {'呢': ['FUNCTION'], '。': ['PUNCT']}).eval()

    def test_padding_does_not_change_prediction(self):
        single = self.model.encode(['我 在 [MASK] 。'])
        padded = self.model.encode(['我 在 [MASK] 。', '刘邦 我 在 我 在 [MASK] 。'])
        a = self.model(**single)
        b = self.model(**padded)
        torch.testing.assert_close(a['attribute_logits'][0], b['attribute_logits'][0, :6])
        self.assertEqual(a['g'].shape[-1], 8)
        self.assertEqual(a['h'].shape[-1], 8)

    def test_filter_training_has_only_local_gradients(self):
        unrelated_embedding = torch.nn.Embedding(8, 16)
        output = self.model(**self.model.encode(['我 在 [MASK] 。']))
        output['attribute_logits'].square().mean().backward()
        self.assertIsNone(unrelated_embedding.weight.grad)
        self.assertGreater(float(self.model.embedding.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.model.keys.grad.abs().sum()), 0.)

    def test_roundtrip_and_unknown_candidate_neutrality(self):
        batch = self.model.encode(['我 在 [MASK] 。'])
        out = self.model(**batch)
        bias = self.model.bias(out, ['刘邦', '新词', '呢', '[MASK]'])
        torch.testing.assert_close(bias[..., :2], torch.zeros_like(bias[..., :2]))
        self.assertTrue(bool((bias[..., -1] < -100).all()))
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)
            restored = IndependentGrammarFilter.from_pretrained(path).eval()
            torch.testing.assert_close(restored(**batch)['attribute_logits'], out['attribute_logits'])

    def test_names_share_grammar_input_and_multi_attribute_seeds_are_valid(self):
        a = self.model.encode(['刘邦 在 [MASK] 。'])
        b = self.model.encode(['未见人名 在 [MASK] 。'])
        self.assertTrue(torch.equal(a['input_ids'], b['input_ids']))
        model = IndependentGrammarFilter(['人'], {'人': ['PERSON', 'ENTITY']}).eval()
        logits = torch.full((1, 1, 13), -20.)
        logits[..., 2] = 20.
        penalty = model.bias({'attribute_logits': logits}, ['人'])
        self.assertLess(abs(float(penalty.detach().item())), 1e-4)

    def test_unrelated_clause_does_not_change_local_slot(self):
        batch = self.model.encode(['刘邦 去 。 我 在 [MASK] 。',
                                   '我 来了 。 我 在 [MASK] 。'])
        output = self.model(**batch)['attribute_logits']
        pos = batch['input_ids'][0].tolist().index(self.model.tokenizer.mask_token_id)
        torch.testing.assert_close(output[0, pos], output[1, pos])

    def test_ambiguous_in_slot_preserves_particle_despite_wrong_confidence(self):
        batch = self.model.encode(['我 在 [MASK] 。'])
        output = self.model(**batch)
        output['attribute_logits'] = torch.full_like(output['attribute_logits'], -20.)
        output['attribute_logits'][..., 6] = 20.  # overconfident ACTION
        bias = self.model.bias(output, ['呢', '。', '刘邦'])
        pos = batch['input_ids'][0].tolist().index(self.model.tokenizer.mask_token_id)
        self.assertEqual(float(bias[0, pos, 0].detach()), 0.)
        self.assertLess(float(bias[0, pos, 1].detach()), -4.)


if __name__ == '__main__':
    unittest.main()
