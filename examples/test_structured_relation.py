"""CPU regressions for 18D integration, learning and scoped rollback."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
import torch

from bert_simple.dynamic_qk_model import (
    ContextualRelationRouter,
    DynamicQKLocalRelationMarginBertForMaskedLM,
    DynamicQKRelationBank,
)
from bert_simple.context_spaces import (
    DEFAULT_RELATION_CANDIDATE_COUNT,
    MAX_RELATION_CANDIDATE_COUNT,
    build_context_space_triples,
)
from bert_simple.model import BertConfig
from bert_simple.tokenizer import SimpleBertTokenizer
from bert_simple.structured_relation import Structured3DRelationFFN, StructuredRelationScores
from bert_simple.relation_correction import capture_relation_snapshot, analyze_relation_correction, _run
from validate_relation_training_data import validate_groups


def tiny_model(enabled=True):
    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(["青王 封 林舟 为 东侯 。 陆衡 西侯"])
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=12, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=24,
                        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    model = DynamicQKLocalRelationMarginBertForMaskedLM(
        config, tokenizer, [(0, 1, 2), (3, 4, 5)], torch.zeros(len(tokenizer)),
        torch.zeros(len(tokenizer)), route_dim=4,
        relation_ffn_hidden_size=32 if enabled else 0,
    )
    tokens = "[CLS] 青王 封 林舟 为 [MASK] 。 [SEP]".split()
    ids = torch.tensor([[tokenizer.token_to_id[token] for token in tokens]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  relation_triples=[(0, 1, 2), (3, 4, 5)],
                  apply_grammar_attributes=False, apply_frequency_prior=False)
    return model, tokenizer, inputs


class RelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def test_features_and_chunked_gradients(self):
        q, k = torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])
        features = Structured3DRelationFFN.features(q, k)
        self.assertEqual(features.shape, (18,))
        torch.testing.assert_close(features[-9:], torch.outer(q, k).flatten())
        scores = StructuredRelationScores(chunk_size=2)
        torch.nn.init.normal_(scores.ffn.network[-1].weight, std=.1)
        hidden = torch.randn(1, 5, 6, requires_grad=True)
        routes = torch.softmax(torch.randn(1, 5, 2), -1)
        qbank, kbank = torch.randn(2, 2, 3, 3), torch.randn(2, 2, 3, 3)
        args = (1, hidden, routes, [(0, 1, 2), (3, 4, 5)], qbank, kbank)
        actual = scores(*args)
        grad = torch.autograd.grad(actual.square().sum(), hidden)[0]
        scores.chunk_size = 100
        expected = scores(*args)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(grad, torch.autograd.grad(expected.square().sum(), hidden)[0])

    def test_sparse_pair_path_matches_full_scores(self):
        torch.manual_seed(23)
        scorer = StructuredRelationScores(hidden_size=7, scale=0.3, chunk_size=2)
        hidden = torch.randn(1, 6, 9)
        routes = torch.softmax(torch.randn(1, 6, 3), dim=-1)
        qbank = torch.randn(2, 3, 3, 3)
        kbank = torch.randn(2, 3, 3, 3)
        triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        full = scorer(1, hidden, routes, triples, qbank, kbank)
        sparse = scorer.forward_pairs(
            1,
            hidden,
            routes,
            triples,
            qbank,
            kbank,
            query_positions=[2, 4],
            key_positions=[0, 3, 5],
        )
        expected = full[:, :, [2, 4]][..., [0, 3, 5]]
        torch.testing.assert_close(sparse, expected)

    def test_group_path_aggregates_all_relation_members(self):
        torch.manual_seed(31)
        scorer = StructuredRelationScores(hidden_size=11, scale=0.3, chunk_size=2)
        torch.nn.init.normal_(scorer.ffn.network[-1].weight, std=0.1)
        hidden = torch.randn(1, 6, 12, requires_grad=True)
        routes = torch.softmax(torch.randn(1, 6, 3), dim=-1)
        qbank = torch.randn(2, 3, 3, 3)
        kbank = torch.randn(2, 3, 3, 3)
        triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        grouped = scorer.forward_context_groups(
            1, hidden, routes, triples, qbank, kbank,
            query_positions=[4],
            context_groups=[[0, 1, 2], [3]],
        )
        self.assertEqual(tuple(grouped.shape), (1, 2, 1, 2))
        grouped.sum().backward()
        for position in (0, 1, 2, 3, 4):
            self.assertGreater(float(hidden.grad[0, position].abs().sum()), 0.0)

        with torch.no_grad():
            one_member = scorer.forward_context_groups(
                1, hidden.detach(), routes, triples, qbank, kbank,
                query_positions=[4],
                context_groups=[[0], [3]],
            )
        # Adding members changes the pooled context representation; this is
        # precisely the behavior that a token-pair lookup cannot express.
        self.assertFalse(torch.equal(grouped.detach(), one_member))

    def test_context_bank_is_fixed_and_covers_all_256_dimensions(self):
        triples = build_context_space_triples(256)
        self.assertEqual(len(triples), 86)
        self.assertEqual(len(set(triples)), 86)
        covered = {dimension for triple in triples for dimension in triple}
        self.assertEqual(covered, set(range(256)))

        candidates = build_context_space_triples(
            256, candidate_count=DEFAULT_RELATION_CANDIDATE_COUNT
        )
        self.assertEqual(len(candidates), 1500)
        self.assertEqual(len(set(candidates)), 1500)
        self.assertEqual(
            {dimension for triple in candidates for dimension in triple},
            set(range(256)),
        )
        self.assertEqual(MAX_RELATION_CANDIDATE_COUNT, 1500)
        with self.assertRaises(ValueError):
            build_context_space_triples(256, candidate_count=1501)

    def test_relation_group_keeps_connectors_and_does_not_cap_members_at_eight(self):
        from train_frozen_relation_filter import _sample_positions

        sample = {
            "sample_id": "appoint-group",
            "tokens": [
                "汉王", "封", "张耳", "为", "[MASK]", "，",
                "命", "他", "镇守", "赵地", "然后", "回宫", "又", "遣",
                "使者", "巡行", "诸郡", "以", "安", "民", "。",
            ],
            "role_spans": {
                "actor": [0, 1],
                "predicate": [1, 2],
                "recipient": [2, 3],
            },
            "background_spans": [[6, 15]],
        }
        mask, positive, negative = _sample_positions(sample)
        self.assertEqual(mask, 5)
        # CLS adds one to data positions; the connective “为” is retained.
        self.assertEqual(positive, [1, 2, 3, 4])
        self.assertGreater(len(negative), 8)

    def test_dynamic_registry_is_context_only_and_fixed(self):
        from bert_mlm_dynamic_word_spaces import ContextSpaceRegistry

        tokenizer = SimpleBertTokenizer()
        tokenizer.train_from_texts(["甲 乙 丙 丁"])
        registry = ContextSpaceRegistry(tokenizer, max_spaces=4, initial_spaces=1)
        ids = torch.tensor(
            [[tokenizer.cls_token_id, tokenizer.token_to_id["甲"], tokenizer.token_to_id["乙"], tokenizer.sep_token_id]]
        )
        sample = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        mask = registry.candidate_mask(sample)
        self.assertEqual(registry.active_spaces, 4)
        self.assertFalse(hasattr(registry, "word_to_spaces"))
        self.assertTrue(bool(mask[0, 1:].all()))

    def test_relation_and_router_chunking_preserve_values(self):
        torch.manual_seed(29)
        triples = [(0, 1, 2), (3, 4, 5), (1, 6, 7), (0, 5, 8)]
        wide = DynamicQKRelationBank(
            triples, num_layers=2, num_heads=2, route_dim=5,
            relation_space_chunk_size=64,
        )
        narrow = DynamicQKRelationBank(
            triples, num_layers=2, num_heads=2, route_dim=5,
            relation_space_chunk_size=2,
        )
        narrow.load_state_dict(wide.state_dict(), strict=True)
        hidden = torch.randn(2, 7, 10)
        routes = torch.softmax(torch.randn(2, 7, len(triples)), dim=-1)
        torch.testing.assert_close(
            narrow.local_scores(1, hidden, routes),
            wide.local_scores(1, hidden, routes),
        )

        router = ContextualRelationRouter(10, 5, 0.02)
        descriptors = torch.randn(13, 5)
        candidate_mask = torch.ones(2, 7, 13, dtype=torch.bool)
        chunked = router(hidden, descriptors, candidate_mask, space_chunk_size=3)
        whole = router(hidden, descriptors, candidate_mask, space_chunk_size=64)
        torch.testing.assert_close(chunked, whole)

    def test_zero_initialization_and_old_checkpoint(self):
        base, _, inputs = tiny_model(False)
        model, _, _ = tiny_model(True)
        incompatible = model.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(all("structured_scores" in key for key in incompatible.missing_keys))
        base.eval(); model.eval()
        torch.testing.assert_close(base(**inputs)[0], model(**inputs)[0], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            base.grammar_attribute_filter.tokenizer.save_pretrained(directory)
            # Simulate an old configuration without any FFN configuration keys.
            path = Path(directory) / "dynamic_qk_config.json"
            config = json.loads(path.read_text())
            config = {k: v for k, v in config.items() if not k.startswith("relation_ffn_")}
            path.write_text(json.dumps(config))
            restored = type(base).from_pretrained(directory).eval()
            self.assertEqual(restored.relation_ffn_hidden_size, 0)
            torch.testing.assert_close(base(**inputs)[0], restored(**inputs)[0])

    def test_training_adapter_group_and_save_reload(self):
        model, tokenizer, inputs = tiny_model()
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[0, 5] = tokenizer.token_to_id["东侯"]
        params = list(model.relation_adapter.parameters())
        optimizer = torch.optim.SGD(params, lr=.3)
        head = model.relation_adapter.dynamic_qk.structured_scores.ffn.network[-1]
        initial = head.weight.detach().clone()
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels)[0]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            raw = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            masked = model.relation_adapter.mask_gradients(raw, [0, 1])
            for parameter, gradient in zip(params, masked):
                parameter.grad = gradient
            optimizer.step()
        self.assertFalse(torch.equal(initial, head.weight))
        first = model.relation_adapter.dynamic_qk.structured_scores.ffn.network[0]
        self.assertGreater(float(first.weight.grad.abs().sum()), 0)
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            restored = type(model).from_pretrained(directory).eval()
            self.assertEqual(restored.relation_ffn_hidden_size, 32)
            torch.testing.assert_close(model(**inputs)[0], restored(**inputs)[0])

    def test_rollback_changes_only_selected_bias_entries_and_cleans_hooks(self):
        model, tokenizer, inputs = tiny_model()
        before = capture_relation_snapshot(model, **inputs)
        module = model.relation_adapter.dynamic_qk.structured_scores
        with torch.no_grad():
            module.ffn.network[-1].weight.normal_(std=2.)
        after = capture_relation_snapshot(model, **inputs)
        edges = [(1, 0, 1), (1, 1, 3)]
        rolled = _run(model, inputs, before, edges)
        expected = after.scores[1].clone()
        for layer, head, key in edges:
            expected[0, head, 5, key] = before.scores[layer][0, head, 5, key]
        torch.testing.assert_close(rolled.scores[1], expected)
        self.assertFalse(torch.equal(rolled.logits, after.logits))
        report = analyze_relation_correction(model, before, tokenizer.token_to_id["东侯"], **inputs)
        self.assertEqual(len(report["singles"]), 2)
        self.assertEqual(len(report["pairs"]), 1)
        self.assertFalse(module._forward_hooks)
        self.assertTrue(model.training)
        torch.testing.assert_close(capture_relation_snapshot(model, **inputs).logits, after.logits)
        changed = dict(inputs, relation_triples=[(0, 1, 2)])
        with self.assertRaisesRegex(ValueError, "identical"):
            analyze_relation_correction(model, before, 1, **changed)

    def test_no_candidates_no_bias(self):
        model, _, inputs = tiny_model()
        inputs["relation_triples"] = []
        snapshot = capture_relation_snapshot(model, **inputs)
        self.assertEqual(snapshot.scores, {})
        report = analyze_relation_correction(model, snapshot, 5, **inputs)
        self.assertEqual(report["singles"], [])

    def test_existing_trainer_updates_shared_ffn_with_replay(self):
        from train_shiji_fact_memory_dynamic_qk import implementation
        _run_update = implementation._run_update
        model, tokenizer, inputs = tiny_model()
        for parameter in model.grammar_attribute_filter.parameters():
            parameter.requires_grad_(False)
        global_parameters = [p for name, p in model.named_parameters()
                             if p.requires_grad and not name.startswith("relation_adapter.")]
        adapter_parameters = list(model.relation_adapter.parameters())
        parameters = global_parameters + adapter_parameters
        optimizer = torch.optim.AdamW(parameters, lr=.001)
        labels = torch.full_like(inputs["input_ids"], -100)
        target = tokenizer.token_to_id["东侯"]
        labels[0, 5] = target
        sample = dict(inputs, token_type_ids=torch.zeros_like(inputs["input_ids"]),
                      labels=labels, fact_token_id=target, fact_position=5,
                      hard_negative_ids=[tokenizer.token_to_id["西侯"]])
        before = capture_relation_snapshot(model, **inputs)
        stats = _run_update(model, parameters, global_parameters, adapter_parameters,
                            optimizer, sample,
                            [{"sample": sample, "relation_triples": inputs["relation_triples"]}],
                            1, .5, .02, .25, 2., -.35, .85)
        self.assertGreater(stats["relation_ffn_gradient_norm"], 0)
        self.assertGreater(stats["relation_ffn_update_norm"], 0)
        report = analyze_relation_correction(model, before, target, only_corrected=True, **inputs)
        self.assertIn("corrected", report)
        json.dumps(report)


class DataSpecTests(unittest.TestCase):
    def example(self):
        doc = (ROOT / "RELATION_TRAINING_DATA_SPEC.md").read_text(encoding="utf-8")
        return json.loads(doc.split("```json\n", 1)[1].split("```", 1)[0])

    def test_published_example_and_rejections(self):
        group = self.example()
        self.assertEqual(validate_groups([group])["contrast"], 1)
        for mutation in ("leak", "span", "answer", "pair"):
            bad = copy.deepcopy(group)
            if mutation == "leak":
                bad["samples"][0]["tokens"].append("东侯")
            elif mutation == "span":
                bad["samples"][0]["background_spans"] = [[7, 9]]
            elif mutation == "answer":
                bad["samples"][0]["answer"] = "西侯"
            else:
                bad["pairs"][1]["changed_slots"] = ["actor"]
            with self.assertRaises(ValueError):
                validate_groups([bad])

    def test_split_leak(self):
        group = self.example()
        duplicate = copy.deepcopy(group)
        duplicate.update(group_id="other", split="test")
        with self.assertRaisesRegex(ValueError, "source crosses"):
            validate_groups([group, duplicate])


if __name__ == "__main__":
    unittest.main()
