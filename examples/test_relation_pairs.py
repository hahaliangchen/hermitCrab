"""Protocol v2, label-independent routing and end-to-end JSONL training."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from bert_simple.dynamic_qk_model import DynamicQKLocalRelationMarginBertForMaskedLM
from bert_simple.relation_pair_training import (
    build_model, training_tokenizer, visible_inputs, sample_logits, paired_loss, evaluate,
)
from bert_simple.tokenizer import SimpleBertTokenizer
from validate_relation_training_data import load_groups, validate_groups
from train_relation_pairs import train, parser


def example():
    text = (ROOT / "RELATION_TRAINING_DATA_SPEC.md").read_text()
    return json.loads(text.split("```json\n", 1)[1].split("```", 1)[0])


class ProtocolTests(unittest.TestCase):
    def test_v2_and_v1_compatibility(self):
        group = example()
        self.assertEqual(validate_groups([group])["schema_v2_groups"], 1)
        group["schema_version"] = 1
        for fact in group["facts"]:
            fact.pop("qualifiers")
        for sample in group["samples"]:
            sample.pop("surface_kind")
            sample.pop("negative_reasons")
        group["pairs"][1]["changed_slots"] = ["recipient"]
        self.assertEqual(validate_groups([group])["schema_v1_groups"], 1)

    def test_relation_type_and_qualifier_changes(self):
        group = example()
        group["facts"][1]["relation_type"] = "REAPPOINT"
        group["pairs"][1]["changed_slots"] = ["relation_type"]
        validate_groups([group])
        group = example()
        group["facts"][0]["qualifiers"] = {"time": "元年"}
        group["facts"][1]["qualifiers"] = {"time": "二年"}
        group["pairs"][1]["changed_slots"] = ["qualifiers.time"]
        validate_groups([group])
        group["pairs"][1]["changed_slots"] = ["qualifiers.place"]
        with self.assertRaisesRegex(ValueError, "did not change"):
            validate_groups([group])

    def test_surface_template_negative_reasons_and_relation_key(self):
        for mutation, message in (("surface", "source surface"), ("template", "role template"),
                                  ("reason", "negative_reasons"), ("key", "direction")):
            group = example()
            if mutation == "surface":
                group["samples"][0]["surface_kind"] = "source"
            elif mutation == "template":
                group["facts"][1]["roles"].pop("predicate")
            elif mutation == "reason":
                group["samples"][0]["negative_reasons"] = {}
            else:
                group["facts"][0]["relation_key"] = dict(relation_type="APPOINT", source_slot="actor",
                                                         target_slot="recipient", direction="recipient_to_actor")
            with self.assertRaisesRegex(ValueError, message):
                validate_groups([group])

    def test_normalized_cross_split_duplicate(self):
        group = example()
        other = copy.deepcopy(group)
        other.update(group_id="other", source_group="other-source", split="dev")
        for fact in other["facts"]:
            fact["fact_id"] += "-other"
            fact["evidence"]["source_id"] = "other-source"
            fact["qualifiers"] = {"time": "different period"}
        for sample in other["samples"]:
            sample["sample_id"] += "-other"
            sample["fact_id"] += "-other"
            sample["tokens"][-1] = "．"
        for pair in other["pairs"]:
            pair["a"] += "-other"
            pair["b"] += "-other"
        with self.assertRaisesRegex(ValueError, "duplicate text across split"):
            validate_groups([group, other])

    def test_optional_vocab_check(self):
        group = example()
        vocabulary = training_tokenizer([group]).token_to_id.copy()
        validate_groups([group], vocabulary)
        vocabulary.pop("西侯")
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            validate_groups([group], vocabulary)

    def test_split_group_counts_cover_every_group(self):
        train_group = example()
        dev_group = copy.deepcopy(train_group)
        dev_group.update(
            group_id="synthetic-dev-001",
            source_group="synthetic-story-dev-001",
            split="dev",
        )
        replacements = {
            "青王": "赤王",
            "林舟": "山舟",
            "陆衡": "河衡",
            "东侯": "南侯",
            "西侯": "北侯",
        }

        def replace_text(value):
            result = str(value)
            for old, new in replacements.items():
                result = result.replace(old, new)
            return result

        for fact in dev_group["facts"]:
            fact["fact_id"] += "-dev"
            fact["evidence"]["source_id"] = "synthetic-story-dev-001"
            fact["evidence"]["quote"] = replace_text(fact["evidence"]["quote"])
            fact["roles"] = {
                role: replace_text(value) for role, value in fact["roles"].items()
            }
        for sample in dev_group["samples"]:
            sample["sample_id"] += "-dev"
            sample["fact_id"] += "-dev"
            sample["tokens"] = [replacements.get(token, token) for token in sample["tokens"]]
            sample["answer"] = replacements.get(sample["answer"], sample["answer"])
            sample["hard_negatives"] = [
                replacements.get(token, token) for token in sample["hard_negatives"]
            ]
            sample["negative_reasons"] = {
                replacements.get(key, key): replace_text(value)
                for key, value in sample["negative_reasons"].items()
            }
        for pair in dev_group["pairs"]:
            pair["a"] += "-dev"
            pair["b"] += "-dev"

        report = validate_groups([train_group, dev_group])
        self.assertEqual(report["train_groups"], 1)
        self.assertEqual(report["dev_groups"], 1)
        self.assertEqual(report["train_samples"], 3)
        self.assertEqual(report["dev_samples"], 3)

    def test_generated_v2_training_data_is_valid(self):
        dataset = ROOT / "data" / "shiji" / "manifests" / "relation_training_v2_draft.jsonl"
        groups = list(load_groups(str(dataset)))
        report = validate_groups(groups)
        total_groups = sum(report.get(f"{split}_groups", 0) for split in ("train", "dev", "test"))
        self.assertGreaterEqual(report["schema_v2_groups"], 10)
        self.assertEqual(total_groups, report["schema_v2_groups"])
        self.assertGreater(report["train_groups"], 0)
        self.assertGreater(report["dev_groups"], 0)
        self.assertGreater(report["test_groups"], 0)
        self.assertGreater(report["source_groups"], 0)


class PairTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.group = example()
        self.tokenizer = training_tokenizer([self.group])
        self.model = build_model(self.tokenizer, hidden_size=12, spaces=2).eval()

    def test_gold_metadata_cannot_change_forward_or_candidates(self):
        original = self.group["samples"][0]
        changed = copy.deepcopy(original)
        changed.update(answer="西侯", fact_id="invented", target_slot="recipient",
                       role_spans={}, background_spans=[], hard_negatives=["东侯"], relation_key={"wrong": True})
        with torch.no_grad():
            a = sample_logits(self.model, self.tokenizer, original, 128)
            b = sample_logits(self.model, self.tokenizer, changed, 128)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        inputs = visible_inputs(self.model, self.tokenizer, original["tokens"])
        self.assertEqual(len(inputs["relation_triples"]), 2)
        self.assertNotIn("labels", inputs)

    def test_vocab_does_not_read_holdout_or_evidence(self):
        holdout = copy.deepcopy(self.group)
        holdout["split"] = "test"
        holdout["samples"][0]["answer"] = "独有职位"
        holdout["samples"][0]["tokens"][0] = "独有人物"
        self.group["facts"][0]["evidence"]["quote"] += " 证据独有词 "
        tokenizer = training_tokenizer([self.group, holdout])
        for word in ("独有职位", "独有人物", "证据独有词"):
            self.assertNotIn(word, tokenizer.token_to_id)
        report = evaluate(self.model, tokenizer, [holdout])
        self.assertEqual(report["test"]["oov_answer_count"], 1)
        self.assertEqual(report["test"]["input_oov_samples"], 1)

    def test_losses_have_gradients_and_contrast_has_no_kl(self):
        logits = {s["sample_id"]: torch.randn(len(self.tokenizer), requires_grad=True)
                  for s in self.group["samples"]}
        loss, stats = paired_loss(logits, self.group["samples"], self.group["pairs"], self.tokenizer)
        self.assertGreater(stats["consistency"], 0)
        loss.backward()
        self.assertTrue(all(scores.grad.abs().sum() > 0 for scores in logits.values()))
        _, stats = paired_loss(logits, self.group["samples"], [self.group["pairs"][1]], self.tokenizer)
        self.assertEqual(stats["consistency"], 0)
        changed = copy.deepcopy(self.group["samples"])
        changed[0]["hard_negatives"] = ["未知负例"]
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            paired_loss(logits, changed, self.group["pairs"], self.tokenizer)

    def test_long_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "truncate"):
            visible_inputs(self.model, self.tokenizer, self.group["samples"][0]["tokens"], 4)

    def test_jsonl_train_probe_reload_and_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "sample.jsonl"
            dataset.write_text(json.dumps(self.group, ensure_ascii=False) + "\n", encoding="utf-8")
            output = root / "run"
            args = parser().parse_args([str(dataset), "--output-dir", str(output), "--epochs", "2",
                                       "--hidden-size", "12", "--spaces", "2", "--probe-every", "1"])
            metrics = train(args)
            self.assertEqual(metrics["train"]["samples"], 3)
            records = [json.loads(line) for line in (output / "training_log.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all("correction_probe" in record for record in records))
            restored = DynamicQKLocalRelationMarginBertForMaskedLM.from_pretrained(str(output)).eval()
            tokenizer = SimpleBertTokenizer.from_pretrained(str(output))
            self.assertEqual(evaluate(restored, tokenizer, [self.group]), metrics)
            cli = subprocess.run([sys.executable, str(ROOT / "examples/predict_relation_pairs.py"), str(output),
                                  "青王 封 林舟 为 [MASK] 。"], capture_output=True, text=True, check=True)
            self.assertEqual(len(json.loads(cli.stdout)["predictions"]), 5)
            warm = parser().parse_args([str(dataset), "--output-dir", str(root / "warm"),
                                       "--init-checkpoint", str(output), "--epochs", "1"])
            self.assertEqual(train(warm)["train"]["samples"], 3)
            with self.assertRaises(FileExistsError):
                train(args)


if __name__ == "__main__":
    unittest.main()
