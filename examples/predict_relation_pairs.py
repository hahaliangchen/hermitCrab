"""Use exactly the same visible-text candidate policy as JSONL training."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from bert_simple.dynamic_qk_model import DynamicQKLocalRelationMarginBertForMaskedLM
from bert_simple.tokenizer import SimpleBertTokenizer
from bert_simple.relation_pair_training import CANDIDATE_POLICY, visible_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("text", help="space-tokenized text containing one [MASK]")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    settings = json.loads((checkpoint / "relation_pair_config.json").read_text())
    if settings["candidate_policy"] != CANDIDATE_POLICY:
        raise ValueError("unsupported candidate policy")
    model = DynamicQKLocalRelationMarginBertForMaskedLM.from_pretrained(str(checkpoint)).eval()
    tokenizer = SimpleBertTokenizer.from_pretrained(str(checkpoint))
    tokens = args.text.split()
    with torch.no_grad():
        inputs = visible_inputs(model, tokenizer, tokens, settings["max_length"])
        scores = model(**inputs)[0][0, tokens.index("[MASK]")+1]
        probabilities, ids = scores.softmax(-1).topk(min(5, len(tokenizer)))
    print(json.dumps({"predictions": [{"token": tokenizer.id_to_token[int(i)], "probability": float(p)}
                                     for i, p in zip(ids, probabilities)],
                      "oov_tokens": sorted(set(tokens) - tokenizer.token_to_id.keys())}, ensure_ascii=False, indent=2))
