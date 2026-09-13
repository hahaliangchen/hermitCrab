import argparse
import os
import sys
from typing import List

import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# add project root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_simple import BertConfig, BertForMaskedLM


def build_tokenizer(path_or_name: str) -> BertTokenizerFast:
    tok = BertTokenizerFast.from_pretrained(path_or_name)
    return tok


def build_dataset(texts: List[str]) -> Dataset:
    return Dataset.from_dict({"text": texts})


def load_texts(path: str) -> List[str]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    if not texts:
        raise ValueError(f"Training file is empty: {path}")
    return texts


def main(data_path: str, tokenizer_path: str, output_dir: str = None):
    device = torch.device("cpu")

    # 1) tokenizer
    tokenizer = build_tokenizer(tokenizer_path)

    # 2) data
    texts = load_texts(data_path)
    raw_ds = build_dataset(texts)

    max_length = 128

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    tokenized = raw_ds.map(tokenize_function, batched=True, remove_columns=["text"]).with_format("torch")

    # 3) data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 4) minimal BERT config & model
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=max_length + 10,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    model = BertForMaskedLM(config)
    # 保证词表大小与 tokenizer 对齐（同时会重绑 LM 头权重）
    model.resize_token_embeddings(len(tokenizer))

    # 5) trainer
    out_dir = output_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs", "hf-trainer-mlm")
    )
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 手动保存（使用我们实现的 save_pretrained）
    model.save_pretrained(out_dir)
    print(f"Saved to {out_dir}")

    # 6) inference demo
    model.eval()
    input_text = "改立沛公为汉[MASK]，统治巴蜀、汉中之地，建都南郑,项羽封有功的部将，却偏偏让您到南[MASK]去，分明是流放您。"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)
    for b, pos in mask_positions:
        scores = logits[b, pos]
        topk = torch.topk(scores, k=5)
        tokens = topk.indices.tolist()
        values = topk.values.tolist()
        print(f"[MASK] at position {pos.item()} -> ", [tokenizer.decode([t]) for t in tokens],
              [round(v, 2) for v in values])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the HF Trainer MLM variant.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "shiji_baihua.txt")),
    )
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--output-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "hf-trainer-mlm")),
    )
    args = parser.parse_args()
    main(args.data_path, args.tokenizer_path, args.output_dir)
