import argparse
import os
import random
import sys
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# add project root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_simple import BertConfig, BertForMaskedLM
from bert_simple.tokenizer import SimpleBertTokenizer


DEFAULT_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "shiji_baihua.txt"))
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "outputs", "simple-tokenizer-mlm")
)


def _resolve_path(path: str) -> str:
    """Resolve normal Windows paths and paths copied as /D:/... ."""
    path = os.path.expandvars(os.path.expanduser(path))
    if len(path) >= 3 and path[0] in ("/", "\\") and path[2] == ":":
        path = path[1:]
    return os.path.abspath(path)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: SimpleBertTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        encs = [
            tokenizer.encode(
                t,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_special_tokens_mask=True,
            )
            for t in texts
        ]
        # store lists for speed
        self.examples: List[Dict[str, List[int]]] = [
            {
                "input_ids": e["input_ids"],
                "token_type_ids": e["token_type_ids"],
                "attention_mask": e["attention_mask"],
                "special_tokens_mask": e.get("special_tokens_mask", [0] * len(e["input_ids"]))
            }
            for e in encs
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def mlm_collate(batch: List[Dict[str, List[int]]], tokenizer: SimpleBertTokenizer, mlm_probability: float = 0.15):
    # 将 batch 中的 input_ids 转换为 PyTorch 张量，也就是把多条数据堆叠起来
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    # 将 batch 中的 token_type_ids (区分句子A/B) 转换为张量
    token_type_ids = torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long)
    # 将 batch 中的 attention_mask (标记哪些是填充的pad) 转换为张量
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    # 将 batch 中的 special_tokens_mask (标记哪些是特殊字符如[CLS],[SEP]) 转换为布尔张量
    special_tokens_mask = torch.tensor([b["special_tokens_mask"] for b in batch], dtype=torch.bool)

    # 复制一份原始输入作为标签(Label)，这就是我们要预测的"标准答案"
    labels = input_ids.clone()
    # 确定哪些位置可以被遮掩(Mask)：不能是特殊字符([CLS]等)，也不能是填充字符(pad)
    maskable = (~special_tokens_mask) & (attention_mask == 1)
    # 生成一个和输入形状一样的随机数矩阵，用于决定哪些位置被选中
    rand = torch.rand(input_ids.shape)
    # 最终确定被遮掩的位置：必须是可遮掩的，且随机概率小于15%(mlm_probability)
    mask_positions = maskable & (rand < mlm_probability)

    # A short batch can randomly select no positions, which makes cross entropy
    # return NaN when every label is -100. Keep one valid position if possible.
    if not bool(mask_positions.any()):
        candidates = maskable.nonzero(as_tuple=False)
        if candidates.numel() == 0:
            raise ValueError("Batch contains no non-special, non-padding tokens to mask")
        row, col = candidates[torch.randint(candidates.size(0), (1,)).item()].tolist()
        mask_positions[row, col] = True

    # 将所有"没被选中"的位置的标签设为 -100，PyTorch计算Loss时会自动忽略 -100
    labels[~mask_positions] = -100

    # 再次生成随机数，用于决定这15%被选中的位置具体怎么个变法 (80-10-10策略)
    probs = torch.rand(input_ids.shape)
    # 80% 的概率：真的替换成 [MASK] 符号
    mask80 = mask_positions & (probs < 0.8)
    # 10% 的概率：替换成词表里随机的一个词 (用于干扰模型)
    rand10 = mask_positions & (probs >= 0.8) & (probs < 0.9)
    # 10% 的概率：保持原样不动 (让模型学会相信输入)
    # 复制一份输入id，准备进行修改，生成最终喂给模型的 masked_input_ids
    masked_input_ids = input_ids.clone()
    # 将属于 80% 那部分的位置，替换为 [MASK] 的 ID
    masked_input_ids[mask80] = tokenizer.mask_token_id

    # 获取词表大小
    vocab_size = len(tokenizer)
    # Do not inject PAD/CLS/SEP/MASK as random content tokens.
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    valid_token_ids = [idx for idx in range(vocab_size) if idx not in special_ids]
    if not valid_token_ids:
        raise ValueError("Tokenizer has no regular tokens for MLM random replacement")
    valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long)
    random_indices = torch.randint(
        low=0, high=valid_token_ids.numel(), size=input_ids.shape, dtype=torch.long
    )
    random_words = valid_token_ids[random_indices]
    # 将属于 10% 随机替换那部分的位置，填入随机生成的词
    masked_input_ids[rand10] = random_words[rand10]
    # 剩下的 10% (keep10) 不做任何操作，保留原词
    # keep10: do nothing

    # 返回处理好的结果：带掩码的输入、句子类型ID、注意力掩码、以及计算Loss用的标签
    return masked_input_ids, token_type_ids, attention_mask, labels


def train_and_save(
    output_dir: Optional[str] = None,
    training_file: str = DEFAULT_DATA_PATH,
    epochs: int = 1,
    max_steps: Optional[int] = None,
    batch_size: int = 16,
    max_length: int = 128,
    hidden_size: int = 256,
):
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be greater than zero when provided")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    if hidden_size <= 0 or hidden_size % 4 != 0:
        raise ValueError("hidden_size must be a positive multiple of 4")

    set_seed(42)
    # CPU is intentional for this training setup; do not silently switch to CUDA.
    device = torch.device("cpu")

    # 1) Load the explicitly selected corpus. Never fall back to dummy data:
    # a missing path must fail loudly instead of producing a misleading model.
    txt_path = _resolve_path(training_file)
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Training file not found: {txt_path}")
    print(f"Using training file: {txt_path}")

    texts = []
    print(f"Loading texts from {txt_path} ...")
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    if not texts:
        raise ValueError(f"Training file is empty: {txt_path}")
    print(f"Loaded {len(texts)} lines from file.")

    # 2) build tokenizer offline from texts
    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)

    # 3) config/model
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=max_length + 10,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = BertForMaskedLM(config).to(device)

    # 4) dataset/loader
    dataset = TextDataset(texts, tokenizer, max_length=max_length)

    def _collate(batch):
        return mlm_collate(batch, tokenizer, mlm_probability=0.15)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    # 5) train
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    steps = 0
    stop_training = False
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_masked_tokens = 0
        for input_ids, token_type_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, *_ = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            steps += 1
            epoch_steps += 1
            epoch_loss += loss.item()
            epoch_masked_tokens += int((labels != -100).sum().item())
            if steps % 50 == 0:
                print(f"step {steps}: loss={loss.item():.4f}")
            if max_steps is not None and steps >= max_steps:
                stop_training = True
                break
        print(
            f"epoch {epoch}/{epochs}: avg_loss={epoch_loss / max(epoch_steps, 1):.4f}, "
            f"masked_tokens={epoch_masked_tokens}"
        )
        if stop_training:
            break

    # 6) save
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model+tokenizer to {output_dir}")

    return output_dir


@torch.no_grad()
def demo_inference(model_dir: str):
    device = torch.device("cpu")
    model = BertForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()
    tokenizer = SimpleBertTokenizer.from_pretrained(model_dir)

    text = "改立沛公为汉[MASK]，统治巴蜀、汉中之地，建都南郑"
    enc = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    outputs = model(**enc)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs

    mask_positions = (enc["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    for b, pos in mask_positions:
        scores = logits[b, pos]
        topk = torch.topk(scores, k=5)
        tokens = topk.indices.tolist()
        values = topk.values.tolist()
        print("Top-5:", [tokenizer.decode([t]) for t in tokens], [round(v, 2) for v in values])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the simple BERT MLM model on a UTF-8 text file."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=DEFAULT_DATA_PATH,
        help=f"Training text file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--skip-demo", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = train_and_save(
        training_file=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
    )
    if not args.skip_demo:
        demo_inference(out)
