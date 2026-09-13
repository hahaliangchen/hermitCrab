import os
import sys
import random
from dataclasses import asdict

import torch
from torch.utils.data import Dataset, DataLoader

# add project root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_simple import BertConfig, BertForMaskedLM


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomMLMDataset(Dataset):
    def __init__(self, size=1024, seq_len=32, vocab_size=200, mask_token_id=103, pad_token_id=0, mask_prob=0.15):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_prob = mask_prob
        assert 0 <= mask_token_id < vocab_size, "mask_token_id must be in [0, vocab_size)"
        self.data = self._make()

    def _make(self):
        data = []
        for _ in range(self.size):
            # Avoid sampling special ids 0 (PAD) and mask id as normal tokens.
            ids = torch.randint(low=1, high=self.vocab_size, size=(self.seq_len,))
            ids = torch.where(ids == self.mask_token_id, (ids + 1) % self.vocab_size, ids)

            # Create mask positions
            mask_positions = torch.rand(self.seq_len) < self.mask_prob
            labels = ids.clone()
            labels[~mask_positions] = -100  # only compute loss on masked positions
            ids = ids.clone()
            ids[mask_positions] = self.mask_token_id

            attention_mask = torch.ones(self.seq_len, dtype=torch.long)
            token_type_ids = torch.zeros(self.seq_len, dtype=torch.long)
            data.append((ids, token_type_ids, attention_mask, labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    token_type_ids = torch.stack([b[1] for b in batch], dim=0)
    attention_mask = torch.stack([b[2] for b in batch], dim=0)
    labels = torch.stack([b[3] for b in batch], dim=0)
    return input_ids, token_type_ids, attention_mask, labels


def train_and_save(output_dir: str = None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny config for quick demo
    seq_len = 32
    config = BertConfig(
        vocab_size=200,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=seq_len + 10,
        pad_token_id=0,
        mask_token_id=103,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    model = BertForMaskedLM(config).to(device)

    dataset = RandomMLMDataset(size=1024, seq_len=seq_len, vocab_size=config.vocab_size, mask_token_id=config.mask_token_id)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    model.train()
    steps = 0
    for epoch in range(3):
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
            if steps % 50 == 0:
                print(f"step {steps}: loss={loss.item():.4f}")
        # keep it quick
        if steps >= 300:
            break

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "mlm-mini"))
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    return output_dir, config


@torch.no_grad()
def demo_inference(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()

    # Build a toy input with one [MASK]
    cfg = BertConfig.from_json_file(os.path.join(model_dir, "config.json"))
    seq_len = 16
    x = torch.randint(low=5, high=cfg.vocab_size, size=(1, seq_len))
    pos = 7
    x[0, pos] = cfg.mask_token_id
    token_type_ids = torch.zeros_like(x)
    attention_mask = torch.ones_like(x)

    logits = model(input_ids=x.to(device), token_type_ids=token_type_ids.to(device), attention_mask=attention_mask.to(device))
    if isinstance(logits, tuple):
        logits = logits[0]

    topk = torch.topk(logits[0, pos], k=5)
    print("Masked position:", pos)
    print("Top-5 predicted token ids:", topk.indices.tolist())
    print("Logits:", [round(v, 3) for v in topk.values.tolist()])


if __name__ == "__main__":
    out_dir, _ = train_and_save()
    demo_inference(out_dir)
