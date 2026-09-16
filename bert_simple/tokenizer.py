import json
import os
import re
from collections import Counter
from typing import List, Dict, Any, Optional


CJK_RANGE = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
)


def _is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    for a, b in CJK_RANGE:
        if a <= cp <= b:
            return True
    return False


class SimpleBertTokenizer:
    """
    Minimal tokenizer for mixed CN/EN text.
    - Tokens are split by whitespace and kept as-is.
    - The training data is expected to be pre-segmented, so Chinese words are
      not split into individual CJK characters.

    Provides: tokenize, encode, batch_encode, save_pretrained, from_pretrained
    Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
    """

    def __init__(self):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.ent_token = "[ENT]"

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        # initialize with specials in fixed order
        for tok in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token, self.ent_token]:
            self._add_token(tok)

    # --- Special token ids ---
    @property
    def ent_token_id(self) -> int:
        return self.token_to_id[self.ent_token]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id[self.cls_token]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id[self.sep_token]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.mask_token]

    # --- Core ---
    def __len__(self) -> int:
        return len(self.id_to_token)

    def _add_token(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        idx = len(self.id_to_token)
        self.token_to_id[tok] = idx
        self.id_to_token.append(tok)
        return idx

    def add_tokens(self, tokens: List[str]) -> int:
        added = 0
        for t in tokens:
            if t not in self.token_to_id:
                self._add_token(t)
                added += 1
        return added

    def train_from_texts(self, texts: List[str], min_freq: int = 1):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        for tok, freq in counter.items():
            if freq >= min_freq:
                self._add_token(tok)

    def tokenize(self, text: str) -> List[str]:
        # Identify special tokens and preserve them as single units
        special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        # Escape for regex
        pattern = "|".join([re.escape(t) for t in special_tokens])

        tokens: List[str] = []
        # Split by special tokens first, then split the remaining text only by
        # whitespace. The corpus already contains word segmentation, so every
        # whitespace-delimited chunk remains one token, including multi-character
        # Chinese words such as "大禹" and "督导".
        for part in re.split(f"({pattern})", text.strip()):
            if not part:
                continue
            if part in special_tokens:
                tokens.append(part)
            else:
                for chunk in re.split(r"\s+", part.strip()):
                    if chunk:
                        tokens.append(chunk)
        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None,
        return_special_tokens_mask: bool = False,
        add_new_tokens: bool = False,
    ) -> Dict[str, Any]:
        tokens = self.tokenize(text)
        if add_new_tokens:
            self.add_tokens(tokens)
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

        num_special_tokens = 2 if add_special_tokens else 0
        if max_length is not None and len(ids) + num_special_tokens > max_length:
            if truncation:
                ids = ids[:max_length - num_special_tokens]
            else:
                raise ValueError("Sequence length exceeds max_length and truncation=False")

        special_tokens_mask: List[int] = []
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
            special_tokens_mask = [1] + [0] * (len(ids) - 2) + [1]
        else:
            special_tokens_mask = [0] * len(ids)

        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        if padding and max_length is not None and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            token_type_ids = token_type_ids + [0] * pad_len
            special_tokens_mask = special_tokens_mask + [1] * pad_len  # treat pad as special for masking exclusion

        out = {
            "input_ids": ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        if return_special_tokens_mask:
            out["special_tokens_mask"] = special_tokens_mask

        if return_tensors == "pt":
            import torch

            out = {k: torch.tensor(v, dtype=torch.long).unsqueeze(0) for k, v in out.items()}
            if return_special_tokens_mask:
                out["special_tokens_mask"] = out["special_tokens_mask"].long()
        return out

    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        outs = [self.encode(t, **kwargs) for t in texts]
        if not outs:
            return {}
        keys = outs[0].keys()

        if kwargs.get("return_tensors") == "pt":
            import torch

            return {k: torch.cat([o[k] for o in outs], dim=0) for k in keys}

        batch = {k: [] for k in keys}
        for o in outs:
            for k in keys:
                batch[k].append(o[k])

        import torch
        for k in keys:
            batch[k] = torch.tensor(batch[k], dtype=torch.long)
        return batch

    # --- IO ---
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        cfg = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "SimpleBertTokenizer":
        tok = cls()
        # reset and load
        with open(os.path.join(load_directory, "vocab.json"), "r", encoding="utf-8") as f:
            tok.token_to_id = json.load(f)
        # rebuild id_to_token sorted by id
        items = sorted(tok.token_to_id.items(), key=lambda x: x[1])
        tok.id_to_token = [t for t, _ in items]
        # load special tokens (optional)
        cfg_path = os.path.join(load_directory, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            tok.pad_token = cfg.get("pad_token", tok.pad_token)
            tok.unk_token = cfg.get("unk_token", tok.unk_token)
            tok.cls_token = cfg.get("cls_token", tok.cls_token)
            tok.sep_token = cfg.get("sep_token", tok.sep_token)
            tok.mask_token = cfg.get("mask_token", tok.mask_token)
        return tok

    # --- utility ---
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        toks = []
        for i in ids:
            if skip_special_tokens and i in {
                self.pad_token_id,
                self.cls_token_id,
                self.sep_token_id,
                self.mask_token_id,
            }:
                continue
            tok = self.id_to_token[i] if 0 <= i < len(self.id_to_token) else self.unk_token
            toks.append(tok)
        # The corpus is word-segmented, so decoding restores one space between
        # every non-special token. This also preserves one-character words such
        # as "在" and "的" instead of merging them into neighboring words.
        return " ".join(toks).strip()
