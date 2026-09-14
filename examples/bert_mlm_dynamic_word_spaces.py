"""BERT MLM with contextual, learnable routing over dynamic 3D Q/K spaces.

The model has two separate attention paths:

* The first BERT layer is ordinary self-attention and creates a contextual
  hidden state for every token position.
* From ``route_start_layer`` onward, a learnable route Q/K scores the
  candidate spaces allowed by the token/context map.  The selected soft
  weights mix space-specific 3D Q/K scores into a per-head local attention
  bias, which is added to ordinary BERT attention.

The registry is only a candidate map and usage record.  It never supplies a
forward route by itself.  Route weights are computed inside each forward pass
and are passed explicitly to attention, so one call cannot affect the next.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
import bert_mlm_dimension_combinations as combination_base
from bert_simple.model import _build_relative_position_bias


DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "shiji_baihua_zhangchen_gaozu_long_context.txt"
    )
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs",
        "bert-mlm-dynamic-word-spaces-contextual",
    )
)
DYNAMIC_CONFIG_NAME = "dynamic_model_config.json"


class WordSpaceRegistry:
    """Token-to-candidate-space map and non-parametric usage bookkeeping.

    ``word_to_spaces`` contains references to candidate spaces, not copies of
    their matrices.  A few initial spaces are exposed to every regular token
    so the learnable router has more than one candidate from the first step.
    New spaces may be added only after a strong conflict persists for several
    steps; a single noisy cosine is not enough to split a space.
    """

    def __init__(
        self,
        tokenizer: combination_base.SimpleBertTokenizer,
        max_spaces: int,
        initial_spaces: int = 4,
    ):
        if max_spaces < 1:
            raise ValueError("max_spaces must be positive")
        if not 1 <= initial_spaces <= max_spaces:
            raise ValueError("initial_spaces must be between 1 and max_spaces")
        self.max_spaces = max_spaces
        self.initial_spaces = initial_spaces
        self.active_spaces = initial_spaces
        self.word_to_spaces: Dict[int, Set[int]] = {}
        self.space_contexts: List[Counter] = [Counter() for _ in range(max_spaces)]
        self.space_usage: List[int] = [0 for _ in range(max_spaces)]
        self.special_ids = {
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        }
        self.mask_token_id = tokenizer.mask_token_id
        self.tokenizer = tokenizer
        self._seed_initial_candidates()

    def _seed_initial_candidates(self) -> None:
        initial = set(range(self.initial_spaces))
        for token_id in range(len(self.tokenizer)):
            if token_id not in self.special_ids and token_id != self.mask_token_id:
                self.word_to_spaces[token_id] = set(initial)

    def allocate(self) -> int:
        if self.active_spaces >= self.max_spaces:
            raise RuntimeError("dynamic local-space capacity exhausted")
        space_id = self.active_spaces
        self.active_spaces += 1
        return space_id

    def candidates(self, token_id: int) -> List[int]:
        candidates = self.word_to_spaces.get(int(token_id))
        if not candidates:
            candidates = set(range(min(self.initial_spaces, self.active_spaces)))
        choices = sorted(space for space in candidates if space < self.active_spaces)
        return choices or [0]

    def _context_candidates(self, input_ids: Sequence[int], valid: Sequence[int]) -> Set[int]:
        spaces: Set[int] = set()
        for token_id, is_valid in zip(input_ids, valid):
            if (
                is_valid
                and token_id not in self.special_ids
                and token_id != self.mask_token_id
            ):
                spaces.update(self.candidates(int(token_id)))
        if not spaces:
            spaces.update(range(self.active_spaces))
        return spaces

    def candidate_mask(self, sample: Dict[str, object]) -> torch.Tensor:
        """Build ``[batch, sequence, max_spaces]`` candidate references.

        A masked position cannot use its gold label to choose candidates.  It
        uses the union of candidate spaces exposed by visible context tokens.
        This keeps the MLM route free of target leakage.
        """

        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        result = torch.zeros(
            batch_size,
            seq_len,
            self.max_spaces,
            dtype=torch.bool,
            device=input_ids.device,
        )
        for batch_index in range(batch_size):
            row_ids = [int(value) for value in input_ids[batch_index].tolist()]
            row_valid = [int(value) for value in attention_mask[batch_index].tolist()]
            context_spaces = self._context_candidates(row_ids, row_valid)
            for position, (token_id, is_valid) in enumerate(zip(row_ids, row_valid)):
                if not is_valid:
                    continue
                if token_id == self.mask_token_id:
                    choices = context_spaces
                elif token_id in self.special_ids:
                    choices = {0}
                else:
                    choices = set(self.candidates(token_id))
                for space_id in choices:
                    if 0 <= space_id < self.active_spaces:
                        result[batch_index, position, space_id] = True
        return result

    def touched_token_ids(self, sample: Dict[str, object]) -> Set[int]:
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        touched: Set[int] = set()
        for row_ids, row_labels, row_valid in zip(
            input_ids.tolist(), labels.tolist(), attention_mask.tolist()
        ):
            for token_id, label, is_valid in zip(row_ids, row_labels, row_valid):
                if not is_valid:
                    continue
                candidate = int(label) if label >= 0 else int(token_id)
                if candidate not in self.special_ids and candidate != self.mask_token_id:
                    touched.add(candidate)
        return touched

    def note_usage(self, sample: Dict[str, object], route_weights: torch.Tensor) -> None:
        """Record top-1 route usage without affecting the forward pass."""

        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        top_spaces = route_weights.detach().argmax(dim=-1).cpu().tolist()
        for batch_index, (row_ids, row_labels, row_valid) in enumerate(
            zip(input_ids.tolist(), labels.tolist(), attention_mask.tolist())
        ):
            context = {
                int(token_id)
                for token_id, is_valid in zip(row_ids, row_valid)
                if is_valid
                and token_id not in self.special_ids
                and token_id != self.mask_token_id
            }
            for position, (token_id, label, is_valid) in enumerate(
                zip(row_ids, row_labels, row_valid)
            ):
                if not is_valid:
                    continue
                target_id = int(label) if label >= 0 else int(token_id)
                if target_id in self.special_ids or target_id == self.mask_token_id:
                    continue
                space_id = int(top_spaces[batch_index][position])
                if not 0 <= space_id < self.active_spaces:
                    continue
                self.word_to_spaces.setdefault(target_id, set(range(self.initial_spaces))).add(
                    space_id
                )
                self.space_contexts[space_id].update(
                    other for other in context if other != target_id
                )
                self.space_usage[space_id] += 1

    def active_ids_from_mask(self, candidate_mask: torch.Tensor) -> Set[int]:
        nonzero = candidate_mask.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            return set()
        return {int(value) for value in nonzero[:, -1].tolist()}

    def attach_words(self, word_ids: Iterable[int], space_id: int) -> None:
        for token_id in set(int(word_id) for word_id in word_ids):
            if token_id not in self.special_ids and token_id != self.mask_token_id:
                self.word_to_spaces.setdefault(token_id, set(range(self.initial_spaces))).add(
                    space_id
                )

    def to_json(self) -> Dict[str, object]:
        mapping = {}
        for token_id, spaces in sorted(self.word_to_spaces.items()):
            if 0 <= token_id < len(self.tokenizer.id_to_token):
                mapping[self.tokenizer.id_to_token[token_id]] = sorted(spaces)
        space_summary = []
        for space_id in range(self.active_spaces):
            context_tokens = [
                self.tokenizer.id_to_token[token_id]
                for token_id, _ in self.space_contexts[space_id].most_common(12)
                if 0 <= token_id < len(self.tokenizer.id_to_token)
            ]
            space_summary.append(
                {
                    "space_id": space_id,
                    "usage": self.space_usage[space_id],
                    "context_tokens": context_tokens,
                }
            )
        return {
            "active_spaces": self.active_spaces,
            "max_spaces": self.max_spaces,
            "initial_spaces": self.initial_spaces,
            "word_to_spaces": mapping,
            "spaces": space_summary,
        }

    @classmethod
    def from_json(
        cls,
        tokenizer: combination_base.SimpleBertTokenizer,
        path: str,
        max_spaces: Optional[int] = None,
    ) -> "WordSpaceRegistry":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        stored_max = int(data.get("max_spaces", max_spaces or 1))
        capacity = max_spaces or stored_max
        registry = cls(
            tokenizer,
            max_spaces=capacity,
            initial_spaces=min(int(data.get("initial_spaces", 1)), capacity),
        )
        registry.active_spaces = min(int(data.get("active_spaces", 1)), capacity)
        registry.word_to_spaces = {}
        for token, spaces in data.get("word_to_spaces", {}).items():
            token_id = tokenizer.token_to_id.get(token)
            if token_id is not None:
                registry.word_to_spaces[int(token_id)] = {
                    int(space) for space in spaces if 0 <= int(space) < registry.active_spaces
                }
        for summary in data.get("spaces", []):
            space_id = int(summary.get("space_id", -1))
            if not 0 <= space_id < capacity:
                continue
            registry.space_usage[space_id] = int(summary.get("usage", 0))
            for token in summary.get("context_tokens", []):
                token_id = tokenizer.token_to_id.get(token)
                if token_id is not None:
                    registry.space_contexts[space_id][int(token_id)] += 1
        return registry


ATTRIBUTE_NAMES = (
    "PLACE",
    "TIME",
    "PERSON",
    "ORG",
    "TITLE",
    "NUMBER",
    "ACTION",
    "ENTITY",
    "FUNCTION",
    "PUNCT",
    "INTERROGATIVE",
    "CLAUSE",
    "UNKNOWN",
)
ATTRIBUTE_INDEX = {name: index for index, name in enumerate(ATTRIBUTE_NAMES)}

# 这些是低成本的结构锚点，不是完整词典。内容词的属性仍由模型从 MLM
# 梯度中学习；一个词可以同时拥有多个候选角色。
ATTRIBUTE_SEED_SPECS = {
    "在": {"attributes": ["FUNCTION"], "roles": ["PREPOSITION"]},
    "于": {"attributes": ["FUNCTION"], "roles": ["PREPOSITION"]},
    "从": {"attributes": ["FUNCTION"], "roles": ["PREPOSITION"]},
    "向": {"attributes": ["FUNCTION"], "roles": ["PREPOSITION"]},
    "到": {"attributes": ["FUNCTION"], "roles": ["PREPOSITION", "MOTION"]},
    "去": {"attributes": ["ACTION"], "roles": ["MOTION"]},
    "来": {"attributes": ["ACTION"], "roles": ["MOTION"]},
    "回": {"attributes": ["ACTION"], "roles": ["MOTION"]},
    "的": {"attributes": ["FUNCTION"], "roles": ["PARTICLE"]},
    "之": {"attributes": ["FUNCTION"], "roles": ["PARTICLE"]},
    "者": {"attributes": ["FUNCTION"], "roles": ["PARTICLE"]},
    "所": {"attributes": ["FUNCTION"], "roles": ["PARTICLE"]},
    "和": {"attributes": ["FUNCTION"], "roles": ["CONJUNCTION"]},
    "与": {"attributes": ["FUNCTION"], "roles": ["CONJUNCTION"]},
    "及": {"attributes": ["FUNCTION"], "roles": ["CONJUNCTION"]},
    "而": {"attributes": ["FUNCTION"], "roles": ["CONJUNCTION"]},
    "是": {"attributes": ["FUNCTION"], "roles": ["COPULA"]},
    "为": {"attributes": ["FUNCTION"], "roles": ["COPULA"]},
    "如果": {"attributes": ["FUNCTION"], "roles": ["CONDITION"]},
    "若": {"attributes": ["FUNCTION"], "roles": ["CONDITION"]},
    "因为": {"attributes": ["FUNCTION"], "roles": ["CAUSE"]},
    "由于": {"attributes": ["FUNCTION"], "roles": ["CAUSE"]},
    # 常见副词、助词和处置/被动标记也作为低成本功能词锚点，
    # 这样在内容槽位中可以被 grammar gate 可靠地筛掉。
    "了": {"attributes": ["FUNCTION"]},
    "也": {"attributes": ["FUNCTION"]},
    "就": {"attributes": ["FUNCTION"]},
    "又": {"attributes": ["FUNCTION"]},
    "把": {"attributes": ["FUNCTION"]},
    "被": {"attributes": ["FUNCTION"]},
    "都": {"attributes": ["FUNCTION"]},
    "还": {"attributes": ["FUNCTION"]},
    "便": {"attributes": ["FUNCTION"]},
    "仍": {"attributes": ["FUNCTION"]},
    "才": {"attributes": ["FUNCTION"]},
    "则": {"attributes": ["FUNCTION"]},
    "只": {"attributes": ["FUNCTION"]},
    "却": {"attributes": ["FUNCTION"]},
    "不": {"attributes": ["FUNCTION"]},
    "没": {"attributes": ["FUNCTION"]},
    "没有": {"attributes": ["FUNCTION"]},
    "已经": {"attributes": ["FUNCTION"]},
    "而且": {"attributes": ["FUNCTION"]},
    "但是": {"attributes": ["FUNCTION"]},
    "不过": {"attributes": ["FUNCTION"]},
    "所以": {"attributes": ["FUNCTION"]},
    "因此": {"attributes": ["FUNCTION"]},
    "于是": {"attributes": ["FUNCTION"]},
    "呢": {"attributes": ["FUNCTION"]},
    "吗": {"attributes": ["FUNCTION"]},
    "啊": {"attributes": ["FUNCTION"]},
    "吧": {"attributes": ["FUNCTION"]},
    # 疑问词不是硬标签：哪/哪里偏地点，何时偏时间，谁偏人物，
    # 什么保留通用实体属性；INTERROGATIVE 仍然是它们共有的结构属性。
    "哪": {"attributes": ["INTERROGATIVE", "PLACE"], "roles": ["INTERROGATIVE"]},
    "哪里": {"attributes": ["INTERROGATIVE", "PLACE"], "roles": ["INTERROGATIVE"]},
    "何时": {"attributes": ["INTERROGATIVE", "TIME"], "roles": ["INTERROGATIVE"]},
    "什么": {"attributes": ["INTERROGATIVE", "ENTITY"], "roles": ["INTERROGATIVE"]},
    "谁": {"attributes": ["INTERROGATIVE", "PERSON"], "roles": ["INTERROGATIVE"]},
    # 少量内容词锚点只用于把属性空间定向，不代表完整词典标注。
    "县": {"attributes": ["PLACE", "ENTITY"], "roles": ["PLACE_FOLLOWER"]},
    "郡": {"attributes": ["PLACE", "ENTITY"], "roles": ["PLACE_FOLLOWER"]},
    "城": {"attributes": ["PLACE", "ENTITY"], "roles": ["PLACE_FOLLOWER"]},
    "地": {"attributes": ["PLACE", "ENTITY"], "roles": ["PLACE_FOLLOWER"]},
    "国": {"attributes": ["PLACE", "ENTITY"], "roles": ["PLACE_FOLLOWER"]},
    "人": {"attributes": ["PERSON", "ENTITY"]},
    "王": {"attributes": ["PERSON", "TITLE"]},
    "侯": {"attributes": ["PERSON", "TITLE"]},
    "将军": {"attributes": ["PERSON", "TITLE"]},
    "时候": {"attributes": ["TIME"]},
    "时": {"attributes": ["TIME"]},
    "年": {"attributes": ["TIME", "NUMBER"]},
    "月": {"attributes": ["TIME", "NUMBER"]},
    "日": {"attributes": ["TIME", "NUMBER"]},
}

PUNCTUATION_CHARS = set(
    "，。！？；：、‘’“”「」『』（）()[]{}《》〈〉——…-.,!?;:'\""
)

CONTENT_ATTRIBUTES = {
    "PLACE",
    "TIME",
    "PERSON",
    "ORG",
    "TITLE",
    "NUMBER",
    "ACTION",
    "ENTITY",
    "INTERROGATIVE",
    "CLAUSE",
    "UNKNOWN",
}

# 这里只写结构允许的粗粒度集合，不写死具体答案。过滤采用 soft penalty，
# 这样“在”既可以接地点，也可以接时间；不可靠时仍保留 UNKNOWN。
GRAMMAR_ROLE_ALLOWED = {
    # “在”既可能是介词（在沛县），也可能是状态/存在谓词（我在呢）。
    # 因此保留 FUNCTION，但仍排除 PUNCT；真正的地点/时间倾向交给
    # contextual slot_head 决定。
    "PREPOSITION": {
        "PLACE",
        "TIME",
        "ENTITY",
        "FUNCTION",
        "INTERROGATIVE",
        "UNKNOWN",
    },
    "MOTION": {"PLACE", "TIME", "ENTITY", "INTERROGATIVE", "UNKNOWN"},
    "PARTICLE": set(CONTENT_ATTRIBUTES),
    "CONJUNCTION": set(CONTENT_ATTRIBUTES),
    "COPULA": set(CONTENT_ATTRIBUTES),
    "CONDITION": {"CLAUSE", "ACTION", "ENTITY", "UNKNOWN"},
    "CAUSE": {"CLAUSE", "ACTION", "ENTITY", "UNKNOWN"},
    "INTERROGATIVE": set(CONTENT_ATTRIBUTES),
    "TIME_FOLLOWER": {"TIME", "NUMBER", "ENTITY", "UNKNOWN"},
    "PLACE_FOLLOWER": {"PLACE", "ENTITY", "UNKNOWN"},
    "NAME_FOLLOWER": {"PERSON", "ENTITY", "UNKNOWN"},
    "QUANTITY": {"NUMBER", "ENTITY", "UNKNOWN"},
    "AFTER_PREPOSITION_OBJECT": {"ACTION", "ENTITY", "UNKNOWN"},
}


class WordAttributeRegistry:
    """少量词/字种子 + 可学习属性的初始化和语法触发器。

    这里不为整个词表人工标注属性。种子只固定少量功能词、疑问词和标点的
    方向；未标注词从 UNKNOWN 开始，并通过属性 gate 的 MLM 梯度逐渐分化。
    """

    def __init__(self, tokenizer: combination_base.SimpleBertTokenizer):
        self.tokenizer = tokenizer
        self.attribute_names = list(ATTRIBUTE_NAMES)
        self.seed_attributes: Dict[int, Set[int]] = {}
        self.seed_roles: Dict[int, Set[str]] = {}
        for token, spec in ATTRIBUTE_SEED_SPECS.items():
            token_id = tokenizer.token_to_id.get(token)
            if token_id is None:
                continue
            self.seed_attributes[int(token_id)] = {
                ATTRIBUTE_INDEX[name] for name in spec["attributes"]
            }
            self.seed_roles[int(token_id)] = set(spec.get("roles", []))
        for token_id, token in enumerate(tokenizer.id_to_token):
            if token_id in {
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.mask_token_id,
            }:
                continue
            if token and all(character in PUNCTUATION_CHARS for character in token):
                self.seed_attributes.setdefault(token_id, set()).add(
                    ATTRIBUTE_INDEX["PUNCT"]
                )

        self._role_token_ids: Dict[str, Set[int]] = {}
        for token_id, roles in self.seed_roles.items():
            for role in roles:
                self._role_token_ids.setdefault(role, set()).add(token_id)
        self._add_role_tokens(
            tokenizer,
            {
                "时候",
                "时",
                "年",
                "月",
                "日",
                "期间",
                "次年",
                "当时",
                "此时",
            },
            "TIME_FOLLOWER",
        )
        self._add_role_tokens(
            tokenizer,
            {
                "起义",
                "驻扎",
                "建都",
                "定都",
                "建都在",
                "定都在",
                "居住",
                "住在",
                "位于",
                "城",
                "县",
                "郡",
                "地",
                "国",
            },
            "PLACE_FOLLOWER",
        )
        # 词表按空格切词，常见的“定都在/前往/投靠”会作为一个 token；
        # 把这些复合触发词显式放回相应的粗语法关系中。
        self._add_role_tokens(
            tokenizer,
            {"建都在", "定都在", "位于", "住在"},
            "PREPOSITION",
        )
        self._add_role_tokens(
            tokenizer,
            {"前往", "来到", "进入", "到达", "投奔", "投靠", "回到", "迁往", "返回"},
            "MOTION",
        )
        self._add_role_tokens(
            tokenizer,
            {"本名", "原名", "姓名", "又称", "称为", "名为", "仍为"},
            "NAME_FOLLOWER",
        )
        self._add_role_tokens(
            tokenizer,
            {"有", "约有", "大约有", "共有", "拥有", "多少"},
            "QUANTITY",
        )

    def _add_role_tokens(
        self,
        tokenizer: combination_base.SimpleBertTokenizer,
        tokens: Set[str],
        role: str,
    ) -> None:
        for token in tokens:
            token_id = tokenizer.token_to_id.get(token)
            if token_id is not None:
                self._role_token_ids.setdefault(role, set()).add(int(token_id))

    def seed_ids(self) -> Dict[int, List[int]]:
        return {
            int(token_id): sorted(int(index) for index in attributes)
            for token_id, attributes in self.seed_attributes.items()
        }

    def role_ids(self) -> Dict[str, List[int]]:
        return {
            role: sorted(int(token_id) for token_id in token_ids)
            for role, token_ids in sorted(self._role_token_ids.items())
        }

    def allowed_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build a soft grammar-slot mask with shape ``[B,T,K]``.

        Previous/next trigger words constrain the slot, but the returned mask
        is only a prior for the learnable gate.  An empty intersection falls
        back to the union, preventing a malformed sentence from making every
        target impossible.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        result = torch.ones(
            batch_size,
            seq_len,
            len(ATTRIBUTE_NAMES),
            dtype=torch.bool,
            device=input_ids.device,
        )
        token_rows = input_ids.tolist()
        for batch_index, row in enumerate(token_rows):
            for position in range(seq_len):
                previous = row[position - 1] if position > 0 else None
                following = row[position + 1] if position + 1 < seq_len else None
                constraints: List[Set[str]] = []
                for role, token_ids in self._role_token_ids.items():
                    if previous in token_ids or following in token_ids:
                        constraints.append(GRAMMAR_ROLE_ALLOWED[role])
                # “在 垓下 [MASK]”这类结构中，当前位置是介词宾语之后的
                # 谓词槽，不应再次套用“在 + 宾语”的地点约束。
                preposition_ids = self._role_token_ids.get("PREPOSITION", set())
                if (
                    position >= 2
                    and row[position - 2] in preposition_ids
                    and previous not in {None, self.tokenizer.mask_token_id}
                ):
                    constraints.append(
                        GRAMMAR_ROLE_ALLOWED["AFTER_PREPOSITION_OBJECT"]
                    )
                if not constraints:
                    continue
                allowed = set.intersection(*constraints)
                if not allowed:
                    allowed = set.union(*constraints)
                result[batch_index, position] = torch.tensor(
                    [name in allowed for name in ATTRIBUTE_NAMES],
                    dtype=torch.bool,
                    device=input_ids.device,
                )
        return result

    def to_json(self) -> Dict[str, object]:
        return {
            "attribute_names": self.attribute_names,
            "seed_attributes": {
                self.tokenizer.id_to_token[token_id]: [
                    self.attribute_names[index] for index in sorted(attributes)
                ]
                for token_id, attributes in sorted(self.seed_attributes.items())
            },
            "seed_roles": {
                self.tokenizer.id_to_token[token_id]: sorted(roles)
                for token_id, roles in sorted(self.seed_roles.items())
            },
            "role_token_ids": self.role_ids(),
        }


class ContextualWordAttributeGate(nn.Module):
    """先由 grammar hidden 判断槽位，再用可学习词属性给词表加分。"""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        initializer_range: float,
        seed_ids: Optional[Dict[int, Sequence[int]]] = None,
        role_ids: Optional[Dict[str, Sequence[int]]] = None,
        gate_scale: float = 0.75,
        seed_loss_weight: float = 0.02,
        alignment_loss_weight: float = 0.05,
        soft_constraint_penalty: float = 3.0,
        known_seed_filter_penalty: float = 8.0,
    ):
        super().__init__()
        self.num_attributes = len(ATTRIBUTE_NAMES)
        self.unknown_index = ATTRIBUTE_INDEX["UNKNOWN"]
        self.gate_scale = gate_scale
        self.seed_loss_weight = seed_loss_weight
        self.alignment_loss_weight = alignment_loss_weight
        self.soft_constraint_penalty = soft_constraint_penalty
        self.known_seed_filter_penalty = known_seed_filter_penalty
        self.slot_head = nn.Linear(hidden_size, self.num_attributes)
        self.token_attribute_logits = nn.Parameter(
            torch.empty(vocab_size, self.num_attributes)
        )

        seed_ids = seed_ids or {}
        seed_mask = torch.zeros(vocab_size, self.num_attributes, dtype=torch.bool)
        seed_targets = torch.zeros(vocab_size, self.num_attributes)
        for token_id, attributes in seed_ids.items():
            if not 0 <= int(token_id) < vocab_size:
                continue
            valid_attributes = sorted(
                int(index)
                for index in attributes
                if 0 <= int(index) < self.num_attributes
            )
            if not valid_attributes:
                continue
            seed_mask[int(token_id), valid_attributes] = True
            seed_targets[int(token_id), valid_attributes] = 1.0 / len(valid_attributes)
        self.register_buffer("seed_mask", seed_mask)
        self.register_buffer("seed_targets", seed_targets)

        role_ids = role_ids or {}
        # 保存模型时只重建 checkpoint 中实际存在的 role 行，避免新版本
        # 新增语法 role 后旧权重的 role_token_mask 形状发生不匹配。
        role_names = sorted(role_ids) if role_ids else sorted(GRAMMAR_ROLE_ALLOWED)
        role_token_mask = torch.zeros(len(role_names), vocab_size, dtype=torch.bool)
        for role_index, role in enumerate(role_names):
            for token_id in role_ids.get(role, []):
                if 0 <= int(token_id) < vocab_size:
                    role_token_mask[role_index, int(token_id)] = True
        self.role_names = role_names
        self.register_buffer("role_token_mask", role_token_mask)
        self.reset_parameters(initializer_range)

    def reset_parameters(self, initializer_range: float) -> None:
        with torch.no_grad():
            self.slot_head.weight.normal_(mean=0.0, std=initializer_range)
            self.slot_head.bias.zero_()
            # 未标注词先保留 UNKNOWN，避免属性层在训练初期把整个词表硬筛空。
            self.token_attribute_logits.fill_(-0.5)
            self.token_attribute_logits[:, self.unknown_index] = 1.5
            seeded_rows = self.seed_mask.any(dim=-1)
            self.token_attribute_logits[seeded_rows] = -1.5
            self.token_attribute_logits[self.seed_mask] = 3.5

    def _grammar_allowed_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = input_ids.shape
        allowed = torch.ones(
            batch_size,
            seq_len,
            self.num_attributes,
            dtype=torch.bool,
            device=input_ids.device,
        )
        if self.role_token_mask.numel() == 0:
            return allowed
        rows = input_ids.tolist()
        role_masks = self.role_token_mask.detach().cpu()
        for batch_index, row in enumerate(rows):
            for position in range(seq_len):
                neighbors = []
                if position > 0:
                    neighbors.append(row[position - 1])
                if position + 1 < seq_len:
                    neighbors.append(row[position + 1])
                role_indices = [
                    role_index
                    for role_index, token_mask in enumerate(role_masks)
                    if any(
                        0 <= int(token_id) < token_mask.numel()
                        and bool(token_mask[int(token_id)])
                        for token_id in neighbors
                    )
                ]
                constraints = [
                    GRAMMAR_ROLE_ALLOWED.get(self.role_names[index], set(CONTENT_ATTRIBUTES))
                    for index in role_indices
                ]
                preposition_indices = [
                    index
                    for index, role in enumerate(self.role_names)
                    if role == "PREPOSITION"
                ]
                preposition_ids = {
                    int(token_id)
                    for index in preposition_indices
                    for token_id in self.role_token_mask[index].nonzero(
                        as_tuple=False
                    ).flatten().tolist()
                }
                previous = row[position - 1] if position > 0 else None
                if (
                    position >= 2
                    and row[position - 2] in preposition_ids
                    and previous is not None
                ):
                    constraints.append(
                        GRAMMAR_ROLE_ALLOWED["AFTER_PREPOSITION_OBJECT"]
                    )
                if not constraints:
                    continue
                allowed_names = set.intersection(*constraints)
                if not allowed_names:
                    allowed_names = set.union(*constraints)
                allowed[batch_index, position] = torch.tensor(
                    [name in allowed_names for name in ATTRIBUTE_NAMES],
                    dtype=torch.bool,
                    device=input_ids.device,
                )
        return allowed

    def forward(
        self,
        grammar_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        grammar_allowed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if grammar_allowed_mask is None:
            grammar_allowed_mask = self._grammar_allowed_mask(input_ids)
        grammar_allowed_mask = grammar_allowed_mask.to(
            device=grammar_hidden.device, dtype=torch.bool
        )
        slot_logits = self.slot_head(grammar_hidden)
        # 不允许的属性只受到软惩罚，不做不可逆 hard mask。
        slot_logits = slot_logits - (
            ~grammar_allowed_mask
        ).to(dtype=slot_logits.dtype) * self.soft_constraint_penalty
        slot_probs = torch.softmax(slot_logits, dim=-1)
        token_probs = torch.softmax(self.token_attribute_logits, dim=-1)
        compatibility = torch.einsum("btk,vk->btv", slot_probs, token_probs)
        attribute_bias = compatibility.clamp_min(1e-6).log() * self.gate_scale
        # 已知的结构锚点可以安全地做强约束：例如 grammar 槽不允许 PUNCT
        # 时，标点种子直接降权；未标注内容词仍只接受可学习的 soft gate。
        seeded = self.seed_mask.any(dim=-1)
        seed_compatible = torch.einsum(
            "btk,vk->btv",
            grammar_allowed_mask.to(dtype=slot_probs.dtype),
            self.seed_mask.to(device=slot_probs.device, dtype=slot_probs.dtype),
        ).gt(0.0)
        known_incompatible = seed_compatible.logical_not() & seeded.view(1, 1, -1)
        attribute_bias = attribute_bias - known_incompatible.to(
            dtype=attribute_bias.dtype
        ) * self.known_seed_filter_penalty
        return attribute_bias, {
            "slot_probs": slot_probs,
            "token_probs": token_probs,
            "grammar_allowed_mask": grammar_allowed_mask,
        }

    def auxiliary_loss(
        self,
        info: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        zero = self.token_attribute_logits.sum() * 0.0
        seeded = self.seed_mask.any(dim=-1)
        if bool(seeded.any()):
            seeded_logits = self.token_attribute_logits[seeded]
            seeded_targets = self.seed_targets[seeded]
            seed_loss = F.kl_div(
                F.log_softmax(seeded_logits, dim=-1),
                seeded_targets,
                reduction="batchmean",
            )
        else:
            seed_loss = zero

        if labels is None:
            return self.seed_loss_weight * seed_loss
        valid = labels.ge(0)
        if not bool(valid.any()):
            return self.seed_loss_weight * seed_loss
        target_ids = labels.clamp_min(0)
        target_attributes = info["token_probs"].index_select(
            0, target_ids.reshape(-1)
        ).view(*target_ids.shape, self.num_attributes)
        slot_probs = info["slot_probs"]
        alignment = F.kl_div(
            slot_probs.clamp_min(1e-8).log()[valid],
            target_attributes.detach()[valid],
            reduction="batchmean",
        )
        return (
            self.seed_loss_weight * seed_loss
            + self.alignment_loss_weight * alignment
        )

    def stats(
        self,
        info: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        slot_probs = info["slot_probs"]
        entropy = float(
            (
                -slot_probs.clamp_min(1e-8)
                * slot_probs.clamp_min(1e-8).log()
            )
            .sum(-1)
            .mean()
            .item()
        )
        result = {"slot_entropy": entropy}
        if labels is not None and bool(labels.ge(0).any()):
            target_ids = labels.clamp_min(0)
            target_attributes = info["token_probs"].index_select(
                0, target_ids.reshape(-1)
            ).view(*target_ids.shape, self.num_attributes)
            valid = labels.ge(0)
            target_prob = (
                (slot_probs * target_attributes).sum(-1)[valid].mean().item()
            )
            result["target_attribute_compatibility"] = float(target_prob)
        return result

    def seed_id_map(self) -> Dict[int, List[int]]:
        rows = self.seed_mask.detach().cpu()
        return {
            int(token_id): [int(index) for index in rows[token_id].nonzero().flatten().tolist()]
            for token_id in range(rows.size(0))
            if bool(rows[token_id].any())
        }

    def role_id_map(self) -> Dict[str, List[int]]:
        rows = self.role_token_mask.detach().cpu()
        return {
            role: [int(token_id) for token_id in rows[role_index].nonzero().flatten().tolist()]
            for role_index, role in enumerate(self.role_names)
            if bool(rows[role_index].any())
        }


class DynamicWordSpaceBank(nn.Module):
    """Per-layer, per-head bank of space-specific 3D Q/K transforms."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_spaces: int,
        hidden_size: int,
        route_dim: int,
        space_dim: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_spaces = max_spaces
        self.hidden_size = hidden_size
        self.route_dim = route_dim
        self.space_dim = space_dim
        self.p_q = nn.Parameter(
            torch.empty(num_layers, num_heads, max_spaces, hidden_size, space_dim)
        )
        self.p_k = nn.Parameter(
            torch.empty(num_layers, num_heads, max_spaces, hidden_size, space_dim)
        )
        self.a_q = nn.Parameter(
            torch.empty(num_layers, num_heads, max_spaces, space_dim, space_dim)
        )
        self.a_k = nn.Parameter(
            torch.empty(num_layers, num_heads, max_spaces, space_dim, space_dim)
        )
        self.space_descriptors = nn.Parameter(torch.empty(max_spaces, route_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.p_q.normal_(mean=0.0, std=0.02)
            self.p_k.normal_(mean=0.0, std=0.02)
            self.space_descriptors.normal_(mean=0.0, std=0.02)
            eye = torch.eye(
                self.space_dim,
                device=self.a_q.device,
                dtype=self.a_q.dtype,
            ).view(1, 1, 1, self.space_dim, self.space_dim)
            self.a_q.copy_(eye.expand_as(self.a_q))
            self.a_k.copy_(eye.expand_as(self.a_k))
            self.a_q.add_(torch.randn_like(self.a_q) * 0.01)
            self.a_k.add_(torch.randn_like(self.a_k) * 0.01)

    def initialize_slot(self, slot_id: int, source_slot: int = 0) -> None:
        if not 0 <= slot_id < self.max_spaces:
            raise ValueError("invalid local-space slot")
        if not 0 <= source_slot < self.max_spaces:
            raise ValueError("invalid source local-space slot")
        with torch.no_grad():
            self.p_q[:, :, slot_id].copy_(
                self.p_q[:, :, source_slot]
                + torch.randn_like(self.p_q[:, :, source_slot]) * 0.005
            )
            self.p_k[:, :, slot_id].copy_(
                self.p_k[:, :, source_slot]
                + torch.randn_like(self.p_k[:, :, source_slot]) * 0.005
            )
            self.a_q[:, :, slot_id].copy_(
                self.a_q[:, :, source_slot]
                + torch.randn_like(self.a_q[:, :, source_slot]) * 0.005
            )
            self.a_k[:, :, slot_id].copy_(
                self.a_k[:, :, source_slot]
                + torch.randn_like(self.a_k[:, :, source_slot]) * 0.005
            )
            self.space_descriptors[slot_id].copy_(
                self.space_descriptors[source_slot]
                + torch.randn_like(self.space_descriptors[source_slot]) * 0.005
            )

    def local_qk(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all candidate Q/K activations as ``[B,T,heads,S,3]``."""

        p_q = self.p_q[layer_index]
        p_k = self.p_k[layer_index]
        u_q = torch.einsum("btd,hsdk->bthsk", hidden_states, p_q)
        u_k = torch.einsum("btd,hsdk->bthsk", hidden_states, p_k)
        a_q = self.a_q[layer_index]
        a_k = self.a_k[layer_index]
        q = torch.einsum("bthsk,hskm->bthsm", u_q, a_q)
        k = torch.einsum("bthsk,hskm->bthsm", u_k, a_k)
        return q, k

    def local_scores(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Mix compatible spaces into a distinct local score per head."""

        q, k = self.local_qk(layer_index, hidden_states)
        q = q.permute(0, 2, 1, 3, 4)  # [B, heads, T, S, 3]
        k = k.permute(0, 2, 1, 3, 4)
        pair_scores = torch.einsum("bhtsd,bhjsd->bhtjs", q, k)
        weights = route_weights.to(dtype=pair_scores.dtype)
        scores = torch.einsum("bhtjs,bts,bjs->bhtj", pair_scores, weights, weights)
        return scores / math.sqrt(float(self.space_dim))

    def slot_gradient_vector(self, slot_id: int) -> torch.Tensor:
        pieces = []
        for parameter in (self.p_q, self.p_k, self.a_q, self.a_k):
            selected = parameter[:, :, slot_id]
            if parameter.grad is None:
                pieces.append(torch.zeros_like(selected).reshape(-1))
            else:
                pieces.append(parameter.grad[:, :, slot_id].detach().reshape(-1))
        selected = self.space_descriptors[slot_id]
        if self.space_descriptors.grad is None:
            pieces.append(torch.zeros_like(selected).reshape(-1))
        else:
            pieces.append(self.space_descriptors.grad[slot_id].detach().reshape(-1))
        return torch.cat(pieces)


class ContextualSpaceRouter(nn.Module):
    """Learnable route Q/K operating on contextual token states and spaces."""

    def __init__(self, hidden_size: int, route_dim: int, initializer_range: float):
        super().__init__()
        self.query = nn.Linear(hidden_size, route_dim)
        self.key = nn.Linear(route_dim, route_dim, bias=False)
        self.initializer_range = initializer_range
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.query.weight.normal_(mean=0.0, std=self.initializer_range)
            self.query.bias.zero_()
            self.key.weight.normal_(mean=0.0, std=self.initializer_range)

    def forward(
        self,
        contextual_hidden: torch.Tensor,
        space_descriptors: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(self.query(contextual_hidden), dim=-1)
        key = F.normalize(self.key(space_descriptors), dim=-1)
        logits = torch.einsum("btr,sr->bts", query, key)
        candidate_mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
        has_candidate = candidate_mask.any(dim=-1)
        safe_mask = candidate_mask.clone()
        safe_mask[..., 0] = safe_mask[..., 0] | ~has_candidate
        logits = logits.masked_fill(~safe_mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        # Padding positions have no candidates and must not contribute local
        # scores.  The temporary fallback above only prevents NaNs.
        return weights * candidate_mask.to(dtype=weights.dtype)


class WordSpaceSelfAttention(nn.Module):
    """Standard attention plus per-head, context-routed 3D local scores."""

    def __init__(
        self,
        original: nn.Module,
        bank: DynamicWordSpaceBank,
        layer_index: int,
        local_score_scale: float,
    ):
        super().__init__()
        self.qkv = original.qkv
        self.out_proj = original.out_proj
        self.dropout = original.dropout
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.all_head_size = original.all_head_size
        self.relative_position_max_distance = original.relative_position_max_distance
        self.relative_position_bias = original.relative_position_bias
        self.layer_index = layer_index
        self.local_score_scale = local_score_scale
        # The model owns the bank.  Do not register another state-dict path
        # through every attention wrapper.
        object.__setattr__(self, "bank", bank)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._transpose_for_scores(query)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        relative_bias = _build_relative_position_bias(
            self.relative_position_bias,
            self.relative_position_max_distance,
            hidden_states.size(1),
            hidden_states.device,
            attn_scores.dtype,
        )
        if relative_bias is not None:
            attn_scores = attn_scores + relative_bias
        if route_weights is not None:
            local_scores = self.bank.local_scores(
                self.layer_index,
                hidden_states,
                route_weights,
            )
            # local_scores is [B, heads, T, T]; every head gets its own bank.
            attn_scores = attn_scores + self.local_score_scale * local_scores
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:2], self.all_head_size)
        return self.out_proj(context), attn_probs


class DynamicWordSpaceBertForMaskedLM(combination_base.BertForMaskedLM):
    """BERT MLM whose dynamic route is computed from contextual hidden states."""

    def __init__(
        self,
        config: combination_base.BertConfig,
        bank: Optional[DynamicWordSpaceBank] = None,
        local_score_scale: float = 0.5,
        route_start_layer: int = 1,
        route_dim: int = 32,
        active_spaces: Optional[int] = None,
        attribute_seed_ids: Optional[Dict[int, Sequence[int]]] = None,
        attribute_role_ids: Optional[Dict[str, Sequence[int]]] = None,
        attribute_gate_scale: float = 0.75,
        attribute_seed_loss_weight: float = 0.02,
        attribute_alignment_loss_weight: float = 0.05,
        grammar_soft_constraint_penalty: float = 3.0,
        known_seed_filter_penalty: float = 8.0,
    ):
        super().__init__(config)
        if route_start_layer < 1:
            raise ValueError("route_start_layer must leave at least one grammar layer")
        if config.num_hidden_layers <= route_start_layer:
            raise ValueError("route_start_layer must leave at least one dynamic layer")
        if bank is None:
            bank = DynamicWordSpaceBank(
                config.num_hidden_layers,
                config.num_attention_heads,
                max_spaces=active_spaces or 1,
                hidden_size=config.hidden_size,
                route_dim=route_dim,
            )
        if bank.num_layers != config.num_hidden_layers:
            raise ValueError("bank and config have different layer counts")
        if bank.num_heads != config.num_attention_heads:
            raise ValueError("bank and config have different head counts")
        self.space_bank = bank
        self.router = ContextualSpaceRouter(
            config.hidden_size,
            bank.route_dim,
            config.initializer_range,
        )
        self.route_start_layer = route_start_layer
        self.route_dim = bank.route_dim
        self.local_score_scale = local_score_scale
        self.active_spaces = min(active_spaces or bank.max_spaces, bank.max_spaces)
        self.attribute_gate = ContextualWordAttributeGate(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            initializer_range=config.initializer_range,
            seed_ids=attribute_seed_ids,
            role_ids=attribute_role_ids,
            gate_scale=attribute_gate_scale,
            seed_loss_weight=attribute_seed_loss_weight,
            alignment_loss_weight=attribute_alignment_loss_weight,
            soft_constraint_penalty=grammar_soft_constraint_penalty,
            known_seed_filter_penalty=known_seed_filter_penalty,
        )
        for layer_index in range(route_start_layer, config.num_hidden_layers):
            layer = self.bert.encoder.layer[layer_index]
            layer.attention.self_attn = WordSpaceSelfAttention(
                layer.attention.self_attn,
                bank,
                layer_index,
                local_score_scale,
            )

    def _default_candidate_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            valid = input_ids.ne(self.config.pad_token_id)
        else:
            valid = attention_mask.to(device=input_ids.device).bool()
        candidate_mask = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.space_bank.max_spaces,
            dtype=torch.bool,
            device=input_ids.device,
        )
        candidate_mask[..., : self.active_spaces] = valid.unsqueeze(-1)
        return candidate_mask

    def _encode_with_route(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        candidate_space_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, ...]],
        torch.Tensor,
    ]:
        hidden_states = self.bert.embeddings(input_ids, token_type_ids)
        extended_mask = self.bert._build_attention_mask(input_ids, attention_mask)
        if candidate_space_mask is None:
            candidate_space_mask = self._default_candidate_mask(input_ids, attention_mask)
        if candidate_space_mask.shape != (
            input_ids.size(0),
            input_ids.size(1),
            self.space_bank.max_spaces,
        ):
            raise ValueError(
                "candidate_space_mask must have shape "
                f"[batch, sequence, {self.space_bank.max_spaces}]"
            )
        candidate_space_mask = candidate_space_mask.to(
            device=hidden_states.device, dtype=torch.bool
        )

        route_weights: Optional[torch.Tensor] = None
        grammar_hidden: Optional[torch.Tensor] = None
        all_attentions = () if output_attentions else None
        for layer_index, layer_module in enumerate(self.bert.encoder.layer):
            if layer_index == self.route_start_layer:
                grammar_hidden = hidden_states
                route_weights = self.router(
                    hidden_states,
                    self.space_bank.space_descriptors,
                    candidate_space_mask,
                )
            layer_route = route_weights if layer_index >= self.route_start_layer else None
            hidden_states, layer_attn = layer_module(
                hidden_states,
                extended_mask,
                output_attentions=output_attentions,
                route_weights=layer_route,
            )
            if output_attentions:
                all_attentions = all_attentions + (layer_attn,)
        if grammar_hidden is None:
            grammar_hidden = hidden_states
        return hidden_states, route_weights, all_attentions, grammar_hidden

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        candidate_space_mask: Optional[torch.Tensor] = None,
        grammar_allowed_mask: Optional[torch.Tensor] = None,
        return_route_weights: bool = False,
        return_attribute_info: bool = False,
        **kwargs,
    ):
        sequence_output, route_weights, all_attentions, grammar_hidden = self._encode_with_route(
            input_ids,
            token_type_ids,
            attention_mask,
            candidate_space_mask,
            output_attentions,
        )
        attribute_bias, attribute_info = self.attribute_gate(
            grammar_hidden,
            input_ids,
            grammar_allowed_mask=grammar_allowed_mask,
        )
        logits = self.lm_head(sequence_output) + attribute_bias
        if labels is not None:
            mlm_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = mlm_loss + self.attribute_gate.auxiliary_loss(
                attribute_info,
                labels,
            )
            result = (loss, logits, all_attentions)
        else:
            result = (logits, all_attentions)
        extras = []
        if return_route_weights:
            extras.append(route_weights)
        if return_attribute_info:
            extras.append(attribute_info)
        return result + tuple(extras)

    def save_pretrained(
        self,
        save_directory: str,
        active_spaces: Optional[int] = None,
    ) -> None:
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as handle:
            handle.write(self.config.to_json_string())
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        metadata = {
            "format_version": 3,
            "max_spaces": self.space_bank.max_spaces,
            "route_dim": self.route_dim,
            "route_start_layer": self.route_start_layer,
            "local_score_scale": self.local_score_scale,
            "active_spaces": int(active_spaces or self.active_spaces),
            "position_embedding_type": self.config.position_embedding_type,
            "attribute_names": list(ATTRIBUTE_NAMES),
            "attribute_gate_scale": self.attribute_gate.gate_scale,
            "attribute_seed_loss_weight": self.attribute_gate.seed_loss_weight,
            "attribute_alignment_loss_weight": self.attribute_gate.alignment_loss_weight,
            "grammar_soft_constraint_penalty": self.attribute_gate.soft_constraint_penalty,
            "known_seed_filter_penalty": self.attribute_gate.known_seed_filter_penalty,
            "attribute_seed_ids": {
                str(token_id): [int(index) for index in attributes]
                for token_id, attributes in self.attribute_gate.seed_id_map().items()
            },
            "attribute_role_ids": {
                role: [int(token_id) for token_id in token_ids]
                for role, token_ids in self.attribute_gate.role_id_map().items()
            },
        }
        with open(
            os.path.join(save_directory, DYNAMIC_CONFIG_NAME),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "DynamicWordSpaceBertForMaskedLM":
        config = combination_base.BertConfig.from_json_file(
            os.path.join(load_directory, "config.json")
        )
        metadata_path = os.path.join(load_directory, DYNAMIC_CONFIG_NAME)
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Dynamic model metadata not found: {metadata_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        attribute_seed_ids = {
            int(token_id): [int(index) for index in attributes]
            for token_id, attributes in metadata.get("attribute_seed_ids", {}).items()
        }
        attribute_role_ids = {
            str(role): [int(token_id) for token_id in token_ids]
            for role, token_ids in metadata.get("attribute_role_ids", {}).items()
        }
        bank = DynamicWordSpaceBank(
            config.num_hidden_layers,
            config.num_attention_heads,
            int(metadata["max_spaces"]),
            config.hidden_size,
            int(metadata["route_dim"]),
        )
        model = cls(
            config,
            bank=bank,
            local_score_scale=float(metadata["local_score_scale"]),
            route_start_layer=int(metadata["route_start_layer"]),
            route_dim=int(metadata["route_dim"]),
            active_spaces=int(metadata.get("active_spaces", bank.max_spaces)),
            attribute_seed_ids=attribute_seed_ids,
            attribute_role_ids=attribute_role_ids,
            attribute_gate_scale=float(metadata.get("attribute_gate_scale", 0.75)),
            attribute_seed_loss_weight=float(
                metadata.get("attribute_seed_loss_weight", 0.02)
            ),
            attribute_alignment_loss_weight=float(
                metadata.get("attribute_alignment_loss_weight", 0.05)
            ),
            grammar_soft_constraint_penalty=float(
                metadata.get("grammar_soft_constraint_penalty", 3.0)
            ),
            known_seed_filter_penalty=float(
                metadata.get("known_seed_filter_penalty", 8.0)
            ),
        )
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
        model.active_spaces = int(metadata.get("active_spaces", bank.max_spaces))
        return model


def _loss_for_sample(
    model: DynamicWordSpaceBertForMaskedLM,
    sample: Dict[str, object],
) -> torch.Tensor:
    result = model(
        input_ids=sample["input_ids"],
        token_type_ids=sample["token_type_ids"],
        attention_mask=sample["attention_mask"],
        labels=sample["labels"],
        candidate_space_mask=sample["candidate_space_mask"],
        grammar_allowed_mask=sample.get("grammar_allowed_mask"),
    )
    return result[0]


@torch.no_grad()
def _route_and_loss(
    model: DynamicWordSpaceBertForMaskedLM,
    sample: Dict[str, object],
) -> Tuple[float, torch.Tensor, Dict[str, torch.Tensor]]:
    result = model(
        input_ids=sample["input_ids"],
        token_type_ids=sample["token_type_ids"],
        attention_mask=sample["attention_mask"],
        labels=sample["labels"],
        candidate_space_mask=sample["candidate_space_mask"],
        grammar_allowed_mask=sample.get("grammar_allowed_mask"),
        return_route_weights=True,
        return_attribute_info=True,
    )
    return float(result[0].item()), result[-2], result[-1]


@torch.no_grad()
def _evaluate_memory(
    model: DynamicWordSpaceBertForMaskedLM,
    memory: Sequence[Dict[str, object]],
) -> Tuple[float, float]:
    if not memory:
        return 0.0, 0.0
    was_training = model.training
    model.eval()
    losses: List[float] = []
    forgetting: List[float] = []
    for item in memory:
        loss = float(_loss_for_sample(model, item["sample"]).item())
        losses.append(loss)
        forgetting.append(max(0.0, loss - float(item["reference_loss"])))
    if was_training:
        model.train()
    return sum(losses) / len(losses), sum(forgetting) / len(forgetting)


def _assign_gradients(
    parameters: Sequence[torch.nn.Parameter], gradients: Sequence[torch.Tensor]
) -> None:
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient


def _select_replay_index(
    memory: Sequence[Dict[str, object]], sample: Dict[str, object], step: int
) -> Tuple[Optional[int], int]:
    if not memory:
        return None, 0
    current_tokens = set(sample["touched_token_ids"])
    overlaps = [len(current_tokens & set(item["token_ids"])) for item in memory]
    max_overlap = max(overlaps)
    candidates = [index for index, value in enumerate(overlaps) if value == max_overlap]
    selected = candidates[(step - 1) % len(candidates)]
    return selected, max_overlap


def _route_stats(
    route_weights: torch.Tensor,
    candidate_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, object]:
    valid = attention_mask.bool().to(device=route_weights.device)
    candidate_mask = candidate_mask.to(device=route_weights.device)
    probabilities = route_weights[valid]
    if probabilities.numel() == 0:
        return {"entropy": 0.0, "top1_usage": {}}
    entropy = float(
        (-(probabilities.clamp_min(1e-9) * probabilities.clamp_min(1e-9).log()).sum(-1)).mean().item()
    )
    top1 = route_weights.argmax(dim=-1)[valid].tolist()
    counts = Counter(int(value) for value in top1)
    total = max(len(top1), 1)
    return {
        "entropy": entropy,
        "top1_usage": {str(space): count / total for space, count in sorted(counts.items())},
        "candidate_count_mean": float(candidate_mask[valid].sum(-1).float().mean().item()),
    }


def train(
    training_file: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_sentences: int = 100,
    max_length: int = 128,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: Optional[int] = None,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.0,
    epochs: int = 20,
    replay_weight: float = 0.5,
    max_spaces: int = 32,
    initial_spaces: int = 4,
    route_dim: int = 32,
    route_start_layer: int = 1,
    local_score_scale: float = 0.5,
    attribute_gate_scale: float = 0.75,
    attribute_seed_loss_weight: float = 0.02,
    attribute_alignment_loss_weight: float = 0.05,
    grammar_soft_constraint_penalty: float = 3.0,
    known_seed_filter_penalty: float = 8.0,
    conflict_threshold: float = -0.5,
    conflict_patience: int = 5,
    mlm_probability: float = 0.15,
    seed: int = 42,
    log_every: int = 100,
) -> str:
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if num_hidden_layers <= route_start_layer:
        raise ValueError("route_start_layer must leave at least one dynamic layer")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not 1 <= initial_spaces <= max_spaces:
        raise ValueError("initial_spaces must be between 1 and max_spaces")
    if conflict_patience < 1:
        raise ValueError("conflict_patience must be positive")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    combination_base.set_seed(seed)
    resolved_training_file = combination_base._resolve_path(training_file)
    texts = combination_base.load_texts(resolved_training_file, max_sentences)
    tokenizer = combination_base.SimpleBertTokenizer()
    tokenizer.train_from_texts(texts, min_freq=1)
    attribute_registry = WordAttributeRegistry(tokenizer)
    print(f"Using training file: {resolved_training_file}")
    print(f"Loaded {len(texts)} sentences; word-level vocab={len(tokenizer)}")
    print(
        f"Attribute seeds: {len(attribute_registry.seed_ids())} tokens; "
        f"grammar roles: {len(attribute_registry.role_ids())}"
    )

    config = combination_base.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_length + 10,
        position_embedding_type="relative",
        relative_position_max_distance=max_length,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    bank = DynamicWordSpaceBank(
        num_layers=num_hidden_layers,
        num_heads=num_attention_heads,
        max_spaces=max_spaces,
        hidden_size=hidden_size,
        route_dim=route_dim,
    )
    registry = WordSpaceRegistry(tokenizer, max_spaces, initial_spaces=initial_spaces)
    model = DynamicWordSpaceBertForMaskedLM(
        config,
        bank=bank,
        local_score_scale=local_score_scale,
        route_start_layer=route_start_layer,
        route_dim=route_dim,
        active_spaces=registry.active_spaces,
        attribute_seed_ids=attribute_registry.seed_ids(),
        attribute_role_ids=attribute_registry.role_ids(),
        attribute_gate_scale=attribute_gate_scale,
        attribute_seed_loss_weight=attribute_seed_loss_weight,
        attribute_alignment_loss_weight=attribute_alignment_loss_weight,
        grammar_soft_constraint_penalty=grammar_soft_constraint_penalty,
        known_seed_filter_penalty=known_seed_filter_penalty,
    ).to(torch.device("cpu"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    encodings: List[Dict[str, List[int]]] = []
    for text in texts:
        encodings.append(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_special_tokens_mask=True,
            )
        )

    def make_sample(encoded: Dict[str, List[int]], sample_seed: int) -> Dict[str, object]:
        sample = combination_base._make_masked_example(
            encoded,
            tokenizer,
            seed=sample_seed,
            mlm_probability=mlm_probability,
        )
        sample["touched_token_ids"] = sorted(registry.touched_token_ids(sample))
        sample["candidate_space_mask"] = registry.candidate_mask(sample)
        sample["grammar_allowed_mask"] = attribute_registry.allowed_mask(
            sample["input_ids"]
        )
        return sample

    memory: List[Dict[str, object]] = []
    memory_by_source: Dict[int, Dict[str, object]] = {}
    slot_history: Dict[int, torch.Tensor] = {}
    conflict_streak: Dict[int, int] = {}
    logs: List[Dict[str, object]] = []
    total_added_spaces = 0
    total_conflict_slots = 0
    total_conflicts_observed = 0
    total_replay_steps = 0
    total_replay_overlap = 0
    global_step = 0
    output_path = combination_base._resolve_path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    with open(
        os.path.join(output_path, "training_log.jsonl"), "w", encoding="utf-8"
    ) as log_handle:
        model.train()
        for epoch in range(epochs):
            for sample_index, encoded in enumerate(encodings):
                global_step += 1
                new_sample = make_sample(
                    encoded,
                    seed + epoch * 1000003 + sample_index * 1009,
                )
                replay_index, replay_overlap = _select_replay_index(
                    memory, new_sample, global_step
                )
                if replay_index is not None:
                    total_replay_steps += 1
                    total_replay_overlap += replay_overlap

                # Probe only for the conflict decision.  The real update below
                # is recomputed after a possible new candidate space is added.
                optimizer.zero_grad(set_to_none=True)
                probe_loss = _loss_for_sample(model, new_sample)
                probe_gradients = combination_base._gradient_tuple(
                    probe_loss, parameters
                )
                _assign_gradients(parameters, probe_gradients)
                active_current = registry.active_ids_from_mask(
                    new_sample["candidate_space_mask"]
                )
                conflicted: List[Tuple[int, float]] = []
                for space_id in sorted(active_current):
                    current_gradient = bank.slot_gradient_vector(space_id)
                    previous_gradient = slot_history.get(space_id)
                    current_norm = float(current_gradient.norm().item())
                    previous_norm = (
                        float(previous_gradient.norm().item())
                        if previous_gradient is not None
                        else 0.0
                    )
                    if previous_gradient is None or current_norm <= 1e-12 or previous_norm <= 1e-12:
                        continue
                    cosine = float(
                        torch.dot(current_gradient, previous_gradient).item()
                        / max(current_norm * previous_norm, 1e-12)
                    )
                    if cosine < conflict_threshold:
                        conflict_streak[space_id] = conflict_streak.get(space_id, 0) + 1
                        total_conflicts_observed += 1
                        if conflict_streak[space_id] >= conflict_patience:
                            conflicted.append((space_id, cosine))
                    else:
                        conflict_streak[space_id] = 0

                added_this_step = 0
                if conflicted and registry.active_spaces < max_spaces:
                    old_space, _ = min(conflicted, key=lambda item: item[1])
                    new_space = registry.allocate()
                    bank.initialize_slot(new_space, source_slot=old_space)
                    registry.attach_words(new_sample["touched_token_ids"], new_space)
                    conflict_streak[old_space] = 0
                    model.active_spaces = registry.active_spaces
                    added_this_step = 1
                    total_added_spaces += 1
                    total_conflict_slots += len(conflicted)
                    # Old memories keep their fixed MLM targets but receive the
                    # expanded candidate map for the next replay.
                    for item in memory:
                        item["sample"]["candidate_space_mask"] = registry.candidate_mask(
                            item["sample"]
                        )
                new_sample["candidate_space_mask"] = registry.candidate_mask(new_sample)

                optimizer.zero_grad(set_to_none=True)
                new_loss = _loss_for_sample(model, new_sample)
                old_loss_value = 0.0
                total_loss = new_loss
                if replay_index is not None:
                    old_loss = _loss_for_sample(model, memory[replay_index]["sample"])
                    old_loss_value = float(old_loss.item())
                    total_loss = total_loss + replay_weight * old_loss
                total_gradients = combination_base._gradient_tuple(total_loss, parameters)
                _assign_gradients(parameters, total_gradients)
                optimizer.step()

                # Read route diagnostics from the updated model.  These weights
                # are never stored on an attention module.
                model.eval()
                current_new_loss, route_weights, attribute_info = _route_and_loss(
                    model, new_sample
                )
                route_metrics = _route_stats(
                    route_weights,
                    new_sample["candidate_space_mask"],
                    new_sample["attention_mask"],
                )
                attribute_metrics = model.attribute_gate.stats(
                    attribute_info,
                    new_sample["labels"],
                )
                model.train()
                registry.note_usage(new_sample, route_weights)

                valid_positions = new_sample["attention_mask"].bool().to(
                    device=route_weights.device
                )
                active_after = {
                    int(value)
                    for value in route_weights.argmax(dim=-1)[valid_positions].tolist()
                    if 0 <= int(value) < registry.active_spaces
                }
                for space_id in registry.active_ids_from_mask(
                    new_sample["candidate_space_mask"]
                ):
                    current_gradient = bank.slot_gradient_vector(space_id)
                    if float(current_gradient.norm().item()) <= 1e-12:
                        continue
                    previous_gradient = slot_history.get(space_id)
                    if previous_gradient is None:
                        slot_history[space_id] = current_gradient.clone()
                    else:
                        slot_history[space_id] = 0.8 * previous_gradient + 0.2 * current_gradient

                if sample_index not in memory_by_source:
                    memory_item = {
                        "sample": new_sample,
                        "reference_loss": current_new_loss,
                        "active_spaces": sorted(active_after),
                        "token_ids": sorted(new_sample["touched_token_ids"]),
                    }
                    memory.append(memory_item)
                    memory_by_source[sample_index] = memory_item

                memory_loss = 0.0
                forgetting = 0.0
                if (
                    global_step == 1
                    or global_step % log_every == 0
                    or (epoch == epochs - 1 and sample_index == len(encodings) - 1)
                ):
                    memory_loss, forgetting = _evaluate_memory(model, memory)
                    model.train()
                    print(
                        f"epoch {epoch + 1}/{epochs}, step {global_step}: "
                        f"new_loss={float(new_loss.item()):.4f}, "
                        f"replay_loss={old_loss_value:.4f}, "
                        f"active_spaces={registry.active_spaces}, "
                        f"added={added_this_step}, "
                        f"route_entropy={route_metrics['entropy']:.4f}, "
                        f"slot_entropy={attribute_metrics['slot_entropy']:.4f}, "
                        f"memory_loss={memory_loss:.4f}, "
                        f"forgetting={forgetting:.4f}"
                    )

                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "sample_index": sample_index,
                    "new_loss": float(new_loss.item()),
                    "replay_loss": old_loss_value,
                    "replay_index": replay_index,
                    "replay_overlap_tokens": replay_overlap,
                    "current_new_loss": current_new_loss,
                    "memory_size": len(memory),
                    "active_spaces": registry.active_spaces,
                    "added_spaces": added_this_step,
                    "conflicted_slots": [
                        {"space_id": space_id, "cosine": cosine}
                        for space_id, cosine in conflicted
                    ],
                    "route_metrics": route_metrics,
                    "attribute_metrics": attribute_metrics,
                    "memory_loss": memory_loss,
                    "forgetting": forgetting,
                }
                logs.append(record)
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()

    model.save_pretrained(output_path, active_spaces=registry.active_spaces)
    tokenizer.save_pretrained(output_path)
    with open(
        os.path.join(output_path, "word_space_registry.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(registry.to_json(), handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(output_path, "word_attribute_registry.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(attribute_registry.to_json(), handle, ensure_ascii=False, indent=2)
    summary = {
        "stage": "bert_mlm_contextual_learnable_3d_route",
        "training_file": resolved_training_file,
        "device": "cpu",
        "sentences": len(texts),
        "epochs": epochs,
        "vocab_size": len(tokenizer),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "local_space_capacity": max_spaces,
        "initial_local_spaces": initial_spaces,
        "active_local_spaces": registry.active_spaces,
        "total_added_spaces": total_added_spaces,
        "total_conflict_slots": total_conflict_slots,
        "total_conflicts_observed": total_conflicts_observed,
        "conflict_threshold": conflict_threshold,
        "conflict_patience": conflict_patience,
        "route_dim": route_dim,
        "route_start_layer": route_start_layer,
        "local_score_scale": local_score_scale,
        "attribute_names": list(ATTRIBUTE_NAMES),
        "attribute_seed_count": len(attribute_registry.seed_ids()),
        "attribute_gate_scale": attribute_gate_scale,
        "attribute_seed_loss_weight": attribute_seed_loss_weight,
        "attribute_alignment_loss_weight": attribute_alignment_loss_weight,
        "grammar_soft_constraint_penalty": grammar_soft_constraint_penalty,
        "known_seed_filter_penalty": known_seed_filter_penalty,
        "replay_weight": replay_weight,
        "replay_steps": total_replay_steps,
        "total_replay_overlap_tokens": total_replay_overlap,
        "word_mapping_links": sum(
            len(spaces) for spaces in registry.word_to_spaces.values()
        ),
        "words_with_multiple_spaces": sum(
            1 for spaces in registry.word_to_spaces.values() if len(spaces) > 1
        ),
        "final_memory_loss": logs[-1]["memory_loss"],
        "final_forgetting": logs[-1]["forgetting"],
    }
    with open(
        os.path.join(output_path, "dynamic_word_space_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved dynamic word-space outputs to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT MLM with contextual learnable 3D-space routing."
    )
    parser.add_argument("data_path", nargs="?", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-sentences", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--max-spaces", type=int, default=32)
    parser.add_argument("--initial-spaces", type=int, default=4)
    parser.add_argument("--route-dim", type=int, default=32)
    parser.add_argument("--route-start-layer", type=int, default=1)
    parser.add_argument("--local-score-scale", type=float, default=0.5)
    parser.add_argument("--attribute-gate-scale", type=float, default=0.75)
    parser.add_argument("--attribute-seed-loss-weight", type=float, default=0.02)
    parser.add_argument(
        "--attribute-alignment-loss-weight", type=float, default=0.05
    )
    parser.add_argument(
        "--grammar-soft-constraint-penalty", type=float, default=3.0
    )
    parser.add_argument(
        "--known-seed-filter-penalty", type=float, default=8.0
    )
    parser.add_argument("--conflict-threshold", type=float, default=-0.5)
    parser.add_argument("--conflict-patience", type=int, default=5)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        training_file=args.data_path,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        replay_weight=args.replay_weight,
        max_spaces=args.max_spaces,
        initial_spaces=args.initial_spaces,
        route_dim=args.route_dim,
        route_start_layer=args.route_start_layer,
        local_score_scale=args.local_score_scale,
        attribute_gate_scale=args.attribute_gate_scale,
        attribute_seed_loss_weight=args.attribute_seed_loss_weight,
        attribute_alignment_loss_weight=args.attribute_alignment_loss_weight,
        grammar_soft_constraint_penalty=args.grammar_soft_constraint_penalty,
        known_seed_filter_penalty=args.known_seed_filter_penalty,
        conflict_threshold=args.conflict_threshold,
        conflict_patience=args.conflict_patience,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
        log_every=args.log_every,
    )
