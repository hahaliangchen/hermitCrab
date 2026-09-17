import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    # New models use relative position bias.  ``absolute`` remains available
    # only so checkpoints written before the switch can still be inspected.
    position_embedding_type: str = "relative"
    relative_position_max_distance: int = 128
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    mask_token_id: int = 103
    initializer_range: float = 0.02

    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d):
        values = dict(d)
        # Configs written by the pre-relative-position implementation did not
        # have a position type field and contain an absolute-position table.
        # Infer the legacy mode for those files; newly constructed configs
        # default to relative positions.
        values.setdefault("position_embedding_type", "absolute")
        return cls(**values)

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.position_embedding_type not in {"relative", "absolute"}:
            raise ValueError(
                "position_embedding_type must be 'relative' or 'absolute'"
            )
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.position_embedding_type == "absolute"
            else None
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids: torch.LongTensor, token_type_ids: Optional[torch.LongTensor] = None):
        batch_size, seq_length = input_ids.size()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeds
        if self.position_embeddings is not None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def _build_relative_position_bias(
    relative_position_bias: Optional[nn.Embedding],
    max_distance: int,
    seq_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Return a head-specific ``[1, heads, T, T]`` relative-position bias.

    The table is indexed by the signed key-minus-query distance.  Distances
    outside the configured range share the nearest boundary bucket, so the
    representation has no dependence on an absolute sequence offset.
    """
    if relative_position_bias is None:
        return None
    positions = torch.arange(seq_length, device=device)
    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
    relative_positions = relative_positions.clamp(-max_distance, max_distance)
    bucket_ids = relative_positions + max_distance
    bias = relative_position_bias(bucket_ids)
    return bias.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.relative_position_max_distance = config.relative_position_max_distance
        self.relative_position_bias = (
            nn.Embedding(
                2 * config.relative_position_max_distance + 1,
                self.num_heads,
            )
            if config.position_embedding_type == "relative"
            else None
        )

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape).permute(0, 2, 1, 3)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ``route_weights`` is accepted by the common attention interface so
        # experimental attention implementations can receive per-position
        # routing explicitly.  The standard BERT attention does not use it.
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._transpose_for_scores(query)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        relative_bias = _build_relative_position_bias(
            self.relative_position_bias,
            self.relative_position_max_distance,
            hidden_states.size(1),
            hidden_states.device,
            attn_scores.dtype,
        )
        if relative_bias is not None:
            attn_scores = attn_scores + relative_bias
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:2] + (self.all_head_size,)
        context = context.view(*new_context_shape)
        out = self.out_proj(context)
        return out, attn_probs


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        route_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_out, attn_probs = self.self_attn(
            hidden_states,
            attention_mask,
            route_weights=route_weights,
            **kwargs,
        )
        hidden_states = self.dropout(self_out) + hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states, attn_probs if output_attentions else None


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states) + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        route_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attn_probs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            route_weights=route_weights,
            **kwargs,
        )
        inter = self.intermediate(attn_output)
        layer_output = self.output(inter, attn_output)
        return layer_output, attn_probs


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        all_attentions = () if output_attentions else None
        for layer_module in self.layer:
            hidden_states, layer_attn = layer_module(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            if output_attentions:
                all_attentions = all_attentions + (layer_attn,)
        return hidden_states, all_attentions


class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token = hidden_states[:, 0]
        pooled = self.dense(first_token)
        pooled = self.activation(pooled)
        return pooled


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def _build_attention_mask(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        else:
            attention_mask = attention_mask.to(device=input_ids.device)
        extended = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -10000.0
        return extended

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        emb = self.embeddings(input_ids, token_type_ids)
        ext_mask = self._build_attention_mask(input_ids, attention_mask)
        seq_output, all_attentions = self.encoder(emb, ext_mask, output_attentions=output_attentions)
        pooled_output = self.pooler(seq_output)
        return seq_output, pooled_output, all_attentions

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            f.write(self.config.to_json_string())
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory: str):
        config = BertConfig.from_json_file(os.path.join(load_directory, "config.json"))
        model = cls(config)
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model


class BertForMaskedLM(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        sequence_output, _, all_attentions = self.bert(input_ids, token_type_ids, attention_mask, output_attentions=output_attentions)
        logits = self.lm_head(sequence_output)
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            return loss, logits, all_attentions
        return logits, all_attentions

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            f.write(self.config.to_json_string())
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory: str):
        config = BertConfig.from_json_file(os.path.join(load_directory, "config.json"))
        model = cls(config)
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.lm_head.weight = model.bert.embeddings.word_embeddings.weight
        return model

    # --- Helpers for compatibility with HF Trainer or utilities ---
    def get_input_embeddings(self) -> nn.Embedding:
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.bert.embeddings.word_embeddings = new_embeddings
        # re-tie
        self.lm_head.weight = self.bert.embeddings.word_embeddings.weight
        self.lm_head.out_features = new_embeddings.num_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_emb = self.bert.embeddings.word_embeddings
        old_num_tokens, emb_dim = old_emb.weight.shape
        if new_num_tokens == old_num_tokens:
            return old_emb
        new_emb = nn.Embedding(new_num_tokens, emb_dim, padding_idx=self.config.pad_token_id)
        new_emb.to(device=old_emb.weight.device, dtype=old_emb.weight.dtype)
        # init and copy
        new_emb.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_emb.weight.data[:num_to_copy] = old_emb.weight.data[:num_to_copy]
        if self.config.pad_token_id is not None and 0 <= self.config.pad_token_id < new_num_tokens:
            new_emb.weight.data[self.config.pad_token_id].zero_()
        self.set_input_embeddings(new_emb)
        self.config.vocab_size = new_num_tokens
        return new_emb
