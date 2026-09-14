"""Independent 8+8 dimensional grammar filter; no main-model parameters/input hidden.

Input is pre-segmented text. Unknown candidate attributes are deliberately neutral.
Q/K learns context-to-attribute compatibility, not factual answer ranking.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn

from .model import BertConfig, BertLayer
from .tokenizer import SimpleBertTokenizer

ATTRIBUTES = ('PLACE', 'TIME', 'PERSON', 'ORG', 'TITLE', 'NUMBER', 'ACTION',
              'ENTITY', 'FUNCTION', 'PUNCT', 'INTERROGATIVE', 'CLAUSE', 'UNKNOWN')
STRUCTURES = ('NAME', 'LOCATION', 'TEMPORAL', 'PREDICATE', 'OBJECT', 'IDENTITY',
              'QUANTITY', 'CONDITION', 'COORDINATION', 'PARTICLE', 'BOUNDARY', 'OPEN')

# Explicit coarse grammar lexicon, independent of names in the main vocabulary.
# Synonyms share input symbols; candidate word identities remain unchanged.
ALIASES = {'名叫': '叫', '本名': '姓名', '原名': '姓名', '名为': '叫',
           '时候': '时', '动身': '出发', '启程': '出发', '过来': '来',
           '一名': '一位', '成为': '是', '若': '如果', '结伴': '一起',
           '一同': '一起', '共同': '一起', '拥有': '有', '共有': '有',
           '还在': '正在', '还': '已经', '居住': '起义'}
GRAMMAR_WORDS = set(('叫 姓名 的 是 在 住在 前往 去 起义 出发 于 时 期间 到达 '
                    '正在 呢 已经 开始 让 来 请 一位 担任 被 任命 为 有 个 部下 '
                    '带了 人 如果 就 因为 没有 只要 一起 来了 去了 到了 '
                    '后来 当时 今天 听说 据说 这位 。 ？ ， [MASK] [UNK]').split())


class IndependentGrammarFilter(nn.Module):
    def __init__(self, tokens, seeds=None):
        super().__init__()
        self.tokenizer = SimpleBertTokenizer()
        self.tokenizer.add_tokens(tokens)
        self.tokenizer.add_tokens(sorted(GRAMMAR_WORDS))
        self.embedding = nn.Embedding(len(self.tokenizer), 16, padding_idx=0)
        config = BertConfig(hidden_size=16, num_attention_heads=2,
                            intermediate_size=48, hidden_dropout_prob=0.,
                            attention_probs_dropout_prob=0.,
                            relative_position_max_distance=32)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(2)])
        # Retain immediate relative neighbors alongside contextual attention;
        # distant clauses must not erase a local trigger such as 姓名 是 [MASK].
        self.grammar = nn.Sequential(nn.Linear(48, 8), nn.Tanh())
        self.attribute = nn.Sequential(nn.Linear(48, 8), nn.Tanh())
        self.structure_head = nn.Linear(8, len(STRUCTURES))
        self.query = nn.Linear(16, 8)
        self.keys = nn.Parameter(torch.randn(len(ATTRIBUTES), 8) * .1)
        self.candidate_logits = nn.Parameter(torch.zeros(len(self.tokenizer), len(ATTRIBUTES)))
        targets = torch.zeros_like(self.candidate_logits)
        for token, names in (seeds or {}).items():
            if token in self.tokenizer.token_to_id:
                indices = [ATTRIBUTES.index(name) for name in names]
                targets[self.tokenizer.token_to_id[token], indices] = 1 / len(indices)
        self.register_buffer('seed_targets', targets)
        with torch.no_grad():
            self.candidate_logits.fill_(-6.)
            self.candidate_logits[:, -1] = 6.
            known = targets.sum(-1).gt(0)
            self.candidate_logits[known] = targets[known].clamp_min(1e-5).log()

    def encode(self, texts):
        texts = [' '.join(normalized if normalized in GRAMMAR_WORDS else '[UNK]'
                          for token in self.tokenizer.tokenize(text)
                          for normalized in [ALIASES.get(token, token)]) for text in texts]
        return self.tokenizer.batch_encode(texts, padding=True,
            max_length=max(len(self.tokenizer.tokenize(t)) + 2 for t in texts),
            return_tensors='pt')

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        embedded = self.embedding(input_ids)
        hidden = embedded
        mask = (~attention_mask.bool())[:, None, None, :].to(hidden.dtype) * -10000
        # Coarse clause scopes: delimiter belongs to the clause on its left.
        # This encodes a structural boundary, not an absolute token position.
        delimiters = torch.zeros_like(input_ids, dtype=torch.bool)
        for token in ('，', '。', '？'):
            delimiters |= input_ids.eq(self.tokenizer.token_to_id[token])
        scopes = delimiters.long().cumsum(-1) - delimiters.long()
        same_scope = scopes[:, :, None].eq(scopes[:, None, :])
        mask = mask + (~same_scope[:, None]).to(hidden.dtype) * -10000
        for layer in self.layers:
            hidden, _ = layer(hidden, mask)
        zero = torch.zeros_like(embedded[:, :1])
        previous = torch.cat((zero, embedded[:, :-1]), 1)
        following = torch.cat((embedded[:, 1:], zero), 1)
        local_context = torch.cat((hidden, previous, following), -1)
        g = self.grammar(local_context)
        h = self.attribute(local_context)
        query = self.query(torch.cat((g, h), -1))
        attribute_logits = query @ self.keys.T / math.sqrt(8)
        # A broad structural rule protects ALL legal alternatives in an
        # underdetermined slot, even if the learned head is overconfident.
        # 我在呢 / 我在沛县 / 我在跑; 到了哪 / 到了吗.
        allowed = torch.zeros(*input_ids.shape, len(ATTRIBUTES), dtype=torch.bool,
                              device=input_ids.device)
        prev_ids = torch.cat((torch.zeros_like(input_ids[:, :1]), input_ids[:, :-1]), 1)
        next_ids = torch.cat((input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])), 1)
        terminal = next_ids.eq(self.tokenizer.token_to_id['。']) | next_ids.eq(self.tokenizer.token_to_id['？'])
        for trigger, names in [('在', ('PLACE', 'TIME', 'ACTION', 'ENTITY', 'FUNCTION', 'INTERROGATIVE')),
                               ('到了', ('PLACE', 'ENTITY', 'FUNCTION', 'INTERROGATIVE'))]:
            positions = prev_ids.eq(self.tokenizer.token_to_id[trigger]) & terminal
            for name in names:
                allowed[..., ATTRIBUTES.index(name)] |= positions
        return {'g': g, 'h': h, 'structure_logits': self.structure_head(g),
                'attribute_logits': attribute_logits, 'ambiguous_allowed': allowed}

    def candidate_probabilities(self, candidate_tokens):
        ids = [self.tokenizer.token_to_id.get(t, 1) for t in candidate_tokens]
        probs = self.candidate_logits.softmax(-1)[ids]
        # Unseen/unlabelled words must not be assigned a made-up semantic type.
        known = self.seed_targets[ids].sum(-1).gt(0)
        unknown = torch.zeros_like(probs)
        unknown[:, -1] = 1
        return torch.where(known[:, None], probs, unknown)

    def bias(self, output, candidate_tokens, scale=3.):
        desired = output['attribute_logits'].softmax(-1)
        candidate = self.candidate_probabilities(candidate_tokens)
        # A multi-attribute seed (e.g. PERSON+ENTITY) must not be penalized
        # simply because its probability mass is shared between valid types.
        membership = candidate / candidate.max(-1, keepdim=True).values.clamp_min(1e-6)
        compatibility = desired @ membership.T + candidate[:, -1]
        # Abstain when uncertain. Finite penalties preserve ambiguity.
        confidence = desired.max(-1).values
        strength = ((confidence - .5) / .4).clamp(0, 1)
        allowed = output.get('ambiguous_allowed')
        if allowed is not None:
            ambiguous = allowed.any(-1)
            rule_compatible = (allowed.to(candidate.dtype) @ membership.T).gt(.5)
            rule_compatible |= candidate[:, -1].gt(.5)
            compatibility = torch.where(ambiguous.unsqueeze(-1),
                                        rule_compatible.to(candidate.dtype), compatibility)
            strength = torch.where(ambiguous, torch.ones_like(strength), strength)
        bias = scale * strength.unsqueeze(-1) * compatibility.clamp(1e-4, 1).log()
        # Special symbols are never valid lexical completions.
        for i, token in enumerate(candidate_tokens):
            if token in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'):
                bias[..., i] = -10000
        return bias

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(path))
        (path / 'grammar_config.json').write_text(json.dumps({
            'version': 4, 'grammar_dim': 8, 'attribute_dim': 8,
            'attributes': ATTRIBUTES, 'structures': STRUCTURES,
        }, ensure_ascii=False, indent=2), encoding='utf8')
        torch.save(self.state_dict(), path / 'grammar_filter.pt')

    @classmethod
    def from_pretrained(cls, directory):
        path = Path(directory)
        metadata = json.loads((path / 'grammar_config.json').read_text())
        if metadata['version'] != 4 or tuple(metadata['attributes']) != ATTRIBUTES:
            raise ValueError('Unsupported grammar filter schema')
        tokenizer = SimpleBertTokenizer.from_pretrained(str(path))
        model = cls(tokenizer.id_to_token)
        model.load_state_dict(torch.load(path / 'grammar_filter.pt',
                                        map_location='cpu', weights_only=True))
        return model


@torch.no_grad()
def filtered_prediction(main, main_tokenizer, grammar, text):
    """Evaluate an existing dynamic BERT without its old coupled attribute gate.

    Token positions align through raw segmented text, never by assuming matching IDs.
    Callers set both models to eval mode. All candidates are scored before top-k.
    """
    batch = main_tokenizer.encode(text, return_tensors='pt')
    batch = {k: v.to(next(main.parameters()).device) for k, v in batch.items()}
    hidden, _, _, _ = main._encode_with_route(batch['input_ids'],
        batch['token_type_ids'], batch['attention_mask'], None, False)
    raw = main.lm_head(hidden)
    small = grammar.encode([text])
    small = {k: v.to(next(grammar.parameters()).device) for k, v in small.items()}
    info = grammar(**small)
    bias = grammar.bias(info, main_tokenizer.id_to_token).to(raw.device)
    return raw, raw + bias, info
