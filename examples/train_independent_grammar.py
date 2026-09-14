"""Train only the independent grammar filter, then audit the frozen MLM.

Run: python3 examples/train_independent_grammar.py --epochs 100
Synthetic labels describe structures; they do not teach historical facts.
The final surface form of each family is held out from optimization/selection.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bert_simple.grammar_filter import (IndependentGrammarFilter, ATTRIBUTES,
                                       STRUCTURES, filtered_prediction)
from bert_simple.tokenizer import SimpleBertTokenizer
from bert_mlm_dynamic_word_spaces import ATTRIBUTE_SEED_SPECS, PUNCTUATION_CHARS

# Each family includes exceptions to a simple immediately-previous-token rule.
# Last form is a test-only surface construction, with disjoint subject words.
FAMILIES = [
 ('NAME', 'PERSON', ['{s} 叫 [MASK] 。', '{s} 的 姓名 是 [MASK] 。',
                     '{s} 名叫 [MASK] 。', '这位 {s} 叫 [MASK] 。']),
 ('LOCATION', 'PLACE', ['{s} 在 [MASK] 起义 。', '{s} 住在 [MASK] 。',
                        '{s} 前往 [MASK] 。', '{s} 在 [MASK] 居住 。']),
 ('TEMPORAL', 'TIME', ['{s} 在 [MASK] 时 出发 。', '{s} 于 [MASK] 时 启程 。',
                       '{s} 在 [MASK] 期间 到达 沛县 。', '{s} 在 [MASK] 时候 动身 。']),
 ('PREDICATE', 'ACTION', ['{s} 正在 [MASK] 。', '{s} 在 [MASK] 呢 。',
                          '{s} 已经 开始 [MASK] 。', '{s} 还在 [MASK] 呢 。']),
 ('OBJECT', 'PERSON', ['{s} 叫 [MASK] 去 沛县 。', '{s} 让 [MASK] 来 。',
                       '{s} 请 [MASK] 出发 。', '{s} 叫 [MASK] 过来 。']),
 ('IDENTITY', 'TITLE', ['{s} 是 一位 [MASK] 。', '{s} 担任 [MASK] 。',
                        '{s} 被 任命 为 [MASK] 。', '{s} 成为 一名 [MASK] 。']),
 ('QUANTITY', 'NUMBER', ['{s} 有 [MASK] 个 部下 。', '{s} 带了 [MASK] 人 。',
                         '{s} 共有 [MASK] 个 部下 。', '{s} 拥有 [MASK] 个 部下 。']),
 ('CONDITION', 'CLAUSE', ['如果 [MASK] ， {s} 就 出发 。',
                         '因为 [MASK] ， {s} 没有 出发 。',
                         '只要 [MASK] ， {s} 就 出发 。',
                         '若 [MASK] ， {s} 就 出发 。']),
 ('COORDINATION', 'FUNCTION', ['{s} [MASK] 他 一起 出发 。',
                              '{s} [MASK] 她 一同 起义 。',
                              '{s} [MASK] 大家 共同 出发 。',
                              '{s} [MASK] 我 结伴 出发 。']),
 ('PARTICLE', 'FUNCTION', ['{s} 在 [MASK] 。', '{s} 来了 [MASK] ？',
                          '{s} 去了 [MASK] ？', '{s} 还 在 [MASK] 。']),
 ('BOUNDARY', 'PUNCT', ['{s} 已经 出发 [MASK]', '{s} 在 沛县 起义 [MASK]',
                       '{s} 的 姓名 是 张三 [MASK]', '{s} 已经 到达 沛县 [MASK]']),
 ('OPEN', 'INTERROGATIVE', ['{s} 在 [MASK] ？', '{s} 去 [MASK] ？',
                           '{s} 叫 [MASK] ？', '{s} 到了 [MASK] ？']),
]


def dataset():
    sets = [[], [], []]
    subjects = [['他', '她', '我', '将军', '大王', '士兵', '使者', '朋友'],
                ['臣子', '客人'], ['高祖', '首领', '这人']]
    prefixes = ['', '后来 ', '当时 ', '今天 ', '听说 ', '据说 ']
    for structure, attr, forms in FAMILIES:
        for split in range(3):
            for form in (forms[-1:] if split == 2 else forms[:-1]):
                for subject in subjects[split]:
                    for prefix in prefixes:
                        sets[split].append((prefix + form.format(s=subject),
                                           STRUCTURES.index(structure), ATTRIBUTES.index(attr)))
    assert not set(x[0] for x in sets[0]) & set(x[0] for x in sets[2])
    # Composition curriculum: the presence of a comma alone does not imply
    # a conditional clause. Preserve the masked clause's original supervision.
    for split in (0, 1):
        base = list(sets[split])
        for text, structure, attr in base:
            sets[split].append(('先前 士兵 已经 出发 ， ' + text, structure, attr))
            if text.endswith(' 。'):
                sets[split].append((text[:-2] + ' ， 使者 已经 来了 。', structure, attr))
                if STRUCTURES[structure] not in ('PARTICLE', 'BOUNDARY', 'OPEN'):
                    sets[split].append((text[:-2] + ' 呢 。', structure, attr))
    return sets


def pack(model, examples):
    encoded = model.encode([x[0] for x in examples])
    positions = encoded['input_ids'].eq(model.tokenizer.mask_token_id).long().argmax(-1)
    return encoded, positions, torch.tensor([x[1] for x in examples]), torch.tensor([x[2] for x in examples])


def evaluate(model, examples):
    model.eval()
    with torch.no_grad():
        encoded, pos, structure, attr = pack(model, examples)
        out = model(**encoded)
        rows = torch.arange(len(pos))
        sp = out['structure_logits'][rows, pos].argmax(-1)
        ap = out['attribute_logits'][rows, pos].argmax(-1)
        probs = out['attribute_logits'][rows, pos].softmax(-1)
        loss = F.cross_entropy(out['structure_logits'][rows, pos], structure)
        loss = loss + F.cross_entropy(out['attribute_logits'][rows, pos], attr)
        wrong = []
        admissible_attributes = torch.zeros(len(examples), dtype=torch.bool)
        for i, example in enumerate(examples):
            admissible = {int(attr[i])}
            # These surface forms cannot uniquely identify a slot type:
            # 到了吗 / 到了哪; 我在呢 / 我在沛县 / 我在跑.
            if '到了 [MASK] ？' in example[0]:
                admissible.update((8, 10, 0))
            if '在 [MASK] 。' in example[0] and '住在' not in example[0]:
                admissible.update((8, 0, 1, 6, 7))
            admissible_attributes[i] = int(ap[i]) in admissible
            if ap[i] != attr[i] or sp[i] != structure[i]:
                wrong.append({'text': example[0], 'expected_structure': STRUCTURES[structure[i]],
                              'predicted_structure': STRUCTURES[sp[i]],
                              'expected_attribute': ATTRIBUTES[attr[i]],
                              'predicted_attribute': ATTRIBUTES[ap[i]],
                              'confidence': float(probs[i].max())})
        # Test both suppression of distractors and preservation of legal functions.
        candidate_names = ['在', '的', '，', '。', '呢', '去', '刘邦']
        bias = model.bias(out, candidate_names)[rows, pos]
        content = ~torch.isin(attr, torch.tensor([8, 9, 10]))
        legal_function = attr.eq(8)
        def mean_or_none(values):
            return float(values.float().mean()) if values.numel() else None
        return {'loss': float(loss), 'structure_accuracy': float(sp.eq(structure).float().mean()),
                'attribute_accuracy': float(ap.eq(attr).float().mean()),
                'admissible_attribute_accuracy': float(admissible_attributes.float().mean()),
                'content_function_punct_suppression_rate': mean_or_none(bias[content, :4] < -4),
                'legal_function_false_suppression_rate': mean_or_none(bias[legal_function, 4] < -4),
                'count': len(examples), 'errors': wrong}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def challenge(model):
    """Additional composition checks; admissible types need not be unique."""
    cases = [
        ('那位 将军 的 本名 是 [MASK] 。', ['PERSON', 'ENTITY']),
        ('高祖 叫 [MASK] 去 见 刘邦 。', ['PERSON', 'ENTITY']),
        ('今天 刘邦 在 [MASK] 时 起义 。', ['TIME']),
        ('据说 高祖 住在 [MASK] 。', ['PLACE', 'ENTITY']),
        ('将军 在 沛县 ， 刘邦 住在 [MASK] 。', ['PLACE', 'ENTITY']),
        ('刘邦 在 [MASK] 时 ， 将军 出发 。', ['TIME']),
        ('高祖 的 姓名 是 [MASK] ， 他 在 沛县 起义 。', ['PERSON', 'ENTITY']),
        ('如果 [MASK] ， 刘邦 就 前往 沛县 。', ['CLAUSE', 'ACTION']),
        ('刘邦 [MASK] 张耳 一起 前往 沛县 。', ['FUNCTION']),
        ('我 还 在 [MASK] 。', ['FUNCTION', 'PLACE', 'TIME', 'ACTION', 'ENTITY']),
        ('我 在 [MASK] ？', ['INTERROGATIVE', 'PLACE', 'ACTION', 'FUNCTION']),
        ('将军 正在 [MASK] ， 刘邦 已经 出发 。', ['ACTION']),
    ]
    records = []
    for text, allowed in cases:
        encoded = model.encode([text])
        position = encoded['input_ids'][0].tolist().index(model.tokenizer.mask_token_id)
        with torch.no_grad():
            info = model(**encoded)
            probabilities = info['attribute_logits'][0, position].softmax(-1)
        predicted = ATTRIBUTES[probabilities.argmax()]
        records.append({'text': text, 'admissible': allowed, 'predicted': predicted,
                        'correct': predicted in allowed, 'confidence': float(probabilities.max())})
    return {'accuracy': sum(x['correct'] for x in records)/len(records), 'cases': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--output', default=str(ROOT / 'outputs/independent-grammar-filter'))
    parser.add_argument('--main-model', default=str(ROOT / 'outputs/bert-mlm-dynamic-word-spaces-contextual'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    rng = random.Random(17)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    main_path = Path(args.main_model)
    original_hash = digest(main_path / 'pytorch_model.bin')
    main_tokenizer = SimpleBertTokenizer.from_pretrained(str(main_path))
    train, validation, test = dataset()
    seeds = {t: spec['attributes'] for t, spec in ATTRIBUTE_SEED_SPECS.items()}
    for token in main_tokenizer.id_to_token:
        if token and all(c in PUNCTUATION_CHARS for c in token):
            seeds[token] = ['PUNCT']
    tokens = list(main_tokenizer.id_to_token)
    for text, _, _ in train + validation + test:
        tokens.extend(text.split())
    model = (IndependentGrammarFilter.from_pretrained(output) if args.evaluate_only
             else IndependentGrammarFilter(tokens, seeds))
    optimizer = torch.optim.Adam(model.parameters(), lr=.004)
    previous_report = json.loads((output / 'report.json').read_text()) if args.evaluate_only else {}
    initial_test = previous_report.get('initial_test') if args.evaluate_only else evaluate(model, test)
    best, best_state = (-1., float('-inf')), {k: v.detach().clone() for k, v in model.state_dict().items()}
    logs = json.loads((output / 'training_log.json').read_text()) if args.evaluate_only else []
    for epoch in range(0 if args.evaluate_only else args.epochs):
        model.train()
        shuffled = rng.sample(train, len(train))
        total_loss = 0.
        for start in range(0, len(train), 64):
            encoded, pos, structure, attr = pack(model, shuffled[start:start+64])
            out = model(**encoded)
            rows = torch.arange(len(pos))
            known = model.seed_targets.sum(-1).gt(0)
            seed_loss = -(model.seed_targets[known] * model.candidate_logits[known].log_softmax(-1)).sum(-1).mean()
            loss = F.cross_entropy(out['structure_logits'][rows, pos], structure)
            loss = loss + F.cross_entropy(out['attribute_logits'][rows, pos], attr) + .1 * seed_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(pos)
        val = evaluate(model, validation)
        score = (val['structure_accuracy'] + val['attribute_accuracy'], -val['loss'])
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        logs.append({'epoch': epoch+1, 'loss': total_loss/len(train),
                     'validation_structure': val['structure_accuracy'],
                     'validation_attribute': val['attribute_accuracy']})
        if (epoch+1) % 10 == 0:
            print(json.dumps(logs[-1]), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    if not args.evaluate_only:
        model.save_pretrained(output)
    restored = IndependentGrammarFilter.from_pretrained(output).eval()
    probe = model.encode(['这位 高祖 叫 [MASK] 。'])
    with torch.no_grad():
        reload_diff = float((model(**probe)['attribute_logits'] - restored(**probe)['attribute_logits']).abs().max())
    from bert_mlm_dynamic_word_spaces import DynamicWordSpaceBertForMaskedLM
    main_model = DynamicWordSpaceBertForMaskedLM.from_pretrained(str(main_path)).eval()
    main_model.requires_grad_(False)
    probes = []
    for text in ['这位 高祖 叫 [MASK] 。', '高祖 叫 [MASK] 。',
                 '高祖 的 原名 是 [MASK] 。', '刘邦 在 [MASK] 起义 。',
                 '我 在 [MASK] 。', '我 在 [MASK] ？']:
        raw, final, info = filtered_prediction(main_model, main_tokenizer, model, text)
        position = main_tokenizer.encode(text)['input_ids'].index(main_tokenizer.mask_token_id)
        target = main_tokenizer.token_to_id['刘邦']
        def rank(logits):
            return int((logits[0, position] > logits[0, position, target]).sum()) + 1
        def top(logits):
            return [main_tokenizer.id_to_token[i] for i in logits[0, position].topk(10).indices.tolist()]
        probes.append({'text': text, 'raw_top10': top(raw), 'filtered_top10': top(final),
                       'liubang_raw_rank': rank(raw), 'liubang_filtered_rank': rank(final),
                       'attribute': ATTRIBUTES[info['attribute_logits'][0, position].argmax()],
                       'confidence': float(info['attribute_logits'][0, position].softmax(-1).max())})
    # A real main-model weight mutation must have zero effect on the independent branch.
    with torch.no_grad():
        before = model(**probe)['attribute_logits'].clone()
        parameter = next(main_model.parameters())
        saved = parameter.clone()
        parameter.add_(.1)
        independence_diff = float((model(**probe)['attribute_logits'] - before).abs().max())
        parameter.copy_(saved)
    def normalized(example):
        return tuple(model.encode([example[0]])['input_ids'][0].tolist())
    seen_inputs = {normalized(x) for x in train}
    novel_test = [x for x in test if normalized(x) not in seen_inputs]
    report = {'train_count': len(train), 'validation_count': len(validation),
              'normalized_train_unique_count': len(seen_inputs),
              'normalized_test_overlap_count': len(test) - len(novel_test),
              'normalized_novel_test': evaluate(model, novel_test) if novel_test else None,
              'composition_challenge': challenge(model),
              'test_protocol': 'surface holdout used for development; normalized novel subset reported separately; synthetic only',
              'initial_test': initial_test, 'train': evaluate(model, train),
              'validation': evaluate(model, validation), 'heldout_test': evaluate(model, test),
              'reload_max_diff': reload_diff, 'main_weight_change_filter_max_diff': independence_diff,
              'main_checkpoint_unchanged': original_hash == digest(main_path / 'pytorch_model.bin'),
              'main_checkpoint_sha256': original_hash, 'probes': probes,
              'limitations': ['Synthetic closed-family benchmark, not general Chinese grammar mastery.',
                              'Unseeded candidate words retain UNKNOWN and neutral filtering.',
                              'Independent filter does not backpropagate into or improve main factual memory.']}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (output / 'training_log.json').write_text(json.dumps(logs, indent=2), encoding='utf8')
    (output / 'dataset.json').write_text(json.dumps({'train': train, 'validation': validation, 'test': test}, ensure_ascii=False), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('initial_test', 'train', 'validation', 'heldout_test')}, ensure_ascii=False), flush=True)
    print('heldout:', {k:v for k,v in report['heldout_test'].items() if k != 'errors'}, flush=True)


if __name__ == '__main__':
    main()
