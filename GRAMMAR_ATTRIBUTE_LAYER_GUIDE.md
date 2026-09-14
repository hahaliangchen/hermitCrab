# 固定语法状态机与可学习词语属性层

这是当前版本的训练说明。旧的
`GRAMMAR_LAYER_TRAINING_GUIDE.md` 描述的是 `IndependentGrammarFilter`：它
仍然是一个带小型 Transformer 的旧方案。本文件描述现在正在验证的新方案。

## 1. 两层职责

当前把“语法”和“词是什么”明确拆开：

```text
空格分词句子 + [MASK]
             │
             ├─ FixedGrammarAutomaton
             │      固定规则匹配 → 当前位置允许的属性集合
             │
             └─ WordAttributeLayer
                    可训练 word × attribute 表 → 候选词属性
                                      │
                                      └─ candidate bias
```

### 固定语法层

源码是 `bert_simple/grammar_automaton.py`。它不参加梯度更新，只匹配遮罩
位置左右的固定模式，例如：

* `姓名 是 [MASK]` → `NAME`，优先允许 `PERSON/ENTITY`；
* `在 [MASK] 时` → `TEMPORAL`，允许 `TIME`；
* `已经 [MASK]` → `PREDICATE`，允许 `ACTION`；
* `有 [MASK] 个` → `QUANTITY`，允许 `NUMBER`。

规则有优先级；同一优先级的多条规则不会强行选一个解释，而是合并允许
集合。没有匹配时返回 `OPEN`，过滤器 abstain，不应误杀主模型候选。这是
第一版局部状态机，长距离语义仍交给 BERT。

### 可学习词语属性层

源码是 `bert_simple/word_attribute_layer.py`。当前不是为每个词生成一套
巨大的 Q/K 矩阵，而是维护一个透明的可训练表：

```text
word_attribute_logits: [vocab_size, 13]
```

13 个属性是 `PLACE/TIME/PERSON/ORG/TITLE/NUMBER/ACTION/ENTITY/FUNCTION/
PUNCT/INTERROGATIVE/CLAUSE/UNKNOWN`。每个词一行、每个属性一个 logit，
经过 sigmoid 后可以同时拥有多个属性，例如：

```text
刘邦  → PERSON + ENTITY
魏国  → PLACE  + ENTITY
将军  → PERSON + TITLE
```

因此这里的 13 是属性标签数量，不是 BERT 的 hidden size，也不是
`256` 维三维组合。未知词由 `known_token_mask` 标记为未知，在候选评分中
保持中性，避免“没学过”被误判成“不合法”。

## 2. 数据构建

数据脚本为 `examples/build_word_attribute_data.py`，默认读取已经分词的：

```text
data/shiji/segmented/shiji_segmented.txt
```

它产生：

```text
data/shiji/manifests/word_attribute_dataset.json
```

数据包含两类信号：

1. 少量稳定词典作为强弱标注，例如人物、地名、官职、功能词、时间词和
   数量词；
2. 从真实句子中挖出的遮罩槽位作为弱标注，例如 `已经 [MASK]` 的动作槽。

词典证据权重为 8，槽位证据权重为 1。这样一个地名不会因为一次噪声较大
的触发词就被盲目合并成 `PERSON/TITLE`。这仍是弱监督数据，不等于人工
完成了整部《史记》的词典标注。

重新构建命令：

```bash
python examples/build_word_attribute_data.py
```

## 3. 训练

训练入口为 `examples/train_grammar_attribute_layers.py`：

```bash
python examples/train_grammar_attribute_layers.py \
  --data data/shiji/manifests/word_attribute_dataset.json \
  --tokenizer outputs/bert-mlm-fact-memory-zhangchen-gaozu-20x-256 \
  --epochs 200 \
  --learning-rate 0.03 \
  --output outputs/grammar-attribute-filter
```

这一步只优化词属性表，不更新主 BERT 的 Embedding、QKV、FFN 或 MLM
输出层。固定语法自动机也没有梯度，因此不会引入主模型的灾难性遗忘。
真正接入主模型时，`GrammarAttributeFilter.candidate_bias()` 会在主模型
输出 logits 后、取 top-k 前提供加性 bias。

## 4. 当前实验记录

最近一次 CPU 训练产物在：

```text
outputs/grammar-attribute-filter/
```

对应的 `report.json` 显示：

* 语料：10,631 句；
* 槽位样本：1,413 条；
* 词属性数据中有标签的词：1,186 个；
* 当前主模型词表：2,596 个 token，其中 327 个能和这些标签对齐；
* 训练 200 轮，loss 从约 `0.693` 降到约 `0.043`；
* 固定语法槽位属性命中率约 `97.10%`，非回退匹配率约 `75.16%`。

词表对齐数量只有 327，是因为当前主模型 checkpoint 来自之前的较小节选，
不是完整 10,631 句语料的词表。后续要覆盖全部史记词汇，需要用完整语料
重建主 tokenizer，并同步重建主 BERT；不能把两个不同词表的属性行直接
当成同一个 token。

词属性训练集上的 `micro_f1=1.0` 和 `exact_match=1.0` 只说明这张直接
参数表拟合了当前弱标注行，不能证明对未见词或未见篇章的泛化。下一轮应
按篇章/段落留出词和句式，报告真正的盲测结果。

## 5. 回归测试

```bash
python -m unittest examples.test_grammar_attribute_layers examples.test_independent_grammar
```

当前结果：9 个测试全部通过。测试覆盖固定规则、同一词多个属性、未知词
中性放行和句末特殊 token 处理。

## 6. 与主 BERT 的边界

当前版本已经完成“训练语法层和词属性层”的独立实验，但还没有把 bias
硬接进 `BertForMaskedLM.forward()`。这是有意保留的边界：先验证属性表和
状态机，再做主模型对照实验。接入时应同时比较：

* 原始 MLM logits；
* 加固定语法 bias；
* 加固定语法 bias + 词属性 bias。

并记录事实词 Top-1/Top-5、合法功能词误压制率和旧样本遗忘率，确认过滤层
是在改善候选排序，而不是用过强规则把答案空间直接剪坏。
