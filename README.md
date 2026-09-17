# BERT Simple（三维关系与动态 Q/K 实验版）

### 核心架构思想与设计哲学（必读）
📖 **[认知双系统与稀疏流形门控：HermitCrab 设计思想与对话精粹](DESIGN_PHILOSOPHY_AND_COGNITIVE_ARCHITECTURE.md)**  
详细阐述了 System 1 语感习惯与 System 2 18 维流形推理的认知解耦、全注意力的“勤奋假象”、过拟合本源与内嵌式 If-Else 稀疏门控机制。

📐 **[前置 3D 关系流形与因果电路架构设计规范](PRE_FFN_3D_MANIFOLD_AND_CAUSAL_CIRCUIT_DESIGN.md)**  
详细定义了前置 3D 审查 FFN、合向量去成分、3 维捆绑归零、GELU 晶体管电路门控、反向拔插头测试与 ReLU 平方纯净稀疏注意力机制。


### 18 维关系 FFN 与纠错回退（实验功能）

数据生成规范见 [RELATION_TRAINING_DATA_SPEC.md](RELATION_TRAINING_DATA_SPEC.md)，
实现说明见 [RELATION_FFN_IMPLEMENTATION.md](RELATION_FFN_IMPLEMENTATION.md)。
协议 v2 与成对训练的改进说明见 [RELATION_TRAINING_V2_IMPROVEMENTS.md](RELATION_TRAINING_V2_IMPROVEMENTS.md)。
架构诊断与问题复盘日志见 [PROBLEM_LOG_RELATION_SUBSPACE_AND_GENERALIZATION_2026-09-16.md](PROBLEM_LOG_RELATION_SUBSPACE_AND_GENERALIZATION_2026-09-16.md)。
动态事实模型可设置 `relation_ffn_hidden_size=32` 开启共享的 18→32→1 FFN。
默认关闭以兼容旧模型；新 JSONL 请用 `examples/train_relation_pairs.py`，包含全词表 CE、
负例 margin、不变性对称 KL，以及抽样纠错回退。训练与答题共用不读取答案的候选策略。
当前关系训练默认使用 `max_length=256`（含 `[CLS]`/`[SEP]`）和至少 12 个内容 token；
超长或过短样本直接拒绝，不静默截断，也不为 token 两两组合分配关系空间。
`examples/train_shiji_structured_relation.py` 保留为旧 manifest 对照，其答案条件候选不能
用于证明未知答案泛化。尚无真实数据泛化效果结论。

这是一个极简的 BERT 实现，当前用于研究三维关系组合、动态三维 Q/K 和事实记忆保持。Matrix-Free 方向已废弃，不再作为当前架构目标。

## 核心特性 (为什么它很特别)

### 1. 极简主义 BERT 代码
*   **零臃肿**: 所有的核心逻辑 (`BertEmbeddings`, `SelfAttention`, `Encoder`) 都被浓缩在 `bert_simple/model.py` 中。
*   **可读性强**: 不同于 HuggingFace 的工业级实现，这个实现保留了 embedding、self-attention、encoder 和 MLM 的完整主链，便于直接改造。

### 2. 三维动态关系通道
*   **静态三维组合**: 训练时限制和保护关系更新通道，允许共享参数小幅适应，严重冲突时再隔离。
*   **动态三维 Q/K**: 先用普通 BERT 第一层生成每个位置自己的上下文表示，再由可学习 route Q/K 在固定共享三维通道 bank 中选择权重；后续层使用每个 head 独立的动态三维局部分数。空间不按 token 对、事实或句子新增。
*   **相对位置注意力**: 新模型不再把绝对位置向量加到 token 上，而是在每个 attention head 的 QK 分数中加入有方向的相对距离 bias；序列整体平移不会改变位置关系。
*   **可调试性**: 模型支持 `output_attentions=True`，也会记录路由熵、候选数量和 space 使用情况。
*   **独立语法—词属性过滤器**: `bert_simple/grammar_filter.py` 使用独立 embedding、相对位置注意力、8 维 `g_t` 和 8 维 `h_t`，通过独立 Q/K 预测属性兼容度。过滤器不读取主 BERT hidden，不更新主模型，主模型权重改变也不会改变过滤器输出。旧 checkpoint 的耦合属性 gate 保留兼容，新推理入口绕过它，避免重复降权。

### 3. 动态 "历史专家" 分词器
*   **中文优先设计**: `SimpleBertTokenizer` 按语料中的空格分词，已分词的中文词保持为一个 token。
*   **动态词表**: 支持 `train_from_texts` 方法，它可以扫描历史文本（如《史记》），并瞬间构建一个只包含相关字符的自定义词表。彻底解决古文生僻字的 OOV (Out of Vocabulary) 问题。

## 使用方法

### 基于自定义文本训练 (例如：史记 / 历史文本)
训练脚本 `examples/bert_mlm_dynamic_word_spaces.py` 使用现有语料实现上下文路由和动态三维 Q/K。

**命令:**
```bash
python examples/bert_mlm_dynamic_word_spaces.py \
  examples/shiji_baihua_zhangchen_gaozu_long_context.txt \
  --epochs 20 --hidden-size 256
```
*   **输入**: 一个文本文件，每行一句话/一段话。
*   **流程**:
    1.  扫描文件以构建动态词表。
    2.  从头开始训练一个使用相对位置注意力的 BERT 掩码语言模型 (MLM)。
    3.  训练并保存模型、固定共享三维空间使用统计和 `word_attribute_registry.json` 到 `outputs/bert-mlm-dynamic-word-spaces-contextual`。

如果不提供文件参数，动态三维 Q/K 脚本会默认读取项目中现有的 `examples/shiji_baihua_zhangchen_gaozu_long_context.txt`。训练从随机初始化开始，不依赖旧权重；默认使用 CPU、hidden_size=256，可以通过命令行参数调整训练轮数和共享路由空间数量。旧的 `word_to_spaces`/词对注册表不再生成。

### 事实记忆分支中的动态 Q/K

当前事实记忆主线保留了 MLM 输出侧的 3×3 relation matrix，同时把上下文路由的动态 Q/K 接入 `route_start_layer` 之后的 self-attention。固定共享 bank 默认有 1500 个候选，其中前 86 个基础通道覆盖完整 256 维；具体路由由当前 token 位置自己的 contextual hidden 通过 router 动态计算。关系 sidecar 训练时把 MASK 对应的全部关系上下文成员先聚合，再由特殊 FFN 判断整组上下文是否支持答案，不能退化为 `词A || 词B` 的事实空间。局部 Q/K 分数在 attention softmax 之前加入。relation matrix、动态 Q/K 矩阵和 router 一起进入 adapter 更新组，训练日志会记录它们的梯度范数与实际更新范数。

```bash
python examples/train_full_relation_filter_stages.py \
  --base-checkpoint outputs/bert-mlm-fact-memory-shiji-grammar-attribute-256/before_fact \
  --dataset data/shiji/manifests/relation_training_v2_draft.jsonl \
  --output outputs/context-relation-filter-full256-stage1 \
  --memory-budget-mb 1024
```

该入口先冻结主 BERT，按特殊 FFN → contextual route → 动态 Q/K 三阶段训练关系 sidecar。
原有 `train_shiji_fact_memory_dynamic_qk.py` 和 `train_shiji_fact_memory_local_relation_margin_v2.py`
仍可作为历史/静态对照，但它们的 token-pair allocator 结果不能作为当前泛化方案或新训练入口。

### 关系增强的《史记》事实记忆数据

事实记忆数据由当前仓库的分词语料 `data/shiji/segmented/shiji_segmented.txt` 构建（当前为 10,631 行），不再只依赖 100 行示例节选。构建命令：

```bash
python examples/build_shiji_fact_dataset.py
```

结果写入 `data/shiji/manifests/fact_memory_dataset.json`，并同步到 `outputs/fact-memory/facts_dataset.json`；覆盖报告在 `data/shiji/manifests/fact_memory_report.json`。除原有任命、攻伐、战果和籍贯事实外，关系样本还记录标题—人物、别名、本名/字号、父子、祖孙、兄弟、婚姻和世系关系，并保留 `relation_type`、`relation_direction`、`linked_entity`、`source_line` 等字段，供动态路由和事实回放使用。

报告中的关系覆盖率是“当前分词源文件中出现明确关系模式的行”的覆盖率，不等同于已经完成《史记》全部章节的人工语义标注。提取器采用词典与句式规则，遇到分词粘连且无法安全做单词元遮罩的姓名会保守跳过。

### 单独训练语法过滤器（保留已训练主 BERT）

> 完整的数据存放目录、真实语料训练命令与指标规范请参阅：[GRAMMAR_LAYER_TRAINING_GUIDE.md](GRAMMAR_LAYER_TRAINING_GUIDE.md)。

```bash
# 基于最新提取的《史记》1.2万条真实语法属性数据集训练：
python examples/train_independent_grammar.py --dataset data/shiji/manifests/dataset.json --epochs 50

# 基于传统合成规则模板训练：
python examples/train_independent_grammar.py --epochs 100
python examples/train_independent_grammar.py --evaluate-only
python examples/test_independent_grammar.py
```

默认读取已有主模型的词表，在 `outputs/independent-grammar-filter` 保存独立权重、完整数据划分、训练曲线和 `report.json`。训练使用可枚举的粗结构模板以及前后分句组合，不向主模型灌入模板事实；留出每类最后一种句式与独立主语做开发回归，验证集用于选择 checkpoint（准确率相同时比较验证损失）。报告另列归一化后未见的输入和更复杂的组合句。数据仍属于合成闭集，指标不能等同于掌握全部中文语法。

接入已加载的动态 BERT：

```python
from bert_simple.grammar_filter import IndependentGrammarFilter, filtered_prediction
grammar = IndependentGrammarFilter.from_pretrained("outputs/independent-grammar-filter").eval()
main_model.eval()
raw_logits, filtered_logits, info = filtered_prediction(
    main_model, main_tokenizer, grammar, "这位 高祖 叫 [MASK] 。")
```

两边分别编码同一份空格分词文本，通过词字符串对齐候选词表。过滤全词表后再取 top-k；未知内容词保持中性，低置信度时减弱过滤，合法的功能词/标点槽位也参与训练。当前未标注词的语义属性尚未完成自动归纳，不能把这部分中性放行描述为已经学会人物或地点分类。过滤器只能减少结构不合适的候选，不能自动增强主 BERT 事实记忆。

过滤器输入使用显式的小型结构词表，结构同义形式共享输入符号，词表外内容词归一成通用占位符，避免记住训练主语。该归一化保持 token 数量与位置，不改变主模型的输入或候选词身份。不同原句归一化后可能相同，因此报告包含归一化重复数量。新领域的结构词需要扩展词表；对长句、嵌套及多解结构还需独立验证。

`在 [MASK] 。/？`、`到了 [MASK] 。/？` 使用显式多解允许集合保护合法属性，避免分类头过度自信时误筛“呢/吗/哪”。这是结构规则保护，不等于模型已学会完整的多标签属性分布。其余结构使用学习分数。`--evaluate-only` 重算报告，不重新训练或改写过滤器权重。

### 可视化 (也就是那个 "显微镜")
要查看模型内部的“思维过程/注意力”：
```bash
python visualize_heads.py
```
这将会为一句示例文本绘制出 Attention 热力图，展示每对 Token 之间的连接强度。
