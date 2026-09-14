# 2026-09-14 开发记录与后续计划

## 一、今天的总体结论

当前项目的主线已经从“只依赖共享 Q/K 和词频”调整为：

```text
分词后的史记文本
        ↓
语法结构 / 词语属性先验
        ↓
BERT 上下文表示
        ↓
事实记忆 + 局部 3×3 关系适配器
        ↓
后续再接生成式解码器
```

当前仍然使用 256 维 BERT 作为底座。三维组合不是预先创建 276 万套矩阵，
而是从 256 维地址空间中为实际出现的关系分配少量局部三维组合，并通过关系
映射表复用这些组合。

本次最后没有启动训练，也没有删除已有权重。新增的 SGD 入口已经完成语法和
导入检查，等待下一轮正式对照实验。

## 二、今天完成的代码与数据工作

### 1. 语法层和词语属性层

- 增加独立的语法属性过滤器及其模型代码：
  `bert_simple/grammar_attribute_filter.py`、
  `bert_simple/grammar_attribute_model.py`。
- 增加独立的结构自动机和词语属性层：
  `bert_simple/grammar_automaton.py`、
  `bert_simple/word_attribute_layer.py`。
- 支持从分词后的《史记》数据集加载训练、验证和测试划分。
- 语法层负责缩小合法候选范围，词语属性层提供 `PERSON`、`PLACE`、
  `TITLE`、功能词、标点等粗粒度属性；未标注词仍保留 `UNKNOWN`，避免错误的
  硬过滤。
- `examples/train_independent_grammar.py` 增加 `--dataset` 参数；主 BERT
  不存在时可以跳过主模型探针，独立语法层仍能训练和评估。

相关文档：

- `GRAMMAR_LAYER_TRAINING_GUIDE.md`
- `GRAMMAR_ATTRIBUTE_LAYER_GUIDE.md`
- `GRAMMAR_ATTRIBUTE_LAYER_V2_CHANGELOG.md`
- `GRAMMAR_ATTRIBUTE_LAYER_V3_CHANGELOG.md`

### 2. 《史记》数据与事实记忆数据集

- 保留原始、句级和分词后的史记数据：
  `data/shiji/raw/`、`data/shiji/sentences/`、`data/shiji/segmented/`。
- 增加词语属性数据集和事实记忆数据集：
  `data/shiji/manifests/dataset.json`、
  `data/shiji/manifests/word_attribute_dataset.json`、
  `data/shiji/manifests/fact_memory_dataset.json`。
- 增加事实数据构建脚本、事实位置处理、报告生成和回归测试。
- 事实记忆训练继续从已完成语法/属性基础的 checkpoint 开始，基础 checkpoint
  保存在：
  `outputs/bert-mlm-fact-memory-shiji-grammar-attribute-256/before_fact`。

### 3. 词频先验的角色调整

词频先验现在只作为“没有足够语法、属性和上下文证据时的低权重兜底”：

- 内容词的词频偏置较低；
- 功能词偏置更低；
- 标点符号偏置最低；
- 事实记忆阶段词频先验冻结，不让词频梯度重新主导事实判断；
- 语法和属性证据优先于词频证据。

这解决了模型在 `[MASK]` 位置过度偏爱句号、`了`、`的`、`去` 等高频候选的
一部分问题，但不能单独保证答案正确，仍需要上下文和事实损失共同约束。

### 4. 局部三维关系适配器

增加了 `bert_simple/local_relation_adapter_model.py` 和 v2 版本：
`bert_simple/local_relation_adapter_model_v2.py`。

当前关系记忆机制为：

1. 句子中实际出现的关系三元组分配局部三维维度组合；
2. 每个实际关系拥有一个 3×3 双线性关系矩阵，而不是为所有组合预分配矩阵；
3. 每个词通过关系映射表记录自己参与过的关系组合；
4. 候选 key 做 L2 归一化，局部关系分数使用单独的尺度；
5. 使用上下文词和当前 Top-8 候选作为 hard negative；
6. 使用 margin loss 让正确事实和近似错误事实拉开差距；
7. 保留 replay 梯度，保护旧事实；
8. 全局 BERT 路径只保留很弱的 0.02 梯度门控，局部关系矩阵承担主要事实更新。

这符合“局部强更新、全局弱更新、保留粘连、关系冲突时再扩容”的设计目标。

### 5. 已完成的 v2 对照结果

实验目录：
`outputs/bert-mlm-fact-memory-shiji-local-relation-margin-v2-256`。

已记录的结果如下：

- 事实测试集训练前：Top-1 `0%`，Top-5 `0%`，loss `7.48216`；
- 事实测试集训练后：Top-1 `37.0%`，Top-5 `44.4%`，loss `5.23366`；
- hard-negative 平均 margin：训练前 `-5.1716`，训练后 `+0.3477`；
- 最终事实记忆 loss：`2.82981`；
- 最终遗忘率：`3.46%`；
- 局部关系贡献平均绝对值约 `2.975`，全局更新贡献平均绝对值约 `1.433`。

手工检查的两个上下文例子：

- `汉王 立 [MASK] 为 赵王 。`：目标“张耳”排名第一；
- `汉王 立 张耳 为 [MASK] 。`：目标“赵王”排名第一，且与“张耳”拉开明显
  margin。

上述两个句子属于训练数据中的事实样本，因此只能证明记忆路径和局部关系分数
确实在工作，不能称为盲测泛化结果。后续必须增加按篇章留出的测试集。

## 三、关于 AdamW 的当前决定

AdamW 是优化器，不是 Transformer 层。它的自适应一阶/二阶矩会对梯度做归一化，
因此代码中把全局梯度乘以 `0.02` 后，实际参数更新不一定仍然保持 2% 的比例；
这会削弱“局部强、全局弱”的可解释性。

因此当前事实记忆和局部关系实验暂时不使用 AdamW：

- 新增入口：
  `examples/train_shiji_fact_memory_local_relation_margin_sgd.py`；
- 使用 SGD + momentum `0.9` + Nesterov；
- 局部关系和全局路径使用相同原始学习率 `5e-4`；
- 全局路径经过 `0.02` 梯度门控，所以有效全局更新约为局部更新的 2%；
- 旧的 AdamW 入口和已有实验结果保留，未来可以做 A/B 对照；
- 本次只完成入口和检查，没有启动 SGD 训练。

## 四、接下来要做什么

### P0：完成 SGD 与 AdamW 的公平对照

使用相同的初始 checkpoint、数据划分、随机种子、epoch、事实样本和 margin
参数，分别运行 SGD 与旧 AdamW 版本，比较：

- 事实 Top-1 / Top-5；
- 事实 loss；
- 全词表平均 margin；
- hard-negative 平均 margin；
- 通用 MLM 记忆损失；
- 遗忘率；
- 全局路径和局部关系路径的实际贡献；
- 关系矩阵数量、维度使用率和关系映射复用率。

重点观察 SGD 是否让 `0.02` 的全局门控真正体现为小幅更新，以及局部矩阵是否
能够保持足够的跨关系粘连。

### P1：改成严格的留出评估

- 按篇章而不是随机句子划分 train/dev/test；
- 测试未参与事实记忆训练的问法和长上下文；
- 分别测试“汉王 → 张耳”“汉王立张耳为赵王”等直接事实与改写句式；
- 报告训练集记忆、同篇章泛化、跨篇章泛化三个指标，避免把背诵当推理。

### P2：完善局部关系路由

- 为每个词维护候选关系空间映射，而不是只依赖当前激活的三元组；
- 由上下文 hidden 生成 route query，对候选局部空间做 softmax 路由；
- 保存旧事实的路由分布，并加入路由保持约束；
- 记录候选数量、Top-k 命中率、路由熵和空间使用频率，防止所有词坍缩到同一
  个局部空间；
- 继续采用宽松冲突判定，只在连续、可观测的事实下降后才增加关系容量。

### P3：把语法和属性从“先验/元信息”接入主 logits

当前语法和词语属性已经可以独立训练与评估，但需要在主 BERT 的 logits 路径中
明确接入：

```text
上下文 hidden
   ↓
语法结构 + 词语属性兼容度
   ↓
候选 logits 的软偏置
   ↓
局部事实关系分数
   ↓
最终 logits
```

接入时继续使用软偏置，不做过于激进的硬屏蔽；标点、功能词的低频先验不能
覆盖明确的语法和事实证据。

### P4：最后再进入生成式问答

当 BERT 的上下文、语法、属性和事实记忆评估稳定后，再在其上增加因果解码器：

```text
问题 + 史料 → BERT 编码器 → 上下文向量
                         ↓ Cross-Attention
                  因果解码器逐词生成答案
```

生成式阶段使用问题—答案数据、teacher forcing 和 next-token loss。BERT 负责
提供上下文与关系表示，解码器负责多句答案的顺序生成。

## 五、当前复现入口

只做语法检查/导入检查：

```powershell
C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile examples\train_shiji_fact_memory_local_relation_margin_sgd.py
C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -c "import sys; sys.path.insert(0, 'examples'); import train_shiji_fact_memory_local_relation_margin_sgd"
```

下一轮正式训练入口：

```powershell
C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe examples\train_shiji_fact_memory_local_relation_margin_sgd.py
```

正式训练前应确认输出目录不存在，避免误覆盖已有实验结果。
