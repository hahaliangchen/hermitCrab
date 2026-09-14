# 独立语法与词语属性层训练指南

本文档系统说明项目中**独立语法结构与词语属性层（IndependentGrammarFilter）**的训练流程、执行命令、评估标准以及相关训练数据文件的具体存放路径。

---

## 一、 架构概述与训练定位

在本项目中，语法和词语属性层被设计为一个**完全独立于主模型**的轻量级先验过滤模块（源码：[`bert_simple/grammar_filter.py`](bert_simple/grammar_filter.py)）：

* **网络结构**：专属 16 维 Embedding，2 层专属 mini-Transformer（hidden=16, heads=2），粗分句硬注意力截断（`，。？`）。
* **表示空间**：8 维句法结构向量 $g_t$（预测 12 类结构）+ 8 维词属性向量 $h_t$（通过 Query 与 13 类属性 Key 点积）。
* **隔离设计**：训练过程**不向主模型回传任何梯度，零副作用，零灾难性遗忘**。训练完成后，以加性 Logit Bias 的形式在推理阶段净化主模型的候选词池。

---

## 二、 训练数据文件清单与位置

依据项目规范 [`GRAMMAR_DATA_PLAN.md`](GRAMMAR_DATA_PLAN.md)，所有从《史记经典故事》提取处理的数据均已归档在标准四级目录中：

| 数据级别 | 文件路径 | 格式与规模 | 说明与用途 |
| :--- | :--- | :--- | :--- |
| **1. 原始正文与页面元数据** | [`data/shiji/raw/shiji_stories_raw.txt`](data/shiji/raw/shiji_stories_raw.txt)<br>[`data/shiji/raw/shiji_pages.jsonl`](data/shiji/raw/shiji_pages.jsonl) | 纯文本 / JSONL<br>167 个正文页 | 已剔除全部页眉、页脚、版权、乱序页码与版面分栏断行，保留完整段落与章节。 |
| **2. 规范分句语料库** | [`data/shiji/sentences/shiji_sentences.txt`](data/shiji/sentences/shiji_sentences.txt) | 纯文本（一句一行）<br>**10,631 句** | 严格按 `。！？；` 切分，保留引语完整性，为分词与槽位挖掘提供句子级母本。 |
| **3. 空格分词语料** | [`data/shiji/segmented/shiji_segmented.txt`](data/shiji/segmented/shiji_segmented.txt) | 空格分隔文本<br>**10,631 句 (200,126 词元)** | 结合专有历史实体字典（人名/地名/官职）分词，完全兼容本项目的 `SimpleBertTokenizer`。 |
| **4. 语法属性标准数据集** | [`data/shiji/manifests/dataset.json`](data/shiji/manifests/dataset.json)<br>*(镜像)* [`outputs/independent-grammar-filter/shiji_dataset.json`](outputs/independent-grammar-filter/shiji_dataset.json) | JSON 格式<br>**共 12,409 条样本** | **训练脚本直接读取的核心文件**。包含 `train` (9,967 条)、`validation` (1,320 条)、`test` (1,122 条独立盲测)。 |
| **5. 数据生成脚本** | [`examples/build_shiji_grammar_data.py`](examples/build_shiji_grammar_data.py) | Python 脚本 | 用于从 PDF 重新提取、清洗、分词并构建上述全套数据集的一键脚本。 |

### 数据集样本结构示例 (`dataset.json`)：
```json
[
  "他 随从 沛公 攻取 阳武 以西 至 洛阳 等 地 ， 在 [MASK] 以北 大败 秦军 。",
  1,  // 结构ID: 1 对应 LOCATION (地点槽位)
  0   // 属性ID: 0 对应 PLACE (地点属性)
]
```

---

## 三、 训练执行命令

训练主入口脚本为 [`examples/train_independent_grammar.py`](examples/train_independent_grammar.py)。

### 1. 使用真实《史记》数据集训练（推荐）
在终端运行以下命令，使用刚刚构建的真实史记语料训练 50~100 轮：

```bash
python examples/train_independent_grammar.py \
  --dataset data/shiji/manifests/dataset.json \
  --epochs 50 \
  --output outputs/independent-grammar-filter
```

> **提示**：如果需要调整学习率或 Batch 大小，可以在脚本中微调（默认 `lr=0.004`, `batch_size=64`，CPU 上 1 轮约 15~20 秒）。

### 2. 仅做评估与报告生成（不重新训练）
当模型已训练完毕并保存在输出目录中时，可执行 `--evaluate-only`：

```bash
python examples/train_independent_grammar.py \
  --dataset data/shiji/manifests/dataset.json \
  --evaluate-only \
  --output outputs/independent-grammar-filter
```

### 3. 合成规则基准训练（对比基线）
如果不带 `--dataset` 参数，脚本将默认回退到纯人工合成的 12 组模板闭集：

```bash
python examples/train_independent_grammar.py --epochs 100
```

### 4. 行为回归自动化测试
每次修改过滤器结构后，建议运行单元测试，验证 Padding 不变性、梯度绝对隔离、UNKNOWN 中性放行与歧义槽位规则保护：

```bash
python examples/test_independent_grammar.py
```

---

## 四、 命令行参数详解

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--dataset` | `None` | 训练数据集路径。指定为 `data/shiji/manifests/dataset.json` 即使用真实史记语料；留空则使用内置合成模板。 |
| `--epochs` | `100` | 训练轮数。使用真实语料时，通常 30~50 轮即可达到 90%+ 的验证集准确率。 |
| `--output` | `outputs/independent-grammar-filter` | 模型权重、分词词表、训练日志及评估报告的保存目录。 |
| `--evaluate-only` | `False` | 若添加该开关，则跳过训练，直接加载现有模型并在测试集上计算全套指标。 |
| `--main-model` | `outputs/bert-mlm-dynamic-word-spaces-contextual` | （可选）主模型检查点目录。脚本会自动校验主模型参数是否保持未改动，并测试事实补词排名变化。若不存在会自动跳过或寻找备选。 |

---

## 五、 训练产物与检查指标

训练完成后，会在 `--output` 目录下生成以下 4 个核心文件：

1. **`grammar_filter.pt`**：训练好的独立层 PyTorch 权重文件；
2. **`grammar_config.json`**：模型超参数元数据（版本、8 维结构空间、8 维属性空间、结构与属性映射列表）；
3. **`training_log.json`**：每个 epoch 的 Loss、验证集结构准确率和属性准确率曲线；
4. **`report.json`**：全量评估报告。

### 关键放行指标要求：
* **结构分类准确率 (`structure_accuracy`)**：盲测集应 $\ge 90\%$；
* **属性打标准确率 (`attribute_accuracy`)**：盲测集应 $\ge 90\%$；
* **合法功能词误压制率 (`legal_function_false_suppression_rate`)**：必须严格为 **$0.0\%$**（确保“我在呢”等合法功能词绝不被误杀）；
* **主模型未受篡改 (`main_checkpoint_unchanged`)**：必须为 `True`。

---

## 六、 推理集成示例（如何在主模型中使用）

训练好的独立过滤器通过以下代码与主 BERT 结合：

```python
import torch
from bert_simple.grammar_filter import IndependentGrammarFilter, filtered_prediction
from bert_simple.tokenizer import SimpleBertTokenizer

# 1. 加载主模型与分词器
main_tokenizer = SimpleBertTokenizer.from_pretrained("outputs/bert-mlm-dimension-combinations-zhangchen-gaozu-long-100-256")
main_model = ... # 加载你的主模型

# 2. 加载训练好的独立语法过滤器
grammar = IndependentGrammarFilter.from_pretrained("outputs/independent-grammar-filter").eval()

# 3. 联合预测
text = "这位 高祖 叫 [MASK] 。"
raw_logits, final_logits, info = filtered_prediction(
    main=main_model,
    main_tokenizer=main_tokenizer,
    grammar=grammar,
    text=text
)

# final_logits 即为叠加了语法与属性偏置后的预测结果
```
