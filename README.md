# BERT Simple (无矩阵研究版 / Matrix-Free Research Edition)

这是一个极简的、剥离了冗余代码的 BERT 实现（约300行），专为**第一性原理 (First Principles)** 理解和架构实验（如 Matrix-Free / 几何 AI）而设计。

## 核心特性 (为什么它很特别)

### 1. 极简主义 "第一性原理" 代码
*   **零臃肿**: 所有的核心逻辑 (`BertEmbeddings`, `SelfAttention`, `Encoder`) 都被浓缩在 `bert_simple/model.py` 中。
*   **可读性强**: 不同于 HuggingFace 那动辄 5000+ 行的工业级代码，这个实现是为了让单个工程师能够完全读懂、理解并进行**重构**而设计的。

### 2. "透明盒子" 架构 (为 Matrix-Free 做准备)
*   **可调试性**: 模型经过了特殊补丁，支持 `output_attentions=True`。
*   **可视化**: 你可以提取出原始的 Attention 矩阵（自注意力分数），精确地观察“谁在看谁”（详见 `visualize_heads.py`）。这对于在转向 Matrix-Free 几何方法之前，验证“硬件饱和/算力浪费”至关重要。

### 3. 动态 "历史专家" 分词器
*   **中文优先设计**: `SimpleBertTokenizer` 将汉字视为原子的逻辑单元（不需要复杂的 BPE 分词）。
*   **动态词表**: 支持 `train_from_texts` 方法，它可以扫描历史文本（如《史记》），并瞬间构建一个只包含相关字符的自定义词表。彻底解决古文生僻字的 OOV (Out of Vocabulary) 问题。

## 使用方法

### 基于自定义文本训练 (例如：史记 / 历史文本)
训练脚本 `examples/simple_tokenizer_mlm.py` 已增强，支持加载外部文件。

**命令:**
```bash
python examples/simple_tokenizer_mlm.py "D:\project\bert-simple\examples\shiji_baihua.txt" --epochs 1 --hidden-size 256
```
*   **输入**: 一个文本文件，每行一句话/一段话。
*   **流程**:
    1.  扫描文件以构建动态词表。
    2.  从头开始训练一个 BERT 掩码语言模型 (MLM)。
    3.  将模型保存到 `outputs/simple-tokenizer-mlm`。

如果不提供文件参数，会默认读取 examples/shiji_baihua.txt；如果指定路径不存在，脚本会直接报错，避免误用模拟数据。训练脚本默认使用 CPU、hidden_size=256，也可以通过命令行参数调整训练轮数、batch size 和最大步数。

### 可视化 (也就是那个 "显微镜")
要查看模型内部的“思维过程/注意力”：
```bash
python visualize_heads.py
```
这将会为一句示例文本绘制出 Attention 热力图，展示每对 Token 之间的连接强度。
