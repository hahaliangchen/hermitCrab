---
name: mlm_training_analysis
description: 深度分析 bert-simple 项目中 BertForMaskedLM 的完整运行机制，涵盖词向量生成、注意力机制（QKV/多头/Mask）、Dropout、Pooler、Loss计算到反向传播的完整链路。每个机制均采用"理论 + 代码 + 模拟结果"三段式结构记录。
---

# bert-simple 源码深度分析手册

> **文档约定**：每个机制均按"**理论** → **代码** → **模拟结果**"三段式展开，确保可以直接定位代码位置、理解物理含义，并直观感受张量在运行时的形态变化。

---

## 1. 词向量（Word Embeddings）的本质

### 理论
**一个词向量包含的信息，并非只有它本身，而是它曾经所在的所有上下文杂糅汇聚在一起的信息。** 每次这个词出现在训练语料中，梯度反传就会对它在词表里对应的那一行数值做一次微小的调整，最终这个向量成为该词在所有历史语境中信息的"语义质心"。

静态词向量（查表拿出的初始值）是起点，是"平均脸"；经过 Transformer 的上下文融合后输出的动态向量，才是"当前这句话的精确语义"。

### 代码
`bert_simple/model.py` → `BertEmbeddings.forward`：
```python
inputs_embeds   = self.word_embeddings(input_ids)     # 按 Token ID 查表取行
token_type_embeds = self.token_type_embeddings(token_type_ids)
position_embeds = self.position_embeddings(position_ids)

# 三种向量直接相加，作为进入 Transformer 的初始词向量
embeddings = inputs_embeds + token_type_embeds + position_embeds
embeddings = self.LayerNorm(embeddings)
embeddings = self.dropout(embeddings)
```

### 模拟结果
```
输入：input_ids = [[101, 6321, 5922, 4374, 102, 0, 0]]
        （即 [CLS, 楚, 霸, 王, SEP, PAD, PAD]，batch=1, seq=7）

word_embeddings(input_ids) → shape: [1, 7, 256]
  └─ 第 0 行（[CLS]）: [-0.02,  0.31, ...,  0.17]   ← 随机初始化的 256 维向量
  └─ 第 1 行（楚）:    [ 0.55,  0.23, ..., -0.11]   ← 被历史语料雕刻过的向量
  └─ 第 5 行（PAD）:   [ 0.00,  0.00, ...,  0.00]   ← padding_idx=0，强制全零

三种向量相加后 embeddings → shape: [1, 7, 256]  （shape 不变，数值叠加）
```

---

## 2. Q、K、V 矩阵：需求者 / 特征标签 / 信息实体

### 理论
Self-Attention 的哲学可以用三个角色来理解：
- **Q（Query/寻找者）**：当前词主动提问——"我需要什么样的上下文来补全我的语义？"
- **K（Key/被寻找者）**：每个词挂出自己的"特征名牌"——"我在什么情况下会被别人需要？"
- **V（Value/贡献者）**：实际被搬运走的信息实体——"匹配上之后，你真正拿走的是这部分内容"

**Q·K 的点积**= 两个词之间的"需求与供给契合度"，分数越高，注意力越强。训练 Q 是让模型学会"应该关注什么"，训练 K 是让模型学会"应该被谁关注"。

### 代码
`bert_simple/model.py` → `MultiHeadSelfAttention`:
```python
# 用一个宽度为 3 倍的全连接层，一次性生成 Q、K、V
self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)

def forward(self, hidden_states, attention_mask=None):
    qkv = self.qkv(hidden_states)          # 矩阵乘法，hidden_size → 3*hidden_size
    query, key, value = qkv.chunk(3, dim=-1)  # 沿最后一维切成三等份
```

### 模拟结果
```
hidden_states shape: [16, 128, 256]   （16句话，128个词，每词256维）

self.qkv.weight shape（存在内存里的）: [768, 256]  ← 永远是2D矩阵

nn.Linear 广播规则：只对最后一维动刀，前面的维度透传
hidden_states [16, 128, 256] × weight.T [256, 768]
→ qkv shape: [16, 128, 768]

qkv.chunk(3, dim=-1):
→ query shape: [16, 128, 256]
→ key   shape: [16, 128, 256]
→ value shape: [16, 128, 256]
```

---

## 3. 多头注意力：`_transpose_for_scores` 的维度变形

### 理论
`hidden_size` 被均匀切分给多个"注意力头"，每个头负责在自己的特征子空间里独立观察上下文，捕捉不同类型的语义关系（如语法结构、情感极性、指代关系等）。

**关键约束**：`hidden_size` 必须能被 `num_attention_heads` 整除。
```python
head_dim = hidden_size // num_attention_heads  # 每个头分配到的维度数
all_head_size = num_heads * head_dim           # 等于 hidden_size（完全相等）
```

**算力魔法**：把 1 个 `256维` 大矩阵分成 4 个 `64维` 小矩阵并行计算，计算量一样，但同时从 4 个视角观察语义。

### 代码
`bert_simple/model.py` → `_transpose_for_scores`:
```python
def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
    # 第一步：view 切分——把最后一维从 256 展开成 (4个头, 64维/头)
    new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
    x = x.view(*new_shape)
    # 第二步：permute 换位——把"头"的维度挪到 seq 前面，送给 GPU 并行
    x = x.permute(0, 2, 1, 3)
    return x
```

### 模拟结果
```
输入 query shape: [16, 128, 256]
  └─ 每个词身上背着一个 256 维的大总包

.view(*[16, 128, 4, 64])
→ shape: [16, 128, 4, 64]
  └─ 大包裹被均匀拆成 4 个 64 维的小包，仍绑在各词身上

.permute(0, 2, 1, 3)
→ shape: [16, 4, 128, 64]
  └─ 重组为：4 个平行世界，每个世界里有完整的 128 词的句子（每词 64 维）

此后 GPU 看到的形状：
matmul(query[16,4,128,64], key.T[16,4,64,128])
→ attn_scores shape: [16, 4, 128, 128]   ← 每个头，每个词对所有词的得分表
```

---

## 4. Attention Score 矩阵的物理含义与缩放因子

### 理论
`attn_scores` 的最后两维 `[seq_Q, seq_K]` 是一张"兴趣得分表"：
- 每一**行**：某个词（Q）对整句话所有词的原始兴趣得分
- 每一**列**：某个词（K）被所有词"看了一眼"所得到的分数

**为什么要除以 `sqrt(head_dim)`？**
随着 `head_dim` 增大，Q·K 点积的数值会线性增大（维度越多累加越多），导致某几个词的得分极高，Softmax 后注意力极度尖锐（一个词独拿 99.9%，其他接近 0），梯度也随之消失。除以 $\sqrt{d_k}$ 将数值压缩到温和范围，保持注意力分布平滑。

### 代码
`bert_simple/model.py` → `MultiHeadSelfAttention.forward`:
```python
attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
# key.transpose(-1, -2) 将 [16,4,128,64] 变为 [16,4,64,128]，才能做矩阵乘法
```

### 模拟结果
```
         [CLS]  楚     霸     王    [SEP]  [PAD]
[CLS] → [ 0.9,  0.2,  0.1,  0.3,   0.8,  0.0 ]
楚    → [ 0.1,  3.2,  2.1,  1.8,   0.2,  0.0 ]
霸    → [ 0.2,  2.8,  3.5,  2.9,   0.1,  0.0 ]
王    → [ 0.3,  1.9,  2.7,  3.1,   0.2,  0.0 ]
[PAD] → [ 0.5,  1.1,  0.8,  0.9,   0.4,  0.0 ]

未除以 sqrt(64) 前：数值可能高达 ±50，Softmax 后极度尖锐
除以 sqrt(64)=8 后：数值压缩到 ±6，Softmax 后分布更均匀
```

---

## 5. Attention Mask：精准封死 PAD 列

### 理论
**核心哲学：PAD 词可以"问"（作为 Q），但绝对不能"答"（作为 K）。**

- PAD 作为 Q 去乱问其他词：它得到的输出会被 `ignore_index=-100` 完全豁免，不计算 Loss，不产生梯度，对网络无任何影响。
- PAD 作为 K 被其他真实词注意到：其他真实词会从无意义的 PAD 里"借"走噪音信息，污染 Loss，产生错误的梯度更新，**灾难性的**。

所以 Mask 策略是：**让 PAD 列（作为 K 的那一维）的得分变成负无穷**，过完 Softmax 后注意力为 0。

### 代码
`bert_simple/model.py` → `_build_attention_mask`：
```python
# 阶段1：DataLoader 输出的原始 mask（1=真实词，0=PAD）
# attention_mask shape: [batch, seq_len]

# 阶段2：转换为 0.0 / -10000.0 的浮点 Mask
extended = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -10000.0
# 两次 unsqueeze 后 shape: [batch, 1, 1, seq_len]
```

之后在 Attention 中被加入：
```python
if attention_mask is not None:
    attn_scores = attn_scores + attention_mask  # attention_mask 此时已是浮点 Mask
```

### 模拟结果
```
原始 attention_mask（来自 DataLoader）:
[[1, 1, 1, 1, 1, 0, 0]]   shape: [1, 7]

1.0 - mask:
[[0, 0, 0, 0, 0, 1, 1]]

× -10000.0:
[[0., 0., 0., 0., 0., -10000., -10000.]]

两次 unsqueeze 后 shape: [1, 1, 1, 7]

广播加到 attn_scores [1, 4, 7, 7] 时：
  attn_scores: [batch,     heads, seq_Q, seq_K ]
  mask:        [batch,         1,     1, seq_K ]
                               ↑        ↑
                          广播heads  广播seq_Q
                          
效果：所有 Q（不管哪行）遇到 PAD 列（第5、6列）时，得分变成 -9995.x
经 softmax(e^{-9995}) ≈ 0，这两列的注意力权重被彻底消灭。
```

---

## 6. `model.train()` 与 `nn.Dropout` 的分工

### 理论
- **`model.train()`**：只是一个全局喇叭，递归地把所有子模块的状态位 `self.training` 改为 `True`，本身不处理任何数据。
- **`nn.Dropout`**：是真正执行随机丢弃的层。它的行为完全由 `self.training` 这个状态位决定：
  - **训练时**：随机置零（p=0.1，即 10% 维度归零），同时将剩余维度放大 `1/(1-p)` 倍，保持期望不变（能量守恒）。
  - **推理时**：像透明玻璃一样，输入什么输出什么，绝对不丢弃任何信息。

**丢弃的方向选择**：完全随机，与数值高低无关。这正是 Dropout 的核心：**不针对"一枝独秀"的神经元，而是通过随机恐吓，逼迫所有神经元都必须分担学习责任。**

### 代码
PyTorch `nn.Dropout` 内部逻辑（伪代码）：
```python
def forward(self, x):
    if self.training:  # 由 model.train() / model.eval() 控制
        # 随机生成 0/1 二进制掩码（bernoulli 分布）
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
        return x * mask / (1 - self.p)  # 置零 + 等比例放大
    else:
        return x  # 推理模式：透明玻璃，原封不动
```

在 `bert_simple/model.py` 中广泛存在：
```python
# BertEmbeddings（第52行）、MultiHeadSelfAttention（第79行）、BertAttention（第111行）、BertOutput（第135行）
self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 默认 p=0.1
```

### 模拟结果
```
输入向量 x = [1.0, 2.0, 3.0, 4.0, 5.0]   p=0.5（用 0.5 方便演示）

训练模式（model.train()）：
  随机掩码        = [  1,   0,   1,   0,   1]
  掩码后          = [1.0, 0.0, 3.0, 0.0, 5.0]
  ÷(1-0.5)放大   = [2.0, 0.0, 6.0, 0.0, 10.0]  ← 幸存者被放大，期望不变

推理模式（model.eval()）：
  输出            = [1.0, 2.0, 3.0, 4.0, 5.0]   ← 完全透明，原样输出
```

---

## 7. 特征纠缠：维度不可解释性的根源

### 理论
**Dropout 是特征纠缠的元凶之一，但即使关掉 Dropout，Transformer 自身机制才是根本原因。**

三大根本原因：
1. **维度挤压（Superposition）**：`hidden_size=256` 的空间要装下成千上万种语言特征。唯一出路是让多个无关属性在同一个维度上叠加共用，导致任何单一维度都无法单独解读。
2. **稠密全连接矩阵（Dense Linear）**：`nn.Linear` 默认满阵，没有稀疏约束，所有输入维度参与所有输出维度的计算，梯度只管降 Loss，不管特征是否正交独立。
3. **FFN 的放大-压缩循环**：先 4 倍放大（`256→1024`）再压缩回（`1024→256`），这种暴力压缩注定让每个维度都要同时背负多重信息包袱。

### 代码
`bert_simple/model.py` → `BertIntermediate` + `BertOutput`：
```python
# 放大：解开纠缠的"宽敞工作室"
self.dense = nn.Linear(config.hidden_size, config.intermediate_size)  # 256 → 1024
self.act   = nn.GELU()

# 压缩：暴力打包回原维度，信息包袱加重
self.dense = nn.Linear(config.intermediate_size, config.hidden_size)   # 1024 → 256
```

### 模拟结果（概念示意）
```
词向量第 42 维的数值 = 0.65

问：这个 0.65 代表"动词属性"还是"正向情感"？
答：不知道。

它实际上是以下属性的线性叠加：
  0.65 ≈ 0.3 × (动词特征)
        + 0.2 × (正向情感特征)
        + 0.1 × (专有名词特征)
        + 0.05 × (否定语气特征)
        + ...（数十种微弱信号的混合）

这正是学术界（Anthropic 等）开发
稀疏自编码器（SAE, Sparse Autoencoder）的动机：
把 768 维里纠缠的特征展开到 10 万维以上，强迫稀疏化，
才能看清楚"原来这 768 维里藏着哪些可解释的特征"。
```

---

## 8. `BertPooler`：句子级摘要提取器

### 理论
**`[CLS]` Token 不是临时生成的，它是词表里货真价实的一员（ID=2），有自己的 Embedding 向量行，随训练不断更新。**

- **静态 `[CLS]`（词表里的那一行）**：经过大量句子训练后，进化出了"最高效地吸收全句信息"的元能力——训练出的"面试官套路"，不存储任何具体句子内容。
- **动态 `[CLS]`（Transformer 输出的那一个向量）**：走完全部 Transformer 层之后，这个位置已经把当前这句话所有词的信息融为一炉——这才是"当前句子的全文摘要"。

### 代码
`bert_simple/model.py` → `BertPooler.forward`：
```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    first_token = hidden_states[:, 0]   # 只取第 0 位 [CLS] 的输出向量
    pooled = self.dense(first_token)    # 线性变换，为分类任务重组特征
    pooled = self.activation(pooled)    # Tanh 激活，压缩到 [-1, 1]
    return pooled
```

### 模拟结果
```
句子 A："[CLS] 项羽 自立 为 西楚 霸王 [SEP]"
句子 B："[CLS] 汉王 也 前往 封国 [SEP]"

两句输入同一个 [CLS] 初始向量（词表第 2 行），同样的 Transformer 参数

句子 A 输出的 pooled_output[0] ≈ 偏向"楚霸王/统治/建立"相关语义方向
句子 B 输出的 pooled_output[0] ≈ 偏向"汉王/封地/行进"相关语义方向

用途：
  seq_output  → 每个词各自的向量 → 用于填空(MLM)/命名实体识别(NER) 等词级任务
  pooled_output → 整句的摘要向量 → 用于情感分析/文本分类等句级任务
```

---

## 9. `BertLayer`：完整的 Transformer Block

### 理论
`self.layer` 里的每一个元素不是单纯的"自注意力机制"，而是一个完整的 **Transformer Block**，包含四个精密配合的组件：

1. **`BertAttention`（大熔炉）**：Self-Attention + Add & Norm，唯一让词与词之间发生信息交流的地方。
2. **`BertIntermediate`（放大阶段）**：FFN 前半段，`hidden_size → 4×hidden_size`，过 GELU 激活，暂时解开特征纠缠。
3. **`BertOutput`（压缩阶段）**：FFN 后半段，`4×hidden_size → hidden_size`，压缩回原维度 + Add & Norm。
4. **残差连接（幕后英雄）**：`output = dropout(sublayer(x)) + x`，保证梯度通路畅通，模型才能堆叠 12 层不崩。

### 代码
`bert_simple/model.py` → `BertLayer.forward`：
```python
def forward(self, hidden_states, attention_mask=None, output_attentions=False):
    # ① 自注意力 + 残差 + LayerNorm
    attn_output, attn_probs = self.attention(hidden_states, attention_mask, output_attentions)
    # ② 前馈网络（放大 → 激活 → 压缩）+ 残差 + LayerNorm
    inter = self.intermediate(attn_output)
    layer_output = self.output(inter, attn_output)
    return layer_output, attn_probs
```

### 模拟结果
```
hidden_states 进入 BertLayer:  [16, 128, 256]

① BertAttention:
   → 每个词"看"了所有词，吸收了上下文信息
   → 输出 shape: [16, 128, 256]（shape 不变，但数值已融合上下文）

② BertIntermediate:
   → Linear(256→1024) + GELU
   → 输出 shape: [16, 128, 1024]  ← 维度放大4倍，特征暂时解纠缠

③ BertOutput:
   → Linear(1024→256) + dropout + 残差 + LayerNorm
   → 输出 shape: [16, 128, 256]  ← 回到原维度

经历 1 层后：每个词的向量已经"见过"了它周围所有词
经历 4 层后：词向量已经是多轮深度上下文融合的结果
```

---

## 10. Loss 计算到权重更新的完整链路

### 理论
一次完整的训练迭代（Step）包含五个物理动作，理解每一步的"为什么"是理解深度学习训练机制的关键：
- **Forward**：算出"当前有多差"（Loss）
- **zero_grad**：清除上一步的历史污染
- **Backward**：算出"每个参数应该往哪个方向改"（梯度）
- **step**：真正执行改动（更新权重）

### 代码
`examples/simple_tokenizer_mlm.py` 第 157-165 行：
```python
# ① Forward：残缺句子 → 模型 → Loss（标量）
loss, _ = model(
    input_ids=input_ids,        # 含 [MASK] 的残缺版句子
    token_type_ids=token_type_ids,
    attention_mask=attention_mask,
    labels=labels,              # 含 -100 的完整答案标签
)

# ② 清空上一步残留的梯度（set_to_none 更彻底，省显存）
optimizer.zero_grad(set_to_none=True)

# ③ Backward：Autograd 沿计算图反推每一层的梯度
loss.backward()

# ④ 更新权重：AdamW 用自适应学习率将梯度应用到全部参数
optimizer.step()
```

MLM Loss 的计算位置（`model.py` → `BertForMaskedLM.forward`）：
```python
logits = self.lm_head(sequence_output)  # 映射回词表维度
loss = F.cross_entropy(
    logits.view(-1, self.config.vocab_size),
    labels.view(-1),
    ignore_index=-100   # 非 Mask 位置（标签为 -100）完全不计入 Loss
)
```

### 模拟结果
```
MLM 数据构造（mlm_collate 函数）：

原句：[CLS] 项羽 自立 为 西楚 霸王 [SEP]
标签：[ -100, -100, -100, -100,  "霸",  -100, -100]
输入：[CLS]  项羽  自立  为   [MASK]  霸王  [SEP]  ← 80% 概率替换为[MASK]

Forward 输出：
  logits shape: [1, 7, vocab_size]  （每个位置对应词表的概率分布）
  只有第 4 位（霸）的 logits 参与 cross_entropy 计算
  loss = -log(P(霸 | 上下文)) 

梯度流向（backward）：
  loss
   ↓ lm_head.weight（= word_embeddings.weight，同一块内存）
   ↓ BertLayer×4（从第4层反推到第1层）
       → out_proj, qkv, FFN dense, LayerNorm 参数
   ↓ word_embeddings.weight（被两个方向的梯度同时累加更新）

optimizer.step() 后：
  全部 Transformer 层参数 + 词嵌入矩阵 均被微调
  每个词向量在"见到更多语境"后，语义方向更加精确
```

---

## 11. 词表与 Tokenizer 的配合机制

### 理论
**词表大小必须在训练循环开始之前确定好**，之后不能变，否则 `nn.Embedding` 矩阵尺寸与优化器状态不匹配，会引发崩溃。

- **标准 BERT 做法**：预先整理好 vocab.txt（约 21000 个汉字），遇到未知词变 `[UNK]`，信息彻底丢失。
- **本项目做法**：通过 `train_from_texts` 扫描全部语料后建立词表，所有出现过的词都能进词表，模拟"从内容里习字"的自然语言学习路径。

### 代码
`bert_simple/tokenizer.py`：
```python
# 训练前一次性扫描全部语料建立词表
def train_from_texts(self, texts: List[str], min_freq: int = 1):
    counter = Counter()
    for text in texts:
        counter.update(self.tokenize(text))
    for tok, freq in counter.items():
        if freq >= min_freq:
            self._add_token(tok)  # 只加进词表，不初始化向量

# encode 时的 add_new_tokens 参数（推理时动态扩充，需配合 resize_token_embeddings 使用）
def encode(self, text, add_new_tokens: bool = False, ...):
    tokens = self.tokenize(text)
    if add_new_tokens:
        self.add_tokens(tokens)
    ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
```

### 模拟结果
```
时间线：

① tokenizer.train_from_texts(texts)
   扫描语料 → 发现 3487 个独特词（含特殊 Token）
   词表建立完成，size=3487

② model = BertForMaskedLM(BertConfig(vocab_size=3487))
   nn.Embedding(3487, 256) 建好，所有行随机初始化

③ optimizer = AdamW(model.parameters(), lr=5e-4)
   优化器接管全部参数，开始记录动量状态

④ 训练 400 steps 后：
   词表 size 没有任何变化（仍然是 3487）
   但每一行的向量数值已经被梯度雕刻得面目全非，从随机混沌走向有意义的语义分布
```

---

*文档最后更新：2026-04-02*
*来源项目：`bert-simple`*
*核心文件：`bert_simple/model.py`、`bert_simple/tokenizer.py`、`examples/simple_tokenizer_mlm.py`*
