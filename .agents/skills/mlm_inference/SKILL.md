---
name: mlm_inference
description: 演示如何使用 simple-bert 组件进行掩码语言模型（MLM）推理。
---

# MLM 推理 (MLM Inference)

该 Skill 记录了本项目中使用 `BertForMaskedLM` 和 `SimpleBertTokenizer` 进行掩码语言模型（MLM）推理的标准流程。当你需要实现 MLM 预测功能时，可以直接遵循此规范，无需重复查阅代码或提醒。

## 工作流程

1. **加载模型和分词器 (Tokenizer)**
   - 确保推理流程在 `@torch.no_grad()` 装饰器下运行，以避免不必要的梯度计算和显存占用。
   - 从预训练目录或已保存的路径中加载 `BertForMaskedLM` 模型和 `SimpleBertTokenizer`。
   - 务必将模型设置为评估模式 (`model.eval()`)。

2. **对输入文本进行编码 (Encoding)**
   - 使用 `tokenizer.encode` 将包含 `[MASK]` 标记的文本字符串转换为模型可读取的输入。
   - 注意指定参数 `return_tensors="pt"` 以直接获取 PyTorch 的张量格式。
   - 将生成的张量字典 (`enc`) 转移到对应的计算设备 (如 `cuda` 或 `cpu`)。

3. **模型前向传播 (Forward Pass)**
   - 以关键字参数 (kwargs) 形式将输入数据传入模型：`outputs = model(**enc)`。
   - 从输出结果中提取原始预测得分 `logits`（若模型输出为 Tuple，得分通常是第一个元素）。

4. **提取与解码预测结果**
   - 通过比对输入序列中与 `tokenizer.mask_token_id` 相等的 ID，定位 `[MASK]` 标记所在的精确位置。
   - 使用获得的 batch 和序列位置索引，从 `logits` 中取出这些位置的对应得分。
   - 通过 `torch.topk` 获取得分最高的若干个预测候选 token (例如 Top-5)。
   - 最后利用 `tokenizer.decode` 将预测的整数 ID 转换回人类可读的字符串。

## 代码示例

以下代码展示了与 `simple_tokenizer_mlm.py` 中一致的标准 MLM 推理函数实现：

```python
import torch
from bert_simple import BertForMaskedLM
from bert_simple.tokenizer import SimpleBertTokenizer

@torch.no_grad()
def demo_inference(model_dir: str, text: str):
    """
    运行 MLM 推理演示
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()
    tokenizer = SimpleBertTokenizer.from_pretrained(model_dir)

    # 1. 文本编码 - 确保输入包含 "[MASK]" 标记
    # 例如：text = "改立沛公为汉[MASK]，统治巴蜀、汉中之地，建都南郑"
    enc = tokenizer.encode(
        text, 
        add_special_tokens=True, 
        max_length=128, 
        truncation=True, 
        padding=True, 
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # 2. 前向传播
    outputs = model(**enc)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs

    # 3. 预测被遮蔽 (MASK) 的词
    mask_positions = (enc["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    for b, pos in mask_positions:
        # 获取当前 MASK 位置的预测分数
        scores = logits[b, pos]
        # 取出排名前 5 的词的索引和分数
        topk = torch.topk(scores, k=5)
        tokens = topk.indices.tolist()
        values = topk.values.tolist()
        
        # 4. 解码输出
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        rounded_values = [round(v, 2) for v in values]
        print("Top-5 预测词:", decoded_tokens, "\n预测得分:", rounded_values)
```
