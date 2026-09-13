import os
import sys
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# add project root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bert_simple import BertConfig, BertForMaskedLM
from bert_simple.tokenizer import SimpleBertTokenizer

def mlm_collate(batch: List[Dict[str, List[int]]], tokenizer: SimpleBertTokenizer, mlm_probability: float = 0.15):
    # 将 batch 中的 input_ids 转换为 PyTorch 张量，也就是把多条数据堆叠起来
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    # 将 batch 中的 token_type_ids (区分句子A/B) 转换为张量
    token_type_ids = torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long)
    # 将 batch 中的 attention_mask (标记哪些是填充的pad) 转换为张量
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    # 将 batch 中的 special_tokens_mask (标记哪些是特殊字符如[CLS],[SEP]) 转换为布尔张量
    special_tokens_mask = torch.tensor([b["special_tokens_mask"] for b in batch], dtype=torch.bool)

    # 复制一份原始输入作为标签(Label)，这就是我们要预测的"标准答案"
    labels = input_ids.clone()
    # 确定哪些位置可以被遮掩(Mask)：不能是特殊字符([CLS]等)，也不能是填充字符(pad)
    maskable = (~special_tokens_mask) & (attention_mask == 1)
    # 生成一个和输入形状一样的随机数矩阵，用于决定哪些位置被选中
    rand = torch.rand(input_ids.shape)
    # 最终确定被遮掩的位置：必须是可遮掩的，且随机概率小于15%(mlm_probability)
    mask_positions = maskable & (rand < mlm_probability)

    # 将所有"没被选中"的位置的标签设为 -100，PyTorch计算Loss时会自动忽略 -100
    labels[~mask_positions] = -100

    # 再次生成随机数，用于决定这15%被选中的位置具体怎么个变法 (80-10-10策略)
    probs = torch.rand(input_ids.shape)
    # 80% 的概率：真的替换成 [MASK] 符号
    mask80 = mask_positions & (probs < 0.8)
    # 10% 的概率：替换成词表里随机的一个词 (用于干扰模型)
    rand10 = mask_positions & (probs >= 0.8) & (probs < 0.9)
    # 10% 的概率：保持原样不动 (让模型学会相信输入)
    keep10 = mask_positions & (probs >= 0.9)

    # 复制一份输入id，准备进行修改，生成最终喂给模型的 masked_input_ids
    masked_input_ids = input_ids.clone()
    # 将属于 80% 那部分的位置，替换为 [MASK] 的 ID
    masked_input_ids[mask80] = tokenizer.mask_token_id

    # 获取词表大小
    vocab_size = len(tokenizer)
    # 生成一个全是随机词ID的矩阵
    random_words = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=torch.long)
    # 将属于 10% 随机替换那部分的位置，填入随机生成的词
    masked_input_ids[rand10] = random_words[rand10]
    # 剩下的 10% (keep10) 不做任何操作，保留原词
    # keep10: do nothing

    # 返回处理好的结果：带掩码的输入、句子类型ID、注意力掩码、以及计算Loss用的标签
    return masked_input_ids, token_type_ids, attention_mask, labels


if __name__ == "__main__":
    # ==========================================
    # 以下为构造测试数据供调试 mlm_collate 使用的代码
    # ==========================================
    print("--- 1. 正在初始化并训练测试用 Tokenizer ---")
    # 造两条简单的文本用来训练词表，保证词表里有字
    dummy_texts = [
        "你好，这是一个专门用来调试的句子。",
        "机器学习和深度学习非常有趣，但是细节很多。"
    ]
    
    tokenizer = SimpleBertTokenizer()
    tokenizer.train_from_texts(dummy_texts, min_freq=1)
    
    print(f"词表大小: {len(tokenizer)}")
    print(f"[MASK] token id: {tokenizer.mask_token_id}")
    print(f"[PAD] token id: {tokenizer.pad_token_id}")

    print("\n--- 2. 正在构造 Batch 数据 ---")
    # 自动计算最长句子的长度（加上头尾2个特殊字符的坑位）
    max_len = max(len(t) for t in dummy_texts) + 2
    encoded_texts = [
        tokenizer.encode(
            text=t,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            # 设置 padding=True 会自动用 pad_token_id 补齐到相同长度，这样才能堆叠成 batch 张量
            padding=True, 
            return_special_tokens_mask=True
        )
        for t in dummy_texts
    ]
    
    # 组装成 List[Dict[str, List[int]]] 的形式
    batch = []
    for e in encoded_texts:
        item = {
            "input_ids": e["input_ids"],
            "token_type_ids": e["token_type_ids"],
            "attention_mask": e["attention_mask"],
            # 防止有些 tokenizer 实现没返回 special_tokens_mask 的兼容处理
            "special_tokens_mask": e.get("special_tokens_mask", [0] * len(e["input_ids"]))
        }
        batch.append(item)
        
    print(f"构造好了 {len(batch)} 条训练数据。")
    print(f"序列长度 (max_length): {len(batch[0]['input_ids'])}")

    print("\n--- 3. 准备调用 mlm_collate (请在此处打断点) ---")
    # 💡 提示：可以在下面这行打个断点，如果是 VSCode / PyCharm 则直接 F5 / Debug 运行当前文件即可进入调试
    masked_input_ids, token_type_ids, attention_mask, labels = mlm_collate(
        batch=batch, 
        tokenizer=tokenizer, 
        mlm_probability=0.15
    )
    
    print("\n--- 4. mlm_collate 执行完成，结果如下 ---")
    print(f">> masked_input_ids (shape: {masked_input_ids.shape}):\n{masked_input_ids}")
    print(f">> labels (shape: {labels.shape}):\n{labels}")
    print("\n调试完成，你可以通过断点观察 mask_positions、rand、mask80 等中间变量是如何生成的了！")
