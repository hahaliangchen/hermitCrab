# 18D 关系 FFN 实现与验证

## 模型路径

`bert_simple/structured_relation.py` 实现 `Structured3DRelationFFN`。
输入 `[q,k,abs(q-k),flatten(q outer k)]`，18→32→GELU→1（即 Linear(18,32)、
GELU、Linear(32,1)），含 bias 共 641 个参数。共享于所有动态层、head、space；
最后一层零初始化，初始输出为 0，首次梯度先更新输出层，随后传播到隐藏层。

对每个共享空间 s，使用该空间的三维 Q/K，定义注意力中的局部偏置：

```text
R_ij = ffn_scale × Σ_s route_i,s × route_j,s × FFN18(q_i,s, k_j,s)
attention_logits = ordinary_logits + relative_bias
                 + dynamic_qk_scale × (dynamic_dot_scores + R)
                 + padding_mask
```

默认 `ffn_scale=0.1`、`dynamic_qk_scale=0.5`。保留普通 V、原 Transformer FFN
以及 MLM 输出侧事实 adapter。三个组件有不同作用，本次未用关系 FFN 替代全部 FFN。
q/k 来自当前层的上下文表示，但其坐标不预设“人物轴”“动作轴”等人类语义。

这里要区分“注意力的 pairwise logit”和“关系训练的监督单位”。注意力本身仍然
需要计算 query/key 对的分数；但关系 sidecar 训练不把每个 token 对当成一条事实。
`forward_context_groups()` 会把一个 MASK 对应的完整关系上下文组的所有成员先做
route-weighted pooling，再交给 18D FFN，并与完整背景组比较：

```text
C_group,s = Pool({ k_j,s : j 属于该关系上下文组 })
R_group = FFN18(q_MASK,s, C_group,s)
loss = softplus(margin - R_positive_group + R_background_group)
```

因此“汉王 封 张耳 为 [MASK]”作为一个组合上下文进入 FFN；不会产生
`汉王 || 封`、`封 || 张耳`、`是 || 的` 这类事实空间。角色 span 只提供训练时的
弱监督成员集合，推理时不读取答案、关系 ID 或 gold span。

新 FFN 参数归属 `relation_adapter.dynamic_qk.structured_scores`，随 adapter 参数组
接受新样本及 replay 梯度。它是共享参数，不套用单关系矩阵的行掩码，因此不能声称
各关系完全隔离或零遗忘。旧配置无 FFN 字段时自动关闭，保存恢复记录三项 FFN 配置。

## 冻结主 BERT 的第一阶段训练

`examples/train_frozen_relation_filter.py` 用已经训练好的 256 维主 BERT 产生隐藏状态，
然后只训练关系 sidecar：上下文路由、动态 3×3 Q/K 和特殊关系 FFN。主模型的通用
FFN 保持原始 `256 → 1024 → 256`，embedding、attention、V、MLM head 均不更新。
角色 span 是弱监督，要求 MASK 对完整标注角色组的关系分数高于完整背景组；没有可用负例的极短
样本会跳过。输出保存为 `relation_filter.pt`，并在配置中记录基础模型维度和开发集最佳轮次。

本阶段训练结果位于 `outputs/frozen-relation-filter-stage1`；语法/词属性两个 sidecar
分别位于 `outputs/independent-grammar-filter-stage1` 和
`outputs/grammar-attribute-filter-stage1`。特殊 FFN 的训练集分数明显高于开发/测试集，
因此只保留开发集最佳轮次，正式接回主 BERT 前仍需扩充结构训练数据并做多种子验证。

完整 bank 的训练入口是 `examples/train_full_relation_filter_stages.py`。它不再读取旧的
14,501 条 token-pair map，也不再为每个词对分配空间；它只为监督所需的 MASK—关系组/
背景组计算训练图。
`Structured3DRelationFFN.forward_pairwise()` 同时按 query/key block 计算，默认 block 为
`32 × 32`；候选空间路由按 512 个空间分块，动态 Q/K 的空间维度按 128 个空间分块。
这里的 `expand` 只发生在受限 block 内并且是 view，不是 `repeat`；3×3 外积使用
`torch.matmul`。256 维主 hidden 使用固定共享的 1500 个三维候选，其中前 86 个
基础通道覆盖全部 hidden 维度；所有 token 位置共享这套 bank，每个位置由自己的
contextual hidden 得到不同 route。这样既不丢掉 256 维，也不把空间数量绑定到词对。

完整训练器默认使用 1500 个关系候选、`--memory-budget-mb 1024`，按进程 RSS 检查并记录峰值；隐藏状态缓存和
完整 bank 初始化使用 `torch.no_grad()`，缓存完成后释放冻结的主 BERT。超过预算会抛出
带阶段和样本位置的错误，不继续训练。该保护不能替代 block 上限，所以正式运行前仍要
先做最坏样本探针。当前 256 维、1500 个共享候选、最长 128 token 的真实草稿探针，
三阶段各一步峰值约 384 MiB，低于 1 GiB 限制。

## 找不同与去成分

`bert_simple/relation_correction.py` 提供：

```python
before = capture_relation_snapshot(model, **probe_inputs)
# 正常计算训练 loss、backward、optimizer.step()
report = analyze_relation_correction(
    model, before, target_id=answer_id, top_k=2, **probe_inputs
)
```

`probe_inputs` 仅包含 input_ids、attention_mask、token_type_ids、relation_triples、
candidate_space_mask 和两个先验开关中需要的字段。batch=1，恰好一个 MASK。
两次 probe 的输入、候选必须相同；自动切换 eval/no_grad 并恢复各模块原 training 状态。
快照保存在 CPU，含每层 R 与 attention；目前保存完整矩阵，适合短句抽样，长句需要
进一步裁剪。R 的单位是乘公共 dynamic_qk_scale 之前的 FFN 偏置。

筛选考虑 `abs(ΔR)` 和 `abs(ΔAttention)`，同时纳入增强和减弱的关系。选取不同 key
位置的两条或三条 `(layer, head, key_position)` 边，query 固定为 MASK。排除 padding
和特殊 token。它是启发式候选筛选，不保证覆盖早期层的间接路径或高阶组合。

回退通过作用域内 hook，只替换选定 R 元素为旧值，保留新模型参数，重新计算 softmax
及后续网络。函数退出时移除 hook。支持单条及所有候选对的联合回退。
2 候选：3 次干预前向；3 候选：6 次干预前向；另需前后各一次完整 probe。
本次尚不做三条一起回退；二阶探测不能排除三阶或更高阶效应。

答案指标为正确 token 与“更新前最强错误 token”的 logit margin，所有回退固定
同一个负例。记录单条下降、联合下降及联合下降减两条下降之和。不要把非加性直接
等同于 AND 门，更不能把 attention 最大的词直接当作原因。

探测只检验 FFN 通道变化，未撤回普通 Q/K、V、输出事实 adapter 等路径的更新。
无效回退不代表词不重要；有效回退也不证明世界中的因果或全局唯一解释。
记录包含 corrected 标志；只有前错后对的记录才能称作纠错经验。

## 启动与数据接口

新 JSONL 已接入 `examples/train_relation_pairs.py`，规范升级为 v2，详见
[改进说明](RELATION_TRAINING_V2_IMPROVEMENTS.md)。正式新数据应使用这个入口。

需要安装 requirements.txt 中的 PyTorch。以下旧 manifest 命令仅保留作历史对照：

```bash
python examples/train_shiji_structured_relation.py \
  --fact-dataset data/shiji/manifests/fact_memory_dataset.json \
  --init-checkpoint outputs/bert-mlm-fact-memory-shiji-grammar-attribute-256/before_fact \
  --output-dir outputs/structured-relation-new-run \
  --fact-epochs 1 --correction-probe-every 100
```

checkpoint 词表必须与旧 manifest 一致，输出目录不能已有模型。
该旧流程使用真实答案构造关系候选，指标不能作为无答案提示的泛化证据。
`correction-probe-every=0` 关闭探测。抽样步的结果写入 `training_log.jsonl` 的
`correction_probe`；非错→对仅记录前后结果，跳过干预前向。

外部 AI 按新规范生成的 JSONL 先使用下面命令校验：

```bash
python examples/validate_relation_training_data.py generated.jsonl
python examples/test_structured_relation.py
```

三阶段 sidecar 的正式入口示例：

```bash
python examples/train_full_relation_filter_stages.py \
  --base-checkpoint outputs/bert-mlm-fact-memory-shiji-grammar-attribute-256/before_fact \
  --dataset data/shiji/manifests/relation_training_v2_draft.jsonl \
  --output outputs/context-relation-filter-full256-stage1 \
  --memory-budget-mb 1024
```

新 JSONL **不能直接传给旧 manifest 训练入口**；用 `train_relation_pairs.py` 直接消费。
成对消费器和一致性损失已经接入；纠错经验辅助损失仍未接入，记录回退结果不等于
模型已学会利用回退证据。效果需要在数据到位后做多种子、留出对照实验。

## 本次验证结果

在临时 Python 3.12 / PyTorch 2.14 CPU 环境运行
`python examples/test_structured_relation.py`，14 项测试通过：18D 构造、分块输出/梯度等价、
零初始化对照、新旧配置保存恢复、adapter 梯度更新、现有训练器 replay 更新路径、
稀疏 pair 与完整分数等价、完整关系组聚合、固定 bank 覆盖 256 维、空间/路由分块等价、
单条/联合回退及 hook 清理、无候选路径、数据示例与错误/泄漏拒绝（部分合并在同一测试）。
此外，1500 个共享候选的三阶段各一步真实草稿最坏样本探针峰值约 384 MiB，未进行
正式长程训练。
训练 CLI 帮助和 Python 编译检查通过。未运行真实语料的长程训练或效果对照。
