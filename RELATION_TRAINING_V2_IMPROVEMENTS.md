# 关系训练协议 v2 与成对训练接口改进说明

## 本次解决的问题

v1 有 18D 关系 FFN、纠错回退和数据校验，但新 JSONL 无法直接训练；旧事实构造器
用真实答案 token 与上下文 token 分配关系组合，测试时可能通过候选空间泄露答案。
本次新增独立的成对训练/推理流程，让输入路径仅接受可见 tokens；补全规范字段、
校验器以及成对损失。18D 模型结构保留，不把人工语义关系类型强制绑定到固定坐标轴。

## 对评审意见的处理

| 问题 | 改进结果 |
|---|---|
| JSONL 没有消费器 | 新增 train_relation_pairs.py，直接读取、校验并训练整组样本 |
| 关系/时间变化无法标注 | changed_slots 支持 relation_type、roles.*、qualifiers.*、target_slot |
| 时间和事件阶段混淆 | facts 必填 qualifiers 对象，可包含 time/place/condition/stage |
| 原文与改写不区分 | 样本必填 surface_kind；source 必须是可恢复的连续引文摘录 |
| 负例没有接入新格式 | 全词表 CE + 显式负例 softplus margin；必填 negative_reasons |
| pairs 未参与训练 | invariance 加全词表对称 KL；contrast 保留各自正确答案监督 |
| 候选依赖真实答案 | 新路径使用共享固定 bank + 上下文可学习路由，不读取答案/角色元数据 |
| 建议增加 relation_key | 接受为可选标注审计字段，校验方向，不据此选择模型空间 |
| 校验缺口 | 增加 Unicode/空白/标点归一化去重、角色模板、限定条件、可选词表检查 |
| 事实真实性无法自动证明 | 字面覆盖给出 warning，来源支持/别名/负例真假仍明确需要审计 |
| 长上下文跨 split | 读取原始逐句语料，检查 long_context 覆盖的 source line；与其他 split 冲突的整组跳过 |
| max_length 只约束 variant | 基础样本、variant 都按含 CLS/SEP 的上限检查；超长关系组跳过，不静默截断 |
| 背景变体过弱 | 优先按完整句/分句边界扩展，记录 background_spans；单纯标点变体只作为明确的 paraphrase 回退 |

v1 校验兼容保留，但新生成数据统一交付 schema_version=2。v1 的裸角色名路径仍可读；
v2 必须用 roles.recipient 这样的路径。升版时需要补 qualifiers、surface_kind 和
negative_reasons；不能自动把旧文本全标 source，也不能由程序捏造负例理由。

## 新模型输入路径与防泄漏

```text
可见 tokens（含 MASK）
  → 固定训练词表编码
  → 普通上下文层
  → 在共享三维空间 bank 中做上下文路由
  → 动态 Q/K + 18D Relation FFN
  → 全词表 logits

answer / hard_negatives / pairs → 损失
roles / qualifiers / relation_key / spans / evidence → 校验、审计
```

实现位于 `bert_simple/relation_pair_training.py`。`visible_inputs` 不接受答案或样本
元数据，只接受模型、分词器、tokens 和长度限制。所有非 padding 位置使用同一小型
候选 bank，具体权重仍由模型已有的上下文路由计算。默认 4 个空间，每个使用独立的
三个 hidden 维度。bank 在训练/评估/推理中固定，不因测试事实增设空间。

这种保守方案有意取消答案条件的 token-pair 筛选，避免接口不一致。它暂时没有
可见文本驱动的稀疏检索，4 个空间的容量是否足够需要实验，不宣称它比旧 bank 更强。
所有空间共享 FFN；旧 MLM 输出侧关系 adapter 仍保留，但也仅使用同一固定 bank。

这段 4-space 配置是协议 v2 的小型成对训练器，不是 256 维关系 sidecar 的最终容量。
正式 sidecar 使用 `examples/train_full_relation_filter_stages.py`：固定共享 bank 覆盖
完整 256 维（1500 个固定共享三维候选；前 86 个基础通道覆盖全部维度），并以
MASK—完整关系上下文组/完整背景组作为特殊 FFN 的监督单位。两者都不读取
token-pair map，但训练目标和容量不能混为一谈。

词表只从 train 样本的 tokens、answer、hard_negatives 建立，不读取证据、dev/test
答案扩展。答案参与训练词表是监督训练的一部分，不能参与单个样本的候选选择。
训练词表外输入/标签/负例直接拒绝。评估输入 OOV 映射 UNK，答案 OOV 计作失败，
分别报告覆盖率，绝不把真实答案映射成 UNK 后算命中。

默认从头训练 64 hidden / 2 layers / 4 heads / 4 spaces 的实验模型。
`--init-checkpoint` 只接受由本新流程保存的 checkpoint，保留其模型/词表，重新创建
优化器（warm start，不是断点精确续训）。不直接沿用旧答案条件空间的 checkpoint。
训练用固定 epoch，最终评估 dev/test，不用 test 选择 checkpoint。

## 损失定义

每组 2～8 个样本，分别读取自己的 MASK 位置，前向输出全词表 logits。

```text
CE = mean(每个样本的全词表交叉熵)
Margin = mean_sample mean_negative softplus(2 - logit_answer + logit_negative)
Consistency = mean_invariance 0.5 × [KL(P_a || P_b) + KL(P_b || P_a)]
Loss = CE + 0.25 × Margin + 0.1 × Consistency
```

权重和 margin 都可通过 CLI 配置。KL 两侧都参与梯度；不直接对齐不同长度句子的
attention 矩阵。contrast 不强迫输出一致，通过各自 CE 和显式负例训练区分。
负例不裁剪词表、不限定预测候选；理由文本不送进模型。

该训练器使用 AdamW 和梯度裁剪，不复用旧答案依赖的梯度空间分配，也尚未加入
通用 MLM replay/锚点保护。模型的事实保持能力需要单独验证，不能由低 loss 推断。

## 找不同与去成分的接入

`--probe-every N` 每 N 个组更新前后检查该组第一个样本，eval 模式消除 dropout 差异。
发生错→对时，沿用 18D FFN 分数变化与 attention 变化筛选两条候选，然后进行
两次单独回退和一次联合回退。使用更新前最强错误答案作为固定负例计算 margin。

结果保存到 `training_log.jsonl`，包含 step/group_id/sample_id、答案/负例 ID、
边的 layer/head/key_position、前后分数与 attention、回退影响。
key_position 包含自动插入的 CLS 偏移（数据 role_spans 不含 CLS）。
本入口日志的 tokens 字段是原始输入，不含 CLS/SEP；解释 key_position 时需减 1。
抽样不会捕获所有纠错；回退证据当前仅用于诊断，不作为新增辅助损失。
成对损失训练与“学习纠错经验”是不同功能，后者仍需独立设计和对照实验。

## 校验器的边界

自动检查包括：

- MASK/答案/负例与 span 结构、背景角色不重叠；
- 同一 relation_type 的角色键模板统一（v2）；
- qualifiers 合法性、changed_slots 路径实际变化；
- source 摘录恢复后一致性，改写类型显式声明；
- 来源、同义 ID 的规范事实、归一化文本跨 split 隔离；
- 可选 --vocab 下输入/答案/负例覆盖；
- relation_key 角色和方向一致，negative_reasons 覆盖全部负例。

不自动证明：原文事实真假、引文蕴含关系、别名指代、负例错误、时间限定在文本中
是否充分、语义级改写泄漏。角色值未字面出现在证据时记 lexical_evidence_warnings，
不把这种情况直接判错，也不能把“有字面覆盖”当作证据充分。
schema v2 的统计包含事实数、来源组数、关系类型数、样本数与各类 pair 数；
surface_kind 和 evidence.kind 分开统计。原始语料总句数需引用独立来源报告，
不能由本 JSONL 的样本数替代，语法属性标注也不属于本协议。

## 运行命令

```bash
python examples/build_relation_training_v2.py
python examples/validate_relation_training_data.py generated.jsonl
python examples/train_relation_pairs.py generated.jsonl \
  --output-dir outputs/relation-pairs-v2-run1 --epochs 10 --probe-every 100
python examples/predict_relation_pairs.py outputs/relation-pairs-v2-run1 \
  '青王 封 林舟 为 [MASK] 。'
python examples/test_structured_relation.py
python examples/test_relation_data_validation.py
python examples/test_relation_pairs.py
```

输出目录必须不存在，防止覆盖已有实验。产物包括模型、词表、relation_pair_config.json
（候选策略、数据 SHA256、种子、超参）、training_log.jsonl、metrics.json。
metrics 分 train/dev/test 报 Top-1、整组全对数量、invariance/contrast 两句都正确率、
答案 OOV 和含输入 OOV 的样本数量。空 split 指标为 null，不伪装成 0 准确率。

旧 train_shiji_structured_relation.py 和旧数据构造代码保留为历史对照，没有静默
重写其结果或候选含义。其旧测试不能用作新流程的无泄漏基线，比较时必须重新评估。

## 验证与尚未完成事项

已从 `data/shiji/manifests/fact_memory_dataset.json` 生成一份可供协议验证和人工审核的
关系训练草稿：`data/shiji/manifests/relation_training_v2_draft.jsonl`，统计报告在同目录
的 `relation_training_v2_draft_report.json`。当前包含 193 个 schema v2 组、386 个事实、
772 个样本，按 split 分为 train/dev/test = 171/10/12 组。生成器只接收原事实清单中恰好
有两个事实的来源组，并跳过跨 split 的 37 个语义事实组、25 个长上下文跨 split 组和 407
个单事实组，避免为了凑对比样本而制造泄漏或伪造事实。基础样本和变体都满足默认
`max_length=128`（含 CLS/SEP）；变体优先使用完整句/分句背景并写入 `background_spans`。
原始负例沿用清单中的 distractors，并标注为需要人工审核；当前有 218 条 lexical evidence
warning，表示角色或抽象谓词没有逐字出现在引文中，不是事实已被自动证明。该文件是
draft，不等同于一万多句史记原料的完整关系标注，训练前应按来源、别名、时间限定和负例
逐组抽查。

当前工作区默认 `.venv` 是 Python 3.14 且没有安装 PyTorch；数据校验、无依赖回归测试、
Python 编译检查和 `git diff --check` 已通过。模型测试改用
`/home/goodtime/project/rag-go/hermit_crab/.venv/bin/python`（Python 3.12.14、PyTorch
2.14.0+cpu）运行：`test_structured_relation.py` 的 8 项、`test_relation_pairs.py` 的
12 项全部通过。该环境提示缺少 NumPy，但不影响这些测试；运行更广泛实验前仍应按
requirements.txt 补齐依赖。测试和 smoke test 不替代真实数据上的准确率实验，也没有
使用或覆盖现有真实训练产物。

已完成合成示例的端到端训练、每步纠错探测、模型/词表保存重载、独立预测 CLI、
warm start、输出目录覆盖拒绝测试，以及协议字段/归一化泄漏/标签不影响前向的测试。
这证明实现链路可运行，不证明真实史料上的准确率或因果解释质量。

尚未执行大规模真实数据训练、多种子效果对照、稀疏候选检索、纠错经验辅助损失、
通用能力保持评估。外部 AI 可以按 v2 先生成 100 组，经校验和来源抽查后再扩充。
