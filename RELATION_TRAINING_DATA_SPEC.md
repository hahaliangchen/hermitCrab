# 关系 FFN 对照训练数据规范 v2

本文件可直接交给数据生成 AI。目标是训练事实补全、背景不变性和关键角色区分，
配合模型内部“更新前后找不同 → 单条/联合回退 → 记录纠错经验”。
架构基线固定为 `[q,k,|q-k|,vec(q⊗k)]` 共 **18 维**，共享 FFN 为 **18→32→1**。
数据无需生成三维坐标、attention、因果贡献或“模型改对原因”；这些必须由真实运行产生。

## 1. 交付格式

- UTF-8 JSONL，一行一个完整对照组；不输出 Markdown 围栏、注释或省略号。
- 每组 2～8 个样本，至少一个不变性对照；整批同时包含关键条件变化的对照。
- 独立报告各 split 的组数、样本数、事实数、关系类型、对照类型及去重结果。
- 初期建议先提交 100 组供人工验收，再扩展规模。建议按来源组划分约 80/10/10，
  比例是目标，不得为凑比例拆开同一来源或事实。
- 使用 `examples/validate_relation_training_data.py 文件.jsonl` 做结构校验。
  校验通过只代表结构和显式约束合格，不证明事实真实或语言无歧义。

## 2. 事实与证据

每个事实必须具有 `fact_id`、`relation_type`、`roles`、`qualifiers`、`evidence`。
`roles` 保存具体值，例如 actor=汉王、predicate=封、recipient=张耳、title=赵王。
别名、命名、地点等关系使用对应角色名；同类关系的角色命名必须统一。
同一个 relation_type 的角色键集合必须一致。`qualifiers` 是对象，无条件限制时为 `{}`；
可用键为 `time`、`place`、`condition`、`stage`，值为非空字符串。例如
`{"time":"汉五年","stage":"改封之后"}`。影响答案唯一性的限定必须在 tokens 中表达，
仅在 qualifiers 中写出、却不在文本中给出，不足以消歧。

可选 `relation_key` 只作标注审计/未来辅助监督：
`{"relation_type":"APPOINT","source_slot":"actor","target_slot":"recipient","direction":"actor_to_recipient"}`。
两端角色必须属于 roles，direction 必须与两端一致。这里 target_slot 是关系边终点，
与样本“本次提问的 target_slot”不同；它不分配三维坐标，也不参与模型候选选择。

真实史料只使用委托方提供、可以核对的原文；没有证据就不生成该事实。
`evidence` 包含 `source_id`（文件/URL/篇章位置）、`quote`（支持答案的原文）、
`kind`（`provided_source` 或 `synthetic`）。不能捏造引文和链接。
合成规则实验使用明确虚构的人名、称号和情境，标记 synthetic，与史料评估分开统计。
对于多时期任命、同名人物或相互冲突的事实，必须在输入中保留消歧条件；不能唯一
确定单一答案的样本不进入本协议。不能仅凭“汉王+封+某人”就假设任意时期职位唯一。

## 3. 字段约定

组字段：`schema_version`=2、全局唯一 `group_id`、`source_group`、
`split`（train/dev/test）、`facts` 数组、`samples` 数组、`pairs` 数组。

样本字段：

- `sample_id`：全局唯一。
- `fact_id`：引用本组事实。
- `tokens`：已经分词的字符串数组，恰好一个 `[MASK]`，无需 CLS/SEP/PAD。
  人名、职位等答案必须完整作为一个 token；目前 MLM 一次预测一个 token。
  生成器必须同时对基础样本和所有变体检查 `len(tokens)+2 <= max_length`（预留 CLS/SEP），
  超长组应过滤或按保留 MASK、谓词和必要角色的规则裁剪，不能交给训练入口才失败。
- `answer`：与该事实 `roles[target_slot]` 完全一致的一个 token。
- `target_slot`：本次提问的角色，例如 title 或 recipient。
- `surface_kind`：必填 `source|paraphrase|synthetic|adversarial`。
  source 表示遮蔽答案前的句子是证据原文的连续摘录；paraphrase 是有依据的改写；
  synthetic 是合成表达；adversarial 是刻意增加干扰的表达。后两者也必须有事实依据，
  不能将虚构事件当成真实史料。事实 evidence.kind 与句子 surface_kind 分开统计。
- `role_spans`：已出现的关键角色到 `[start,end]` 的映射，0 起始、左闭右开，
  指向本句 tokens（不包含 CLS）。标注任命事件中的“张耳”，不能标背景事件里的同名词。
  不标被遮蔽的 target_slot；不要求所有角色都有 span，省略时必须仍可确定答案。
- `background_spans`：背景片段的 `[start,end]` 列表；不能覆盖 MASK 或关键角色。
  原文背景变体应尽量按完整句/分句边界扩展；新增背景范围必须标出，不能用只多一个标点
  或孤立 token 的“扩展”冒充干扰上下文。
- `hard_negatives`：至少一个符合答案角色类型的错误候选，必须确实不是该问法的有效答案。
  不能因为另一个人物出现在背景中，就把他机械地当成职位槽的负例。
- `negative_reasons`：必填对象，对每一个 hard_negative 说明在当前条件下为何错误，
  如 `{"西侯":"该职位属于陆衡，当前被任命者为林舟"}`。负例可出现在背景中，
  但不得是本次问法的另一个合理答案。原因是审计信息，不送入模型。

对照字段：`a`、`b` 为本组 sample_id，`kind` 为 invariance 或 contrast，
`changed_slots` 为变化字段路径列表（背景/同义改写的不变性对照用空列表）。
允许 `relation_type`、`roles.<角色名>`、`qualifiers.time|place|condition|stage`、`target_slot`。
例如 `["roles.recipient"]` 或 `["qualifiers.time"]`；改变提问目标时包含 `target_slot`。
标注被主动改变的条件，不必重复列出随之变化的答案角色；每条路径必须实际发生变化。
同义动作表述可在 roles.predicate 中使用一致的规范名，表面词由 tokens 和 spans 记录。

这些标注都是训练/评估元信息。**推理输入仅为 tokens，不能把 answer、fact_id、
target_slot、gold role_spans 或来源引文编码成候选空间以泄露答案。** 候选空间只能由
可见文本和训练阶段建立的注册表产生；内部“三维维度组合”不等于数据里的语义三元组。
当前 256 维 sidecar 将 role_spans 及其与 MASK 之间的必要连接词作为一个完整关系上下文组，
把 background_spans 与其余非关系上下文作为背景组，先做组聚合再训练特殊 FFN；不会为
任意两个 token 建立独立三维空间。缺少 role_spans 时只能将 MASK 所在完整分句作为弱上下文，
并依靠背景组进行对照，不能把这种弱标注误称为精确角色解析。

## 4. 必须覆盖的对照

1. 不变性：同一 fact_id 和 target_slot，答案相同；增删无关背景、移动背景位置、
   同义改写、主被动改写。包含相近人名干扰、背景中同名实体、不同长度和词序。
2. 关键条件变化：替换事件角色、关系或时间条件，必须引用另一条有证据的事实；
   答案确实变化才标 contrast。不能任意替换人名后沿用原答案。
3. 提问目标变化：例如从问“封为什么”变为问“谁被封”，答案与目标角色同步变化。
4. 删除必要信息导致歧义：不强行保留原标签，排除或单独交付为未来多解/拒答集。

建议样本约一半用于不变性，另一半覆盖关键变化和目标切换；不要只做一种任命模板。
完整输入不得出现 answer 或直接泄露答案的同义表达。不要把 MASK 一律放在同一个位置，
不要让背景长度、标点、模板编号、split 标记成为答案线索。

## 5. 划分与评估

同一事实（含 qualifiers）、来源篇章及其所有改写必须保持在同一 split。所有对照组和 source_group
不可跨 split；同一内容换 ID 仍视为泄漏。长上下文覆盖的原始句子范围也必须保持在同一 split，
不能因为 fact_id 和完整输入不同就放过跨句/跨篇章泄漏。分词归一化后的输入不可跨 split 重复。
自动归一化为 Unicode NFKC、去空白、统一句号和弯引号，能检查分词边界变化和全半角差异。
同义改写、别名及语义重复仍需额外审计，校验器不能自动证明没有这些泄漏。
默认 test 测试留出来源上的能力，但未见且没有上下文证据的历史事实不能凭空回答：
报告中分清事实是否已学、是否可从可见证据推出。

另需评估“已学事实的新表达”时，建立单独的 evaluation suite：允许事实重叠，
但改写模板、背景组合及文本不得参与训练，并明确标记它不是未见事实测试。
不要混进默认的事实隔离 split 以免混淆指标。

同时报告答案 Top-1、同组正确率、不变性对照两句都正确率、contrast 两句都正确率、
旧事实保持率。两句都答错却相同不算不变性成功。

## 6. 完整合成示例（仅展示格式，不是历史事实）

下面 JSON 排成多行便于阅读；实际交付压为一行。

```json
{
  "schema_version": 2,
  "group_id": "synthetic-001",
  "source_group": "synthetic-story-001",
  "split": "train",
  "facts": [
    {"fact_id": "syn-f1", "relation_type": "APPOINT", "roles": {"actor": "青王", "predicate": "封", "recipient": "林舟", "title": "东侯"}, "qualifiers": {}, "evidence": {"kind": "synthetic", "source_id": "synthetic-story-001", "quote": "虚构故事：林舟与陆衡归来后，青王封林舟为东侯，封陆衡为西侯。"}},
    {"fact_id": "syn-f2", "relation_type": "APPOINT", "roles": {"actor": "青王", "predicate": "封", "recipient": "陆衡", "title": "西侯"}, "qualifiers": {}, "evidence": {"kind": "synthetic", "source_id": "synthetic-story-001", "quote": "虚构故事：林舟与陆衡归来后，青王封林舟为东侯，封陆衡为西侯。"}}
  ],
  "samples": [
    {"sample_id": "syn-s1", "fact_id": "syn-f1", "tokens": ["青王", "封", "林舟", "为", "[MASK]", "。"], "answer": "东侯", "target_slot": "title", "surface_kind": "synthetic", "role_spans": {"actor": [0,1], "predicate": [1,2], "recipient": [2,3]}, "background_spans": [], "hard_negatives": ["西侯"], "negative_reasons": {"西侯":"该职位属于陆衡，当前被任命者为林舟"}},
    {"sample_id": "syn-s2", "fact_id": "syn-f1", "tokens": ["林舟", "与", "陆衡", "归来", "后", "，", "青王", "封", "林舟", "为", "[MASK]", "。"], "answer": "东侯", "target_slot": "title", "surface_kind": "adversarial", "role_spans": {"actor": [6,7], "predicate": [7,8], "recipient": [8,9]}, "background_spans": [[0,6]], "hard_negatives": ["西侯"], "negative_reasons": {"西侯":"陆衡是干扰实体，任命对象仍为林舟"}},
    {"sample_id": "syn-s3", "fact_id": "syn-f2", "tokens": ["青王", "封", "陆衡", "为", "[MASK]", "。"], "answer": "西侯", "target_slot": "title", "surface_kind": "synthetic", "role_spans": {"actor": [0,1], "predicate": [1,2], "recipient": [2,3]}, "background_spans": [], "hard_negatives": ["东侯"], "negative_reasons": {"东侯":"该职位属于林舟，当前被任命者为陆衡"}}
  ],
  "pairs": [
    {"a": "syn-s1", "b": "syn-s2", "kind": "invariance", "changed_slots": []},
    {"a": "syn-s1", "b": "syn-s3", "kind": "contrast", "changed_slots": ["roles.recipient"]}
  ]
}
```

## 7. 与训练流程的接口

新入口 `examples/train_relation_pairs.py` 直接消费本 JSONL，无需转换成旧 manifest。
基础损失为所有样本的全词表答案 CE，追加 hard-negative softplus margin；对 invariance
对照追加全词表分布的对称 KL。contrast 不施加一致性，保留各自 CE 和负例监督。
默认 `loss = mean(CE) + 0.25*mean(margin_loss) + 0.1*mean(symmetric_KL)`，
margin 为 2.0；负例字符串映射为词表 ID，绝不限制推理候选词表。
位置对齐应依据各自 MASK 位置，不直接逐位置对齐不同长度句子的 attention。

纠错记录使用同一输入、同一候选集合、关闭 dropout 的更新前后前向。
注意力与关系分数变化用于候选筛选；两条候选分别回退和联合回退，比较固定正确/错误
候选的 margin。记录模型版本、训练步、sample_id、负例 ID、层/头/位置及干预结果。
这些结果是模型内部、局部条件下的证据，不允许数据 AI 预填“原子因果组”。

新入口默认随机初始化，64 hidden、2 层、4 heads、4 个共享三维空间；所有输入采用同一
小型候选 bank，上下文路由学习权重。不会读取答案构造 token pair，不根据 gold 角色标签
分配空间。训练、评估、`examples/predict_relation_pairs.py` 共用 visible_inputs。
词表仅由 train 样本的可见 tokens、答案及负例构建，不读取 dev/test 或证据原文扩展词表。
训练 OOV 拒绝；评估答案 OOV 计为答错并单独报告，输入 OOV 映射 UNK 并报告数量。
超过 max_length 的样本拒绝，不静默裁掉关键角色。默认每 100 个组更新抽样一次纠错，
每次检查组内第一个样本；可能漏掉其他样本和时刻。回退经验目前记日志，不追加自动归因损失。

```bash
python examples/validate_relation_training_data.py generated.jsonl
python examples/validate_relation_training_data.py generated.jsonl --vocab path/to/vocab.json
python examples/train_relation_pairs.py generated.jsonl --output-dir outputs/relation-pairs-run1
python examples/predict_relation_pairs.py outputs/relation-pairs-run1 '青王 封 林舟 为 [MASK] 。'
```

校验器仍接受 v1，兼容缺省 qualifiers、旧 changed_slots；新数据必须生成 v2。
校验器检查来源/事实/归一化文本隔离、同类角色模板、source 摘录一致性和字段路径变化。
`lexical_evidence_warnings` 只提示角色字面值未出现在引文中，不能证明事实错误或真实；
别名/指代、时间消歧、负例真实性和同义泄漏仍需来源审计。词表检查仅在提供 --vocab 时运行。
旧 `train_shiji_structured_relation.py` 保留作历史对照，其中答案参与候选构造，不能用它
证明未知答案场景的泛化。正式新数据训练请用 train_relation_pairs.py。
