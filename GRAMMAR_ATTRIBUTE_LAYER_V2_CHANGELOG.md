# 语法状态机第二版变更

本轮在 `bert_simple/grammar_automaton.py` 中加入了：

* `SUBJECT`：识别谓词前的主语，包括带话题前缀的形式；
* `PASSIVE`：识别 `被` 后施事和无施事被动动作两种情况；
* `NEGATION`：识别 `不/未/没有/无/未曾/莫` 后的槽位；
* `CAUSATIVE`：识别 `派/让/使/令/命令` 形成的兼语/连动槽位；
* `CONDITION`：用条件栈处理 `如果/若/只要/因为/虽然` 等分句，逗号弹出
  当前条件层，支持条件嵌套。

真实《史记》语料新增的遮罩槽位统计：

```text
SUBJECT    369
PASSIVE    552
NEGATION  1382
CAUSATIVE 1020
CONDITION  548
```

扩展数据由 `examples/build_extended_word_attribute_data.py` 生成，输出仍
是 `data/shiji/manifests/word_attribute_dataset.json`。当前总槽位样本为
5,281 条；用更新后的固定状态机评估，属性命中率约 97.97%，非回退匹配率约
94.28%。这些指标是规则覆盖检查，不是神经网络训练准确率，也不是对全部
中文语法的保证。

固定语法层仍不更新主 BERT。词属性层继续单独用多标签 BCE 学习；主模型接入
时只把属性兼容度作为候选 logit bias，并保留 `OPEN` 状态的中性放行。
