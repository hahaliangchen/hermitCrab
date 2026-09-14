# 语法状态机与训练数据 V3

本轮最终采用“规则先验证、再入库”的做法。扩展脚本
`examples/build_extended_word_attribute_data.py` 生成一个遮罩样本后，会先
调用 `FixedGrammarAutomaton`；如果期望结构和状态机实际返回结构不一致，
该样本不会被计入新增槽位。这样可以避开 `叫` 的命名/兼语歧义，以及 `被`
后面既可能是人物又可能是动作的歧义。

固定状态机现在覆盖：

* `SUBJECT`：谓词前主语，包括带前置话题的句子；
* `PASSIVE`：被动施事和无施事被动动作；
* `NEGATION`：不、未、没有、无、未曾、莫；
* `CAUSATIVE`：派、让、使、令、命令，以及带连续动词的叫；
* `CONDITION`：条件分句栈，支持条件嵌套。

最终真实语料统计为 10,631 句、5,555 条槽位样本：

```text
SUBJECT    705
PASSIVE    552
NEGATION  1379
CAUSATIVE  962
CONDITION  544
```

最终状态一致性检查中，上述新增槽位的结构错配数为 0。训练产物位于
`outputs/grammar-attribute-filter`；固定规则覆盖检查为：属性命中率约
98.18%，非回退匹配率约 94.56%。这两个数字衡量的是规则覆盖，不是完整
中文语法的泛化准确率。
