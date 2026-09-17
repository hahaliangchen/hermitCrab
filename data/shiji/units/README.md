# 《史记》历史单元关系抽取训练数据集清单 (Shiji Unit Datasets)

本数据集将《史记》10631 句语料与标注事实，依据太史公本纪、世家、列传的脉络，划分为 **6 个主题历史单元**。
每个单元构成一个高内聚、低跨度的局部历史语义网络，杜绝跨朝代/跨篇章的名人光环先验偏置（如蔺相如吞噬魏子、刘邦吞噬楚怀王等问题）。

## 数据集完整性与无泄漏保证
1. **全局原子连通分量划分**：输入 178 组，保留 173 组；凡共享同一证据句（`source_id`）、事实 ID 或语义事实（`semantic_key`）的组，先在 6 个单元之间统一聚类，再分配到同一个 train/dev/test。
2. **高置信度清洗**：剔除 5 组明确的伪实体、截断实体或错误关系，并在 `shiji_units_quality_report.json` 中保留逐组原因；剩余规则抽取仍需人工抽查。
3. **双层文件组织**：既提供标准组级合并文件（供 `train_relation_pairs.py` 训练），也提供子目录拆分文件及扁平化样本文件（`samples_*.jsonl`，供常规 MLM/评估使用）。

## 单元概览表

| 单元 ID | 单元名称 | 总组数 | 样本总数 | Train 组(条) | Dev 组(条) | Test 组(条) | 核心实体 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `unit1_chuhan_founding` | **楚汉争霸与开国功臣** | 59 | 236 | 41 (164) | 9 (36) | 9 (36) | 刘邦, 周亚夫, 将军, 赵王, 韩信, 汉高祖 |
| `unit2_warring_states_retainers` | **战国风云与将相门客** | 18 | 72 | 12 (48) | 3 (12) | 3 (12) | 蔺相如, 舍人, 楚国, 顷襄王, 大将, 廉颇 |
| `unit3_han_generals_frontier` | **大汉边疆与帝国名将** | 34 | 136 | 24 (96) | 5 (20) | 5 (20) | 侯, 中郎将, 张骞, 赵信, 卫青, 将军 |
| `unit4_qin_empire_reform` | **大秦帝国与法家变法** | 47 | 188 | 33 (132) | 7 (28) | 7 (28) | 李斯, 丞相, 太子, 蒙恬, 成汤, 大将 |
| `unit5_spring_autumn_hegemony` | **春秋诸侯与先秦霸业** | 11 | 44 | 7 (28) | 2 (8) | 2 (8) | 太子, 申生, 唐叔虞, 大夫, 周成王, 周武王 |
| `unit6_han_society_and_economy` | **汉代儒林与货殖风云** | 4 | 16 | 2 (8) | 1 (4) | 1 (4) | 皇帝, 秦二世, 刘彻, 汉武帝, 卫青, 大将 |

---

## 清洗与评估注意事项

- 全局连通分量数：131；跨单元连通分量：3。
- 每个单元的 split 以 group 为单位分配；8 组以上的单元至少保留 2 组 dev 和 2 组 test。少于 8 组的单元只能保留单组 holdout，结果仅作探索性参考。
- 本次高置信度剔除：
  - `shiji-rel-v2-line_5469_KINSHIP_16_19`：国君 is a generic title, not an identified person, in 侠累是韩国国君的叔父.
  - `shiji-rel-v2-line_5760_appoint_15_17`：拜官吏为老师 is a learning relationship, not an appointment; 人/官吏/老师 are generic roles.
  - `shiji-rel-v2-line_6472_appoint_5_7`：the appointment extractor used 前往 as the actor in 派张良前往立韩信为齐王.
  - `shiji-rel-v2-line_9307_KINSHIP_2_6`：王孙 is a truncated parent in 卓王孙的女儿卓文君; the exact parent is not represented.
  - `shiji-rel-v2-line_9997_appoint_9_14`：义 is a truncated mention in 义的弟弟义纵; the one-token schema cannot safely recover 义纵.
- hard negatives 继承自原事实清单，仍带有人工复核标记，不能视为已完成的负例标注。

## 各单元详细说明

### 【楚汉争霸与开国功臣】 (`unit1_chuhan_founding`)
- **历史背景**：围绕汉高祖刘邦、项羽、韩信、张耳、陈余、周勃、萧何等人的封王拜将、诛杀背叛与楚汉争霸历史。
- **涵盖篇章**：张耳陈余列传, 汉初勋戚列传(刘敬/叔孙通/季布), 淮阴侯列传, 绛侯周勃世家, 陈涉世家, 项羽本纪, 高祖本纪
- **主文件**：`data/shiji/units/unit1_chuhan_founding.jsonl`
- **拆分目录**：`data/shiji/units/splits/unit1_chuhan_founding/`
  - `train.jsonl` / `samples_train.jsonl`：41 组 (164 条样本)
  - `dev.jsonl` / `samples_dev.jsonl`：9 组 (36 条样本)
  - `test.jsonl` / `samples_test.jsonl`：9 组 (36 条样本)
- **高频角色/实体候选**（仅作统计，不等同于知识库实体）：刘邦, 周亚夫, 将军, 赵王, 韩信, 汉高祖, 大将, 灌婴, 列侯, 楚王, 项籍, 项羽, 汉王, 武臣, 郦商

### 【战国风云与将相门客】 (`unit2_warring_states_retainers`)
- **历史背景**：围绕孟尝君养士收租、廉颇蔺相如完璧归赵与将相和、田单火牛阵、孙子吴起兵学等战国风云。
- **涵盖篇章**：兵学将相列传(伯夷/孙子/吴起), 廉颇蔺相如列传, 田单/屈原/刺客列传
- **主文件**：`data/shiji/units/unit2_warring_states_retainers.jsonl`
- **拆分目录**：`data/shiji/units/splits/unit2_warring_states_retainers/`
  - `train.jsonl` / `samples_train.jsonl`：12 组 (48 条样本)
  - `dev.jsonl` / `samples_dev.jsonl`：3 组 (12 条样本)
  - `test.jsonl` / `samples_test.jsonl`：3 组 (12 条样本)
- **高频角色/实体候选**（仅作统计，不等同于知识库实体）：蔺相如, 舍人, 楚国, 顷襄王, 大将, 廉颇, 将军, 赵奢, 周武王, 西伯昌, 老子, 楚怀王, 子兰, 假, 宫

### 【大汉边疆与帝国名将】 (`unit3_han_generals_frontier`)
- **历史背景**：围绕汉武帝开疆拓土、卫青霍去病出击匈奴、飞将军李广抗匈、张骞出使大宛西域等边疆战功。
- **涵盖篇章**：卫将军骠骑列传, 司马相如列传, 大宛列传, 张释之冯唐列传, 李将军列传
- **主文件**：`data/shiji/units/unit3_han_generals_frontier.jsonl`
- **拆分目录**：`data/shiji/units/splits/unit3_han_generals_frontier/`
  - `train.jsonl` / `samples_train.jsonl`：24 组 (96 条样本)
  - `dev.jsonl` / `samples_dev.jsonl`：5 组 (20 条样本)
  - `test.jsonl` / `samples_test.jsonl`：5 组 (20 条样本)
- **高频角色/实体候选**（仅作统计，不等同于知识库实体）：侯, 中郎将, 张骞, 赵信, 卫青, 将军, 周亚夫, 公孙弘, 丞相, 张汤, 廷尉, 头曼, 单于, 公孙敖, 南越王

### 【大秦帝国与法家变法】 (`unit4_qin_empire_reform`)
- **历史背景**：围绕商鞅变法、秦始皇一统天下、李斯蒙恬辅政与沙丘权谋、酷吏执法与早周先秦世系。
- **涵盖篇章**：周本纪, 商君/苏秦列传, 夏本纪, 李斯/蒙恬列传, 殷本纪, 秦始皇本纪, 酷吏列传
- **主文件**：`data/shiji/units/unit4_qin_empire_reform.jsonl`
- **拆分目录**：`data/shiji/units/splits/unit4_qin_empire_reform/`
  - `train.jsonl` / `samples_train.jsonl`：33 组 (132 条样本)
  - `dev.jsonl` / `samples_dev.jsonl`：7 组 (28 条样本)
  - `test.jsonl` / `samples_test.jsonl`：7 组 (28 条样本)
- **高频角色/实体候选**（仅作统计，不等同于知识库实体）：李斯, 丞相, 太子, 蒙恬, 成汤, 大将, 胡亥, 高祖, 国君, 相, 卫鞅, 长史, 李由任, 姬发, 将军

### 【春秋诸侯与先秦霸业】 (`unit5_spring_autumn_hegemony`)
- **历史背景**：围绕春秋五霸齐桓晋文争霸、勾践卧薪尝胆灭吴、孔子周游列国与儒家先贤修齐治平。
- **涵盖篇章**：孔子世家, 春秋诸侯世家(齐/晋/楚), 越王勾践世家
- **主文件**：`data/shiji/units/unit5_spring_autumn_hegemony.jsonl`
- **拆分目录**：`data/shiji/units/splits/unit5_spring_autumn_hegemony/`
  - `train.jsonl` / `samples_train.jsonl`：7 组 (28 条样本)
  - `dev.jsonl` / `samples_dev.jsonl`：2 组 (8 条样本)
  - `test.jsonl` / `samples_test.jsonl`：2 组 (8 条样本)
- **高频角色/实体候选**（仅作统计，不等同于知识库实体）：太子, 申生, 唐叔虞, 大夫, 周成王, 周武王, 管仲, 谋士, 季子, 司空, 常寿过, 夫差, 霸王, 勾践, 丘

### 【汉代儒林与货殖风云】 (`unit6_han_society_and_economy`)
- **历史背景**：围绕汉代儒学兴起独尊儒术、货殖列传天下商贾富商大贾、滑稽讽谏与文景之治社会百态。
- **涵盖篇章**：儒林列传, 孝文本纪/武帝, 货殖/滑稽列传
- **主文件**：`data/shiji/units/unit6_han_society_and_economy.jsonl`
- **拆分目录**：`data/shiji/units/splits/unit6_han_society_and_economy/`
  - `train.jsonl` / `samples_train.jsonl`：2 组 (8 条样本)
  - `dev.jsonl` / `samples_dev.jsonl`：1 组 (4 条样本)
  - `test.jsonl` / `samples_test.jsonl`：1 组 (4 条样本)
- **高频角色/实体候选**（仅作统计，不等同于知识库实体）：皇帝, 秦二世, 刘彻, 汉武帝, 卫青, 大将, 韩商, 韩生

## 使用方法与训练示例

### 1. 使用 relation_pair 模型进行单元训练
```bash
# 训练战国风云与将相门客单元 (Unit 2)
python examples/train_relation_pairs.py data/shiji/units/unit2_warring_states_retainers.jsonl \
    --output-dir outputs/unit2-256d-4l \
    --hidden-size 256 \
    --num-layers 4 \
    --epochs 20
```

### 2. 使用 evaluate_by_unit 进行多单元多维评测与探针诊断
```bash
# 评估模型在各单元上的准确率、负例击败率及核心探针 (如 魏子 vs 蔺相如)
python examples/evaluate_by_unit.py outputs/unit2-256d-4l
```

### 3. 直接读取扁平化样本 (samples_*.jsonl)
每条样本为一行独立 JSON：
```json
{
  "group_id": "shiji-rel-v2-line_4781_TITLE_LINK_7_8",
  "sample_id": "shiji_fact_0370:base",
  "split": "dev",
  "tokens": ["宦者", "令缪贤", "说", "：", "“", "我", "的", "[MASK]", "蔺相如", "可以", "充任", "使者", "。", "”"],
  "answer": "舍人",
  "target_slot": "object",
  "hard_negatives": ["御史大夫", "郎中令", "中山王", "汉武帝", "太子"]
}
```