"""Build a source-verified Shiji fact-memory dataset.

This builder deliberately keeps the data conservative:

* every answer is an exact one-token span in the source line;
* the mask is applied at that span, never at the first textual occurrence;
* every variant carries its own answer and target role;
* relation names are canonical labels, while the original verb is retained as
  evidence;
* naming expressions such as “项羽 又 叫 项籍” are not treated as commands;
* source lines, rather than individual variants, determine train/dev/test split.

The extractor is still heuristic Chinese pattern matching.  The output should
therefore be treated as a verified candidate set, not as a complete semantic
graph.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import jieba.posseg as pseg
except ImportError:  # pragma: no cover - a deterministic lexical fallback is bundled
    pseg = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "shiji"
SEGMENTED_PATH = DATA_DIR / "segmented" / "shiji_segmented.txt"
MANIFESTS_DIR = DATA_DIR / "manifests"
OUTPUT_DIR = ROOT / "outputs" / "fact-memory"
ATTRIBUTE_MANIFEST = MANIFESTS_DIR / "word_attribute_dataset.json"

PUNCTUATION = {
    "，", "。", "；", "：", "！", "？", "、", "“", "”", "‘", "’",
    ",", ".", ";", ":", "!", "?", "(", ")", "（", "）", "《", "》", "—", "——",
}
SENTENCE_END = {"。", "！", "？", "；", "!", "?"}
COMMA_MARKS = {"，", ",", "；", ";", "：", ":"}

PERSON_DISTRACTORS = [
    "刘邦", "项羽", "张耳", "陈余", "韩信", "范增", "陈胜", "吴广", "蒯通",
    "萧何", "曹参", "樊哙", "张良", "吕不韦", "嬴政", "蒙恬", "李斯", "扁鹊",
    "屈原", "贾谊", "廉颇", "蔺相如", "赵括", "白起", "孙武", "孙膑", "庞涓",
    "伍子胥", "勾践", "夫差", "齐桓公", "晋文公", "楚庄王", "商汤", "夏桀",
    "周武王", "姜尚", "卫青", "霍去病", "魏公子", "信陵君", "平原君",
    "孟尝君", "武臣", "邵骚", "秦始皇", "章邯", "英布", "彭越", "周勃",
    "大禹", "商鞅", "郑国", "赵王歇", "张敖", "宋义", "周公旦", "微子",
    "箕子", "荆轲",
]

MORE_PERSON_NAMES = {
    "寒浞", "羿", "少康", "靡", "孔甲", "刘累", "桀", "纣王", "妲己", "武丁",
    "姬昌", "吕望", "周公", "召公", "秦王政", "赵高", "项梁", "项籍", "项伯",
    "灌婴", "周兰", "龙且", "李良", "陈平", "周亚夫", "李广", "李息", "李陵",
    "张骞", "陆贾", "陆生", "赵佗", "蒙骜", "蒙武", "蒙毅", "卫鞅", "吴起",
    "田单", "田广", "子玉", "重耳", "晋惠公", "晋献公", "楚昭王", "楚怀王",
    "楚成王", "秦穆公", "秦惠王", "秦昭王", "秦孝公", "秦襄公", "秦庄襄王",
    "文帝", "景帝", "孝文帝", "孝景帝", "汉武帝", "汉高祖", "高祖", "高帝",
}
# The attribute manifest is intentionally conservative and was built for a
# different task.  These names occur in explicit Shiji identity/kinship/title
# constructions, so keep them in a separate relation lexicon instead of
# treating every frequent token as a person.
RELATION_PERSON_NAMES = {
    "禹", "舜", "尧", "鲧", "启", "太康", "相", "杼", "浇", "伢", "丹朱", "妹喜",
    "成汤", "伊尹", "契", "简狄", "姜原", "昭明", "公刘", "鞠", "太丁", "外丙", "仲壬",
    "太甲", "沃丁", "武乙", "比干", "微子启", "微子", "箕子", "西伯昌", "伯禽", "唐叔虞",
    "叔虞", "周成王", "周武王", "周厉王", "周幽王", "褒姒", "宜臼", "伯服", "申生", "夷吾",
    "子圉", "子兰", "熊心", "项庄", "项伯", "项梁", "项羽", "项籍", "樊哙", "吕雉", "吕后",
    "吕公", "吕禄", "吕须", "鲁元公主", "刘如意", "刘盈", "刘恒", "刘彻", "嬴政", "秦王政",
    "子楚", "安国君", "庄襄王", "秦二世", "赵姬", "华阳夫人", "扶苏", "胡亥", "王绾", "冯劫", "冯去疾",
    "蒙恬", "蒙毅", "蒙武", "蒙骜", "李斯", "李由任", "李由", "荆轲", "樊於期", "秦舞阳",
    "高渐离", "太子丹", "燕昭王", "燕王喜", "公叔痤", "孝公", "景监", "张仪", "苏秦", "苏代",
    "苏厉", "孙膑", "庞涓", "田忌", "田文", "孟尝君", "管仲", "鲍叔牙", "晏婴", "崔杼", "庄公",
    "勾践", "夫差", "伍子胥", "文种", "范蠡", "孔子", "孔丘", "仲尼", "师襄子", "颜回", "曾参",
    "子贡", "子路", "老子", "老聃", "庄子", "庄周", "孟子", "荀子", "贾谊", "贾生", "司马相如",
    "卓文君", "卓王孙", "司马迁", "董仲舒", "主父偃", "公孙弘", "公孙敖", "张骞", "卫青", "霍去病",
    "李广", "李敢", "苏武", "赵佗", "赵胡", "冒顿", "头曼", "昆莫", "淳维", "孙叔敖", "优孟",
    "东方朔", "朱买臣", "张汤", "晁错", "袁盎", "郅都", "汲黯", "邓通", "夏侯婴", "郦食其", "陆贾",
    "叔孙通", "刘敬", "随何", "贯高", "丁公", "季布", "季心", "英布", "彭越", "卢绾", "臧荼",
    "武臣", "邵骚", "韩广", "李良", "宋义", "宋襄", "陈平", "周勃", "周亚夫", "周昌", "申屠嘉",
    "冯唐", "赵禹", "王温舒", "王吉", "许负", "任敞", "司马安", "司马尚", "公孙弘", "李蔡",
    "李沮", "公孙贺", "公孙度", "李息", "李广利", "任安", "乐率", "余善", "闽越王郢", "刘次景",
    "乘舒", "刘孝", "刘爽", "刘广", "刘建德", "张释之", "田蚡", "田胜", "田甲", "淳于髡", "淳于意",
    "淳于司马", "扁鹊", "秦越人", "文种", "子西", "子反", "子玉", "观从", "观起", "熊渠", "熊丽",
    "熊艾", "熊胜", "熊杨", "熊延", "熊通", "鬻熊", "季连", "昆吾", "参胡", "彭祖", "会人",
    "曹姓", "季连", "熊珍", "熊绎", "假", "宗", "注", "宫", "昌", "丘", "傅说", "郑国",
    "寒浞", "后羿", "羿", "少康", "后缗", "相", "太康", "仲康", "姬发", "有扈氏", "颛顼", "高阳", "昌意",
    "妺喜", "武丁", "盘庚", "阳甲", "帝乙", "纣王", "周文", "周章", "陈涉", "陈胜", "范增",
    "夏侯婴", "夏侯灶", "夏侯颇", "陈婴", "朱房", "胡武", "周市", "田臧", "召平", "周文", "韩成",
    "赵王歇", "张敖", "田荣", "田横", "田广", "田都", "田安", "项声", "项它", "钟离昧", "季布",
    "彭越", "黥布", "卢绾", "陈豨", "赵利", "樊将军", "王翳", "吕马童", "荣", "德", "阏于", "余",
    "非", "端", "越", "寄", "乘", "舜", "爽", "孝", "无采", "岑娶", "蝉", "鬻", "熊",
    "韩生", "韩商", "靳", "子长", "仲", "季札", "季子", "懿子", "樊伉", "南子", "崔明",
    "申后", "太公", "商均", "常寿过", "郢", "赵尧", "李沮", "李蔡", "赵信", "曹襄",
    "陆终", "卓", "王孙", "黄帝", "秦始皇帝", "夏姬", "禹帝", "外丙", "仲壬", "卷章", "吴回",
    "熊狂", "熊胜", "熊杨", "熊挚", "熊延", "熊萼", "熊仪", "熊霜", "伯霜", "名伯霜", "名仲雪",
    "名叔堪", "名季", "周襄王", "顷襄王", "共王", "康王", "郏敖", "子带", "允常", "庆忌", "子良",
    "宣姜", "卫宣公", "文姜", "齐襄公", "公孙无知", "连称", "连妃", "仲姬", "戎姬", "无诡",
    "郦生", "郦商", "郦寄", "郦况", "郦疥", "赵成", "阎乐", "窦广国", "栾贲", "聂政", "聂荣",
    "严仲子", "侠累", "赵奢", "卫长子", "刘长", "刘礼", "刘濞", "刘安", "刘启", "刘弗陵", "昆莫",
}
PERSON_NAMES = set(PERSON_DISTRACTORS) | MORE_PERSON_NAMES
PERSON_NAMES.update(RELATION_PERSON_NAMES)

PLACE_NAMES = {
    "沛县", "大梁", "外黄", "苦陉", "咸阳", "邯郸", "临淄", "郢都", "姑苏",
    "鸿门", "乌江", "垓下", "巨鹿", "钜鹿", "荥阳", "彭城", "函谷关", "白马津",
    "渔阳", "大泽乡", "范阳", "楚国", "赵国", "魏国", "韩国", "燕国", "齐国",
    "秦国", "蜀地", "汉中", "陈县", "蕲县", "河北", "河南", "信都", "襄国",
    "废丘", "井陉", "会稽", "朝歌", "傅岩", "项城县", "狄", "岐山", "洛邑",
    "宛城", "关中", "三秦", "常山", "太原", "长平", "阏与", "南越", "西域",
    "蓬莱", "阳周", "句注山", "雁门", "定襄", "云中", "陇西", "月氏", "柏人城",
    "成武", "阳城", "户牖乡",
}
POLITICAL_PLACES = {
    "楚国", "赵国", "魏国", "韩国", "燕国", "齐国", "秦国", "南越", "匈奴",
    "汉", "楚", "赵", "魏", "齐", "秦", "周", "晋",
}

TITLE_WORDS = {
    "汉王", "项王", "沛公", "高祖", "高帝", "天子", "皇帝", "皇上", "大王",
    "楚王", "赵王", "魏王", "齐王", "秦王", "怀王", "文帝", "景帝", "孝文帝",
    "孝景帝", "武帝", "汉武帝", "始皇帝", "县令", "太子", "国君", "王", "将军",
    "大将", "大将军", "上将军", "次将", "末将", "丞相", "右丞相", "左丞相",
    "相国", "太尉", "太傅", "太师", "郡守", "国尉", "中尉", "郎中", "中郎将",
    "校尉", "司马", "大夫", "舍人", "侯", "列侯", "王后", "诸侯王", "世子",
    "安平君", "常山王", "南越王", "右贤王", "左贤王", "霸王", "西楚霸王", "武信君",
    "商君", "稷嗣君", "奉阳君", "马服君", "涿侯", "条侯", "冠军侯", "绛侯", "留侯",
    "胶东王", "真定王", "辽东王", "衡山王", "中山王", "广川王", "齐王", "使者", "使臣",
    "御史", "侍臣", "侍从", "门客", "谋士", "辩士", "小臣", "执法官", "部下", "父老",
    "豪杰", "士卒", "宾客", "臣子", "军师", "刺客", "接班人", "陛下", "文信侯",
}

# A few source lines keep a title and a name together as one token.  These
# compounds are treated as title mentions by the relation extractor while the
# older action extractor continues to use the conservative TITLE_WORDS set.
TITLE_NAME_COMPOUNDS = {
    "汉高祖", "汉高帝", "汉武帝", "汉文帝", "汉景帝", "秦始皇", "始皇帝",
    "西楚霸王", "南越武王", "齐威王", "赵孝成王", "赵惠文王", "魏惠王",
}

# Titles which are common in the later chapters but were absent from the
# small hand-written vocabulary above.  This is still an explicit lexicon:
# relation extraction never promotes an arbitrary frequent noun to a title.
TITLE_WORDS.update(
    {
        "文王", "武王", "太后", "王太后", "皇后", "太皇太后", "夫人", "单于", "大单于",
        "国相", "太仆", "长史", "令尹", "少傅", "少师", "御史大夫", "太史令", "大行",
        "卫尉", "治粟都尉", "骑都尉", "骁骑都尉", "郡监", "郡尉", "司空", "上卿", "卿",
        "博士", "大司马", "太仓令", "侍医", "廷尉", "典客", "中大夫", "郎中令", "侍中",
        "庶长", "太傅", "太师", "太子太傅", "太仆令", "列侯", "侯爵", "侯王", "王妃",
        "昭平侯", "昌文侯", "临武侯", "广野君", "淮阴侯", "骠骑将军", "车骑将军",
        "前将军", "后将军", "左将军", "右将军", "中将军", "轻车将军", "贰师将军",
    }
)

KINSHIP_WORDS = {
    "儿子", "女儿", "父亲", "母亲", "父", "母", "子", "女", "兄", "弟",
    "哥哥", "弟弟", "姐姐", "妹妹", "姊妹", "兄弟", "祖先", "后代",
    "叔父", "伯父", "舅舅", "妻", "妻子", "夫", "丈夫", "夫人", "妾", "孙子", "孙女",
    "玄孙", "子孙", "长子", "次子", "三子", "四子", "五子", "六子", "第六子",
    "大儿子", "小儿子", "长女", "小女", "父母", "父子", "子女", "后母", "母后",
    "祖父", "祖母", "曾孙", "外孙", "姊姊", "嫂子", "姐夫", "妯娌", "妻儿",
}

NAMING_MARKERS = {
    "本名", "名叫", "名为", "名曰", "叫", "叫作", "叫做", "又叫", "又称",
    "称", "称为", "称作", "号称", "俗称", "自称", "自称为", "自号", "人称",
    "名字", "姓名", "姓", "名", "取名", "取名为", "起名", "起名叫", "字", "号",
}

# Some lexical labels are useful for the older action/attribute task but do
# not denote a title mention in a relation sentence.  In particular, 将 is
# usually the coverb “to/将”, not the noun “general”.
RELATION_TITLE_BLOCKERS = {"将"}

RELATION_NON_NAME_WORDS = {
    "的", "了", "着", "过", "地", "得", "而", "而且", "则", "乃", "遂", "因", "故",
    "是", "为", "在", "于", "从", "被", "把", "给", "向", "对", "到", "往", "来", "去",
    "以", "与", "和", "及", "或", "且", "若", "如", "所", "者", "其", "此", "彼", "这", "那",
    "他", "她", "它", "自己", "大家", "有人", "人", "们", "有", "无", "不", "未", "已", "也",
    "又", "还", "都", "皆", "才", "即", "便", "就", "仍", "再", "很", "最", "十分", "非常",
    "一个", "一", "两", "三", "四", "五", "几个", "许多", "很多", "各", "每", "某", "谁", "什么",
    "三个", "四个", "五个", "六个", "七个", "八个", "九个", "十个", "两位", "三位", "四位",
    "五位", "六位", "七位", "八位", "九位", "十位", "一名", "两名", "三名", "几名", "数人",
    "数十", "数百", "数千", "数万", "数十万", "数百万", "多人", "几人",
    "说", "曰", "言", "告", "问", "听", "见", "看", "想", "知", "认为", "成为", "成为了",
    "时", "随从", "面前", "中", "上", "下", "外", "内", "那里", "来到", "到达", "知道",
    "听说", "亲自", "去世", "得到", "让", "用", "没有", "担任", "赐给", "封为", "被封",
    "令", "使", "可以", "能够", "不能", "不会", "愿意", "骠骑", "贰师",
    "常山", "淮南", "长沙", "南越", "右贤", "左贤", "下邳", "东园", "角里",
    "从此", "向来", "叫旦", "生", "生下", "回答", "听到", "见到", "曾经", "相会", "邻居",
    "共有", "及其", "部下", "进来", "床前", "暗中", "诋毁", "失传", "劝", "作",
    "后人", "祖先", "后裔", "国人", "齐国人", "宋国人", "各自", "分别", "按照",
    "其为", "不是", "为卫", "在位", "建在", "封在", "封于", "封宪王", "名宫",
    "西", "秦末", "听从", "往见", "起来", "赋予", "劝谏", "劝阻", "随行", "数万", "数千",
    "之命", "之令", "辖区", "忠厚", "大治", "制", "诏", "赐号", "号令",
    "名望", "建在", "建立", "称霸", "长者", "概念", "名称", "称号", "弓", "他人",
    "曲子", "短时间", "千里", "界外", "有病", "总是", "孙子", "玄孙", "里", "弓", "礼",
    "正", "却", "希望", "次子", "长子", "三子", "四子", "五子", "六子",
    "第六子", "孩子", "婴儿", "家", "人家", "贫苦人家", "后代子孙", "有个", "个", "一块", "那块",
    "短时间", "名叫", "名字", "姓名", "取名", "起名", "称号", "名称", "时候", "总是", "当今",
    "奉", "只有", "选择", "排行", "就是", "时期", "继位", "即位", "去世", "死亡", "死", "建立", "始祖",
    "担任", "任", "出任", "说", "曰", "回答", "作", "做", "告", "报告", "见", "听", "听说", "史",
    "封", "立", "拜", "攻打", "进攻", "派", "派遣", "给", "到", "来", "去", "求救", "接受",
    "进入", "跟随", "自立", "称帝", "称王", "成了", "成为", "很", "还", "已经", "正在", "后来", "出于",
    "轵", "邑", "深井", "妻嫂", "匈奴", "乌孙王", "先祖", "圣人", "人类", "贫苦人家",
    "时候", "当时", "后来", "以前", "以后", "之后", "之前", "于是", "因此", "然后", "随后",
    "开始", "终于", "已经", "正在", "当中", "其中", "之中", "方面", "地方", "事情", "事件",
    "国家", "天下", "百姓", "人民", "民众", "君臣", "朝廷", "宫廷", "王朝", "时代", "年", "月", "日",
    "将", "把", "而", "便", "遂", "仍", "皆", "并", "并且",
}

GROUP_WORDS = {
    "军队", "大军", "部队", "部众", "楚军", "汉军", "秦军", "赵军", "燕军",
    "齐军", "骑兵", "骑士", "士兵", "兵卒", "百官", "诸侯", "民众", "百姓",
    "官员", "将领", "大臣", "贵人", "侍卫", "工匠", "姬妾", "士人", "九夷",
    "楚人", "秦人", "汉人", "赵人", "燕人", "军营",
}

ACTION_WORDS = {
    "攻打", "攻下", "攻占", "进攻", "进击", "讨伐", "征讨", "围困", "夺取",
    "平定", "收复", "打败", "击败", "杀死", "杀害", "处死", "斩杀", "起兵",
    "兴兵", "发兵", "率兵", "率领", "进兵", "出兵", "进军", "行刺", "寻找",
    "任命", "追击", "追杀", "袭击", "攻破", "击破", "击杀", "俘虏", "俘获",
    "生擒", "擒获", "捕获", "迎击", "救援", "救出", "会师", "跟随", "投奔",
    "归降", "降服", "起用", "召见", "接见", "拜见", "说服", "娶", "嫁", "封", "拜",
    "立", "派", "派遣", "命令", "委托",
}

PRONOUNS_AND_FUNCTIONS = {
    "我", "你", "您", "吾", "余", "朕", "寡人", "孤", "尔", "汝", "他", "她", "它",
    "他们", "她们", "自己", "有人", "大家", "我们", "你们",
    "这", "这个", "那", "那些", "有个", "一个", "一次", "后来", "于是", "然后",
    "同时", "随后", "现在", "当时", "不久", "可能", "准备", "打算", "愿意", "必须",
    "如果", "虽然", "即使", "因为", "由于", "而且", "并且", "以及", "共同", "一起",
    "前去", "向东", "向西", "向南", "向北", "往北", "进发", "出来", "打开",
    "再次", "一举", "终于", "正在", "已经", "还是", "就此", "不再", "永无休止",
    "十分高兴", "很大", "如何", "为了", "确实", "真的", "强行", "当众", "私下",
    "恰好", "立即", "首先", "一部分", "之中", "以东", "以北", "地区", "土地",
    "天下", "城市", "城池", "各县", "一带", "什么",
}
CONTINUATION_MARKERS = {
    "并", "并且", "又", "还", "便", "就", "随后", "同时", "接着", "于是", "再", "仍",
    "然后",
}
DIRECTION_MARKERS = {
    "向东", "向西", "向南", "向北", "往东", "往西", "往南", "往北", "进军", "进兵",
    "出兵", "率兵",
}

CAUSATIVE_VERBS = {"派", "命令", "派遣", "令", "委托"}
APPOINTMENT_VERBS = {"立", "封", "拜", "任命", "任用", "出任", "聘为"}
ORIGIN_WORDS = {"是", "本是", "原为", "原是"}
CAMPAIGN_RELATIONS = {
    "攻打": "ATTACK",
    "进攻": "ATTACK",
    "进击": "ATTACK",
    "攻下": "CONQUER",
    "攻占": "CONQUER",
    "夺取": "CONQUER",
    "平定": "CONQUER",
    "收复": "CONQUER",
    "讨伐": "PUNISH",
    "征讨": "PUNISH",
    "围困": "SIEGE",
}
COMBAT_RELATIONS = {
    "打败": "DEFEAT",
    "击败": "DEFEAT",
    "杀死": "KILL",
    "杀害": "KILL",
    "处死": "KILL",
    "斩杀": "KILL",
}
LOCATION_EVENTS = {
    "起义", "即位", "称王", "建都", "驻扎", "大败", "自杀", "病逝", "起兵", "诛杀",
}

DISTRACTOR_POOLS = {
    "PERSON": PERSON_DISTRACTORS,
    "PLACE": sorted(PLACE_NAMES),
    "TITLE": sorted(TITLE_WORDS),
    "GROUP": sorted(GROUP_WORDS),
}


def load_attribute_lexicon() -> Dict[str, Set[str]]:
    """Load previously learned lexical labels as an optional extra signal."""

    labels: Dict[str, Set[str]] = {}
    if not ATTRIBUTE_MANIFEST.exists():
        return labels
    try:
        payload = json.loads(ATTRIBUTE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return labels
    for word, values in payload.get("word_labels", {}).items():
        labels[str(word)] = {str(value) for value in values}
    return labels


ATTRIBUTE_LEXICON = load_attribute_lexicon()


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def split_for_source(source_line: int) -> str:
    bucket = stable_seed("split", source_line) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def _fallback_name_hint(tokens: Sequence[str], index: int) -> bool:
    """Recognize names in the small set of explicit relation slots.

    The source is already whitespace-tokenized.  When jieba is unavailable,
    this keeps the builder reproducible and still recognizes a name after a
    title, naming marker, or kinship marker without guessing from frequency.
    """

    token = tokens[index]
    if (
        token in PUNCTUATION
        or token in PRONOUNS_AND_FUNCTIONS
        or token in GROUP_WORDS
        or token in ACTION_WORDS
        or token in KINSHIP_WORDS
        or token in NAMING_MARKERS
        or token in PLACE_NAMES
        or token in RELATION_NON_NAME_WORDS
    ):
        return False
    previous = tokens[index - 1] if index > 0 else ""
    previous_two = tokens[index - 2] if index > 1 else ""
    following = tokens[index + 1] if index + 1 < len(tokens) else ""
    if (
        (previous in TITLE_WORDS or previous in TITLE_NAME_COMPOUNDS)
        and previous not in RELATION_TITLE_BLOCKERS
    ):
        return True
    if (
        (following in TITLE_WORDS or following in TITLE_NAME_COMPOUNDS)
        and following not in RELATION_TITLE_BLOCKERS
    ):
        return True
    if previous in NAMING_MARKERS or previous_two in NAMING_MARKERS:
        return True
    if previous in KINSHIP_WORDS:
        return True
    # A name in an explicit action or copular slot is still useful when the
    # optional jieba tagger is unavailable.  Keep this local and lexical so
    # frequency cannot turn arbitrary prose into a person label.
    if previous in ACTION_WORDS or following in ACTION_WORDS:
        return True
    if previous in {"为", "是", "被", "曰", "言", "告", "问", "谓", "使"}:
        return True
    return False


def pos_tags(line: str) -> Dict[str, str]:
    if pseg is not None:
        clean_line = line.replace(" ", "")
        return {word: flag for word, flag in pseg.cut(clean_line)}

    tokens = line.split()
    tags: Dict[str, str] = {}
    for index, token in enumerate(tokens):
        if token in PLACE_NAMES or token.endswith(("国", "县", "郡", "城", "关")):
            tags[token] = "ns"
        elif token in PERSON_NAMES or _fallback_name_hint(tokens, index):
            tags[token] = "nr"
        elif token in ACTION_WORDS:
            tags[token] = "v"
        elif token in PUNCTUATION:
            tags[token] = "w"
        else:
            tags[token] = "n"
    return tags


def token_attributes(token: str, default: Optional[str] = None) -> List[str]:
    """Return coarse multi-label attributes for one target token."""

    labels = set(ATTRIBUTE_LEXICON.get(token, set()))
    if token in GROUP_WORDS:
        labels.discard("PERSON")
        labels.discard("TITLE")
        labels.add("GROUP")
    elif token in TITLE_WORDS or token in TITLE_NAME_COMPOUNDS:
        labels.add("TITLE")
    elif token in PERSON_NAMES:
        labels.add("PERSON")
    elif token in PLACE_NAMES or token.endswith(("国", "县", "郡", "城", "关")):
        labels.add("PLACE")
    elif token in ACTION_WORDS:
        labels.add("ACTION")

    if re.fullmatch(r"[0-9０-９]+", token):
        labels.add("NUMBER")

    semantic_labels = {
        "PERSON", "PLACE", "TITLE", "GROUP", "ORG", "ACTION", "NUMBER",
    }
    if labels & semantic_labels and "PUNCT" not in labels:
        labels.add("ENTITY")
    if not labels and default:
        labels.add(default)
        if default not in {"ACTION", "NUMBER"}:
            labels.add("ENTITY")
    if not labels:
        labels.add("ENTITY")

    if "GROUP" in labels:
        labels.discard("PERSON")
        labels.discard("TITLE")
        labels.add("ENTITY")
    return sorted(labels)


def is_place_like(token: str, tag_map: Dict[str, str]) -> bool:
    attrs = set(token_attributes(token))
    return "PLACE" in attrs or tag_map.get(token, "").startswith("ns")


def is_person_like(token: str, tag_map: Dict[str, str]) -> bool:
    if token in PRONOUNS_AND_FUNCTIONS or token in GROUP_WORDS:
        return False
    attrs = set(token_attributes(token))
    if "PERSON" in attrs or "TITLE" in attrs:
        return True
    flag = tag_map.get(token, "")
    return flag.startswith(("nr", "nrf")) and "PLACE" not in attrs


def is_title_like(token: str) -> bool:
    """Return whether a token is a historical title mention."""

    if token in RELATION_TITLE_BLOCKERS:
        return False
    # Do not promote stale weak labels such as 部下/父老 to titles.  The
    # relation extractor needs a small, explicit title lexicon; the broader
    # attribute labels remain available to the original task.
    return token in TITLE_WORDS or token in TITLE_NAME_COMPOUNDS


def _is_atomic_relation_name(token: str) -> bool:
    """Reject glued prose/function phrases from one-token name slots."""

    if (
        token in RELATION_NON_NAME_WORDS
        or token in PRONOUNS_AND_FUNCTIONS
        or token in PUNCTUATION
        or token in NAMING_MARKERS
        or token in KINSHIP_WORDS
        or token in ACTION_WORDS
        or token in GROUP_WORDS
        or token in PLACE_NAMES
        or token.endswith(("国", "县", "郡", "城", "关"))
    ):
        return False
    lexical_labels = ATTRIBUTE_LEXICON.get(token, set())
    if lexical_labels & {"FUNCTION", "PUNCT", "NUMBER", "TIME", "PLACE", "ACTION", "GROUP"}:
        return False
    if not (1 <= len(token) <= 4) or any(char.isdigit() for char in token):
        return False
    if any(char in token for char in "一二三四五六七八九十百千万亿两"):
        return False
    if any(
        marker and (token.startswith(marker) or token.endswith(marker))
        for marker in (
            "叫", "名叫", "叫作", "叫做", "又称", "称为", "称作", "为", "的", "了",
        )
    ):
        return False
    if len(token) > 2 and any(
        char in token for char in "的了着过地得而则乃遂因故是为在于从被把给向对到往来去以与和及或且若如所者其此彼这那他她我你您有无不未已也又还都皆才即便就仍再"
    ):
        return False
    if any(token.startswith(prefix) and token != prefix for prefix in TITLE_WORDS):
        return False
    return True


def is_named_entity_candidate(
    token: str,
    tag_map: Dict[str, str],
    *,
    allow_title: bool = True,
) -> bool:
    """Recognize an explicit person/title slot without accepting prose words."""

    if (
        not token
        or token in PUNCTUATION
        or token in PRONOUNS_AND_FUNCTIONS
        or token in GROUP_WORDS
        or token in ACTION_WORDS
        or token in KINSHIP_WORDS
        or token in NAMING_MARKERS
        or token in PLACE_NAMES
        or token.endswith(("国", "县", "郡", "城", "关"))
        or token in RELATION_NON_NAME_WORDS
        or token in RELATION_TITLE_BLOCKERS
    ):
        return False
    if allow_title and is_title_like(token):
        return True
    attrs = set(token_attributes(token))
    if "PERSON" in attrs:
        return True
    flag = tag_map.get(token, "")
    return (
        flag.startswith(("nr", "nrf"))
        and "PLACE" not in attrs
        and _is_atomic_relation_name(token)
    )


def is_person_entity_candidate(token: str, tag_map: Dict[str, str]) -> bool:
    """Recognize a person/name slot while excluding a standalone title."""

    # The older attribute lexicon contains a few political place names as
    # PERSON examples (for example 郑国).  In an explicit person relation,
    # the place suffix is stronger evidence than that stale label.
    if token in PLACE_NAMES or token.endswith(("国", "县", "郡", "城", "关")):
        return False
    if not is_named_entity_candidate(token, tag_map):
        return False
    if is_title_like(token):
        return False
    if token in PERSON_NAMES:
        return True
    attrs = set(token_attributes(token))
    if "PERSON" in attrs:
        return True
    if pseg is not None and tag_map.get(token, "").startswith(("nr", "nrf")):
        return True
    return False


def is_actor_candidate(token: str, tag_map: Dict[str, str]) -> bool:
    if token in PUNCTUATION or token in PRONOUNS_AND_FUNCTIONS:
        return False
    if token in ACTION_WORDS or not token:
        return False
    attrs = set(token_attributes(token))
    if attrs & {"PERSON", "TITLE", "GROUP"}:
        return True
    if token in POLITICAL_PLACES:
        return True
    flag = tag_map.get(token, "")
    return flag.startswith(("nr", "nrf")) and "PLACE" not in attrs


def candidate_span(
    tokens: Sequence[str], index: int, tag_map: Dict[str, str]
) -> Tuple[int, int, str]:
    """Expand an obvious title-name span such as 太子 丹."""

    start = index
    end = index + 1
    if (
        index > 0
        and tokens[index - 1] in TITLE_WORDS
        and tokens[index - 1] not in GROUP_WORDS
    ):
        start = index - 1
    elif (
        index + 1 < len(tokens)
        and tokens[index] in TITLE_WORDS
        and is_person_like(tokens[index + 1], tag_map)
    ):
        end = index + 2
    return start, end, " ".join(tokens[start:end])


def find_subject(
    tokens: Sequence[str], verb_index: int, tag_map: Dict[str, str]
) -> Optional[Dict[str, object]]:
    """Find a conservative local agent; pronouns are never resolved."""

    sentence_start = 0
    for index in range(verb_index - 1, -1, -1):
        if tokens[index] in SENTENCE_END:
            sentence_start = index + 1
            break

    comma_start = sentence_start
    for index in range(verb_index - 1, sentence_start - 1, -1):
        if tokens[index] in COMMA_MARKS:
            comma_start = index + 1
            break

    local_candidates = [
        index
        for index in range(comma_start, verb_index)
        if is_actor_candidate(tokens[index], tag_map)
    ]
    local_prefix = tokens[comma_start:verb_index]
    marker_continuation = bool(local_prefix) and (
        local_prefix[0] in CONTINUATION_MARKERS
        or local_prefix[0] in DIRECTION_MARKERS
    )

    if local_candidates and not marker_continuation:
        index = local_candidates[-1]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "explicit_nearest",
            "attributes": token_attributes(tokens[index]),
        }

    prior_candidates = [
        index
        for index in range(sentence_start, comma_start)
        if is_actor_candidate(tokens[index], tag_map)
    ]
    if marker_continuation and prior_candidates:
        index = prior_candidates[0]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "continued_sentence_actor",
            "attributes": token_attributes(tokens[index]),
        }

    if local_candidates:
        index = local_candidates[-1]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "explicit_after_marker",
            "attributes": token_attributes(tokens[index]),
        }

    if local_prefix and local_prefix[0] in DIRECTION_MARKERS and prior_candidates:
        index = prior_candidates[0]
        start, end, text = candidate_span(tokens, index, tag_map)
        return {
            "text": text,
            "start": start,
            "end": end,
            "mode": "continued_direction_actor",
            "attributes": token_attributes(tokens[index]),
        }
    return None


def find_causative_target(
    tokens: Sequence[str], verb_index: int, tag_map: Dict[str, str]
) -> Optional[int]:
    """Find a person before a generic role, e.g. 命令 骑兵 将领 灌婴."""

    end = min(len(tokens), verb_index + 7)
    person_candidates: List[int] = []
    fallback_candidates: List[int] = []
    for index in range(verb_index + 1, end):
        token = tokens[index]
        if token in PUNCTUATION:
            break
        if token in {"自己", "他们", "大家", "有人", "人", "们"}:
            continue
        if is_person_like(token, tag_map):
            person_candidates.append(index)
        elif set(token_attributes(token)) & {"TITLE", "GROUP"}:
            fallback_candidates.append(index)

    if person_candidates:
        return person_candidates[0]
    if fallback_candidates:
        return fallback_candidates[0]
    return None


def find_appointment_parts(
    tokens: Sequence[str], verb_index: int, tag_map: Dict[str, str]
) -> Optional[Tuple[int, int]]:
    """Return person and single-token role positions for an appointment."""

    for_index = None
    for index in range(verb_index + 2, min(len(tokens), verb_index + 7)):
        if tokens[index] in SENTENCE_END or tokens[index] in {"，", "；", "："}:
            break
        if tokens[index] == "为":
            for_index = index
            break
    if for_index is None:
        return None

    person_candidates = [
        index
        for index in range(verb_index + 1, for_index)
        if is_person_like(tokens[index], tag_map)
    ]
    if person_candidates:
        person_index = person_candidates[-1]
    else:
        fallback = [
            index
            for index in range(verb_index + 1, for_index)
            if tokens[index] not in TITLE_WORDS
            and tokens[index] not in PRONOUNS_AND_FUNCTIONS
            and tokens[index] not in PUNCTUATION
            and tokens[index] not in RELATION_NON_NAME_WORDS
            and tokens[index] not in KINSHIP_WORDS
            and tokens[index] not in GROUP_WORDS
            and not re.fullmatch(r"[0-9０-９]+", tokens[index])
            and not is_place_like(tokens[index], tag_map)
        ]
        if not fallback:
            return None
        person_index = fallback[-1]

    role_index = for_index + 1
    if role_index >= len(tokens) or tokens[role_index] in PUNCTUATION:
        return person_index, -1
    # Do not truncate multi-token roles such as 常山 王 or 上 将军.
    if (
        role_index + 1 < len(tokens)
        and tokens[role_index + 1] not in PUNCTUATION
        and tokens[role_index] in {"上", "下", "左", "右", "中"}
    ):
        return person_index, -1
    return person_index, role_index


def mask_at(tokens: Sequence[str], start: int, end: int) -> List[str]:
    return list(tokens[:start]) + ["[MASK]"] + list(tokens[end:])


def make_long_context(
    sentences: Sequence[str],
    line_index: int,
    target_index: int,
    window: int = 2,
) -> Tuple[str, str, int]:
    start_line = max(0, line_index - window)
    end_line = min(len(sentences), line_index + window + 1)
    long_tokens: List[str] = []
    absolute_target = -1
    for index in range(start_line, end_line):
        line_tokens = sentences[index].split()
        if index == line_index:
            absolute_target = len(long_tokens) + target_index
        long_tokens.extend(line_tokens)
    if absolute_target < 0:
        raise ValueError("target line was not included in long context")
    return (
        " ".join(long_tokens),
        " ".join(mask_at(long_tokens, absolute_target, absolute_target + 1)),
        absolute_target,
    )


def answer_is_visible(answer: str, masked_text: str) -> bool:
    return answer in masked_text.split()


def choose_distractors(
    answer: str,
    attributes: Sequence[str],
    exclude: Iterable[str],
    source_line: int,
    target_index: int,
    long_context: str,
    k: int = 5,
) -> List[str]:
    primary = next(
        (
            label
            for label in ("PERSON", "PLACE", "TITLE", "GROUP")
            if label in attributes
        ),
        "PERSON",
    )
    pool = DISTRACTOR_POOLS.get(primary, PERSON_DISTRACTORS)
    excluded = set(exclude)
    excluded.add(answer)
    evidence_tokens = set(long_context.split())
    candidates = [
        value
        for value in pool
        if value not in excluded and value not in evidence_tokens
    ]
    rng = random.Random(stable_seed("distractors", source_line, target_index, answer))
    rng.shuffle(candidates)
    return candidates[:k]


def make_fact(
    *,
    sentences: Sequence[str],
    line_index: int,
    tokens: Sequence[str],
    target_index: int,
    target_role: str,
    subject: Dict[str, object],
    relation_type: str,
    grammar: str,
    relation_surface: str,
    relation_evidence: str,
    object_value: str,
    object_attributes: Sequence[str],
    default_answer_attribute: Optional[str] = None,
    fact_group_id: Optional[str] = None,
    confidence: Optional[float] = None,
    confidence_basis: str = "source_span_verified+pattern_match",
) -> Optional[Dict]:
    if target_index < 0 or target_index >= len(tokens):
        return None
    answer = tokens[target_index]
    if answer in PUNCTUATION or answer in PRONOUNS_AND_FUNCTIONS:
        return None
    answer_attributes = token_attributes(answer, default_answer_attribute)
    source_masked = " ".join(mask_at(tokens, target_index, target_index + 1))
    if answer_is_visible(answer, source_masked):
        # Repeated answers are intentionally excluded from one-token MLM data.
        return None

    long_context, long_masked, long_target = make_long_context(
        sentences, line_index, target_index
    )
    variants: List[Dict] = []
    if not answer_is_visible(answer, long_masked):
        variants.append(
            {
                "type": "long_context_masked",
                "context": long_context,
                "masked_text": long_masked,
                "answer": answer,
                "target_role": target_role,
                "answer_attributes": list(answer_attributes),
                "target_span": [long_target, long_target + 1],
            }
        )

    source_line = line_index + 1
    subject_text = str(subject["text"])
    subject_attrs = list(subject.get("attributes", []))
    fact_group_id = fact_group_id or (
        f"line_{source_line}_{relation_type}_{target_index}"
    )
    distractors = choose_distractors(
        answer,
        answer_attributes,
        {answer, subject_text, object_value, *tokens},
        source_line,
        target_index,
        long_context,
    )
    return {
        "fact_group_id": fact_group_id,
        "source_line": source_line,
        "context": " ".join(tokens),
        "long_context": long_context,
        "masked_text": source_masked,
        "answer": answer,
        "target_role": target_role,
        "target_span": [target_index, target_index + 1],
        "subject": subject_text,
        "subject_attributes": subject_attrs,
        "relation": relation_type,
        "relation_type": relation_type,
        "relation_surface": relation_surface,
        "relation_evidence": relation_evidence,
        "object": object_value,
        "object_attributes": list(object_attributes),
        "grammar": grammar,
        "answer_attributes": list(answer_attributes),
        "confidence": (
            float(confidence)
            if confidence is not None
            else (0.96 if subject.get("mode") == "explicit_nearest" else 0.91)
        ),
        "confidence_basis": confidence_basis,
        "subject_resolution": subject.get("mode"),
        "distractors": distractors,
        "variants": variants,
        "split": split_for_source(source_line),
    }


def source_evidence(tokens: Sequence[str], start: int, end: int) -> str:
    return " ".join(tokens[max(0, start - 2) : min(len(tokens), end + 6)])


def _relation_anchor_before(
    tokens: Sequence[str],
    index: int,
    tag_map: Dict[str, str],
    max_distance: int = 8,
    ignore_discourse_blockers: bool = False,
) -> Optional[int]:
    """Find an explicit named anchor for a naming expression."""

    lower = max(0, index - max_distance)
    for candidate_index in range(index - 1, lower - 1, -1):
        token = tokens[candidate_index]
        if token in SENTENCE_END:
            break
        if token in COMMA_MARKS:
            if ignore_discourse_blockers and token in {"，", ","}:
                continue
            break
        if token in KINSHIP_WORDS or token in {
            "有个", "一个", "人", "国号", "称号", "国人", "后裔", "祖先", "后代",
            "从此", "后来", "于是", "所以", "因为", "因此", "然后", "随后",
        }:
            if ignore_discourse_blockers and token in {
                "后来", "于是", "所以", "因此", "然后", "随后",
            }:
                continue
            return None
        if token in PUNCTUATION or token in {"的", "其", "他", "她", "这", "那"}:
            continue
        if is_named_entity_candidate(token, tag_map) or (
            candidate_index == index - 1 and _is_atomic_relation_name(token)
        ):
            return candidate_index
    return None


def _naming_target_indices(
    tokens: Sequence[str],
    marker_index: int,
    tag_map: Dict[str, str],
    allow_multiple: bool,
    allow_title_prefix: bool = False,
) -> List[int]:
    """Collect named targets after a name/title marker."""

    targets: List[int] = []
    upper = min(len(tokens), marker_index + 9)
    for index in range(marker_index + 1, upper):
        token = tokens[index]
        if token in SENTENCE_END:
            break
        if token in {"，", ",", "；", ";"}:
            if allow_multiple and token in {"，", ","}:
                lookahead = index + 1
                while lookahead < upper and tokens[lookahead] in {
                    "、", "或", "和", "与", "及", "也", "又",
                }:
                    lookahead += 1
                if lookahead < upper and is_named_entity_candidate(
                    tokens[lookahead], tag_map
                ):
                    continue
            break
        if token in PUNCTUATION or token in {"其", "他", "她", "自己", "为", "作", "是"}:
            continue
        if token in {"、", "或", "和", "与", "及", "也", "又"}:
            if not allow_multiple:
                break
            continue
        if is_named_entity_candidate(token, tag_map):
            if targets and index == targets[-1] + 1:
                if not (
                    allow_title_prefix
                    and len(targets) == 1
                    and is_title_like(tokens[targets[0]])
                    and is_person_entity_candidate(token, tag_map)
                ):
                    break
            elif (
                not targets
                and not allow_title_prefix
                and index + 1 < upper
                and is_named_entity_candidate(tokens[index + 1], tag_map)
            ):
                # The source tokenizer sometimes splits one historical name
                # into adjacent one-token pieces (e.g. 鬻 熊).  The schema
                # requires an exact one-token answer, so omit the ambiguous
                # partial span instead of teaching a false relation.
                return []
            targets.append(index)
            if not allow_multiple:
                break
            continue
        # Do not jump over an unknown first token to a later prose word.  The
        # one-token schema cannot safely recover a glued multi-token name;
        # accepting the later word creates examples such as 自称 沛公刚 起义
        # -> 起义.
        break
    return targets


def _link_relation_types(left: str, right: str) -> Tuple[str, str, str, str]:
    """Choose directional relation labels for two linked entities."""

    left_title = is_title_like(left)
    right_title = is_title_like(right)
    if left_title and not right_title:
        return "TITLE_OF", "HAS_TITLE", "TITLE_LINK", "title_to_person"
    if right_title and not left_title:
        return "HAS_TITLE", "TITLE_OF", "TITLE_LINK", "person_to_title"
    return "ALIAS_OF", "ALIAS_OF", "ALIAS", "alias"


def _make_link_facts(
    *,
    sentences: Sequence[str],
    line_index: int,
    tokens: Sequence[str],
    left_index: int,
    right_index: int,
    relation_surface: str,
    relation_evidence: str,
    confidence: float = 0.97,
    relation_types: Optional[Tuple[str, str]] = None,
    grammar_override: Optional[str] = None,
    directions: Optional[Tuple[str, str]] = None,
    left_default_override: Optional[str] = None,
    right_default_override: Optional[str] = None,
) -> List[Dict]:
    """Create both directions of an explicit title/name/entity link."""

    if left_index == right_index:
        return []
    left = tokens[left_index]
    right = tokens[right_index]
    if not left or not right:
        return []
    inferred_relation_type, inferred_inverse_type, inferred_grammar, inferred_direction = _link_relation_types(left, right)
    if relation_types is None:
        relation_type, inverse_type = inferred_relation_type, inferred_inverse_type
    else:
        relation_type, inverse_type = relation_types
    grammar = grammar_override or inferred_grammar
    if directions is None:
        forward_direction = inferred_direction
        inverse_direction = (
            "person_to_title" if inferred_direction == "title_to_person" else
            "title_to_person" if inferred_direction == "person_to_title" else "alias"
        )
    else:
        forward_direction, inverse_direction = directions
    left_default = left_default_override or ("TITLE" if is_title_like(left) else "PERSON")
    right_default = right_default_override or ("TITLE" if is_title_like(right) else "PERSON")
    group_id = f"line_{line_index + 1}_{grammar}_{left_index}_{right_index}"
    facts: List[Dict] = []

    forward = make_fact(
        sentences=sentences,
        line_index=line_index,
        tokens=tokens,
        target_index=right_index,
        target_role="object",
        subject={
            "text": left,
            "attributes": token_attributes(left, left_default),
            "mode": "explicit_relation",
        },
        relation_type=relation_type,
        grammar=grammar,
        relation_surface=relation_surface,
        relation_evidence=relation_evidence,
        object_value=right,
        object_attributes=token_attributes(right, right_default),
        default_answer_attribute=right_default,
        fact_group_id=group_id,
        confidence=confidence,
        confidence_basis="source_span_verified+explicit_relation_pattern",
    )
    if forward is not None:
        forward["relation_direction"] = forward_direction
        forward["linked_entity"] = {"left": left, "right": right}
        facts.append(forward)

    inverse = make_fact(
        sentences=sentences,
        line_index=line_index,
        tokens=tokens,
        target_index=left_index,
        target_role="object",
        subject={
            "text": right,
            "attributes": token_attributes(right, right_default),
            "mode": "explicit_relation",
        },
        relation_type=inverse_type,
        grammar=grammar,
        relation_surface=relation_surface,
        relation_evidence=relation_evidence,
        object_value=left,
        object_attributes=token_attributes(left, left_default),
        default_answer_attribute=left_default,
        fact_group_id=group_id,
        confidence=confidence,
        confidence_basis="source_span_verified+explicit_relation_pattern",
    )
    if inverse is not None:
        inverse["relation_direction"] = inverse_direction
        inverse["linked_entity"] = {"left": left, "right": right}
        facts.append(inverse)
    return facts


def _is_single_title_role(tokens: Sequence[str], index: int) -> bool:
    """Reject a partial answer from a multi-token title such as 常山 王."""

    if index < 0 or index >= len(tokens) or not is_title_like(tokens[index]):
        return False
    if index > 0 and tokens[index - 1] in {
        "常山", "淮南", "长沙", "南越", "右贤", "左贤", "绛", "胶东", "中山",
        "真定", "辽东", "衡山", "广川", "涿", "冠军", "舞阳", "淮阴", "伏波",
        "楼船", "车骑", "骠骑", "轻车", "前", "后", "左", "右", "中",
    }:
        return False
    if index + 1 < len(tokens) and tokens[index + 1] in {
        "王", "侯", "君", "公", "帝", "后", "将军", "相",
    }:
        return False
    return True


def _is_apposition_person_candidate(
    tokens: Sequence[str], index: int, title_index: int, tag_map: Dict[str, str]
) -> bool:
    """Recognize the name side of an adjacent title/name apposition."""

    if index < 0 or index >= len(tokens) or index == title_index:
        return False
    if index == title_index + 1:
        # A title after a reporting/action verb is often an object phrase,
        # not a title/name apposition: “说 秦王 李斯” and “打 秦王 宫”.
        if title_index > 0 and tokens[title_index - 1] in {
            "说", "曰", "言", "告", "问", "谓", "认为", "听说", "所说",
            "打", "攻打", "进攻", "攻下", "攻占", "夺取", "用", "以", "在", "于",
        }:
            return False
    elif index == title_index - 1:
        # Caption fragments and possessive phrases are common in the source
        # corpus.  In “墓地 宫 南越王” and “的 周勃 将军”, the adjacent
        # token is not the holder of the title.
        before_name = tokens[index - 1] if index > 0 else ""
        if before_name in {
            "的", "墓地", "画像", "画像石", "图", "出土", "博物馆", "石雕",
            "铜器", "故地", "故里", "本纪", "列传",
        }:
            return False
        if title_index + 1 < len(tokens) and tokens[title_index + 1] in {
            "在", "于", "时", "的", "和", "与", "及", "被", "把", "将",
        }:
            return False
    if is_person_entity_candidate(tokens[index], tag_map):
        return True
    # Apposition is a high-value relation and is easy to poison with a
    # neighboring verb (e.g. “文王 奉” or “天子 派”).  Unknown names must first
    # be added to the explicit relation lexicon; arbitrary one-token nouns are
    # not promoted merely because they touch a title.
    return False


def _title_role_after_appointment(
    tokens: Sequence[str], verb_index: int
) -> Optional[int]:
    """Return a one-token title after active or passive appointment wording."""

    verb = tokens[verb_index]
    if verb.endswith("为") and verb not in {"以为", "认为"}:
        role_index = verb_index + 1
        if (
            role_index + 2 < len(tokens)
            and is_title_like(tokens[role_index])
            and tokens[role_index + 1] == "的"
            and _is_single_title_role(tokens, role_index + 2)
        ):
            role_index += 2
        return role_index if _is_single_title_role(tokens, role_index) else None
    for index in range(verb_index + 1, min(len(tokens), verb_index + 7)):
        if tokens[index] in SENTENCE_END or tokens[index] in {"，", "；", "："}:
            break
        if tokens[index] != "为":
            continue
        role_index = index + 1
        if (
            role_index + 2 < len(tokens)
            and is_title_like(tokens[role_index])
            and tokens[role_index + 1] == "的"
            and _is_single_title_role(tokens, role_index + 2)
        ):
            role_index += 2
        return role_index if _is_single_title_role(tokens, role_index) else None
    return None


def _extract_title_and_alias_facts(
    sentences: Sequence[str],
    line_index: int,
    tokens: Sequence[str],
    tag_map: Dict[str, str],
) -> List[Dict]:
    """Extract explicit title/person and name/alias links from one line."""

    facts: List[Dict] = []
    seen_pairs: Set[Tuple[int, int, str]] = set()

    def add_pair(
        left_index: int,
        right_index: int,
        surface: str,
        start: int,
        end: int,
        *,
        explicit_marker: bool = False,
    ) -> None:
        if left_index == right_index:
            return
        left = tokens[left_index]
        right = tokens[right_index]
        left_is_named = is_named_entity_candidate(left, tag_map) or (
            explicit_marker and _is_atomic_relation_name(left)
        )
        right_is_named = is_named_entity_candidate(right, tag_map) or (
            explicit_marker and _is_atomic_relation_name(right)
        )
        if not (left_is_named and right_is_named):
            return
        if explicit_marker:
            # A naming marker must connect a person/title to another
            # person/title (or a short explicit name), never an abstract noun
            # such as 制/诏 or a quantity such as 数万.
            if not (
                is_title_like(left)
                or is_person_entity_candidate(left, tag_map)
            ):
                return
            if not (
                is_title_like(right)
                or is_person_entity_candidate(right, tag_map)
                or (
                    surface
                    in {
                        "本名", "名叫", "名为", "名曰", "又叫", "又称", "叫", "叫作", "叫做",
                        "称作", "自称", "自称为", "自号", "名字", "字", "起名", "起名叫",
                        "取名", "取名为",
                    }
                    and _is_atomic_relation_name(right)
                )
            ):
                return
        key = (left_index, right_index, surface)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        facts.extend(
            _make_link_facts(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                left_index=left_index,
                right_index=right_index,
                relation_surface=surface,
                relation_evidence=source_evidence(tokens, start, end),
            )
        )

    # Appositions such as 汉王 刘邦, 高祖 成汤, and 张耳 常山王.
    for index, token in enumerate(tokens):
        if not is_title_like(token):
            continue
        if not _is_single_title_role(tokens, index):
            continue
        # When both sides happen to look like names, prefer the following
        # token.  This avoids a heading such as “曹参 汉高祖 刘邦” being
        # recorded as if 曹参 were the holder of 汉高祖.
        if index + 1 < len(tokens) and _is_apposition_person_candidate(
            tokens, index + 1, index, tag_map
        ):
            add_pair(index, index + 1, "title_name_apposition", index, index + 2)
        elif index > 0 and _is_apposition_person_candidate(
            tokens, index - 1, index, tag_map
        ):
            add_pair(index, index - 1, "title_name_apposition", index - 1, index + 1)

    # Explicit naming/alias expressions.  Multiple targets are allowed only
    # for list-like markers such as 又称 ... 、 ... 或 ... .
    for marker_index, marker in enumerate(tokens):
        if marker not in NAMING_MARKERS or marker in {"姓", "名", "名字", "姓名"}:
            continue
        if marker == "叫":
            previous = tokens[marker_index - 1] if marker_index else ""
            if previous not in {"又", "名字", "姓名", "名", "起名", "取名"}:
                continue
        previous = tokens[marker_index - 1] if marker_index else ""
        if marker in {"字", "号"}:
            # A short backward window prevents an image caption such as
            # “冯唐 … ‘卫’ 字 瓦当” from becoming a person alias.
            anchor_distance = 3
        elif marker in {"叫", "起名叫", "取名为"} and (
            marker != "叫" or previous in {"起名", "取名"}
        ):
            # “孔子 … 起名叫 丘” may have a descriptive clause between the
            # subject and the marker, but it remains within this sentence.
            anchor_distance = 16
        else:
            anchor_distance = 8
        left_index = _relation_anchor_before(
            tokens,
            marker_index,
            tag_map,
            max_distance=anchor_distance,
            ignore_discourse_blockers=marker in {"起名叫", "取名为", "字", "号"},
        )
        if left_index is None:
            continue
        if marker in {"取名", "取名为"} and any(
            token in {"地方", "地名", "地点", "所在之地"}
            for token in tokens[left_index + 1 : marker_index]
        ):
            # “把黄帝升天的地方取名为鼎湖” is a place-naming event, not
            # an alias between two people.  Place events are intentionally
            # left to the location/event extractor.
            continue
        allow_multiple = marker in {
            "又称", "又叫", "称为", "称作", "俗称", "称", "自称", "自称为",
        }
        targets = _naming_target_indices(
            tokens,
            marker_index,
            tag_map,
            allow_multiple,
            allow_title_prefix=marker in {"自称", "自称为"},
        )

        # “武王 自称 太子 姬发” contains a title prefix before the actual
        # name.  Link both title→person and subject→person, while avoiding the
        # misleading title↔title alias that the generic path would create.
        if (
            marker in {"自称", "自称为"}
            and len(targets) >= 2
            and targets[1] == targets[0] + 1
            and is_title_like(tokens[targets[0]])
            and is_person_entity_candidate(tokens[targets[1]], tag_map)
        ):
            add_pair(
                targets[0],
                targets[1],
                marker,
                targets[0],
                targets[1] + 1,
                explicit_marker=True,
            )
            add_pair(
                left_index,
                targets[1],
                marker,
                left_index,
                targets[1] + 1,
                explicit_marker=True,
            )
            targets = targets[1:]

        for target_index in targets:
            add_pair(
                left_index,
                target_index,
                marker,
                left_index,
                target_index + 1,
                explicit_marker=True,
            )

    # A common form is “人物 的 名字 叫 名称”, where the nearest token before
    # 叫 is the noun 名字 rather than the entity itself.
    for name_index, token in enumerate(tokens):
        if token not in {"名字", "姓名"} or name_index < 2:
            continue
        next_token = tokens[name_index + 1]
        if next_token not in {"叫", "是", "为"}:
            continue
        owner_index = name_index - 2 if tokens[name_index - 1] == "的" else name_index - 1
        if owner_index < 0 or not is_named_entity_candidate(tokens[owner_index], tag_map):
            continue
        targets = _naming_target_indices(tokens, name_index + 1, tag_map, False)
        for target_index in targets:
            add_pair(
                owner_index,
                target_index,
                "名字",
                owner_index,
                target_index + 1,
                explicit_marker=True,
            )

    return facts


KINSHIP_RELATION_SPECS = {
    # Relation types and directions are written from the left entity to the
    # right entity.  The inverse row is emitted by _make_link_facts.
    "儿子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "女儿": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "女": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "孙子": ("GRANDPARENT_OF", "GRANDCHILD_OF", "grandparent_to_grandchild", "grandchild_to_grandparent"),
    "孙女": ("GRANDPARENT_OF", "GRANDCHILD_OF", "grandparent_to_grandchild", "grandchild_to_grandparent"),
    "曾孙": ("GRANDPARENT_OF", "GRANDCHILD_OF", "grandparent_to_grandchild", "grandchild_to_grandparent"),
    "玄孙": ("GRANDPARENT_OF", "GRANDCHILD_OF", "grandparent_to_grandchild", "grandchild_to_grandparent"),
    # Rank-qualified children still express the same parent/child edge.  The
    # rank remains in relation_surface so the model can distinguish “长子”
    # from an unqualified “儿子” when the context supports it.
    "长子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "次子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "三子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "四子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "五子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "六子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "第六子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "大儿子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "小儿子": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "长女": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "小女": ("PARENT_OF", "CHILD_OF", "parent_to_child", "child_to_parent"),
    "父亲": ("CHILD_OF", "PARENT_OF", "child_to_parent", "parent_to_child"),
    "母亲": ("CHILD_OF", "PARENT_OF", "child_to_parent", "parent_to_child"),
    "父": ("CHILD_OF", "PARENT_OF", "child_to_parent", "parent_to_child"),
    "母": ("CHILD_OF", "PARENT_OF", "child_to_parent", "parent_to_child"),
    "后母": ("CHILD_OF", "PARENT_OF", "child_to_parent", "parent_to_child"),
    "母后": ("CHILD_OF", "PARENT_OF", "child_to_parent", "parent_to_child"),
    "哥哥": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "弟弟": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "姐姐": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "妹妹": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "姊妹": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "兄弟": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "兄": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "弟": ("SIBLING_OF", "SIBLING_OF", "sibling_to_sibling", "sibling_to_sibling"),
    "妻": ("SPOUSE_OF", "SPOUSE_OF", "spouse_to_spouse", "spouse_to_spouse"),
    "妻子": ("SPOUSE_OF", "SPOUSE_OF", "spouse_to_spouse", "spouse_to_spouse"),
    "夫": ("SPOUSE_OF", "SPOUSE_OF", "spouse_to_spouse", "spouse_to_spouse"),
    "丈夫": ("SPOUSE_OF", "SPOUSE_OF", "spouse_to_spouse", "spouse_to_spouse"),
    "夫人": ("SPOUSE_OF", "SPOUSE_OF", "spouse_to_spouse", "spouse_to_spouse"),
    "妾": ("SPOUSE_OF", "SPOUSE_OF", "spouse_to_spouse", "spouse_to_spouse"),
    "叔父": ("NEPHEW_OR_NIECE_OF", "UNCLE_OR_AUNT_OF", "nephew_to_uncle", "uncle_to_nephew"),
    "伯父": ("NEPHEW_OR_NIECE_OF", "UNCLE_OR_AUNT_OF", "nephew_to_uncle", "uncle_to_nephew"),
    "舅舅": ("NEPHEW_OR_NIECE_OF", "UNCLE_OR_AUNT_OF", "nephew_to_uncle", "uncle_to_nephew"),
    "祖先": ("DESCENDANT_OF", "ANCESTOR_OF", "descendant_to_ancestor", "ancestor_to_descendant"),
    "先祖": ("DESCENDANT_OF", "ANCESTOR_OF", "descendant_to_ancestor", "ancestor_to_descendant"),
    "后代": ("ANCESTOR_OF", "DESCENDANT_OF", "ancestor_to_descendant", "descendant_to_ancestor"),
    "后代子孙": ("ANCESTOR_OF", "DESCENDANT_OF", "ancestor_to_descendant", "descendant_to_ancestor"),
}


def _is_kinship_entity_candidate(
    tokens: Sequence[str], index: int, tag_map: Dict[str, str]
) -> bool:
    """Accept a name/title in an explicit kinship slot, not a nearby verb."""

    if index < 0 or index >= len(tokens):
        return False
    token = tokens[index]
    if is_title_like(token) or token in PERSON_NAMES:
        return True
    # The fallback tagger deliberately does not infer people from frequency.
    # In a kinship slot an unlisted atomic noun is still too easy to confuse
    # with a verb or a caption word (“效法 先祖”, “儿子 穷困”), so require an
    # explicit relation lexicon entry.  If jieba is installed, its person tag
    # can extend the lexicon without weakening the no-tagger path.
    return bool(
        pseg is not None
        and tag_map.get(token, "").startswith(("nr", "nrf"))
        and _is_atomic_relation_name(token)
    )


def _invert_relation_spec(
    spec: Tuple[str, str, str, str]
) -> Tuple[str, str, str, str]:
    return spec[1], spec[0], spec[3], spec[2]


def _extract_kinship_facts(
    sentences: Sequence[str],
    line_index: int,
    tokens: Sequence[str],
    tag_map: Dict[str, str],
) -> List[Dict]:
    """Extract explicit family, lineage, spouse, and sibling links."""

    facts: List[Dict] = []
    seen_pairs: Set[Tuple[int, int, str, str]] = set()

    def add_relation(
        left_index: int,
        right_index: int,
        surface: str,
        start: int,
        end: int,
        spec: Tuple[str, str, str, str],
    ) -> None:
        if left_index == right_index:
            return
        if not (
            _is_kinship_entity_candidate(tokens, left_index, tag_map)
            and _is_kinship_entity_candidate(tokens, right_index, tag_map)
        ):
            return
        key = (left_index, right_index, surface, spec[0])
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        facts.extend(
            _make_link_facts(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                left_index=left_index,
                right_index=right_index,
                relation_surface=surface,
                relation_evidence=source_evidence(tokens, start, end),
                confidence=0.95,
                relation_types=(spec[0], spec[1]),
                grammar_override="KINSHIP",
                directions=(spec[2], spec[3]),
            )
        )

    def resolve_pronoun_owner(index: int) -> Optional[int]:
        """Resolve a local 他/她/其 to the nearest explicit antecedent."""

        lower = max(0, index - 24)
        for candidate_index in range(index - 1, lower - 1, -1):
            token = tokens[candidate_index]
            if token in SENTENCE_END:
                break
            if token in COMMA_MARKS or token in PUNCTUATION:
                continue
            if _is_kinship_entity_candidate(tokens, candidate_index, tag_map):
                return candidate_index
            if (
                token in PRONOUNS_AND_FUNCTIONS
                or token in RELATION_NON_NAME_WORDS
                or token in ACTION_WORDS
                or token in KINSHIP_WORDS
                or token in NAMING_MARKERS
            ):
                continue
            # Do not bridge an unknown prose noun while resolving a pronoun;
            # this keeps a nearby unrelated name from becoming the parent.
            break
        return None

    def owner_before_role(
        role_index: int, *, allow_list_commas: bool = False
    ) -> Optional[int]:
        skip = {
            "有", "有个", "一个", "一位", "两位", "两个", "三个", "四个", "五个", "六个",
            "七个", "八个", "几个", "多个", "个", "一", "两", "三", "四", "五", "六",
            "第一个", "第二个", "第一", "第二", "第三", "第四", "第五", "第六",
            "长", "次", "大", "小", "生", "生了", "生下", "了", "的",
            "自己", "我", "他", "她", "其",
        }
        lower = max(0, role_index - (32 if allow_list_commas else 8))
        for index in range(role_index - 1, lower - 1, -1):
            token = tokens[index]
            if token in SENTENCE_END:
                break
            if token in COMMA_MARKS:
                if allow_list_commas and token in {"，", ","}:
                    continue
                break
            if token in {"他", "她", "其"}:
                resolved = resolve_pronoun_owner(index)
                if resolved is not None:
                    return resolved
                continue
            if (
                token in skip
                or token in KINSHIP_WORDS
                or token in NAMING_MARKERS
                or token in PUNCTUATION
            ):
                continue
            if _is_kinship_entity_candidate(tokens, index, tag_map):
                if allow_list_commas and (
                    (index > 0 and tokens[index - 1] in {
                        "叫", "名叫", "名为", "名曰", "叫作", "叫做",
                    })
                    or (token.startswith("名") and len(token) > 1)
                ):
                    # Skip a previously named sibling while walking back
                    # through a list such as “长子 昆吾，次子 参胡”.
                    continue
                return index
            # Do not jump across a prose clause looking for a distant name.
            if token not in {"的", "了", "生", "有", "个"}:
                break
        return None

    for role_index, role in enumerate(tokens):
        spec = KINSHIP_RELATION_SPECS.get(role)
        if spec is None:
            continue

        # A 的 儿子/弟弟 是 B.  This is the inverse surface order of the
        # more common “A 的 儿子 B” construction.
        if (
            role_index >= 2
            and tokens[role_index - 1] == "的"
            and role_index + 2 < len(tokens)
            and tokens[role_index + 1] == "是"
            and _is_kinship_entity_candidate(tokens, role_index - 2, tag_map)
            and _is_kinship_entity_candidate(tokens, role_index + 2, tag_map)
        ):
            add_relation(
                role_index - 2,
                role_index + 2,
                role,
                role_index - 2,
                role_index + 3,
                spec,
            )

        # A 是 B 的 儿子/女儿.  Permit a short descriptive/political-place
        # bridge between 是 and B, as in “高阳，是黄帝的孙子” and
        # “秦始皇帝，是秦国庄襄王的儿子”.  The nearest explicit person
        # before 是 is the subject; the owner is always immediately before 的.
        if (
            role_index >= 3
            and tokens[role_index - 1] == "的"
            and _is_kinship_entity_candidate(tokens, role_index - 2, tag_map)
        ):
            for copula_index in range(role_index - 3, max(-1, role_index - 12), -1):
                if tokens[copula_index] != "是":
                    continue
                subject_index = None
                candidate_index = copula_index - 1
                while candidate_index >= 0 and tokens[candidate_index] in PUNCTUATION:
                    candidate_index -= 1
                if (
                    candidate_index >= 0
                    and _is_kinship_entity_candidate(tokens, candidate_index, tag_map)
                ):
                    subject_index = candidate_index
                if subject_index is not None:
                    add_relation(
                        subject_index,
                        role_index - 2,
                        role,
                        subject_index,
                        role_index + 1,
                        _invert_relation_spec(spec),
                    )
                break

        # A 的 儿子 B / A 的 母亲 B, including a name marker after the role.
        if role_index >= 2 and tokens[role_index - 1] == "的":
            owner_index = role_index - 2
            if not _is_kinship_entity_candidate(tokens, owner_index, tag_map):
                if tokens[owner_index] in {"他", "她", "其"}:
                    owner_index = resolve_pronoun_owner(owner_index)
                else:
                    owner_index = None
                if owner_index is None:
                    continue
            target_marker = role_index + 1
            if target_marker >= len(tokens):
                continue
            if tokens[target_marker] in {"叫", "名叫", "名为", "名曰", "叫作", "叫做"}:
                for target_index in _naming_target_indices(
                    tokens, target_marker, tag_map, True
                ):
                    add_relation(
                        owner_index,
                        target_index,
                        role,
                        owner_index,
                        target_index + 1,
                        spec,
                    )
            elif _is_kinship_entity_candidate(tokens, target_marker, tag_map):
                add_relation(
                    owner_index,
                    target_marker,
                    role,
                    owner_index,
                    target_marker + 1,
                    spec,
                )

        # A 有个儿子/弟弟 叫 B, A 生了个女儿 名叫 B, or the name marker is
        # separated by a comma: “有个后代子孙，名字叫做鬻熊”.
        target_marker = role_index + 1
        rank_role = role in {
            "长子", "次子", "三子", "四子", "五子", "六子", "第六子",
            "大儿子", "小儿子", "长女", "小女",
        }
        split_rank = role == "子" and role_index > 0 and tokens[role_index - 1] in {
            "一", "两", "三", "四", "五", "六", "七", "八", "第六",
        }
        if target_marker < len(tokens) and tokens[target_marker] in {
            "叫", "名叫", "名为", "名曰", "叫作", "叫做",
        }:
            owner_index = owner_before_role(
                role_index, allow_list_commas=rank_role or split_rank
            )
            if owner_index is None:
                continue
            for target_index in _naming_target_indices(
                tokens, target_marker, tag_map, True
            ):
                add_relation(
                    owner_index,
                    target_index,
                    role,
                    owner_index,
                    target_index + 1,
                    spec,
                )
        elif (
            target_marker < len(tokens)
            and _is_kinship_entity_candidate(tokens, target_marker, tag_map)
        ):
            owner_index = owner_before_role(
                role_index, allow_list_commas=rank_role or split_rank
            )
            if owner_index is not None:
                add_relation(
                    owner_index,
                    target_marker,
                    role,
                    owner_index,
                    target_marker + 1,
                    spec,
                )
        elif (
            target_marker < len(tokens)
            and tokens[target_marker] in PUNCTUATION
            and target_marker + 2 < len(tokens)
            and tokens[target_marker + 1] in {"名字", "姓名"}
            and tokens[target_marker + 2] in {"叫", "名叫", "名为", "名曰", "叫作", "叫做"}
        ):
            owner_index = owner_before_role(
                role_index, allow_list_commas=rank_role or split_rank
            )
            if owner_index is None:
                continue
            for target_index in _naming_target_indices(
                tokens, target_marker + 2, tag_map, True
            ):
                add_relation(
                    owner_index,
                    target_index,
                    role,
                    owner_index,
                    target_index + 1,
                    spec,
                )

    # Direct spouse verbs are relationship-bearing facts even when the source
    # does not use 的/有个 wording.
    for verb_index, verb in enumerate(tokens):
        if verb not in {"娶", "嫁给", "嫁", "娶了", "嫁给了"}:
            continue
        if verb_index == 0 or verb_index + 1 >= len(tokens):
            continue
        left_index = verb_index - 1
        right_index = verb_index + 1
        while right_index < len(tokens) and tokens[right_index] in {"了", "给"}:
            right_index += 1
        if right_index >= len(tokens):
            continue

        # Resolve “娶 吕后 的 妹妹 吕须 为 妻” to the final person in the
        # kinship phrase.  Falling back to the first title would incorrectly
        # teach that 樊哙 married 吕后 rather than 吕须.
        nested_target = None
        for role_index in range(right_index + 1, min(len(tokens), right_index + 6)):
            if tokens[role_index] != "的":
                continue
            nested_role = tokens[role_index + 1] if role_index + 1 < len(tokens) else ""
            if nested_role not in KINSHIP_RELATION_SPECS:
                continue
            candidate_index = role_index + 2
            if candidate_index < len(tokens) and (
                is_person_entity_candidate(tokens[candidate_index], tag_map)
                or is_title_like(tokens[candidate_index])
            ):
                nested_target = candidate_index
            break
        if nested_target is not None:
            right_index = nested_target
        spec = KINSHIP_RELATION_SPECS["妻子"]
        if not (
            (is_person_entity_candidate(tokens[left_index], tag_map) or is_title_like(tokens[left_index]))
            and (is_person_entity_candidate(tokens[right_index], tag_map) or is_title_like(tokens[right_index]))
        ):
            continue
        add_relation(
            left_index,
            right_index,
            verb,
            left_index,
            right_index + 1,
            spec,
        )

    return facts


def extract_all_facts(sentences: Sequence[str]) -> List[Dict]:
    facts: List[Dict] = []
    for line_index, line in enumerate(sentences):
        tokens = line.split()
        if len(tokens) < 4 or len(tokens) > 120:
            continue
        tag_map = pos_tags(line)

        # Context-dependent identity links are kept alongside action facts so
        # the training set can learn both stable aliases (沛公/汉王/高祖) and
        # ambiguous titles (赵王/楚王) from their surrounding evidence.
        facts.extend(_extract_title_and_alias_facts(sentences, line_index, tokens, tag_map))
        facts.extend(_extract_kinship_facts(sentences, line_index, tokens, tag_map))

        # 1. Assignment/command.  Naming expressions using 叫 are excluded.
        for verb_index, verb in enumerate(tokens):
            if verb not in CAUSATIVE_VERBS:
                continue
            target_index = find_causative_target(tokens, verb_index, tag_map)
            subject = find_subject(tokens, verb_index, tag_map)
            if target_index is None or subject is None:
                continue
            target = tokens[target_index]
            attrs = token_attributes(target, "PERSON")
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type="ASSIGN",
                grammar="CAUSATIVE",
                relation_surface=verb,
                relation_evidence=source_evidence(tokens, verb_index, target_index + 1),
                object_value=target,
                object_attributes=attrs,
                default_answer_attribute="PERSON",
                fact_group_id=f"line_{line_index + 1}_assign_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

        # 2. Appointment/ennoblement.  Person and single-token title targets
        # are separate records with separate answers.
        for verb_index, verb in enumerate(tokens):
            if verb not in APPOINTMENT_VERBS:
                continue
            parts = find_appointment_parts(tokens, verb_index, tag_map)
            subject = find_subject(tokens, verb_index, tag_map)
            if parts is None or subject is None:
                continue
            person_index, role_index = parts
            person = tokens[person_index]
            role = tokens[role_index] if role_index >= 0 else ""
            role_attributes = token_attributes(role, "TITLE") if role else ["TITLE", "ENTITY"]
            evidence = source_evidence(
                tokens, verb_index, max(person_index + 1, role_index + 1)
            )
            group_id = f"line_{line_index + 1}_appoint_{person_index}_{role_index}"
            person_fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=person_index,
                target_role="object",
                subject=subject,
                relation_type="APPOINT",
                grammar="APPOINTMENT",
                relation_surface=verb,
                relation_evidence=evidence,
                object_value=person,
                object_attributes=token_attributes(person, "PERSON"),
                default_answer_attribute="PERSON",
                fact_group_id=group_id,
            )
            if person_fact is not None:
                person_fact["role"] = role
                person_fact["role_attributes"] = role_attributes
                facts.append(person_fact)
            if role_index >= 0:
                role_fact = make_fact(
                    sentences=sentences,
                    line_index=line_index,
                    tokens=tokens,
                    target_index=role_index,
                    target_role="role",
                    subject=subject,
                    relation_type="APPOINT",
                    grammar="APPOINTMENT",
                    relation_surface=verb,
                    relation_evidence=evidence,
                    object_value=person,
                    object_attributes=token_attributes(person, "PERSON"),
                    default_answer_attribute="TITLE",
                    fact_group_id=group_id,
                )
                if role_fact is not None:
                    role_fact["role"] = role
                    role_fact["role_attributes"] = role_attributes
                    facts.append(role_fact)
                if _is_single_title_role(tokens, role_index):
                    facts.extend(
                        _make_link_facts(
                            sentences=sentences,
                            line_index=line_index,
                            tokens=tokens,
                            left_index=role_index,
                            right_index=person_index,
                            relation_surface=f"{verb}为",
                            relation_evidence=evidence,
                            confidence=0.96,
                            relation_types=("TITLE_OF", "HAS_TITLE"),
                            directions=("title_to_person", "person_to_title"),
                            right_default_override="PERSON",
                        )
                    )

        # 2b. Passive title assignment, e.g. 被封为侯 / 被任命为将军.
        # The original appointment extractor looks for the person after the
        # verb; this pass handles the common person-before-verb construction.
        for person_index, person in enumerate(tokens):
            if not is_person_entity_candidate(person, tag_map):
                continue
            for verb_index in range(person_index + 1, min(len(tokens), person_index + 7)):
                if tokens[verb_index] not in APPOINTMENT_VERBS:
                    continue
                passive_slice = tokens[person_index + 1 : verb_index]
                passive_before_person = (
                    person_index > 0 and tokens[person_index - 1] == "被"
                )
                if "被" not in passive_slice and not passive_before_person:
                    continue
                # The appointed person must be immediately before 被 (or have
                # only a short aspect/adverbial bridge).  A broad look-back
                # would misread “受 萧何 推荐 ， 被 刘邦 拜 为 大将” as if
                # 萧何 had been appointed.
                passive_index = (
                    person_index - 1
                    if passive_before_person
                    else person_index + 1 + passive_slice.index("被")
                )
                if not passive_before_person and passive_index - person_index > 2:
                    continue
                role_index = _title_role_after_appointment(tokens, verb_index)
                if role_index is None:
                    continue
                facts.extend(
                    _make_link_facts(
                        sentences=sentences,
                        line_index=line_index,
                        tokens=tokens,
                        left_index=role_index,
                        right_index=person_index,
                        relation_surface=f"被{tokens[verb_index]}为",
                        relation_evidence=source_evidence(
                            tokens,
                            min(person_index, passive_index),
                            role_index + 1,
                        ),
                        confidence=0.96,
                        relation_types=("TITLE_OF", "HAS_TITLE"),
                        directions=("title_to_person", "person_to_title"),
                        right_default_override="PERSON",
                    )
                )

        # 3. Origin/place.  Person and place are separate prediction targets.
        for person_index in range(len(tokens) - 2):
            if not is_person_like(tokens[person_index], tag_map):
                continue
            if tokens[person_index + 1] not in ORIGIN_WORDS:
                continue
            place_index = person_index + 2
            if not is_place_like(tokens[place_index], tag_map):
                continue
            person = tokens[person_index]
            place = tokens[place_index]
            subject = {
                "text": person,
                "attributes": token_attributes(person, "PERSON"),
                "mode": "explicit_origin_subject",
            }
            evidence = source_evidence(tokens, person_index, place_index + 1)
            group_id = f"line_{line_index + 1}_origin_{person_index}_{place_index}"
            person_fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=person_index,
                target_role="subject",
                subject=subject,
                relation_type="BORN_IN",
                grammar="ORIGIN",
                relation_surface=tokens[person_index + 1],
                relation_evidence=evidence,
                object_value=place,
                object_attributes=token_attributes(place, "PLACE"),
                default_answer_attribute="PERSON",
                fact_group_id=group_id,
            )
            if person_fact is not None:
                facts.append(person_fact)
            place_fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=place_index,
                target_role="object",
                subject=subject,
                relation_type="BORN_IN",
                grammar="ORIGIN",
                relation_surface=tokens[person_index + 1],
                relation_evidence=evidence,
                object_value=place,
                object_attributes=token_attributes(place, "PLACE"),
                default_answer_attribute="PLACE",
                fact_group_id=group_id,
            )
            if place_fact is not None:
                facts.append(place_fact)

        # 4. Campaign/attack facts with canonical relation types.
        for verb_index, verb in enumerate(tokens):
            relation_type = CAMPAIGN_RELATIONS.get(verb)
            if relation_type is None or verb_index + 1 >= len(tokens):
                continue
            target_index = verb_index + 1
            target = tokens[target_index]
            target_attrs = token_attributes(target)
            if target in PRONOUNS_AND_FUNCTIONS or target in PUNCTUATION:
                continue
            if not (set(target_attrs) & {"PERSON", "PLACE", "GROUP", "TITLE"}):
                continue
            subject = find_subject(tokens, verb_index, tag_map)
            if subject is None:
                continue
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type=relation_type,
                grammar="CAMPAIGN",
                relation_surface=verb,
                relation_evidence=source_evidence(tokens, verb_index, target_index + 1),
                object_value=target,
                object_attributes=target_attrs,
                default_answer_attribute="PERSON",
                fact_group_id=f"line_{line_index + 1}_{relation_type}_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

        # 5. Combat outcomes.
        for verb_index, verb in enumerate(tokens):
            relation_type = COMBAT_RELATIONS.get(verb)
            if relation_type is None or verb_index + 1 >= len(tokens):
                continue
            target_index = verb_index + 1
            target = tokens[target_index]
            target_attrs = token_attributes(target, "PERSON")
            if target in PRONOUNS_AND_FUNCTIONS or target in PUNCTUATION:
                continue
            if not (set(target_attrs) & {"PERSON", "PLACE", "GROUP", "TITLE"}):
                continue
            subject = find_subject(tokens, verb_index, tag_map)
            if subject is None:
                continue
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type=relation_type,
                grammar="COMBAT_OUTCOME",
                relation_surface=verb,
                relation_evidence=source_evidence(tokens, verb_index, target_index + 1),
                object_value=target,
                object_attributes=target_attrs,
                default_answer_attribute="PERSON",
                fact_group_id=f"line_{line_index + 1}_{relation_type}_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

        # 6. Place attached to a compact historical event.
        for place_index in range(len(tokens) - 2):
            if tokens[place_index] != "在":
                continue
            target_index = place_index + 1
            event_index = place_index + 2
            if not is_place_like(tokens[target_index], tag_map):
                continue
            if tokens[event_index] not in LOCATION_EVENTS:
                continue
            subject = find_subject(tokens, place_index, tag_map)
            if subject is None:
                continue
            place = tokens[target_index]
            fact = make_fact(
                sentences=sentences,
                line_index=line_index,
                tokens=tokens,
                target_index=target_index,
                target_role="object",
                subject=subject,
                relation_type="EVENT_AT",
                grammar="LOCATION_EVENT",
                relation_surface=tokens[event_index],
                relation_evidence=source_evidence(tokens, place_index, event_index + 1),
                object_value=place,
                object_attributes=token_attributes(place, "PLACE"),
                default_answer_attribute="PLACE",
                fact_group_id=f"line_{line_index + 1}_event_at_{target_index}",
            )
            if fact is not None:
                facts.append(fact)

    return facts


def validate_record(record: Dict, original_sentences: Sequence[str]) -> List[str]:
    errors: List[str] = []
    source_line = record.get("source_line")
    if not isinstance(source_line, int) or not (1 <= source_line <= len(original_sentences)):
        return ["invalid_source_line"]
    context = record.get("context")
    ground_truth = original_sentences[source_line - 1]
    if context != ground_truth:
        errors.append("context_mismatch")

    tokens = str(context).split()
    answer = record.get("answer")
    span = record.get("target_span")
    if not isinstance(span, list) or len(span) != 2:
        errors.append("invalid_target_span")
        return errors
    start, end = span
    if not (isinstance(start, int) and isinstance(end, int) and end == start + 1):
        errors.append("non_single_token_target")
    elif not (0 <= start < len(tokens)) or tokens[start] != answer:
        errors.append("target_span_answer_mismatch")

    expected_mask = " ".join(mask_at(tokens, start, end)) if 0 <= start < len(tokens) else ""
    if record.get("masked_text") != expected_mask:
        errors.append("masked_text_not_exact")
    if str(record.get("masked_text", "")).split().count("[MASK]") != 1:
        errors.append("base_mask_count")
    if answer_is_visible(str(answer), str(record.get("masked_text", ""))):
        errors.append("base_answer_leakage")

    if not record.get("relation_type") or record.get("relation") != record.get("relation_type"):
        errors.append("relation_not_canonical")
    if not record.get("answer_attributes"):
        errors.append("missing_answer_attributes")
    if record.get("split") not in {"train", "dev", "test"}:
        errors.append("invalid_split")

    for variant in record.get("variants", []):
        if variant.get("answer") != answer:
            errors.append("variant_answer_mismatch")
        masked = str(variant.get("masked_text", ""))
        if masked.split().count("[MASK]") != 1:
            errors.append("variant_mask_count")
        if answer_is_visible(str(answer), masked):
            errors.append("variant_answer_leakage")
        if not variant.get("answer_attributes"):
            errors.append("variant_missing_attributes")
    return errors


def validate_and_deduplicate(
    facts: Sequence[Dict], original_sentences: Sequence[str]
) -> Tuple[List[Dict], Counter]:
    valid: List[Dict] = []
    rejected: Counter = Counter()
    seen: Set[Tuple[object, ...]] = set()
    for fact in facts:
        errors = validate_record(fact, original_sentences)
        if errors:
            rejected.update(errors)
            continue
        key = (
            fact["source_line"],
            fact["target_span"][0],
            fact["relation_type"],
            fact["target_role"],
            fact["answer"],
        )
        if key in seen:
            rejected["duplicate_fact"] += 1
            continue
        seen.add(key)
        valid.append(fact)
    valid.sort(
        key=lambda item: (
            item["source_line"],
            item["target_span"][0],
            item["relation_type"],
            item["target_role"],
        )
    )
    return valid, rejected


def build_report(
    facts: Sequence[Dict],
    rejected: Counter,
    source_sentence_count: int,
) -> Dict:
    relation_grammars = {"TITLE_LINK", "ALIAS", "KINSHIP"}
    relation_facts = [
        fact for fact in facts if fact.get("grammar") in relation_grammars
    ]
    relation_source_lines = {
        fact["source_line"] for fact in relation_facts if "source_line" in fact
    }
    relation_pairs = {
        (
            fact.get("relation_type"),
            (fact.get("linked_entity") or {}).get("left"),
            (fact.get("linked_entity") or {}).get("right"),
        )
        for fact in relation_facts
        if (fact.get("linked_entity") or {}).get("left")
        and (fact.get("linked_entity") or {}).get("right")
    }
    quality = Counter()
    for fact in facts:
        quality["exact_one_base_mask"] += int(
            fact["masked_text"].split().count("[MASK]") == 1
        )
        quality["base_answer_leakage"] += int(
            answer_is_visible(fact["answer"], fact["masked_text"])
        )
        for variant in fact.get("variants", []):
            quality["variant_count"] += 1
            quality["variant_answer_mismatch"] += int(
                variant.get("answer") != fact["answer"]
            )
            quality["variant_answer_leakage"] += int(
                answer_is_visible(fact["answer"], variant["masked_text"])
            )
    return {
        "schema_version": 2,
        "total_facts": len(facts),
        "source_sentences_count": source_sentence_count,
        "source_lines_used": len({fact["source_line"] for fact in facts}),
        "source_line_coverage_ratio": (
            len({fact["source_line"] for fact in facts}) / source_sentence_count
            if source_sentence_count
            else 0.0
        ),
        "relation_fact_count": len(relation_facts),
        "relation_source_lines_used": len(relation_source_lines),
        "relation_source_coverage_ratio": (
            len(relation_source_lines) / source_sentence_count
            if source_sentence_count
            else 0.0
        ),
        "relation_pair_count": len(relation_pairs),
        "relation_direction_distribution": dict(
            Counter(
                fact.get("relation_direction", "unspecified")
                for fact in relation_facts
            )
        ),
        "relation_surface_distribution": dict(
            Counter(fact.get("relation_surface", "") for fact in relation_facts)
        ),
        "grammar_distribution": dict(Counter(fact["grammar"] for fact in facts)),
        "relation_type_distribution": dict(
            Counter(fact["relation_type"] for fact in facts)
        ),
        "target_role_distribution": dict(
            Counter(fact["target_role"] for fact in facts)
        ),
        "split_distribution": dict(Counter(fact["split"] for fact in facts)),
        "attribute_distribution": dict(
            Counter(tuple(fact["answer_attributes"]) for fact in facts)
        ),
        "variant_distribution": dict(
            Counter(len(fact.get("variants", [])) for fact in facts)
        ),
        "distractor_distribution": dict(
            Counter(len(fact.get("distractors", [])) for fact in facts)
        ),
        "average_heuristic_confidence": (
            sum(float(fact["confidence"]) for fact in facts) / len(facts)
            if facts
            else 0.0
        ),
        "quality_checks": dict(quality),
        "rejected_candidates": dict(rejected),
        "notes": [
            "confidence is a rule-based score, not a measured accuracy",
            "naming/alias expressions using 叫 are excluded from CAUSATIVE",
            "all variants carry an independent answer label",
            "source_line determines the split to prevent context leakage",
            "relation coverage is measured against the current segmented source file, not a claim of complete semantic annotation of every Shiji chapter",
            "relation extraction is a conservative lexicon-and-pattern candidate set; glued multi-token names may be skipped when a one-token mask would be ambiguous",
        ],
        "sample_preview": list(facts[:3]),
    }


def main() -> None:
    print(f"=== Loading segmented sentences from {SEGMENTED_PATH} ===")
    sentences = SEGMENTED_PATH.read_text(encoding="utf-8").splitlines()
    print(f"Loaded {len(sentences)} sentences.")

    print("=== Mining source-verified fact candidates ===")
    raw_facts = extract_all_facts(sentences)
    print(f"Mined {len(raw_facts)} raw candidates.")

    print("=== Validating exact spans, masks, variants, and splits ===")
    facts, rejected = validate_and_deduplicate(raw_facts, sentences)
    print(f"Validated {len(facts)} facts.")
    if rejected:
        print("Rejected:", dict(rejected))

    for index, fact in enumerate(facts, start=1):
        fact["fact_id"] = f"shiji_fact_{index:04d}"

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = MANIFESTS_DIR / "fact_memory_dataset.json"
    mirror_path = OUTPUT_DIR / "facts_dataset.json"
    report_path = MANIFESTS_DIR / "fact_memory_report.json"
    serialized = json.dumps(facts, ensure_ascii=False, indent=2)
    dataset_path.write_text(serialized, encoding="utf-8")
    mirror_path.write_text(serialized, encoding="utf-8")
    report_path.write_text(
        json.dumps(
            build_report(facts, rejected, len(sentences)),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved dataset to {dataset_path}")
    print(f"Mirrored dataset to {mirror_path}")
    print(f"Saved report to {report_path}")
    print(json.dumps(build_report(facts, rejected, len(sentences)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
