"""
§13.1 分母、§13.8 帖子类型、§13.7 时间限制。

这三节改的都是**同一个量怎么被测出来**，而不是哪些用户进分析（那是
samples.py）、也不是"某个用户的领域参与怎么算"里的词表/账号名单
（vocabulary.py / accounts.py）。因此本模块的产出只有一个用途：

    如果换成另一种同样合理的测法，结论会不会变？

--------------------------------------------------------------------------
最重要的一条：**每一个 variant_label 都必须写明分母**
--------------------------------------------------------------------------
这一族存在的全部意义，就是证明"分母的选择没有推着结论走"。一行不写明
分母的结果，读起来与主口径的结果一模一样——读者无法判断它到底是"表达帖
分母"还是"全部帖子分母"，这一族于是自己废掉自己。主流水线在这件事上已经
栽过一次：早先的代码报告过一个分母含义不明的比值，是一次复核才发现的。

所以本模块的每个 label 都长成 `X=...;denominator=...` 的形状，
gender_domain/tests/test_measures_robustness.py 里有三条测试逐行扫描
label，缺一个 `denominator=` 就失败。note 里还会再写一遍分子、分母与
实际样本量，让只拿到 measures.parquet 一张表的读者也看得见这一行测的
是什么。

--------------------------------------------------------------------------
六个量是写死的，所以"另一种测法"只能写进 topical_share 这一列
--------------------------------------------------------------------------
harness.QUANTITIES 把六个预先设定量写死（方案文档 §11.3），本模块不允许
新增估计量。因此 §13.1 的字符密度、每字命中数、领域月份占比这些替代测量
**只能写进 `{domain}_topical_share` 这一列**，由同一个分数 logit 估同一个
形状的量（性别在一个 0-1 占比上的平均边际效应）。

这件事必须靠 label 说清楚，否则 outcome 列写着 topical_share、里面装的却
是字符密度，读者会把它当成主口径。因此这类变体的 label 一律写成
`topical_measure=<分子>;denominator=<分母>`，note 里再写一遍
`topical_share_column_holds=<真实测量>`。**绝不**在 outcome 列上另造名字：
那会让下游 synthesis 按 (outcome, domain, term) 反查六个量时找不到这一行，
等于这个变体没跑过。

顺带一条口径问题：「每千字命中数」不是占比（一条 100 字、命中 1 次的帖子
是每千字 10 次），而分数 logit 以取值落在 [0,1] 为前提
（stats_utils.check_proportion_range 会直接报错）。所以本模块在**每字**
命中数上拟合——它恒定落在 [0,1]（词表匹配区间不重叠，命中次数 ≤ 命中
字符数 ≤ 总字符数），与每千字口径只差一个常数因子 1000，估计量的符号、
显著性、跨变体的比较全部不变。label 与 note 都写明这一点。

--------------------------------------------------------------------------
§13.8：帖子类型的重新聚合走 incidence 的 posts 帧，不新写一条聚合路径
--------------------------------------------------------------------------
incidence.build_post_term_incidence 的 posts 帧本来就带着 post_type 与
n_chars（POST_FRAME_COLUMNS 里写着"刻意多带 month 与 post_type"），而
incidence.topical_by_user 的分子分母都由 posts 帧的 `is_expressive` 列定义、
且 posts 参数是必填的（"哪一份帖子集合定义了分母"必须由调用方明说）。

因此帖子类型变体的做法是：**把 posts 帧的 is_expressive 列换成"属于这次
选中的帖子类型、且清洗后字符数 > 0"**，其余原样交给 topical_by_user。
这不是绕过 incidence 的接口，正是它设计出来要被这么用的那个参数；也因此
不需要往 incidence 里加任何列。

过滤器取主口径（原创 + 带评论转发，字符数 > 0）时，这个掩码与表 A 的
is_expressive 逐帖相等，重聚合必须逐用户重现表 C——这与 Task 2 的全词表
重建是同一类检验，是"重新聚合这条路走得通"的证明，测试里钉死了它。

§13.1 的「全部帖子」分母是**另一个**掩码：它包含纯转发，也**不排除零字符
帖**——build_user_tables 里写着这个并行口径"刻意保持字面上的全部帖子，
不采纳零字符帖排除规则，一旦跟着主口径改就失去比较价值"。所以它与 §13.8
的 `all_text`（三种类型但仍要求字符数 > 0）不是同一个变体，两者的 label
分别写明这一点。

一条必须说明的性质：帖子类型只作用于内容测量，进入指示来自表 B 的转发
事件，**按构造不受帖子类型影响**。因此这些变体的 entry / did_entry 三个量
若照常产出，会与基线逐字相同——那是坐在分布正中心的幽灵行，会同时抬高
synthesis 的方向一致率、压低离散度。所以它们**逐行写成 NaN**（行数不变），
"进入指示不受帖子类型影响"这个发现用一行注明原因的行说一次。
完整理由见下方 ENTRY_QUANTITIES 上方的长注释——特别是"为什么只在 note 里
写一个标记不管用"。同一条处理也适用于 §13.1 的替代测量变体：把字符密度
写进 topical_share 列同样碰不到进入指示。

与此配套的一条：**与基线逐字相同的口径一律不产出结果行**（进入指示以全部
同性别用户为分母、话题占比以表达帖为分母、以及异常月份规则一个月都没选中
时的全年重估）。它们改为作为**重建检验**运行——断言重现表 C，把"已核验
相等"写进 manifest。这与 Task 2 的全集重建是同一条纪律，而且比一行重复的
估计有信息量得多：它证明重聚合这条路本身是对的。

--------------------------------------------------------------------------
§13.7：留一月与分月估计**从 models_temporal 的输出里读**，绝不重算
--------------------------------------------------------------------------
月度参与率与十二次留一月重估已经在 gender_domain/models_temporal.py 里
算过并写进 analysis_data/results/（monthly_rates.parquet 与
leave_one_month_out.parquet）。本模块**读那两份文件**，把留一月的性别 AME
原样搬进稳健性结果表（贴上变体身份、note 注明来源），不自己再拟合一遍。

理由不是省时间：同一个会被写进论文的数字由两个模块各算一遍，迟早会因为
某一处口径改动而不一致，而读者没有办法判断该信哪一个。models_temporal 的
留一月估计带着月份固定效应与按用户聚类的标准误，本模块无论如何也不该在
一个不同的规格下重新造一个"留一月估计"。

文件不存在时留一行注明原因的 NaN 行（走 vocabulary._note_only_rows，与
samples 处理 demographic_gender 缺失时是同一套做法），**绝不改为自己算**。

§13.7 里 models_temporal 没有覆盖的三件事由本模块自己做，因为它们不是
"留一个月"而是"换一段窗口/换一批用户"：上半年 vs 下半年、剔除行为量异常
的月份、以及只保留活跃满 3 / 6 个月的用户。前两者按月重新聚合（进入指示
来自表 D 的用户—月面板，内容测量来自 incidence 的 posts 帧按 month 过滤），
后者只是对表 C 做一次行选择。

--------------------------------------------------------------------------
异常月份的规则：事先写死、机械执行、并记下它选中了哪几个月
--------------------------------------------------------------------------
    一个月是异常月，当且仅当它的**行为量**与全年月度中位数的偏差同时满足
        |v_m - median(v)| > 3.0 × 1.4826 × MAD(v)
        |v_m - median(v)| > 0.5 × median(v)
    即 |v_m - median(v)| > max(两个阈值)。

三条设计说明：

1. **只看行为量，不看六个结果变量。** 行为量取"发帖数 + 转发帖数"
   （与 samples.TOTAL_ACTIVITY_COLUMNS 同一个定义，不另立一个），在表 D 上
   按月求和。按结果变量挑月份就是在对因变量做选择，而读者从输出上完全
   分辨不出这两种做法的差别——这正是"规则必须事先写死"这条要求想挡住的。

2. **中位数 + MAD 而不是均值 + 标准差。** 十二个观测里，一个极端月份会把
   标准差撑大到足以把自己遮住（掩蔽效应）；MAD 对此免疫。1.4826 是让 MAD
   在正态下与标准差同尺度的常数，3 倍是 Hampel 判别子的惯用阈值。

3. **再加一条 50% 的相对偏差下限。** 只用 Hampel 那一半有一个退化情形：
   一年几乎恒定时 MAD 会塌到 0，阈值随之变成 0，于是任何一个哪怕只差 1 的
   月份都会被判成"异常"。相对偏差下限把这种情形挡住；反过来，只用相对
   偏差又对真实的月度波动过于迟钝。两条同时要求（取 max）才既不退化、
   也不迟钝。

规则文本、逐月行为量、中位数、MAD、阈值、以及**它选中了哪几个月**，全部
写进诊断表与 manifest。看过哪个月不方便再挑月份不是稳健性检验，而读者从
输出上分不出这两者——除非规则和它的选择结果都被记下来。

使用方法:
    python -m gender_domain.robustness.measures build --year 2020
"""

import os
import tempfile
from collections import namedtuple

import fire
import numpy as np
import pandas as pd

from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import id_rules as ir
from gender_domain import models_core as mc
from gender_domain import models_temporal as mt
from gender_domain import stats_utils as su
from gender_domain import text_rules as tr
from gender_domain.robustness import harness
from gender_domain.robustness import incidence as inc
from gender_domain.robustness import samples as smp
# _annotate_note / _note_only_rows 一律复用 vocabulary 的实现（accounts 与
# samples 也是这么做的）：几个 family 的 note 拼接规则一旦分叉，跨 family
# 的比较就不再是同一件事。本族全部变体都是确定性的，没有任何抽样，因此
# 不需要 replicate_seeds / _domain_seed——每一行的 seed 如实写 None
# （harness 的规则：确定性变体记 None，不是省略这一列）。
from gender_domain.robustness import vocabulary as voc

DOMAINS = but.DOMAINS

VARIANT_FAMILY_DENOMINATOR = "denominators"        # §13.1
VARIANT_FAMILY_POST_TYPE = "post_types"            # §13.8
VARIANT_FAMILY_TEMPORAL = "temporal_restrictions"  # §13.7

# ---------------------------------------------------------------------------
# 表列
# ---------------------------------------------------------------------------

# 表 C 里本模块用得到的列：vocabulary 那份共享清单，加上 §13.1 要用的几个
# 并行口径列（表 C 早就并行输出了它们，见 build_user_tables 里"供 13.1 分母
# 稳健性使用"那一段）。
USER_TABLE_COLUMNS = list(voc.USER_TABLE_COLUMNS) + [
    column
    for domain in DOMAINS
    for column in (
        "{}_topical_share_allposts".format(domain),
        "{}_char_density".format(domain),
        "{}_hits_per_1k".format(domain),
        "{}_source_months".format(domain),
        "{}_source_month_share".format(domain),
    )
]

# 表 D 里本模块用得到的列。§13.7 的月份限制靠它重新聚合活动量与进入指示：
# 内容测量可以从 incidence 的 posts 帧按 month 过滤重算，但进入指示来自
# 表 B 的转发事件，而用户×账号矩阵没有月份轴——表 D 才是"逐月的进入"这件
# 事的权威来源。
PANEL_COLUMNS = [
    "user_id", "month", "gender", "n_posts", "n_retweets", "n_active_days",
] + ["{}_source_count".format(domain) for domain in DOMAINS]

# ---------------------------------------------------------------------------
# §13.8 帖子类型
# ---------------------------------------------------------------------------

# 主口径的帖子类型：口径的唯一来源是 text_rules，本模块不另立定义
PRIMARY_POST_TYPES = tuple(tr.EXPRESSIVE_TYPES)
PLAIN_POST_TYPE = "retweet_plain"
ALL_TEXT_POST_TYPES = PRIMARY_POST_TYPES + (PLAIN_POST_TYPE,)

# 每个变体：(key, 选中的帖子类型, 是否要求字符数 > 0, 分母的人话描述,
# 是否就是主口径)。`is_baseline=True` 的那一条**不作为变体产出结果行**：
# 它与基线逐字相同，摆一行进结果表只会在分布正中心多一个幽灵变体，把方向
# 一致率抬高、把离散度压低（samples.py 拒绝复制 untreated 结果去冒充 log1p
# 变体，是同一条纪律）。它改为作为**重建检验**运行：断言它重现表 C，并把
# "已核验相等"写进 manifest——这比多一行重复的估计有用得多。
PostTypeVariant = namedtuple(
    "PostTypeVariant",
    ["key", "post_types", "require_nonzero_chars", "denominator", "is_baseline"],
)

POST_TYPE_VARIANTS = (
    PostTypeVariant("original_only", ("original",), True,
                    "original_posts_with_nonzero_chars", False),
    PostTypeVariant("primary", PRIMARY_POST_TYPES, True,
                    "expressive_posts(original+commented_retweets,nonzero_chars)",
                    True),
    PostTypeVariant("plain_retweets_only", (PLAIN_POST_TYPE,), True,
                    "plain_retweets_with_nonzero_chars", False),
    PostTypeVariant("all_text", ALL_TEXT_POST_TYPES, True,
                    "all_post_types_with_nonzero_chars", False),
)

# ---------------------------------------------------------------------------
# 「这个变体按构造动不到哪几个量」必须写成 NaN，不能只写进 note
# ---------------------------------------------------------------------------
#
# 帖子类型过滤只作用于内容测量；把替代测量写进 topical_share 列同样只作用于
# 内容测量。两者都碰不到进入指示——它来自表 B 的转发事件，不是内容测量。
# 于是这些变体的 entry_public / entry_celebrity / did_entry 三个量会给出与
# 基线**逐字相同**的估计。
#
# 早先的做法是照常产出这三个量、在 note 里写一句"按构造不变"。这不管用，
# 有两个各自独立的理由：
#
# 1. **散文到不了算术。** synthesis 的方向一致率数的是"与基线同号的行"，
#    一堆坐在正中心的重复行会同时抬高一致率、压低离散度——一句只有人能读到
#    的 note 拦不住这件事。
# 2. **那个标记本身就不可用。** note 是通过 voc._annotate_note 贴的，它按
#    设计**盖整帧**：标记会同时落在 topical 的三个量上。下游若按这个字符串
#    过滤，会把 §13.8 真正要检验的 topical_public / topical_celebrity /
#    did_topical 一起丢掉，除非它自己在字符串匹配背后重新推导"六个量里哪
#    几个是 entry"——那等于把 harness 的映射抄一遍。
#
# 因此改为**逐行**把这三个量写成 NaN（行数不变，estimate_all 的行数不变量
# 原样成立），NaN 本来就是本套件"这里没测"的既有信号，不需要任何新约定；
# 而"进入指示不受影响"这个发现，按既有做法用**一行**注明原因的行说一次。
#
# 哪几个量算 entry 从 harness.QUANTITY_META 现场推导，不在这里另抄一份：
# 六个量的定义只能有一个来源。
ENTRY_QUANTITIES = tuple(
    key for key, meta in harness.QUANTITY_META.items()
    if meta["outcome"] == mt.OUTCOME_ENTRY
)
# 这三个量在这类变体里被写成 NaN 时，行内 note 写的原因
NOTE_ENTRY_NOT_ESTIMATED = (
    "entry_not_estimated_in_this_variant:"
    "the_variant_only_changes_the_content_measure;"
    "entry_comes_from_table_B_retweet_events_and_is_identical_to_the_baseline;"
    "emitted_as_NaN_so_it_is_not_counted_as_an_independent_variant"
)
# "进入指示不受帖子类型影响"这个发现，用一行说一次
FINDING_ENTRY_UNAFFECTED_POST_TYPE = (
    "post_types=any;denominator=all_same_gender_users(as_in_the_baseline);"
    "finding=entry_unaffected_by_post_type"
)
FINDING_ENTRY_UNAFFECTED_POST_TYPE_NOTE = (
    "post_type_filtering_cannot_change_entry:"
    "entry_is_a_table_B_retweet_event_not_a_content_measure;"
    "stated_once_here_instead_of_duplicating_the_baseline_estimate_in_every_variant"
)
FINDING_ENTRY_UNAFFECTED_MEASURE = (
    "topical_measure=any;denominator=all_same_gender_users(as_in_the_baseline);"
    "finding=entry_unaffected_by_the_topical_measure"
)
FINDING_ENTRY_UNAFFECTED_MEASURE_NOTE = (
    "substituting_an_alternative_topical_measure_cannot_change_entry:"
    "entry_is_a_table_B_retweet_event_not_a_content_measure;"
    "stated_once_here_instead_of_duplicating_the_baseline_estimate_in_every_variant"
)

# ---------------------------------------------------------------------------
# §13.1 分母与替代测量
# ---------------------------------------------------------------------------

# 进入指示的分母（= 谁进样本）
ENTRY_DENOMINATOR_ALL = "all_same_gender_users"
ENTRY_DENOMINATOR_RETWEETERS = "users_with_n_retweets>0"

# 替代测量的键
MEASURE_EXPRESSIVE_SHARE = "expressive_share"
MEASURE_ALLPOSTS_SHARE = "allposts_share"
MEASURE_CHAR_DENSITY = "char_density"
MEASURE_HITS_PER_CHAR = "hits_per_char"
MEASURE_SOURCE_MONTHS_PANEL = "source_month_share_panel"
MEASURE_SOURCE_MONTHS_POSTS = "source_month_share_posts_only"

# 进 label 的分子/分母文字。写成常量而不是散落的字面量，是为了让下面那张
# 「label -> 权威列」的对照表与 MEASURE_SPECS 只可能一起改。
NUM_TOPICAL_POSTS = "topical_expressive_posts"
NUM_HIT_POSTS = "hit_posts"
NUM_HIT_CHARS = "hit_characters"
NUM_HIT_COUNT = "hit_count"
NUM_SOURCE_MONTHS = "domain_source_months"

DEN_EXPRESSIVE_POSTS = "expressive_posts(main)"
DEN_ALL_POSTS = "all_posts_including_zero_char_posts"
DEN_EXPRESSIVE_CHARS = "expressive_characters"
DEN_EXPRESSIVE_CHARS_PER_1K = "expressive_characters(reported_per_character)"
DEN_ACTIVE_MONTHS_PANEL = "active_months_panel(includes_retweet_only_months)"
DEN_ACTIVE_MONTHS_POSTS = "active_months_from_posts_only"

MeasureSpec = namedtuple(
    "MeasureSpec", ["key", "numerator", "denominator", "note", "is_baseline"]
)

# 每个替代测量的分子/分母，全部写成人话直接进 label 与 note。
# `is_baseline=True` 的那一条与基线逐字相同，因此**不作为变体产出结果行**，
# 只作为重建检验运行（理由见 PostTypeVariant 上方的说明）。
MEASURE_SPECS = (
    MeasureSpec(MEASURE_EXPRESSIVE_SHARE, NUM_TOPICAL_POSTS,
                DEN_EXPRESSIVE_POSTS, None, True),
    MeasureSpec(MEASURE_ALLPOSTS_SHARE, NUM_HIT_POSTS, DEN_ALL_POSTS,
                "parallel_column_in_table_C:{domain}_topical_share_allposts", False),
    MeasureSpec(MEASURE_CHAR_DENSITY, NUM_HIT_CHARS,
                DEN_EXPRESSIVE_CHARS, None, False),
    MeasureSpec(MEASURE_HITS_PER_CHAR, NUM_HIT_COUNT,
                DEN_EXPRESSIVE_CHARS_PER_1K,
                "reported_per_character;multiply_by_1000_for_hits_per_1k_scale;"
                "fractional_logit_requires_a_proportion", False),
    MeasureSpec(MEASURE_SOURCE_MONTHS_PANEL, NUM_SOURCE_MONTHS,
                DEN_ACTIVE_MONTHS_PANEL, None, False),
    MeasureSpec(MEASURE_SOURCE_MONTHS_POSTS, NUM_SOURCE_MONTHS,
                DEN_ACTIVE_MONTHS_POSTS,
                "clipped_at_1_like_build_user_tables_source_share", False),
)
MEASURE_BY_KEY = {spec.key: spec for spec in MEASURE_SPECS}

# ---------------------------------------------------------------------------
# 把 label 与它实际算出来的东西绑在一起
# ---------------------------------------------------------------------------
#
# MEASURE_SPECS 里的 `key` 决定算什么，`numerator`/`denominator` 决定 label
# 写什么，两者之间原本没有任何东西相连——把两条 spec 的分母字符串对调一下，
# 结果表上"分母 = 全部帖子"那一行装的就会是表达帖口径的份额，而整套测试
# 一条都不会失败。**一个自信地写错的分母比一个没写分母更糟**：这一族存在的
# 唯一意义就是证明分母的选择没有推着结论走，而主流水线已经为同一类错误付过
# 一次复核的代价。
#
# 下面这张表按 **(分子, 分母) 这一对进 label 的字符串** 反查表 C 里的权威
# 列，`check_measure_matches_its_label` 逐用户比对。对调分母会让查表落空
# （或查到另一列），当场报错。**六条 spec 必须条条有登记**：查不到就报错，
# 否则"对调之后查不到、于是不检查"会变成一个静默的逃生口。
#
# column 为 None 的那一条是表 C 里确实没有对应列的口径（只看发帖的活跃月份
# 分母正是这个变体要引入的东西），它仍然必须登记在册，只是检查内容改成
# "断言表 C 里没有这一列"，而不是跳过。
LabelCheck = namedtuple("LabelCheck", ["column", "scale"])

LABEL_TO_TABLE_C = {
    (NUM_TOPICAL_POSTS, DEN_EXPRESSIVE_POSTS):
        LabelCheck("{}_topical_share", 1.0),
    (NUM_HIT_POSTS, DEN_ALL_POSTS):
        LabelCheck("{}_topical_share_allposts", 1.0),
    (NUM_HIT_CHARS, DEN_EXPRESSIVE_CHARS):
        LabelCheck("{}_char_density", 1.0),
    (NUM_HIT_COUNT, DEN_EXPRESSIVE_CHARS_PER_1K):
        LabelCheck("{}_hits_per_1k", 1000.0),
    (NUM_SOURCE_MONTHS, DEN_ACTIVE_MONTHS_PANEL):
        LabelCheck("{}_source_month_share", 1.0),
    (NUM_SOURCE_MONTHS, DEN_ACTIVE_MONTHS_POSTS):
        LabelCheck(None, 1.0),
}

# ---------------------------------------------------------------------------
# §13.7 时间限制
# ---------------------------------------------------------------------------

FIRST_HALF_MONTHS = tuple(range(1, 7))
SECOND_HALF_MONTHS = tuple(range(7, 13))
ALL_MONTHS = tuple(range(1, 13))
DEFAULT_MIN_ACTIVE_MONTHS = (3, 6)

# 活跃月份数的口径：分母含"只转发不发帖"的月份，与
# build_user_tables.combine_user_table 的持续性分母是同一列。
ACTIVE_MONTHS_COLUMN = "n_active_months_panel"

# 异常月份规则（**事先写死**，见模块文档）
ANOMALY_K = 3.0
MAD_SCALE = 1.4826
ANOMALY_MIN_RELATIVE_DEVIATION = 0.5
ANOMALY_VOLUME_VARIABLE = "+".join(smp.TOTAL_ACTIVITY_COLUMNS)
ANOMALY_RULE = (
    "abs(monthly_volume - median(monthly_volume)) > "
    "max({k} * {scale} * MAD(monthly_volume), "
    "{rel} * median(monthly_volume)); "
    "volume = {variable} summed over users within the month; "
    "two_sided; pre_specified_before_looking_at_any_outcome"
).format(k=ANOMALY_K, scale=MAD_SCALE,
         rel=ANOMALY_MIN_RELATIVE_DEVIATION, variable=ANOMALY_VOLUME_VARIABLE)

# 逐月行为量诊断行的 variant_label（它不是一个变体，是规则本身的账本）
ANOMALY_DIAGNOSTIC_LABEL = "anomaly_rule_monthly_behaviour_volume"

# 留一月重估的来源与"读不到"时的说明
LOMO_SOURCE_NOTE = (
    "source=models_temporal.leave_one_month_out;"
    "read_from_analysis_data/results;not_recomputed_here"
)
LOMO_UNAVAILABLE_LABEL = (
    "leave_one_month_out_unavailable;denominator=user_months_with_month_fixed_effects"
)
LOMO_UNAVAILABLE_NOTE = (
    "models_temporal_output_not_found:"
    "run_python_-m_gender_domain.models_temporal_build_first;"
    "leave_one_month_out_must_not_be_recomputed_in_the_robustness_layer"
)

# ---------------------------------------------------------------------------
# 诊断表 schema
# ---------------------------------------------------------------------------
#
# 逐 (family, variant_label) 一行；异常月份规则额外逐月一行。分五组：
# 1) 身份与规则；2) 这一行测的是什么（分子/分母/装在哪一列）；
# 3) 月份与行为量（异常月规则的完整账本）；4) 帖子类型与选中帖数；
# 5) 样本量与逐性别的剔除量——一条对男女生效比例不同的限制会改变被比较的
#    两个总体，只报一个总数看不出这件事。
DIAGNOSTIC_SCHEMA = (
    "variant_family",
    "variant_label",
    "rule",
    "measure_column",
    "numerator",
    "denominator",
    "month",
    "monthly_volume",
    "volume_median",
    "volume_mad",
    "volume_deviation",
    "anomaly_threshold",
    "is_anomalous",
    "months_retained",
    "months_excluded",
    "post_types",
    "n_posts_selected",
    "n_posts_total",
    "n_users_before",
    "n_users_after",
    "n_male_before",
    "n_female_before",
    "n_male_after",
    "n_female_after",
    "n_dropped_male",
    "n_dropped_female",
    "dropped_share_male",
    "dropped_share_female",
    "note",
)

MeasureContext = namedtuple(
    "MeasureContext",
    ["year", "user_table", "panel", "vocab", "incidence", "lomo", "monthly_rates",
     "source_paths"],
)


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def robustness_dir():
    """稳健性层唯一允许写入的目录"""
    return os.path.join(config.OUTPUT_DIR, "robustness")


def results_path():
    return os.path.join(robustness_dir(), "measures.parquet")


def diagnostics_path():
    return os.path.join(robustness_dir(), "measure_diagnostics.parquet")


def manifest_dir(year):
    return os.path.join(robustness_dir(), "measures_{}".format(year))


def temporal_results_dir():
    """models_temporal 写结果的目录（对本模块**只读**）"""
    return os.path.join(config.OUTPUT_DIR, "results")


def leave_one_month_out_path():
    return os.path.join(temporal_results_dir(), mt.LOMO_FILE)


def monthly_rates_path():
    return os.path.join(temporal_results_dir(), mt.MONTHLY_FILE)


# ---------------------------------------------------------------------------
# 上下文
# ---------------------------------------------------------------------------

def _load_user_table(year):
    path = os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到表 C: {}（由 python -m gender_domain.build_user_tables build "
            "生成）".format(path)
        )
    frame = pd.read_parquet(path, engine="pyarrow", columns=USER_TABLE_COLUMNS)
    frame["user_id"] = ir.normalize_id_series(frame["user_id"])
    print("表 C {:,} 个用户".format(len(frame)))
    return frame, path


def _load_panel(year):
    path = os.path.join(config.OUTPUT_DIR, "user_month_domain_{}.parquet".format(year))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到表 D: {}（由 python -m gender_domain.build_user_tables build "
            "生成）。§13.7 的月份限制要靠它重新聚合逐月的进入指示。".format(path)
        )
    frame = pd.read_parquet(path, engine="pyarrow", columns=PANEL_COLUMNS)
    frame["user_id"] = ir.normalize_id_series(frame["user_id"])
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").astype("Int64")
    print("表 D {:,} 个用户-月".format(len(frame)))
    return frame, path


def _read_temporal_output(path, label):
    """读 models_temporal 的一份结果表；不存在时返回 None 并说明

    **只读，绝不重算。** 读不到的后果是一行注明原因的 NaN 行，而不是本模块
    自己拟合一个"留一月估计"——同一个会被写进论文的数字由两个模块各算一遍，
    迟早会不一致，而读者没有办法判断该信哪一个。
    """
    if not os.path.exists(path):
        print(
            "提示: 未找到 models_temporal 的{}（{}）。§13.7 的这一部分会输出"
            "一行注明原因的结果，**不会**在稳健性层重新拟合。".format(label, path)
        )
        return None
    frame = pd.read_parquet(path, engine="pyarrow", columns=list(su.RESULT_SCHEMA))
    print("读取 models_temporal 的{}: {}（{} 行）".format(label, path, len(frame)))
    return frame


def build_context(year=config.YEAR, domains=DOMAINS):
    """加载表 C、表 D、两份词表、逐领域的帖子×词矩阵，以及 models_temporal 的输出"""
    user_table, user_path = _load_user_table(year)
    panel, panel_path = _load_panel(year)

    vocab = {}
    incidences = {}
    for domain in domains:
        vocab[domain] = voc.load_vocabulary(domain, year)
        # 本族从头到尾没读过 posts["weibo_id"]（§13.4 才需要它回溯原帖），
        # 那是 posts 帧里最大的一列，不载入直接省掉真实规模下的约 3.4 GB。
        incidences[domain] = inc.build_post_term_incidence(
            year, domain, keep_weibo_id=False)

    lomo = _read_temporal_output(leave_one_month_out_path(), "留一月重估")
    monthly = _read_temporal_output(monthly_rates_path(), "分月参与率")

    context = MeasureContext(
        year=year, user_table=user_table, panel=panel, vocab=vocab,
        incidence=incidences, lomo=lomo, monthly_rates=monthly,
        source_paths={
            "user_table": user_path,
            "panel": panel_path,
            "leave_one_month_out": leave_one_month_out_path(),
            "monthly_rates": monthly_rates_path(),
        },
    )
    check_primary_reconstruction(context, domains)
    return context


# ---------------------------------------------------------------------------
# 重建检验：与基线逐字相同的那几个"变体"改成在这里被核验，而不是产出重复行
# ---------------------------------------------------------------------------

def _mismatch_count(expected, got, tolerance=0.0):
    """逐元素比对两个已经对齐的数组，NaN 与 NaN 算相等，返回不一致的个数

    占比列里 NaN 有确切含义（分母为 0），把它当成"对不上"会让每一次重建
    检验都在零表达帖用户身上假报警；把它当成"随便什么都算相等"又会漏掉真的
    错误。所以两边都是 NaN 才算相等，一边是 NaN 一边不是就算不一致。
    """
    expected = np.asarray(expected, dtype=float)
    got = np.asarray(got, dtype=float)
    both_nan = np.isnan(expected) & np.isnan(got)
    close = np.isclose(expected, got, rtol=0.0, atol=tolerance, equal_nan=False)
    return int((~(both_nan | close)).sum())


def check_primary_reconstruction(context, domains=DOMAINS, tolerance=1e-9):
    """主口径的帖子类型重聚合必须与表 C 逐用户相等——**对不上直接报错**

    这是本模块的"全集重建"检验（词表侧的同名检验在 incidence，账号侧在
    accounts）。它同时也是 §13.8 主口径那一"变体"的去处：它与基线逐字相同，
    因此不产出结果行，改为在这里被核验（结论写进 manifest）。

    三样都要比：分子（topical_posts）、**分母（n_expressive_posts）**、以及
    占比本身。只比分子会让"分子分母同时错、比值恰好对上"的情形溜过去，而
    分母正是这一族全部要害所在。

    这个检验licenses 本族每一个按帖子类型或按月份重新聚合的变体，因此它
    **不能只打印一句警告然后接着算**：对不上就说明后面每一个数都在同一个
    方向上错，接着算出来的东西没有任何人应该读。
    """
    report = {}
    for domain in domains:
        result = topical_for_post_types(context, domain, PRIMARY_POST_TYPES)
        expected = context.user_table.set_index("user_id")
        got = result.set_index("user_id")
        expected = expected.reindex(got.index)
        if int(expected.index.isna().sum()) or len(got) != len(context.user_table):
            raise ValueError(
                "{} 的主口径重聚合覆盖的用户集合与表 C 不一致（重聚合 {} 人、"
                "表 C {} 人）：分母的用户全集对不上，后面每一个变体都会错。".format(
                    domain, len(got), len(context.user_table))
            )
        checks = {
            "topical_posts": _mismatch_count(
                expected["{}_topical_posts".format(domain)], got["topical_posts"]),
            "n_expressive_posts": _mismatch_count(
                expected["n_expressive_posts"], got["n_expressive_posts"]),
            "topical_share": _mismatch_count(
                expected["{}_topical_share".format(domain)], got["topical_share"],
                tolerance=tolerance),
        }
        bad = {name: n for name, n in checks.items() if n}
        if bad:
            raise ValueError(
                "{} 的主口径帖子类型重聚合与表 C 对不上（逐列不一致用户数 {}）——"
                "§13.8 的帖子类型变体与 §13.7 的按月重聚合都会在同一个方向上错，"
                "请先查 incidence 的重建口径，不要读本次的任何结果。".format(domain, bad)
            )
        report[domain] = {"n_users": int(len(got)), "verified_equal_to_table_c": True}
        print("{} 主口径重聚合 == 表 C（{} 个用户，分子/分母/占比三者全等）".format(
            domain, len(got)))
    return report


def check_measure_matches_its_label(context, spec, domains=DOMAINS, tolerance=1e-9):
    """label 上写的分子/分母，必须就是这一行真的算出来的东西

    做法见 LABEL_TO_TABLE_C 上方的说明：按 **(分子, 分母) 这一对进 label 的
    字符串** 反查表 C 里的权威列，逐用户比对。把两条 spec 的分母对调之后，
    查表会落空或查到另一列，这里当场报错——否则一个自信地写错的分母会一路
    走到结果表上，而这一族存在的唯一意义就是分母。
    """
    key = (spec.numerator, spec.denominator)
    if key not in LABEL_TO_TABLE_C:
        raise ValueError(
            "替代测量 {} 的 (分子, 分母) = {} 没有在 LABEL_TO_TABLE_C 里登记。"
            "每一条 spec 都必须登记，否则'改了 label 就查不到、于是不检查'会变成"
            "一个静默的逃生口——而 label 与实际测量对不上正是本族最不能出的错。"
            .format(spec.key, key)
        )
    check = LABEL_TO_TABLE_C[key]
    report = {}
    for domain in domains:
        computed = topical_for_measure(context, domain, spec.key)
        if check.column is None:
            # 表 C 里确实没有对应列（只看发帖的活跃月份分母正是这个变体要
            # 引入的东西）。仍然登记在册、仍然算一遍，只是没有可对照的真值。
            report[domain] = {"authoritative_column": None,
                              "n_users": int(len(computed)),
                              "verified": "no_column_in_table_c_by_design"}
            continue
        column = check.column.format(domain)
        if column not in context.user_table.columns:
            raise ValueError(
                "{} 声称对应表 C 的 {} 列，但表 C 里没有这一列".format(spec.key, column))
        merged = context.user_table[["user_id", column]].merge(
            computed[["user_id", "topical_share"]], on="user_id", how="left")
        mismatch = _mismatch_count(
            merged[column], merged["topical_share"] * check.scale,
            tolerance=tolerance)
        if mismatch:
            raise ValueError(
                "label 与实际测量对不上: {} 的 label 写着 numerator={} / "
                "denominator={}，按这个 label 应当等于表 C 的 {} 列，但有 {} 个"
                "用户对不上。一个写错的分母比一个没写的分母更糟——本族存在的"
                "唯一意义就是分母。".format(
                    domain, spec.numerator, spec.denominator, column, mismatch)
            )
        report[domain] = {"authoritative_column": column, "scale": check.scale,
                          "verified": True}
    return report


def check_full_year_reconstruction(context, tolerance=1e-9):
    """保留全部 12 个月的按月重聚合必须回到表 C

    §13.7 的"剔除异常月份"在规则一个月都没选中时，退化成一个与基线逐字相同
    的全年重估。那一版不作为变体产出（幽灵变体的道理见 PostTypeVariant 上方
    的说明），改为在这里被核验：全月份窗口的活动量与进入指示必须等于表 C。
    """
    frame = user_frame_for_months(context, ALL_MONTHS)
    expected = context.user_table.set_index("user_id").reindex(
        frame.set_index("user_id").index)
    got = frame.set_index("user_id")
    if len(got) != len(context.user_table):
        raise ValueError(
            "保留全部 12 个月的重聚合覆盖 {} 人，表 C 有 {} 人：按月重聚合的"
            "用户全集对不上。".format(len(got), len(context.user_table)))
    bad = {}
    for column in ("n_posts", "n_retweets", "n_active_days"):
        n = _mismatch_count(expected[column], got[column])
        if n:
            bad[column] = n
    for domain in DOMAINS:
        column = "{}_source_entered".format(domain)
        n = _mismatch_count(expected[column].astype(float),
                            got[column].astype(float))
        if n:
            bad[column] = n
    if bad:
        raise ValueError(
            "保留全部 12 个月的重聚合与表 C 对不上（逐列不一致用户数 {}）——"
            "§13.7 的半年限制与异常月份限制都会在同一个方向上错。".format(bad))
    print("全年（12 个月）按月重聚合 == 表 C（{} 个用户）".format(len(got)))
    return {"n_users": int(len(got)), "verified_equal_to_table_c": True}


def verify_reconstructions(context, domains=DOMAINS):
    """本族全部"与基线逐字相同"的口径都在这里被核验，结论进 manifest

    这三条替代的是三个原本会被产出、却与基线一模一样的"变体"：
      - §13.1 的 entry_sample=all_users（帧原样不动，就是基线本身）；
      - §13.1 的 topical_measure=expressive_posts（主口径）与 §13.8 的
        post_types=original+retweet_comment（同一件事的两个入口）；
      - §13.7 在异常月份规则一个月都没选中时的全年重估。
    产出重复行会在分布正中心多出幽灵变体；核验相等并把结论记下来，既去掉了
    幽灵、又比那一行重复的估计更有信息量（它证明重建路径是对的）。
    """
    report = {
        "primary_post_type_reconstruction": check_primary_reconstruction(
            context, domains),
        "full_year_month_reconstruction": check_full_year_reconstruction(context),
        "entry_denominator_all_users_is_the_baseline": {
            "n_users": int(len(context.user_table)),
            "n_male": int((context.user_table["gender"] == "m").sum()),
            "n_female": int((context.user_table["gender"] == "f").sum()),
            "note": ("the_main_specification_already_uses_all_same_gender_users;"
                     "emitting_it_as_a_variant_would_duplicate_the_baseline"),
        },
        "measure_label_matches_measurement": {
            spec.key: check_measure_matches_its_label(context, spec, domains)
            for spec in MEASURE_SPECS
        },
    }
    return report


# ---------------------------------------------------------------------------
# 帖子类型掩码：§13.8 与 §13.1 的「全部帖子」分母共用同一个实现
# ---------------------------------------------------------------------------

def post_type_mask(posts, post_types, require_nonzero_chars=True):
    """逐帖布尔掩码：属于给定帖子类型（可选：且清洗后字符数 > 0）

    `require_nonzero_chars=True` 时，取 PRIMARY_POST_TYPES 得到的掩码与表 A
    的 is_expressive 列逐帖相等（text_rules.is_expressive 的定义就是"类型
    属于 EXPRESSIVE_TYPES 且 n_chars > 0"）——这正是主口径必须重现表 C 的
    原因，也是这个参数存在的原因：§13.1 的「全部帖子」分母**刻意**不排除
    零字符帖（见模块文档），因此要把它设成 False。
    """
    selected = posts["post_type"].isin(list(post_types)).to_numpy(dtype=bool)
    if require_nonzero_chars:
        selected = selected & (posts["n_chars"].to_numpy(dtype=np.int64) > 0)
    return selected


def posts_with_selection(posts, post_types, require_nonzero_chars=True,
                         months=None):
    """把 posts 帧的 is_expressive 换成"这次选中的帖子"，其余列原样保留

    incidence.topical_by_user / char_measures_by_user 的分子分母都由 posts
    帧的 is_expressive 列定义，而 posts 参数是必填的（"哪一份帖子集合定义
    了分母"必须由调用方明说）。因此换掉这一列就是换分母的正确做法，不需要
    往 incidence 里加任何东西。

    months 不为 None 时**按行过滤**（而不是并进掩码）：月份限制要的是
    "这段窗口里这个用户长什么样"，窗口外的帖子连同它们的用户都不该出现在
    这一版的聚合结果里；而帖子类型限制要的是"同一批用户换一个分母"，窗口
    外的帖子仍然属于这个用户，只是不进分子分母。两者语义不同，所以一个
    过滤行、一个改掩码。
    """
    frame = posts if months is None else posts[
        posts["month"].isin(list(months)).to_numpy(dtype=bool)
    ]
    out = frame.copy()
    out["is_expressive"] = post_type_mask(
        out, post_types, require_nonzero_chars=require_nonzero_chars)
    return out


def topical_for_post_types(context, domain, post_types,
                           require_nonzero_chars=True, months=None):
    """给定帖子类型（与可选的月份窗口）下的逐用户 topical 指标

    词表恒为全词表：本模块改的是分母，不是词表——那是 §13.3 的事。
    """
    incidence = context.incidence[domain]
    posts = posts_with_selection(
        incidence.posts, post_types,
        require_nonzero_chars=require_nonzero_chars, months=months)
    return inc.topical_by_user(incidence, context.vocab[domain], posts)


def char_for_post_types(context, domain, post_types=PRIMARY_POST_TYPES,
                        require_nonzero_chars=True, months=None):
    """给定帖子类型下的逐用户字符口径指标（§13.1 的字符密度与每字命中数）"""
    incidence = context.incidence[domain]
    posts = posts_with_selection(
        incidence.posts, post_types,
        require_nonzero_chars=require_nonzero_chars, months=months)
    return inc.char_measures_by_user(incidence, context.vocab[domain], posts)


# ---------------------------------------------------------------------------
# §13.1 替代测量：全部写进 {domain}_topical_share 这一列
# ---------------------------------------------------------------------------

def _measure_frame(user_id, topical_posts, topical_share):
    """voc.user_frame_with_topical 需要的三列形状"""
    return pd.DataFrame({
        "user_id": np.asarray(user_id, dtype=object).astype(str),
        "topical_posts": np.asarray(topical_posts, dtype=np.float64),
        "topical_share": np.asarray(topical_share, dtype=np.float64),
    })


def topical_for_measure(context, domain, measure):
    """某个替代测量下的逐用户 (topical_posts, topical_share)

    六个量写死在 harness 里，因此每一种替代测量都只能写进
    `{domain}_topical_share` 这一列，由同一个分数 logit 估同一个形状的量。
    label 与 note 负责说清楚这一列此刻装的到底是什么（见模块文档）。
    """
    if measure == MEASURE_EXPRESSIVE_SHARE:
        result = topical_for_post_types(context, domain, PRIMARY_POST_TYPES)
        return _measure_frame(result["user_id"], result["topical_posts"],
                              result["topical_share"])

    if measure == MEASURE_ALLPOSTS_SHARE:
        # 「全部帖子」分母刻意不排除零字符帖，与表 C 的
        # {domain}_topical_share_allposts 是同一个口径（见模块文档）
        result = topical_for_post_types(
            context, domain, ALL_TEXT_POST_TYPES, require_nonzero_chars=False)
        return _measure_frame(result["user_id"], result["topical_posts"],
                              result["topical_share"])

    if measure in (MEASURE_CHAR_DENSITY, MEASURE_HITS_PER_CHAR):
        result = char_for_post_types(context, domain)
        if measure == MEASURE_CHAR_DENSITY:
            share = result["char_density"]
        else:
            # 每千字命中数不是占比（100 字命中 1 次 = 每千字 10 次），
            # 分数 logit 会因越界直接报错。换算成每字命中数：恒在 [0,1]，
            # 与每千字口径只差常数因子 1000（见模块文档）。
            share = result["hits_per_1k"] / 1000.0
        return _measure_frame(result["user_id"], result["n_hits"], share)

    if measure in (MEASURE_SOURCE_MONTHS_PANEL, MEASURE_SOURCE_MONTHS_POSTS):
        frame = context.user_table
        months = frame["{}_source_months".format(domain)].astype(float)
        if measure == MEASURE_SOURCE_MONTHS_PANEL:
            denominator = frame[ACTIVE_MONTHS_COLUMN].astype(float)
        else:
            denominator = frame["n_active_months"].astype(float)
        share = but._safe_divide(months, denominator)
        # 只转发不发帖的月份进得了 source_months、进不了"只看帖子"的活跃
        # 月份数，因此后一个分母下份额可能 > 1。与
        # build_user_tables.combine_user_table 对 source_share 的处理一致，
        # 截到 1 并在 note 里说明，绝不让越界值进分数 logit。
        share = share.clip(upper=1.0)
        return _measure_frame(frame["user_id"], months, share)

    raise ValueError("未知的替代测量: {}，只支持 {}".format(
        measure, sorted(MEASURE_BY_KEY)))


# ---------------------------------------------------------------------------
# §13.7 月份限制：进入指示与活动量从表 D 重新聚合
# ---------------------------------------------------------------------------

def monthly_behaviour_volume(panel):
    """逐月的行为量（发帖数 + 转发帖数，在该月的全部用户上求和）

    **只看行为量，不看任何一个结果变量。** 异常月份的规则完全建立在这个
    序列上；用结果变量挑月份就是在对因变量做选择，而读者从输出上分辨不出
    这两种做法的差别。定义沿用 samples.TOTAL_ACTIVITY_COLUMNS，不另立一个。
    """
    frame = panel.copy()
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
    frame = frame[frame["month"].notna()]
    volume = smp.total_activity(frame)
    grouped = pd.Series(volume.to_numpy(dtype=float),
                        index=frame["month"].astype(int).to_numpy())
    out = grouped.groupby(level=0).sum().sort_index()
    out.index.name = "month"
    out.name = "monthly_volume"
    return out


def anomalous_months(volumes):
    """按事先写死的规则选出异常月份，返回 (月份元组, 规则的完整账本)

    规则（模块文档里有完整说明与理由）：
        |v_m - median| > max(3.0 × 1.4826 × MAD, 0.5 × median)

    两个条件同时要求（取 max）：Hampel 那一半让规则对量级免疫、不被单个
    极端月份掩蔽；50% 的相对偏差下限挡住"MAD 塌到 0 之后任何微小波动都算
    异常"的退化情形。
    """
    values = pd.Series(volumes).astype(float).sort_index()
    info = {
        "median": float("nan"), "mad": float("nan"), "threshold": float("nan"),
        "deviations": {}, "rule": ANOMALY_RULE,
        "volume_variable": ANOMALY_VOLUME_VARIABLE,
    }
    if values.empty:
        return (), info

    median = float(np.median(values.to_numpy()))
    deviations = (values - median).abs()
    mad = float(np.median(deviations.to_numpy()))
    threshold = max(ANOMALY_K * MAD_SCALE * mad,
                    ANOMALY_MIN_RELATIVE_DEVIATION * abs(median))
    flagged = tuple(int(m) for m in values.index[deviations > threshold])

    info.update({
        "median": median, "mad": mad, "threshold": float(threshold),
        "deviations": {int(m): float(d) for m, d in deviations.items()},
        "volumes": {int(m): float(v) for m, v in values.items()},
        "flagged": list(flagged),
    })
    print(
        "异常月份规则: 中位数 {:.1f}，MAD {:.1f}，阈值 {:.1f}，选中 {}".format(
            median, mad, threshold, list(flagged) or "无")
    )
    return flagged, info


def user_frame_for_months(context, months):
    """只保留给定月份时的用户帧（活动量、进入指示、内容测量全部重算）

    进入指示来自表 D 逐月的 `{domain}_source_count` 求和（> 0 即为进入）：
    用户×账号矩阵没有月份轴，表 D 才是"逐月的进入"这件事的权威来源。
    活动量协变量（发帖、转发、活跃天数、活跃月份数）同样只在窗口内累计——
    否则 M1 会拿全年的活动量去调整一个只有半年的结果变量。
    内容测量来自 incidence 的 posts 帧按 month 过滤后重新聚合。

    窗口内完全没有活动的用户不在返回结果里：他们在这段窗口里不存在，
    给他们造一行"0 帖 0 参与"是编出来的观测。
    """
    months = tuple(int(m) for m in months)
    panel = context.panel
    sub = panel[panel["month"].isin(list(months)).to_numpy(dtype=bool)]

    aggregation = {
        "n_posts": ("n_posts", "sum"),
        "n_retweets": ("n_retweets", "sum"),
        "n_active_days": ("n_active_days", "sum"),
        "n_active_months": ("month", "nunique"),
    }
    for domain in DOMAINS:
        column = "{}_source_count".format(domain)
        aggregation[column] = (column, "sum")
    grouped = sub.groupby("user_id", observed=True).agg(**aggregation).reset_index()

    frame = context.user_table[["user_id", "gender", "region"]].merge(
        grouped, on="user_id", how="inner")
    for column in ("n_posts", "n_retweets", "n_active_days", "n_active_months"):
        frame[column] = frame[column].fillna(0).astype(np.int64)
    for domain in DOMAINS:
        count_col = "{}_source_count".format(domain)
        frame[count_col] = frame[count_col].fillna(0).astype(np.int64)
        frame["{}_source_entered".format(domain)] = frame[count_col] > 0
    # 活跃月份数在这一版里就是窗口内的活跃月份数，两列保持一致，免得下游
    # 拿到一个"窗口内 3 个月、面板列却写着 12 个月"的自相矛盾的用户帧
    frame[ACTIVE_MONTHS_COLUMN] = frame["n_active_months"]
    frame["n_expressive_posts"] = 0

    topical = {}
    for domain in DOMAINS:
        result = topical_for_post_types(
            context, domain, PRIMARY_POST_TYPES, months=months)
        topical[domain] = result
    frame = voc.user_frame_with_topical(frame, topical)
    # 表达帖数取内容测量那一侧的分母，保持同一个窗口口径
    denominators = topical[DOMAINS[0]][["user_id", "n_expressive_posts"]]
    merged = frame[["user_id"]].merge(denominators, on="user_id", how="left")
    frame["n_expressive_posts"] = merged["n_expressive_posts"].fillna(0).to_numpy(
        dtype=np.int64)
    return frame


def active_month_restriction(user_table, min_months):
    """只保留活跃月份数 >= min_months 的用户

    活跃月份数取 n_active_months_panel（分母含只转发不发帖的月份），与
    build_user_tables.combine_user_table 的持续性分母是同一列——两处用不同
    的"活跃月份"会让同一个词在两张表里指不同的东西。
    """
    values = pd.to_numeric(user_table[ACTIVE_MONTHS_COLUMN], errors="coerce")
    keep = (values >= int(min_months)).fillna(False).to_numpy(dtype=bool)
    return user_table[keep].copy()


# ---------------------------------------------------------------------------
# 诊断行
# ---------------------------------------------------------------------------

def _gender_counts(frame):
    return (int((frame["gender"] == "m").sum()),
            int((frame["gender"] == "f").sum()))


def diagnostics_row(variant_family, variant_label, rule=None, before=None,
                    after=None, measure_column=None, numerator=None,
                    denominator=None, month=None, monthly_volume=None,
                    volume_median=None, volume_mad=None, volume_deviation=None,
                    anomaly_threshold=None, is_anomalous=None,
                    months_retained=None, months_excluded=None, post_types=None,
                    n_posts_selected=None, n_posts_total=None, note=None):
    """一行诊断：这个变体测的是什么、动了哪些月/哪些帖子、剩下的样本长什么样

    逐性别的剔除量单独成列，因为一条对男女生效比例不同的限制会改变被比较
    的两个总体，只报一个总数看不出这件事。
    """
    n_male_before = n_female_before = None
    n_male_after = n_female_after = None
    if before is not None:
        n_male_before, n_female_before = _gender_counts(before)
    if after is not None:
        n_male_after, n_female_after = _gender_counts(after)

    def _dropped(before_n, after_n):
        if before_n is None or after_n is None:
            return None, None
        dropped = before_n - after_n
        return dropped, (dropped / before_n if before_n else float("nan"))

    n_dropped_male, share_male = _dropped(n_male_before, n_male_after)
    n_dropped_female, share_female = _dropped(n_female_before, n_female_after)

    row = {
        "variant_family": variant_family,
        "variant_label": variant_label,
        "rule": rule,
        "measure_column": measure_column,
        "numerator": numerator,
        "denominator": denominator,
        "month": None if month is None else int(month),
        "monthly_volume": _as_float(monthly_volume),
        "volume_median": _as_float(volume_median),
        "volume_mad": _as_float(volume_mad),
        "volume_deviation": _as_float(volume_deviation),
        "anomaly_threshold": _as_float(anomaly_threshold),
        "is_anomalous": is_anomalous,
        "months_retained": _format_months(months_retained),
        "months_excluded": _format_months(months_excluded),
        "post_types": None if post_types is None else "+".join(post_types),
        "n_posts_selected": _as_int(n_posts_selected),
        "n_posts_total": _as_int(n_posts_total),
        "n_users_before": None if before is None else int(len(before)),
        "n_users_after": None if after is None else int(len(after)),
        "n_male_before": n_male_before,
        "n_female_before": n_female_before,
        "n_male_after": n_male_after,
        "n_female_after": n_female_after,
        "n_dropped_male": n_dropped_male,
        "n_dropped_female": n_dropped_female,
        "dropped_share_male": share_male,
        "dropped_share_female": share_female,
        "note": note,
    }
    return {key: row.get(key) for key in DIAGNOSTIC_SCHEMA}


def _as_float(value):
    return None if value is None else float(value)


def _as_int(value):
    return None if value is None else int(value)


def _format_months(months):
    if months is None:
        return None
    if not len(months):
        return "none"
    return ",".join("{:02d}".format(int(m)) for m in sorted(months))


def _append_frame(rows, path, schema, label):
    """按给定 schema 增量落盘（与 samples._append_frame 同一套原子改名做法）

    单独实现是因为这张附表的 schema 不是 ROBUSTNESS_SCHEMA：
    harness.append_rows 读旧文件时显式点名共享 schema 的列，用它写诊断表
    会把诊断列全丢掉。与 harness.append_rows 一样**不去重**：同一个 year
    跑两次 build() 会让表翻倍（重跑前先删旧文件）。
    """
    if path is None:
        return None
    frame = pd.DataFrame(list(rows), columns=list(schema))
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path, engine="pyarrow", columns=list(schema))
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.reset_index(drop=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".append_mea_tmp_", suffix=".parquet", dir=out_dir)
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print("已增量写入{}: {}（新增 {} 行，累计 {} 行）".format(
        label, path, len(frame), len(combined)))
    return path


def append_diagnostics(rows, path):
    return _append_frame(rows, path, DIAGNOSTIC_SCHEMA, "诊断")


# ---------------------------------------------------------------------------
# 跑一个变体
# ---------------------------------------------------------------------------

def _measure_note(diagnostic, extra=None):
    """把"这一行测的是什么、分母是什么、剩下多少人"压成一句写进结果行

    诊断表里有全套数字，但只拿到 measures.parquet 一张表的读者也必须看得见
    这一行的分母——这一族的全部意义就在这件事上。
    """
    parts = []
    if diagnostic.get("numerator"):
        parts.append("numerator={}".format(diagnostic["numerator"]))
    if diagnostic.get("denominator"):
        parts.append("denominator={}".format(diagnostic["denominator"]))
    if diagnostic.get("measure_column"):
        parts.append("topical_share_column_holds={}".format(
            diagnostic["measure_column"]))
    if diagnostic.get("months_retained"):
        parts.append("months_retained={}".format(diagnostic["months_retained"]))
    if diagnostic.get("post_types"):
        parts.append("post_types={}".format(diagnostic["post_types"]))
    if diagnostic.get("n_users_after") is not None:
        parts.append("n_users={}".format(diagnostic["n_users_after"]))
    if diagnostic.get("n_dropped_male") is not None:
        parts.append("n_dropped_male={}".format(diagnostic["n_dropped_male"]))
        parts.append("n_dropped_female={}".format(diagnostic["n_dropped_female"]))
    if extra:
        parts.append(extra)
    return ";".join(parts)


def _quantity_row_mask(rows, quantities):
    """结果行里属于给定几个量的那些行（按 harness 自己的 (outcome, domain, term)）

    三元组从 harness.QUANTITY_META 现场取，不在本模块另抄一份：六个量分别是
    什么，只能有一个来源。
    """
    mask = np.zeros(len(rows), dtype=bool)
    for quantity in quantities:
        meta = harness.QUANTITY_META[quantity]
        mask |= (
            (rows["outcome"] == meta["outcome"]).to_numpy(dtype=bool)
            & (rows["domain"] == meta["domain"]).to_numpy(dtype=bool)
            & (rows["term"] == meta["term"]).to_numpy(dtype=bool)
        )
    return mask


def nan_out_quantities(rows, quantities, note, n_input):
    """把给定几个量的行写成 NaN 并注明原因，**行数不变**

    用途见 ENTRY_QUANTITIES 上方的长注释：一个按构造动不到某个量的变体，
    若照常产出那个量，会在分布正中心多出一堆与基线逐字相同的行，把方向
    一致率抬高、把离散度压低。NaN 是本套件既有的"这里没测"信号（harness
    的 _nan_quantity_row 用的就是它），因此这里沿用同一个形状：数值列全部
    NaN、n_obs 记 0、n_dropped 记全部输入用户，并在 note 上追加原因。

    保留全部行是刻意的：estimate_all 的"恰好 len(QUANTITIES) × len(layers)
    行"这条不变量，是下游判断某个变体是否真的跑过的唯一依据。
    """
    mask = _quantity_row_mask(rows, quantities)
    if not mask.any():
        return rows
    out = rows.copy()
    # note 的拼接规则一律走 voc._annotate_note（本套件唯一的那一套），
    # 只是作用在被选中的这几行上——它按设计盖整帧，所以要先切片再贴回去
    patched = voc._annotate_note(out.loc[mask], note)
    for column in ("estimate", "se", "ci_low", "ci_high"):
        patched[column] = np.nan
    patched["n_obs"] = 0
    patched["n_dropped"] = int(n_input)
    patched["drop_reason"] = None
    out.loc[mask, list(harness.ROBUSTNESS_SCHEMA)] = patched[
        list(harness.ROBUSTNESS_SCHEMA)].to_numpy(dtype=object)
    return out


def _run_variant(frame, variant_family, variant_label, diagnostic, out_path,
                 diag_path, extra_note=None, nan_quantities=None,
                 nan_note=None):
    """一个测量变体：估一次六个量、写一行诊断，两处都立刻落盘

    nan_quantities 不为空时，那几个量的行在落盘前被写成 NaN（行数不变）：
    这个变体按构造动不到它们，照常产出会在结果表里制造与基线逐字相同的
    幽灵行。
    """
    rows = harness.estimate_all(
        frame, variant_family=variant_family, variant_label=variant_label,
        replicate=0, seed=None,
    )
    rows = voc._annotate_note(rows, _measure_note(diagnostic, extra=extra_note))
    if nan_quantities:
        rows = nan_out_quantities(rows, nan_quantities, nan_note, len(frame))
    if out_path is not None:
        harness.append_rows(rows, out_path)
    append_diagnostics([diagnostic], diag_path)
    return rows


def _finding_row(variant_family, variant_label, note, out_path):
    """把"这一族按构造动不到某个量"这个发现，用一行说一次

    走 vocabulary._note_only_rows（samples 处理做不到的变体、accounts 处理
    缺类别时用的同一套做法），不新造第二种"说明性行"。
    """
    return voc._note_only_rows(
        variant_label, note, outcome=mt.OUTCOME_ENTRY, domain="public",
        out_path=out_path, variant_family=variant_family,
    )


def _apply_measure(context, user_table, measure):
    """把某个替代测量写进 {domain}_topical_share / {domain}_topical_posts"""
    topical = {
        domain: topical_for_measure(context, domain, measure)
        for domain in DOMAINS
    }
    return voc.user_frame_with_topical(user_table, topical)


# ---------------------------------------------------------------------------
# §13.1 分母变体
# ---------------------------------------------------------------------------

def run_denominator_variants(year=config.YEAR, context=None, out_path=None,
                             diag_path=None):
    """§13.1：进入指示的分母（只看转发过的人）、话题占比的分母（全部帖子）、
    字符密度、每字（千字）命中数、领域月份 ÷ 活跃月份

    两条口径**不作为变体产出结果行**，因为它们与基线逐字相同：
      - 进入指示"以全部同性别用户为分母"就是主口径本身（帧原样不动）；
      - 话题占比"以表达帖为分母"同样是主口径。
    它们改由 verify_reconstructions 核验并写进 manifest（理由见
    PostTypeVariant 上方的说明）。

    替代测量只改内容测量，碰不到进入指示，因此这些变体的 entry_public /
    entry_celebrity / did_entry 三个量一律写成 NaN，"进入指示不受影响"这个
    发现用一行说一次。

    每一个 variant_label 都写成 `X=...;denominator=...`，见模块文档。
    """
    context = context or build_context(year)
    frame = context.user_table
    collected = []

    # --- 进入指示的分母：只有"只看转发过的人"是一个真的变体 ---
    subset = frame[frame["n_retweets"].astype(float) > 0].copy()
    label = "entry_sample=retweeters_only;denominator={}".format(
        ENTRY_DENOMINATOR_RETWEETERS)
    diagnostic = diagnostics_row(
        VARIANT_FAMILY_DENOMINATOR, label,
        rule="restrict_to_users_who_retweeted_at_least_once",
        before=frame, after=subset,
        measure_column="{domain}_source_entered",
        numerator="users_who_entered_the_domain",
        denominator=ENTRY_DENOMINATOR_RETWEETERS,
        note="the_other_arm(denominator={})_is_the_baseline_itself_and_is_"
             "verified_rather_than_re_emitted".format(ENTRY_DENOMINATOR_ALL),
    )
    print("{}: 保留 {:,} / {:,} 人".format(label, len(subset), len(frame)))
    collected.append(_run_variant(
        subset, VARIANT_FAMILY_DENOMINATOR, label, diagnostic, out_path, diag_path))

    # --- 话题占比的分母/测量：全部写进 {domain}_topical_share 这一列 ---
    emitted = 0
    for spec in MEASURE_SPECS:
        # label 上写的分子/分母必须就是真的算出来的东西，对不上当场报错。
        # 主口径那一条也要查——它虽然不产出结果行，重建检验仍然要跑。
        check_measure_matches_its_label(context, spec)
        if spec.is_baseline:
            print("跳过与基线逐字相同的口径 {}（已核验重现表 C，不再产出重复行）"
                  .format(spec.key))
            continue
        label = "topical_measure={};denominator={}".format(
            spec.numerator, spec.denominator)
        user_df = _apply_measure(context, frame, spec.key)
        diagnostic = diagnostics_row(
            VARIANT_FAMILY_DENOMINATOR, label,
            rule="substitute_{}_into_topical_share_column".format(spec.key),
            before=frame, after=user_df,
            measure_column="{}/{}".format(spec.numerator, spec.denominator),
            numerator=spec.numerator, denominator=spec.denominator,
            note=spec.note,
        )
        print("{}: 已写入 topical_share 列".format(label))
        collected.append(_run_variant(
            user_df, VARIANT_FAMILY_DENOMINATOR, label, diagnostic, out_path,
            diag_path, extra_note=spec.note,
            nan_quantities=ENTRY_QUANTITIES, nan_note=NOTE_ENTRY_NOT_ESTIMATED))
        emitted += 1

    if emitted:
        collected.append(_finding_row(
            VARIANT_FAMILY_DENOMINATOR, FINDING_ENTRY_UNAFFECTED_MEASURE,
            FINDING_ENTRY_UNAFFECTED_MEASURE_NOTE, out_path))

    return pd.concat(collected, ignore_index=True)


# ---------------------------------------------------------------------------
# §13.8 帖子类型变体
# ---------------------------------------------------------------------------

def run_post_type_variants(year=config.YEAR, context=None, out_path=None,
                           diag_path=None):
    """§13.8：只看原创 / 只看纯转发 / 全部有字帖

    三个变体都按帖子类型重新聚合表 A（复用 Task 2 的逐帖矩阵与 posts 帧）。
    主口径（原创 + 带评论转发）**不在这里产出结果行**：它与基线逐字相同，
    改由 check_primary_reconstruction 核验它逐用户重现表 C（分子、分母、
    占比三者），结论写进 manifest。

    帖子类型碰不到进入指示（它来自表 B 的转发事件，不是内容测量），因此
    每个变体的 entry_public / entry_celebrity / did_entry 三个量一律写成
    NaN，这个发现用一行说一次——散文到不了下游的算术，NaN 到得了。
    """
    context = context or build_context(year)
    frame = context.user_table
    collected = []
    emitted = 0

    for variant in POST_TYPE_VARIANTS:
        if variant.is_baseline:
            print("跳过与基线逐字相同的口径 post_types={}（已核验重现表 C，"
                  "不再产出重复行）".format("+".join(variant.post_types)))
            continue
        label = "post_types={};denominator={}".format(
            "+".join(variant.post_types), variant.denominator)
        topical = {}
        for domain in DOMAINS:
            topical[domain] = topical_for_post_types(
                context, domain, variant.post_types,
                require_nonzero_chars=variant.require_nonzero_chars)
        posts = context.incidence[DOMAINS[0]].posts
        n_selected = int(post_type_mask(
            posts, variant.post_types,
            require_nonzero_chars=variant.require_nonzero_chars).sum())
        n_total = int(len(posts))

        user_df = voc.user_frame_with_topical(frame, topical)
        diagnostic = diagnostics_row(
            VARIANT_FAMILY_POST_TYPE, label,
            rule="reaggregate_table_A_over_post_types={}".format(
                "+".join(variant.post_types)),
            before=frame, after=user_df,
            measure_column="topical_posts/{}".format(variant.denominator),
            numerator="topical_posts_of_the_selected_types",
            denominator=variant.denominator,
            post_types=variant.post_types,
            n_posts_selected=n_selected, n_posts_total=n_total,
            note="entry_quantities_emitted_as_NaN:see_the_finding_row",
        )
        print("{}: 选中帖 {:,} / {:,}".format(label, n_selected, n_total))
        collected.append(_run_variant(
            user_df, VARIANT_FAMILY_POST_TYPE, label, diagnostic, out_path,
            diag_path,
            nan_quantities=ENTRY_QUANTITIES, nan_note=NOTE_ENTRY_NOT_ESTIMATED))
        emitted += 1

    if emitted:
        collected.append(_finding_row(
            VARIANT_FAMILY_POST_TYPE, FINDING_ENTRY_UNAFFECTED_POST_TYPE,
            FINDING_ENTRY_UNAFFECTED_POST_TYPE_NOTE, out_path))

    return pd.concat(collected, ignore_index=True)


# ---------------------------------------------------------------------------
# §13.7 时间限制
# ---------------------------------------------------------------------------

def _month_variant(context, months, label_prefix, rule, out_path, diag_path,
                   excluded=None, extra_note=None):
    """一个按月份窗口重新聚合的变体"""
    frame = context.user_table
    restricted = user_frame_for_months(context, months)
    label = "{};denominator=user_activity_within_months({})".format(
        label_prefix, _format_months(months))
    diagnostic = diagnostics_row(
        VARIANT_FAMILY_TEMPORAL, label, rule=rule, before=frame, after=restricted,
        measure_column="topical_posts/expressive_posts_within_the_window",
        numerator="entry_and_topical_within_the_retained_months",
        denominator="user_activity_within_months({})".format(_format_months(months)),
        months_retained=months, months_excluded=excluded,
        note=extra_note,
    )
    print("{}: 窗口内 {:,} / {:,} 人".format(label, len(restricted), len(frame)))
    return _run_variant(restricted, VARIANT_FAMILY_TEMPORAL, label, diagnostic,
                        out_path, diag_path, extra_note=extra_note)


def _anomaly_diagnostics(volumes, info, flagged):
    """异常月份规则的完整账本：逐月一行，写明量、偏差、阈值与是否被选中"""
    rows = []
    for month, volume in volumes.items():
        deviation = abs(float(volume) - info["median"])
        rows.append(diagnostics_row(
            VARIANT_FAMILY_TEMPORAL, ANOMALY_DIAGNOSTIC_LABEL,
            rule=ANOMALY_RULE,
            measure_column=ANOMALY_VOLUME_VARIABLE,
            numerator=ANOMALY_VOLUME_VARIABLE,
            denominator="not_applicable:this_row_records_the_rule_not_an_estimate",
            month=int(month), monthly_volume=float(volume),
            volume_median=info["median"], volume_mad=info["mad"],
            volume_deviation=deviation, anomaly_threshold=info["threshold"],
            is_anomalous=bool(int(month) in flagged),
            note="pre_specified_rule_applied_mechanically;"
                 "volume_is_behaviour_not_any_of_the_six_outcomes",
        ))
    return rows


def run_leave_one_month_out(year=config.YEAR, context=None, out_path=None):
    """§13.7 留一月重估：**从 models_temporal 的输出里读**，不重算

    models_temporal.leave_one_month_out 已经在带月份固定效应、按用户聚类
    标准误的规格下算过十二次留一月加一行全年参照。本函数把其中性别 AME 的
    行原样搬进稳健性结果表并贴上变体身份，note 注明来源。

    读不到那份文件时留一行注明原因的 NaN 行（走 vocabulary._note_only_rows），
    **绝不改为自己拟合一遍**：同一个会被写进论文的数字由两个模块各算一遍，
    迟早会不一致，而读者没有办法判断该信哪一个。
    """
    context = context or build_context(year)
    if context.lomo is None:
        return voc._note_only_rows(
            LOMO_UNAVAILABLE_LABEL, LOMO_UNAVAILABLE_NOTE,
            outcome=mt.OUTCOME_ENTRY, domain="public", out_path=out_path,
            variant_family=VARIANT_FAMILY_TEMPORAL,
        )

    source = context.lomo
    picked = source[source["term"] == mc.TERM_AME]
    rows = []
    for record in picked.to_dict("records"):
        label = "leave_one_month_out:{};denominator={}".format(
            record["model"],
            "user_months_with_month_fixed_effects(cluster_se_user)",
        )
        rows.append(harness.attach_variant_identity(
            record, VARIANT_FAMILY_TEMPORAL, label, 0, None))
    frame = pd.DataFrame(rows, columns=list(harness.ROBUSTNESS_SCHEMA))
    frame = voc._annotate_note(frame, LOMO_SOURCE_NOTE)
    print("§13.7 留一月重估: 从 models_temporal 读入 {} 行（未重算）".format(len(frame)))
    if out_path is not None:
        harness.append_rows(frame, out_path)
    return frame


def run_temporal_restrictions(year=config.YEAR, context=None, out_path=None,
                              diag_path=None,
                              min_active_months=DEFAULT_MIN_ACTIVE_MONTHS,
                              applied=None):
    """§13.7：上半年 vs 下半年、剔除行为量异常的月份、活跃满 3 / 6 个月的用户，
    外加从 models_temporal 读进来的留一月重估

    Args:
        applied: 可选的 dict，就地填入**这一次实际应用**的异常月份选择
            （规则、逐月行为量、中位数/MAD/阈值、选中的月份、保留的月份、
            以及这个变体到底有没有产出）。build() 用它写 manifest，而不是
            自己再算一遍 anomalous_months——manifest 应当报告"实际用了什么"，
            不是"照着同一段代码再推一次"。今天两者必然相同，但那是一个巧合，
            不是一条被保证的性质。
    """
    context = context or build_context(year)
    frame = context.user_table
    collected = []

    collected.append(_month_variant(
        context, FIRST_HALF_MONTHS, "months=first_half",
        rule="restrict_every_measure_to_months_01-06", out_path=out_path,
        diag_path=diag_path, excluded=SECOND_HALF_MONTHS,
    ))
    collected.append(_month_variant(
        context, SECOND_HALF_MONTHS, "months=second_half",
        rule="restrict_every_measure_to_months_07-12", out_path=out_path,
        diag_path=diag_path, excluded=FIRST_HALF_MONTHS,
    ))

    # --- 异常月份：规则事先写死，逐月账本进诊断表 ---
    volumes = monthly_behaviour_volume(context.panel)
    flagged, info = anomalous_months(volumes)
    append_diagnostics(_anomaly_diagnostics(volumes, info, set(flagged)), diag_path)
    retained = tuple(m for m in volumes.index.astype(int) if m not in set(flagged))
    anomaly_label = "months=anomalous_excluded({})".format(_format_months(flagged))
    emitted_anomaly = None
    if not flagged:
        # 规则一个月都没选中：剔除异常月份等于什么都不剔，这一版与基线逐字
        # 相同（复核时实测确认是一次一模一样的全年重估）。产出它只会在分布
        # 正中心多一个幽灵变体，因此改为核验"全年重聚合 == 表 C"并把结论写进
        # manifest——与主口径那一版同一条纪律。这在真实数据上完全可能发生。
        emitted_anomaly = "not_emitted:rule_flagged_no_month"
        print("异常月份规则一个月都没选中：这一版与基线逐字相同，改为核验"
              "全年重聚合是否重现表 C，不产出重复行")
        check_full_year_reconstruction(context)
    elif not retained:
        # 规则把整年都判成异常：留一行注明原因的 NaN 行，绝不悄悄退回
        # "不排除任何月份"——那会让一个跑不出来的变体伪装成一次通过。
        emitted_anomaly = "note_only:rule_flagged_every_month"
        collected.append(voc._note_only_rows(
            "{};denominator=user_activity_within_months(none)".format(anomaly_label),
            "anomaly_rule_flagged_every_month:no_window_left_to_estimate_on;"
            "rule={}".format(ANOMALY_RULE),
            outcome=mt.OUTCOME_ENTRY, domain="public", out_path=out_path,
            variant_family=VARIANT_FAMILY_TEMPORAL,
        ))
    else:
        emitted_anomaly = "estimated"
        collected.append(_month_variant(
            context, retained, anomaly_label,
            rule=ANOMALY_RULE, out_path=out_path, diag_path=diag_path,
            excluded=flagged,
            extra_note="anomaly_rule_pre_specified;selected_months={}".format(
                _format_months(flagged)),
        ))

    if applied is not None:
        applied.update({
            "anomaly_rule": ANOMALY_RULE,
            "anomaly_volume_variable": ANOMALY_VOLUME_VARIABLE,
            "monthly_behaviour_volume": {int(m): float(v) for m, v in volumes.items()},
            "monthly_volume_median": info["median"],
            "monthly_volume_mad": info["mad"],
            "anomaly_threshold": info["threshold"],
            "anomalous_months": [int(m) for m in flagged],
            "retained_months": [int(m) for m in retained],
            "anomaly_variant": emitted_anomaly,
        })

    # --- 活跃月份限制：剔的是用户，逐性别记账 ---
    for minimum in min_active_months:
        kept = active_month_restriction(frame, minimum)
        label = ("users_with_min_active_months={};"
                 "denominator=all_same_gender_users_active_in_at_least_{}_months"
                 ).format(int(minimum), int(minimum))
        diagnostic = diagnostics_row(
            VARIANT_FAMILY_TEMPORAL, label,
            rule="drop_users_with_{}<{}".format(ACTIVE_MONTHS_COLUMN, int(minimum)),
            before=frame, after=kept,
            measure_column="unchanged:both_measures_stay_on_the_full_year",
            numerator="unchanged", denominator=(
                "all_same_gender_users_active_in_at_least_{}_months".format(
                    int(minimum))),
            note="restriction_drops_users_not_months;"
                 "gender_specific_drop_counts_are_recorded_because_a_restriction_"
                 "that_removes_men_and_women_at_different_rates_changes_the_comparison",
        )
        print("{}: 保留 {:,} / {:,} 人（男 -{} / 女 -{}）".format(
            label, len(kept), len(frame),
            diagnostic["n_dropped_male"], diagnostic["n_dropped_female"]))
        collected.append(_run_variant(
            kept, VARIANT_FAMILY_TEMPORAL, label, diagnostic, out_path, diag_path))

    collected.append(run_leave_one_month_out(year, context=context,
                                             out_path=out_path))
    return pd.concat(collected, ignore_index=True)


# ---------------------------------------------------------------------------
# 全流程
# ---------------------------------------------------------------------------

def build(year=config.YEAR, min_active_months=DEFAULT_MIN_ACTIVE_MONTHS):
    """§13.1 + §13.8 + §13.7 全部变体，逐变体增量落盘，最后写 manifest"""
    context = build_context(year)
    out_path = results_path()
    diag_path = diagnostics_path()
    os.makedirs(robustness_dir(), exist_ok=True)

    # 与基线逐字相同的那几个口径在这里被**核验**，而不是被产出成重复的行
    reconstruction = verify_reconstructions(context)

    applied = {}
    run_denominator_variants(year, context=context, out_path=out_path,
                             diag_path=diag_path)
    run_post_type_variants(year, context=context, out_path=out_path,
                           diag_path=diag_path)
    run_temporal_restrictions(year, context=context, out_path=out_path,
                              diag_path=diag_path,
                              min_active_months=min_active_months,
                              applied=applied)

    results = pd.read_parquet(out_path, engine="pyarrow",
                              columns=list(harness.ROBUSTNESS_SCHEMA))
    diagnostics = pd.read_parquet(diag_path, engine="pyarrow",
                                  columns=list(DIAGNOSTIC_SCHEMA))

    manifest = config.build_manifest(
        step="robustness_measures_{}".format(year),
        inputs=[
            os.path.relpath(path, config.OUTPUT_DIR)
            for path in (context.source_paths["user_table"],
                         context.source_paths["panel"])
        ],
        params={
            "year": year,
            # 本族全部变体都是确定性的：没有任何抽样，因此没有种子；结果行的
            # seed 列一律如实写 None（harness 的规则），不是省略这一列。
            "deterministic_no_sampling": True,
            "entry_denominators": [ENTRY_DENOMINATOR_ALL,
                                   ENTRY_DENOMINATOR_RETWEETERS],
            "topical_measures": [
                {"key": spec.key, "numerator": spec.numerator,
                 "denominator": spec.denominator,
                 "emitted_as_variant": not spec.is_baseline}
                for spec in MEASURE_SPECS
            ],
            "post_type_variants": [
                {"key": variant.key,
                 "post_types": list(variant.post_types),
                 "require_nonzero_chars": bool(variant.require_nonzero_chars),
                 "denominator": variant.denominator,
                 "emitted_as_variant": not variant.is_baseline}
                for variant in POST_TYPE_VARIANTS
            ],
            "primary_post_types": list(PRIMARY_POST_TYPES),
            # 与基线逐字相同的口径不产出结果行，改为在这里被核验：这一项
            # 记的就是"核验过、且相等"，它比一行重复的估计有信息量得多。
            "reconstruction_checks": reconstruction,
            # 这几个量在替代测量/帖子类型变体里被写成 NaN，因为那些变体按
            # 构造动不到它们（进入指示来自表 B 的转发事件）
            "entry_quantities_emitted_as_nan_in_measure_and_post_type_variants":
                list(ENTRY_QUANTITIES),
            "first_half_months": list(FIRST_HALF_MONTHS),
            "second_half_months": list(SECOND_HALF_MONTHS),
            "min_active_months": [int(m) for m in min_active_months],
            "active_months_column": ACTIVE_MONTHS_COLUMN,
            # 异常月份：规则、它看的是什么量、逐月的量、以及它选中了哪几个月。
            # 这四样缺一不可——看过哪个月不方便再挑月份，读者从输出上分不出来。
            # **全部取自 run_temporal_restrictions 实际应用的那一次**（applied），
            # 不在这里重新推导一遍：manifest 该报告的是用了什么。
            "anomaly_k": ANOMALY_K,
            "anomaly_mad_scale": MAD_SCALE,
            "anomaly_min_relative_deviation": ANOMALY_MIN_RELATIVE_DEVIATION,
            "anomaly_rule": applied["anomaly_rule"],
            "anomaly_volume_variable": applied["anomaly_volume_variable"],
            "anomalous_months": applied["anomalous_months"],
            "anomaly_retained_months": applied["retained_months"],
            "anomaly_variant": applied["anomaly_variant"],
            "monthly_behaviour_volume": applied["monthly_behaviour_volume"],
            "monthly_volume_median": applied["monthly_volume_median"],
            "monthly_volume_mad": applied["monthly_volume_mad"],
            "anomaly_threshold": applied["anomaly_threshold"],
            # §13.7 的留一月估计不在这里算：读 models_temporal 的输出
            "leave_one_month_out_source": os.path.relpath(
                context.source_paths["leave_one_month_out"], config.OUTPUT_DIR),
            "leave_one_month_out_available": bool(context.lomo is not None),
            "monthly_rates_source": os.path.relpath(
                context.source_paths["monthly_rates"], config.OUTPUT_DIR),
            "monthly_rates_available": bool(context.monthly_rates is not None),
            "diagnostics_file": os.path.basename(diag_path),
        },
        counts={
            "result_rows": int(len(results)),
            "diagnostic_rows": int(len(diagnostics)),
            "variant_labels": int(results["variant_label"].nunique()),
            "users_total": int(len(context.user_table)),
            "users_male": int((context.user_table["gender"] == "m").sum()),
            "users_female": int((context.user_table["gender"] == "f").sum()),
            "user_months": int(len(context.panel)),
            "posts_total": int(len(context.incidence[DOMAINS[0]].posts)),
        },
    )
    config.write_manifest(manifest, manifest_dir(year))
    return {"results_path": out_path, "diagnostics_path": diag_path}


def _cli(fn):
    """CLI 包装：把 out_path / diag_path 的默认值补成模块的正式路径

    各 run_* 函数的路径参数默认是 None（供测试与 build() 显式传路径），
    但从命令行单跑一个子命令时，None 意味着"算了一整轮、一个字节都没写"，
    而且不会有任何提示。这层包装只在 CLI 入口生效。
    """
    def _run(*args, **kwargs):
        kwargs.setdefault("out_path", results_path())
        kwargs.setdefault("diag_path", diagnostics_path())
        os.makedirs(robustness_dir(), exist_ok=True)
        print("输出目录: {}".format(robustness_dir()))
        return fn(*args, **kwargs)

    _run.__doc__ = fn.__doc__
    _run.__name__ = fn.__name__
    return _run


if __name__ == "__main__":
    fire.Fire({
        "build": build,
        "denominator_variants": _cli(run_denominator_variants),
        "post_type_variants": _cli(run_post_type_variants),
        "temporal_restrictions": _cli(run_temporal_restrictions),
    })
