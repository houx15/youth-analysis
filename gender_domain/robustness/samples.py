"""
§13.2 极端用户与长尾、§13.6 性别样本数量、§13.9 性别字段与用户类型。

这三节改的都是**哪些用户进分析**（或者这些用户的活动量以什么形态进模型），
而不是测量本身——与 accounts / vocabulary 改的是"某个用户的领域参与怎么算"
恰好互补。因此本模块不碰任何 incidence 矩阵，只对表 C 做行选择与列变换，
再原样交给 harness.estimate_all。

--------------------------------------------------------------------------
最重要的一条：阈值一律在两性合并的样本上算，男女必须是同一个数
--------------------------------------------------------------------------
§13.2 的原文是"男女共同的 pooled P95/P99 阈值"，并且明确写着：

    不把男女分别按各自 95% 分位点截尾作为唯一版本。不同性别使用不同删除
    阈值会改变所比较的总体，现有娱乐转发均值还会因此发生方向反转。

"按男性自己的 P95 截男性、按女性自己的 P95 截女性"看上去很对称，实际上是
在比较两个被各自重新定义过的总体：男性的 P95 与女性的 P95 落在完全不同的
绝对活跃度上，截完之后"男性组"里最活跃的人可能仍然比"女性组"里最活跃的人
活跃一倍。本模块因此只提供 `applied_cut_points`，它算一次合并分位点、把
同一个数同时返回给 m 和 f，**没有留下逐性别算阈值的入口**；
gender_domain/tests/test_samples_robustness.py 里有一条测试钉死
`cuts["m"] == cuts["f"] == cuts["pooled"]`，并且钉死这个数与逐性别分位点
都不相等（夹具刻意让男性更活跃）。

同一条纪律适用于"删除总体最高 1%/5% 用户"：那是**合并**的 1%，不是各删
各的 1%。男性更活跃时，合并的前 1% 里男性比女性多，两性被删掉的人数**本来
就该不对称**——这正是这个变体要展示的东西，不是需要被"修正"的偏差。诊断表
逐变体记 n_dropped_male / n_dropped_female，让这份不对称直接可见。

--------------------------------------------------------------------------
"不处理极端值 vs log(1+x)"这一对在本流水线里做不到，因此只留一行说明
--------------------------------------------------------------------------
§13.2 列出的前两种处理是"不处理"与"log(1+x)"。但 M1/M2 的活动量协变量
**本来就是 log1p**：models_core.add_log_activity 把 n_posts / n_retweets
变成 log_posts / log_retweets，MODEL_LAYERS 里写死用的是后者。也就是说，
主规格已经是"log(1+x)"那一版；本模块只能改用户表，改不了协变量变换，
所以"原始尺度协变量"那一版在这里跑不出来。

**做不到的事只留一行注明原因的 NaN 行，绝不写成一组估计值。** 早先的做法
是把 untreated 的结果原样复制一份挂上 log1p_activity 的标签、再用 note
说明"按构造相同"——那段 note 只能被人读到，**下游的算术读不到**：
synthesis 的 direction_consistency 是"与基线同号的**变体标签**占多少"，
一个坐在正中心的复制标签会同时抬高方向一致率、压低估计值的离散度，把
"这是同一次拟合"悄悄变成 §13.2 的一票赞成。因此改为走
`voc._note_only_rows`（本模块处理 demographic_gender 缺失时用的同一套
做法），让 synthesis 把 §13.2 记成"检验不完整"，而不是记成一次通过。

真要做出这个对比，有两条路，**都是主流水线改动加一次重跑，不属于本任务**：
往 models_core.MODEL_LAYERS 里加一层 M1_raw（能顺着 estimate_all 现有的
layers 参数走通，但 MODEL_LAYERS 在 models_core 里有四处无条件遍历，会漏
进主结果表），或者给 estimate_all 加一个协变量覆盖参数（更干净）。

截尾只作用于**活动量协变量**（ACTIVITY_COLUMNS），不动结果变量：六个预先
设定量的结果变量是一个 0/1 指示与一个 0-1 占比，本来就有界，对它们做
winsorise 没有意义。

--------------------------------------------------------------------------
§13.6：等量抽样必须自报抽到了什么，"平衡样本"必须自证平衡
--------------------------------------------------------------------------
- **等量抽样**（`run_equal_size_gender`）：男性全留，女性每次随机抽与男性
  相同的人数。每一次抽样都往诊断表写一行**这次抽到的女性的活动量分布**
  （发帖/转发/活跃天数的均值、中位数、P90），并且在开头写一行全样本参照行
  （variant_label = "full_sample_reference"）。只报估计值不报样本长什么样，
  读者就只能假定子样本有代表性——而那恰恰是这个变体要检验的东西。
- **活动量平衡**（`run_activity_balanced`）：方法名写进 variant_label
  （`ps_match` 或退化时的 `cem_match`），并且平衡前后的标准化均值差
  （standardised mean difference）逐协变量写进 sample_balance.parquet。
  一个不自证的 "balanced" 标签比不做平衡更糟：标签宣称了数据并不支持的
  东西。平衡后 SMD 没有变小时，这张表会直接把这件事显示出来。

  匹配方法取"倾向得分分层 + 层内 1:1 抽取"：先用 logit 估
  P(男性 | 活动量协变量)，再按**合并样本**的得分分位数切层，层内两性各取
  min(n_男, n_女) 人。选它是因为它在 20 万用户上是 O(n log n)（逐对最近邻
  匹配是 O(n_男 × n_女)，在 5.2 万 × 17.4 万上根本跑不完），而且分层键
  是一维的，不会像多维粗化精确匹配（CEM）那样在细分格子里找不到配对。
  倾向得分模型拟合失败时退回按协变量分位数格子的 CEM，并且**把实际用的
  方法写进 variant_label 与 note**，绝不让标签说 ps_match、跑的却是别的。

--------------------------------------------------------------------------
§13.9：demographic_gender 从哪来，以及规则必须事先写死
--------------------------------------------------------------------------
`demographic_gender` 不在表 C 里，也不在表 A 里（build_post_table 的
OUTPUT_COLUMNS 没有透传它）。它在 merged_profiles/merged_user_profiles.parquet
里——而这个文件正是 profile_join 已经在读的那一个。因此本次把它加进
`profile_join.PROFILE_COLUMNS`，让它随其它画像列一起进
`user_domain_{year}_with_profile.parquet`；这不需要重跑表 A（那是一次全年
文本重扫），也不需要动 build_post_table。**本模块读不到这一列时，两个依赖
它的变体各输出一行注明原因的 NaN 行，绝不把"字段缺失"当成"两个字段一致"。**

用户类型三条规则全部事先写死在模块常量里，并且写进 manifest：

- **机构账号**：verified_type ∈ INSTITUTIONAL_VERIFIED_TYPES（1-7，微博常见
  编码里的政府/企业/媒体/网站/应用/团体机构/待审核）。profile_join 顶部
  已经声明过项目没有拿到官方码表，这里沿用同一条保守假设并把取值列表写进
  manifest，读者可以直接看到这个判断是什么、以及改它要改哪里。
- **认证个人**：verified_flag 为真且不属于机构类别（微博编码里 0 是个人认证）。
  普通用户：verified_flag 为假。verified_flag 缺失的用户两边都不进——
  "不知道"不是"普通用户"。
- **疑似自动化/营销**：AUTOMATION_RULE 写死的两条子规则取或——
  (a) 总活动量（发帖 + 转发）高于**合并样本**的 P99.9；
  (b) 日均发帖量（发帖数 ÷ 活跃天数）高于**合并样本**的 P99.9。
  两条都是合并分位点，都不含任何看过结果之后才定的绝对数字。这是一个
  筛子，不是一个分类器：它标的是"行为形状像机器"的那一小撮尾巴，标出来
  多少人、阈值是多少，全部写进诊断表。

使用方法:
    python -m gender_domain.robustness.samples build --year 2020 --n_replicates 200
"""

import os
import tempfile
from collections import namedtuple

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf

from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import id_rules as ir
from gender_domain import models_core as mc
from gender_domain.robustness import harness
# replicate_seeds / _annotate_note / _note_only_rows / _domain_seed 一律复用
# vocabulary 的实现（accounts 也是这么做的）：三个 family 的种子派生规则或
# note 拼接规则一旦分叉，跨 family 的比较就不再是同一件事。
from gender_domain.robustness import vocabulary as voc

DOMAINS = but.DOMAINS

# 四个 variant family。等量抽样是几百个 replicate 的分布，与确定性变体混在
# 同一个 family 里会让下游"变体之间的离散度"把两种东西加在一起，所以它
# 单独成一个 family（与 accounts_bootstrap 同一个道理）。
VARIANT_FAMILY_EXTREME = "extreme_values"      # §13.2
VARIANT_FAMILY_EQUAL_SIZE = "gender_equal_size"  # §13.6 等量抽样（多 replicate）
VARIANT_FAMILY_BALANCE = "gender_sample"       # §13.6 活动量平衡（确定性）
VARIANT_FAMILY_USER_TYPE = "user_type"         # §13.9

# 表 C 里六个量真正用得到的列，与 vocabulary / accounts 共用同一份清单。
BASE_USER_COLUMNS = list(voc.USER_TABLE_COLUMNS)

# 画像列。§13.9 全部变体、§13.6 的认证状态平衡都依赖它们，而且 §13.9 是
# 唯一允许估 M2 的一节（方案文档 §11.4：M2 会收窄样本，混进其它比较会
# 混淆结论），M2 的三个画像协变量也在这里。
PROFILE_USER_COLUMNS = [
    "verified_flag",
    "verified_type",
    "user_type",
    "log_fans",
    "log_friends",
    "profile_complete",
    "demographic_gender",
]

# 被极端值处理作用的活动量列。结果变量（0/1 进入指示、0-1 占比）本来就
# 有界，不在这里面——见模块文档。
ACTIVITY_COLUMNS = ("n_posts", "n_retweets", "n_active_days", "n_active_months")

# "总活动量"的定义：发帖数 + 转发帖数。删顶端 x% 用户按它排序。
TOTAL_ACTIVITY_COLUMNS = ("n_posts", "n_retweets")

# §13.2 的 log(1+x) 对照：跑不出来，只留一行注明原因的 NaN 行（见模块文档）。
LOG1P_LABEL = "log1p_activity_unavailable"
LOG1P_NOTE = (
    "raw_scale_activity_covariates_not_estimable:"
    "models_core.MODEL_LAYERS_fixes_the_covariate_transform_to_log1p;"
    "main_specification_is_already_log1p;"
    "requires_a_covariate_override_in_estimate_all_or_an_M1_raw_layer"
    "(main_pipeline_change+rerun)"
)

DEFAULT_WINSOR_QUANTILES = (0.95, 0.99)
DEFAULT_TRIM_SHARES = (0.01, 0.05)
DEFAULT_N_REPLICATES = 200
DEFAULT_N_BINS = 20

# §13.6 平衡的协变量：总发帖量、总转发量、活跃天数、认证状态（简报原文）。
# 前两项取 models_core.add_log_activity 派生的 log1p 版本——与模型里进
# M1 的是同一个变量，平衡的就该是它，而不是另一个尺度上的同名量。
BALANCE_COVARIATES = ("log_posts", "log_retweets", "n_active_days", "verified_flag")
BALANCE_METHOD_PS = "ps_match"
BALANCE_METHOD_CEM = "cem_match"

# §13.9 机构认证的 verified_type 取值。项目没有官方码表（见 profile_join
# 顶部的说明），这里沿用微博常见编码：1 政府、2 企业、3 媒体、4 网站、
# 5 应用、6 团体机构、7 待审核。写成模块常量并写进 manifest，是为了让
# "这是一个判断"这件事在结果旁边可见，而不是埋在某个 if 里。
INSTITUTIONAL_VERIFIED_TYPES = (1, 2, 3, 4, 5, 6, 7)
# 未认证的编码（profile_join 的同一条假设）
UNVERIFIED_VERIFIED_TYPE = -1

# 疑似自动化/营销的分位点与规则文本。**事先写死**：一个看过结果之后才挑
# 出来的阈值不是稳健性检验。
AUTOMATION_QUANTILE = 0.999
AUTOMATION_RULE = (
    "pooled_p{q}_of_total_activity(n_posts+n_retweets)"
    " OR pooled_p{q}_of_posts_per_active_day(n_posts/n_active_days)"
).format(q=AUTOMATION_QUANTILE)

# demographic_gender 的取值归一化：只认字母与中文两种写法，数字编码
# （1/2 到底谁是男谁是女）没有权威依据，一律记成"无法映射"而不是猜。
_GENDER_TOKENS = {
    "m": "m", "male": "m", "man": "m", "男": "m", "男性": "m",
    "f": "f", "female": "f", "woman": "f", "女": "f", "女性": "f",
}

# 诊断表 schema。逐 (family, variant_label, replicate, threshold_variable)
# 一行，分四组：1) 身份与规则；2) **阈值**（pooled/male/female 三列并排，
# 后两列必须相等，这是本任务最重要的一条性质，直接摆在表上让人一眼看见）；
# 3) 样本量与被剔除人数（逐性别，删顶端的不对称就在这里）；4) 活动量分布
# （逐性别的均值/中位数/P90，等量抽样的每一次抽样都写一行）。
DIAGNOSTIC_SCHEMA = (
    "variant_family",
    "variant_label",
    "replicate",
    "seed",
    "rule",
    "threshold_variable",
    "threshold_quantile",
    "threshold_pooled",
    "threshold_male",
    "threshold_female",
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
    "female_to_male_ratio",
    "mean_posts_male",
    "median_posts_male",
    "p90_posts_male",
    "mean_posts_female",
    "median_posts_female",
    "p90_posts_female",
    "mean_retweets_male",
    "median_retweets_male",
    "p90_retweets_male",
    "mean_retweets_female",
    "median_retweets_female",
    "p90_retweets_female",
    "mean_active_days_male",
    "median_active_days_male",
    "p90_active_days_male",
    "mean_active_days_female",
    "median_active_days_female",
    "p90_active_days_female",
    "verified_share_male",
    "verified_share_female",
    "note",
)

# 平衡表 schema。逐 (变体, 阶段, 协变量) 一行，stage ∈ {"before", "after"}。
BALANCE_SCHEMA = (
    "variant_family",
    "variant_label",
    "stage",
    "covariate",
    "n_male",
    "n_female",
    "mean_male",
    "mean_female",
    "sd_male",
    "sd_female",
    "std_mean_diff",
    "variance_ratio",
    "note",
)

SampleContext = namedtuple(
    "SampleContext",
    ["year", "user_table", "source_path", "has_profile", "has_demographic_gender"],
)

# _domain_seed 的 stream 取值。vocabulary 占了 0（抽词）与 1（抽帖子），
# accounts 占了 2（账号 bootstrap）；样本层用 3（等量抽样）与 4（层内匹配
# 抽取），这样即使几个 family 用同一个 replicate 种子，随机流也互不重合。
_STREAM_EQUAL_SIZE = 3
_STREAM_MATCH = 4


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def robustness_dir():
    """稳健性层唯一允许写入的目录"""
    return os.path.join(config.OUTPUT_DIR, "robustness")


def results_path():
    return os.path.join(robustness_dir(), "samples.parquet")


def diagnostics_path():
    return os.path.join(robustness_dir(), "sample_diagnostics.parquet")


def balance_path():
    return os.path.join(robustness_dir(), "sample_balance.parquet")


def manifest_dir(year):
    return os.path.join(robustness_dir(), "samples_{}".format(year))


# ---------------------------------------------------------------------------
# 读表：优先带画像的那一份，因为 §13.9 与认证状态平衡都要它
# ---------------------------------------------------------------------------

def _read_available_columns(path, columns):
    """按 columns 读 parquet，但只读文件里真的有的那些列

    parquet 读取必须显式点名列（全局约定），而这里又必须容忍"这份表是在
    profile_join 加上 demographic_gender 之前生成的"。先读 schema 再取交集，
    两条都能满足；缺了哪些列打印出来，不静默。
    """
    available = set(pq.ParquetFile(path).schema.names)
    wanted = [c for c in columns if c in available]
    missing = [c for c in columns if c not in available]
    if missing:
        print("提示: {} 缺少列 {}，这些列不参与本次稳健性检验".format(
            os.path.basename(path), missing))
    return pd.read_parquet(path, engine="pyarrow", columns=wanted), set(wanted)


def load_user_table(year=config.YEAR):
    """(用户表, 来源文件, 是否带画像列)

    优先读 user_domain_{year}_with_profile.parquet：§13.9 的全部变体、§13.6
    的认证状态平衡、以及 §13.9 唯一允许的 M2 层都依赖画像列。没有这份文件
    时退回原始表 C 并**打印说明**，依赖画像列的变体随后会各自输出注明原因
    的行，而不是在这里抛异常把另外两节也一起拖垮。
    """
    with_profile = os.path.join(
        config.OUTPUT_DIR, "user_domain_{}_with_profile.parquet".format(year))
    plain = os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year))

    if os.path.exists(with_profile):
        frame, present = _read_available_columns(
            with_profile, BASE_USER_COLUMNS + PROFILE_USER_COLUMNS)
        print("读取带画像的表 C: {}（{:,} 个用户）".format(with_profile, len(frame)))
        return frame, with_profile, True
    if not os.path.exists(plain):
        raise FileNotFoundError(
            "未找到表 C: {}（由 python -m gender_domain.build_user_tables build "
            "生成）".format(plain)
        )
    frame, present = _read_available_columns(plain, BASE_USER_COLUMNS)
    print(
        "提示: 未找到 {}，改用 {}。§13.9 的画像相关变体与 M2 层将无法运行，"
        "它们会各自输出一行注明原因的结果。".format(
            os.path.basename(with_profile), plain)
    )
    return frame, plain, False


def demographic_gender_coverage(frame):
    """demographic_gender 有多少取值真的能映射成 m/f

    只写一个"这一列在不在"的布尔值不够：这一列可能存在、却有一大半取值
    认不出来（例如存成数字编码，见 _normalise_gender_token）。这三个数字
    进 manifest，读者不必用减法去猜性别字段变体到底覆盖了哪些人。
    """
    n_total = int(len(frame))
    if "demographic_gender" not in frame.columns:
        return {"n_total": n_total, "n_mappable": 0, "n_unmappable": n_total,
                "mappable_share": 0.0}
    n_mappable = int(frame["demographic_gender"].map(_normalise_gender_token).notna().sum())
    return {
        "n_total": n_total,
        "n_mappable": n_mappable,
        "n_unmappable": n_total - n_mappable,
        "mappable_share": (n_mappable / n_total) if n_total else float("nan"),
    }


def build_context(year=config.YEAR):
    """加载用户表，并判断 demographic_gender 到底能不能用"""
    frame, source_path, has_profile = load_user_table(year)
    frame = frame.copy()
    frame["user_id"] = ir.normalize_id_series(frame["user_id"])

    has_demographic = "demographic_gender" in frame.columns
    if has_demographic:
        n_usable = demographic_gender_coverage(frame)["n_mappable"]
        if n_usable == 0:
            print(
                "提示: demographic_gender 这一列存在但没有任何可识别的取值，"
                "性别字段一致性变体无法运行（绝不把「字段缺失」当成「两个字段一致」）"
            )
            has_demographic = False
        else:
            print("demographic_gender 可用: {:,} / {:,} 个用户能映射成 m/f".format(
                n_usable, len(frame)))
    else:
        print(
            "提示: 用户表里没有 demographic_gender 列（它在 merged_profiles 里，"
            "由 profile_join 带进带画像的表 C）。性别字段一致性变体本次无法运行。"
        )

    n_male = int((frame["gender"] == "m").sum())
    n_female = int((frame["gender"] == "f").sum())
    print("样本: 男性 {:,} 人、女性 {:,} 人（女/男 = {:.2f}）".format(
        n_male, n_female, n_female / n_male if n_male else float("nan")))
    return SampleContext(
        year=year, user_table=frame, source_path=source_path,
        has_profile=has_profile, has_demographic_gender=has_demographic,
    )


# ---------------------------------------------------------------------------
# §13.2 阈值：一律 pooled，男女同一个数
# ---------------------------------------------------------------------------

def total_activity(frame):
    """总活动量 = 发帖数 + 转发帖数，删顶端 x% 用户按它排序

    定义写在一个函数里而不是散落在各处，是为了让"总活动量到底指什么"只有
    一个答案：诊断表里的阈值、结果行 note 里的阈值、测试里独立算出来的
    期望值，三者必须指同一个量。
    """
    total = np.zeros(len(frame), dtype=float)
    for column in TOTAL_ACTIVITY_COLUMNS:
        total = total + frame[column].to_numpy(dtype=float)
    return pd.Series(total, index=frame.index)


def _pooled_values(frame, values):
    """参与阈值计算的取值：只取 gender ∈ {m, f} 的用户

    "pooled"在 §13.2 里的意思是"男女合起来"，所以分位点的总体就是这两组人
    的并集；性别缺失/其它取值的用户本来也进不了以性别为焦点变量的模型
    （models_core.prepare_model_frame 会剔掉他们），不该参与定义阈值。
    """
    mask = frame["gender"].isin(("m", "f")).to_numpy(dtype=bool)
    pooled = np.asarray(values, dtype=float)[mask]
    return pooled[~np.isnan(pooled)]


def pooled_quantile(frame, column, quantile):
    """某一列在**两性合并样本**上的分位点；无有效取值时返回 NaN"""
    values = _pooled_values(frame, frame[column].to_numpy(dtype=float))
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, float(quantile)))


def applied_cut_points(frame, column, quantile):
    """{"pooled": t, "m": t, "f": t}——男女用的就是同一个 t

    这个函数**不提供**逐性别算阈值的入口，这是刻意的：§13.2 明确指出
    "不同性别使用不同删除阈值会改变所比较的总体，现有娱乐转发均值还会因此
    发生方向反转"。三个键返回同一个数，是为了让诊断表能把 threshold_male
    与 threshold_female 两列并排写出来，读者不用读代码就能确认这一条。
    """
    cut = pooled_quantile(frame, column, quantile)
    return {"pooled": cut, "m": cut, "f": cut}


def winsorise_pooled(frame, quantile, columns=ACTIVITY_COLUMNS):
    """把活动量列在**合并**分位点处截顶，返回 (新表, {列: 阈值})

    只截上尾：§13.2 关心的是长尾里那一小撮极端活跃用户，下尾的 0 是有定义
    的真实事实（没发过帖），截它没有任何意义。截尾改的是数值不是样本，
    因此行数不变——models_core.add_log_activity 随后会在截过顶的值上取
    log1p，进 M1 的就是截过顶的协变量。
    """
    out = frame.copy()
    thresholds = {}
    for column in columns:
        cut = pooled_quantile(frame, column, quantile)
        thresholds[column] = cut
        if not np.isnan(cut):
            out[column] = out[column].clip(upper=cut)
    return out, thresholds


def trim_top_users(frame, share):
    """删掉总活动量在**合并**样本前 share 之上的用户

    规则：总活动量 **严格大于**合并样本的 (1 - share) 分位点的用户被删。
    用阈值而不是"按名次取前 ceil(share × n) 个"，是为了不在并列处把一批
    活动量完全相同的用户劈成两半（谁被删谁留下就成了排序的偶然）；代价是
    实际删掉的比例可能与 share 略有出入，所以诊断表逐变体记真实的删除人数
    与比例，而不是假设它恰好等于 share。

    Returns:
        (删完之后的表, 阈值, 被删掉的布尔掩码)
    """
    activity = total_activity(frame)
    cut = float("nan")
    values = _pooled_values(frame, activity.to_numpy(dtype=float))
    if values.size:
        cut = float(np.quantile(values, 1.0 - float(share)))
    dropped = pd.Series(
        activity.to_numpy(dtype=float) > cut if not np.isnan(cut)
        else np.zeros(len(frame), dtype=bool),
        index=frame.index,
    )
    return frame[~dropped.to_numpy(dtype=bool)].copy(), cut, dropped


# ---------------------------------------------------------------------------
# §13.9 掩码：三条规则全部事先写死
# ---------------------------------------------------------------------------

def _normalise_gender_token(value):
    """把 demographic_gender 的取值归一化成 "m"/"f"，认不出来就是 None

    只认字母与中文两种写法。数字编码（1/2 谁是男谁是女）项目里没有权威
    依据，一律判成"无法映射"——猜错方向会让"性别字段一致"这条结论整个
    反过来，而这正是本节要检验的东西。
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    token = str(value).strip().lower()
    return _GENDER_TOKENS.get(token)


def _mapped_demographic_gender(frame):
    if "demographic_gender" not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)
    return frame["demographic_gender"].map(_normalise_gender_token)


def gender_fields_agree_mask(frame):
    """两个性别字段都能映射、且取值相同的用户

    映射不出来的用户既不算"一致"也不算"冲突"——"不知道"是第三种状态，
    把它算进任何一边都是在编数据。
    """
    mapped = _mapped_demographic_gender(frame)
    gender = frame["gender"].map(_normalise_gender_token)
    return (mapped.notna() & gender.notna() & (mapped == gender)).astype(bool)


def gender_conflict_mask(frame):
    """两个性别字段都能映射、但取值相反的用户"""
    mapped = _mapped_demographic_gender(frame)
    gender = frame["gender"].map(_normalise_gender_token)
    return (mapped.notna() & gender.notna() & (mapped != gender)).astype(bool)


def _verified_type_numeric(frame):
    if "verified_type" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame["verified_type"], errors="coerce")


def institutional_mask(frame):
    """机构认证账号：verified_type ∈ INSTITUTIONAL_VERIFIED_TYPES"""
    numeric = _verified_type_numeric(frame)
    return numeric.isin(list(INSTITUTIONAL_VERIFIED_TYPES)).astype(bool)


def _verified_flag_bool(frame):
    """verified_flag 转成 (是否认证, 是否已知) 两个布尔数组"""
    if "verified_flag" not in frame.columns:
        unknown = np.zeros(len(frame), dtype=bool)
        return unknown, unknown
    values = pd.to_numeric(
        frame["verified_flag"].astype("object"), errors="coerce"
    ).to_numpy(dtype=float)
    known = ~np.isnan(values)
    return (values == 1.0) & known, known


def verified_individual_mask(frame):
    """认证个人：已认证、且不属于机构认证类别

    认证状态未知的用户不算认证个人（也不算普通用户，见 ordinary_user_mask）。
    """
    verified, known = _verified_flag_bool(frame)
    institutional = institutional_mask(frame).to_numpy(dtype=bool)
    return pd.Series(verified & known & (~institutional), index=frame.index)


def ordinary_user_mask(frame):
    """普通用户：认证状态已知且为"未认证"

    未知一律不进——把"没匹配到画像"算成"普通用户"会让这个对照组混进一批
    根本没有画像的账号，而 §13.9 要比的正是"普通用户 vs 认证个人"。
    """
    verified, known = _verified_flag_bool(frame)
    return pd.Series((~verified) & known, index=frame.index)


def automation_mask(frame, quantile=AUTOMATION_QUANTILE):
    """疑似自动化/营销账号，返回 (掩码, {阈值变量: 阈值})

    规则见模块常量 AUTOMATION_RULE：两条子规则取或，两条都是**合并样本**
    的分位点。日均发帖量在活跃天数为 0 时无定义（NaN），不标记——把
    "算不出来"标成"疑似机器人"会凭空剔掉一批用户。
    """
    activity = total_activity(frame).to_numpy(dtype=float)
    days = frame["n_active_days"].to_numpy(dtype=float)
    per_day = np.divide(
        frame["n_posts"].to_numpy(dtype=float), days,
        out=np.full(len(frame), np.nan), where=days > 0,
    )

    thresholds = {
        "total_activity": float(np.quantile(_pooled_values(frame, activity), quantile))
        if _pooled_values(frame, activity).size else float("nan"),
        "posts_per_active_day": float(
            np.quantile(_pooled_values(frame, per_day), quantile)
        ) if _pooled_values(frame, per_day).size else float("nan"),
    }
    flagged = np.zeros(len(frame), dtype=bool)
    if not np.isnan(thresholds["total_activity"]):
        flagged |= activity > thresholds["total_activity"]
    if not np.isnan(thresholds["posts_per_active_day"]):
        flagged |= np.where(np.isnan(per_day), False,
                            per_day > thresholds["posts_per_active_day"])
    return pd.Series(flagged, index=frame.index), thresholds


# ---------------------------------------------------------------------------
# §13.6 等量抽样
# ---------------------------------------------------------------------------

def equal_size_draw(frame, seed):
    """男性全留，女性随机抽与男性相同的人数（不放回）

    抽之前先按 user_id 排序，让同一个 seed 的抽样结果与输入表的行序无关
    （parquet 的行序不该影响一个可复现的抽样）。女性人数不足时全取，并由
    调用方在 note 里说明——这在真实数据上不会发生（女性 17.4 万、男性
    5.2 万），但夹具或子样本上可能发生，静默地"抽到多少算多少"会让
    "等量"这个标签失真。
    """
    male = frame[frame["gender"] == "m"]
    female = frame[frame["gender"] == "f"].sort_values("user_id")
    n_male = len(male)
    if len(female) <= n_male:
        drawn = female
    else:
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(len(female), size=n_male, replace=False))
        drawn = female.iloc[picked]
    return pd.concat([male, drawn], ignore_index=False)


# ---------------------------------------------------------------------------
# §13.6 活动量平衡：倾向得分分层 + 层内 1:1 抽取（退化时用 CEM）
# ---------------------------------------------------------------------------

def _balance_frame(frame):
    """派生平衡与匹配用的协变量列，log 尺度一律走 models_core"""
    out = mc.add_log_activity(frame)
    out = mc.add_male_indicator(out)
    verified, known = _verified_flag_bool(frame)
    out["verified_numeric"] = verified.astype(float)
    out["verified_unknown"] = (~known).astype(float)
    if "verified_flag" in out.columns:
        out["verified_flag"] = np.where(known, verified.astype(float), np.nan)
    else:
        out["verified_flag"] = np.nan
    return out


def balance_statistics(frame, covariate):
    """一个协变量在男女之间的均衡统计（标准化均值差为核心）

    标准化均值差取 (mean_m - mean_f) / sqrt((var_m + var_f) / 2)，即两组
    方差的平均作为标准化尺度——这是 Rosenbaum-Rubin 的常用写法，不依赖
    哪一组是"处理组"，因此匹配前后用的是同一把尺子。分母为 0（两组都是
    常量）时返回 NaN 并在 note 里说明，绝不返回 0（"完美平衡"和"没法算"
    必须分得开）。
    """
    values = pd.to_numeric(frame[covariate], errors="coerce")
    male = values[frame["gender"].to_numpy() == "m"].dropna()
    female = values[frame["gender"].to_numpy() == "f"].dropna()
    stats = {
        "n_male": int(len(male)),
        "n_female": int(len(female)),
        "mean_male": float(male.mean()) if len(male) else float("nan"),
        "mean_female": float(female.mean()) if len(female) else float("nan"),
        "sd_male": float(male.std(ddof=1)) if len(male) > 1 else float("nan"),
        "sd_female": float(female.std(ddof=1)) if len(female) > 1 else float("nan"),
        "std_mean_diff": float("nan"),
        "variance_ratio": float("nan"),
        "note": None,
    }
    if len(male) > 1 and len(female) > 1:
        var_male = float(male.var(ddof=1))
        var_female = float(female.var(ddof=1))
        pooled_sd = np.sqrt((var_male + var_female) / 2.0)
        if pooled_sd > 0:
            stats["std_mean_diff"] = float(
                (stats["mean_male"] - stats["mean_female"]) / pooled_sd)
        else:
            stats["note"] = "zero_pooled_sd"
        if var_female > 0:
            stats["variance_ratio"] = float(var_male / var_female)
    else:
        stats["note"] = "too_few_users_in_one_gender"
    return stats


def balance_rows(frame, variant_family, variant_label, stage,
                 covariates=BALANCE_COVARIATES):
    """平衡表的一批行：一个阶段（before/after）× 每个协变量一行"""
    prepared = _balance_frame(frame)
    rows = []
    for covariate in covariates:
        stats = balance_statistics(prepared, covariate)
        stats.update({
            "variant_family": variant_family,
            "variant_label": variant_label,
            "stage": stage,
            "covariate": covariate,
        })
        rows.append({key: stats[key] for key in BALANCE_SCHEMA})
    return rows


def _propensity_score(frame):
    """P(男性 | 活动量协变量)，拟合失败时返回 None

    这是一个**匹配工具**，不是本研究的模型：它只用来把用户排到一维得分上
    好分层，估计量本身仍然由 harness 调用 models_core 得到。因此这里不做
    任何稳健标准误、也不报告它的系数。
    """
    candidates = ["log_posts", "log_retweets", "n_active_days",
                  "verified_numeric", "verified_unknown"]
    terms = [c for c in candidates
             if c in frame.columns and frame[c].nunique(dropna=True) >= 2]
    if not terms:
        return None

    # 协变量先做 z 标准化再进 logit。这不改变得分的排序（因而不改变分层），
    # 但把设计矩阵的条件数压下来：发帖量与活跃天数差着两个数量级时，
    # Newton 步长会算出一个奇异的 Hessian（实测就是 LinAlgError）。
    work = pd.DataFrame({mc.FOCAL_COLUMN: frame[mc.FOCAL_COLUMN].to_numpy(dtype=float)})
    for term in terms:
        values = frame[term].to_numpy(dtype=float)
        sd = float(np.nanstd(values))
        work[term] = (values - float(np.nanmean(values))) / (sd if sd > 0 else 1.0)
    formula = "{} ~ {}".format(mc.FOCAL_COLUMN, " + ".join(terms))

    # 两档优化器：Newton 在准分离下容易给出奇异 Hessian，BFGS 只用梯度、
    # 在同样的数据上通常还能收敛。两档都失败才退回 CEM，并且**把实际用的
    # 方法写进 variant_label**——标签绝不允许说 ps_match 却跑了别的。
    for method in ("newton", "bfgs"):
        try:
            fit = smf.logit(formula, data=work).fit(
                method=method, disp=False, maxiter=200)
            score = np.asarray(fit.predict(work), dtype=float)
        except Exception as exc:  # noqa: BLE001 —— 匹配工具失败不该炸掉整个变体
            print("提示: 倾向得分模型（{}）拟合失败（{}: {}）".format(
                method, type(exc).__name__, exc))
            continue
        if np.isnan(score).any():
            print("提示: 倾向得分（{}）里出现缺失".format(method))
            continue
        return score
    print("提示: 倾向得分无法估计，退回按协变量分位数格子的 CEM 匹配")
    return None


def _quantile_bins(values, n_bins):
    """按**合并样本**的分位数把一列切成至多 n_bins 层，返回整数层号

    分位点同样是 pooled 的：分层是为了在同一个活动量水平上比较男女，用
    某一个性别自己的分位点切层会让"同一层"在两性之间指的不是同一件事。
    """
    values = np.asarray(values, dtype=float)
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=int)
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, int(n_bins) + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right").astype(int)


def matching_strata(frame, n_bins=DEFAULT_N_BINS):
    """(层号数组, 用的是哪种方法)

    优先：倾向得分的分位数层（一维，20 万用户上 O(n log n)，而且每一层
    几乎必然两性都有人）。倾向得分算不出来时退回 CEM：四个协变量各切
    分位数格子，格子的笛卡尔积做层——它在细分格子上更容易找不到配对，
    所以只作为退路，而且方法名会一路写进 variant_label 与 note。
    """
    prepared = _balance_frame(frame)
    score = _propensity_score(prepared)
    if score is not None:
        return _quantile_bins(score, n_bins), BALANCE_METHOD_PS

    keys = []
    for covariate in ("log_posts", "log_retweets", "n_active_days"):
        keys.append(_quantile_bins(
            prepared[covariate].to_numpy(dtype=float), max(int(n_bins) // 4, 2)))
    keys.append(prepared["verified_numeric"].fillna(-1).to_numpy(dtype=int))
    combined = np.zeros(len(frame), dtype=np.int64)
    for key in keys:
        combined = combined * 1000 + key.astype(np.int64)
    return combined, BALANCE_METHOD_CEM


def match_within_strata(frame, n_bins=DEFAULT_N_BINS, seed=0):
    """层内 1:1 抽取，返回 (匹配后的表, 方法, 诊断信息)

    每一层取 k = min(该层男性数, 该层女性数) 人：两性在每一层里人数相同，
    因此层内的活动量分布也就被强制对齐。层内人多的一侧随机取 k 个
    （种子由主 seed 确定性派生，可复现），取之前先按 user_id 排序，让结果
    与输入行序无关。
    """
    strata, method = matching_strata(frame, n_bins=n_bins)
    work = frame.copy()
    work["_stratum"] = strata
    rng = np.random.default_rng(voc._domain_seed(seed, "public", stream=_STREAM_MATCH))

    picked = []
    n_strata_used = 0
    for stratum in sorted(set(strata.tolist())):
        block = work[work["_stratum"] == stratum]
        male = block[block["gender"] == "m"].sort_values("user_id")
        female = block[block["gender"] == "f"].sort_values("user_id")
        k = min(len(male), len(female))
        if k == 0:
            continue
        n_strata_used += 1
        for group in (male, female):
            if len(group) == k:
                picked.append(group)
            else:
                index = np.sort(rng.choice(len(group), size=k, replace=False))
                picked.append(group.iloc[index])

    if not picked:
        matched = frame.iloc[0:0].copy()
    else:
        matched = pd.concat(picked, ignore_index=False).sort_index()
    info = {
        "method": method,
        "n_strata_total": int(len(set(strata.tolist()))),
        "n_strata_matched": int(n_strata_used),
    }
    return matched, method, info


# ---------------------------------------------------------------------------
# 诊断行
# ---------------------------------------------------------------------------

_GENDER_SUFFIX = {"m": "male", "f": "female"}


def _gender_stats(frame, gender):
    """某个性别的活动量分布（均值/中位数/P90）与认证比例"""
    subset = frame[frame["gender"] == gender]
    out = {}
    for name, column in (("posts", "n_posts"), ("retweets", "n_retweets"),
                         ("active_days", "n_active_days")):
        values = subset[column].to_numpy(dtype=float) if len(subset) else np.array([])
        out["mean_{}_{}".format(name, _GENDER_SUFFIX[gender])] = (
            float(np.mean(values)) if values.size else float("nan"))
        out["median_{}_{}".format(name, _GENDER_SUFFIX[gender])] = (
            float(np.median(values)) if values.size else float("nan"))
        out["p90_{}_{}".format(name, _GENDER_SUFFIX[gender])] = (
            float(np.quantile(values, 0.9)) if values.size else float("nan"))
    verified, known = _verified_flag_bool(subset)
    out["verified_share_{}".format(_GENDER_SUFFIX[gender])] = (
        float(verified[known].mean()) if known.any() else float("nan"))
    return out


def diagnostics_row(before, after, variant_family, variant_label, rule,
                    replicate=0, seed=None, threshold_variable=None,
                    threshold_quantile=np.nan, cut_points=None, note=None):
    """一行诊断：这个变体动了谁、阈值是多少、剩下的样本长什么样

    cut_points 直接来自 applied_cut_points，因此 threshold_male 与
    threshold_female 必然相等——把它们并排写进表里，是为了读者不必读代码
    就能确认"阈值是 pooled 的"这条最重要的性质。
    """
    cut_points = cut_points or {}
    n_male_before = int((before["gender"] == "m").sum())
    n_female_before = int((before["gender"] == "f").sum())
    n_male_after = int((after["gender"] == "m").sum())
    n_female_after = int((after["gender"] == "f").sum())

    row = {
        "variant_family": variant_family,
        "variant_label": variant_label,
        "replicate": int(replicate),
        "seed": float(seed) if seed is not None else np.nan,
        "rule": rule,
        "threshold_variable": threshold_variable,
        "threshold_quantile": float(threshold_quantile),
        "threshold_pooled": float(cut_points.get("pooled", np.nan)),
        "threshold_male": float(cut_points.get("m", np.nan)),
        "threshold_female": float(cut_points.get("f", np.nan)),
        "n_users_before": int(len(before)),
        "n_users_after": int(len(after)),
        "n_male_before": n_male_before,
        "n_female_before": n_female_before,
        "n_male_after": n_male_after,
        "n_female_after": n_female_after,
        "n_dropped_male": n_male_before - n_male_after,
        "n_dropped_female": n_female_before - n_female_after,
        "dropped_share_male": (
            (n_male_before - n_male_after) / n_male_before if n_male_before
            else float("nan")),
        "dropped_share_female": (
            (n_female_before - n_female_after) / n_female_before if n_female_before
            else float("nan")),
        "female_to_male_ratio": (
            n_female_after / n_male_after if n_male_after else float("nan")),
        "note": note,
    }
    row.update(_gender_stats(after, "m"))
    row.update(_gender_stats(after, "f"))
    return {key: row.get(key) for key in DIAGNOSTIC_SCHEMA}


def _append_frame(rows, path, schema, label):
    """按给定 schema 增量落盘（与 accounts._append_frame 同一套原子改名做法）

    单独实现是因为这两张附表的 schema 都不是 ROBUSTNESS_SCHEMA：
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
        prefix=".append_smp_tmp_", suffix=".parquet", dir=out_dir)
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


def append_balance(rows, path):
    return _append_frame(rows, path, BALANCE_SCHEMA, "平衡")


# ---------------------------------------------------------------------------
# 跑一个变体
# ---------------------------------------------------------------------------

def _sample_note(diagnostic, prefix=None, extra_cuts=None):
    """把"这个变体动了多少人、阈值是多少"压成一句写进结果行的 note

    诊断表里有全套数字，但只拿到 samples.parquet 一张表的读者也必须看得见
    这次删了谁：一个"删掉总体最高 1% 之后结论还在"的说法，在读者知道删掉的
    是 900 个男性和 30 个女性之前是不完整的。
    """
    parts = [] if not prefix else [prefix]
    parts.append("threshold_scope=pooled")
    if diagnostic.get("threshold_variable"):
        parts.append("{}_cut={:.4f}".format(
            diagnostic["threshold_variable"], diagnostic["threshold_pooled"]))
    # extra_cuts：一个变体动了多个变量时（截尾一次截四列），其余变量的
    # 截点也必须写进 note，否则它们只存在于诊断表里
    for variable, cut in (extra_cuts or []):
        parts.append("{}_cut={:.4f}".format(variable, float(cut)))
    parts.append("n_users={}".format(diagnostic["n_users_after"]))
    parts.append("n_dropped_male={}".format(diagnostic["n_dropped_male"]))
    parts.append("n_dropped_female={}".format(diagnostic["n_dropped_female"]))
    return ";".join(parts)


def _run_variant(frame, variant_family, variant_label, diagnostic, out_path,
                 diag_path, replicate=0, seed=None, layers=("M0", "M1"),
                 note=None):
    """一个样本变体：估一次六个量、写一行诊断，两处都立刻落盘"""
    rows = harness.estimate_all(
        frame, variant_family=variant_family, variant_label=variant_label,
        replicate=replicate, seed=seed, layers=layers,
    )
    rows = voc._annotate_note(rows, _sample_note(diagnostic, prefix=note))
    if out_path is not None:
        harness.append_rows(rows, out_path)
    append_diagnostics([diagnostic], diag_path)
    return rows


# ---------------------------------------------------------------------------
# §13.2 变体
# ---------------------------------------------------------------------------

def run_extreme_value_variants(year=config.YEAR, context=None, out_path=None,
                               diag_path=None,
                               winsor_quantiles=DEFAULT_WINSOR_QUANTILES,
                               trim_shares=DEFAULT_TRIM_SHARES):
    """§13.2：不处理 / log(1+x) / 合并 P95、P99 截尾 / 删顶端 1%、5%"""
    context = context or build_context(year)
    frame = context.user_table
    collected = []

    untreated_diag = diagnostics_row(
        frame, frame, VARIANT_FAMILY_EXTREME, "untreated",
        rule="no_extreme_value_treatment",
        note="activity_covariates_enter_M1_as_log1p_by_models_core",
    )
    rows = _run_variant(frame, VARIANT_FAMILY_EXTREME, "untreated", untreated_diag,
                        out_path, diag_path)
    collected.append(rows)

    # log(1+x)：**做不到，因此只留一行注明原因的 NaN 行**，见下面的长注释
    log_diag = dict(untreated_diag)
    log_diag["variant_label"] = LOG1P_LABEL
    log_diag["rule"] = "not_estimable:covariate_transform_is_fixed_in_models_core"
    log_diag["note"] = LOG1P_NOTE
    append_diagnostics([log_diag], diag_path)
    collected.append(voc._note_only_rows(
        LOG1P_LABEL, LOG1P_NOTE, outcome="source_entered", domain="public",
        out_path=out_path, variant_family=VARIANT_FAMILY_EXTREME,
    ))
    print("§13.2 “不处理 vs log(1+x)”这一对做不到（协变量变换写死在 "
          "models_core.MODEL_LAYERS 里），已按本模块处理不可用变体的同一套"
          "做法留一行注明原因的 NaN 行")

    for quantile in winsor_quantiles:
        label = "winsorised_pooled_p{}".format(int(round(float(quantile) * 100)))
        winsorised, thresholds = winsorise_pooled(frame, quantile)
        diagnostics = []
        for column in ACTIVITY_COLUMNS:
            cuts = applied_cut_points(frame, column, quantile)
            diagnostics.append(diagnostics_row(
                frame, winsorised, VARIANT_FAMILY_EXTREME, label,
                rule="winsorise_upper_tail_at_pooled_quantile",
                threshold_variable=column, threshold_quantile=quantile,
                cut_points=cuts,
                note="clipped_upper_only",
            ))
        print("{}: 合并阈值 {}".format(
            label, {c: round(float(t), 3) for c, t in thresholds.items()}))
        estimate_rows = harness.estimate_all(
            winsorised, variant_family=VARIANT_FAMILY_EXTREME, variant_label=label,
            replicate=0, seed=None,
        )
        # note 必须带上**全部四个**活动量列的阈值：只写 diagnostics[0]
        # （n_posts）会让另外三列的截点只存在于诊断表里，拿到
        # samples.parquet 一张表的读者看不到这个变体到底截了什么。
        estimate_rows = voc._annotate_note(
            estimate_rows,
            _sample_note(diagnostics[0], extra_cuts=[
                (row["threshold_variable"], row["threshold_pooled"])
                for row in diagnostics[1:]
            ]),
        )
        if out_path is not None:
            harness.append_rows(estimate_rows, out_path)
        append_diagnostics(diagnostics, diag_path)
        collected.append(estimate_rows)

    for share in trim_shares:
        label = "trim_pooled_top{}pct".format(int(round(float(share) * 100)))
        trimmed, cut, dropped = trim_top_users(frame, share)
        diagnostic = diagnostics_row(
            frame, trimmed, VARIANT_FAMILY_EXTREME, label,
            rule="drop_users_above_pooled_quantile_of_total_activity",
            threshold_variable="total_activity",
            threshold_quantile=1.0 - float(share),
            cut_points={"pooled": cut, "m": cut, "f": cut},
            note="strictly_greater_than_cut(no_tie_splitting)",
        )
        print("{}: 阈值 {:.1f}，删掉 {} 人（男 {} / 女 {}）".format(
            label, cut, int(dropped.sum()),
            diagnostic["n_dropped_male"], diagnostic["n_dropped_female"]))
        collected.append(_run_variant(
            trimmed, VARIANT_FAMILY_EXTREME, label, diagnostic, out_path, diag_path))

    return pd.concat(collected, ignore_index=True)


# ---------------------------------------------------------------------------
# §13.6 变体
# ---------------------------------------------------------------------------

def run_equal_size_gender(year=config.YEAR, n_replicates=DEFAULT_N_REPLICATES,
                          seed=0, context=None, out_path=None, diag_path=None):
    """§13.6：反复从女性中抽与男性等量的样本，每次都记下抽到了什么"""
    context = context or build_context(year)
    frame = context.user_table
    seeds = voc.replicate_seeds(seed, n_replicates)

    # 全样本参照行：没有它，读者拿到每次抽样的分布也无从判断"有没有代表性"
    append_diagnostics([diagnostics_row(
        frame, frame, VARIANT_FAMILY_EQUAL_SIZE, "full_sample_reference",
        rule="reference_full_sample_not_an_estimate",
        note="reference_row_only:no_estimate_attached",
    )], diag_path)

    n_male = int((frame["gender"] == "m").sum())
    n_female = int((frame["gender"] == "f").sum())
    print("§13.6 等量抽样: 男性 {:,} 人，从 {:,} 名女性中每次抽 {:,} 人，共 {} 次".format(
        n_male, n_female, min(n_male, n_female), n_replicates))

    collected = []
    for replicate in range(n_replicates):
        replicate_seed = seeds[replicate]
        draw_seed = voc._domain_seed(replicate_seed, "public",
                                     stream=_STREAM_EQUAL_SIZE)
        drawn = equal_size_draw(frame, draw_seed)
        note = None
        if n_female <= n_male:
            note = "female_pool_not_larger_than_male_count:all_women_kept"
        diagnostic = diagnostics_row(
            frame, drawn, VARIANT_FAMILY_EQUAL_SIZE,
            "equal_size_rep{}".format(replicate),
            rule="draw_women_without_replacement_to_match_male_count",
            replicate=replicate, seed=replicate_seed, note=note,
        )
        collected.append(_run_variant(
            drawn, VARIANT_FAMILY_EQUAL_SIZE, "equal_size_rep{}".format(replicate),
            diagnostic, out_path, diag_path, replicate=replicate,
            seed=replicate_seed,
        ))
    return (
        pd.concat(collected, ignore_index=True) if collected
        else pd.DataFrame(columns=list(harness.ROBUSTNESS_SCHEMA))
    )


def run_activity_balanced(year=config.YEAR, n_bins=DEFAULT_N_BINS, seed=0,
                          context=None, out_path=None, diag_path=None,
                          bal_path=None):
    """§13.6：按发帖量、转发量、活跃天数与认证状态做 1:1 匹配

    variant_label 里写着实际用的方法（ps_match / cem_match）与四个协变量，
    平衡前后的标准化均值差写进 sample_balance.parquet。

    参数名是 bal_path 而不是 balance_path：后者会遮蔽模块级的
    `balance_path()` 函数，函数体内再想拿默认路径就拿不到了。
    """
    context = context or build_context(year)
    frame = context.user_table
    matched, method, info = match_within_strata(frame, n_bins=n_bins, seed=seed)
    label = "balanced_{}_1to1_on_posts_retweets_days_verified".format(method)

    rows_before = balance_rows(frame, VARIANT_FAMILY_BALANCE, label, "before")
    rows_after = balance_rows(matched, VARIANT_FAMILY_BALANCE, label, "after")
    append_balance(rows_before + rows_after, bal_path)
    for before, after in zip(rows_before, rows_after):
        print("平衡 {}: SMD {:.3f} -> {:.3f}".format(
            before["covariate"], before["std_mean_diff"], after["std_mean_diff"]))

    diagnostic = diagnostics_row(
        frame, matched, VARIANT_FAMILY_BALANCE, label,
        rule="match_1to1_within_pooled_strata:{}".format(method),
        seed=seed,
        note="method={};n_strata_matched={}/{};covariates={}".format(
            method, info["n_strata_matched"], info["n_strata_total"],
            "+".join(BALANCE_COVARIATES)),
    )
    return _run_variant(
        matched, VARIANT_FAMILY_BALANCE, label, diagnostic, out_path, diag_path,
        seed=seed,
        note="balance_method={}".format(method),
    )


# ---------------------------------------------------------------------------
# §13.9 变体
# ---------------------------------------------------------------------------

# §13.9 是唯一允许估 M2 的一节（方案文档 §11.4）：这一节问的正是画像字段
# 本身，M2 收窄样本带来的混淆在这里是题中应有之义，在别处不是。
USER_TYPE_LAYERS = ("M0", "M1", "M2")


def run_gender_field_variants(year=config.YEAR, context=None, out_path=None,
                              diag_path=None,
                              automation_quantile=AUTOMATION_QUANTILE):
    """§13.9：性别字段一致性、机构账号、疑似自动化账号、认证个人 vs 普通用户"""
    context = context or build_context(year)
    frame = context.user_table
    collected = []

    def _subset_variant(mask, label, rule, note=None, thresholds=None,
                        threshold_variable=None, threshold_quantile=np.nan):
        subset = frame[mask.to_numpy(dtype=bool)].copy()
        cuts = None
        if thresholds is not None and threshold_variable is not None:
            cut = thresholds.get(threshold_variable, float("nan"))
            cuts = {"pooled": cut, "m": cut, "f": cut}
        diagnostic = diagnostics_row(
            frame, subset, VARIANT_FAMILY_USER_TYPE, label, rule=rule,
            threshold_variable=threshold_variable,
            threshold_quantile=threshold_quantile, cut_points=cuts, note=note,
        )
        print("{}: 保留 {:,} / {:,} 人（男 -{} / 女 -{}）".format(
            label, len(subset), len(frame),
            diagnostic["n_dropped_male"], diagnostic["n_dropped_female"]))
        return _run_variant(
            subset, VARIANT_FAMILY_USER_TYPE, label, diagnostic, out_path,
            diag_path, layers=USER_TYPE_LAYERS, note=note,
        )

    def _keep_all_but(mask):
        """取反：被这条规则标中的用户剔除，其余保留"""
        return pd.Series(~mask.to_numpy(dtype=bool), index=frame.index)

    if context.has_demographic_gender:
        collected.append(_subset_variant(
            gender_fields_agree_mask(frame), "gender_fields_agree",
            rule="keep_users_whose_gender_and_demographic_gender_agree",
            note="unmappable_demographic_gender_counts_as_neither_agree_nor_conflict",
        ))
        collected.append(_subset_variant(
            _keep_all_but(gender_conflict_mask(frame)),
            "exclude_gender_conflict",
            rule="drop_users_whose_two_gender_fields_disagree",
            note="only_users_with_two_mappable_and_different_values_are_dropped",
        ))
    else:
        note = (
            "demographic_gender_not_available_in_user_table:"
            "add_it_to_profile_join.PROFILE_COLUMNS_and_rebuild_"
            "user_domain_with_profile;agreement_must_not_be_assumed"
        )
        for label in ("gender_fields_agree_unavailable",
                      "exclude_gender_conflict_unavailable"):
            collected.append(voc._note_only_rows(
                label, note, outcome="source_entered", domain="public",
                out_path=out_path, variant_family=VARIANT_FAMILY_USER_TYPE,
            ))

    collected.append(_subset_variant(
        _keep_all_but(institutional_mask(frame)),
        "exclude_institutional_accounts",
        rule="drop_verified_type_in_{}".format(list(INSTITUTIONAL_VERIFIED_TYPES)),
        note="verified_type_code_table_is_an_assumption(see profile_join)",
    ))

    automated, thresholds = automation_mask(frame, quantile=automation_quantile)
    collected.append(_subset_variant(
        _keep_all_but(automated),
        "exclude_suspected_automated_or_marketing",
        rule=AUTOMATION_RULE,
        thresholds=thresholds, threshold_variable="total_activity",
        threshold_quantile=automation_quantile,
        note="pre_specified_rule;posts_per_active_day_cut={:.4f}".format(
            thresholds["posts_per_active_day"]),
    ))

    collected.append(_subset_variant(
        verified_individual_mask(frame), "verified_individuals_only",
        rule="keep_verified_and_not_institutional",
        note="unknown_verification_excluded",
    ))
    collected.append(_subset_variant(
        ordinary_user_mask(frame), "ordinary_users_only",
        rule="keep_known_unverified_users",
        note="unknown_verification_excluded",
    ))
    return pd.concat(collected, ignore_index=True)


# ---------------------------------------------------------------------------
# 全流程
# ---------------------------------------------------------------------------

def build(year=config.YEAR, n_replicates=DEFAULT_N_REPLICATES, seed=0,
          n_bins=DEFAULT_N_BINS, winsor_quantiles=DEFAULT_WINSOR_QUANTILES,
          trim_shares=DEFAULT_TRIM_SHARES,
          automation_quantile=AUTOMATION_QUANTILE):
    """§13.2 + §13.6 + §13.9 全部变体，逐变体增量落盘，最后写 manifest"""
    context = build_context(year)
    out_path = results_path()
    diag_path = diagnostics_path()
    bal_path = balance_path()
    os.makedirs(robustness_dir(), exist_ok=True)

    run_extreme_value_variants(year, context=context, out_path=out_path,
                               diag_path=diag_path,
                               winsor_quantiles=winsor_quantiles,
                               trim_shares=trim_shares)
    run_equal_size_gender(year, n_replicates=n_replicates, seed=seed,
                          context=context, out_path=out_path, diag_path=diag_path)
    run_activity_balanced(year, n_bins=n_bins, seed=seed, context=context,
                          out_path=out_path, diag_path=diag_path,
                          bal_path=bal_path)
    run_gender_field_variants(year, context=context, out_path=out_path,
                              diag_path=diag_path,
                              automation_quantile=automation_quantile)

    results = pd.read_parquet(out_path, engine="pyarrow",
                              columns=list(harness.ROBUSTNESS_SCHEMA))
    diagnostics = pd.read_parquet(diag_path, engine="pyarrow",
                                  columns=list(DIAGNOSTIC_SCHEMA))
    balance = pd.read_parquet(bal_path, engine="pyarrow",
                              columns=list(BALANCE_SCHEMA))
    matched = balance[balance["stage"] == "after"]
    demographic_coverage = demographic_gender_coverage(context.user_table)
    # 删顶端的不对称只能在 §13.2 这一族里看：把最大值取遍全部 family，
    # 真实数据上最大的那个来自 verified_individuals_only（它按定义只留下
    # 几千个认证账号，两性都被剔掉 97% 以上），与截尾一点关系都没有。
    extreme = diagnostics[diagnostics["variant_family"] == VARIANT_FAMILY_EXTREME]
    manifest = config.build_manifest(
        step="robustness_samples_{}".format(year),
        inputs=[os.path.relpath(context.source_path, config.OUTPUT_DIR)],
        params={
            "year": year,
            "n_replicates": n_replicates,
            "seed": seed,
            "seeds": voc.replicate_seeds(seed, n_replicates),
            # 本任务最重要的一条纪律，写进 manifest 让它可被引用
            "threshold_scope": "pooled_across_genders",
            "winsor_quantiles": [float(q) for q in winsor_quantiles],
            "trim_shares": [float(s) for s in trim_shares],
            "total_activity_definition": "+".join(TOTAL_ACTIVITY_COLUMNS),
            "activity_columns": list(ACTIVITY_COLUMNS),
            "equal_size_rule": "draw_women_without_replacement_to_match_male_count",
            "balance_method": str(
                diagnostics.loc[
                    diagnostics["variant_family"] == VARIANT_FAMILY_BALANCE, "rule"
                ].iloc[0]
            ) if (diagnostics["variant_family"] == VARIANT_FAMILY_BALANCE).any()
            else None,
            "balance_covariates": list(BALANCE_COVARIATES),
            "balance_n_bins": n_bins,
            "institutional_verified_types": list(INSTITUTIONAL_VERIFIED_TYPES),
            "automation_rule": AUTOMATION_RULE,
            "automation_quantile": float(automation_quantile),
            "user_type_layers": list(USER_TYPE_LAYERS),
            # demographic_gender 到底有没有：没有的话结果表里是两行注明原因
            # 的行，不是"两个字段一致"
            "demographic_gender_available": bool(context.has_demographic_gender),
            # 光有一个布尔值不够：真实数据里这一列可能存在、却有一大半取值
            # 认不出来（例如存成数字编码，见 _normalise_gender_token）。把
            # 可映射比例直接写下来，读者不必用减法去猜这个变体到底覆盖了谁。
            "demographic_gender_mappable_share": (
                float(demographic_coverage["mappable_share"])),
            "demographic_gender_mappable_users": (
                int(demographic_coverage["n_mappable"])),
            "demographic_gender_unmappable_users": (
                int(demographic_coverage["n_unmappable"])),
            "profile_columns_available": bool(context.has_profile),
            "source_table": os.path.basename(context.source_path),
            "diagnostics_file": os.path.basename(diag_path),
            "balance_file": os.path.basename(bal_path),
        },
        counts={
            "result_rows": int(len(results)),
            "diagnostic_rows": int(len(diagnostics)),
            "balance_rows": int(len(balance)),
            "variant_labels": int(results["variant_label"].nunique()),
            "users_total": int(len(context.user_table)),
            "users_male": int((context.user_table["gender"] == "m").sum()),
            "users_female": int((context.user_table["gender"] == "f").sum()),
            # 删顶端到底删掉了谁：两性不对称本身就是 §13.2 的发现。**只取
            # extreme_values 这一族**，否则最大值会来自 §13.9 的子样本变体，
            # 名字说的和数字来源就不是一回事了。
            "extreme_values_max_dropped_share_male": (
                float(extreme["dropped_share_male"].max()) if len(extreme) else None),
            "extreme_values_max_dropped_share_female": (
                float(extreme["dropped_share_female"].max()) if len(extreme) else None),
            "worst_smd_after_matching": (
                float(matched["std_mean_diff"].abs().max())
                if len(matched) else None
            ),
        },
    )
    config.write_manifest(manifest, manifest_dir(year))
    return {
        "results_path": out_path,
        "diagnostics_path": diag_path,
        "balance_path": bal_path,
    }


def _cli(fn, needs_balance=False):
    """CLI 包装：把 out_path / diag_path 的默认值补成模块的正式路径

    各 run_* 函数的路径参数默认是 None（供测试与 build() 显式传路径），
    但从命令行单跑一个子命令时，None 意味着"算了一整轮、一个字节都没写"，
    而且不会有任何提示。这层包装只在 CLI 入口生效：没显式给路径就写正式
    路径，并把写到哪里打印出来。
    """
    def _run(*args, **kwargs):
        kwargs.setdefault("out_path", results_path())
        kwargs.setdefault("diag_path", diagnostics_path())
        if needs_balance:
            kwargs.setdefault("bal_path", balance_path())
        os.makedirs(robustness_dir(), exist_ok=True)
        print("输出目录: {}".format(robustness_dir()))
        return fn(*args, **kwargs)

    _run.__doc__ = fn.__doc__
    _run.__name__ = fn.__name__
    return _run


if __name__ == "__main__":
    fire.Fire({
        "build": build,
        "extreme_values": _cli(run_extreme_value_variants),
        "equal_size_gender": _cli(run_equal_size_gender),
        "activity_balanced": _cli(run_activity_balanced, needs_balance=True),
        "gender_field_variants": _cli(run_gender_field_variants),
    })
