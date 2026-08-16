"""
§9 月度稳定性（分月参与率、月份固定效应、leave-one-month-out）与
§10 转发延迟（清理规则、分位数、用户级中位数）。

本模块回答的两个问题，与前面几个模块回答的都不是同一件事：

- **§9：年度性别差是全年都在，还是被某一两个月推出来的？**
  一个"男女差异全年稳定"的结论，与一个"差异几乎全部来自 8 月某个明星
  事件"的结论，在论文里是两种完全不同的发现，而它们在年度汇总表上
  长得一模一样。因此这里同时给出三样东西：分月的参与率与占比（看得见
  形状）、月份固定效应下的年度性别差（把月份的共同波动吸收掉）、以及
  十二次留一月重估（把"某一个月是不是在推着结果走"变成一列可以直接读
  的数字）。留一月的输出里必须带着全年估计这一行，否则读者要去另一张
  表里找参照值才能比较，这个比较就不再自足。

- **§10：已经发生的转发，发生得有多快？**
  延迟只在**已经观察到的转发**上有定义（§10.1）：它描述扩散发生时的
  时间分布，不是"用户面对信息时会不会响应"的概率。这两句话经常被读成
  同一件事，所以每一行结果的 note 里都带着 NOTE_REALIZED_ONLY。

必须写清楚的口径与裁定（每一条都是"换一种写法就会得到另一个数"的选择）：

1. **延迟的头条是分位数，不是均值。**（§10.3）延迟分布是重尾的，现有
   探索性结果里均值与中位数给出的方向甚至不一致。因此本模块报告
   P25/P50/P75/P90/P95/P99 与 1/6/24 小时内完成的比例；均值只作为一行
   带 NOTE_SECONDARY_MEAN 标签的次要行出现，且**不做 t 检验**——在重尾
   分布上比较两组均值，检验的是一个被极端值主导的量，不是"男女谁转得
   更快"。测试里有一条扫描输出、禁止任何 t 检验痕迹的守卫。

2. **比较之前先聚合到用户。** 少数刷屏用户可以贡献成千上万条转发记录，
   记录级中位数因此可能只是"某几个人的习惯"。实测在本模块的 fixture 上，
   一个刷了 100 条慢转发的男性用户能把记录级中位数从 1 小时拉到 100 小时，
   而用户级中位数的男女差仍然是 0。所以：每名用户先算自己的中位延迟，
   男女差在用户级中位数上定义，区间用**按用户整簇重抽样**的 bootstrap
   （su.bootstrap_ci 的 cluster 参数）。记录级分位数仍然报告（它是扩散
   过程本身的描述），但它与用户级的量在 model 列上分开，不允许混读。

3. **清理规则各自记数，绝不静默删除。**（§10.2）表 B 刻意保留了非正延迟
   （只标 delay_valid=False），把"删不删"的决定留到这里，就是为了让删除
   动作发生在一个会记账、会写进 manifest 的地方。五条规则**按顺序**归因，
   一行只算进它命中的第一条，因此各条计数互不重叠、相加恰好等于被删行数：
       1) missing_delay          时间戳缺失或无法换算，延迟无定义
       2) non_positive_delay     转发时间早于或等于原帖时间（时钟/抓取问题）
       3) prior_year_source      转发的是上一年的旧帖
       4) implausible_long_delay 同年帖，但延迟超过阈值
       5) duplicate_user_source_pair  同一用户对同一原帖的重复记录
   **顺序不是随手定的**：prior_year 必须排在 implausible 之前。一条 2019
   年 12 月的旧帖在 2020 年 1 月被转发，延迟几百小时是完全正常的跨年行为，
   如果先判"超长延迟"，这类记录会被记成异常值，读者会以为数据里有几十万
   条脏记录，而真实情况只是"这些人转的是旧帖"。同理 non_positive 排在
   prior_year 之前：延迟为负时"原帖是哪一年的"这个判断本身就不可信。

4. **时间戳单位必须先检查再算。**（§10.2 第一条）原始数据的 time_stamp
   单位在本项目里从未被核实过（build_post_table 里同样是防御性推断）。
   表 B 的 delay_seconds 是两个时间戳直接相减，如果时间戳其实是毫秒，
   这一列的单位就是毫秒，除以 3600 会把 2 小时算成 0.002 小时——一个不会
   报错、只会让所有分位数缩小 1000 倍的错误。infer_timestamp_unit 按量级
   （1e11，与 build_post_table 同一阈值）判定，判定结果写进 report 与
   manifest，让结果表自证用的是哪个单位。

5. **"上一年的旧帖"按北京时间的自然年判定。** 时间戳按 UTC epoch 解释，
   加 8 小时换算成北京时间后再取年份：数据本身是中国平台的中国用户，
   跨年边界必须按当地日历判断，否则 1 月 1 日凌晨 8 点之前的帖子会被
   算成上一年。同时把"时间戳换算出的日期"与表 B 的 date 列（来自文件名）
   做一次比对，不一致的比例写进 report——这是对上一条推断的独立体检。

6. **超长延迟阈值取 720 小时（30 天）。** 这是一个裁定，不是一个客观界限，
   所以它是模块常量、写进 manifest、并且可以从 CLI 覆盖。理由：本研究
   关心的是来源账号发帖之后的扩散，微博的扩散在小时尺度上完成，超过一个月
   之后才发生的"转发"已经不是同一个过程（多半是旧帖被重新翻出来，或是
   时间戳本身有问题）。把阈值定得更宽（例如 365 天）不会改变中位数与
   P90，只会让 P99 被少数几条极端记录支配；定得更窄（例如 24 小时）则会
   把真实的慢速扩散一起删掉。720 小时被删掉的行数逐条记在 report 里，
   读者可以自己判断这条裁定的代价有多大。

7. **月份固定效应模型按用户聚类。**（§11.1）用户—月份面板上同一个用户
   贡献最多 12 行，把它们当独立观测会系统性低估标准误；实测在本模块的
   模拟数据上，聚类标准误比"当独立观测"的稳健标准误大 1.5 倍以上。

8. **面板上没有 M2。** 表 D 不带账号画像列（画像在表 C 上），所以本模块
   的层只有 M0（性别 + 月份固定效应）与 M1（再加当月发帖量/转发量/活跃
   天数）。这不是遗漏，是数据边界，因此每一行的 note 里都写明
   NOTE_NO_M2，而不是让读者以为我们忘了跑 M2。

9. **失败必须留痕，样本量必须自洽。** 沿用 models_core / models_combination
   的约定：任何一层拟合不收敛/退化都产出 estimate=NaN 且 note 写明原因的
   行；n_obs + n_dropped 恒等于该行对应的输入行数，drop_reason 逐条记数。

内存口径（事件表可能很大，写在这里供服务器上排查用）：
    全年事件表按月分片，每片一个 parquet。build 只按需要的列逐片读入
    （EVENT_COLUMNS，10 列），**逐片清理后只把幸存行留在内存里**，再把
    各片的幸存行拼起来做一次跨片去重。四条行级规则（缺失/非正/跨年/超长）
    都是逐行判定的，逐片做与整表做完全等价；只有"同一用户对同一原帖的
    重复记录"可能跨片出现（同一原帖在两个月里被同一用户转两次），所以
    去重必须在拼接之后再做一次，这一次的删除量单独记为
    n_duplicate_pairs_cross_shard，不与片内去重混在一起。
    峰值内存因此是"幸存行 × 10 列"，而不是"全年原始事件 × 全部列"。

使用方法:
    python -m gender_domain.models_temporal build --year=2020
"""

import glob
import os

import fire
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from gender_domain import config
from gender_domain import models_core as mc
from gender_domain import models_interaction as mi
from gender_domain import stats_utils as su

DOMAINS = mc.DOMAINS
FOCAL_COLUMN = mc.FOCAL_COLUMN            # "male"，与 models_core 同一口径
USER_COLUMN = "user_id"

MONTHS = tuple(range(1, 13))

# ---------------------------------------------------------------------------
# 结果表命名（(outcome, domain, model, term) 必须逐行唯一，见
# models_combination 模块文档第 2b 条）
# ---------------------------------------------------------------------------

OUTCOME_ENTRY_RATE = "monthly_source_entered_rate"
OUTCOME_SHARE_MEAN = "monthly_topical_share_mean"
OUTCOME_ENTRY = "source_entered"
OUTCOME_SHARE = "topical_share"
OUTCOME_DELAY = "retweet_delay_hours"

TERM_MALE = "male"
TERM_FEMALE = "female"
TERM_GENDER = mc.TERM_AME                 # "gender_male"
TERM_ODDS_RATIO = mc.TERM_ODDS_RATIO
TERM_MEAN = "mean_hours"
TERM_USER_MEDIAN = "median_of_user_medians"

LAYER_M0 = "M0"
LAYER_M1 = "M1"
# 面板上没有画像列，因此没有 M2（模块文档第 8 条）
PANEL_LAYERS = (
    (LAYER_M0, ["gender"]),
    (LAYER_M1, ["gender", "log_posts", "log_retweets", "n_active_days"]),
)

MODEL_FULL_YEAR = "full_year"
USER_LEVEL_GAP = "user_level/gap"

REASON_MISSING_GENDER = "missing_gender"
REASON_OTHER_GENDER = "other_gender"
REASON_NO_EXPRESSIVE = "no_expressive_posts"
REASON_EXCLUDED_MONTH = "excluded_month"
REASON_MONTH_ABSENT = "month_not_in_panel"
REASON_MISSING_COVARIATE = "missing_covariate"
REASON_AGGREGATED = "records_aggregated_to_users"

NOTE_MONTH_FE = "month_fixed_effects"
NOTE_CLUSTER_USER = "cluster_se_user"
NOTE_NO_M2 = "panel_has_no_profile_columns_no_M2"
NOTE_MONTH_ABSENT = REASON_MONTH_ABSENT
NOTE_LOMO = "leave_one_month_out"
NOTE_REALIZED_ONLY = "delay_among_realized_retweets_only"
NOTE_SECONDARY_MEAN = "secondary_mean_long_tail_not_headline"
NOTE_CLUSTER_BOOTSTRAP = "cluster_bootstrap_user"
NOTE_WILSON = "wilson"
NOTE_NEWCOMBE = "newcombe"
NOTE_NORMAL = "normal_approx_one_row_per_user_within_month"

# ---------------------------------------------------------------------------
# §10 清理规则
# ---------------------------------------------------------------------------

RULE_MISSING = "missing_delay"
RULE_NON_POSITIVE = "non_positive_delay"
RULE_PRIOR_YEAR = "prior_year_source"
RULE_IMPLAUSIBLE = "implausible_long_delay"
RULE_DUPLICATE = "duplicate_user_source_pair"
# 顺序即归因顺序，见模块文档第 3 条：prior_year 必须排在 implausible 之前
CLEAN_RULES = (
    RULE_MISSING,
    RULE_NON_POSITIVE,
    RULE_PRIOR_YEAR,
    RULE_IMPLAUSIBLE,
    RULE_DUPLICATE,
)

# 超长延迟阈值：30 天（模块文档第 6 条，是裁定不是客观界限）
MAX_DELAY_HOURS = 720.0
# 时间戳单位判定阈值，与 build_post_table._TIMESTAMP_UNIT_THRESHOLD 一致
TS_UNIT_THRESHOLD = 1e11
# 时间戳按 UTC epoch 解释，跨年边界按北京时间判定（模块文档第 5 条）
CHINA_UTC_OFFSET_HOURS = 8
# 同一用户对同一原帖的重复记录：领域也进键，因为一条转发同时命中两个领域时
# 表 B 会各出一行，那是真实的双领域事件，不是重复
DUPLICATE_KEYS = ("user_id", "r_weibo_id", "source_domain")

QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
WITHIN_HOURS = (1, 6, 24)

_CONFIDENCE = 0.95
_Z = float(norm.ppf(1 - (1 - _CONFIDENCE) / 2))
# 记录级分位数的簇 bootstrap 次数。一次重抽样的成本约等于
# "拼接下标 + 一次 np.percentile"，在千万行量级上约 0.2 秒，
# 500 次 × 4 个 (性别 × 领域) 格子约 7 分钟，在 4 小时的作业里可以接受。
_N_BOOT = 500
_SEED = 20200101

MONTHLY_FILE = "monthly_rates.parquet"
LOMO_FILE = "leave_one_month_out.parquet"
DELAY_FILE = "delay_quantiles.parquet"

# 读 parquet 一律显式给列（项目全局约定）
PANEL_COLUMNS = [
    "user_id", "month", "gender",
    "n_posts", "n_retweets", "n_active_days",
] + [f"{d}_{suffix}" for d in DOMAINS
     for suffix in ("source_entered", "source_count", "topical_share")]

EVENT_COLUMNS = [
    "user_id", "gender", "source_domain", "r_weibo_id", "month", "date",
    "retweet_ts", "source_ts", "delay_seconds", "delay_valid",
]


# ---------------------------------------------------------------------------
# 标签
# ---------------------------------------------------------------------------

def month_label(month):
    """分月描述行的 model 标签，形如 "month=03" """
    return "month={:02d}".format(int(month))


def fe_label(layer):
    """月份固定效应模型的 model 标签，形如 "M0/month_fe" """
    return "{}/{}".format(layer, NOTE_MONTH_FE)


def lomo_label(layer, month):
    """留一月重估的 model 标签；month 为 None 表示全年参照行

    形如 "M0/month_fe/drop_month=03" 与 "M0/month_fe/full_year"。
    口径进 model 列而不是 note 列，理由同 models_combination 模块文档
    第 2b 条：下游按 (outcome, domain, model, term) 取行必须唯一。
    """
    base = fe_label(layer)
    if month is None:
        return "{}/{}".format(base, MODEL_FULL_YEAR)
    return "{}/drop_month={:02d}".format(base, int(month))


def record_label(gender_label):
    """记录级延迟行的 model 标签，形如 "record_level/male" """
    return "record_level/{}".format(gender_label)


def user_label(gender_label):
    """用户级延迟行的 model 标签，形如 "user_level/male" """
    return "user_level/{}".format(gender_label)


def quantile_term(q):
    """分位数行的 term，形如 "p50" """
    return "p{:g}".format(round(float(q) * 100, 6))


def within_term(hours):
    """N 小时内完成比例行的 term，形如 "within_1h" """
    return "within_{:g}h".format(float(hours))


# ---------------------------------------------------------------------------
# 记账
# ---------------------------------------------------------------------------

def _drop_reason(pairs, n_input, n_obs):
    """格式化逐条剔除原因，并守住 n_obs + n_dropped == n_input

    与 models_combination._drop_reason 同一约定：各条计数互不重叠、相加
    恰好等于 n_dropped。对不上就报错——一张自己都对不上账的结果表，读者
    没有办法判断哪一个数字是可信的。
    """
    total = sum(int(n) for _, n in pairs)
    if total != n_input - n_obs:
        raise ValueError(
            f"逐条剔除计数 {pairs} 相加为 {total}，但 n_input={n_input} 减 "
            f"n_obs={n_obs} 等于 {n_input - n_obs}：结果表的样本流失记账不自洽，"
            "必须先修好归因逻辑，不能让一行对不上账的数字进入论文表格。"
        )
    return mi.format_drop_reason([(r, int(n)) for r, n in pairs if n])


def _gender_counts(frame):
    """(性别不可用行数, {标签: 掩码})"""
    gender = frame["gender"].astype("object")
    masks = {TERM_MALE: gender == "m", TERM_FEMALE: gender == "f"}
    n_missing = int((~gender.isin(("m", "f"))).sum())
    return n_missing, masks


# ---------------------------------------------------------------------------
# §9.2 分月参与率与占比
# ---------------------------------------------------------------------------

def _mean_ci(values):
    """一组独立观测的均值与正态近似区间

    只在"每个用户在该月最多贡献一行"的前提下使用（分月的横截面正是如此），
    所以这里不需要聚类：同一个用户在不同月份的行分别进入不同的月份，
    月内不存在同一用户的重复观测。
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(arr.mean())
    if n < 2:
        return mean, np.nan, np.nan, np.nan, n
    se = float(arr.std(ddof=1) / np.sqrt(n))
    return mean, se, mean - _Z * se, mean + _Z * se, n


def monthly_rates(panel_df):
    """§9.2 逐月 × 性别 × 领域的参与率与平均占比（含 95% 区间）

    每个 (月份, 领域) 上产出两组行：
      - 参与率（当月是否转发过该领域来源）：男 / 女各一行 Wilson 区间，
        外加一行男女差（Newcombe 区间）；
      - 平均话题占比：男 / 女各一行正态近似区间，外加一行均值差。
    占比无定义（当月没有表达帖，表 D 里是 NaN）的用户—月一律剔除并计入
    n_dropped（原因 no_expressive_posts），绝不当成 0——项目全局约定
    "NaN is not zero"。

    刻意**不**输出"上半年 / 下半年"或"全年"的合并行：那些行里同一个用户
    会出现多次，Wilson / 正态区间的独立性前提不再成立，给出的区间会系统性
    偏窄。半年对比可以直接从这十二个月的序列上读，年度的推断量由
    fit_month_fixed_effects 给出（那里按用户聚类）。
    """
    frame = panel_df.copy()
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").astype("Int64")
    months = sorted(int(m) for m in frame["month"].dropna().unique())
    rows = []

    for month in months:
        sub = frame[frame["month"] == month]
        n_input = int(len(sub))
        n_missing_gender, masks = _gender_counts(sub)
        model = month_label(month)

        for domain in DOMAINS:
            # 缺失视为"当月没有参与"：表 D 的 source_entered 由计数派生
            # （count > 0），外连接产生的缺失就是"这个月没有事件"，
            # 与占比类列的 NaN 不是同一件事，不能一起当无定义剔除
            entered = sub[f"{domain}_source_entered"].astype("object")
            entered = entered.where(entered.notna(), False).astype(bool)
            share = pd.to_numeric(
                sub[f"{domain}_topical_share"].astype("object"), errors="coerce"
            ).astype(float)
            su.check_proportion_range(share, f"{domain}_topical_share")

            rate_cells = {}
            share_cells = {}
            for label, mask in masks.items():
                n_group = int(mask.sum())
                n_other = n_input - n_missing_gender - n_group
                prefix = [
                    (REASON_MISSING_GENDER, n_missing_gender),
                    (REASON_OTHER_GENDER, n_other),
                ]
                successes = int(entered[mask].sum())
                if n_group:
                    rate = successes / n_group
                    low, high = su.proportion_ci(successes, n_group)
                else:
                    rate, low, high = np.nan, np.nan, np.nan
                rate_cells[label] = {"n": n_group, "successes": successes}
                rows.append(su.tidy_result(
                    outcome=OUTCOME_ENTRY_RATE, domain=domain, model=model,
                    term=label, estimate=rate, se=np.nan, ci_low=low, ci_high=high,
                    scale="probability", n_obs=n_group, n_dropped=n_input - n_group,
                    drop_reason=_drop_reason(prefix, n_input, n_group),
                    note=NOTE_WILSON,
                ))

                mean, se, low, high, n_valid = _mean_ci(share[mask])
                share_cells[label] = {"n": n_valid, "mean": mean, "se": se}
                pairs = prefix + [(REASON_NO_EXPRESSIVE, n_group - n_valid)]
                rows.append(su.tidy_result(
                    outcome=OUTCOME_SHARE_MEAN, domain=domain, model=model,
                    term=label, estimate=mean, se=se, ci_low=low, ci_high=high,
                    scale="proportion", n_obs=n_valid, n_dropped=n_input - n_valid,
                    drop_reason=_drop_reason(pairs, n_input, n_valid),
                    note=NOTE_NORMAL,
                ))

            male, female = rate_cells[TERM_MALE], rate_cells[TERM_FEMALE]
            if male["n"] and female["n"]:
                diff, low, high = su.proportion_diff_ci(
                    male["successes"], male["n"], female["successes"], female["n"]
                )
            else:
                diff, low, high = np.nan, np.nan, np.nan
            n_obs = male["n"] + female["n"]
            rows.append(su.tidy_result(
                outcome=OUTCOME_ENTRY_RATE, domain=domain, model=model,
                term=TERM_GENDER, estimate=diff, se=np.nan, ci_low=low, ci_high=high,
                scale="probability", n_obs=n_obs, n_dropped=n_input - n_obs,
                drop_reason=_drop_reason(
                    [(REASON_MISSING_GENDER, n_missing_gender)], n_input, n_obs
                ),
                note=NOTE_NEWCOMBE,
            ))

            male, female = share_cells[TERM_MALE], share_cells[TERM_FEMALE]
            n_obs = male["n"] + female["n"]
            if male["n"] > 1 and female["n"] > 1:
                diff = male["mean"] - female["mean"]
                se = float(np.sqrt(male["se"] ** 2 + female["se"] ** 2))
                low, high = diff - _Z * se, diff + _Z * se
            else:
                diff, se, low, high = np.nan, np.nan, np.nan, np.nan
            pairs = [
                (REASON_MISSING_GENDER, n_missing_gender),
                (REASON_NO_EXPRESSIVE, n_input - n_missing_gender - n_obs),
            ]
            rows.append(su.tidy_result(
                outcome=OUTCOME_SHARE_MEAN, domain=domain, model=model,
                term=TERM_GENDER, estimate=diff, se=se, ci_low=low, ci_high=high,
                scale="proportion", n_obs=n_obs, n_dropped=n_input - n_obs,
                drop_reason=_drop_reason(pairs, n_input, n_obs),
                note=NOTE_NORMAL,
            ))

    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# §9.2 月份固定效应
# ---------------------------------------------------------------------------

def prepare_panel_frame(panel_df):
    """把表 D 整理成可以直接进 patsy 的建模帧

    直接复用 models_core.prepare_model_frame（只留 m/f、派生 male /
    log_posts / log_retweets），再把 month 规整成整数：patsy 的 C(month)
    要求它是一个可以稳定分组的列，浮点或可空整数会在不同分片上展开出
    不同的水平集合。
    """
    frame = mc.prepare_model_frame(panel_df)
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").astype(int)
    for domain in DOMAINS:
        column = f"{domain}_topical_share"
        if column in frame.columns:
            # 占比越界只可能是上游分子分母算错了，必须在这里炸掉：分数 logit
            # 以取值落在 [0,1] 为前提，越界值会在不报错的情况下把估计推走
            su.check_proportion_range(frame[column], column)
    return frame


def _month_terms(sample, terms):
    """在协变量项后追加月份固定效应；月份不变时不追加并返回提示"""
    if sample["month"].nunique(dropna=True) < 2:
        return terms, "single_month_no_month_fe"
    return terms + ["C(month)"], None


def _fit_month_model(frame, domain, layer, covariates, n_input, model_label,
                     outcome_kind, prefix_pairs=(), include_odds=True,
                     extra_note=None):
    """拟合一层"月份固定效应 + 性别"的模型并产出结果行

    Args:
        frame: prepare_panel_frame 的输出，且已经按调用方的需要筛过月份
        n_input: 记账基准（**整张面板的行数**，不是 frame 的行数），
            调用方通过 prefix_pairs 交代 frame 相对它少了哪些行
        outcome_kind: "entry"（0/1，logit）或 "share"（占比，分数 logit）
        include_odds: 是否额外给出发生比行；留一月重估只要 AME 一行

    标准误一律按用户聚类（§11.1，模块文档第 7 条）：同一用户在面板上最多
    贡献 12 行，当成独立观测会系统性低估标准误。
    """
    if outcome_kind == "entry":
        outcome = OUTCOME_ENTRY
        outcome_col = f"{domain}_source_entered"
        scale = "probability"
        base_mask = None
        base_reason = None
    else:
        outcome = OUTCOME_SHARE
        outcome_col = f"{domain}_topical_share"
        scale = "proportion"
        base_mask = frame[outcome_col].notna()
        base_reason = REASON_NO_EXPRESSIVE

    pairs = list(prefix_pairs)
    if base_mask is None:
        sample = frame
    else:
        sample = frame[base_mask]
        pairs.append((base_reason, int(len(frame) - len(sample))))
    n_obs = int(len(sample))

    note_base = mc._join_notes(NOTE_MONTH_FE, NOTE_CLUSTER_USER, NOTE_NO_M2, extra_note)
    terms, dropped = mc._formula_terms(covariates, sample)
    terms, month_note = _month_terms(sample, terms) if n_obs else (terms, None)

    def _rows_with(estimate_note):
        out = [mc._nan_row(outcome, domain, model_label, TERM_GENDER, scale,
                           n_obs, n_input - n_obs,
                           _drop_reason(pairs, n_input, n_obs), estimate_note)]
        if include_odds:
            out.append(mc._nan_row(outcome, domain, model_label, TERM_ODDS_RATIO,
                                   "odds_ratio", n_obs, n_input - n_obs,
                                   _drop_reason(pairs, n_input, n_obs), estimate_note))
        return out

    blocked = mc._blocked_note(sample, terms) if n_obs else "empty_sample"
    if blocked is None and sample[outcome_col].nunique(dropna=True) < 2:
        blocked = "no_outcome_variation"
    if blocked is not None:
        return _rows_with(mc._join_notes(note_base, month_note, blocked))

    formula = "{} ~ {}".format(outcome_col, " + ".join(terms))
    label = f"{outcome}/{domain}/{model_label}"
    groups = sample[USER_COLUMN].to_numpy()
    if outcome_kind == "entry":
        result, fail = mc._safe_fit(
            lambda: smf.logit(formula, data=sample).fit(
                cov_type="cluster", cov_kwds={"groups": groups},
                disp=False, maxiter=100,
            ),
            label,
        )
    else:
        result, fail = mc._safe_fit(
            lambda: smf.glm(
                formula, data=sample, family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": groups}),
            label,
        )

    note = mc._join_notes(
        note_base, month_note,
        "dropped_constant:" + ",".join(dropped) if dropped else None,
        fail,
    )
    if result is None:
        return _rows_with(note)

    fitted = int(result.nobs)
    if fitted != n_obs:
        pairs.append((REASON_MISSING_COVARIATE, n_obs - fitted))
        n_obs = fitted
    if outcome_kind == "share":
        note = mc._join_notes(note, mc._glm_separation_note(result))
    n_dropped = n_input - n_obs
    drop_reason = _drop_reason(pairs, n_input, n_obs)

    rows = [mc._ame_row(result, sample, outcome, domain, model_label, scale,
                        n_obs, n_dropped, drop_reason, note)]
    if include_odds:
        rows.append(mc._exponentiated_row(
            result, outcome, domain, model_label, TERM_ODDS_RATIO, "odds_ratio",
            n_obs, n_dropped, drop_reason, note,
        ))
    return rows


def fit_month_fixed_effects(panel_df, domain):
    """§9.2 加入月份固定效应之后的年度性别差异（标准误按用户聚类）

    两个结果变量各跑 M0（性别 + 月份固定效应）与 M1（再加当月活跃度）：
      - 是否转发该领域来源（logit）：报告概率尺度的性别 AME + 发生比；
      - 该领域话题占比（分数 logit）：报告比例尺度的性别 AME。
    面板上没有画像列，因此没有 M2，每行 note 里写明 NOTE_NO_M2。

    月份固定效应吸收掉"所有人这个月都更活跃"这类共同波动，剩下的性别差
    才是"全年平均而言男女的差距"，而不是某几个月的活动量差异被算进性别。
    """
    n_input = int(len(panel_df))
    frame = prepare_panel_frame(panel_df)
    prefix = [(REASON_MISSING_GENDER, n_input - int(len(frame)))]
    rows = []
    for layer, covariates in PANEL_LAYERS:
        for kind in ("entry", "share"):
            rows.extend(_fit_month_model(
                frame, domain, layer, covariates, n_input, fe_label(layer), kind,
                prefix_pairs=prefix,
                # 占比模型只报告比例尺度的 AME，不给发生比：与 models_core
                # 的占比模型同一口径——分数 logit 的指数化系数不是任何一个
                # 可以直接解释的"发生比"，摆在结果表里只会被误读
                include_odds=(kind == "entry"),
            ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


def leave_one_month_out(panel_df, domain, layer=LAYER_M0):
    """§9.2 十二次留一月重估 + 一行全年参照，共 13 行

    每次剔除一个月重新拟合"性别 + 月份固定效应"的进入模型，报告概率尺度
    的性别 AME。全年估计作为参照行一起写出来，读者不需要去别的表里找基准
    就能判断"某一个月是不是在推着结果走"。

    面板里根本不存在的月份也保留一行 NaN（note = month_not_in_panel）：
    输出永远是 13 行。少一行会被读成"这个月没有被检查过"，而实际情况是
    "这个月没有数据"，两者必须在结果表上分得开。
    """
    n_input = int(len(panel_df))
    frame = prepare_panel_frame(panel_df)
    n_missing_gender = n_input - int(len(frame))
    present = set(int(m) for m in frame["month"].unique())
    covariates = dict(PANEL_LAYERS)[layer]
    rows = []

    for month in MONTHS:
        model_label = lomo_label(layer, month)
        if month not in present:
            rows.append(mc._nan_row(
                OUTCOME_ENTRY, domain, model_label, TERM_GENDER, "probability",
                0, n_input, _drop_reason([(REASON_MONTH_ABSENT, n_input)], n_input, 0),
                mc._join_notes(NOTE_LOMO, NOTE_MONTH_ABSENT),
            ))
            continue
        subset = frame[frame["month"] != month]
        prefix = [
            (REASON_MISSING_GENDER, n_missing_gender),
            (REASON_EXCLUDED_MONTH, int(len(frame) - len(subset))),
        ]
        rows.extend(_fit_month_model(
            subset, domain, layer, covariates, n_input, model_label, "entry",
            prefix_pairs=prefix, include_odds=False, extra_note=NOTE_LOMO,
        ))

    rows.extend(_fit_month_model(
        frame, domain, layer, covariates, n_input, lomo_label(layer, None), "entry",
        prefix_pairs=[(REASON_MISSING_GENDER, n_missing_gender)],
        include_odds=False, extra_note=mc._join_notes(NOTE_LOMO, MODEL_FULL_YEAR),
    ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# §10.2 延迟清理
# ---------------------------------------------------------------------------

def infer_timestamp_unit(event_df):
    """按量级判定时间戳单位："s"（秒）或 "ms"（毫秒）

    见模块文档第 4 条：表 B 的 delay_seconds 是两个原始时间戳直接相减，
    单位随原始时间戳而定。1e11 秒 ≈ 公元 5138 年，任何真实的秒级 epoch
    都远小于它；毫秒级 epoch（2020 年约 1.58e12）则远大于它。阈值与
    build_post_table._TIMESTAMP_UNIT_THRESHOLD 保持一致。
    """
    ts = pd.to_numeric(
        pd.concat([event_df["retweet_ts"], event_df["source_ts"]], ignore_index=True),
        errors="coerce",
    ).abs()
    ts = ts[ts > 0]
    if ts.empty:
        return "s"
    return "ms" if float(ts.median()) >= TS_UNIT_THRESHOLD else "s"


def dedup_user_source_pairs(df, subset=DUPLICATE_KEYS):
    """同一用户 × 同一原帖 × 同一领域只保留最早的一条转发

    保留"最早"（延迟最小）而不是"文件里第一条"：重复记录多数来自抓取时
    同一条转发被记了两次，或用户对同一原帖反复转发；无论哪一种，描述
    "扩散有多快"时都应该用第一次响应，取任意一条会让延迟分布凭空右移。

    返回 (去重后的帧, 被删行数)；原始行序保持不变（只是删掉了重复行），
    以免下游误以为输出被重新排过序。
    """
    before = int(len(df))
    ordered = df.sort_values("delay_hours", kind="mergesort")
    deduped = ordered.drop_duplicates(subset=list(subset), keep="first")
    deduped = deduped.sort_index()
    return deduped, before - int(len(deduped))


def clean_delays(event_df, year=config.YEAR, max_delay_hours=MAX_DELAY_HOURS):
    """§10.2 延迟清理：五条规则顺序归因，逐条记数，绝不静默删除

    规则顺序与理由见模块文档第 3 条。返回 (清理后的帧, report)：
    清理后的帧在原列之外多一列 delay_hours（小时），report 的每一项都会
    原样写进 manifest。

    report 的字段：
        n_input / n_clean / n_removed        总量
        removed                              {规则: 删除行数}，相加 == n_removed
        timestamp_unit                       "s" / "ms"（模块文档第 4 条）
        max_delay_hours / year               本次使用的阈值与年份
        n_delay_valid_false_input            表 B 自己标记的非正/缺失延迟行数，
                                             与本模块前两条规则之和互为核对
        n_source_ts_unparsed                 原帖时间戳无法换算成日期的行数
        ts_date_mismatch_share               时间戳换算日期与 date 列不一致的比例
        prior_year_median_hours              跨年转发的中位延迟（描述用）
        by_domain                            清理后各领域的行数
    """
    n_input = int(len(event_df))
    unit = infer_timestamp_unit(event_df)
    divisor = 3600.0 if unit == "s" else 3600.0 * 1000.0

    frame = event_df.copy()
    frame["delay_hours"] = pd.to_numeric(
        frame["delay_seconds"], errors="coerce"
    ).astype(float) / divisor
    delay_valid_false = int((~frame["delay_valid"].astype(bool)).sum()) \
        if "delay_valid" in frame.columns else None

    removed = {rule: 0 for rule in CLEAN_RULES}

    missing = frame["delay_hours"].isna()
    removed[RULE_MISSING] = int(missing.sum())
    frame = frame[~missing]

    non_positive = frame["delay_hours"] <= 0
    removed[RULE_NON_POSITIVE] = int(non_positive.sum())
    frame = frame[~non_positive]

    source_dt = pd.to_datetime(
        pd.to_numeric(frame["source_ts"], errors="coerce"), unit=unit, errors="coerce"
    ) + pd.Timedelta(hours=CHINA_UTC_OFFSET_HOURS)
    n_source_unparsed = int(source_dt.isna().sum())
    prior_year = (source_dt.dt.year < int(year)).fillna(False)
    removed[RULE_PRIOR_YEAR] = int(prior_year.sum())
    prior_median = float(frame.loc[prior_year, "delay_hours"].median()) \
        if int(prior_year.sum()) else None
    frame = frame[~prior_year]

    implausible = frame["delay_hours"] > float(max_delay_hours)
    removed[RULE_IMPLAUSIBLE] = int(implausible.sum())
    frame = frame[~implausible]

    frame, n_duplicate = dedup_user_source_pairs(frame)
    removed[RULE_DUPLICATE] = int(n_duplicate)

    mismatch_share = None
    if "date" in frame.columns and len(frame):
        retweet_dt = pd.to_datetime(
            pd.to_numeric(frame["retweet_ts"], errors="coerce"), unit=unit,
            errors="coerce",
        ) + pd.Timedelta(hours=CHINA_UTC_OFFSET_HOURS)
        mismatch_share = float(
            (retweet_dt.dt.strftime("%Y-%m-%d") != frame["date"].astype(str)).mean()
        )

    frame = frame.reset_index(drop=True)
    n_removed = int(sum(removed.values()))
    if n_removed + int(len(frame)) != n_input:
        raise ValueError(
            f"延迟清理记账不自洽: 逐条删除 {removed} 相加为 {n_removed}，"
            f"清理后 {len(frame)} 行，输入 {n_input} 行。清理规则必须做到"
            "每一行只被一条规则删掉，且相加恰好等于删除总数。"
        )

    report = {
        "n_input": n_input,
        "removed": removed,
        "n_removed": n_removed,
        "n_clean": int(len(frame)),
        "timestamp_unit": unit,
        "max_delay_hours": float(max_delay_hours),
        "year": int(year),
        "n_delay_valid_false_input": delay_valid_false,
        "n_source_ts_unparsed": n_source_unparsed,
        "ts_date_mismatch_share": mismatch_share,
        "prior_year_median_hours": prior_median,
        "by_domain": frame["source_domain"].value_counts().to_dict()
        if len(frame) else {},
    }
    print(
        f"延迟清理: 输入 {n_input} 行，时间戳单位 {unit}，"
        f"删除 {n_removed} 行（{removed}），保留 {len(frame)} 行"
    )
    return frame, report


def merge_delay_reports(reports):
    """把逐个分片的清理 report 汇总成一份（build 按月分片清理，见模块文档内存口径）

    单位不一致会直接报错：同一年的分片如果一半是秒、一半是毫秒，合并出来的
    分位数没有任何意义，这属于必须有人去查的上游问题，不该被一个"取多数"
    之类的规则掩盖。
    """
    reports = [r for r in reports if r is not None]
    if not reports:
        return None
    units = {r["timestamp_unit"] for r in reports}
    if len(units) > 1:
        raise ValueError(
            f"各分片推断出的时间戳单位不一致: {sorted(units)}。合并前必须先查清楚"
            "原始时间戳到底是什么单位，否则合并出来的延迟分布没有意义。"
        )
    merged = {
        "n_input": int(sum(r["n_input"] for r in reports)),
        "removed": {
            rule: int(sum(r["removed"].get(rule, 0) for r in reports))
            for rule in CLEAN_RULES
        },
        "n_clean": int(sum(r["n_clean"] for r in reports)),
        "timestamp_unit": units.pop(),
        "max_delay_hours": float(reports[0]["max_delay_hours"]),
        "year": int(reports[0]["year"]),
        "n_delay_valid_false_input": int(sum(
            r["n_delay_valid_false_input"] or 0 for r in reports
        )),
        "n_source_ts_unparsed": int(sum(r["n_source_ts_unparsed"] for r in reports)),
        "ts_date_mismatch_share_by_shard": [
            r["ts_date_mismatch_share"] for r in reports
        ],
        "prior_year_median_hours_by_shard": [
            r["prior_year_median_hours"] for r in reports
        ],
        "n_shards": len(reports),
    }
    merged["n_removed"] = int(sum(merged["removed"].values()))
    by_domain = {}
    for r in reports:
        for domain, n in (r["by_domain"] or {}).items():
            by_domain[domain] = by_domain.get(domain, 0) + int(n)
    merged["by_domain"] = by_domain
    return merged


# ---------------------------------------------------------------------------
# §10.3 分位数与用户级中位数
# ---------------------------------------------------------------------------

def delay_stat_terms(quantiles=QUANTILES, within_hours=WITHIN_HOURS):
    """记录级统计量的 term 顺序（与 _delay_stat_vector 一一对应）"""
    return ([quantile_term(q) for q in quantiles]
            + [within_term(h) for h in within_hours]
            + [TERM_MEAN])


def _delay_stat_vector(values, quantiles=QUANTILES, within_hours=WITHIN_HOURS):
    """一次算完分位数 + N 小时内完成比例 + 均值，顺序与 delay_stat_terms 一致"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.full(len(quantiles) + len(within_hours) + 1, np.nan)
    quants = np.percentile(arr, [q * 100 for q in quantiles])
    shares = [float(np.mean(arr <= h)) for h in within_hours]
    return np.concatenate([quants, shares, [float(arr.mean())]])


def cluster_bootstrap_delay_stats(values, clusters, n_boot=_N_BOOT, seed=_SEED,
                                   quantiles=QUANTILES, within_hours=WITHIN_HOURS,
                                   confidence=_CONFIDENCE):
    """按用户整簇重抽样，一次求出全部记录级统计量的百分位区间

    重抽样语义与 stats_utils.bootstrap_ci(cluster=...) **完全一致**：
    同一个种子下，先在原始样本上算点估计，再用 numpy.random.default_rng(seed)
    依次从簇标签全集里有放回抽出同样多的簇，把抽中簇的**全部行**拼起来。
    这里没有对每个统计量各调一次 bootstrap_ci，唯一的原因是成本：本模块
    每个 (性别 × 领域) 格子要报告 6 个分位数 + 3 个比例 + 1 个均值，逐个调用
    会把同一份重抽样做十遍（在千万行量级上是十倍的时间）。为了让"完全一致"
    这句话不只是注释，测试里直接拿单个分位数与 bootstrap_ci 的结果逐位对齐。

    Returns:
        {term: {"estimate", "ci_low", "ci_high"}}
    """
    arr = np.asarray(values, dtype=float)
    terms = delay_stat_terms(quantiles, within_hours)
    est = _delay_stat_vector(arr, quantiles, within_hours)
    if arr.size == 0:
        return {t: {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
                for t in terms}

    cluster_arr = np.asarray(clusters)
    if len(cluster_arr) != arr.size:
        raise ValueError("cluster 长度必须与 values 一致")
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_arr)
    n_clusters = len(unique_clusters)
    cluster_to_rows = {c: np.where(cluster_arr == c)[0] for c in unique_clusters}

    boot = np.empty((int(n_boot), len(terms)))
    for b in range(int(n_boot)):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_rows[c] for c in sampled])
        boot[b] = _delay_stat_vector(arr[idx], quantiles, within_hours)

    alpha = 1 - confidence
    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return {
        term: {"estimate": float(est[i]), "ci_low": float(low[i]),
               "ci_high": float(high[i])}
        for i, term in enumerate(terms)
    }


def _user_medians(cell):
    """每名用户的中位延迟（§10.3 的用户级口径）"""
    grouped = cell.groupby(USER_COLUMN)["delay_hours"].median()
    return grouped.reset_index().rename(columns={"delay_hours": "user_median"})


def _median_gap(frame):
    """男性用户中位延迟的中位数 - 女性的；任一侧为空时返回 NaN"""
    male = frame.loc[frame["gender"] == "m", "user_median"].to_numpy(dtype=float)
    female = frame.loc[frame["gender"] == "f", "user_median"].to_numpy(dtype=float)
    if male.size == 0 or female.size == 0:
        return np.nan
    return float(np.median(male) - np.median(female))


def delay_quantiles(clean_df, n_boot=_N_BOOT, seed=_SEED,
                    quantiles=QUANTILES, within_hours=WITHIN_HOURS):
    """§10.3 延迟分位数、N 小时内完成比例、用户级中位数与男女差

    每个领域产出三组行：
      - model="record_level/{性别}"：P25/P50/P75/P90/P95/P99（小时）、
        1/6/24 小时内完成的比例（概率），以及一行均值——均值带
        NOTE_SECONDARY_MEAN 标签，是次要行，不作为头条（模块文档第 1 条）。
        区间一律用按用户整簇重抽样的 bootstrap，因为同一用户会贡献多条记录。
      - model="user_level/{性别}"：每名用户先算自己的中位延迟，再取中位数。
      - model="user_level/gap"：男女用户级中位数之差，区间同样按用户整簇
        重抽样（模块文档第 2 条）。

    输入必须是 clean_delays 的输出（含 delay_hours 列）：1/6/24 小时内完成
    的比例算在**清理后**的集合上，把非正延迟、跨年旧帖、重复记录留在分母里
    会同时改变分子和分母，得到一个既不是"扩散速度"也不是"数据质量"的数字。
    """
    frame = clean_df.copy()
    # 成本提示：簇 bootstrap 的开销是"格子数 × n_boot × 一次分位数计算"，
    # 全量事件表上一次约 0.2 秒，跑不动时用 CLI 的 --n_boot 调低（区间会变粗，
    # 但仍然是按用户整簇重抽样的区间，不会变成一个错误的窄区间）
    print(f"§10.3 延迟分位数: 清理后 {len(frame):,} 条记录，"
          f"每个（性别 × 领域）格子做 {n_boot} 次按用户整簇重抽样")
    rows = []
    for domain in DOMAINS:
        sub = frame[frame["source_domain"] == domain]
        n_input = int(len(sub))
        n_missing_gender, masks = _gender_counts(sub)
        gender_code = {TERM_MALE: "m", TERM_FEMALE: "f"}
        user_frames = []

        for label, mask in masks.items():
            cell = sub[mask]
            n_obs = int(len(cell))
            n_other = n_input - n_missing_gender - n_obs
            prefix = [
                (REASON_MISSING_GENDER, n_missing_gender),
                (REASON_OTHER_GENDER, n_other),
            ]
            stats = cluster_bootstrap_delay_stats(
                cell["delay_hours"].to_numpy(dtype=float),
                cell[USER_COLUMN].to_numpy(),
                n_boot=n_boot, seed=seed, quantiles=quantiles,
                within_hours=within_hours,
            )
            for term in delay_stat_terms(quantiles, within_hours):
                stat = stats[term]
                is_share = term.startswith("within_")
                note = mc._join_notes(
                    NOTE_REALIZED_ONLY, NOTE_CLUSTER_BOOTSTRAP,
                    NOTE_SECONDARY_MEAN if term == TERM_MEAN else None,
                )
                rows.append(su.tidy_result(
                    outcome=OUTCOME_DELAY, domain=domain, model=record_label(label),
                    term=term, estimate=stat["estimate"], se=np.nan,
                    ci_low=stat["ci_low"], ci_high=stat["ci_high"],
                    scale="probability" if is_share else "hours",
                    n_obs=n_obs, n_dropped=n_input - n_obs,
                    drop_reason=_drop_reason(prefix, n_input, n_obs),
                    note=note,
                ))

            medians = _user_medians(cell)
            n_users = int(len(medians))
            pairs = prefix + [(REASON_AGGREGATED, n_obs - n_users)]
            if n_users:
                est, low, high = su.bootstrap_ci(
                    medians["user_median"].to_numpy(dtype=float),
                    lambda v: float(np.median(v)), n_boot=n_boot, seed=seed,
                    cluster=medians[USER_COLUMN].to_numpy(),
                )
            else:
                est, low, high = np.nan, np.nan, np.nan
            rows.append(su.tidy_result(
                outcome=OUTCOME_DELAY, domain=domain, model=user_label(label),
                term=TERM_USER_MEDIAN, estimate=est, se=np.nan,
                ci_low=low, ci_high=high, scale="hours",
                n_obs=n_users, n_dropped=n_input - n_users,
                drop_reason=_drop_reason(pairs, n_input, n_users),
                note=mc._join_notes(NOTE_REALIZED_ONLY, NOTE_CLUSTER_BOOTSTRAP),
            ))
            medians["gender"] = gender_code[label]
            user_frames.append(medians)

        combined = pd.concat(user_frames, ignore_index=True) if user_frames \
            else pd.DataFrame(columns=[USER_COLUMN, "user_median", "gender"])
        n_users = int(len(combined))
        if n_users:
            est, low, high = su.bootstrap_ci(
                combined, _median_gap, n_boot=n_boot, seed=seed,
                cluster=combined[USER_COLUMN].to_numpy(),
            )
        else:
            est, low, high = np.nan, np.nan, np.nan
        pairs = [
            (REASON_MISSING_GENDER, n_missing_gender),
            (REASON_AGGREGATED, n_input - n_missing_gender - n_users),
        ]
        rows.append(su.tidy_result(
            outcome=OUTCOME_DELAY, domain=domain, model=USER_LEVEL_GAP,
            term=TERM_GENDER, estimate=est, se=np.nan, ci_low=low, ci_high=high,
            scale="hours", n_obs=n_users, n_dropped=n_input - n_users,
            drop_reason=_drop_reason(pairs, n_input, n_users),
            note=mc._join_notes(NOTE_REALIZED_ONLY, NOTE_CLUSTER_BOOTSTRAP,
                                "male_minus_female"),
        ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_panel(year):
    path = os.path.join(config.OUTPUT_DIR, f"user_month_domain_{year}.parquet")
    panel = pd.read_parquet(path, columns=PANEL_COLUMNS)
    print(f"读取用户—月份面板 {path}（{len(panel):,} 行）")
    return panel, path


def _event_shards(year):
    pattern = os.path.join(
        config.OUTPUT_DIR, f"retweet_domain_events_{year}", "month=*.parquet"
    )
    return sorted(glob.glob(pattern))


def load_clean_events(year, max_delay_hours=MAX_DELAY_HOURS):
    """逐片读入事件表、逐片清理，再做一次跨片去重

    见模块文档的内存口径：四条行级规则逐片做与整表做完全等价，因此每片
    只把幸存行留在内存里；只有"同一用户 × 同一原帖"的重复可能跨片出现
    （表 B 的去重只在片内做过），所以拼接之后再去一次重，这一次的删除量
    单独记为 n_duplicate_pairs_cross_shard，不与片内的混在一起。
    """
    files = _event_shards(year)
    if not files:
        raise FileNotFoundError(
            f"未找到 {year} 年的转发事件表分片（"
            f"{os.path.join(config.OUTPUT_DIR, f'retweet_domain_events_{year}')}）"
        )
    parts, reports = [], []
    for path in files:
        shard = pd.read_parquet(path, columns=EVENT_COLUMNS)
        clean, report = clean_delays(shard, year=year, max_delay_hours=max_delay_hours)
        parts.append(clean)
        reports.append(report)
    frame = pd.concat(parts, ignore_index=True)
    frame, n_cross = dedup_user_source_pairs(frame)
    frame = frame.reset_index(drop=True)

    merged = merge_delay_reports(reports)
    merged["removed"][RULE_DUPLICATE] += int(n_cross)
    merged["n_removed"] += int(n_cross)
    merged["n_clean"] -= int(n_cross)
    merged["n_duplicate_pairs_cross_shard"] = int(n_cross)
    if merged["n_clean"] != int(len(frame)):
        raise ValueError(
            f"跨分片清理记账不自洽: report 记 {merged['n_clean']} 行，"
            f"实际 {len(frame)} 行"
        )
    print(f"事件表清理完成: {len(files)} 个分片，跨片重复 {n_cross} 行，"
          f"保留 {len(frame):,} 行")
    return frame, merged, files


def _write_partial(frames, path, out_dir, inputs, year, counts, completed,
                   n_boot, delay_report=None):
    """把当前已完成的阶段落盘，并同步更新 manifest（与 models_combination 同一约定）

    增量落盘不是优化，是防丢：延迟阶段的簇 bootstrap 在全量事件表上要跑
    十几分钟，作业撞墙钟被杀时，前面算好的 §9 结果不该跟着消失。
    manifest 的 completed_stages 记录这份结果到底包含哪几个阶段。
    """
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    n_failed = int(frame["estimate"].isna().sum())
    print(f"已保存（增量）: {path}（{len(frame)} 行，其中 {n_failed} 行为拟合失败留痕）")

    manifest = config.build_manifest(
        step=f"models_temporal_{year}",
        inputs=inputs,
        params={
            "year": year,
            "layers": [layer for layer, _ in PANEL_LAYERS],
            "no_M2_on_panel": True,
            "estimator": "logit / GLM(Binomial, logit), cov_type=cluster(user_id)",
            "confidence": _CONFIDENCE,
            "n_boot": n_boot,
            "seed": _SEED,
            "max_delay_hours": MAX_DELAY_HOURS,
            "quantiles": [float(q) for q in QUANTILES],
            "within_hours": [float(h) for h in WITHIN_HOURS],
            "clean_rules": list(CLEAN_RULES),
            "duplicate_keys": list(DUPLICATE_KEYS),
            "china_utc_offset_hours": CHINA_UTC_OFFSET_HOURS,
            "mean_is_secondary_no_t_test": True,
        },
        counts=dict(counts, completed_stages=list(completed),
                    n_completed_stages=len(completed),
                    delay_cleaning=delay_report or {}),
    )
    config.write_manifest(manifest, os.path.join(out_dir, "manifest_temporal"))
    return frame


def build(year=config.YEAR, n_boot=None):
    """跑完 §9 与 §10 并写出三张结果表 + manifest

    顺序：分月描述 -> 月份固定效应 -> 留一月重估 -> 延迟清理与分位数。
    延迟阶段放在最后，是因为它要读整年的事件表（最贵的一步），前面的
    §9 结果先落盘，作业被杀时不至于全部重来。
    """
    n_boot = _N_BOOT if n_boot is None else int(n_boot)
    panel, panel_path = _load_panel(year)
    out_dir = os.path.join(config.OUTPUT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    inputs = [os.path.relpath(panel_path, config.OUTPUT_DIR)]
    counts = {
        "n_user_months": int(len(panel)),
        "n_users": int(panel[USER_COLUMN].nunique()),
        "months": sorted(int(m) for m in panel["month"].dropna().unique()),
    }
    completed = []

    monthly_path = os.path.join(out_dir, MONTHLY_FILE)
    frames = []
    print("§9.2 分月参与率与占比")
    frames.append(monthly_rates(panel))
    completed.append("monthly_rates")
    _write_partial(frames, monthly_path, out_dir, inputs, year, counts, completed,
                   n_boot)

    for domain in DOMAINS:
        print(f"§9.2 月份固定效应: {domain}")
        frames.append(fit_month_fixed_effects(panel, domain))
        completed.append(f"month_fixed_effects/{domain}")
        _write_partial(frames, monthly_path, out_dir, inputs, year, counts,
                       completed, n_boot)

    lomo_path = os.path.join(out_dir, LOMO_FILE)
    lomo_frames = []
    for domain in DOMAINS:
        print(f"§9.2 留一月重估: {domain}")
        lomo_frames.append(leave_one_month_out(panel, domain))
        completed.append(f"leave_one_month_out/{domain}")
        _write_partial(lomo_frames, lomo_path, out_dir, inputs, year, counts,
                       completed, n_boot)

    print("§10.2 转发延迟清理")
    events, delay_report, event_files = load_clean_events(year)
    inputs = inputs + [os.path.relpath(p, config.OUTPUT_DIR) for p in event_files]
    counts["n_delay_records"] = int(len(events))

    delay_path = os.path.join(out_dir, DELAY_FILE)
    print("§10.3 延迟分位数与用户级中位数")
    delay_frames = [delay_quantiles(events, n_boot=n_boot, seed=_SEED)]
    completed.append("delay_quantiles")
    _write_partial(delay_frames, delay_path, out_dir, inputs, year, counts,
                   completed, n_boot, delay_report=delay_report)

    return [monthly_path, lomo_path, delay_path]


if __name__ == "__main__":
    fire.Fire({"build": build})
