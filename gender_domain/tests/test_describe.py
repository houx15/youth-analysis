"""
gender_domain.describe 的单元测试（表 1：样本与活动特征；表 2：原始性别差异）。

所有期望数字都在注释里写出推导过程：表 1 用等差数列构造样本，使
pandas 线性插值分位数可以手算到位；表 2 用一组用户逐个手工设计
n_retweets/n_expressive_posts/source_months 等原始计数，据此推出
source_entered/source_share/topical_share/source_month_share 四个
结果变量该取什么值——这些值不是先跑代码再抄下来的，是先算好写在这里，
再拿代码去对。少数依赖 stats_utils 内部公式（Newcombe/risk ratio 对数
尺度区间）的期望区间，直接调用 stats_utils 对应函数作为交叉验证的
"标准答案"，因为那些函数本身已经在 test_stats_utils.py 里单独验证过。
"""

import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import describe as de
from gender_domain import stats_utils as su


# ---------------------------------------------------------------------------
# 表 1：sample_activity_table
# ---------------------------------------------------------------------------

def _table1_fixture():
    """两个性别各 4 个用户，四个活动变量都是等差数列，方便手算分位数。

    pandas Series.quantile 默认线性插值（R-7 方法），n=4 时
    index = q*(n-1) = 3q；q=0.75 -> index=2.25，落在第 2、3 个元素
    （0-indexed）之间，frac=0.25，value = a[2] + 0.25*(a[3]-a[2])。
    这里所有数组都是等距的，代入即可手算，已用独立 pandas 调用核对
    （见任务报告）。
    """
    return pd.DataFrame({
        "user_id": ["m1", "m2", "m3", "m4", "f1", "f2", "f3", "f4"],
        "gender": ["m", "m", "m", "m", "f", "f", "f", "f"],
        "n_posts": [10, 20, 30, 40, 100, 200, 300, 400],
        "n_retweets": [1, 2, 3, 4, 10, 20, 30, 40],
        "n_active_days": [5, 10, 15, 20, 50, 100, 150, 200],
        "n_active_months": [1, 2, 3, 4, 5, 6, 7, 8],
    })


def test_sample_activity_table_has_one_row_per_gender_and_variable():
    out = de.sample_activity_table(_table1_fixture())
    # 2 个性别 x 4 个活动变量 = 8 行
    assert len(out) == 8
    assert set(out["gender"]) == {"m", "f"}
    assert set(out["variable"]) == {"n_posts", "n_retweets", "n_active_days", "n_active_months"}


def test_sample_activity_table_n_users_correct_per_gender():
    out = de.sample_activity_table(_table1_fixture())
    row = out[(out["gender"] == "m") & (out["variable"] == "n_posts")].iloc[0]
    assert row["n_users"] == 4


def test_sample_activity_table_n_posts_male_mean_median_percentiles():
    out = de.sample_activity_table(_table1_fixture())
    row = out[(out["gender"] == "m") & (out["variable"] == "n_posts")].iloc[0]
    # [10,20,30,40]：mean=median=25；p75=30+0.25*10=32.5；p90=30+0.7*10=37.0；
    # p95=30+0.85*10=38.5；p99=30+0.97*10=39.7
    assert row["mean"] == pytest.approx(25.0)
    assert row["median"] == pytest.approx(25.0)
    assert row["p75"] == pytest.approx(32.5)
    assert row["p90"] == pytest.approx(37.0)
    assert row["p95"] == pytest.approx(38.5)
    assert row["p99"] == pytest.approx(39.7)


def test_sample_activity_table_n_retweets_female_mean_median_percentiles():
    out = de.sample_activity_table(_table1_fixture())
    row = out[(out["gender"] == "f") & (out["variable"] == "n_retweets")].iloc[0]
    # [10,20,30,40]：同上比例的分位数
    assert row["mean"] == pytest.approx(25.0)
    assert row["median"] == pytest.approx(25.0)
    assert row["p75"] == pytest.approx(32.5)
    assert row["p90"] == pytest.approx(37.0)
    assert row["p95"] == pytest.approx(38.5)
    assert row["p99"] == pytest.approx(39.7)


def test_sample_activity_table_n_active_months_male_percentiles():
    out = de.sample_activity_table(_table1_fixture())
    row = out[(out["gender"] == "m") & (out["variable"] == "n_active_months")].iloc[0]
    # [1,2,3,4]：mean=median=2.5；p75=3+0.25=3.25；p90=3+0.7=3.7；
    # p95=3+0.85=3.85；p99=3+0.97=3.97
    assert row["mean"] == pytest.approx(2.5)
    assert row["median"] == pytest.approx(2.5)
    assert row["p75"] == pytest.approx(3.25)
    assert row["p90"] == pytest.approx(3.7)
    assert row["p95"] == pytest.approx(3.85)
    assert row["p99"] == pytest.approx(3.97)


# ---------------------------------------------------------------------------
# 表 2：raw_gender_gaps
# ---------------------------------------------------------------------------

def _table2_fixture():
    """6 个男性 + 6 个女性用户，只填 public 域需要的原始列，celebrity 域
    直接复制 public 域的取值（本测试不断言 celebrity 具体数字，只要求
    函数能在两个域上都跑通）。

    男性 (u1..u6) 手工设计：
        n_retweets            = [5, 2, 0, 4, 0, 3]
        public_source_entered = [T, T, F, F, F, T]
        public_source_count   = [3, 2, 0, 0, 0, 1]
        public_source_share   = [.6, 1.0, NaN, 0.0, NaN, 1/3]   (NaN: n_retweets=0)
        n_expressive_posts    = [10, 0, 5, 8, 0, 6]
        public_topical_posts  = [4, 0, 0, 2, 0, 6]
        public_topical_share  = [.4, NaN, 0.0, .25, NaN, 1.0]   (NaN: 无表达帖)
        active_months_panel   = [3, 2, 1, 4, 0, 2]
        public_source_months  = [2, 1, 0, 0, 0, 2]
        public_source_month_share = [2/3, .5, 0.0, 0.0, NaN, 1.0]  (NaN: 无活跃月)

    女性 (v1..v6) 同理：
        n_retweets            = [6, 3, 0, 2, 5, 0]
        public_source_entered = [F, F, F, T, F, F]
        public_source_count   = [0, 0, 0, 1, 0, 0]
        public_source_share   = [0.0, 0.0, NaN, .5, 0.0, NaN]
        n_expressive_posts    = [10, 5, 4, 0, 8, 6]
        public_topical_posts  = [1, 0, 1, 0, 0, 0]
        public_topical_share  = [.1, 0.0, .25, NaN, 0.0, 0.0]
        active_months_panel   = [3, 2, 1, 2, 0, 3]
        public_source_months  = [0, 0, 0, 1, 0, 0]
        public_source_month_share = [0.0, 0.0, 0.0, .5, NaN, 0.0]

    验证：source_share/topical_share/source_month_share 的 NaN 全部来自
    "分母为 0"（未转发 / 无表达帖 / 无活跃月），不是随意填的。
    """
    m_source_entered = [True, True, False, False, False, True]
    m_source_count = [3, 2, 0, 0, 0, 1]
    m_source_share = [0.6, 1.0, np.nan, 0.0, np.nan, 1 / 3]
    m_topical_posts = [4, 0, 0, 2, 0, 6]
    m_topical_share = [0.4, np.nan, 0.0, 0.25, np.nan, 1.0]
    m_source_months = [2, 1, 0, 0, 0, 2]
    m_source_month_share = [2 / 3, 0.5, 0.0, 0.0, np.nan, 1.0]

    f_source_entered = [False, False, False, True, False, False]
    f_source_count = [0, 0, 0, 1, 0, 0]
    f_source_share = [0.0, 0.0, np.nan, 0.5, 0.0, np.nan]
    f_topical_posts = [1, 0, 1, 0, 0, 0]
    f_topical_share = [0.1, 0.0, 0.25, np.nan, 0.0, 0.0]
    f_source_months = [0, 0, 0, 1, 0, 0]
    f_source_month_share = [0.0, 0.0, 0.0, 0.5, np.nan, 0.0]

    df = pd.DataFrame({
        "user_id": [f"m{i}" for i in range(1, 7)] + [f"f{i}" for i in range(1, 7)],
        "gender": ["m"] * 6 + ["f"] * 6,
        "n_retweets": [5, 2, 0, 4, 0, 3, 6, 3, 0, 2, 5, 0],
        "public_source_entered": m_source_entered + f_source_entered,
        "public_source_count": m_source_count + f_source_count,
        "public_source_share": m_source_share + f_source_share,
        "public_topical_posts": m_topical_posts + f_topical_posts,
        "public_topical_share": m_topical_share + f_topical_share,
        "public_source_months": m_source_months + f_source_months,
        "public_source_month_share": m_source_month_share + f_source_month_share,
    })
    for col in [
        "source_entered", "source_count", "source_share", "topical_posts",
        "topical_share", "source_months", "source_month_share",
    ]:
        df[f"celebrity_{col}"] = df[f"public_{col}"]
    return df


def _rows(out, outcome, domain, term, denominator=None):
    sub = out[(out["outcome"] == outcome) & (out["domain"] == domain) & (out["term"] == term)]
    if denominator is not None:
        sub = sub[sub["denominator"] == denominator]
    assert len(sub) == 1, f"期望恰好 1 行: outcome={outcome} term={term} denominator={denominator}, 实际 {len(sub)} 行"
    return sub.iloc[0]


def test_raw_gender_gaps_has_denominator_column():
    out = de.raw_gender_gaps(_table2_fixture())
    assert "denominator" in out.columns


def test_raw_gender_gaps_source_entered_two_denominators_differ():
    out = de.raw_gender_gaps(_table2_fixture())
    # all_same_gender_users：3/6=0.5 (男) vs 1/6=0.16667 (女)
    male_all = _rows(out, "source_entered", "public", "male", "all_same_gender_users")
    female_all = _rows(out, "source_entered", "public", "female", "all_same_gender_users")
    assert male_all["estimate"] == pytest.approx(0.5)
    assert female_all["estimate"] == pytest.approx(1 / 6)
    assert male_all["n_obs"] == 6
    assert female_all["n_obs"] == 6

    # retweeters_only：分母限定 n_retweets>0 的用户。男性 4 人 (u1,u2,u4,u6)
    # 中 3 人进入 -> 3/4=0.75；女性 4 人 (v1,v2,v4,v5) 中 1 人进入 -> 1/4=0.25
    male_rt = _rows(out, "source_entered", "public", "male", "retweeters_only")
    female_rt = _rows(out, "source_entered", "public", "female", "retweeters_only")
    assert male_rt["estimate"] == pytest.approx(0.75)
    assert female_rt["estimate"] == pytest.approx(0.25)
    assert male_rt["n_obs"] == 4
    assert female_rt["n_obs"] == 4
    assert male_rt["n_dropped"] == 2  # u3, u5 无转发
    assert female_rt["n_dropped"] == 2  # v3, v6 无转发

    # 两个分母产生的数值必须不同（否则"标注分母"这件事就没有意义）
    assert male_all["estimate"] != pytest.approx(male_rt["estimate"])
    assert female_all["estimate"] != pytest.approx(female_rt["estimate"])


def test_raw_gender_gaps_source_entered_gap_and_rr_match_stats_utils():
    out = de.raw_gender_gaps(_table2_fixture())
    gap = _rows(out, "source_entered", "public", "gap_pp", "all_same_gender_users")
    rr = _rows(out, "source_entered", "public", "risk_ratio", "all_same_gender_users")

    # 交叉验证：直接调用已单独测试过的 stats_utils 函数作为标准答案
    exp_diff, exp_low, exp_high = su.proportion_diff_ci(3, 6, 1, 6)
    assert gap["estimate"] == pytest.approx(exp_diff)
    assert gap["ci_low"] == pytest.approx(exp_low)
    assert gap["ci_high"] == pytest.approx(exp_high)

    exp_rr, exp_rr_low, exp_rr_high = su.risk_ratio_ci(3, 6, 1, 6)
    assert rr["estimate"] == pytest.approx(exp_rr)
    assert rr["estimate"] == pytest.approx(3.0)  # (3/6)/(1/6) = 3.0
    assert rr["ci_low"] == pytest.approx(exp_rr_low)
    assert rr["ci_high"] == pytest.approx(exp_rr_high)

    # p 值绝不能是唯一headline：本函数根本不产出 p 值列，效应量+区间即已足够
    assert "p_value" not in out.columns


def test_raw_gender_gaps_nan_excluded_from_topical_share_only_not_source_entered():
    out = de.raw_gender_gaps(_table2_fixture())
    # 男性 source_entered（all_same_gender_users）：6 人全部纳入，0 人丢弃
    se_male = _rows(out, "source_entered", "public", "male", "all_same_gender_users")
    assert se_male["n_obs"] == 6
    assert se_male["n_dropped"] == 0

    # 男性 topical_share：u2、u5 无表达帖 -> topical_share=NaN，从这一行剔除，
    # 但没有从 source_entered 那一行剔除（上面已验证 n_obs=6）
    ts_male = _rows(out, "topical_share", "public", "male")
    assert ts_male["n_obs"] == 4
    assert ts_male["n_dropped"] == 2
    assert ts_male["drop_reason"] == "no_expressive_posts"


def test_raw_gender_gaps_topical_share_mean_hand_computed():
    out = de.raw_gender_gaps(_table2_fixture())
    # 男性有效值 [.4, 0.0, .25, 1.0] (u1,u3,u4,u6)，mean = 1.65/4 = 0.4125
    ts_male = _rows(out, "topical_share", "public", "male")
    assert ts_male["estimate"] == pytest.approx(0.4125)
    assert ts_male["ci_low"] <= ts_male["estimate"] <= ts_male["ci_high"]

    # 女性有效值 [.1, 0.0, .25, 0.0, 0.0] (v1,v2,v3,v5,v6)，mean = 0.35/5 = 0.07
    ts_female = _rows(out, "topical_share", "public", "female")
    assert ts_female["estimate"] == pytest.approx(0.07)
    assert ts_female["n_obs"] == 5
    assert ts_female["n_dropped"] == 1  # v4 无表达帖


def test_raw_gender_gaps_source_share_mean_hand_computed():
    out = de.raw_gender_gaps(_table2_fixture())
    # 男性有效值（n_retweets>0）[.6, 1.0, 0.0, 1/3] (u1,u2,u4,u6)
    # sum = 1.9333..., mean = 0.483325
    ss_male = _rows(out, "source_share", "public", "male")
    assert ss_male["n_obs"] == 4
    assert ss_male["n_dropped"] == 2
    assert ss_male["drop_reason"] == "no_retweets"
    assert ss_male["estimate"] == pytest.approx((0.6 + 1.0 + 0.0 + 1 / 3) / 4)

    # 女性有效值 [0.0, 0.0, .5, 0.0] (v1,v2,v4,v5)，mean = 0.125
    ss_female = _rows(out, "source_share", "public", "female")
    assert ss_female["n_obs"] == 4
    assert ss_female["estimate"] == pytest.approx(0.125)


def test_raw_gender_gaps_source_month_share_mean_hand_computed():
    out = de.raw_gender_gaps(_table2_fixture())
    # 男性有效值（活跃月>0）[2/3, .5, 0.0, 0.0, 1.0] (u1,u2,u3,u4,u6)
    sms_male = _rows(out, "source_month_share", "public", "male")
    assert sms_male["n_obs"] == 5
    assert sms_male["n_dropped"] == 1  # u5 无活跃月
    assert sms_male["drop_reason"] == "no_active_months"
    expected_male = (2 / 3 + 0.5 + 0.0 + 0.0 + 1.0) / 5
    assert sms_male["estimate"] == pytest.approx(expected_male)

    # 女性有效值 [0.0, 0.0, 0.0, .5, 0.0] (v1,v2,v3,v4,v6)，mean = 0.1
    sms_female = _rows(out, "source_month_share", "public", "female")
    assert sms_female["n_obs"] == 5
    assert sms_female["estimate"] == pytest.approx(0.1)


def test_raw_gender_gaps_gap_pp_equals_male_minus_female_point_estimate():
    out = de.raw_gender_gaps(_table2_fixture())
    for outcome in ("source_share", "topical_share", "source_month_share"):
        male_row = _rows(out, outcome, "public", "male")
        female_row = _rows(out, outcome, "public", "female")
        gap_row = _rows(out, outcome, "public", "gap_pp")
        assert gap_row["estimate"] == pytest.approx(male_row["estimate"] - female_row["estimate"])
        assert gap_row["ci_low"] <= gap_row["estimate"] <= gap_row["ci_high"]


def test_raw_gender_gaps_top_share_computed_over_users_not_posts():
    out = de.raw_gender_gaps(_table2_fixture())
    # source_entered/source_share 共用 public_source_count 作为"行为量"：
    # 男性 [3,2,0,0,0,1]，sum=6，最大值=3 -> top1%(k=ceil(.01*6)=1)份额=3/6=0.5；
    # q=0.05 同样 k=1，份额也是 0.5
    top1_male = _rows(out, "source_entered", "public", "top1_share_male")
    top5_male = _rows(out, "source_entered", "public", "top5_share_male")
    assert top1_male["estimate"] == pytest.approx(0.5)
    assert top5_male["estimate"] == pytest.approx(0.5)

    # 女性 [0,0,0,1,0,0]，sum=1，最大值=1 -> 份额=1.0（该项行为完全集中在 1 人）
    top1_female = _rows(out, "source_entered", "public", "top1_share_female")
    assert top1_female["estimate"] == pytest.approx(1.0)

    # topical_share 用 public_topical_posts：男性 [4,0,0,2,0,6] sum=12 max=6
    # -> 0.5；女性 [1,0,1,0,0,0] sum=2 max=1 -> 0.5
    ts_top1_male = _rows(out, "topical_share", "public", "top1_share_male")
    ts_top1_female = _rows(out, "topical_share", "public", "top1_share_female")
    assert ts_top1_male["estimate"] == pytest.approx(0.5)
    assert ts_top1_female["estimate"] == pytest.approx(0.5)

    # source_month_share 用 public_source_months：男性 [2,1,0,0,0,2] sum=5 max=2
    # -> 0.4；女性 [0,0,0,1,0,0] sum=1 max=1 -> 1.0
    sms_top1_male = _rows(out, "source_month_share", "public", "top1_share_male")
    sms_top1_female = _rows(out, "source_month_share", "public", "top1_share_female")
    assert sms_top1_male["estimate"] == pytest.approx(0.4)
    assert sms_top1_female["estimate"] == pytest.approx(1.0)


def test_raw_gender_gaps_runs_on_both_domains():
    out = de.raw_gender_gaps(_table2_fixture())
    assert set(out["domain"]) == {"public", "celebrity"}
    # celebrity 域的数据是 public 域的复制，二者结果应完全一致
    male_public = _rows(out, "source_entered", "public", "male", "all_same_gender_users")
    male_celebrity = _rows(out, "source_entered", "celebrity", "male", "all_same_gender_users")
    assert male_public["estimate"] == pytest.approx(male_celebrity["estimate"])


def test_raw_gender_gaps_uses_tidy_result_schema_columns():
    out = de.raw_gender_gaps(_table2_fixture())
    for col in su.RESULT_SCHEMA:
        assert col in out.columns


def test_raw_gender_gaps_note_labels_interval_method_per_row():
    """表 2 必须能自证每一行的区间方法，不能让读者靠猜——见协调者第 2 条裁定。

    二值结果变量 (source_entered) 用二项区间（Wilson/Newcombe/对数尺度
    risk ratio），比例型结果变量 (source_share/topical_share/
    source_month_share) 报告的是用户层面比例的均值，抽样分布不是二项，
    改用两独立样本 bootstrap。note 列必须写清楚具体是哪一种，且两类
    结果变量的取值必须不同，否则读者会默认全表都是 Newcombe。
    """
    out = de.raw_gender_gaps(_table2_fixture())

    male_binary = _rows(out, "source_entered", "public", "male", "all_same_gender_users")
    female_binary = _rows(out, "source_entered", "public", "female", "all_same_gender_users")
    gap_binary = _rows(out, "source_entered", "public", "gap_pp", "all_same_gender_users")
    rr_binary = _rows(out, "source_entered", "public", "risk_ratio", "all_same_gender_users")
    assert male_binary["note"] == "wilson"
    assert female_binary["note"] == "wilson"
    assert gap_binary["note"] == "newcombe"
    assert rr_binary["note"] == "log_scale"

    male_prop = _rows(out, "topical_share", "public", "male")
    female_prop = _rows(out, "topical_share", "public", "female")
    gap_prop = _rows(out, "topical_share", "public", "gap_pp")
    rr_prop = _rows(out, "topical_share", "public", "risk_ratio")
    assert male_prop["note"] == "bootstrap"
    assert female_prop["note"] == "bootstrap"
    assert gap_prop["note"] == "bootstrap_two_sample"
    assert rr_prop["note"] == "bootstrap_two_sample"

    # 核心断言：二值结果变量与比例型结果变量的方法标签必须不同
    assert male_binary["note"] != male_prop["note"]
    assert gap_binary["note"] != gap_prop["note"]
    assert rr_binary["note"] != rr_prop["note"]


# ---------------------------------------------------------------------------
# 表 2 的唯一键：(outcome, domain, model, term) 逐行唯一
# ---------------------------------------------------------------------------

def test_table2_result_keys_are_unique():
    """整张表 2 里 (outcome, domain, model, term) 必须逐行唯一

    这是整套结果表共用的不变量：models_interaction 里用断言守着，
    三个测试模块各测一遍，而绘图层在 15 个地方按这个四元组取行再
    .iloc[0]。表 2 此前有 16 行撞键——source_entered 的两个分母口径
    都写 model="raw"，只靠额外的 denominator 列区分，任何按四元组
    join 的下游都会静默拿到其中任意一行。
    """
    out = de.raw_gender_gaps(_table2_fixture())
    key = ["outcome", "domain", "model", "term"]
    duplicated = out[out.duplicated(subset=key, keep=False)]
    assert duplicated.empty, (
        f"表 2 里有 {len(duplicated)} 行重复键，例如\n"
        f"{duplicated.sort_values(key).head(8).to_string(index=False)}"
    )


def test_source_entered_denominators_are_separate_model_variants():
    """两个分母口径折进 model 列，denominator 列同时保留给人读

    折进 model 的做法与 models_combination（"M1/share_ge_0.1"）、
    models_temporal（"record_level/male/max=720h"）完全一致。
    """
    out = de.raw_gender_gaps(_table2_fixture())
    entered = out[out["outcome"] == "source_entered"]
    # 分母口径的行落在两个 model 变体上；头部集中度行分母不适用，仍是 raw
    by_model = dict(zip(entered["model"], entered["denominator"]))
    assert by_model["raw/all_users"] == de.DENOM_ALL_USERS
    assert by_model["raw/retweeters_only"] == de.DENOM_RETWEETERS
    # 同一个 (domain, term) 在两个变体下各有一行，且互不撞键
    male_all = _rows(out, "source_entered", "public", "male", de.DENOM_ALL_USERS)
    male_rt = _rows(out, "source_entered", "public", "male", de.DENOM_RETWEETERS)
    assert male_all["model"] != male_rt["model"]
    # 分母不同，估计值也确实不同（否则这条测试区分不出任何东西）
    assert male_all["estimate"] != pytest.approx(male_rt["estimate"])


def test_table2_uses_only_the_declared_scale_vocabulary():
    """表 2 里出现的每一个 scale 都必须在 stats_utils 的封闭词表里

    尤其是 risk ratio 那一行：它原来叫含糊的 "ratio"，与 odds_ratio /
    irr 三个名字说同一种画法（对数轴 + 参照线 1）却互不相认。
    """
    out = de.raw_gender_gaps(_table2_fixture())
    assert set(out["scale"].dropna()) <= set(su.RESULT_SCALES)
    rr = _rows(out, "source_entered", "public", "risk_ratio", de.DENOM_ALL_USERS)
    assert rr["scale"] == "risk_ratio"
    assert rr["scale"] in su.RATIO_SCALES


# ---------------------------------------------------------------------------
# CLI：落盘的表也要满足同一条不变量，并带上运行标识
# ---------------------------------------------------------------------------

def test_build_writes_unique_keys_and_stamps_the_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(de.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv(de.config.RUN_ID_ENV, "run-under-test")
    # 表 2 的 fixture 只带结果变量所需的原始计数；表 1 还要四个活动变量，
    # 这里补齐（取值本身与本测试无关，本测试只看键唯一性与运行标识）
    frame = _table2_fixture()
    for var in de.ACTIVITY_VARS:
        if var not in frame.columns:
            frame[var] = np.arange(len(frame), dtype=float) + 1.0
    frame.to_parquet(tmp_path / "user_domain_2020.parquet",
                     engine="pyarrow", index=False)

    table1_path, table2_path = de.build(year=2020)
    table2 = pd.read_parquet(table2_path,
                             columns=list(su.RESULT_SCHEMA) + ["denominator"])
    key = ["outcome", "domain", "model", "term"]
    assert not table2.duplicated(subset=key).any()

    stamps = de.config.read_run_stamps(str(tmp_path / "results"))
    for path in (table1_path, table2_path):
        assert stamps[os.path.basename(path)]["run_id"] == "run-under-test"
