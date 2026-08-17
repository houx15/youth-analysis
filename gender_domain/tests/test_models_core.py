"""
gender_domain.models_core 的单元测试（§6.2–6.5 进入/强度/占比/持续性模型）。

测试设计原则（这类任务最容易出错的地方就在这里，所以写在最前面）：

1. **模拟数据的"真值"是先算好的，不是跑完代码抄下来的。**
   所有模拟都把性别效应直接写死在生成过程里（例如男性进入概率 0.60、
   女性 0.30），期望的平均边际效应（AME）就是 0.30 这个已知量，测试断言
   的是"能不能把这个已知量估回来"，而不是"代码有没有跑通"。

2. **M0 层可以断言到机器精度，不是近似。**
   M0 只有性别一个二值自变量，logit / 分数 logit 在这种饱和模型下的
   最大似然解使得每一组的拟合值恰好等于该组的样本均值，因此
   AME = (男性组样本均值 - 女性组样本均值) 是一个代数恒等式，不是
   大样本近似。这条性质让 M0 的测试可以用 1e-8 量级的容差，一旦有人
   把 AME 悄悄换成"在协变量均值处的边际效应"(MEM) 或者别的近似写法，
   这个断言立刻会挂。加了控制变量的 M1/M2 没有这个恒等式，只能用大样本
   容差（模拟里控制变量与性别独立，所以 AME 仍应接近原始差值）。

3. **必须有一个"报系数就会挂"的测试。**
   test_entry_ame_is_not_the_logit_coefficient 里男女进入概率取 0.90 /
   0.50，真值 AME = 0.40，而 logit 系数是 log(9/1) - log(1/1) ≈ 2.197。
   两者差了 5 倍以上，断言 |estimate - 0.40| < 0.03 的同时还断言
   estimate 落在 [-1, 1] 内——概率尺度的差值天然有界，log-odds 尺度
   没有，任何"直接报回归系数"的实现都过不了。

4. **失败必须留痕。** 不收敛的模型要产出一行 NaN + note，而不是抛异常、
   也不是少一行——少一行在论文里读起来像"这个假设没人做过"。
"""

import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import models_core as mc
from gender_domain import stats_utils as su


# ---------------------------------------------------------------------------
# 模拟数据构造
# ---------------------------------------------------------------------------

def _simulate_users(n_male=3000, n_female=3000, p_male=0.60, p_female=0.30,
                    seed=0, profile_complete_frac=1.0, domain="public"):
    """构造一份最小可用的表 C 模拟数据，性别效应写死在生成过程里

    进入概率：男性 p_male、女性 p_female，与其它协变量独立——这样
    M1/M2 加进来的控制变量在总体上不改变性别的边际效应，M0/M1/M2 三层
    的 AME 真值都是 (p_male - p_female)。
    """
    rng = np.random.default_rng(seed)
    n = n_male + n_female
    gender = np.array(["m"] * n_male + ["f"] * n_female)
    p = np.where(gender == "m", p_male, p_female)
    entered = (rng.random(n) < p)

    n_posts = rng.integers(1, 200, n)
    n_retweets = rng.integers(1, 100, n)
    n_expressive = rng.integers(1, 80, n)
    n_active_days = rng.integers(1, 300, n)
    n_active_months = rng.integers(1, 13, n)

    source_count = np.where(entered, rng.integers(1, 20, n), 0)
    source_months = np.where(entered, rng.integers(1, 6, n), 0)
    source_months = np.minimum(source_months, n_active_months)

    complete = rng.random(n) < profile_complete_frac

    df = pd.DataFrame({
        "user_id": [str(i) for i in range(n)],
        "gender": gender,
        "n_posts": n_posts,
        "n_retweets": n_retweets,
        "n_expressive_posts": n_expressive,
        "n_active_days": n_active_days,
        "n_active_months": n_active_months,
        "n_active_months_panel": n_active_months,
        "region": rng.choice(["north", "south", "west"], n),
        "verified_flag": rng.integers(0, 2, n).astype(float),
        "log_fans": rng.normal(5, 1, n),
        "log_friends": rng.normal(4, 1, n),
        "profile_complete": complete,
    })
    for d in ("public", "celebrity"):
        if d == domain:
            df[f"{d}_source_entered"] = entered
            df[f"{d}_source_count"] = source_count.astype(int)
            df[f"{d}_source_months"] = source_months.astype(int)
        else:
            other = rng.random(n) < 0.4
            df[f"{d}_source_entered"] = other
            df[f"{d}_source_count"] = np.where(other, rng.integers(1, 10, n), 0)
            df[f"{d}_source_months"] = np.where(other, 1, 0)
        # 与 build_user_tables.combine_user_table 一致：占比上限截到 1。
        # 不截的话模拟数据会造出 >1 的占比，而分数 logit 的前提就是 [0,1]
        # （stats_utils.check_proportion_range 现在会直接报错）
        df[f"{d}_source_share"] = (
            df[f"{d}_source_count"] / df["n_retweets"]
        ).clip(upper=1.0)
        df[f"{d}_topical_share"] = rng.random(n) * 0.5
        df[f"{d}_source_month_share"] = (
            df[f"{d}_source_months"] / df["n_active_months_panel"]
        )
    return df


def _raw_gap(df, column):
    """男性组均值 - 女性组均值（只用有定义的观测），即 M0 的 AME 真值"""
    valid = df[df[column].notna()]
    return (
        valid.loc[valid["gender"] == "m", column].astype(float).mean()
        - valid.loc[valid["gender"] == "f", column].astype(float).mean()
    )


def _row(frame, model, term=mc.TERM_AME):
    sub = frame[(frame["model"] == model) & (frame["term"] == term)]
    assert len(sub) == 1, f"期望恰好一行 {model}/{term}，实际 {len(sub)} 行"
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# 派生变量（log_posts / log_retweets / male）
# ---------------------------------------------------------------------------

def test_add_log_activity_uses_log1p():
    df = pd.DataFrame({"n_posts": [0, 1, 9], "n_retweets": [0, 3, 99]})
    out = mc.add_log_activity(df)
    # log1p 而不是 log：n_posts=0 必须得到 0.0，不能是 -inf
    assert out["log_posts"].tolist() == pytest.approx([0.0, np.log(2), np.log(10)])
    assert out["log_retweets"].tolist() == pytest.approx([0.0, np.log(4), np.log(100)])
    # 不改动入参
    assert "log_posts" not in df.columns


def test_add_male_indicator_is_zero_one():
    df = pd.DataFrame({"gender": ["m", "f", "m"]})
    out = mc.add_male_indicator(df)
    assert out[mc.FOCAL_COLUMN].tolist() == [1, 0, 1]


def test_prepare_model_frame_drops_unknown_gender():
    df = _simulate_users(n_male=20, n_female=20, seed=1)
    df.loc[df.index[:3], "gender"] = None
    frame = mc.prepare_model_frame(df)
    assert len(frame) == 37
    assert set(frame["gender"]) == {"m", "f"}


# ---------------------------------------------------------------------------
# §6.2 进入模型
# ---------------------------------------------------------------------------

def test_entry_frame_matches_shared_result_schema():
    df = _simulate_users(n_male=300, n_female=300, seed=2)
    out = mc.fit_entry_models(df, "public")
    assert list(out.columns) == list(su.RESULT_SCHEMA)


def test_entry_reports_all_three_layers():
    df = _simulate_users(n_male=300, n_female=300, seed=3)
    out = mc.fit_entry_models(df, "public")
    assert sorted(out["model"].unique()) == ["M0", "M1", "M2"]


def test_entry_m0_ame_equals_raw_probability_difference_exactly():
    """M0 饱和 logit 的 AME 与原始比例差是代数恒等式，可以断到机器精度"""
    df = _simulate_users(n_male=3000, n_female=3000, p_male=0.60, p_female=0.30, seed=4)
    out = mc.fit_entry_models(df, "public")
    row = _row(out, "M0")
    truth = _raw_gap(df.assign(y=df["public_source_entered"].astype(float)), "y")
    assert row["estimate"] == pytest.approx(truth, abs=1e-8)
    assert row["scale"] == "probability"
    # 生成过程写死的真值是 0.30，模拟误差在 n=6000 下约 ±0.02
    assert row["estimate"] == pytest.approx(0.30, abs=0.03)
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert np.isfinite(row["se"]) and row["se"] > 0


def test_entry_ame_is_not_the_logit_coefficient():
    """报系数而不是边际效应的实现必须在这里挂掉

    男 0.90 / 女 0.50：AME 真值 0.40，而 logit 系数 = logit(0.9) - logit(0.5)
    = log(9) ≈ 2.197。概率尺度的差值必然落在 [-1, 1] 内，log-odds 尺度不会。
    """
    df = _simulate_users(n_male=4000, n_female=4000, p_male=0.90, p_female=0.50, seed=5)
    row = _row(mc.fit_entry_models(df, "public"), "M0")
    assert -1.0 <= row["estimate"] <= 1.0
    assert row["estimate"] == pytest.approx(0.40, abs=0.03)
    # 同一份模型的发生比行才应该在 ~9 附近
    or_row = _row(mc.fit_entry_models(df, "public"), "M0", term=mc.TERM_ODDS_RATIO)
    assert or_row["estimate"] == pytest.approx(9.0, rel=0.15)
    assert or_row["scale"] == "odds_ratio"


def test_entry_m1_and_m2_recover_the_same_known_effect():
    """控制变量与性别独立时，加控制不应改变 AME 真值"""
    df = _simulate_users(n_male=4000, n_female=4000, p_male=0.60, p_female=0.30, seed=6)
    out = mc.fit_entry_models(df, "public")
    for model in ("M0", "M1", "M2"):
        assert _row(out, model)["estimate"] == pytest.approx(0.30, abs=0.03), model


def test_entry_odds_ratio_accompanies_every_layer():
    df = _simulate_users(n_male=500, n_female=500, seed=7)
    out = mc.fit_entry_models(df, "public")
    or_rows = out[out["term"] == mc.TERM_ODDS_RATIO]
    assert sorted(or_rows["model"].tolist()) == ["M0", "M1", "M2"]
    assert (or_rows["estimate"] > 0).all()


def test_m2_fits_fewer_observations_and_records_the_attrition():
    """M2 只用画像完整的用户，M0/M1 保持全样本；差额必须写进 n_dropped"""
    df = _simulate_users(n_male=1500, n_female=1500, seed=8, profile_complete_frac=0.6)
    n_complete = int(df["profile_complete"].sum())
    out = mc.fit_entry_models(df, "public")

    m1 = _row(out, "M1")
    m2 = _row(out, "M2")
    assert m1["n_obs"] == len(df)
    assert m1["n_dropped"] == 0
    assert m2["n_obs"] == n_complete
    assert m2["n_obs"] < m1["n_obs"]
    assert m2["n_dropped"] == len(df) - n_complete
    # drop_reason 是带计数的顺序归因（与 §7.1/§7.2 的行同一种写法）：
    # 只写原因名的话，读者拿不到"这条规则本身丢了多少人"
    assert m2["drop_reason"] == "incomplete_profile={}".format(len(df) - n_complete)


def test_m2_also_requires_region_because_region_is_an_m2_control():
    """region 是 M2 的控制变量，缺 region 同样进不了 M2，一并记为画像不完整"""
    df = _simulate_users(n_male=500, n_female=500, seed=20)
    df.loc[df.index[:80], "region"] = None
    out = mc.fit_entry_models(df, "public")
    m2 = _row(out, "M2")
    assert m2["n_obs"] == len(df) - 80
    assert m2["drop_reason"] == "incomplete_profile=80"
    assert _row(out, "M1")["n_obs"] == len(df)


def test_missing_profile_columns_degrade_only_m2():
    """表 C 没跑过 profile_join 时，M2 整层留痕失败，M0/M1 照常给结果"""
    df = _simulate_users(n_male=400, n_female=400, seed=21)
    df = df.drop(columns=["profile_complete", "verified_flag", "log_fans", "log_friends"])
    out = mc.fit_entry_models(df, "public")
    m2 = _row(out, "M2")
    assert np.isnan(m2["estimate"])
    assert m2["n_obs"] == 0
    assert m2["drop_reason"] == "incomplete_profile={}".format(len(df))
    assert isinstance(m2["note"], str) and m2["note"]
    assert np.isfinite(_row(out, "M1")["estimate"])


def test_n_obs_is_the_rows_actually_fitted_not_the_rows_we_selected():
    """patsy 会静默丢掉协变量为 NaN 的行，n_obs 必须跟着真实拟合行数走

    上游 profile_join 目前保证 profile_complete 蕴含三项控制变量非缺失，
    所以真实数据上不会出现这种情况；但结果表的数字不该依赖另一个模块的
    内部约定才正确。这里刻意造出"画像标记完整、log_fans 却是 NaN"的
    500 个用户，断言 M2 行报的是 3500 而不是 4000，并且
    n_obs + n_dropped 始终等于输入用户数。
    """
    df = _simulate_users(n_male=2000, n_female=2000, seed=22)
    df.loc[df.index[:500], "log_fans"] = np.nan          # profile_complete 仍为 True
    out = mc.fit_entry_models(df, "public")

    m2 = _row(out, "M2")
    assert m2["n_obs"] == len(df) - 500
    assert m2["n_dropped"] == 500
    assert "missing_covariate" in m2["drop_reason"]
    for _, row in out.iterrows():
        assert row["n_obs"] + row["n_dropped"] == len(df), row.to_dict()


def test_fractional_logit_separation_is_flagged_in_the_note():
    """GLM 的 IRLS 在（准）完全分离时照样自称收敛，必须在 note 里点名

    进入模型用 Logit 就是为了避开这一点，但占比/持续性只能用 GLM，
    所以补了系数量级 / 拟合值贴边两道便宜的事后检查。这里让 source_share
    被 n_posts 完全决定（0 或 1），断言这一行仍然产出（不静默丢弃），
    但 note 里带上 possible_separation。
    """
    df = _simulate_users(n_male=1000, n_female=1000, seed=23)
    # 准分离：占比几乎完全由 log_posts 决定，logit 尺度上的真实系数极大，
    # 拟合值大批贴在 0/1 边界，但标准误仍然有限——正是那种"看起来干净"
    # 的危险情形（完全分离反而会被 non_finite_se 直接拦下）
    logp = np.log1p(df["n_posts"].astype(float))
    df["public_source_share"] = 1 / (1 + np.exp(-(-40.0 + 10.0 * logp)))
    out = mc.fit_share_models(df, "public", "source_share")
    m1 = _row(out, "M1")
    assert np.isfinite(m1["estimate"]), "告警不应该把这一行删掉，估计要照常给出"
    assert isinstance(m1["note"], str) and "possible_separation" in m1["note"]


def test_design_width_counts_categorical_dummies():
    """参数个数必须按哑变量展开后算，否则 M2 会少算 30 个参数"""
    sample = pd.DataFrame({
        "male": [0, 1] * 5,
        "log_posts": np.arange(10, dtype=float),
        "region": ["a", "b", "c", "d", "e"] * 2,
    })
    # 截距 1 + male 1 + log_posts 1 + region 5 个水平取 4 个哑变量 = 7
    assert mc._design_width(["male", "log_posts", "C(region)"], sample) == 7


def test_ztnb_retry_chain_has_no_nelder_mead():
    """Nelder–Mead 救不回 M1/M2，而且它的收敛标志不是梯度检验"""
    assert "nm" not in mc.ZTNB_METHODS
    assert mc.ZTNB_METHODS[0] == "newton"


def test_ztnb_irr_note_says_it_is_the_latent_untruncated_mean():
    """IRR 是潜在未截断均值之比，不是"参与者条件均值"之比，结果表要自证"""
    df = _simulate_users(n_male=800, n_female=800, seed=24)
    row = _row(mc.fit_intensity_models(df, "public"), "M0", term=mc.TERM_IRR)
    assert mc.NOTE_IRR_LATENT in row["note"]


def test_non_converging_model_emits_nan_row_with_note():
    """完全分离的 M1：必须留下一行 NaN + note，不抛异常、更不能少这一行"""
    df = _simulate_users(n_male=400, n_female=400, seed=9)
    # 让进入与否被 n_posts 完全决定（完全分离），M0 仍然正常可估
    df["public_source_entered"] = df["n_posts"] > 100

    out = mc.fit_entry_models(df, "public")
    assert sorted(out["model"].unique()) == ["M0", "M1", "M2"]

    m1 = _row(out, "M1")
    assert np.isnan(m1["estimate"])
    assert np.isnan(m1["ci_low"]) and np.isnan(m1["ci_high"])
    assert isinstance(m1["note"], str) and m1["note"]
    # M0 不受影响，仍然给出真实估计
    assert np.isfinite(_row(out, "M0")["estimate"])


def test_no_gender_variation_is_reported_not_crashed():
    df = _simulate_users(n_male=200, n_female=0, seed=10)
    out = mc.fit_entry_models(df, "public")
    row = _row(out, "M0")
    assert np.isnan(row["estimate"])
    assert "gender" in row["note"]


# ---------------------------------------------------------------------------
# §6.3 强度模型（两部分 / hurdle）
# ---------------------------------------------------------------------------

def test_intensity_has_both_parts_and_the_log1p_robustness_row():
    df = _simulate_users(n_male=800, n_female=800, seed=11)
    out = mc.fit_intensity_models(df, "public")
    assert list(out.columns) == list(su.RESULT_SCHEMA)

    part1 = out[out["outcome"] == "source_entered"]
    assert sorted(part1["model"].unique()) == ["M0", "M1", "M2"]
    assert part1["note"].str.contains(mc.NOTE_PART1).all()

    irr = out[out["term"] == mc.TERM_IRR]
    assert sorted(irr["model"].unique()) == ["M0", "M1", "M2"]
    assert (irr["scale"] == "irr").all()

    ols = out[out["note"] == mc.NOTE_LOG1P_OLS]
    assert sorted(ols["model"].unique()) == ["M0", "M1", "M2"]


def test_intensity_part_two_uses_only_participants():
    df = _simulate_users(n_male=800, n_female=800, seed=12)
    n_participants = int((df["public_source_count"] > 0).sum())
    out = mc.fit_intensity_models(df, "public")
    row = _row(out, "M0", term=mc.TERM_IRR)
    assert row["n_obs"] == n_participants
    assert row["n_dropped"] == len(df) - n_participants
    assert "no_source_retweets" in row["drop_reason"]


def test_ztnb_recovers_a_known_incidence_rate_ratio():
    """零截断负二项：男性均值是女性的 2 倍，IRR 应回到 2 附近"""
    from scipy.stats import nbinom

    rng = np.random.default_rng(13)
    n_each = 5000
    gender = np.array(["m"] * n_each + ["f"] * n_each)
    mu = np.where(gender == "m", 6.0, 3.0)
    alpha = 0.5
    size = 1 / alpha
    prob = size / (size + mu)
    counts = nbinom.rvs(size, prob, random_state=13)

    n = len(gender)
    df = pd.DataFrame({
        "user_id": [str(i) for i in range(n)],
        "gender": gender,
        "n_posts": rng.integers(1, 200, n),
        "n_retweets": np.maximum(counts, 1) + rng.integers(0, 50, n),
        "n_expressive_posts": rng.integers(1, 50, n),
        "n_active_days": rng.integers(1, 300, n),
        "n_active_months": rng.integers(1, 13, n),
        "n_active_months_panel": rng.integers(1, 13, n),
        "region": rng.choice(["north", "south"], n),
        "verified_flag": rng.integers(0, 2, n).astype(float),
        "log_fans": rng.normal(5, 1, n),
        "log_friends": rng.normal(4, 1, n),
        "profile_complete": True,
    })
    for d in ("public", "celebrity"):
        c = counts if d == "public" else np.zeros(n, dtype=int)
        df[f"{d}_source_count"] = c.astype(int)
        df[f"{d}_source_entered"] = c > 0
        df[f"{d}_source_months"] = np.minimum(c, df["n_active_months"])
        # 与 build_user_tables.combine_user_table 一致：占比上限截到 1。
        # 不截的话模拟数据会造出 >1 的占比，而分数 logit 的前提就是 [0,1]
        # （stats_utils.check_proportion_range 现在会直接报错）
        df[f"{d}_source_share"] = (
            df[f"{d}_source_count"] / df["n_retweets"]
        ).clip(upper=1.0)
        df[f"{d}_topical_share"] = rng.random(n) * 0.4
        df[f"{d}_source_month_share"] = (
            df[f"{d}_source_months"] / df["n_active_months_panel"]
        )

    row = _row(mc.fit_intensity_models(df, "public"), "M0", term=mc.TERM_IRR)
    assert row["estimate"] == pytest.approx(2.0, rel=0.10)
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["ci_low"] > 1.0


# ---------------------------------------------------------------------------
# §6.3 占比模型（分数 logit）
# ---------------------------------------------------------------------------

def test_share_m0_ame_equals_mean_difference_exactly():
    df = _simulate_users(n_male=2000, n_female=2000, seed=14)
    out = mc.fit_share_models(df, "public", "source_share")
    assert list(out.columns) == list(su.RESULT_SCHEMA)
    row = _row(out, "M0")
    truth = _raw_gap(df, "public_source_share")
    assert row["estimate"] == pytest.approx(truth, abs=1e-8)
    assert row["scale"] == "proportion"


def test_share_drops_users_with_zero_denominator():
    """NaN 不是 0：没有转发的用户 source_share 未定义，必须剔除并记账"""
    df = _simulate_users(n_male=600, n_female=600, seed=15)
    df.loc[df.index[:100], "public_source_share"] = np.nan
    out = mc.fit_share_models(df, "public", "source_share")
    row = _row(out, "M0")
    assert row["n_obs"] == len(df) - 100
    assert row["n_dropped"] == 100
    assert "no_retweets" in row["drop_reason"]

    m2 = _row(out, "M2")
    assert "no_retweets" in m2["drop_reason"]


def test_topical_share_uses_its_own_drop_reason():
    df = _simulate_users(n_male=400, n_female=400, seed=16)
    df.loc[df.index[:50], "public_topical_share"] = np.nan
    row = _row(mc.fit_share_models(df, "public", "topical_share"), "M0")
    assert row["outcome"] == "topical_share"
    assert "no_expressive_posts" in row["drop_reason"]
    assert row["n_dropped"] == 50


# ---------------------------------------------------------------------------
# §6.4 持续性模型
# ---------------------------------------------------------------------------

def test_persistence_m0_ame_equals_month_weighted_mean_difference():
    """M0 层：拟合是月加权的，而 M0 的拟合值在性别内是常数，于是逐用户
    平均与逐月平均恰好相同，AME 等于月加权均值差——这条恒等式能钉住
    "AME 确实来自模型的反事实预测"，但**区分不了**报告口径是按用户平均
    还是按月平均（见下一个测试）。
    """
    df = _simulate_users(n_male=2000, n_female=2000, seed=17)
    out = mc.fit_persistence_models(df, "public")
    assert list(out.columns) == list(su.RESULT_SCHEMA)
    row = _row(out, "M0")

    w = df["n_active_months_panel"].astype(float).to_numpy()
    y = df["public_source_month_share"].astype(float).to_numpy()
    male = (df["gender"] == "m").to_numpy()
    truth = (
        np.average(y[male], weights=w[male]) - np.average(y[~male], weights=w[~male])
    )
    assert row["estimate"] == pytest.approx(truth, abs=1e-8)
    assert row["outcome"] == "source_month_share"
    assert row["scale"] == "proportion"


def test_persistence_ame_averages_over_users_not_over_months():
    """M1 层：钉住"报告的 AME 在用户上等权平均"这个估计量选择

    构造一份活跃月数与协变量强相关的数据，使得逐用户等权平均与以活跃
    月数为权重的平均**明显不同**；断言输出等于前者、且与后者相差远超
    容差。月加权拟合（var_weights）只影响估计效率，不改变报告单位——
    这一点在 fit_persistence_models 的 docstring 里写明，这里用测试锁死。
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(31)
    n_each = 1500
    n = 2 * n_each
    gender = np.array(["m"] * n_each + ["f"] * n_each)
    male = (gender == "m").astype(int)
    # 活跃月数与"参与倾向"强相关：活跃月数多的用户比例接近 0.5（边际
    # 效应最大），活跃月数少的用户比例贴近 0（边际效应很小）
    months = rng.integers(1, 13, n)
    latent = -4.0 + 0.45 * months + 0.9 * male
    share = 1 / (1 + np.exp(-latent))
    share = np.clip(share + rng.normal(0, 0.02, n), 0.0, 1.0)

    df = pd.DataFrame({
        "user_id": [str(i) for i in range(n)],
        "gender": gender,
        "n_posts": (months * 20 + rng.integers(1, 200, n)),
        "n_retweets": rng.integers(1, 100, n),
        "n_expressive_posts": rng.integers(1, 50, n),
        # 活跃天数不能恰好是活跃月数的整数倍，否则设计矩阵奇异
        "n_active_days": months * 20 + rng.integers(1, 20, n),
        "n_active_months": months,
        "n_active_months_panel": months,
        "region": rng.choice(["north", "south"], n),
        "verified_flag": rng.integers(0, 2, n).astype(float),
        "log_fans": rng.normal(5, 1, n),
        "log_friends": rng.normal(4, 1, n),
        "profile_complete": True,
    })
    for d in ("public", "celebrity"):
        df[f"{d}_source_count"] = rng.integers(0, 10, n)
        df[f"{d}_source_entered"] = df[f"{d}_source_count"] > 0
        df[f"{d}_source_months"] = np.minimum(
            np.round(share * months).astype(int), months
        )
        # 与 build_user_tables.combine_user_table 一致：占比上限截到 1。
        # 不截的话模拟数据会造出 >1 的占比，而分数 logit 的前提就是 [0,1]
        # （stats_utils.check_proportion_range 现在会直接报错）
        df[f"{d}_source_share"] = (
            df[f"{d}_source_count"] / df["n_retweets"]
        ).clip(upper=1.0)
        df[f"{d}_topical_share"] = rng.random(n) * 0.4
        df[f"{d}_source_month_share"] = share

    row = _row(mc.fit_persistence_models(df, "public"), "M1")

    # 独立复刻同一个模型，自己算两种平均，看输出对上的是哪一种
    frame = mc.prepare_model_frame(df)
    terms, _ = mc._formula_terms(mc.M1, frame)
    fitted = smf.glm(
        "public_source_month_share ~ " + " + ".join(terms), data=frame,
        family=sm.families.Binomial(),
        var_weights=frame["n_active_months_panel"].astype(float).to_numpy(),
    ).fit(cov_type="HC1")
    d1 = fitted.predict(frame.assign(**{mc.FOCAL_COLUMN: 1})).to_numpy()
    d0 = fitted.predict(frame.assign(**{mc.FOCAL_COLUMN: 0})).to_numpy()
    per_user_diff = d1 - d0
    over_users = float(np.mean(per_user_diff))
    over_months = float(np.average(
        per_user_diff, weights=frame["n_active_months_panel"].astype(float)
    ))

    # fixture 必须真的能区分两种口径，否则这个测试什么也没钉住
    assert abs(over_users - over_months) > 1e-3, (over_users, over_months)
    assert row["estimate"] == pytest.approx(over_users, abs=1e-8)
    assert abs(row["estimate"] - over_months) > 1e-3
    assert "ame_averaged_over_users" in row["note"]
    assert "months_weighted_fit" in row["note"]


# ---------------------------------------------------------------------------
# 不确定性：本模块所有模型都能拿到 cov_params，AME 的 se 不允许是 NaN
# ---------------------------------------------------------------------------

def test_every_successful_estimate_carries_a_finite_interval():
    """没有区间的估计不许出现在结果表里（除非那一行本来就是失败留痕）

    se 只对概率/比例尺度的 AME 断言非空：发生比/IRR 行的区间是对数尺度
    正态近似换算回来的，天然不对称，stats_utils 模块文档第 5 条明确
    禁止为这类区间反推一个 se，所以那些行的 se 就是 NaN，这是刻意的。
    """
    df = _simulate_users(n_male=1200, n_female=1200, seed=18)
    frames = [
        mc.fit_entry_models(df, "public"),
        mc.fit_intensity_models(df, "public"),
        mc.fit_share_models(df, "public", "source_share"),
        mc.fit_share_models(df, "public", "topical_share"),
        mc.fit_persistence_models(df, "public"),
    ]
    for frame in frames:
        ok = frame[frame["estimate"].notna()]
        assert len(ok) > 0
        assert ok["ci_low"].notna().all()
        assert ok["ci_high"].notna().all()
        ame_rows = ok[ok["scale"].isin(("probability", "proportion"))]
        assert len(ame_rows) > 0
        # delta method 的 se 必须真的算出来了：stats_utils 在
        # cov_params() 不可用时会返回 NaN，本模块所有估计量都不允许
        # 落到那个分支（落到了就必须改用重拟合的 cluster bootstrap）
        assert ame_rows["se"].notna().all(), ame_rows[ame_rows["se"].isna()]
        assert np.isfinite(ame_rows["se"].astype(float)).all()


def test_share_model_rejects_an_out_of_range_proportion():
    """占比 > 1 必须炸掉，而不是照常拟合出一个被推走的估计

    分数 logit 以取值落在 [0,1] 内为前提（模块文档第 5 条）。越界值不会
    让 GLM 报错，只会悄悄改变估计——这属于上游分子/分母口径算错，必须有人
    去修，所以 stats_utils.check_proportion_range 直接抛异常。
    """
    df = _simulate_users(n_male=200, n_female=200, seed=31)
    df.loc[df.index[:5], "public_topical_share"] = 1.7
    with pytest.raises(ValueError, match="public_topical_share"):
        mc.fit_share_models(df, "public", "topical_share")


def test_persistence_model_rejects_an_out_of_range_proportion():
    df = _simulate_users(n_male=200, n_female=200, seed=32)
    df.loc[0, "public_source_month_share"] = 1.5
    with pytest.raises(ValueError, match="public_source_month_share"):
        mc.fit_persistence_models(df, "public")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_build_writes_four_result_tables(tmp_path, monkeypatch):
    df = _simulate_users(n_male=400, n_female=400, seed=19)
    monkeypatch.setattr(mc.config, "OUTPUT_DIR", str(tmp_path))
    df.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                  engine="pyarrow", index=False)

    paths = mc.build(year=2020)
    assert len(paths) == 4
    for path in paths:
        assert path.endswith(".parquet")
        frame = pd.read_parquet(path, columns=list(su.RESULT_SCHEMA))
        assert list(frame.columns) == list(su.RESULT_SCHEMA)
        assert sorted(frame["domain"].unique()) == ["celebrity", "public"]
    assert (tmp_path / "results" / "manifest_models_core" / "manifest.json").exists()


def test_build_result_keys_are_unique(tmp_path, monkeypatch):
    """一次完整 build 之后，(outcome, domain, model, term) 必须逐行唯一

    §12 的绘图层在 15 个地方按这个四元组取行再 .iloc[0]。键一旦撞上，
    取到的是哪一行完全取决于行序——不会报错、不会留痕，只会让图上的某个
    点变成另一个模型的估计。models_interaction / models_combination /
    models_temporal 三个模块各有一条这样的测试，喂图 1–3 的这两个模块
    （describe 与 models_core）此前一条都没有，这条补上 models_core 这一半。
    """
    df = _simulate_users(n_male=400, n_female=400, seed=41)
    monkeypatch.setattr(mc.config, "OUTPUT_DIR", str(tmp_path))
    df.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                  engine="pyarrow", index=False)

    key = ["outcome", "domain", "model", "term"]
    for path in mc.build(year=2020):
        frame = pd.read_parquet(path, columns=list(su.RESULT_SCHEMA))
        duplicated = frame[frame.duplicated(subset=key, keep=False)]
        assert duplicated.empty, (
            f"{path} 里有 {len(duplicated)} 行重复键，例如\n"
            f"{duplicated.sort_values(key).head(8).to_string(index=False)}"
        )


def test_build_uses_only_the_declared_scale_vocabulary(tmp_path, monkeypatch):
    """结果表里出现的每一个 scale 都必须在封闭词表里"""
    df = _simulate_users(n_male=300, n_female=300, seed=42)
    monkeypatch.setattr(mc.config, "OUTPUT_DIR", str(tmp_path))
    df.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                  engine="pyarrow", index=False)
    for path in mc.build(year=2020):
        frame = pd.read_parquet(path, columns=list(su.RESULT_SCHEMA))
        assert set(frame["scale"].dropna()) <= set(su.RESULT_SCALES)


def test_build_stamps_every_result_table_with_the_run_id(tmp_path, monkeypatch):
    """四张表都要记进 results/run_stamps.json，且共用同一个 run_id

    这是 C1 那条"崩掉的一步会静默发表上一次的数字"的落点之一：导出层
    只有拿到每张表的运行标识，才能判断盘上这一组表是不是同一批。
    """
    df = _simulate_users(n_male=200, n_female=200, seed=43)
    monkeypatch.setattr(mc.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv(mc.config.RUN_ID_ENV, "run-under-test")
    df.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                  engine="pyarrow", index=False)

    paths = mc.build(year=2020)
    stamps = mc.config.read_run_stamps(str(tmp_path / "results"))
    for path in paths:
        assert stamps[os.path.basename(path)]["run_id"] == "run-under-test"


# ---------------------------------------------------------------------------
# _formula_terms：缺列与常量列是两件不同的事
# ---------------------------------------------------------------------------

def test_formula_terms_drops_nothing_from_a_well_formed_m1_frame():
    """结构良好的 M1 建模帧不该掉任何一个协变量

    这条断言存在的理由：`covariate_note` 只在有东西被剔除时才写 note，
    所以"note 是空的"既可能意味着一切正常，也可能意味着剔除清单根本没
    被拼进去。先钉死"正常情况下确实一个都不掉"，下面两条关于剔除原因的
    测试才有意义。
    """
    df = _simulate_users(n_male=100, n_female=100, seed=44)
    frame = mc.prepare_model_frame(df)
    terms, dropped = mc._formula_terms(mc.M1, frame)
    assert dropped == []
    assert mc.covariate_note(dropped) is None
    assert terms[0] == mc.FOCAL_COLUMN
    assert set(terms[1:]) == set(mc.M1[1:])


def test_formula_terms_distinguishes_an_absent_column_from_a_constant_one():
    """缺列与常量列必须写成两种不同的 note 前缀

    两者是完全不同的事：verified_flag 整列不在建模帧里，是上游构表漏了
    一列、必须有人去修的 bug；verified_flag 在本层样本里恰好都等于 1，
    是这一层样本的真实性质。都写成 dropped_constant 会让前者永远伪装
    成后者，"这一层其实没控制住它"与"这一列压根没送进来"再也分不开。
    """
    df = _simulate_users(n_male=100, n_female=100, seed=45)
    frame = mc.prepare_model_frame(df)
    frame["verified_flag"] = 1.0                 # 常量列
    frame = frame.drop(columns=["log_fans"])     # 整列缺失
    _, dropped = mc._formula_terms(mc.M2, frame)
    assert ("verified_flag", mc.DROP_KIND_CONSTANT) in dropped
    assert ("log_fans", mc.DROP_KIND_MISSING) in dropped

    note = mc.covariate_note(dropped)
    assert "{}:verified_flag".format(mc.DROP_KIND_CONSTANT) in note
    assert "{}:log_fans".format(mc.DROP_KIND_MISSING) in note
    # 两个前缀不能是同一个字符串，否则这条测试什么也没区分
    assert mc.DROP_KIND_CONSTANT != mc.DROP_KIND_MISSING


# ---------------------------------------------------------------------------
# layers 参数：默认必须仍然是三层，主结果层的行为一个字节都不能变
# ---------------------------------------------------------------------------

def test_the_default_layers_are_still_all_three_for_every_fit_function():
    """默认不传 layers 时，四个 fit_* 都必须原样产出 M0/M1/M2

    这是加 layers 参数时唯一真正危险的地方：论文正文的四张结果表要三层
    （模块文档第 2 条），默认值一旦变成 M0/M1，会静悄悄地把 M2 从论文
    自己的表里删掉——那比它想省下的那点机时糟糕得多。
    """
    df = _simulate_users(n_male=300, n_female=300, seed=90)
    for frame in (
        mc.fit_entry_models(df, "public"),
        mc.fit_intensity_models(df, "public"),
        mc.fit_share_models(df, "public", "topical_share"),
        mc.fit_persistence_models(df, "public"),
    ):
        assert set(frame["model"]) == {"M0", "M1", "M2"}


def test_the_default_is_byte_identical_to_asking_for_all_three_layers():
    """default 与显式三层必须逐行逐列相同——这就是"主流程没被动过"的证据"""
    df = _simulate_users(n_male=300, n_female=300, seed=91)
    all_three = ("M0", "M1", "M2")
    pd.testing.assert_frame_equal(
        mc.fit_entry_models(df, "public"),
        mc.fit_entry_models(df, "public", layers=all_three))
    pd.testing.assert_frame_equal(
        mc.fit_share_models(df, "public", "topical_share"),
        mc.fit_share_models(df, "public", "topical_share", layers=all_three))
    pd.testing.assert_frame_equal(
        mc.fit_persistence_models(df, "public"),
        mc.fit_persistence_models(df, "public", layers=all_three))
    pd.testing.assert_frame_equal(
        mc.fit_intensity_models(df, "public"),
        mc.fit_intensity_models(df, "public", layers=all_three))


def test_asking_for_two_layers_drops_m2_and_leaves_the_other_rows_unchanged():
    """稳健性套件要的那两层，其数值必须与三层跑法逐行相同"""
    df = _simulate_users(n_male=300, n_female=300, seed=92)
    full = mc.fit_entry_models(df, "public")
    two = mc.fit_entry_models(df, "public", layers=("M0", "M1"))
    assert set(two["model"]) == {"M0", "M1"}
    expected = full[full["model"].isin(("M0", "M1"))].reset_index(drop=True)
    pd.testing.assert_frame_equal(two.reset_index(drop=True), expected)


def test_selected_layers_keeps_the_declaration_order_of_model_layers():
    """传进来的顺序不影响输出顺序：层的顺序只有 MODEL_LAYERS 一个来源"""
    assert [name for name, _ in mc.selected_layers(("M1", "M0"))] == ["M0", "M1"]
    assert [name for name, _ in mc.selected_layers(None)] == list(mc.MODEL_LAYER_NAMES)


def test_an_unknown_layer_name_raises_instead_of_being_silently_skipped():
    """写错层名如果被静默忽略，得到的是一张"少了几层却看不出少了"的表"""
    df = _simulate_users(n_male=50, n_female=50, seed=93)
    with pytest.raises(ValueError, match="M3"):
        mc.fit_entry_models(df, "public", layers=("M0", "M3"))
    with pytest.raises(ValueError):
        mc.selected_layers(("m1",))


def test_a_skipped_layer_is_never_fitted_at_all():
    """不是"拟合三层再丢掉 M2"，而是根本不去构造 M2 的样本

    这条测试守的是这次改动的**全部收益**：如果哪天有人把 layers 参数
    退化成"照样跑三层、最后再筛一遍"，数值结果不会有任何变化，只有这里
    会挂。
    """
    df = _simulate_users(n_male=200, n_female=200, seed=94)
    seen = []
    original = mc._layer_sample

    def _spy(frame, layer, n_input, base_mask=None, base_reason=None):
        seen.append(layer)
        return original(frame, layer, n_input, base_mask=base_mask,
                        base_reason=base_reason)

    mc._layer_sample = _spy
    try:
        mc.fit_entry_models(df, "public", layers=("M0", "M1"))
        assert seen == ["M0", "M1"]
        seen.clear()
        mc.fit_entry_models(df, "public")
        assert seen == ["M0", "M1", "M2"]
    finally:
        mc._layer_sample = original
