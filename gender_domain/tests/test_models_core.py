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
        df[f"{d}_source_share"] = df[f"{d}_source_count"] / df["n_retweets"]
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
    assert m2["drop_reason"] == "incomplete_profile"


def test_m2_also_requires_region_because_region_is_an_m2_control():
    """region 是 M2 的控制变量，缺 region 同样进不了 M2，一并记为画像不完整"""
    df = _simulate_users(n_male=500, n_female=500, seed=20)
    df.loc[df.index[:80], "region"] = None
    out = mc.fit_entry_models(df, "public")
    m2 = _row(out, "M2")
    assert m2["n_obs"] == len(df) - 80
    assert m2["drop_reason"] == "incomplete_profile"
    assert _row(out, "M1")["n_obs"] == len(df)


def test_missing_profile_columns_degrade_only_m2():
    """表 C 没跑过 profile_join 时，M2 整层留痕失败，M0/M1 照常给结果"""
    df = _simulate_users(n_male=400, n_female=400, seed=21)
    df = df.drop(columns=["profile_complete", "verified_flag", "log_fans", "log_friends"])
    out = mc.fit_entry_models(df, "public")
    m2 = _row(out, "M2")
    assert np.isnan(m2["estimate"])
    assert m2["n_obs"] == 0
    assert m2["drop_reason"] == "incomplete_profile"
    assert isinstance(m2["note"], str) and m2["note"]
    assert np.isfinite(_row(out, "M1")["estimate"])


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
        df[f"{d}_source_share"] = df[f"{d}_source_count"] / df["n_retweets"]
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

def test_persistence_ame_equals_month_weighted_mean_difference():
    """分母是活跃月数，因此 M0 的 AME 是以活跃月数为权重的均值差"""
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
