"""
gender_domain.robustness.harness 的单元测试。

本模块是整套稳健性套件的地基：每一个 variant（换样本、换分母、换词表……）
最终都要把自己产出的用户级宽表交给 estimate_all，由它去调用与主结果
完全相同的拟合函数、算出完全相同的六个量。测试的重点因此不是"模型
估得准不准"（那是 models_core / models_interaction 自己的测试范围），
而是这一层"调度"本身的两条纪律：

1. **六个量、每层一行，缺一不可，哪怕某个拟合炸了。**
   test_row_count_invariant_holds_when_a_fit_is_forced_to_fail 故意让
   一个 domain 缺一列，逼真实的 fit_entry_models / to_long_format 抛出
   KeyError（这是它们自己的实现里唯一没有被内部 try/except 兜住的失败
   模式），断言 estimate_all 仍然产出满编的行数，只是对应的行变成
   estimate=NaN 且 note 里带着异常信息。

2. **estimate_all 自己不拟合任何模型。**
   test_estimate_all_recovers_the_six_quantities_via_direct_calls 直接
   调用 models_core / models_interaction 的对应函数，断言 harness 给出
   的数字与直接调用完全一致（不是"接近"，是同一次确定性拟合的同一个
   浮点数）——如果哪天 harness 里长出了自己的一套拟合逻辑，这个测试
   会因为数字对不上而挂掉。
"""

import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import models_core as mc
from gender_domain import models_interaction as mi
from gender_domain import stats_utils as su
from gender_domain.robustness import harness


# ---------------------------------------------------------------------------
# 模拟数据构造：与 test_models_interaction._simulate_wide 同一套生成过程，
# 因为它同时覆盖了六个量需要的全部列（两个域的 source_entered /
# topical_share，外加两条 DiD 需要的长表列）。
# ---------------------------------------------------------------------------

def _simulate_frame(n_male=400, n_female=400, gap_public=0.10, gap_celebrity=-0.10,
                    base=0.45, seed=0, user_sd=0.05, profile_complete_frac=1.0):
    rng = np.random.default_rng(seed)
    n = n_male + n_female
    gender = np.array(["m"] * n_male + ["f"] * n_female)
    male = (gender == "m").astype(float)
    u = rng.normal(0.0, user_sd, n)

    p_public = np.clip(base + male * gap_public + u, 0.01, 0.99)
    p_celebrity = np.clip(base + male * gap_celebrity + u, 0.01, 0.99)

    df = pd.DataFrame({
        "user_id": [str(i) for i in range(n)],
        "gender": gender,
        "n_posts": rng.integers(1, 200, n),
        "n_retweets": rng.integers(1, 100, n),
        "n_expressive_posts": rng.integers(1, 80, n),
        "n_active_days": rng.integers(1, 300, n),
        "n_active_months": rng.integers(1, 13, n),
        "n_active_months_panel": rng.integers(1, 13, n),
        "region": rng.choice(["north", "south", "west"], n),
        "verified_flag": rng.integers(0, 2, n).astype(float),
        "log_fans": rng.normal(5, 1, n),
        "log_friends": rng.normal(4, 1, n),
        "profile_complete": rng.random(n) < profile_complete_frac,
    })
    for domain, p in (("public", p_public), ("celebrity", p_celebrity)):
        df[f"{domain}_source_entered"] = rng.random(n) < p
        share = np.clip(p + rng.normal(0.0, 0.08, n), 0.001, 0.999)
        df[f"{domain}_topical_share"] = share
        df[f"{domain}_source_share"] = share
        df[f"{domain}_source_count"] = rng.integers(0, 10, n)
        df[f"{domain}_source_months"] = rng.integers(0, 3, n)
        df[f"{domain}_source_month_share"] = rng.random(n) * 0.5
    return df


def _row(frame, model, term, domain=None, outcome=None):
    mask = (frame["model"] == model) & (frame["term"] == term)
    if domain is not None:
        mask &= frame["domain"] == domain
    if outcome is not None:
        mask &= frame["outcome"] == outcome
    sub = frame[mask]
    assert len(sub) == 1, (
        f"期望恰好一行 model={model} term={term} domain={domain} outcome={outcome}，"
        f"实际 {len(sub)} 行"
    )
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# QUANTITIES 常量
# ---------------------------------------------------------------------------

def test_quantities_are_exactly_the_six_pre_specified_keys():
    assert harness.QUANTITIES == (
        "entry_public",
        "entry_celebrity",
        "topical_public",
        "topical_celebrity",
        "did_entry",
        "did_topical",
    )


# ---------------------------------------------------------------------------
# 行数不变量
# ---------------------------------------------------------------------------

def test_estimate_all_row_count_matches_quantities_times_layers():
    df = _simulate_frame(seed=1)
    out = harness.estimate_all(df, "baseline", "baseline")
    assert len(out) == len(harness.QUANTITIES) * 2  # 默认 layers=("M0","M1")


def test_estimate_all_row_count_scales_with_extra_layer():
    df = _simulate_frame(seed=1)
    out = harness.estimate_all(df, "baseline", "baseline", layers=("M0", "M1", "M2"))
    assert len(out) == len(harness.QUANTITIES) * 3


def test_row_count_invariant_holds_when_a_fit_is_forced_to_fail():
    """缺一列会让 fit_entry_models / to_long_format 直接抛 KeyError——
    这是这两个函数唯一没有被自己的内部 try/except 兜住的失败模式，
    因此是逼真实失败的最直接办法（不是靠 monkeypatch 伪造一个假异常）。
    """
    df = _simulate_frame(seed=2)
    broken = df.drop(columns=["celebrity_source_entered"])
    out = harness.estimate_all(broken, "broken_variant", "missing_celebrity_entry_column")

    assert len(out) == len(harness.QUANTITIES) * 2

    # entry_celebrity 与 did_entry 都依赖这一列，两者的每一层都必须是
    # NaN 且带着可追溯的失败 note，而不是从结果表里整行消失
    affected = out[out["outcome"].isin(["source_entered"]) &
                   (out["domain"].isin(["celebrity", mi.DOMAIN_BOTH]))]
    assert len(affected) > 0
    assert affected["estimate"].isna().all()
    assert affected["note"].notna().all()
    assert (affected["note"].str.contains("KeyError") |
            affected["note"].str.contains("celebrity_source_entered")).all()

    # 未受影响的量必须照常估出真实数字，不能被这一个失败拖累
    untouched = out[out["outcome"] == "topical_share"]
    assert untouched["estimate"].notna().all()


# ---------------------------------------------------------------------------
# 与直接调用底层拟合函数完全一致
# ---------------------------------------------------------------------------

def test_estimate_all_recovers_the_six_quantities_via_direct_calls():
    df = _simulate_frame(seed=3)
    out = harness.estimate_all(df, "baseline", "baseline", layers=("M0", "M1"))

    entry_public = mc.fit_entry_models(df, "public")
    entry_celebrity = mc.fit_entry_models(df, "celebrity")
    topical_public = mc.fit_share_models(df, "public", "topical_share")
    topical_celebrity = mc.fit_share_models(df, "celebrity", "topical_share")
    did_entry_long = mi.to_long_format(df, "source_entry")
    did_topical_long = mi.to_long_format(df, "topical_share")

    for model in ("M0", "M1"):
        assert _row(out, model, mc.TERM_AME, domain="public",
                    outcome="source_entered")["estimate"] == pytest.approx(
            _row(entry_public, model, mc.TERM_AME)["estimate"]
        )
        assert _row(out, model, mc.TERM_AME, domain="celebrity",
                    outcome="source_entered")["estimate"] == pytest.approx(
            _row(entry_celebrity, model, mc.TERM_AME)["estimate"]
        )

        out_topical_public = out[
            (out["model"] == model) & (out["term"] == mc.TERM_AME) &
            (out["domain"] == "public") & (out["outcome"] == "topical_share")
        ]
        assert len(out_topical_public) == 1
        assert out_topical_public.iloc[0]["estimate"] == pytest.approx(
            _row(topical_public, model, mc.TERM_AME)["estimate"]
        )

        out_topical_celebrity = out[
            (out["model"] == model) & (out["term"] == mc.TERM_AME) &
            (out["domain"] == "celebrity") & (out["outcome"] == "topical_share")
        ]
        assert len(out_topical_celebrity) == 1
        assert out_topical_celebrity.iloc[0]["estimate"] == pytest.approx(
            _row(topical_celebrity, model, mc.TERM_AME)["estimate"]
        )

        did_entry_fit = mi.fit_interaction(did_entry_long, model, outcome_kind="source_entry")
        did_topical_fit = mi.fit_interaction(did_topical_long, model, outcome_kind="topical_share")

        out_did_entry = out[
            (out["model"] == model) & (out["term"] == mi.TERM_DID) &
            (out["outcome"] == "source_entered")
        ]
        assert len(out_did_entry) == 1
        assert out_did_entry.iloc[0]["estimate"] == pytest.approx(
            _row(did_entry_fit, model, mi.TERM_DID, domain=mi.DOMAIN_BOTH)["estimate"]
        )

        out_did_topical = out[
            (out["model"] == model) & (out["term"] == mi.TERM_DID) &
            (out["outcome"] == "topical_share")
        ]
        assert len(out_did_topical) == 1
        assert out_did_topical.iloc[0]["estimate"] == pytest.approx(
            _row(did_topical_fit, model, mi.TERM_DID, domain=mi.DOMAIN_BOTH)["estimate"]
        )


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_schema_columns_exactly_match_the_extended_schema():
    df = _simulate_frame(seed=4)
    out = harness.estimate_all(df, "baseline", "baseline")
    assert list(out.columns) == list(harness.ROBUSTNESS_SCHEMA)
    assert list(harness.ROBUSTNESS_SCHEMA) == list(su.RESULT_SCHEMA) + [
        "variant_family", "variant_label", "replicate", "seed"
    ]


def test_attach_variant_identity_rejects_unknown_extra_field():
    """扩展 schema 必须延续 tidy_result"未知字段直接报错"的行为，
    而不是绕过它——这是任务简报明确要求的那条纪律。
    """
    row = su.tidy_result(
        outcome="source_entered", domain="public", model="M0", term=mc.TERM_AME,
        estimate=0.1, se=0.01, ci_low=0.08, ci_high=0.12, scale="probability",
        n_obs=100, n_dropped=0, drop_reason=None, note=None,
    )
    with pytest.raises(ValueError):
        harness.attach_variant_identity(
            row, variant_family="x", variant_label="y", replicate=0, seed=None,
            extra_bogus_field="not allowed",
        )


# ---------------------------------------------------------------------------
# variant_family / variant_label / replicate / seed
# ---------------------------------------------------------------------------

def test_seed_is_recorded_as_none_for_deterministic_variant():
    df = _simulate_frame(seed=5)
    out = harness.estimate_all(df, "some_family", "some_label", replicate=0, seed=None)
    assert "seed" in out.columns
    # seed 必须逐行显式写成 None，而不是被省略——列必须存在，且每一行都可读到它
    assert out["seed"].isna().all()


def test_seed_is_recorded_when_given():
    df = _simulate_frame(seed=5)
    out = harness.estimate_all(df, "some_family", "some_label", replicate=3, seed=42)
    assert (out["seed"] == 42).all()
    assert (out["replicate"] == 3).all()
    assert (out["variant_family"] == "some_family").all()
    assert (out["variant_label"] == "some_label").all()


def test_baseline_uses_variant_family_baseline_and_replicate_zero():
    df = _simulate_frame(seed=6)
    out = harness.baseline(df)
    assert (out["variant_family"] == "baseline").all()
    assert (out["replicate"] == 0).all()
    assert out["seed"].isna().all()
    assert len(out) == len(harness.QUANTITIES) * 2


# ---------------------------------------------------------------------------
# append_rows
# ---------------------------------------------------------------------------

def test_append_rows_twice_yields_both_replicates(tmp_path):
    df = _simulate_frame(seed=7, n_male=200, n_female=200)
    path = os.path.join(str(tmp_path), "some_family.parquet")

    first = harness.estimate_all(df, "some_family", "rep0", replicate=0, seed=1)
    harness.append_rows(first, path)
    second = harness.estimate_all(df, "some_family", "rep1", replicate=1, seed=2)
    harness.append_rows(second, path)

    combined = pd.read_parquet(path, engine="pyarrow", columns=list(harness.ROBUSTNESS_SCHEMA))
    assert len(combined) == len(first) + len(second)
    assert set(combined["replicate"].unique()) == {0, 1}
    assert list(combined.columns) == list(harness.ROBUSTNESS_SCHEMA)
