"""
gender_domain.robustness.synthesis 的单元测试（§13.10 综合判定、§11.3 FDR、
§12.7 规格曲线数据）。

这份测试的重心按重要性排列：

1. **方向一致率的分母只能是"真的估出来了"的行。**
   本套件里有两类 NaN 行：拟合失败留痕，以及"这个变体按构造碰不到这个量"
   （measures 的帖子类型/替代测量变体把三个 entry 量写成 NaN，
   samples 的 log(1+x) 对照整个做不到、只留一行注明原因）。把它们算进
   分母会低报一致率，算进分子会**凭空报出一个从没被检验过的一致**——
   后者是上游两次复核都抓到过的同一个错误。测试因此钉死：同一个量下
   live 行与 NaN 行混在一起时，分母恰好等于 live 行数。

2. **note 是追加的，不是覆盖的。**
   voc._annotate_note 把新说明**前置/后置**拼到已有 note 上，所以任何
   "这一行是不是那类刻意 NaN"的判断只能用子串匹配。测试给刻意 NaN 的行
   贴上一段前缀，再断言它仍然被识别出来。

3. **缺整个 family 的量必须被标成"没测全"，而不是拿现有的行给它下结论。**
   §13.2 的 untreated vs log(1+x) 已知做不到，这条路径在真实数据上必然
   被走到。

4. **judge 报告，不裁定。** §13.10 明确写了稳健不等于"每个版本都显著"。
   测试逐列扫描输出，出现任何布尔"robust/verdict/passes"列就失败。

5. **FDR 只作用于次要分析。** §11.3：六个预先设定量永远不进 BH 校正。

6. **一个账号撑起全部效应时，judge 必须让它浮出来。**

7. **domain 有三个取值。** DiD 行的 domain 是 "both"，任何按 domain 分组
   的动作都必须容得下它。
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import config
from gender_domain import models_core as mc
from gender_domain import models_interaction as mi
from gender_domain import stats_utils as su
from gender_domain.robustness import accounts as acc
from gender_domain.robustness import harness
from gender_domain.robustness import measures as mea
from gender_domain.robustness import samples as smp
from gender_domain.robustness import synthesis as syn
from gender_domain.robustness import vocabulary as voc


# ---------------------------------------------------------------------------
# 合成夹具：用真实 schema 与真实标签造出五个 family 的输出
# ---------------------------------------------------------------------------

def make_row(quantity, model, estimate, se=0.01, family="vocabulary", label="v",
             replicate=0, seed=None, note=None, n_obs=1000):
    """按共享 schema + 变体身份造一行，与各 family 的落盘行完全同构"""
    meta = harness.QUANTITY_META[quantity]
    if estimate == estimate:
        half = 1.96 * se
        ci_low, ci_high = estimate - half, estimate + half
    else:
        se, ci_low, ci_high = np.nan, np.nan, np.nan
    row = su.tidy_result(
        outcome=meta["outcome"], domain=meta["domain"], model=model,
        term=meta["term"], estimate=estimate, se=se, ci_low=ci_low,
        ci_high=ci_high, scale=meta["scale"], n_obs=n_obs, n_dropped=0,
        drop_reason=None, note=note,
    )
    return harness.attach_variant_identity(row, family, label, replicate, seed)


def variant_rows(family, label, estimates, layers=("M0", "M1"), replicate=0,
                 seed=None, note=None, nan_note=None, se=0.01):
    """六个量 × layers 的一个完整变体；estimates 缺的键写成 NaN"""
    rows = []
    for model in layers:
        for quantity in harness.QUANTITIES:
            value = estimates.get((quantity, model), estimates.get(quantity, np.nan))
            value = float(value) if value == value else np.nan
            row_note = note if value == value else (nan_note or note)
            rows.append(make_row(quantity, model, value, se=se, family=family,
                                 label=label, replicate=replicate, seed=seed,
                                 note=row_note))
    return pd.DataFrame(rows, columns=list(harness.ROBUSTNESS_SCHEMA))


# 基线（主结果表里的六个量）。方向：三正三负，好让"同号"这件事有区分度。
BASELINE = {
    ("entry_public", "M0"): 0.10, ("entry_public", "M1"): 0.08,
    ("entry_celebrity", "M0"): -0.20, ("entry_celebrity", "M1"): -0.16,
    ("topical_public", "M0"): 0.05, ("topical_public", "M1"): 0.04,
    ("topical_celebrity", "M0"): -0.03, ("topical_celebrity", "M1"): -0.02,
    ("did_entry", "M0"): 0.30, ("did_entry", "M1"): 0.24,
    ("did_topical", "M0"): 0.08, ("did_topical", "M1"): 0.06,
}


def write_main_results(out_dir):
    """主结果层的三张表：synthesis 的参照从这里显式取，而不是假设有基线行"""
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    entry, share, interaction = [], [], []
    for (quantity, model), value in BASELINE.items():
        row = make_row(quantity, model, value, family="main", label="main")
        row = {col: row[col] for col in su.RESULT_SCHEMA}
        if quantity.startswith("did_"):
            interaction.append(row)
        elif quantity.startswith("entry_"):
            entry.append(row)
        else:
            share.append(row)
    for name, rows in (("models_entry", entry), ("models_share", share),
                       ("interaction_gender_domain", interaction)):
        pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA)).to_parquet(
            os.path.join(results_dir, "{}.parquet".format(name)),
            engine="pyarrow", index=False)
    return results_dir


@pytest.fixture()
def robustness_project(tmp_path, monkeypatch):
    """把 config.OUTPUT_DIR 指到临时目录，写好五个 family 的合成输出"""
    out_dir = str(tmp_path / "analysis_data")
    os.makedirs(os.path.join(out_dir, "robustness"), exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    write_main_results(out_dir)

    rob = os.path.join(out_dir, "robustness")

    # --- 词表族：4 个 replicate，全部与基线同号 ---
    vocab = pd.concat([
        variant_rows(voc.VARIANT_FAMILY, "keep0.8_rep{}".format(i), {
            ("entry_public", "M0"): 0.10 + 0.01 * i,
            ("entry_public", "M1"): 0.08 + 0.01 * i,
            ("entry_celebrity", "M0"): -0.19, ("entry_celebrity", "M1"): -0.15,
            ("topical_public", "M0"): 0.05 + 0.002 * i,
            ("topical_public", "M1"): 0.04,
            ("topical_celebrity", "M0"): -0.03, ("topical_celebrity", "M1"): -0.02,
            ("did_entry", "M0"): 0.29, ("did_entry", "M1"): 0.23,
            ("did_topical", "M0"): 0.08, ("did_topical", "M1"): 0.06,
        }, replicate=i, seed=100 + i, note="retained_terms=650/816")
        for i in range(4)
    ], ignore_index=True)
    # 配对的校准行：与 rep0 逐字相同，只换身份标签。它不是一个独立变体。
    calib = vocab[vocab["variant_label"] == "keep0.8_rep0"].copy()
    calib["variant_family"] = voc.CALIBRATION_FAMILY
    calib["variant_label"] = "keep0.8_rep0_reaggregated"
    vocab = pd.concat([vocab, calib], ignore_index=True)
    vocab.to_parquet(os.path.join(rob, "vocabulary.parquet"),
                     engine="pyarrow", index=False)

    # --- 账号族：一个头部账号把 entry_celebrity 从 -0.20 拉到 -0.02 ---
    accounts = pd.concat([
        variant_rows(acc.VARIANT_FAMILY, "loo_celebrity_rank01_9999", {
            ("entry_public", "M0"): 0.10, ("entry_public", "M1"): 0.08,
            # M0 与 M1 各自相对自己那一层的基线都恰好偏移 90%，这样测试
            # 不依赖 judge 到底在哪一层上取最大值
            ("entry_celebrity", "M0"): -0.02, ("entry_celebrity", "M1"): -0.016,
            ("topical_public", "M0"): 0.05, ("topical_public", "M1"): 0.04,
            ("topical_celebrity", "M0"): -0.03, ("topical_celebrity", "M1"): -0.02,
            ("did_entry", "M0"): 0.12, ("did_entry", "M1"): 0.10,
            ("did_topical", "M0"): 0.08, ("did_topical", "M1"): 0.06,
        }, note="excluded_volume_share=0.31"),
        variant_rows(acc.VARIANT_FAMILY, "loo_celebrity_rank02_8888", {
            ("entry_public", "M0"): 0.10, ("entry_public", "M1"): 0.08,
            ("entry_celebrity", "M0"): -0.19, ("entry_celebrity", "M1"): -0.15,
            ("topical_public", "M0"): 0.05, ("topical_public", "M1"): 0.04,
            ("topical_celebrity", "M0"): -0.03, ("topical_celebrity", "M1"): -0.02,
            ("did_entry", "M0"): 0.29, ("did_entry", "M1"): 0.23,
            ("did_topical", "M0"): 0.08, ("did_topical", "M1"): 0.06,
        }, note="excluded_volume_share=0.02"),
        variant_rows(acc.BOOTSTRAP_FAMILY, "bootstrap_rep0", {
            ("entry_public", "M0"): 0.11, ("entry_public", "M1"): 0.09,
            ("entry_celebrity", "M0"): -0.21, ("entry_celebrity", "M1"): -0.17,
            ("topical_public", "M0"): 0.05, ("topical_public", "M1"): 0.04,
            ("topical_celebrity", "M0"): -0.03, ("topical_celebrity", "M1"): -0.02,
            ("did_entry", "M0"): 0.31, ("did_entry", "M1"): 0.25,
            ("did_topical", "M0"): 0.08, ("did_topical", "M1"): 0.06,
        }, replicate=0, seed=7),
    ], ignore_index=True)
    accounts.to_parquet(os.path.join(rob, "accounts.parquet"),
                        engine="pyarrow", index=False)

    # --- 样本族：untreated + 截尾，外加 §13.2 做不到的那一行 ---
    untreated = variant_rows(
        smp.VARIANT_FAMILY_EXTREME, "untreated", dict(BASELINE),
        note="activity_covariates_enter_M1_as_log1p_by_models_core")
    trimmed = variant_rows(smp.VARIANT_FAMILY_EXTREME, "trim_pooled_top1pct", {
        ("entry_public", "M0"): 0.09, ("entry_public", "M1"): 0.07,
        ("entry_celebrity", "M0"): -0.18, ("entry_celebrity", "M1"): -0.14,
        ("topical_public", "M0"): 0.05, ("topical_public", "M1"): 0.04,
        ("topical_celebrity", "M0"): -0.03, ("topical_celebrity", "M1"): -0.02,
        ("did_entry", "M0"): 0.27, ("did_entry", "M1"): 0.21,
        ("did_topical", "M0"): 0.08, ("did_topical", "M1"): 0.06,
    }, note="strictly_greater_than_cut(no_tie_splitting)")
    # §13.2 的 log(1+x)：整族做不到，只有一行注明原因的行（model/term 都是 None）
    log1p = voc._note_only_rows(
        smp.LOG1P_LABEL, smp.LOG1P_NOTE, outcome="source_entered",
        domain="public", variant_family=smp.VARIANT_FAMILY_EXTREME,
    )
    pd.concat([untreated, trimmed, log1p], ignore_index=True).to_parquet(
        os.path.join(rob, "samples.parquet"), engine="pyarrow", index=False)

    # --- 测量族：帖子类型变体把三个 entry 量写成刻意 NaN ---
    post_type = variant_rows(
        mea.VARIANT_FAMILY_POST_TYPE,
        "post_types=original;denominator=original_posts_with_nonzero_chars", {
            ("topical_public", "M0"): 0.06, ("topical_public", "M1"): 0.05,
            ("topical_celebrity", "M0"): -0.04, ("topical_celebrity", "M1"): -0.03,
            ("did_topical", "M0"): 0.10, ("did_topical", "M1"): 0.08,
        },
        note="post_types=original",
        # note 是追加的：刻意 NaN 的说明前面还挂着别的内容
        nan_note="post_types=original;" + mea.NOTE_ENTRY_NOT_ESTIMATED,
    )
    temporal = variant_rows(
        mea.VARIANT_FAMILY_TEMPORAL,
        "months=1-6;denominator=expressive_posts_in_those_months", {
            ("entry_public", "M0"): 0.09, ("entry_public", "M1"): 0.07,
            ("entry_celebrity", "M0"): -0.17, ("entry_celebrity", "M1"): -0.13,
            ("topical_public", "M0"): 0.04, ("topical_public", "M1"): 0.03,
            ("topical_celebrity", "M0"): -0.03, ("topical_celebrity", "M1"): -0.02,
            ("did_entry", "M0"): 0.26, ("did_entry", "M1"): 0.20,
            ("did_topical", "M0"): 0.07, ("did_topical", "M1"): 0.05,
        })
    pd.concat([post_type, temporal], ignore_index=True).to_parquet(
        os.path.join(rob, "measures.parquet"), engine="pyarrow", index=False)

    return {"out_dir": out_dir, "robustness_dir": rob}


# ---------------------------------------------------------------------------
# load_all：参照必须显式取得，并说明取的是什么
# ---------------------------------------------------------------------------

def test_load_all_collects_every_family_and_labels_each_row_with_its_quantity(
    robustness_project
):
    df = syn.load_all(2020)
    families = set(df["variant_family"])
    assert voc.VARIANT_FAMILY in families
    assert acc.VARIANT_FAMILY in families
    assert smp.VARIANT_FAMILY_EXTREME in families
    assert mea.VARIANT_FAMILY_POST_TYPE in families
    # 六个量的 key 逐行贴上；注明原因的行贴不上，如实留空
    assert set(df["quantity"].dropna()) == set(harness.QUANTITIES)
    note_only = df[df["variant_label"] == smp.LOG1P_LABEL]
    assert len(note_only) == 1
    assert note_only["quantity"].isna().all()


def test_load_all_obtains_the_reference_explicitly_and_says_what_it_used(
    robustness_project
):
    """没有任何 family 产出基线行——参照必须显式取得，并在输出里说明来源"""
    df = syn.load_all(2020)
    # 合成数据里没有任何一行 variant_family == "baseline"，除非 synthesis 自己补
    baseline = df[df["variant_family"] == syn.BASELINE_FAMILY]
    assert len(baseline) > 0, "load_all 必须显式补上参照行，而不是假设数据里有"
    sources = set(df["baseline_source"])
    assert len(sources) == 1
    source = sources.pop()
    assert "results" in source, "参照来源必须写清楚，这里应当是主结果表"
    m0 = baseline[(baseline["quantity"] == "entry_public") & (baseline["model"] == "M0")]
    assert float(m0["estimate"].iloc[0]) == pytest.approx(0.10)


def test_load_all_records_an_unavailable_reference_instead_of_inventing_one(
    robustness_project
):
    os.remove(os.path.join(robustness_project["out_dir"], "results",
                           "models_entry.parquet"))
    os.remove(os.path.join(robustness_project["out_dir"], "results",
                           "models_share.parquet"))
    os.remove(os.path.join(robustness_project["out_dir"], "results",
                           "interaction_gender_domain.parquet"))
    df = syn.load_all(2020, allow_recompute=False)
    assert set(df["baseline_source"]) == {syn.BASELINE_SOURCE_UNAVAILABLE}
    baseline = df[df["variant_family"] == syn.BASELINE_FAMILY]
    assert baseline["estimate"].isna().all()
    assert baseline["note"].str.contains("baseline_unavailable").all()


def test_load_all_handles_the_three_domain_values(robustness_project):
    df = syn.load_all(2020)
    assert set(df["domain"]) >= {"public", "celebrity", mi.DOMAIN_BOTH}
    did = df[df["quantity"] == "did_entry"]
    assert set(did["domain"]) == {mi.DOMAIN_BOTH}


# ---------------------------------------------------------------------------
# direction_consistency：分母只数活着的估计
# ---------------------------------------------------------------------------

def _hand_built_frame():
    """4 个 live 变体（3 个同号、1 个反号）+ 2 个刻意 NaN 变体"""
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY,
                     {("entry_public", "M0"): 0.10, ("entry_public", "M1"): 0.08}),
        variant_rows("vocabulary", "keep0.8_rep0",
                     {("entry_public", "M0"): 0.12, ("entry_public", "M1"): 0.09}),
        variant_rows("vocabulary", "keep0.8_rep1",
                     {("entry_public", "M0"): 0.09, ("entry_public", "M1"): 0.07}),
        variant_rows("accounts", "loo_public_rank01_1",
                     {("entry_public", "M0"): 0.11, ("entry_public", "M1"): 0.09}),
        variant_rows("accounts", "loo_public_rank02_2",
                     {("entry_public", "M0"): -0.01, ("entry_public", "M1"): -0.02}),
        # 刻意 NaN：note 前面还挂着别的内容，必须靠子串匹配才认得出来
        variant_rows(mea.VARIANT_FAMILY_POST_TYPE, "post_types=original",
                     {}, nan_note="post_types=original;"
                                  + mea.NOTE_ENTRY_NOT_ESTIMATED),
        variant_rows(mea.VARIANT_FAMILY_POST_TYPE, "post_types=all_text",
                     {}, nan_note="post_types=all_text;"
                                  + mea.NOTE_ENTRY_NOT_ESTIMATED),
    ]
    return pd.concat(frames, ignore_index=True)


def test_direction_consistency_counts_only_variants_carrying_live_estimates():
    df = _hand_built_frame()
    out = syn.direction_consistency(df)
    row = out[(out["quantity"] == "entry_public") & (out["model"] == "M0")].iloc[0]
    assert int(row["n_live"]) == 4, "分母必须是 4 个真的估出来的变体"
    assert int(row["n_nan"]) == 2, "两行刻意 NaN 必须被单独数出来，而不是消失"
    assert int(row["n_agree"]) == 3
    assert float(row["share_agree"]) == pytest.approx(0.75)
    assert float(row["estimate_min"]) == pytest.approx(-0.01)
    assert float(row["estimate_max"]) == pytest.approx(0.12)


def test_direction_consistency_reports_a_wilson_interval_on_the_share():
    df = _hand_built_frame()
    out = syn.direction_consistency(df)
    row = out[(out["quantity"] == "entry_public") & (out["model"] == "M0")].iloc[0]
    low, high = su.proportion_ci(3, 4)
    assert float(row["share_ci_low"]) == pytest.approx(low)
    assert float(row["share_ci_high"]) == pytest.approx(high)


def test_a_quantity_whose_variants_are_all_nan_is_never_reported_as_agreeing():
    """这是上游两次复核抓到的同一个错误：分母算进 NaN 会报出没测过的一致"""
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY,
                     {("entry_celebrity", "M0"): -0.20}),
        variant_rows(mea.VARIANT_FAMILY_POST_TYPE, "post_types=original", {},
                     nan_note="x;" + mea.NOTE_ENTRY_NOT_ESTIMATED),
    ]
    out = syn.direction_consistency(pd.concat(frames, ignore_index=True))
    row = out[(out["quantity"] == "entry_celebrity") & (out["model"] == "M0")].iloc[0]
    assert int(row["n_live"]) == 0
    assert np.isnan(float(row["share_agree"]))
    assert bool(row["incompletely_tested"])


def test_paired_calibration_rows_are_not_counted_as_independent_variants(
    robustness_project
):
    """vocabulary_calibration 的 _reaggregated 行是 rep0 的逐字拷贝"""
    df = syn.load_all(2020)
    assert voc.CALIBRATION_FAMILY in set(df["variant_family"])
    out = syn.direction_consistency(df)
    row = out[(out["quantity"] == "entry_public") & (out["model"] == "M0")].iloc[0]
    labels = set(syn.variant_pool(df)["variant_label"])
    assert "keep0.8_rep0_reaggregated" not in labels
    assert "keep0.8_rep0" in labels
    pool = syn.variant_pool(df).query("quantity == 'entry_public' and model == 'M0'")
    assert int(row["n_live"]) == int(pool["estimate"].notna().sum())


# ---------------------------------------------------------------------------
# 「没测全」必须是一个标记，不是拿现有的行下结论
# ---------------------------------------------------------------------------

def test_a_quantity_missing_a_whole_family_is_flagged_incompletely_tested():
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY,
                     {("entry_public", "M0"): 0.10}),
        variant_rows("vocabulary", "keep0.8_rep0", {("entry_public", "M0"): 0.11}),
    ]
    out = syn.direction_consistency(pd.concat(frames, ignore_index=True))
    row = out[(out["quantity"] == "entry_public") & (out["model"] == "M0")].iloc[0]
    assert bool(row["incompletely_tested"])
    missing = str(row["families_missing"])
    assert acc.VARIANT_FAMILY in missing
    assert smp.VARIANT_FAMILY_EXTREME in missing


def test_M2_is_only_expected_from_the_profile_family_and_stays_out_of_judge():
    """§11.4：M2 会收窄样本，只有 §13.9 那一族产出，不能拿全套族去比"""
    layers = ("M0", "M1", "M2")
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY, BASELINE,
                     layers=layers),
        variant_rows(smp.VARIANT_FAMILY_USER_TYPE, "verified_individuals_only",
                     {("entry_public", "M2"): 0.09}, layers=layers),
    ]
    df = pd.concat(frames, ignore_index=True)
    out = syn.direction_consistency(df)
    row = out[(out["quantity"] == "entry_public") & (out["model"] == "M2")].iloc[0]
    assert row["families_missing"] is None, "M2 上不该报另外九族缺失"
    assert int(row["n_live"]) == 1
    m0 = out[(out["quantity"] == "entry_public") & (out["model"] == "M0")].iloc[0]
    assert voc.VARIANT_FAMILY in str(m0["families_missing"])
    judged = syn.judge(df)
    assert not [c for c in judged.columns if c.endswith("_M2")]


def test_incomplete_variants_are_detected_by_the_row_count_invariant():
    """estimate_all 恒产出 6 × len(layers) 行——少于此就是这个变体没跑完"""
    good = variant_rows("vocabulary", "keep0.8_rep0", {("entry_public", "M0"): 0.1})
    truncated = variant_rows("accounts", "loo_public_rank01_1",
                             {("entry_public", "M0"): 0.1}).iloc[:3]
    note_only = voc._note_only_rows(
        smp.LOG1P_LABEL, smp.LOG1P_NOTE, outcome="source_entered", domain="public",
        variant_family=smp.VARIANT_FAMILY_EXTREME,
    )
    df = pd.concat([good, truncated, note_only], ignore_index=True)
    out = syn.incomplete_variants(df)
    labels = set(out["variant_label"])
    assert "loo_public_rank01_1" in labels
    assert smp.LOG1P_LABEL in labels
    assert "keep0.8_rep0" not in labels
    reasons = dict(zip(out["variant_label"], out["reason"]))
    assert reasons[smp.LOG1P_LABEL] == syn.REASON_NOTE_ONLY
    assert reasons["loo_public_rank01_1"] == syn.REASON_MISSING_ROWS


# ---------------------------------------------------------------------------
# activity_attenuation：§13.10 的第二条准则
# ---------------------------------------------------------------------------

def test_activity_attenuation_measures_the_shrinkage_from_M0_to_M1(
    robustness_project
):
    df = syn.load_all(2020)
    out = syn.activity_attenuation(df)
    row = out[out["quantity"] == "entry_public"].iloc[0]
    # 基线 0.10 -> 0.08，衰减 20%
    assert float(row["baseline_estimate_M0"]) == pytest.approx(0.10)
    assert float(row["baseline_estimate_M1"]) == pytest.approx(0.08)
    assert float(row["baseline_attenuation"]) == pytest.approx(0.20)
    assert int(row["n_variant_pairs"]) > 0
    assert 0.0 < float(row["attenuation_median"]) < 1.0


def test_activity_attenuation_counts_sign_flips_between_layers():
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY,
                     {("entry_public", "M0"): 0.10, ("entry_public", "M1"): 0.08}),
        variant_rows("vocabulary", "keep0.8_rep0",
                     {("entry_public", "M0"): 0.10, ("entry_public", "M1"): -0.02}),
        variant_rows("vocabulary", "keep0.8_rep1",
                     {("entry_public", "M0"): 0.10, ("entry_public", "M1"): 0.05}),
    ]
    out = syn.activity_attenuation(pd.concat(frames, ignore_index=True))
    row = out[out["quantity"] == "entry_public"].iloc[0]
    assert int(row["n_sign_flip_M0_to_M1"]) == 1
    assert int(row["n_variant_pairs"]) == 2


# ---------------------------------------------------------------------------
# influence_summary：单个账号 / 用户群 / 词表 / 月份的影响
# ---------------------------------------------------------------------------

def test_influence_summary_states_the_threshold_and_names_the_worst_variant(
    robustness_project
):
    df = syn.load_all(2020)
    out = syn.influence_summary(df, threshold=0.5)
    sub = out[(out["quantity"] == "entry_celebrity") & (out["model"] == "M0")
              & (out["influence_unit"] == syn.UNIT_ACCOUNT)].iloc[0]
    # -0.20 -> -0.02 是基线的 90%
    assert float(sub["max_abs_relative_shift"]) == pytest.approx(0.9)
    assert sub["worst_variant_label"] == "loo_celebrity_rank01_9999"
    assert float(sub["threshold"]) == pytest.approx(0.5)
    assert bool(sub["exceeds_threshold"])
    # 词表族没有把这个量推出去
    term = out[(out["quantity"] == "entry_celebrity") & (out["model"] == "M0")
               & (out["influence_unit"] == syn.UNIT_TERM_SET)].iloc[0]
    assert not bool(term["exceeds_threshold"])


def test_influence_summary_covers_the_four_units_named_by_the_brief(
    robustness_project
):
    df = syn.load_all(2020)
    out = syn.influence_summary(df)
    assert {syn.UNIT_ACCOUNT, syn.UNIT_USER_GROUP, syn.UNIT_TERM_SET,
            syn.UNIT_MONTH} <= set(out["influence_unit"])


# ---------------------------------------------------------------------------
# apply_fdr：只作用于次要分析
# ---------------------------------------------------------------------------

def test_apply_fdr_never_touches_the_six_prespecified_quantities():
    prespecified = pd.concat([
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY,
                     {key: value for key, value in BASELINE.items()}, se=0.5),
    ], ignore_index=True)
    secondary = pd.DataFrame([
        su.tidy_result(outcome="intensity", domain="public", model="M0",
                       term=mc.TERM_AME, estimate=0.4, se=0.1,
                       ci_low=0.2, ci_high=0.6, scale="log1p_count",
                       n_obs=100, n_dropped=0, drop_reason=None, note=None),
        su.tidy_result(outcome="persistence", domain="celebrity", model="M0",
                       term=mc.TERM_AME, estimate=0.02, se=0.1,
                       ci_low=-0.18, ci_high=0.22, scale="proportion",
                       n_obs=100, n_dropped=0, drop_reason=None, note=None),
    ], columns=list(su.RESULT_SCHEMA))
    df = pd.concat([prespecified[list(su.RESULT_SCHEMA)], secondary],
                   ignore_index=True)
    out = syn.apply_fdr(df, alpha=0.05)
    assert len(out) == len(df)
    pre = out[out["is_prespecified"]]
    assert len(pre) == 12
    assert pre["q_value"].isna().all()
    assert pre["fdr_rejected"].isna().all()
    sec = out[~out["is_prespecified"]]
    assert sec["q_value"].notna().all()
    assert int(out["fdr_n_tested"].dropna().iloc[0]) == 2


def test_apply_fdr_reproduces_benjamini_hochberg_on_known_p_values():
    # 两个 z 值：4.0（p≈6.3e-5）与 0.2（p≈0.84）
    rows = [
        su.tidy_result(outcome="intensity", domain="public", model="M0",
                       term=mc.TERM_AME, estimate=0.4, se=0.1, ci_low=0.2,
                       ci_high=0.6, scale="log1p_count", n_obs=100, n_dropped=0,
                       drop_reason=None, note=None),
        su.tidy_result(outcome="persistence", domain="public", model="M0",
                       term=mc.TERM_AME, estimate=0.02, se=0.1, ci_low=-0.18,
                       ci_high=0.22, scale="proportion", n_obs=100, n_dropped=0,
                       drop_reason=None, note=None),
    ]
    out = syn.apply_fdr(pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA)))
    q = sorted(out["q_value"].tolist())
    p = sorted(out["p_value"].tolist())
    assert q[0] == pytest.approx(p[0] * 2 / 1)
    assert q[1] == pytest.approx(p[1])
    assert bool(out.sort_values("p_value")["fdr_rejected"].iloc[0])
    assert not bool(out.sort_values("p_value")["fdr_rejected"].iloc[1])


def test_apply_fdr_leaves_rows_without_a_usable_standard_error_untested():
    rows = [
        su.tidy_result(outcome="intensity", domain="public", model="M0",
                       term=mc.TERM_AME, estimate=0.4, se=np.nan, ci_low=np.nan,
                       ci_high=np.nan, scale="log1p_count", n_obs=100,
                       n_dropped=0, drop_reason=None, note="bootstrap_ci"),
    ]
    out = syn.apply_fdr(pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA)))
    assert out["q_value"].isna().all()
    assert "se" in str(out["p_value_source"].iloc[0])


# ---------------------------------------------------------------------------
# specification_curve_data：§12.7 的图数据
# ---------------------------------------------------------------------------

def test_specification_curve_data_is_ordered_by_estimate_and_carries_flags(
    robustness_project
):
    df = syn.load_all(2020)
    out = syn.specification_curve_data(df)
    sub = out[(out["quantity"] == "entry_celebrity") & (out["model"] == "M0")
              & out["estimate_available"]]
    assert sub["estimate"].is_monotonic_increasing
    assert list(sub["rank"]) == list(range(1, len(sub) + 1))
    for column in ("variant_family", "variant_label", "replicate", "seed",
                   "influence_unit", "is_baseline", "crosses_zero",
                   "agrees_with_baseline", "relative_shift"):
        assert column in out.columns
    assert bool(out[out["variant_family"] == syn.BASELINE_FAMILY]["is_baseline"].all())


def test_specification_curve_data_keeps_unestimable_rows_visible(
    robustness_project
):
    df = syn.load_all(2020)
    out = syn.specification_curve_data(df)
    dead = out[~out["estimate_available"]]
    assert len(dead) > 0
    assert dead["rank"].isna().all()
    assert dead["note"].notna().all()


# ---------------------------------------------------------------------------
# judge：报告，不裁定
# ---------------------------------------------------------------------------

# 禁用的**词干**。必须按子串匹配：按整名匹配的话，`is_robust_overall`
# 或 `robust_flag` 会从名字这一关溜过去；再配上一个"名字里带 flag 就放行"
# 的白名单，布尔那一关也一起溜过去。这条守卫是本模块"报告而不裁定"这条
# 规则**唯一**的强制手段，不能留缝。
BANNED_VERDICT_STEMS = (
    "robust", "verdict", "pass", "surviv", "conclusion", "significant",
)


def test_judge_emits_no_boolean_robust_verdict(robustness_project):
    out = syn.judge(syn.load_all(2020))
    for column in out.columns:
        lowered = column.lower()
        hits = [stem for stem in BANNED_VERDICT_STEMS if stem in lowered]
        assert not hits, (
            "judge 不允许出现一个把判断替读者做完的列: {}（命中词干 {}）"
            .format(column, hits)
        )
    # 布尔那一关同样不留白名单：judge 里一个布尔列都不该有
    boolean_columns = [c for c in out.columns if out[c].dtype == bool]
    assert not boolean_columns, (
        "judge 不允许出现任何布尔列: {}".format(boolean_columns))


def test_the_anti_verdict_guard_would_catch_a_disguised_verdict_column():
    """守卫本身必须挡得住 is_robust_overall / robust_flag 这类伪装"""
    for disguised in ("is_robust_overall", "robust_flag", "passes_13_10",
                      "survives_all_variants", "significant_everywhere"):
        lowered = disguised.lower()
        assert any(stem in lowered for stem in BANNED_VERDICT_STEMS), disguised


def test_judge_surfaces_a_single_account_driving_the_whole_effect(
    robustness_project
):
    out = syn.judge(syn.load_all(2020))
    row = out[out["quantity"] == "entry_celebrity"].iloc[0]
    assert float(row["max_relative_shift_single_account"]) == pytest.approx(0.9)
    assert row["worst_single_account_variant"] == "loo_celebrity_rank01_9999"
    # 词表族没有把它推出去，两者必须能被分开读
    assert float(row["max_relative_shift_term_set"]) < 0.5


def test_judge_reports_the_four_criteria_of_1310(robustness_project):
    out = syn.judge(syn.load_all(2020))
    for column in ("direction_share_M0", "direction_share_M1",
                   "baseline_attenuation", "attenuation_median",
                   "max_relative_shift_single_account",
                   "max_relative_shift_user_group",
                   "max_relative_shift_term_set",
                   "max_relative_shift_month",
                   "share_ci_excludes_zero_M1",
                   "completeness", "baseline_source"):
        assert column in out.columns, column
    assert set(out["quantity"]) == set(harness.QUANTITIES)


def test_judge_flags_a_quantity_that_is_missing_a_family_rather_than_judging_it():
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY, BASELINE),
        variant_rows("vocabulary", "keep0.8_rep0",
                     {("entry_public", "M0"): 0.11, ("entry_public", "M1"): 0.09}),
    ]
    out = syn.judge(pd.concat(frames, ignore_index=True))
    row = out[out["quantity"] == "entry_public"].iloc[0]
    assert row["completeness"] != syn.COMPLETENESS_COMPLETE
    assert acc.VARIANT_FAMILY in str(row["families_missing"])


def test_judge_says_which_reference_it_compared_against(robustness_project):
    out = syn.judge(syn.load_all(2020))
    assert out["baseline_source"].notna().all()
    assert "results" in str(out["baseline_source"].iloc[0])


# ---------------------------------------------------------------------------
# build：落盘位置与 manifest
# ---------------------------------------------------------------------------

def test_build_writes_only_under_the_robustness_directory_with_a_manifest(
    robustness_project
):
    paths = syn.build(2020)
    rob = robustness_project["robustness_dir"]
    for path in paths.values():
        assert os.path.abspath(path).startswith(os.path.abspath(rob))
    assert os.path.exists(os.path.join(rob, "synthesis.parquet"))
    manifest_path = os.path.join(syn.manifest_dir(2020), "manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["params"]["baseline_source"]
    assert manifest["params"]["influence_threshold"] == syn.DEFAULT_INFLUENCE_THRESHOLD
    assert manifest["counts"]["quantities"] == len(harness.QUANTITIES)


def test_build_output_is_readable_with_explicit_columns(robustness_project):
    paths = syn.build(2020)
    judged = pd.read_parquet(paths["synthesis"], engine="pyarrow",
                             columns=["quantity", "completeness", "baseline_source"])
    assert set(judged["quantity"]) == set(harness.QUANTITIES)


# ---------------------------------------------------------------------------
# 复核第一轮修复
# ---------------------------------------------------------------------------

# 加上 M2 的基线，好让 M2 上的相对偏移真的算得出来——否则"M2 有没有漏进
# 准则三"这件事无从检验（偏移会因为基线是 NaN 而恒为 NaN）。
BASELINE_WITH_M2 = dict(BASELINE)
BASELINE_WITH_M2[("entry_public", "M2")] = 0.10


def test_exclude_top_k_is_not_classified_as_a_single_account():
    """`{domain}_exclude_top10` 一次剔十个账号，归不到"单个账号"名下"""
    assert syn.influence_unit(acc.VARIANT_FAMILY, "loo_public_rank01_9999") \
        == syn.UNIT_ACCOUNT
    for label in ("public_exclude_top1", "public_exclude_top5",
                  "celebrity_exclude_top10"):
        assert syn.influence_unit(acc.VARIANT_FAMILY, label) == syn.UNIT_ACCOUNT_SET


def test_a_ten_account_deletion_cannot_produce_the_single_account_number():
    """§13.5 的核心问题是"一个账号撑起来的吗"，答案不能由删十个账号得出"""
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY, BASELINE),
        variant_rows(acc.VARIANT_FAMILY, "loo_celebrity_rank01_9999", {
            ("entry_celebrity", "M0"): -0.19, ("entry_celebrity", "M1"): -0.15}),
        variant_rows(acc.VARIANT_FAMILY, "celebrity_exclude_top10", {
            ("entry_celebrity", "M0"): -0.01, ("entry_celebrity", "M1"): -0.01}),
    ]
    df = pd.concat(frames, ignore_index=True)
    row = syn.judge(df)
    row = row[row["quantity"] == "entry_celebrity"].iloc[0]
    assert row["worst_single_account_variant"] == "loo_celebrity_rank01_9999"
    assert float(row["max_relative_shift_single_account"]) < 0.5
    # 删十个账号的那次偏移仍然被报告，只是记在"一批账号"名下
    infl = syn.influence_summary(df)
    sets = infl[(infl["quantity"] == "entry_celebrity")
                & (infl["influence_unit"] == syn.UNIT_ACCOUNT_SET)]
    assert "celebrity_exclude_top10" in set(sets["worst_variant_label"])
    assert float(sets["max_abs_relative_shift"].max()) > 0.9


def _frame_with_an_M2_only_variant():
    layers = ("M0", "M1", "M2")
    return pd.concat([
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY, BASELINE_WITH_M2,
                     layers=layers),
        variant_rows(smp.VARIANT_FAMILY_EXTREME, "trim_pooled_top1pct", {
            ("entry_public", "M0"): 0.11, ("entry_public", "M1"): 0.09},
            layers=layers),
        # 只在 M2 上估出来的 §13.9 变体，相对偏移巨大（受限样本）
        variant_rows(smp.VARIANT_FAMILY_USER_TYPE, "verified_individuals_only",
                     {("entry_public", "M2"): 1.00}, layers=layers),
    ], ignore_index=True)


def test_an_M2_variant_cannot_contribute_to_criterion_three_of_an_M0_M1_row():
    """M2 是受限样本（§11.4），它的偏移不能混进一行其余全是 M0/M1 的记录"""
    df = _frame_with_an_M2_only_variant()
    # 影响力表本身照常逐层记录 M2 的偏移（9.0 = |1.00 - 0.10| / 0.10）
    infl = syn.influence_summary(df)
    m2 = infl[(infl["quantity"] == "entry_public") & (infl["model"] == "M2")
              & (infl["influence_unit"] == syn.UNIT_USER_GROUP)].iloc[0]
    assert float(m2["max_abs_relative_shift"]) == pytest.approx(9.0)
    # judge 那一行必须只看 M0/M1
    row = syn.judge(df)
    row = row[row["quantity"] == "entry_public"].iloc[0]
    assert row["worst_user_group_layer"] in ("M0", "M1")
    assert float(row["max_relative_shift_user_group"]) == pytest.approx(0.125)
    assert int(row["n_units_exceeding_threshold"]) == 0


def test_judge_points_at_where_the_M2_answer_lives_without_making_it_a_criterion():
    df = _frame_with_an_M2_only_variant()
    row = syn.judge(df)
    row = row[row["quantity"] == "entry_public"].iloc[0]
    assert int(row["n_M2_variants"]) == 1
    assert "synthesis_direction.parquet" in str(row["m2_pointer"])
    assert "not_comparable" in str(row["m2_pointer"]).replace("NOT_", "not_")
    # 指路牌不是准则：M2 没有被折进任何一个跨族一致率
    assert "M2" not in str(row["direction_share_M0"])
    assert not [c for c in syn.judge(df).columns if c.endswith("_M2")]


def test_n_agree_is_nan_when_the_comparison_was_never_possible():
    """参照拿不到时，n_agree 写 0 会读成"全体不一致"——必须是 NaN"""
    df = variant_rows("vocabulary", "keep0.8_rep0", {("entry_public", "M0"): 0.11})
    out = syn.direction_consistency(df)
    row = out[(out["quantity"] == "entry_public") & (out["model"] == "M0")].iloc[0]
    assert int(row["n_live"]) == 1
    assert np.isnan(float(row["n_agree"]))
    assert np.isnan(float(row["share_agree"]))
    assert bool(row["incompletely_tested"])


def test_exceeds_threshold_is_none_when_the_comparison_was_never_possible():
    df = variant_rows(acc.VARIANT_FAMILY, "loo_public_rank01_1",
                      {("entry_public", "M0"): 0.11})
    out = syn.influence_summary(df)
    row = out[(out["quantity"] == "entry_public") & (out["model"] == "M0")].iloc[0]
    assert row["exceeds_threshold"] is None
    assert np.isnan(float(row["n_exceeding"]))


def test_n_variant_pairs_uses_the_same_pairs_as_the_quantiles():
    """M0 恰为 0 的配对算不出衰减；计数与分位数不能各用各的口径"""
    frames = [
        variant_rows(syn.BASELINE_FAMILY, syn.BASELINE_FAMILY, BASELINE),
        variant_rows("vocabulary", "keep0.8_rep0",
                     {("entry_public", "M0"): 0.0, ("entry_public", "M1"): 0.0}),
        variant_rows("vocabulary", "keep0.8_rep1",
                     {("entry_public", "M0"): 0.10, ("entry_public", "M1"): 0.08}),
    ]
    out = syn.activity_attenuation(pd.concat(frames, ignore_index=True))
    row = out[out["quantity"] == "entry_public"].iloc[0]
    assert int(row["n_matched_pairs"]) == 2
    assert int(row["n_variant_pairs"]) == 1
    assert int(row["n_pairs_attenuation_undefined"]) == 1
    assert float(row["attenuation_median"]) == pytest.approx(0.20)


def test_the_recomputed_baseline_source_names_the_function_the_code_calls():
    """溯源的人会 grep 这个字符串，它必须指向真正被调用的函数"""
    assert "estimate_all" in syn.BASELINE_SOURCE_RECOMPUTED
    assert "harness.baseline" not in syn.BASELINE_SOURCE_RECOMPUTED


def test_a_result_file_outside_the_fdr_scope_is_reported_not_silently_skipped(
    robustness_project, capsys
):
    directory = os.path.join(robustness_project["out_dir"], "results")
    extra = os.path.join(directory, "models_brand_new_secondary.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(extra, engine="pyarrow", index=False)
    assert "models_brand_new_secondary.parquet" in syn.unlisted_result_files(directory)
    syn.build(2020)
    assert "models_brand_new_secondary.parquet" in capsys.readouterr().out
    with open(os.path.join(syn.manifest_dir(2020), "manifest.json"),
              encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert "models_brand_new_secondary.parquet" in \
        manifest["params"]["result_files_not_in_fdr_scope"]


# ---------------------------------------------------------------------------
# SLURM 作业
# ---------------------------------------------------------------------------

def test_slurm_array_job_runs_one_task_per_family():
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "slurm", "run_robustness.slurm")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert "#SBATCH --array=" in text
    for module in ("vocabulary", "accounts", "samples", "measures",
                   "context_sample"):
        assert "gender_domain.robustness.{}".format(module) in text
    # 降低成本的旋钮必须写在头部注释里
    assert "top_n" in text and "n_replicates" in text
