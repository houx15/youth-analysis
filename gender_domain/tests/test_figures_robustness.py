"""
gender_domain.figures_robustness 的单元测试（§12.7 稳健性附录图）。

与 test_figures.py 同一取舍：图不测长相，测的是这层最容易骗人的四件事——

1. 缺数据时是否当场报错，而不是画一张空坐标轴；
2. 重跑追加出来的重复变体行会不会被数两遍（族表是纯追加写的）；
3. "没测过"的格子会不会被画成"一致率 0%"；
4. 跑不出估计的变体会不会静默消失，而不是在图上被数出来。

合成数据全部在 tmp_path 里，matplotlib 用 Agg 无头后端。
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gender_domain import config
from gender_domain import figures_robustness as fr
from gender_domain.robustness import harness


QUANTITIES = list(fr.QUANTITY_ORDER)


def _capture_saved_figure(monkeypatch):
    """截住 _save_fig 里的 Figure —— 它保存完就 plt.close，事后拿不到"""
    captured = {}

    real_save = fr.fg._save_fig

    def spy(fig, name, fig_dir):
        captured["fig"] = fig
        captured["name"] = name
        return real_save(fig, name, fig_dir)

    monkeypatch.setattr(fr.fg, "_save_fig", spy)
    return captured


def _texts(fig):
    out = []
    for ax in fig.get_axes():
        out.extend(t.get_text() for t in ax.texts)
        if ax.get_title():
            out.append(ax.get_title())
    return out


# ---------------------------------------------------------------------------
# 合成 robustness 目录
# ---------------------------------------------------------------------------

def _spec_curve_rows(quantity, model, estimates, families=None):
    families = families or ["vocabulary"] * len(estimates)
    baseline = 0.05 if not quantity.startswith(("entry_celebrity",
                                                "topical_celebrity")) else -0.03
    rows = []
    for i, (estimate, family) in enumerate(zip(estimates, families)):
        agrees = (np.sign(estimate) == np.sign(baseline)
                  if estimate == estimate else None)
        rows.append({
            "quantity": quantity, "model": model, "variant_family": family,
            "variant_label": "{}_{}".format(family, i), "replicate": i,
            "seed": None, "influence_unit": "term_set", "is_baseline": False,
            "estimate": estimate, "se": 0.01,
            "ci_low": estimate - 0.02 if estimate == estimate else np.nan,
            "ci_high": estimate + 0.02 if estimate == estimate else np.nan,
            "estimate_available": estimate == estimate, "rank": i,
            "n_specifications": len(estimates),
            "crosses_zero": False, "agrees_with_baseline": agrees,
            "relative_shift": 0.1, "baseline_estimate": baseline,
            "baseline_source": "test", "baseline_run_id": "test",
            "baseline_git_sha": "test", "baseline_git_dirty": "false",
            "note": None,
        })
    return rows


@pytest.fixture()
def robustness_figures_project(tmp_path, monkeypatch):
    """写出七张图各自需要的最小输入，并把 config.OUTPUT_DIR 指过去"""
    out_dir = str(tmp_path / "analysis_data")
    rob = os.path.join(out_dir, "robustness")
    os.makedirs(rob, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)

    # --- synthesis：六个量的基线 ---
    baselines = {"entry_public": 0.036, "entry_celebrity": -0.015,
                 "topical_public": 0.098, "topical_celebrity": -0.035,
                 "did_entry": 0.052, "did_topical": 0.153}
    pd.DataFrame([{
        "quantity": q,
        "baseline_estimate_M0": value * 1.2,
        "baseline_estimate_M1": value,
    } for q, value in baselines.items()]).to_parquet(
        os.path.join(rob, "synthesis.parquet"), engine="pyarrow", index=False)

    # --- specification curve：每个量 4 个变体，其中 entry_public 有 1 个 NaN ---
    rows = []
    for quantity in QUANTITIES:
        values = [0.02, 0.04, 0.05, 0.06]
        if quantity == "entry_public":
            values = [0.02, 0.04, np.nan, 0.06]
        if quantity in ("entry_celebrity", "topical_celebrity"):
            values = [-0.05, -0.03, -0.02, 0.01]
        rows.extend(_spec_curve_rows(quantity, "M1", values))
    pd.DataFrame(rows).to_parquet(
        os.path.join(rob, "synthesis_specification_curve.parquet"),
        engine="pyarrow", index=False)

    # --- 逐族一致率：post_types 对 entry_* 完全没测（share_agree = NaN）---
    by_family = []
    for quantity in QUANTITIES:
        for family, share, n_live in (("vocabulary", 1.0, 200),
                                      ("accounts", 1.0, 122),
                                      ("post_types", np.nan, 0)):
            if family == "post_types" and quantity.startswith("topical"):
                share, n_live = 0.6667, 3
            by_family.append({
                "quantity": quantity, "model": "M1", "variant_family": family,
                "share_agree": share, "n_live": n_live,
                "share_of_pooled_live_rows": 0.3,
                "estimate_min": -0.1, "estimate_max": 0.2,
                "estimate_median": 0.05,
            })
    pd.DataFrame(by_family).to_parquet(
        os.path.join(rob, "synthesis_direction_by_family.parquet"),
        engine="pyarrow", index=False)

    # --- 影响力：month 这一类对每个量都没测 ---
    influence = []
    for quantity in QUANTITIES:
        for unit in fr.INFLUENCE_UNITS:
            shift = np.nan if unit == "month" else 0.3
            if unit == "user_group":
                shift = 0.9
            influence.append({
                "quantity": quantity, "model": "M1", "influence_unit": unit,
                "baseline_estimate": baselines[quantity], "n_live": 10,
                "max_abs_relative_shift": shift,
                "worst_variant_family": "user_type",
                "worst_variant_label": "ordinary_users_only",
                "worst_estimate": 0.01, "threshold": 0.5,
                "n_exceeding": 1.0, "exceeds_threshold": shift > 0.5
                if shift == shift else False,
            })
    pd.DataFrame(influence).to_parquet(
        os.path.join(rob, "synthesis_influence.parquet"),
        engine="pyarrow", index=False)

    # --- 衰减 ---
    pd.DataFrame([{
        "quantity": q, "baseline_estimate_M0": value * 1.2,
        "baseline_estimate_M1": value, "baseline_attenuation": 0.167,
        "attenuation_p10": 0.10, "attenuation_p90": 0.22,
        "attenuation_median": 0.167,
    } for q, value in baselines.items()]).to_parquet(
        os.path.join(rob, "synthesis_attenuation.parquet"),
        engine="pyarrow", index=False)

    # --- 词表族：3 个 replicate；来源进入的三个量退化成同一个值 ---
    vocab_rows = []
    for quantity in QUANTITIES:
        meta = harness.QUANTITY_META[quantity]
        degenerate = quantity in ("entry_public", "entry_celebrity", "did_entry")
        for rep in range(3):
            value = (baselines[quantity] if degenerate
                     else baselines[quantity] - 0.001 * rep)
            vocab_rows.append({
                "outcome": meta["outcome"], "domain": meta["domain"],
                "model": "M1", "term": meta["term"], "estimate": value,
                "se": 0.01, "ci_low": value - 0.02, "ci_high": value + 0.02,
                "scale": meta["scale"], "n_obs": 1000, "n_dropped": 0,
                "drop_reason": None, "note": None,
                "variant_family": "vocabulary",
                "variant_label": "keep0.8_rep{}".format(rep),
                "replicate": rep, "seed": rep,
            })
    pd.DataFrame(vocab_rows).to_parquet(
        os.path.join(rob, "vocabulary.parquet"), engine="pyarrow", index=False)

    # --- 账号族：5 个 leave-one-out ---
    acc_rows = []
    for quantity in ("entry_public", "entry_celebrity", "did_entry"):
        meta = harness.QUANTITY_META[quantity]
        for i in range(5):
            value = baselines[quantity] + 0.001 * i
            acc_rows.append({
                "outcome": meta["outcome"], "domain": meta["domain"],
                "model": "M1", "term": meta["term"], "estimate": value,
                "se": 0.01, "ci_low": value - 0.02, "ci_high": value + 0.02,
                "scale": meta["scale"], "n_obs": 1000, "n_dropped": 0,
                "drop_reason": None, "note": None,
                "variant_family": "accounts",
                "variant_label": "loo_public_rank{:02d}_x".format(i),
                "replicate": None, "seed": None,
            })
    pd.DataFrame(acc_rows).to_parquet(
        os.path.join(rob, "accounts.parquet"), engine="pyarrow", index=False)

    # --- 账号集中度 ---
    conc = []
    for domain, n_accounts in (("public", 47), ("celebrity", 216)):
        for gender in ("all", "m", "f"):
            for k, share in ((1, 0.33), (5, 0.88), (10, 0.97)):
                conc.append({"domain": domain, "gender": gender, "k": k,
                             "n_accounts": n_accounts, "n_events": 1000,
                             "n_users_entered": 500, "top_k_share": share,
                             "top_k_share_pooled_ranking": share,
                             "top_accounts": "a;b"})
    pd.DataFrame(conc).to_parquet(
        os.path.join(rob, "account_concentration.parquet"),
        engine="pyarrow", index=False)

    # --- FDR：2 个预先设定量 + 4 个次要分析（其中 1 个没有 p 值）---
    fdr = []
    for i, (p, prespecified) in enumerate([(np.nan, True), (0.001, True),
                                           (1e-30, False), (0.001, False),
                                           (0.9, False), (np.nan, False)]):
        fdr.append({
            "outcome": "o{}".format(i), "domain": "public", "model": "M1",
            "term": "gender_male", "estimate": 0.1, "se": 0.01,
            "ci_low": 0.08, "ci_high": 0.12, "scale": "probability",
            "n_obs": 1000, "n_dropped": 0, "drop_reason": None, "note": None,
            "source_file": "x.parquet", "is_prespecified": prespecified,
            "p_value": p, "p_value_source": "wald",
            "q_value": p if p == p else np.nan,
            "fdr_rejected": (p < 0.05) if (p == p and not prespecified) else False,
            "fdr_alpha": 0.05, "fdr_n_tested": 3,
        })
    pd.DataFrame(fdr).to_parquet(
        os.path.join(rob, "synthesis_fdr.parquet"), engine="pyarrow", index=False)

    return rob


# ---------------------------------------------------------------------------
# 缺文件：报错，不画空图
# ---------------------------------------------------------------------------

def test_a_missing_input_raises_instead_of_drawing_an_empty_panel(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "analysis_data"))
    with pytest.raises(FileNotFoundError) as excinfo:
        fr.figR1_specification_curve(2020, fig_dir=str(tmp_path / "figures"))
    # 报错信息必须写出"下一步该做什么"，否则读者只知道文件不在
    assert "synthesis" in str(excinfo.value)


@pytest.mark.parametrize("name,builder", list(fr.FIGURES))
def test_every_figure_renders_from_the_synthetic_directory(
    name, builder, robustness_figures_project, tmp_path
):
    path = builder(2020, fig_dir=str(tmp_path / "figures"))
    assert os.path.exists(path)
    assert name in os.path.basename(path)
    plt.close("all")


# ---------------------------------------------------------------------------
# 重跑追加出来的重复行：不能被数两遍
# ---------------------------------------------------------------------------

def test_a_rerun_family_file_is_not_counted_twice(robustness_figures_project,
                                                  tmp_path, monkeypatch):
    """族表是纯追加写的，重跑一族会让它整份翻倍；图上的点数必须不变

    figR5/figR6 直接读族表，不经过综合层。综合层已经去重，这一层如果不去，
    同一批数据会在两处画出不同的点数——而两处都不会报错。
    """
    path = os.path.join(robustness_figures_project, "vocabulary.parquet")
    frame = pd.read_parquet(path, engine="pyarrow")
    pd.concat([frame, frame], ignore_index=True).to_parquet(
        path, engine="pyarrow", index=False)

    captured = _capture_saved_figure(monkeypatch)
    fr.figR5_vocabulary_resampling(2020, fig_dir=str(tmp_path / "figures"))
    titles = " ".join(_texts(captured["fig"]))
    # 合成数据是 3 个 replicate；翻倍后如果不去重会写成 6
    assert "3 replicates" in titles
    assert "6 replicates" not in titles
    plt.close("all")


def test_dedup_variants_keeps_the_last_copy():
    frame = pd.DataFrame([
        {"variant_family": "v", "variant_label": "a", "replicate": 0,
         "seed": None, "outcome": "o", "domain": "public", "model": "M1",
         "term": "t", "estimate": 0.1},
        {"variant_family": "v", "variant_label": "a", "replicate": 0,
         "seed": None, "outcome": "o", "domain": "public", "model": "M1",
         "term": "t", "estimate": 0.2},
    ])
    out = fr.dedup_variants(frame)
    assert len(out) == 1
    assert out["estimate"].iloc[0] == 0.2


# ---------------------------------------------------------------------------
# "没测过" ≠ "一致率 0%"
# ---------------------------------------------------------------------------

def test_an_untested_family_is_labelled_not_tested_not_zero_percent(
    robustness_figures_project, tmp_path, monkeypatch
):
    """share_agree 是 NaN 的格子必须写成 not tested，且不能借配色表的颜色

    0% 与"没测"在这张热力图上是两个相反的事实：前者说"换了这个口径方向就
    反了"，后者说"这个口径根本不适用于这个量"。配色表最红的一端是 0%，
    NaN 借了它就等于把"没测"画成最坏的结果。
    """
    captured = _capture_saved_figure(monkeypatch)
    fr.figR2_direction_by_family(2020, fig_dir=str(tmp_path / "figures"))
    fig = captured["fig"]
    texts = _texts(fig)
    assert any("not\ntested" in t for t in texts)
    assert not any(t.startswith("0%") for t in texts)
    plt.close("all")


def test_an_untested_influence_unit_is_labelled_not_a_zero_bar(
    robustness_figures_project, tmp_path, monkeypatch
):
    """max_abs_relative_shift 是 NaN 的单位画成 0 长度的柱子 + "not tested" 标注

    0 长度的柱子单独看会读成"这一类单位完全推不动结论"，那是最乐观的读法，
    与"没测过"正好相反，所以必须有标注。
    """
    captured = _capture_saved_figure(monkeypatch)
    fr.figR3_influence(2020, fig_dir=str(tmp_path / "figures"))
    texts = _texts(captured["fig"])
    assert sum(1 for t in texts if t == "not tested") == len(QUANTITIES)
    plt.close("all")


# ---------------------------------------------------------------------------
# 跑不出估计的变体：必须在图上被数出来
# ---------------------------------------------------------------------------

def test_specifications_without_an_estimate_are_counted_on_the_figure(
    robustness_figures_project, tmp_path, monkeypatch
):
    """NaN 的变体在排序曲线上画不出点，图上必须写出缺了几个"""
    captured = _capture_saved_figure(monkeypatch)
    fr.figR1_specification_curve(2020, fig_dir=str(tmp_path / "figures"))
    texts = _texts(captured["fig"])
    assert any("produced no estimate" in t for t in texts)
    # 合成数据里只有 entry_public 有 1 个 NaN
    assert sum(1 for t in texts if "produced no estimate" in t) == 1
    plt.close("all")


def test_the_specification_count_in_the_title_excludes_nan_rows(
    robustness_figures_project, tmp_path, monkeypatch
):
    """标题里的 n 是"画出来的点数"，不是"变体总数"——分母必须与图上一致"""
    captured = _capture_saved_figure(monkeypatch)
    fr.figR1_specification_curve(2020, fig_dir=str(tmp_path / "figures"))
    titles = [ax.get_title() for ax in captured["fig"].get_axes()]
    # entry_public：4 个变体，1 个 NaN -> 3 个 specification
    assert any("Source entry public affairs" in t and "3 specifications" in t
               for t in titles)
    plt.close("all")


# ---------------------------------------------------------------------------
# 退化分布：不能画成直方图
# ---------------------------------------------------------------------------

def test_a_degenerate_replicate_distribution_says_so_instead_of_drawing_a_bar(
    robustness_figures_project, tmp_path, monkeypatch
):
    """词表不进来源进入的测量，200 个 replicate 完全相同

    直方图会把它画成一根柱子，看上去像"分布很窄"——那是一个关于精度的说法，
    而事实是这个量根本不受词表影响。两者必须区分开。
    """
    captured = _capture_saved_figure(monkeypatch)
    fr.figR5_vocabulary_resampling(2020, fig_dir=str(tmp_path / "figures"))
    texts = _texts(captured["fig"])
    assert any("replicates identical" in t for t in texts)
    plt.close("all")


# ---------------------------------------------------------------------------
# FDR：预先设定的六个量绝不进校正
# ---------------------------------------------------------------------------

def test_prespecified_quantities_stay_out_of_the_bh_panel(
    robustness_figures_project, tmp_path, monkeypatch
):
    """把预先设定量混进 BH 图会让校正看起来更严，实际改变的是别人的 q 值"""
    captured = _capture_saved_figure(monkeypatch)
    fr.figR7_fdr(2020, fig_dir=str(tmp_path / "figures"))
    titles = [ax.get_title() for ax in captured["fig"].get_axes()]
    # 合成数据：4 个次要分析，其中 1 个没有 p 值 -> 3 个进 BH
    assert any("3 secondary tests" in t for t in titles)
    plt.close("all")


def test_the_p_value_axis_floor_is_announced_not_silent(
    robustness_figures_project, tmp_path, monkeypatch
):
    """被地板截住的点必须数出来：地板是画法，不是数据"""
    captured = _capture_saved_figure(monkeypatch)
    fr.figR7_fdr(2020, fig_dir=str(tmp_path / "figures"))
    texts = _texts(captured["fig"])
    # 合成数据里有一个 p=1e-30，低于 1e-20 的地板
    assert any("drawn on the floor" in t for t in texts)
    plt.close("all")
