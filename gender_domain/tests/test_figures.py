"""
gender_domain.figures 的单元测试（§12 正文图与附录图）。

图只测数据处理，不测长相：一张 PDF 好不好看无法在单测里断言，但
"读不到数据时是否明确报错""domain='both' 的差中差有没有被 groupby
悄悄丢掉""拟合失败的 NaN 行有没有被静默跳过""异常月份的判定规则是不是
事前固定的"这四件事都可以断言，而它们恰好是这层最容易出错的地方。

所有测试都跑在合成的 figure_data 目录上（本地没有真实结果），
matplotlib 用 Agg 无头后端。
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gender_domain import figures as fg
from gender_domain import stats_utils as su


SCHEMA = list(su.RESULT_SCHEMA)


def _row(**kwargs):
    """按共享 schema 补齐一行，缺省字段填 None"""
    row = {col: None for col in SCHEMA}
    row.update(kwargs)
    return row


# ---------------------------------------------------------------------------
# 合成 figure_data
# ---------------------------------------------------------------------------

def _table2():
    rows = []
    specs = [
        ("source_entered", "public", "probability", "all_same_gender_users", 0.30, 0.22),
        ("source_entered", "celebrity", "probability", "all_same_gender_users", 0.18, 0.35),
        ("topical_share", "public", "proportion", "users_with_expressive_posts", 0.09, 0.05),
        ("topical_share", "celebrity", "proportion", "users_with_expressive_posts", 0.03, 0.11),
    ]
    for outcome, domain, scale, denom, male, female in specs:
        # source_entered 的两个分母口径是两个 model 变体（describe.py 里
        # 折进 model 列，见 _ENTERED_DENOM_SPECS）；其余结果变量分母单一，
        # model 仍是 "raw"。合成数据必须照着真实模块的形状写，否则测试
        # 守的是一份现实中不存在的表。
        model = fg.PRIMARY_TABLE2_MODEL[outcome]
        for term, est, n in (("male", male, 52000), ("female", female, 174000)):
            rows.append(_row(outcome=outcome, domain=domain, model=model, term=term,
                             estimate=est, ci_low=est - 0.01, ci_high=est + 0.01,
                             scale=scale, n_obs=n, n_dropped=0, note="wilson"))
        rows.append(_row(outcome=outcome, domain=domain, model=model, term="gap_pp",
                         estimate=male - female, ci_low=male - female - 0.01,
                         ci_high=male - female + 0.01, scale=scale,
                         n_obs=226000, n_dropped=0, note="newcombe"))
    frame = pd.DataFrame(rows)
    frame["denominator"] = "all_same_gender_users"
    frame.loc[frame["outcome"] == "topical_share", "denominator"] = \
        "users_with_expressive_posts"
    # 再补一组非主口径分母的行，确认图会按分母筛选而不是把两套口径叠在一起。
    # 它们与主口径行只差 model/denominator，正好也守住"表 2 的四元组唯一"
    extra = frame[frame["outcome"] == "source_entered"].copy()
    extra["denominator"] = "retweeters_only"
    extra["model"] = "raw/retweeters_only"
    extra["estimate"] = extra["estimate"] * 1.4
    return pd.concat([frame, extra], ignore_index=True)


def _models_entry():
    rows = []
    for domain, base in (("public", 0.08), ("celebrity", -0.17)):
        for i, model in enumerate(("M0", "M1", "M2")):
            est = base * (1 - 0.25 * i)
            rows.append(_row(outcome="source_entered", domain=domain, model=model,
                             term="gender_male", estimate=est, se=0.004,
                             ci_low=est - 0.008, ci_high=est + 0.008,
                             scale="probability", n_obs=226000, n_dropped=0))
            rows.append(_row(outcome="source_entered", domain=domain, model=model,
                             term="gender_male_odds_ratio", estimate=1.4, se=0.02,
                             ci_low=1.3, ci_high=1.5, scale="odds_ratio",
                             n_obs=226000, n_dropped=0, note="log_scale"))
    return pd.DataFrame(rows)


def _models_share(with_nan=True):
    rows = []
    for outcome in ("source_share", "topical_share"):
        for domain, base in (("public", 0.04), ("celebrity", -0.08)):
            for i, model in enumerate(("M0", "M1", "M2")):
                est = base * (1 - 0.3 * i)
                rows.append(_row(outcome=outcome, domain=domain, model=model,
                                 term="gender_male", estimate=est, se=0.002,
                                 ci_low=est - 0.004, ci_high=est + 0.004,
                                 scale="proportion", n_obs=180000, n_dropped=46000,
                                 drop_reason="no_expressive_posts=46000"))
    frame = pd.DataFrame(rows)
    if with_nan:
        # M2 在 celebrity/topical_share 上"拟合失败"：估计值 NaN，note 写明原因。
        # 图必须把这一行显式标出来，而不是当它不存在。
        mask = ((frame["outcome"] == "topical_share")
                & (frame["domain"] == "celebrity")
                & (frame["model"] == "M2"))
        frame.loc[mask, ["estimate", "se", "ci_low", "ci_high"]] = np.nan
        frame.loc[mask, "note"] = "not_converged"
    return frame


def _models_persistence():
    rows = []
    for domain, base in (("public", 0.03), ("celebrity", -0.05)):
        for i, model in enumerate(("M0", "M1", "M2")):
            est = base * (1 - 0.2 * i)
            rows.append(_row(outcome="source_month_share", domain=domain, model=model,
                             term="gender_male", estimate=est, se=0.002,
                             ci_low=est - 0.004, ci_high=est + 0.004,
                             scale="proportion", n_obs=200000, n_dropped=26000,
                             note="months_weighted_fit+ame_averaged_over_users"))
    return pd.DataFrame(rows)


def _interaction():
    """交互结果表：四个预测格子 + 两个域内性别差 + 差中差 + 交互系数

    四个格子与两个差、差中差之间的恒等式在真实模块里被守到 1e-10，
    这里的合成数字也照着满足：格子之差 == 域内性别差，两者再相减 == DiD。
    """
    rows = []
    specs = (
        # outcome, scale, 四个格子（男公共、女公共、男娱乐、女娱乐）
        ("source_entered", "probability", (0.30, 0.22, 0.18, 0.35)),
        ("topical_share", "proportion", (0.09, 0.05, 0.03, 0.11)),
    )
    for outcome, scale, cells in specs:
        male_public, female_public, male_celeb, female_celeb = cells
        gap_public_base = male_public - female_public
        gap_celeb_base = male_celeb - female_celeb
        for i, model in enumerate(("M0", "M1", "M2")):
            # 逐层缩水：女性格子不动，男性格子按 (1-0.2i) 向女性靠拢，
            # 于是每一层内部"格子之差 == 域内性别差 == 差中差的两半"始终成立，
            # 与真实模块守到 1e-10 的恒等式同构
            shrink = 1 - 0.2 * i
            gap_public = gap_public_base * shrink
            gap_celeb = gap_celeb_base * shrink
            male_public = female_public + gap_public
            male_celeb = female_celeb + gap_celeb
            did = gap_public - gap_celeb
            rows.append(_row(outcome=outcome, domain="both", model=model,
                             term="gender_male_x_public_did", estimate=did, se=0.006,
                             ci_low=did - 0.012, ci_high=did + 0.012, scale=scale,
                             n_obs=226000, n_dropped=0,
                             note="gee_exchangeable_cluster_user_2rows_per_user"))
            rows.append(_row(outcome=outcome, domain="public", model=model,
                             term="gender_male_gap_public", estimate=gap_public,
                             se=0.004, ci_low=gap_public - 0.008,
                             ci_high=gap_public + 0.008, scale=scale,
                             n_obs=226000, n_dropped=0))
            rows.append(_row(outcome=outcome, domain="celebrity", model=model,
                             term="gender_male_gap_celebrity", estimate=gap_celeb,
                             se=0.004, ci_low=gap_celeb - 0.008,
                             ci_high=gap_celeb + 0.008, scale=scale,
                             n_obs=226000, n_dropped=0))
            rows.append(_row(outcome=outcome, domain="both", model=model,
                             term="gender_male_x_public_coef", estimate=1.1, se=0.05,
                             ci_low=1.0, ci_high=1.2, scale="log_odds",
                             n_obs=226000, n_dropped=0, note="not_the_headline"))
            for term, domain, value in (
                    ("pred_male_public", "public", male_public),
                    ("pred_female_public", "public", female_public),
                    ("pred_male_celebrity", "celebrity", male_celeb),
                    ("pred_female_celebrity", "celebrity", female_celeb),
            ):
                rows.append(_row(outcome=outcome, domain=domain, model=model,
                                 term=term, estimate=value, se=0.003,
                                 ci_low=value - 0.006, ci_high=value + 0.006,
                                 scale=scale, n_obs=226000, n_dropped=0,
                                 note="gee_exchangeable_cluster_user_2rows_per_user"
                                      "+counterfactual_cell"))
    return pd.DataFrame(rows)


def _interaction_with_flagged_cells():
    """把 M1 的三个格子分别改成：区间被截断、区间不可用、整格 NaN"""
    frame = _interaction()
    base = (frame["outcome"] == "source_entered") & (frame["model"] == "M1")

    clipped = base & (frame["term"] == "pred_male_public")
    frame.loc[clipped, "ci_high"] = 1.0
    frame.loc[clipped, "note"] = "counterfactual_cell+" \
        "pred_ci_clipped_to_unit_interval"

    unavailable = base & (frame["term"] == "pred_female_public")
    frame.loc[unavailable, ["se", "ci_low", "ci_high"]] = np.nan
    frame.loc[unavailable, "note"] = "counterfactual_cell+pred_ci_unavailable"

    absent = base & (frame["term"] == "pred_male_celebrity")
    frame.loc[absent, ["estimate", "se", "ci_low", "ci_high"]] = np.nan
    frame.loc[absent, "note"] = "counterfactual_cell+pred_ci_unavailable"
    return frame


def _decomposition():
    rows = []
    shares = {
        ("public", "male"): (0.55, 0.20, 0.15, 0.10),
        ("public", "female"): (0.62, 0.16, 0.14, 0.08),
        ("celebrity", "male"): (0.70, 0.12, 0.11, 0.07),
        ("celebrity", "female"): (0.45, 0.22, 0.18, 0.15),
    }
    for (domain, gender), values in shares.items():
        for ptype, value in zip(
                ("neither", "source_only", "content_only", "both"), values):
            rows.append(_row(outcome="participation_type", domain=domain,
                             model="any_hit", term="{}_{}".format(ptype, gender),
                             estimate=value, ci_low=value - 0.005,
                             ci_high=value + 0.005, scale="probability",
                             n_obs=52000 if gender == "male" else 174000,
                             n_dropped=0, note="content_threshold=any_hit+wilson"))
    # 另一套阈值变体，图必须只画主口径 any_hit，不能把变体叠上去
    variant = pd.DataFrame(rows).copy()
    variant["model"] = "share_ge_0.1"
    return pd.concat([pd.DataFrame(rows), variant], ignore_index=True)


def _combination():
    """多项 logit 结果：四类的性别 AME + 八行各性别的模型预测概率

    预测概率与 AME 满足真实模块断言的恒等式 pred_male[c] - pred_female[c]
    == AME[c]：女性各类别占比固定，男性等于女性加上该层的 AME。
    """
    rows = []
    ames = {"neither": -0.06, "public_only": 0.09,
            "celebrity_only": -0.05, "both": 0.02}
    female = {"neither": 0.22, "public_only": 0.13,
              "celebrity_only": 0.45, "both": 0.20}
    for i, model in enumerate(("M0", "M1", "M2")):
        for category, ame in ames.items():
            est = ame * (1 - 0.2 * i)
            rows.append(_row(outcome="source_combo", domain="both", model=model,
                             term="gender_male_ame_{}".format(category), estimate=est,
                             se=0.003, ci_low=est - 0.006, ci_high=est + 0.006,
                             scale="probability", n_obs=226000, n_dropped=0,
                             note="mnlogit_base:neither"))
            for gender in ("male", "female"):
                value = female[category] + (est if gender == "male" else 0.0)
                rows.append(_row(
                    outcome="source_combo", domain="both", model=model,
                    term="pred_{}_{}".format(gender, category), estimate=value,
                    se=0.002, ci_low=value - 0.004, ci_high=value + 0.004,
                    scale="probability", n_obs=226000, n_dropped=0,
                    note="mnlogit_base:neither+counterfactual_predicted_probability"))
        rows.append(_row(outcome="source_combo", domain="both", model=model,
                         term="gender_male_ame_sum", estimate=0.0, scale="probability",
                         n_obs=226000, n_dropped=0, note="internal_consistency_check"))
    return pd.DataFrame(rows)


def _combination_with_flagged_cells():
    """把 M1 的三个预测格子改成：区间被截断、区间不可用、整格 NaN"""
    frame = _combination()
    base = frame["model"] == "M1"

    clipped = base & (frame["term"] == "pred_male_neither")
    frame.loc[clipped, "ci_low"] = 0.0
    frame.loc[clipped, "note"] = "counterfactual_predicted_probability+" \
        "pred_ci_clipped_to_unit_interval"

    unavailable = base & (frame["term"] == "pred_female_public_only")
    frame.loc[unavailable, ["se", "ci_low", "ci_high"]] = np.nan
    frame.loc[unavailable, "note"] = "counterfactual_predicted_probability+" \
        "pred_ci_unavailable"

    absent = base & (frame["term"] == "pred_male_both")
    frame.loc[absent, ["estimate", "se", "ci_low", "ci_high"]] = np.nan
    frame.loc[absent, "note"] = "counterfactual_predicted_probability+" \
        "pred_ci_unavailable"
    return frame


def _combo_distribution():
    rows = []
    shares = {"male": (0.30, 0.25, 0.30, 0.15), "female": (0.22, 0.13, 0.45, 0.20)}
    for gender, values in shares.items():
        for category, value in zip(
                ("neither", "public_only", "celebrity_only", "both"), values):
            rows.append(_row(outcome="source_combo_observed", domain="both",
                             model="raw", term="{}_{}".format(category, gender),
                             estimate=value, ci_low=value - 0.004,
                             ci_high=value + 0.004, scale="probability",
                             n_obs=52000 if gender == "male" else 174000,
                             n_dropped=0, note="wilson"))
    return pd.DataFrame(rows)


def _monthly():
    rows = []
    for outcome, scale in (("monthly_source_entered_rate", "probability"),
                           ("monthly_topical_share_mean", "proportion")):
        for domain in ("public", "celebrity"):
            for month in range(1, 13):
                base = 0.20 if domain == "public" else 0.30
                # 6 月人为放一个高值：异常月份规则必须能把它挑出来，
                # 而不是靠人事后指认
                bump = 0.12 if month == 6 else 0.0
                for term, offset in (("male", 0.02), ("female", -0.02)):
                    est = base + offset + bump
                    rows.append(_row(outcome=outcome, domain=domain,
                                     model="month={:02d}".format(month), term=term,
                                     estimate=est, ci_low=est - 0.005,
                                     ci_high=est + 0.005, scale=scale,
                                     n_obs=100000, n_dropped=0))
                rows.append(_row(outcome=outcome, domain=domain,
                                 model="month={:02d}".format(month),
                                 term="gender_male", estimate=0.04,
                                 ci_low=0.03, ci_high=0.05, scale=scale,
                                 n_obs=200000, n_dropped=0))
    # 月份固定效应行：model 不是 month=NN，图必须靠 model 前缀筛掉它
    rows.append(_row(outcome="source_entered", domain="public",
                     model="M0/month_fixed_effects", term="gender_male",
                     estimate=0.05, ci_low=0.04, ci_high=0.06, scale="probability",
                     n_obs=2000000, n_dropped=0))
    return pd.DataFrame(rows)


def _distributions(seed=0):
    rng = np.random.default_rng(seed)
    dist = {}
    for measure, scale in (("retweet_count", 3.0), ("delay_hours", 20.0),
                           ("char_density", 0.05)):
        frames = []
        for gender in ("m", "f"):
            for domain in ("public", "celebrity"):
                values = rng.lognormal(mean=np.log(scale), sigma=1.0, size=500)
                frames.append(pd.DataFrame({
                    "measure": measure, "gender": gender, "domain": domain,
                    "value": values,
                }))
        dist[measure] = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for measure, frame in dist.items():
        for (gender, domain), group in frame.groupby(["gender", "domain"]):
            row = {"measure": measure, "gender": gender, "domain": domain,
                   "n_total": 20000, "n_sampled": len(group), "n_missing": 3,
                   "cap": 500, "seed": 20200101}
            for level in fg.QUANTILE_LEVELS:
                row[fg.quantile_column(level, "full")] = float(
                    np.quantile(group["value"], level))
                row[fg.quantile_column(level, "sampled")] = float(
                    np.quantile(group["value"], level))
            summary_rows.append(row)
    return dist, pd.DataFrame(summary_rows)


@pytest.fixture()
def figure_data(tmp_path):
    """写出一整套合成 figure_data，返回 (数据目录, 出图目录)"""
    data_dir = tmp_path / "figure_data"
    data_dir.mkdir()
    fig_dir = tmp_path / "figures"

    frames = {
        "table2_raw_gender_gaps.parquet": _table2(),
        "models_entry.parquet": _models_entry(),
        "models_share.parquet": _models_share(),
        "models_persistence.parquet": _models_persistence(),
        "interaction_gender_domain.parquet": _interaction(),
        "decomposition_source_content.parquet": _decomposition(),
        "combination_multinomial.parquet": _combination(),
        "combo_distribution.parquet": _combo_distribution(),
        "monthly_rates.parquet": _monthly(),
    }
    for name, frame in frames.items():
        frame.to_parquet(str(data_dir / name), engine="pyarrow", index=False)

    dist, summary = _distributions()
    for measure, frame in dist.items():
        frame.to_parquet(str(data_dir / fg.DIST_FILES[measure]),
                         engine="pyarrow", index=False)
    summary.to_parquet(str(data_dir / fg.SUMMARY_FILE), engine="pyarrow", index=False)
    # 图数据清单：真实的 export_figure_data.build 会把它写进 figure_data/，
    # figures 画图前核对它的 year（见 fg.assert_manifest）
    _write_figure_manifest(str(data_dir), year=2020)
    return str(data_dir), str(fig_dir)


def _write_figure_manifest(data_dir, year=2020, run_id="run-under-test"):
    manifest = {
        "step": "export_figure_data_{}".format(year),
        "run_id": run_id,
        "git_sha": "deadbee",
        "git_dirty": False,
        "params": {"year": year, "seed": 20200101, "max_points": 500},
        "counts": {},
    }
    with open(os.path.join(data_dir, fg.FIGURE_MANIFEST_FILE), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------------------
# 每张图都能在无头环境下跑出一个非空 PDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func_name", [
    "fig1_core_outcomes",
    "fig2_adjusted_effects",
    "fig3_interaction",
    "fig4_source_content",
    "fig5_combinations",
    "fig6_monthly",
    "appendix_distributions",
])
def test_each_figure_writes_a_non_empty_pdf(figure_data, func_name):
    data_dir, fig_dir = figure_data
    path = getattr(fg, func_name)(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000       # 空画布也就几百字节
    assert path.endswith(".pdf")


def test_filenames_are_date_prefixed(figure_data):
    data_dir, fig_dir = figure_data
    path = fg.fig1_core_outcomes(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    name = os.path.basename(path)
    assert name.startswith(datetime.now().strftime("%Y%m%d") + "_fig1")


def test_all_writes_every_figure(figure_data):
    data_dir, fig_dir = figure_data
    paths = fg.all(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert len(paths) == 7
    assert all(os.path.getsize(p) > 1000 for p in paths)


# ---------------------------------------------------------------------------
# 缺文件必须明确报错，而不是画一张空图
# ---------------------------------------------------------------------------

def test_fig1_raises_when_required_file_missing(figure_data):
    data_dir, fig_dir = figure_data
    os.remove(os.path.join(data_dir, "table2_raw_gender_gaps.parquet"))
    with pytest.raises(FileNotFoundError) as err:
        fg.fig1_core_outcomes(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    message = str(err.value)
    assert "table2_raw_gender_gaps.parquet" in message
    assert "export_figure_data" in message      # 报错要说清下一步该做什么
    assert not os.path.exists(fig_dir)          # 不留下半张空图


def test_appendix_raises_when_distribution_missing(figure_data):
    data_dir, fig_dir = figure_data
    os.remove(os.path.join(data_dir, fg.DIST_FILES["delay_hours"]))
    with pytest.raises(FileNotFoundError) as err:
        fg.appendix_distributions(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert fg.DIST_FILES["delay_hours"] in str(err.value)


# ---------------------------------------------------------------------------
# domain="both"：差中差不能在按领域分组时被丢掉
# ---------------------------------------------------------------------------

def test_did_rows_keep_domain_both():
    frame = _interaction()
    did = fg.did_rows(frame, outcome="source_entered")
    assert len(did) == 3                         # M0/M1/M2 各一行
    assert set(did["domain"]) == {"both"}
    assert set(did["model"]) == {"M0", "M1", "M2"}


def test_domain_facets_exclude_both():
    """按领域分面时只画两个面板，跨域行另行处理，不产生第三个面板"""
    frame = _interaction()
    assert fg.domain_facets(frame) == ["public", "celebrity"]


def test_did_annotation_contains_estimate_and_interval():
    frame = _interaction()
    text = fg.did_annotation(frame, outcome="source_entered", model="M1")
    assert "0.2" in text                          # M1 的 DiD = 0.25*0.8 = 0.20
    assert "95%" in text


def test_predicted_cells_are_four_and_carry_real_domains():
    """四个预测格子的 domain 是真实领域，不是 both——它们不属于跨域行"""
    frame = _interaction()
    cells = fg.predicted_cells(frame, outcome="source_entered", model="M1")
    assert len(cells) == 4
    assert set(cells["domain"]) == {"public", "celebrity"}
    assert set(cells["term"]) == set(fg.TERM_PRED.values())


def test_predicted_cells_reproduce_the_gaps_and_the_did():
    """图上画的格子与标注的差中差必须来自同一次拟合：恒等式在这里再核一遍"""
    frame = _interaction()
    cells = fg.predicted_cells(frame, outcome="source_entered", model="M1")
    lookup = dict(zip(cells["term"], cells["estimate"]))
    gap_public = lookup["pred_male_public"] - lookup["pred_female_public"]
    gap_celeb = lookup["pred_male_celebrity"] - lookup["pred_female_celebrity"]
    gap_row = fg.select(frame, outcome="source_entered", model="M1",
                        term="gender_male_gap_public").iloc[0]
    assert gap_public == pytest.approx(gap_row["estimate"])
    did_row = fg.did_rows(frame, outcome="source_entered", model="M1").iloc[0]
    assert gap_public - gap_celeb == pytest.approx(did_row["estimate"])


def test_predicted_cells_do_not_disturb_the_did_row():
    """新增的 pred_* 行不能影响差中差与分面的取法"""
    frame = _interaction()
    assert len(fg.did_rows(frame, outcome="source_entered")) == 3
    assert fg.domain_facets(frame) == ["public", "celebrity"]


def test_cell_status_distinguishes_clipped_unavailable_and_missing():
    """三种"不干净"的格子必须能被区分出来，不能都当成正常估计画"""
    frame = _interaction_with_flagged_cells()
    cells = fg.predicted_cells(frame, outcome="source_entered", model="M1")
    status = {row["term"]: fg.cell_status(row) for _, row in cells.iterrows()}
    assert status["pred_male_public"] == "clipped"
    assert status["pred_female_public"] == "ci_unavailable"
    assert status["pred_male_celebrity"] == "missing"
    assert status["pred_female_celebrity"] == "ok"


def test_fig3_draws_flagged_cells_without_dropping_them(figure_data):
    """截断/不可用/缺失的格子都要留下痕迹，图仍然出得来"""
    data_dir, fig_dir = figure_data
    _interaction_with_flagged_cells().to_parquet(
        os.path.join(data_dir, "interaction_gender_domain.parquet"),
        engine="pyarrow", index=False)
    path = fg.fig3_interaction(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.getsize(path) > 1000


def test_fig3_raises_when_interaction_file_missing(figure_data):
    data_dir, fig_dir = figure_data
    os.remove(os.path.join(data_dir, "interaction_gender_domain.parquet"))
    with pytest.raises(FileNotFoundError) as err:
        fg.fig3_interaction(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert "interaction_gender_domain.parquet" in str(err.value)


def test_did_annotation_reports_missing_estimate():
    frame = _interaction()
    mask = ((frame["term"] == "gender_male_x_public_did")
            & (frame["model"] == "M1") & (frame["outcome"] == "source_entered"))
    frame.loc[mask, ["estimate", "ci_low", "ci_high"]] = np.nan
    frame.loc[mask, "note"] = "boot_refit_failed:12"
    text = fg.did_annotation(frame, outcome="source_entered", model="M1")
    assert "boot_refit_failed:12" in text


# ---------------------------------------------------------------------------
# NaN 行：必须被显式标注，不能静默跳过
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 有点估计、没有区间的行：绝不能画成一个宽度为 0 的区间
# ---------------------------------------------------------------------------

def _ci_unavailable_row(**kwargs):
    """点估计在、区间是 NaN 的一行。

    这不是假想情形：stats_utils.average_marginal_effect 与
    models_interaction.difference_in_differences 的 bootstrap 分支都会在
    协方差不可用 / 重拟合失败时返回"有点估计、无区间"的行。
    """
    row = _row(outcome="topical_share", domain="celebrity", model="M2",
               term="gender_male", estimate=0.0405, se=np.nan,
               ci_low=np.nan, ci_high=np.nan, scale="proportion",
               n_obs=180000, n_dropped=46000, note="boot_refit_failed:12")
    row.update(kwargs)
    return pd.Series(row)


def test_estimate_status_calls_a_point_without_interval_ci_unavailable():
    assert fg.estimate_status(_ci_unavailable_row()) == "ci_unavailable"


def test_estimate_status_is_the_same_rule_used_for_predicted_cells():
    """图 3 的 cell_status 与通用的 estimate_status 必须是同一条规则"""
    row = _ci_unavailable_row()
    assert fg.cell_status(row) == fg.estimate_status(row)


def test_draw_estimate_draws_no_interval_when_the_ci_is_missing():
    """没有不确定性的估计不许画成图上最"精确"的那个点"""
    fig, ax = plt.subplots()
    flags = []
    status = fg.draw_estimate(ax, _ci_unavailable_row(), 0, fg.MALE_COLOR, "o",
                              flags=flags, tag="topical_share/celebrity/M2")
    plt.close(fig)
    assert status == "ci_unavailable"
    # errorbar 会往 ax.containers 里塞一个 ErrorbarContainer；这里不该有
    assert len(ax.containers) == 0
    assert flags and "CI unavailable" in flags[0]
    assert "boot_refit_failed:12" in flags[0]


def test_draw_estimate_draws_an_interval_for_a_complete_row():
    fig, ax = plt.subplots()
    flags = []
    row = _ci_unavailable_row(se=0.002, ci_low=0.0365, ci_high=0.0445, note=None)
    status = fg.draw_estimate(ax, row, 0, fg.MALE_COLOR, "o", flags=flags)
    plt.close(fig)
    assert status == "ok"
    assert len(ax.containers) == 1
    assert flags == []


def test_draw_estimate_marks_a_gap_for_a_missing_estimate():
    fig, ax = plt.subplots()
    flags = []
    row = _ci_unavailable_row(estimate=np.nan, note="empty_sample")
    status = fg.draw_estimate(ax, row, 0, fg.MALE_COLOR, "o", flags=flags)
    plt.close(fig)
    assert status == "missing"
    assert len(ax.containers) == 0
    assert flags and "no estimate" in flags[0] and "empty_sample" in flags[0]


def test_did_annotation_reports_a_missing_interval_instead_of_nan():
    frame = _interaction()
    mask = ((frame["term"] == "gender_male_x_public_did")
            & (frame["model"] == "M1") & (frame["outcome"] == "source_entered"))
    frame.loc[mask, ["se", "ci_low", "ci_high"]] = np.nan
    frame.loc[mask, "note"] = "boot_refit_failed:12"
    text = fg.did_annotation(frame, outcome="source_entered", model="M1")
    assert "nan" not in text.lower()
    assert "CI unavailable" in text
    assert "boot_refit_failed:12" in text


def test_clipped_ends_uses_inequalities_not_equality():
    """截断端点用 <=0 / >=1 判定：生产端换一种 clip 写法也不该丢掉箭头"""
    row = _ci_unavailable_row(ci_low=-1e-18, ci_high=1.0 + 1e-18,
                              note="pred_ci_clipped_to_unit_interval")
    assert fg.clipped_ends(row) == ("low", "high")
    row = _ci_unavailable_row(ci_low=0.02, ci_high=1.0,
                              note="pred_ci_clipped_to_unit_interval")
    assert fg.clipped_ends(row) == ("high",)


def test_fig1_flags_a_row_without_an_interval(figure_data):
    """图 1 上出现"有点估计、无区间"的格子时必须留痕，而不是画成零宽区间"""
    data_dir, fig_dir = figure_data
    table2 = _table2()
    mask = ((table2["outcome"] == "source_entered")
            & (table2["domain"] == "public") & (table2["term"] == "male")
            & (table2["denominator"] == "all_same_gender_users"))
    table2.loc[mask, ["ci_low", "ci_high"]] = np.nan
    table2.loc[mask, "note"] = "wilson_unavailable"
    table2.to_parquet(os.path.join(data_dir, "table2_raw_gender_gaps.parquet"),
                      engine="pyarrow", index=False)
    path = fg.fig1_core_outcomes(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.getsize(path) > 1000


def test_fig2_flags_a_row_without_an_interval(figure_data):
    data_dir, fig_dir = figure_data
    share = _models_share(with_nan=False)
    mask = ((share["model"] == "M2") & (share["outcome"] == "topical_share")
            & (share["domain"] == "celebrity"))
    share.loc[mask, ["se", "ci_low", "ci_high"]] = np.nan
    share.loc[mask, "note"] = "boot_refit_failed:12"
    share.to_parquet(os.path.join(data_dir, "models_share.parquet"),
                     engine="pyarrow", index=False)
    path = fg.fig2_adjusted_effects(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# 图 5：多项模型的预测概率（§12.5）
# ---------------------------------------------------------------------------

def test_predicted_combo_cells_are_eight_per_layer():
    frame = _combination()
    cells = fg.predicted_combo_cells(frame, model="M1")
    assert len(cells) == 8
    assert set(cells["domain"]) == {"both"}


def test_predicted_combo_cells_reproduce_the_ames():
    """预测概率之差必须等于同一层报告的 AME——图与表是同一次拟合的两种写法"""
    frame = _combination()
    cells = fg.predicted_combo_cells(frame, model="M1").set_index("term")
    for category in fg.COMBO_CATEGORIES:
        gap = (cells.loc["pred_male_{}".format(category), "estimate"]
               - cells.loc["pred_female_{}".format(category), "estimate"])
        ame = fg.select(frame, model="M1",
                        term="gender_male_ame_{}".format(category)).iloc[0]
        assert gap == pytest.approx(ame["estimate"])


def test_fig5_draws_flagged_predicted_cells_without_dropping_them(figure_data):
    data_dir, fig_dir = figure_data
    _combination_with_flagged_cells().to_parquet(
        os.path.join(data_dir, "combination_multinomial.parquet"),
        engine="pyarrow", index=False)
    path = fg.fig5_combinations(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# 图 3：观测点与预测点必须是两种不同的记号
# ---------------------------------------------------------------------------

def test_observed_marker_differs_from_every_predicted_marker():
    assert fg.OBSERVED_MARKER not in set(fg.GENDER_MARKERS.values())


def test_missing_estimate_rows_are_found():
    frame = _models_share()
    missing = fg.missing_estimate_rows(frame)
    assert len(missing) == 1
    row = missing.iloc[0]
    assert row["model"] == "M2" and row["domain"] == "celebrity"


def test_missing_estimate_label_quotes_the_note():
    frame = _models_share()
    labels = fg.missing_estimate_labels(frame)
    assert len(labels) == 1
    assert "not_converged" in labels[0]


def test_fig2_still_draws_when_a_whole_layer_failed(figure_data):
    """整层 M2 全部 NaN 时仍要出图，并把缺失写在图上"""
    data_dir, fig_dir = figure_data
    frame = _models_share(with_nan=False)
    frame.loc[frame["model"] == "M2",
              ["estimate", "se", "ci_low", "ci_high"]] = np.nan
    frame.loc[frame["model"] == "M2", "note"] = "empty_sample"
    frame.to_parquet(os.path.join(data_dir, "models_share.parquet"),
                     engine="pyarrow", index=False)
    path = fg.fig2_adjusted_effects(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.getsize(path) > 1000


# ---------------------------------------------------------------------------
# 异常月份：事前固定的稳健规则，不事后挑事件
# ---------------------------------------------------------------------------

def test_flag_unusual_months_marks_the_spike():
    values = np.array([0.2] * 11 + [0.5])
    flags = fg.flag_unusual_months(values)
    assert flags.sum() == 1 and flags[-1]


def test_flag_unusual_months_marks_nothing_on_a_flat_series():
    values = np.full(12, 0.2)
    assert fg.flag_unusual_months(values).sum() == 0


def test_flag_unusual_months_uses_mad_not_std():
    """标准差会被离群点自己撑大；MAD 不会——换成 std 这条断言就会失败"""
    values = np.array([0.20, 0.21, 0.19, 0.20, 0.22, 0.18,
                       0.21, 0.19, 0.20, 0.21, 0.20, 0.60])
    flags = fg.flag_unusual_months(values)
    assert flags[-1]
    assert flags[:-1].sum() == 0


def test_flag_unusual_months_ignores_nan():
    values = np.array([0.2] * 10 + [np.nan, 0.5])
    flags = fg.flag_unusual_months(values)
    assert not flags[10]
    assert flags[11]


# ---------------------------------------------------------------------------
# 月度行的解析
# ---------------------------------------------------------------------------

def test_month_from_model_label_parses_only_month_rows():
    frame = _monthly()
    monthly = fg.monthly_rows(frame, outcome="monthly_source_entered_rate",
                              domain="public")
    assert sorted(monthly["month"].unique()) == list(range(1, 13))
    # 月度固定效应那一行（model="M0/month_fixed_effects"）必须被排除
    assert "M0/month_fixed_effects" not in set(monthly["model"])


# ---------------------------------------------------------------------------
# select_one：撞键必须当场失败，不能按行序任取一行
# ---------------------------------------------------------------------------

def test_select_one_returns_the_single_matching_row():
    frame = _interaction()
    row = fg.select_one(frame, outcome="source_entered", domain="both",
                        model="M1", term=fg.TERM_DID)
    assert row is not None
    assert row["model"] == "M1"


def test_select_one_returns_none_when_nothing_matches():
    """没有匹配是一种正当情形：调用方要把它标成 row absent，而不是崩掉"""
    frame = _interaction()
    assert fg.select_one(frame, outcome="source_entered", model="M9",
                         term=fg.TERM_DID) is None


def test_select_one_refuses_to_pick_arbitrarily_when_the_key_is_duplicated():
    """撞键时 .iloc[0] 会按行序任取一行——不报错、不留痕、数字随行序漂移

    这正是表 2 曾经的问题（两个分母口径都写 model="raw"）。图在十几处
    按 (outcome, domain, model, term) 筛完就取第一行，所以这道闸门必须
    设在取行的入口上，而不是指望每个调用方自己检查。
    """
    frame = _interaction()
    duplicated = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(AssertionError, match="必须唯一"):
        fg.select_one(duplicated, outcome="source_entered", domain="both",
                      model="M1", term=fg.TERM_DID)


def test_select_still_allows_multi_row_filtering():
    """select 本身不加唯一性断言：它同样用于取一整张子表（12 个月、一层三行）"""
    frame = _monthly()
    rows = fg.monthly_rows(frame, outcome="monthly_source_entered_rate",
                           domain="public")
    assert len(fg.select(rows, term="male")) == 12


def test_every_figure_survives_a_full_draw_with_the_uniqueness_guard(figure_data):
    """整套图在开着唯一性断言的情况下必须画得出来

    等价于在合成数据上把"结果表四元组唯一"这条不变量跑了一遍：任何一格
    撞键，对应的图会直接抛 AssertionError，而不是悄悄画错。
    """
    data_dir, fig_dir = figure_data
    assert len(fg.all(year=2020, data_dir=data_dir, fig_dir=fig_dir)) == 7


# ---------------------------------------------------------------------------
# 图 6：月度区间缺失同样要留痕（此前被 NaN 区间修复漏掉）
# ---------------------------------------------------------------------------

def test_fig6_marks_a_month_whose_interval_is_missing(figure_data):
    """有点估计、区间是 NaN 的月份必须画成"CI n/a"，不能画成一个普通点

    图 6 原来自己调 fill_between、自己只检查估计值是否有限，于是这种月份
    在一条连续折线上看起来与别的点毫无区别——不确定性缺失被读成不确定性
    为零，正是整个 §12 反复要避免的那种误导。
    """
    data_dir, fig_dir = figure_data
    monthly = _monthly()
    mask = ((monthly["outcome"] == "monthly_source_entered_rate")
            & (monthly["domain"] == "public")
            & (monthly["model"] == "month=06") & (monthly["term"] == "male"))
    assert mask.sum() == 1
    monthly.loc[mask, ["ci_low", "ci_high"]] = np.nan
    monthly.loc[mask, "note"] = "wilson_unavailable"

    # 先确认这一行确实被判成 ci_unavailable（否则下面画出来的图证明不了什么）
    row = monthly[mask].iloc[0]
    assert fg.estimate_status(row) == "ci_unavailable"

    monthly.to_parquet(os.path.join(data_dir, "monthly_rates.parquet"),
                       engine="pyarrow", index=False)
    path = fg.fig6_monthly(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.getsize(path) > 1000


def test_fig6_marks_a_month_with_no_estimate_at_all(figure_data):
    data_dir, fig_dir = figure_data
    monthly = _monthly()
    mask = ((monthly["outcome"] == "monthly_topical_share_mean")
            & (monthly["domain"] == "celebrity")
            & (monthly["model"] == "month=09") & (monthly["term"] == "female"))
    monthly.loc[mask, ["estimate", "ci_low", "ci_high"]] = np.nan
    monthly.loc[mask, "note"] = "empty_sample"
    monthly.to_parquet(os.path.join(data_dir, "monthly_rates.parquet"),
                       engine="pyarrow", index=False)
    path = fg.fig6_monthly(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert os.path.getsize(path) > 1000


def test_fig6_flags_a_missing_interval_through_draw_estimate():
    """区间缺失的月份走的是 draw_estimate 的 ci_unavailable 分支

    直接在坐标轴上验证，不依赖 PDF 的字节数：不画区间容器、留下一条
    带 note 的说明。图 6 与图 1/2/3/5 必须是同一条规则。
    """
    fig, ax = plt.subplots()
    flags = []
    row = pd.Series(_row(outcome="monthly_source_entered_rate", domain="public",
                         model="month=06", term="male", estimate=0.32,
                         ci_low=np.nan, ci_high=np.nan, scale="probability",
                         n_obs=100000, n_dropped=0, note="wilson_unavailable"))
    status = fg.draw_estimate(ax, row, 6, fg.MALE_COLOR, "o", orientation="v",
                              flags=flags, tag="male month=06")
    plt.close(fig)
    assert status == "ci_unavailable"
    assert len(ax.containers) == 0
    assert flags and "CI unavailable" in flags[0] and "month=06" in flags[0]


# ---------------------------------------------------------------------------
# 图数据清单：年份错配必须被察觉（figure_data/ 的文件名不带年份）
# ---------------------------------------------------------------------------

def test_figures_reject_a_manifest_from_a_different_year(figure_data):
    """用 2019 年的导出画 2020 年的图，此前完全无法察觉

    figure_data/ 的文件名不带年份，--year 只进标题与文件名前缀，所以
    错配画出来的图每个数字都是真的，只是属于另一年。
    """
    data_dir, fig_dir = figure_data
    _write_figure_manifest(data_dir, year=2019)
    with pytest.raises(ValueError) as err:
        fg.fig1_core_outcomes(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert "2019" in str(err.value) and "2020" in str(err.value)


def test_figures_require_the_manifest_to_exist(figure_data):
    data_dir, fig_dir = figure_data
    os.remove(os.path.join(data_dir, fg.FIGURE_MANIFEST_FILE))
    with pytest.raises(FileNotFoundError) as err:
        fg.fig2_adjusted_effects(year=2020, data_dir=data_dir, fig_dir=fig_dir)
    assert "export_figure_data" in str(err.value)


def test_manifest_check_covers_every_figure_entry_point(figure_data):
    """七个入口一个都不能漏：漏掉的那一个就是年份错配的后门"""
    data_dir, fig_dir = figure_data
    _write_figure_manifest(data_dir, year=2019)
    for func_name in ("fig1_core_outcomes", "fig2_adjusted_effects",
                      "fig3_interaction", "fig4_source_content",
                      "fig5_combinations", "fig6_monthly",
                      "appendix_distributions"):
        with pytest.raises(ValueError):
            getattr(fg, func_name)(year=2020, data_dir=data_dir, fig_dir=fig_dir)


def test_figure_manifest_filename_matches_the_export_layer():
    """两个模块各写一份常量（figures 跑在本地，不 import 导出层），必须同名"""
    from gender_domain import export_figure_data as ex
    assert fg.FIGURE_MANIFEST_FILE == ex.FIGURE_MANIFEST_FILE


# ---------------------------------------------------------------------------
# 表 2 的两个分母口径：图必须取主口径那个 model 变体
# ---------------------------------------------------------------------------

def test_fig1_reads_the_primary_denominator_variant():
    """两个分母口径在 model 里分开之后，图必须显式挑主口径那一个"""
    table2 = _table2()
    rows = fg.select(table2, outcome="source_entered",
                     model=fg.PRIMARY_TABLE2_MODEL["source_entered"],
                     denominator=fg.PRIMARY_DENOMINATOR["source_entered"])
    # 两个域 × (男/女/gap) = 6 行，非主口径的那一组一行都不许混进来
    assert len(rows) == 6
    assert set(rows["denominator"]) == {"all_same_gender_users"}
    male_public = fg.select_one(rows, domain="public", term="male")
    assert male_public["estimate"] == pytest.approx(0.30)
