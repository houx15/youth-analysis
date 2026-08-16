"""
gender_domain.export_figure_data 的单元测试（§12 图数据导出）。

这一层的职责很窄：把服务器上已经算好的结果表原样搬到一个"能下载"的目录，
再为附录的 ECDF 抽一份有上限的分布样本。因此测试只盯三件事：

1. 抽样必须可复现——同一个 seed 抽两次结果逐点相同，换 seed 会变；
2. 抽样必须保形——抽出来的样本分位数与全量分位数在容差内一致，否则
   附录 ECDF 画的就不是真实分布；
3. 记账必须完整——每一格记下全量样本量、抽样上限、seed 与全量分位数，
   缺一样，下载到本地之后就没法说清"这条曲线代表多少人"。

期望数字的推导都写在注释里：分位数容差按"随机抽 2000 点估计中位数的
标准误"给，不是先跑代码再抄结果。
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import config
from gender_domain import export_figure_data as ex
from gender_domain import stats_utils as su


# ---------------------------------------------------------------------------
# 抽样：可复现、有上限、保形
# ---------------------------------------------------------------------------

def _long_fixture(n_per_group=8000, seed=7, sigma=1.5):
    """构造 gender × domain 四格的长表，取值是重尾的对数正态

    重尾是刻意的：真实的转发次数与延迟都是重尾分布，均匀分布测不出
    "抽样把尾巴削掉了"这种错误。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gender in ("m", "f"):
        for domain in ("public", "celebrity"):
            values = rng.lognormal(mean=1.0, sigma=sigma, size=n_per_group)
            rows.append(pd.DataFrame({
                "measure": "retweet_count",
                "gender": gender,
                "domain": domain,
                "value": values,
            }))
    return pd.concat(rows, ignore_index=True)


def test_sample_distribution_is_reproducible_under_same_seed():
    long_df = _long_fixture()
    a, _ = ex.sample_distribution(long_df, cap=2000, seed=123)
    b, _ = ex.sample_distribution(long_df, cap=2000, seed=123)
    assert len(a) == len(b)
    np.testing.assert_allclose(a["value"].to_numpy(), b["value"].to_numpy())


def test_sample_distribution_changes_with_different_seed():
    long_df = _long_fixture()
    a, _ = ex.sample_distribution(long_df, cap=2000, seed=123)
    b, _ = ex.sample_distribution(long_df, cap=2000, seed=456)
    # 两次抽样都是 2000 点，逐点完全相同的概率可以忽略
    assert not np.allclose(a["value"].to_numpy(), b["value"].to_numpy())


def test_sample_distribution_caps_each_gender_domain_group():
    long_df = _long_fixture(n_per_group=8000)
    sampled, _ = ex.sample_distribution(long_df, cap=2000, seed=0)
    sizes = sampled.groupby(["gender", "domain"]).size()
    assert len(sizes) == 4                     # 四格都在，一格都不能少
    assert set(sizes.unique()) == {2000}


def test_sample_distribution_keeps_small_groups_whole():
    """样本量本来就低于上限的格子必须整格保留，不能被抽稀"""
    long_df = pd.DataFrame({
        "measure": "delay_hours",
        "gender": ["m"] * 10 + ["f"] * 3,
        "domain": ["public"] * 13,
        "value": np.arange(13, dtype=float),
    })
    sampled, summary = ex.sample_distribution(long_df, cap=100, seed=0)
    assert len(sampled) == 13
    assert set(summary["n_sampled"]) == {10, 3}


def test_sample_distribution_preserves_quantiles_within_tolerance():
    """抽样后的分位数必须与全量分位数一致到容差内

    容差不是试出来的，是按样本分位数的标准误算的。取值服从
    lognormal(μ=1, σ=1)，从 n=20000 里不放回抽 m=4000 点，样本 p 分位数的
    标准误约为 sqrt(p(1-p)/m)/f(q_p) × sqrt(1-m/n)（有限总体校正）：
      p50: q=e^1=2.718, f=1/(q·σ·sqrt(2π))=0.147 -> SE≈0.048，相对 1.8%
      p90: q=9.80,  f=0.0179 -> SE≈0.24，相对 2.4%
      p95: q=14.06, f=0.00734 -> SE≈0.42，相对 3.0%
    这里对 p25/p50/p75 给 6% 容差、对 p90/p95 给 10%，都约等于 3 个标准误：
    比这更紧测的是随机数而不是代码，比这更松则漏得掉"抽样把尾巴削平"。
    """
    long_df = _long_fixture(n_per_group=20000, sigma=1.0)
    sampled, _ = ex.sample_distribution(long_df, cap=4000, seed=20200101)
    for (gender, domain), group in long_df.groupby(["gender", "domain"]):
        sub = sampled[(sampled["gender"] == gender) & (sampled["domain"] == domain)]
        for q, rtol in ((0.25, 0.06), (0.50, 0.06), (0.75, 0.06),
                        (0.90, 0.10), (0.95, 0.10)):
            full_q = float(np.quantile(group["value"], q))
            samp_q = float(np.quantile(sub["value"], q))
            assert samp_q == pytest.approx(full_q, rel=rtol), (
                f"{gender}/{domain} 的 p{int(q * 100)} 抽样后偏离全量太多"
            )


def test_summary_records_seed_cap_and_full_quantiles():
    long_df = _long_fixture(n_per_group=5000)
    _, summary = ex.sample_distribution(long_df, cap=1000, seed=999)
    assert set(summary["seed"]) == {999}
    assert set(summary["cap"]) == {1000}
    assert set(summary["n_total"]) == {5000}
    # 全量分位数按 QUANTILE_LEVELS 逐个记下来，图里要靠它写分母/标注
    for level in ex.QUANTILE_LEVELS:
        assert ex.quantile_column(level, "full") in summary.columns
        assert ex.quantile_column(level, "sampled") in summary.columns
    row = summary[(summary["gender"] == "m") & (summary["domain"] == "public")].iloc[0]
    group = long_df[(long_df["gender"] == "m") & (long_df["domain"] == "public")]
    assert row[ex.quantile_column(0.5, "full")] == pytest.approx(
        float(np.quantile(group["value"], 0.5))
    )


def test_summary_counts_missing_values_separately():
    """NaN 不是 0：无定义的取值只能记进 n_missing，不能混进分布"""
    long_df = pd.DataFrame({
        "measure": "char_density",
        "gender": ["m", "m", "m", "f"],
        "domain": ["public"] * 4,
        "value": [0.1, np.nan, 0.3, np.nan],
    })
    sampled, summary = ex.sample_distribution(long_df, cap=10, seed=0)
    assert len(sampled) == 2
    male = summary[summary["gender"] == "m"].iloc[0]
    assert male["n_total"] == 2 and male["n_missing"] == 1


# ---------------------------------------------------------------------------
# 三份分布的构造
# ---------------------------------------------------------------------------

def _user_fixture():
    """6 名用户的表 C 片段，手工设计每一列的取值"""
    return pd.DataFrame({
        "user_id": ["1", "2", "3", "4", "5", "6"],
        "gender": ["m", "m", "f", "f", "f", "x"],       # x 是第三性别，必须被记为流失
        "public_source_count": [0, 5, 2, 0, 9, 3],
        "celebrity_source_count": [1, 0, 7, 4, 0, 2],
        "public_source_entered": [False, True, True, False, True, True],
        "celebrity_source_entered": [True, False, True, True, False, True],
        "public_char_density": [0.0, 0.2, np.nan, 0.4, 0.1, 0.3],
        "celebrity_char_density": [np.nan, 0.5, 0.6, 0.0, 0.2, 0.1],
        "source_combo": ["celebrity_only", "public_only", "both", "celebrity_only",
                         "public_only", "both"],
    })


def test_retweet_count_distribution_is_long_and_keeps_zeros():
    out = ex.retweet_count_distribution(_user_fixture())
    assert list(out.columns) == ex.DIST_COLUMNS
    assert set(out["measure"]) == {"retweet_count"}
    assert set(out["domain"]) == {"public", "celebrity"}
    # 0 次转发是有定义的事实，必须留在分布里（ECDF 的左端就是它）
    male_public = out[(out["gender"] == "m") & (out["domain"] == "public")]
    assert sorted(male_public["value"]) == [0.0, 5.0]


def test_retweet_count_distribution_drops_unknown_gender():
    out = ex.retweet_count_distribution(_user_fixture())
    assert set(out["gender"]) == {"m", "f"}


def test_density_distribution_carries_nan_through_to_be_counted():
    """无表达帖的用户 density 无定义：长表里原样保留 NaN，交给抽样那一步记数

    这一步刻意不把 NaN 提前丢掉——丢掉之后 n_missing 就永远是 0，
    "有多少用户没有定义"这件事就再也无法从导出文件里读出来。
    """
    out = ex.density_distribution(_user_fixture())
    assert set(out["measure"]) == {"char_density"}
    female_public = out[(out["gender"] == "f") & (out["domain"] == "public")]
    # 用户 3 的 public_char_density 是 NaN，用户 4、5 分别是 0.4、0.1
    assert female_public["value"].isna().sum() == 1
    assert sorted(female_public["value"].dropna()) == [0.1, 0.4]


def test_density_nan_is_excluded_from_the_sample_and_counted_as_missing():
    """抽样这一步才把 NaN 排除，并记进 n_missing——绝不填 0 混进分布"""
    long_df = ex.density_distribution(_user_fixture())
    sampled, summary = ex.sample_distribution(long_df, cap=100, seed=0)
    female_public = sampled[(sampled["gender"] == "f")
                            & (sampled["domain"] == "public")]
    assert sorted(female_public["value"]) == [0.1, 0.4]
    row = summary[(summary["gender"] == "f")
                  & (summary["domain"] == "public")].iloc[0]
    assert row["n_total"] == 2 and row["n_missing"] == 1


def test_delay_distribution_uses_delay_hours_column():
    clean = pd.DataFrame({
        "user_id": ["1", "2", "3"],
        "gender": ["m", "f", "f"],
        "source_domain": ["public", "public", "celebrity"],
        "delay_hours": [1.5, 20.0, 3.0],
    })
    out = ex.delay_distribution(clean)
    assert set(out["measure"]) == {"delay_hours"}
    assert sorted(out[out["domain"] == "public"]["value"]) == [1.5, 20.0]


# ---------------------------------------------------------------------------
# 四类来源组合的观测分布（图 5 左半边）
# ---------------------------------------------------------------------------

def test_combo_distribution_uses_shared_result_schema():
    out = ex.combo_distribution(_user_fixture())
    assert list(out.columns) == list(su.RESULT_SCHEMA)
    assert set(out["domain"]) == {"both"}          # 组合是跨域的，不属于任一领域
    assert set(out["model"]) == {"raw"}


def test_combo_distribution_shares_sum_to_one_per_gender():
    out = ex.combo_distribution(_user_fixture())
    for gender_label in ("male", "female"):
        rows = out[out["term"].str.endswith("_" + gender_label)]
        assert len(rows) == 4                       # 四个类别一格不缺
        assert rows["estimate"].sum() == pytest.approx(1.0)


def test_combo_distribution_male_shares_are_hand_checked():
    out = ex.combo_distribution(_user_fixture()).set_index("term")
    # 男性两人：用户 1 celebrity_only、用户 2 public_only
    assert out.loc["celebrity_only_male", "estimate"] == pytest.approx(0.5)
    assert out.loc["public_only_male", "estimate"] == pytest.approx(0.5)
    assert out.loc["both_male", "estimate"] == pytest.approx(0.0)
    assert out.loc["neither_male", "estimate"] == pytest.approx(0.0)


def test_combo_distribution_ci_within_unit_interval():
    out = ex.combo_distribution(_user_fixture())
    assert (out["ci_low"] >= 0).all()
    assert (out["ci_high"] <= 1).all()


def test_combo_distribution_records_other_gender_as_dropped():
    out = ex.combo_distribution(_user_fixture())
    row = out[out["term"] == "both_male"].iloc[0]
    # 6 名用户里 1 名第三性别；男性行的 n_obs 是 2，其余都算流失
    assert row["n_obs"] == 2
    assert row["n_obs"] + row["n_dropped"] == 6
    assert "other_gender" in row["drop_reason"]


# ---------------------------------------------------------------------------
# 结果表搬运与整体 build
# ---------------------------------------------------------------------------

def _write_results(results_dir):
    """写出 12 张结果表的最小骨架，列名与真实模块一致"""
    os.makedirs(results_dir, exist_ok=True)
    schema_frame = pd.DataFrame([{
        "outcome": "source_entered", "domain": "public", "model": "M0",
        "term": "gender_male", "estimate": 0.05, "se": 0.01,
        "ci_low": 0.03, "ci_high": 0.07, "scale": "probability",
        "n_obs": 100, "n_dropped": 0, "drop_reason": None, "note": None,
    }])
    for name in ex.RESULT_FILES:
        if name == "table1_sample_activity.parquet":
            frame = pd.DataFrame([{
                "gender": "m", "n_users": 100, "variable": "n_posts",
                "mean": 10.0, "median": 8.0, "p75": 12.0, "p90": 20.0,
                "p95": 25.0, "p99": 40.0,
            }])
        elif name == "table2_raw_gender_gaps.parquet":
            frame = schema_frame.copy()
            frame["model"] = "raw"
            frame["term"] = "male"
            frame["denominator"] = "all_same_gender_users"
        else:
            frame = schema_frame.copy()
        frame.to_parquet(os.path.join(results_dir, name), engine="pyarrow", index=False)


def _write_events(events_dir, year=2020):
    """两个月的表 B 分片，时间戳用秒，全部落在 year 之内"""
    os.makedirs(events_dir, exist_ok=True)
    base = pd.Timestamp(f"{year}-03-01", tz=None).value // 10 ** 9
    rng = np.random.default_rng(3)
    for month in (3, 4):
        n = 40
        delay = rng.uniform(0.5, 100.0, size=n) * 3600.0
        source_ts = base + rng.integers(0, 20 * 86400, size=n).astype(float)
        retweet_ts = source_ts + delay
        frame = pd.DataFrame({
            "user_id": [str(i) for i in range(n)],
            "weibo_id": [f"w{month}_{i}" for i in range(n)],
            "r_weibo_id": [f"r{month}_{i}" for i in range(n)],
            "r_user_id": ["src"] * n,
            "date": pd.to_datetime(retweet_ts, unit="s").strftime("%Y-%m-%d"),
            "month": month,
            "gender": ["m" if i % 2 else "f" for i in range(n)],
            "source_domain": ["public" if i % 3 else "celebrity" for i in range(n)],
            "source_category": ["news"] * n,
            "retweet_ts": retweet_ts,
            "source_ts": source_ts,
            "delay_seconds": delay,
            "delay_valid": True,
        })
        frame.to_parquet(
            os.path.join(events_dir, "month={:02d}.parquet".format(month)),
            engine="pyarrow", index=False,
        )


@pytest.fixture()
def analysis_dir(tmp_path, monkeypatch):
    """搭一个假的 analysis_data/，把 config.OUTPUT_DIR 指过去"""
    root = tmp_path / "analysis_data"
    root.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", str(root))
    _write_results(str(root / "results"))
    _write_events(str(root / "retweet_domain_events_2020"))
    users = _user_fixture()
    users.to_parquet(str(root / "user_domain_2020.parquet"),
                     engine="pyarrow", index=False)
    return root


def test_copy_result_tables_reports_missing_files_by_name(analysis_dir):
    os.remove(str(analysis_dir / "results" / "models_share.parquet"))
    with pytest.raises(FileNotFoundError) as err:
        ex.copy_result_tables()
    assert "models_share.parquet" in str(err.value)


def test_build_writes_every_file_and_stays_small(analysis_dir):
    ex.build(year=2020, max_points=500, seed=42)
    out_dir = ex.figure_data_dir()
    expected = list(ex.RESULT_FILES) + [ex.COMBO_FILE, ex.SUMMARY_FILE] + \
        list(ex.DIST_FILES.values())
    for name in expected:
        path = os.path.join(out_dir, name)
        assert os.path.exists(path), f"缺少导出文件: {name}"
        size_mb = os.path.getsize(path) / (1024 * 1024)
        assert size_mb < ex.MAX_FILE_MB


def test_build_writes_manifest_with_seed_and_cap(analysis_dir):
    ex.build(year=2020, max_points=500, seed=42)
    manifest_path = os.path.join(
        config.OUTPUT_DIR, "results", "manifest_figures", "manifest.json"
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["params"]["seed"] == 42
    assert manifest["params"]["max_points"] == 500
    assert manifest["counts"]["n_users"] == 6


def test_build_is_reproducible_end_to_end(analysis_dir):
    """同一个 seed 跑两次，抽样文件逐点相同——附录图才谈得上可复现"""
    ex.build(year=2020, max_points=50, seed=42)
    first = pd.read_parquet(
        os.path.join(ex.figure_data_dir(), ex.DIST_FILES["delay_hours"]),
        columns=ex.DIST_COLUMNS,
    )
    ex.build(year=2020, max_points=50, seed=42)
    second = pd.read_parquet(
        os.path.join(ex.figure_data_dir(), ex.DIST_FILES["delay_hours"]),
        columns=ex.DIST_COLUMNS,
    )
    pd.testing.assert_frame_equal(first, second)
