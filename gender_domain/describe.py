"""
表 1（样本与活动特征）与表 2（原始性别差异），对应研究设计 §6.1。

只读表 C（`user_domain_{year}_with_profile.parquet`，缺失时回退到未拼接
画像的 `user_domain_{year}.parquet`），不重新推导任何度量——四个核心结果
变量（source_entered / source_share / topical_share / source_month_share）
已经在 build_user_tables.combine_user_table 里按正确的 NaN 语义算好，
本模块只是原样读取、分组汇总、套上统一的结果 schema。

设计取舍（写在这里，避免下游误以为是随手实现）：
1. NaN 不是 0：source_share/topical_share/source_month_share 在分母为 0
   （从未转发 / 无表达帖 / 无活跃月）时是 NaN，这里一律 `.notna()` 过滤，
   剔除数量写进 n_dropped，绝不 fillna(0)。source_entered 是二值列，
   对每个用户天然有定义，不存在这个问题。
2. 每个比例都标注分母（研究协议 §13.1）：source_entered 输出两个分母
   变体——"all_same_gender_users"（全体同性别用户）与
   "retweeters_only"（限定至少转发过一次的同性别用户，对应旧流水线
   "领域转发者 / 全体转发者"口径）——用 `denominator` 列显式区分，
   不让"这是对谁的比例"这件事只能靠猜。其余三个比例型结果变量分母单一，
   同样填 `denominator` 列说明是哪个子总体（例如 source_share 的分母是
   "至少转发过一次的用户"）。
3. 效应量优先、p 值不出场（研究协议全局约束）：表 2 每一行只有
   estimate/ci_low/ci_high，schema 里根本没有 p 值列。
4. 二值结果变量（source_entered）用 Wilson/Newcombe/对数尺度 risk ratio
   （全部复用 stats_utils，不重新实现）：这是"进入"这个 0/1 事件在全体
   用户里的二项比例，二项区间在这里是精确适用的。比例型结果变量
   (source_share/topical_share/source_month_share) 报告的是"用户层面
   比例的均值"（每个用户自己的转发/表达占比，再对用户取平均），不是
   "全域行为汇总后的比例"——它的抽样分布不是二项分布，Newcombe/对数尺度
   区间的前提不成立，因此均值与均值差改用 bootstrap：均值本身用
   stats_utils.bootstrap_ci；均值差 (gap_pp) 与均值之比 (risk_ratio 行)
   需要两个独立样本各自重抽样再合并统计量——stats_utils.bootstrap_ci 的
   cluster 机制解决的是"同一用户在长表里贡献多行、必须整簇重抽样"这个
   不同的问题，不能套用在"两个独立分组各自重抽样"上，所以这里用
   `_two_sample_bootstrap` 就地实现，不去改 stats_utils（模块文档明确
   不接受这种复用）。哪一行用了哪种区间方法，一律写进 `note` 列
   （"wilson"/"newcombe"/"log_scale"/"bootstrap"/"bootstrap_two_sample"）
   ——表 2 本身要能自证方法，不能指望读者看到二值结果变量用了 Newcombe
   就默认全表都是 Newcombe。
5. 头部集中度（top-1%/top-5% 份额）衡量的是"这项参与行为本身"的集中
   程度，因此用该结果变量对应的原始计数列（如 source_share/
   source_entered 共用的转发计数 `{domain}_source_count`），而不是 0/1
   或比例本身——在 0/1 上算头部份额没有信息量。按性别分别计算，在该
   性别全体用户上计算（不受 denominator 选择影响），调用
   `stats_utils.top_share`，逐用户而非逐帖计算。

使用方法:
    python -m gender_domain.describe build --year=2020
"""

import os

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gender_domain import config
from gender_domain import stats_utils as su

DOMAINS = ("public", "celebrity")

ACTIVITY_VARS = ("n_posts", "n_retweets", "n_active_days", "n_active_months")
_PERCENTILES = (0.75, 0.90, 0.95, 0.99)

# 三个比例型结果变量：(结果变量名, 头部集中度用的原始计数列模板, 缺失原因)
_PROPORTION_OUTCOMES = (
    ("source_share", "{domain}_source_count", "no_retweets", "users_with_retweets"),
    ("topical_share", "{domain}_topical_posts", "no_expressive_posts", "users_with_expressive_posts"),
    ("source_month_share", "{domain}_source_months", "no_active_months", "users_with_active_months"),
)

_TOP_SHARE_QS = ((0.01, "top1_share"), (0.05, "top5_share"))

_BOOT_SEED = 0
_BOOT_N = 1000


# ---------------------------------------------------------------------------
# 表 1：样本与活动特征
# ---------------------------------------------------------------------------

def sample_activity_table(user_df):
    """表 1：按性别列出样本量，以及帖数/转发数/活跃天数/活跃月数的分布

    每个 (性别, 活动变量) 一行，列出 mean/median/p75/p90/p95/p99。这四个
    活动变量在表 C 里对每个用户都有定义（没有表达帖/转发也是 0，不是
    缺失），因此这里不做 NaN 剔除。
    """
    rows = []
    for gender, group in user_df.groupby("gender", dropna=False):
        n_users = len(group)
        for var in ACTIVITY_VARS:
            series = group[var].astype(float)
            quantiles = series.quantile(list(_PERCENTILES))
            rows.append({
                "gender": gender,
                "n_users": n_users,
                "variable": var,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p75": float(quantiles.loc[0.75]),
                "p90": float(quantiles.loc[0.90]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 表 2：原始性别差异
# ---------------------------------------------------------------------------

def _two_sample_bootstrap(male_values, female_values, statistic, n_boot=_BOOT_N, seed=_BOOT_SEED,
                           confidence=0.95):
    """两个独立样本各自重抽样后代入 statistic，取百分位区间

    与 stats_utils.bootstrap_ci 的 cluster 重抽样解决的是不同的问题
    （见模块文档第 4 条），这里就地实现，不复用也不改动那个函数。
    点估计 (est) 直接在原始（未重抽样）的两份样本上计算，重抽样只用来
    刻画区间，这一点与 stats_utils.bootstrap_ci 的约定保持一致。
    """
    m = np.asarray(male_values, dtype=float)
    f = np.asarray(female_values, dtype=float)
    point = statistic(m, f)
    rng = np.random.default_rng(seed)
    n_m, n_f = len(m), len(f)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        bm = m[rng.integers(0, n_m, size=n_m)]
        bf = f[rng.integers(0, n_f, size=n_f)]
        boot[b] = statistic(bm, bf)
    alpha = 1 - confidence
    low, high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(low), float(high)


def _make_row(outcome, domain, term, estimate, ci_low, ci_high, scale, n_obs, n_dropped,
              drop_reason, denominator, note=None):
    """套上共享结果 schema（tidy_result），再补上表 2 专属的 denominator 列

    denominator 不在 stats_utils.RESULT_SCHEMA 里——那是给 models_*.parquet
    用的通用 schema，表 2 需要额外标注分母，tidy_result 会拒绝未知字段，
    所以在拿到 tidy_result 返回的 dict 之后再补这一列，而不是塞进
    tidy_result 的调用参数。
    """
    row = su.tidy_result(
        outcome=outcome, domain=domain, model="raw", term=term,
        estimate=estimate, se=None, ci_low=ci_low, ci_high=ci_high,
        scale=scale, n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
        note=note,
    )
    row["denominator"] = denominator
    return row


def _top_share_rows(outcome, domain, count_col, male_df, female_df):
    """按性别分别算头部 1%/5% 用户对该结果变量对应行为总量的贡献份额"""
    rows = []
    for gender_label, sub in (("male", male_df), ("female", female_df)):
        values = sub[count_col].astype(float).to_numpy()
        for q, term_prefix in _TOP_SHARE_QS:
            share = su.top_share(values, q)
            rows.append(_make_row(
                outcome, domain, f"{term_prefix}_{gender_label}", share,
                None, None, "share_of_behavior", len(sub), 0, None, "n/a",
            ))
    return rows


def _binary_outcome_rows(user_df, domain, male_mask, female_mask):
    """source_entered：二值结果变量，两个分母变体 + 头部集中度"""
    outcome = "source_entered"
    outcome_col = f"{domain}_source_entered"
    count_col = f"{domain}_source_count"

    male_df = user_df[male_mask]
    female_df = user_df[female_mask]
    rows = []

    denom_specs = (
        ("all_same_gender_users", None),
        ("retweeters_only", "n_retweets"),
    )
    for denom_label, filter_col in denom_specs:
        m_sub = male_df if filter_col is None else male_df[male_df[filter_col] > 0]
        f_sub = female_df if filter_col is None else female_df[female_df[filter_col] > 0]
        s_m, n_m = int(m_sub[outcome_col].sum()), len(m_sub)
        s_f, n_f = int(f_sub[outcome_col].sum()), len(f_sub)
        n_dropped_m = len(male_df) - n_m
        n_dropped_f = len(female_df) - n_f
        drop_reason = "excluded_by_denominator" if filter_col is not None else None

        p_m_low, p_m_high = su.proportion_ci(s_m, n_m) if n_m else (np.nan, np.nan)
        p_f_low, p_f_high = su.proportion_ci(s_f, n_f) if n_f else (np.nan, np.nan)
        rows.append(_make_row(
            outcome, domain, "male", s_m / n_m if n_m else np.nan, p_m_low, p_m_high,
            "probability", n_m, n_dropped_m, drop_reason if n_dropped_m else None, denom_label,
            note="wilson",
        ))
        rows.append(_make_row(
            outcome, domain, "female", s_f / n_f if n_f else np.nan, p_f_low, p_f_high,
            "probability", n_f, n_dropped_f, drop_reason if n_dropped_f else None, denom_label,
            note="wilson",
        ))

        diff, d_low, d_high = su.proportion_diff_ci(s_m, n_m, s_f, n_f)
        rows.append(_make_row(
            outcome, domain, "gap_pp", diff, d_low, d_high, "probability",
            n_m + n_f, n_dropped_m + n_dropped_f, None, denom_label, note="newcombe",
        ))

        rr, rr_low, rr_high = su.risk_ratio_ci(s_m, n_m, s_f, n_f)
        rows.append(_make_row(
            outcome, domain, "risk_ratio", rr, rr_low, rr_high, "ratio",
            n_m + n_f, n_dropped_m + n_dropped_f, None, denom_label, note="log_scale",
        ))

    rows.extend(_top_share_rows(outcome, domain, count_col, male_df, female_df))
    return rows


def _proportion_outcome_rows(user_df, domain, outcome, count_col_tpl, drop_reason, denominator,
                              male_mask, female_mask):
    """source_share / topical_share / source_month_share：均值 + bootstrap CI"""
    outcome_col = f"{domain}_{outcome}"
    count_col = count_col_tpl.format(domain=domain)

    male_df = user_df[male_mask]
    female_df = user_df[female_mask]
    male_valid = male_df[male_df[outcome_col].notna()]
    female_valid = female_df[female_df[outcome_col].notna()]
    n_dropped_m = len(male_df) - len(male_valid)
    n_dropped_f = len(female_df) - len(female_valid)

    m_vals = male_valid[outcome_col].to_numpy(dtype=float)
    f_vals = female_valid[outcome_col].to_numpy(dtype=float)

    m_est, m_low, m_high = su.bootstrap_ci(m_vals, np.mean, n_boot=_BOOT_N, seed=_BOOT_SEED)
    f_est, f_low, f_high = su.bootstrap_ci(f_vals, np.mean, n_boot=_BOOT_N, seed=_BOOT_SEED)

    rows = [
        _make_row(
            outcome, domain, "male", m_est, m_low, m_high, "proportion",
            len(male_valid), n_dropped_m, drop_reason if n_dropped_m else None, denominator,
            note="bootstrap",
        ),
        _make_row(
            outcome, domain, "female", f_est, f_low, f_high, "proportion",
            len(female_valid), n_dropped_f, drop_reason if n_dropped_f else None, denominator,
            note="bootstrap",
        ),
    ]

    diff_point, diff_low, diff_high = _two_sample_bootstrap(
        m_vals, f_vals, lambda m, f: m.mean() - f.mean(),
    )
    rows.append(_make_row(
        outcome, domain, "gap_pp", diff_point, diff_low, diff_high, "proportion",
        len(male_valid) + len(female_valid), n_dropped_m + n_dropped_f, None, denominator,
        note="bootstrap_two_sample",
    ))

    def _ratio(m, f):
        f_mean = f.mean()
        return m.mean() / f_mean if f_mean != 0 else np.nan

    ratio_point, ratio_low, ratio_high = _two_sample_bootstrap(m_vals, f_vals, _ratio)
    rows.append(_make_row(
        outcome, domain, "risk_ratio", ratio_point, ratio_low, ratio_high, "ratio",
        len(male_valid) + len(female_valid), n_dropped_m + n_dropped_f, None, denominator,
        note="bootstrap_two_sample",
    ))

    rows.extend(_top_share_rows(outcome, domain, count_col, male_df, female_df))
    return rows


def raw_gender_gaps(user_df):
    """表 2：四个核心结果变量 x 两个领域的原始性别差异

    每个结果变量在每个领域下产出多行（男/女取值、pp 差距、risk ratio、
    头部集中度），用 term 列区分具体是哪个量，用 denominator 列区分分母。
    绝不产出 p 值列——效应量与区间已经是全部内容。
    """
    male_mask = user_df["gender"] == "m"
    female_mask = user_df["gender"] == "f"

    rows = []
    for domain in DOMAINS:
        rows.extend(_binary_outcome_rows(user_df, domain, male_mask, female_mask))
        for outcome, count_tpl, drop_reason, denominator in _PROPORTION_OUTCOMES:
            rows.extend(_proportion_outcome_rows(
                user_df, domain, outcome, count_tpl, drop_reason, denominator,
                male_mask, female_mask,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_user_table(year):
    """优先读画像拼接后的表 C，缺失则回退到未拼接画像的表 C（Task 1 输出）"""
    profile_path = os.path.join(config.OUTPUT_DIR, f"user_domain_{year}_with_profile.parquet")
    base_path = os.path.join(config.OUTPUT_DIR, f"user_domain_{year}.parquet")
    path = profile_path if os.path.exists(profile_path) else base_path
    columns = pq.ParquetFile(path).schema.names
    df = pd.read_parquet(path, columns=columns)
    print(f"读取用户表 {path}（{len(df):,} 用户）")
    return df, path


def build(year=config.YEAR):
    """生成表 1、表 2 并写出 manifest"""
    user_df, source_path = _load_user_table(year)
    table1 = sample_activity_table(user_df)
    table2 = raw_gender_gaps(user_df)

    out_dir = os.path.join(config.OUTPUT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    table1_path = os.path.join(out_dir, "table1_sample_activity.parquet")
    table2_path = os.path.join(out_dir, "table2_raw_gender_gaps.parquet")
    table1.to_parquet(table1_path, engine="pyarrow", index=False)
    table2.to_parquet(table2_path, engine="pyarrow", index=False)
    print(f"已保存: {table1_path}（{len(table1)} 行）")
    print(f"已保存: {table2_path}（{len(table2)} 行）")

    manifest = config.build_manifest(
        step=f"describe_{year}",
        inputs=[os.path.relpath(source_path, config.OUTPUT_DIR)],
        params={"year": year},
        counts={
            "n_users": int(len(user_df)),
            "gender": user_df["gender"].value_counts().to_dict(),
            "table1_rows": int(len(table1)),
            "table2_rows": int(len(table2)),
        },
    )
    # 每个步骤自己的 manifest 子目录，不直接写进 analysis_data/results/：
    # write_manifest 会覆盖同目录下已有的 manifest.json，而后续任务
    # （models_core/models_interaction/...）都会往同一个 results/ 目录
    # 写结果，如果大家的 manifest 都落在 results/ 根目录，最后一个跑的
    # 模块会把前面所有模块的溯源记录悄悄抹掉——这与 Plan 1 里表 A 按月
    # 分片各自建 manifest_month_NN/ 子目录是同一个约定，这里延续它。
    config.write_manifest(manifest, os.path.join(out_dir, "manifest_describe"))
    return table1_path, table2_path


if __name__ == "__main__":
    fire.Fire({"build": build})
