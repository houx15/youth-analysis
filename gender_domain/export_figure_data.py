"""
§12 图数据导出：把服务器上算好的结果搬成一份"能下载"的小文件集。

为什么要有这一层（与旧流水线的 export_viz_data.py 同一个理由）：分析数据
只存在于服务器上，体量根本搬不动；而画图是需要反复改的工作，只能在本地
迭代。所以服务器上跑完结果之后，这里把两类东西写进
analysis_data/figure_data/：

1. **结果表原样搬运**。Tasks 3–7 写出的 12 张结果表本来就只有几十到几百行，
   直接复制过去即可，不做任何再加工——图这一层不许重算任何一个数字。
2. **附录 ECDF 需要的分布抽样**。转发次数、转发延迟、字符 density 三份
   分布是用户级/记录级的原始取值，全量有上百万行，下载不动。这里按
   性别 × 领域每格抽最多 max_points 个点，seed 固定并写进 manifest 与
   汇总表，于是任何时候都能抽出逐点相同的一份样本。

关于抽样的三条硬性口径：

- **抽样是随机的，但必须可复现**。每一格的随机数发生器由 (seed, 格子名)
  共同决定，而不是"一个全局 rng 顺着遍历顺序往下抽"——后者只要格子的
  遍历顺序变了（比如某年某个领域一条记录都没有），所有格子的样本都会跟着
  变，同一个 seed 也不再意味着同一份样本。
- **抽样必须保形**。附录 ECDF 是用来展示分布形状的，如果抽样把尾巴削掉，
  那张图就是错的。均匀随机抽样在每一格内是无偏的，抽出的分位数与全量
  分位数一致到抽样误差内；为了让这件事可以被事后核对，汇总表把全量分位数
  与抽样后的分位数并排写出来，任何人都能自己比一遍。
- **NaN 不是 0**。没有表达帖的用户 char_density 无定义，这类取值只能记进
  n_missing，绝不能填 0 混进分布——那会在 ECDF 的左端凭空堆出一根柱子。

另外这里补了一张结果层没有的小表 combo_distribution.parquet：§8 的多项
logit 只写出了各类别的性别平均边际效应（男−女），没有写出各性别自身的
类别占比，而 §12.5 要求图 5 先展示"四类来源组合的性别分布"。这里的做法是
读表 C 已有的 source_combo 列（由 build_user_tables 生成，本模块不重新
推导它），按性别做一次占比 + Wilson 区间，与 describe.py 里表 2 的做法完全
一致，用的是同一套 stats_utils 函数。

用法：
  python -m gender_domain.export_figure_data build --year=2020
"""

import os
import zlib

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gender_domain import config
from gender_domain import models_temporal as mt
from gender_domain import stats_utils as su

# 输出目录名（在 config.OUTPUT_DIR 之下）。全局约束：本模块只往这里写。
FIGURE_DATA_DIRNAME = "figure_data"
RESULTS_DIRNAME = "results"
MANIFEST_DIRNAME = "manifest_figures"

# Tasks 3–7 写出的结果表，全部原样搬运。缺一张就报错：图这一层不允许
# "少一张表就少画一张图"地静默降级，那样图与表会对不上。
RESULT_FILES = (
    "table1_sample_activity.parquet",
    "table2_raw_gender_gaps.parquet",
    "models_entry.parquet",
    "models_intensity.parquet",
    "models_share.parquet",
    "models_persistence.parquet",
    "interaction_gender_domain.parquet",
    "decomposition_source_content.parquet",
    "combination_multinomial.parquet",
    "monthly_rates.parquet",
    "leave_one_month_out.parquet",
    "delay_quantiles.parquet",
)

# 三份分布样本
DIST_FILES = {
    "retweet_count": "dist_retweet_count.parquet",
    "delay_hours": "dist_delay_hours.parquet",
    "char_density": "dist_char_density.parquet",
}
SUMMARY_FILE = "distribution_summary.parquet"
COMBO_FILE = "combo_distribution.parquet"

DIST_COLUMNS = ["measure", "gender", "domain", "value"]
GROUP_KEYS = ["measure", "gender", "domain"]

DOMAINS = mt.DOMAINS                       # ("public", "celebrity")
GENDER_LABELS = (("m", "male"), ("f", "female"))
COMBO_CATEGORIES = ("neither", "public_only", "celebrity_only", "both")
COMBO_OUTCOME = "source_combo_observed"

# 每个 性别 × 领域 格子最多导出多少个点。5 万点足以把 ECDF 画到肉眼分辨
# 不出与全量的差别（分位数误差在 1/sqrt(5e4) ≈ 0.4% 量级），三份分布合计
# 也只有几 MB，下载无压力。
MAX_POINTS_PER_GROUP = 50000
SAMPLE_SEED = 20200101
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
# 单个导出文件的体量上限（MB）。超过就说明抽样口径设错了，宁可当场报错，
# 也不要让人下载到一半才发现文件有几个 G。
MAX_FILE_MB = 20.0

# 附录延迟 ECDF 用哪个阈值口径。§10.3 的两个口径都写进了 delay_quantiles，
# 但 ECDF 只画主口径（720 小时），并在汇总表里记下用的是哪一个。
DELAY_THRESHOLD = mt.MAX_DELAY_HOURS


def figure_data_dir():
    """图数据输出目录（只此一个写入位置）"""
    return os.path.join(config.OUTPUT_DIR, FIGURE_DATA_DIRNAME)


def results_dir():
    """结果表所在目录（只读）"""
    return os.path.join(config.OUTPUT_DIR, RESULTS_DIRNAME)


def quantile_column(level, kind):
    """分位数列名，kind 取 "full"（全量）或 "sampled"（抽样后）"""
    return "q{:g}_{}".format(level * 100, kind)


def _save_parquet(frame, path):
    """写出 parquet 并打印体量；超过上限时删掉文件再报错

    超限必须连文件一起清掉：留在盘上的话，下一次运行只要跳过这一步
    （或者有人直接去下载目录取文件），就会拿到一个本次运行明确判定为
    "太大、不该存在"的文件，而错误信息早已滚出屏幕。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print("  已保存: {}（{:,} 行, {:.2f} MB）".format(path, len(frame), size_mb))
    if size_mb > MAX_FILE_MB:
        os.remove(path)
        raise ValueError(
            "导出文件 {} 达到 {:.1f} MB，超过 {:.1f} MB 上限：这层的存在意义就是"
            "让文件小到能下载，超限说明抽样口径设错了。该文件已删除，"
            "不会留在下载目录里".format(path, size_mb, MAX_FILE_MB)
        )
    return size_mb


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------

def _group_rng(seed, key):
    """由 (seed, 格子名) 共同决定的随机数发生器

    用 crc32 把格子名折成一个整数，与 seed 一起作为 SeedSequence 的熵。
    这样每一格的样本只取决于它自己的名字与 seed，与遍历顺序、与别的格子
    有没有数据都无关——"同一个 seed 得到同一份样本"这个承诺才站得住。
    """
    digest = zlib.crc32(str(key).encode("utf-8"))
    return np.random.default_rng([int(seed), int(digest)])


def sample_values(values, cap, seed, key):
    """从一格取值里等概率不放回地抽最多 cap 个点，顺序保持原样

    抽完之后按原始顺序（而不是抽样顺序）返回，纯粹是为了让写出的文件
    在同一个 seed 下逐字节稳定，便于核对。
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= cap:
        return values
    idx = _group_rng(seed, key).choice(len(values), size=int(cap), replace=False)
    idx.sort()
    return values[idx]


def sample_distribution(long_df, cap=MAX_POINTS_PER_GROUP, seed=SAMPLE_SEED):
    """按 measure × gender × domain 分格抽样

    Args:
        long_df: 列为 DIST_COLUMNS 的长表
        cap: 每格保留的最大点数
        seed: 随机种子，写进汇总表

    Returns:
        (sampled_df, summary_df)。summary_df 记录每一格的全量样本量、
        缺失数、抽样数、cap、seed，以及全量与抽样后的分位数——附录图的
        分母、标注和"抽样是否保形"的核对都靠它。
    """
    missing_columns = [c for c in DIST_COLUMNS if c not in long_df.columns]
    if missing_columns:
        raise KeyError("分布长表缺少列: {}".format(missing_columns))

    sampled_parts, summary_rows = [], []
    for keys, group in long_df.groupby(GROUP_KEYS, sort=True):
        measure, gender, domain = keys
        raw = pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float)
        # NaN 是"这个量对该用户无定义"，不是 0：单独记数，不进分布
        finite = raw[np.isfinite(raw)]
        n_missing = int(len(raw) - len(finite))
        key = "{}|{}|{}".format(measure, gender, domain)
        drawn = sample_values(finite, cap, seed, key)

        sampled_parts.append(pd.DataFrame({
            "measure": measure, "gender": gender, "domain": domain, "value": drawn,
        }))
        row = {
            "measure": measure, "gender": gender, "domain": domain,
            "n_total": int(len(finite)), "n_missing": n_missing,
            "n_sampled": int(len(drawn)), "cap": int(cap), "seed": int(seed),
        }
        for level in QUANTILE_LEVELS:
            row[quantile_column(level, "full")] = (
                float(np.quantile(finite, level)) if len(finite) else np.nan)
            row[quantile_column(level, "sampled")] = (
                float(np.quantile(drawn, level)) if len(drawn) else np.nan)
        summary_rows.append(row)

    if not sampled_parts:
        sampled = pd.DataFrame({c: pd.Series([], dtype="object" if c != "value"
                                             else "float64")
                                for c in DIST_COLUMNS})
        return sampled, pd.DataFrame(summary_rows)

    sampled = pd.concat(sampled_parts, ignore_index=True)[DIST_COLUMNS]
    return sampled, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# 三份分布的构造（都只是把表 C / 表 B 已有的列摊成长表，不新造度量）
# ---------------------------------------------------------------------------

def _known_gender(frame):
    """只保留 m/f 两组；第三性别与缺失在这里被排除（与结果层同一口径）"""
    gender = frame["gender"].astype("object")
    return frame[gender.isin([g for g, _ in GENDER_LABELS])]


def _long_from_user_columns(user_df, measure, column_template):
    rows = []
    frame = _known_gender(user_df)
    for domain in DOMAINS:
        column = column_template.format(domain=domain)
        if column not in frame.columns:
            raise KeyError("表 C 缺少列 {}".format(column))
        rows.append(pd.DataFrame({
            "measure": measure,
            "gender": frame["gender"].astype(str).to_numpy(),
            "domain": domain,
            "value": pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
        }))
    return pd.concat(rows, ignore_index=True)[DIST_COLUMNS]


def retweet_count_distribution(user_df):
    """来源转发次数（表 C 的 {domain}_source_count），0 是有定义的事实，保留"""
    return _long_from_user_columns(user_df, "retweet_count", "{domain}_source_count")


def density_distribution(user_df):
    """用户级字符 density（表 C 的 {domain}_char_density）

    没有表达帖的用户这一列是 NaN——无定义，不是 0。这里原样带出 NaN，
    由 sample_distribution 记进 n_missing 并排除。
    """
    return _long_from_user_columns(user_df, "char_density", "{domain}_char_density")


def delay_distribution(clean_df):
    """记录级转发延迟（小时），来自 models_temporal.clean_delays 的清理结果

    延迟的清理规则（缺失/非正/跨年/超长/重复五条）只有一处定义，就是
    models_temporal.clean_delays；这里绝不重写，只消费它的输出。
    """
    if "delay_hours" not in clean_df.columns:
        raise KeyError("清理后的事件表缺少 delay_hours 列")
    frame = _known_gender(clean_df)
    frame = frame[frame["source_domain"].isin(DOMAINS)]
    return pd.DataFrame({
        "measure": "delay_hours",
        "gender": frame["gender"].astype(str).to_numpy(),
        "domain": frame["source_domain"].astype(str).to_numpy(),
        "value": pd.to_numeric(frame["delay_hours"], errors="coerce").to_numpy(
            dtype=float),
    })[DIST_COLUMNS]


# ---------------------------------------------------------------------------
# 图 5 左半边：四类来源组合的观测分布
# ---------------------------------------------------------------------------

def combo_distribution(user_df):
    """按性别给出 source_combo 四类的占比与 Wilson 区间（共享结果 schema）

    source_combo 是表 C 已有的列（build_user_tables 生成），本函数只做
    "分组求占比 + 区间"，与 describe.py 的表 2 同一套做法、同一套
    stats_utils 函数。domain 一律记 "both"：来源组合本来就是跨领域的
    分类，硬塞进 public/celebrity 任一侧都是错的。
    """
    if "source_combo" not in user_df.columns:
        raise KeyError(
            "表 C 缺少 source_combo 列：四类来源组合由 build_user_tables 算好，"
            "本模块不重新推导它"
        )
    n_input = int(len(user_df))
    gender = user_df["gender"].astype("object")
    combo = user_df["source_combo"].astype("object")
    known_combo = combo.isin(COMBO_CATEGORIES)

    n_missing_gender = int(gender.isna().sum())
    n_other_gender = int((~gender.isna() & ~gender.isin(["m", "f"])).sum())

    rows = []
    for code, label in GENDER_LABELS:
        mask = (gender == code) & known_combo
        n_obs = int(mask.sum())
        n_unknown_combo = int(((gender == code) & ~known_combo).sum())
        reasons = []
        if n_missing_gender:
            reasons.append("missing_gender={}".format(n_missing_gender))
        if n_other_gender:
            reasons.append("other_gender={}".format(n_other_gender))
        if n_unknown_combo:
            reasons.append("unknown_source_combo={}".format(n_unknown_combo))
        # 另一性别的用户对这一行来说也是"没有进入这个估计"的样本，
        # 必须记进 n_dropped，否则 n_obs + n_dropped 对不上总样本
        n_other_side = n_input - n_obs - n_missing_gender - n_other_gender \
            - n_unknown_combo
        if n_other_side:
            reasons.append("other_gender_group={}".format(n_other_side))
        drop_reason = "+".join(reasons) if reasons else None

        for category in COMBO_CATEGORIES:
            successes = int((mask & (combo == category)).sum())
            if n_obs:
                estimate = successes / n_obs
                low, high = su.proportion_ci(successes, n_obs)
            else:
                estimate, low, high = np.nan, np.nan, np.nan
            rows.append(su.tidy_result(
                outcome=COMBO_OUTCOME, domain="both", model="raw",
                term="{}_{}".format(category, label),
                estimate=estimate, se=None, ci_low=low, ci_high=high,
                scale="probability", n_obs=n_obs, n_dropped=n_input - n_obs,
                drop_reason=drop_reason, note="wilson",
            ))
    return pd.DataFrame(rows)[list(su.RESULT_SCHEMA)]


# ---------------------------------------------------------------------------
# 结果表搬运
# ---------------------------------------------------------------------------

def copy_result_tables(out_dir=None):
    """把 12 张结果表原样复制到 figure_data/，返回 (路径, MB) 列表

    读的时候显式给 columns=（全局约束）：结果表的列集合是已知的，这里
    直接按文件里的 schema 全列读入，但仍然显式传参，不走"默认读全部"。
    """
    src_dir = results_dir()
    out_dir = out_dir or figure_data_dir()
    missing = [name for name in RESULT_FILES
               if not os.path.exists(os.path.join(src_dir, name))]
    if missing:
        raise FileNotFoundError(
            "结果目录 {} 缺少以下结果表: {}。请先在服务器上跑完 "
            "slurm/run_results.slurm（gender_domain.describe / models_* 各模块）"
            "再导出图数据".format(src_dir, ", ".join(missing))
        )

    outputs = []
    for name in RESULT_FILES:
        src = os.path.join(src_dir, name)
        # 先读 schema 拿到列名，再显式传 columns= 正式读入（全局约束：
        # parquet 一律显式给列，不走"默认读全部"）
        columns = pq.ParquetFile(src).schema.names
        frame = pd.read_parquet(src, columns=columns)
        dst = os.path.join(out_dir, name)
        outputs.append((dst, _save_parquet(frame, dst)))
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

USER_TABLE_COLUMNS = ["user_id", "gender", "source_combo"] + [
    "{}_{}".format(domain, suffix)
    for domain in DOMAINS
    for suffix in ("source_count", "char_density")
]


def _load_user_table(year):
    """优先读画像拼接后的表 C，缺失则回退到未拼接画像的表 C"""
    profile_path = os.path.join(
        config.OUTPUT_DIR, "user_domain_{}_with_profile.parquet".format(year))
    base_path = os.path.join(
        config.OUTPUT_DIR, "user_domain_{}.parquet".format(year))
    path = profile_path if os.path.exists(profile_path) else base_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到表 C（{} 或 {}）：图数据导出必须在 build_user_tables / "
            "profile_join 之后运行".format(profile_path, base_path)
        )
    frame = pd.read_parquet(path, columns=USER_TABLE_COLUMNS)
    print("读取用户表 {}（{:,} 用户）".format(path, len(frame)))
    return frame, path


def build(year=config.YEAR, max_points=MAX_POINTS_PER_GROUP, seed=SAMPLE_SEED):
    """导出全部图数据，并写出 manifest

    Args:
        year: 数据年份
        max_points: 每个 性别 × 领域 格子最多导出多少个分布点
        seed: 抽样随机种子，写进 manifest 与汇总表
    """
    out_dir = figure_data_dir()
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("导出 {} 年图数据到 {}".format(year, out_dir))
    print("=" * 60)

    print("\n[1/4] 复制结果表")
    copied = copy_result_tables(out_dir)

    print("\n[2/4] 用户级分布（转发次数、字符 density）与来源组合分布")
    user_df, user_path = _load_user_table(year)
    combo = combo_distribution(user_df)
    _save_parquet(combo, os.path.join(out_dir, COMBO_FILE))

    # measure 由调用处直接给出，不从帧里的第一行反推：某一份分布如果一行
    # 都没有（例如某年某个领域完全没有事件），反推会得到 None，接着在
    # DIST_FILES[None] 上抛一个看不出前因后果的 KeyError
    long_frames = [
        ("retweet_count", retweet_count_distribution(user_df)),
        ("char_density", density_distribution(user_df)),
    ]

    print("\n[3/4] 记录级转发延迟（复用 models_temporal 的清理规则）")
    frames, reports, event_files = mt.load_clean_events(
        year, thresholds=[DELAY_THRESHOLD])
    label = mt.threshold_label(DELAY_THRESHOLD)
    long_frames.append(("delay_hours", delay_distribution(frames[label])))

    print("\n[4/4] 分格抽样并写出")
    summaries = []
    for measure, long_df in long_frames:
        if measure not in DIST_FILES:
            raise KeyError(
                "未知的分布名 {!r}，可用的是 {}".format(
                    measure, sorted(DIST_FILES)))
        if long_df.empty:
            raise ValueError(
                "{} 的长表一行都没有：{} 年的输入里这份分布是空的，"
                "空文件导出去只会让本地画出一张空图，先查上游为什么没有数据".format(
                    measure, year)
            )
        sampled, summary = sample_distribution(long_df, cap=max_points, seed=seed)
        summaries.append(summary)
        _save_parquet(sampled, os.path.join(out_dir, DIST_FILES[measure]))
    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all["delay_threshold"] = label
    _save_parquet(summary_all, os.path.join(out_dir, SUMMARY_FILE))

    total_mb = sum(
        os.path.getsize(os.path.join(out_dir, f)) / (1024 * 1024)
        for f in os.listdir(out_dir)
    )
    print("\n导出完成，共 {} 个文件，合计 {:.2f} MB".format(
        len(os.listdir(out_dir)), total_mb))

    manifest = config.build_manifest(
        step="export_figure_data_{}".format(year),
        inputs=[os.path.relpath(user_path, config.OUTPUT_DIR)]
        + [os.path.relpath(p, config.OUTPUT_DIR) for p in event_files]
        + [os.path.join(RESULTS_DIRNAME, name) for name in RESULT_FILES],
        params={
            "year": year,
            "max_points": int(max_points),
            "seed": int(seed),
            "delay_threshold": label,
            "quantile_levels": list(QUANTILE_LEVELS),
        },
        counts={
            "n_users": int(len(user_df)),
            "n_result_tables": len(copied),
            "n_sampled_by_group": {
                "{}|{}|{}".format(r["measure"], r["gender"], r["domain"]):
                    int(r["n_sampled"])
                for _, r in summary_all.iterrows()
            },
            "n_total_by_group": {
                "{}|{}|{}".format(r["measure"], r["gender"], r["domain"]):
                    int(r["n_total"])
                for _, r in summary_all.iterrows()
            },
            "total_mb": round(total_mb, 3),
            "delay_clean": {
                "n_input": reports[label]["n_input"],
                "n_clean": reports[label]["n_clean"],
            },
        },
    )
    config.write_manifest(manifest, os.path.join(results_dir(), MANIFEST_DIRNAME))
    return out_dir


if __name__ == "__main__":
    fire.Fire({"build": build})
