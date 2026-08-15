"""
表 A：帖子领域测量表（每条去重帖子一行）

对应研究设计 0.4 节 A 表。按月分片，供 SLURM array 逐月并行。

使用方法:
    python -m gender_domain.build_post_table month 2020 3
    python -m gender_domain.build_post_table all 2020
"""

import glob
import os

import fire
import pandas as pd
from tqdm import tqdm

from gender_domain import config
from gender_domain import id_rules as ir
from gender_domain import text_rules as tr

REQUIRED_COLUMNS = [
    "weibo_id",
    "user_id",
    "weibo_content",
    "is_retweet",
    "gender",
    "province",
    "time_stamp",
]

OUTPUT_COLUMNS = [
    "weibo_id",
    "user_id",
    "date",
    "month",
    "gender",
    "province",
    "post_type",
    "chain_stripped",
    "n_chars",
    "public_hit",
    "public_n_hits",
    "public_chars_hit",
    "public_terms",
    "public_density",
    "celebrity_hit",
    "celebrity_n_hits",
    "celebrity_chars_hit",
    "celebrity_terms",
    "celebrity_density",
]


def _drop_gender_null(df):
    """按 gender 是否为空过滤，返回 (过滤后的 df, 丢弃行数)

    单独抽出便于对丢弃计数做单元测试，也便于在 build_month 里把这一步
    的丢弃数量单独记入 manifest，不与其他丢弃来源混在一起。
    """
    before = len(df)
    filtered = df[df["gender"].notna()].copy()
    return filtered, before - len(filtered)


def _dedup_with_count(df, subset):
    """按 subset 去重，返回 (去重后的 df, 丢弃行数)

    月级跨文件去重（build_month 里的第二次去重）复用此函数，方便把丢弃
    数量单独写入 manifest，并可脱离整条流水线单独测试。
    """
    before = len(df)
    deduped = df.drop_duplicates(subset=subset, keep="first")
    return deduped, before - len(deduped)


# 时间戳单位在服务器数据里未经确认，按数值大小做防御性推断：
# 秒级时间戳（约 1.6e9，2020 年前后）远小于毫秒级时间戳（约 1.6e12）。
# 1e11 落在两者中间，足够安全地区分。
_TIMESTAMP_UNIT_THRESHOLD = 1e11


def _parse_timestamp_date(ts):
    """把原始时间戳换算成日期对象；无法解析返回 None

    None 表示"未知"，会被 diagnose_date_mismatch 计入 unknown_count 而不是
    mismatch_count——避免把脏时间戳误判成"日期不一致"。这里只做诊断，
    不影响任何输出列，date 列仍然只以文件名为准。
    """
    if ts is None:
        return None
    if isinstance(ts, float) and ts != ts:  # NaN
        return None
    try:
        value = float(ts)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    unit = "s" if value < _TIMESTAMP_UNIT_THRESHOLD else "ms"
    try:
        return pd.to_datetime(value, unit=unit).date()
    except (ValueError, OverflowError):
        return None


def diagnose_date_mismatch(weibo_ids, time_stamps, filename_date, max_examples=5):
    """诊断 time_stamp 换算日期与文件名日期是否一致（纯诊断，不改变任何输出行）

    date 列的定义不变，仍然只取自文件名（与旧基线口径一致，避免两套表
    因为 date 定义不同而无法比较）。这个函数只是提前量化文件名日期与
    帖子自带时间戳之间可能存在多大偏差，供人工判断是否需要进一步处理。

    Args:
        weibo_ids: 与 time_stamps 对齐的 weibo_id 序列
        time_stamps: 原始时间戳序列（可能是 None/NaN/字符串/数字，单位未知）
        filename_date: 文件名解析出的日期（date 对象）
        max_examples: 最多保留多少个不一致样例的 weibo_id，便于人工抽查

    Returns:
        dict，包含 mismatch_count（时间戳日期与文件名日期不一致的行数）、
        unknown_count（时间戳缺失或无法解析的行数，不计入 mismatch）、
        example_weibo_ids（最多 max_examples 个不一致样例的 weibo_id）
    """
    mismatch_count = 0
    unknown_count = 0
    examples = []
    for wid, ts in zip(weibo_ids, time_stamps):
        parsed = _parse_timestamp_date(ts)
        if parsed is None:
            unknown_count += 1
            continue
        if parsed != filename_date:
            mismatch_count += 1
            if len(examples) < max_examples:
                examples.append(wid)
    return {
        "mismatch_count": mismatch_count,
        "unknown_count": unknown_count,
        "example_weibo_ids": examples,
    }


def process_frame(df, public_matcher, celebrity_matcher):
    """对一天（或任意一批）帖子计算领域测量，纯函数便于测试"""
    df = df.drop_duplicates(subset=["weibo_id"], keep="first").copy()
    # 用 id_rules 归一化 user_id：与表 B（build_retweet_table.py）用同一套
    # 规则，保证两表 user_id 的字符串表示完全一致，Task 5 才能按 user_id
    # 直接 join，不会因为某天文件里出现缺失值导致列被上转型为 float64、
    # 再被裸 astype(str) 变成 "123.0" 这种在两表间对不上的伪 ID。
    df["user_id"] = ir.normalize_id_series(df["user_id"])

    cleaned = df["weibo_content"].map(tr.clean_text)
    df["post_type"] = [
        tr.classify_post_type(flag, text)
        for flag, text in zip(df["is_retweet"], cleaned)
    ]
    # 记录是否剥离过转发链，便于审计该规则的触发率与性别差异
    df["chain_stripped"] = df["weibo_content"].map(tr.has_retweet_chain)

    for prefix, matcher in (
        ("public", public_matcher),
        ("celebrity", celebrity_matcher),
    ):
        measures = [tr.measure_text(text, matcher) for text in cleaned]
        df[f"{prefix}_hit"] = [m["hit"] for m in measures]
        df[f"{prefix}_n_hits"] = [m["n_hits"] for m in measures]
        df[f"{prefix}_chars_hit"] = [m["n_chars_hit"] for m in measures]
        df[f"{prefix}_terms"] = ["|".join(m["terms"]) for m in measures]
        df[f"{prefix}_density"] = [m["density"] for m in measures]

    df["n_chars"] = [len(t) for t in cleaned]
    df["month"] = pd.to_datetime(df["date"]).dt.month

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def _month_files(year, month):
    pattern = os.path.join(config.DATA_DIR, str(year), f"{year}-{month:02d}-*.parquet")
    return sorted(glob.glob(pattern))


def build_month(year=config.YEAR, month=1):
    """处理一个月的日文件，写出一个分片"""
    files = _month_files(year, month)
    if not files:
        print(f"未找到 {year} 年 {month} 月的数据文件，跳过")
        return None

    public_terms = config.load_public_vocabulary(year)
    celebrity_terms = config.load_celebrity_vocabulary(year)
    public_matcher = tr.VocabMatcher(public_terms)
    celebrity_matcher = tr.VocabMatcher(celebrity_terms)
    print(f"词表规模: 公共事务 {len(public_matcher.terms)}，明星 {len(celebrity_matcher.terms)}")

    out_dir = os.path.join(config.OUTPUT_DIR, f"post_domain_measures_{year}")
    os.makedirs(out_dir, exist_ok=True)

    parts = []
    rows_in = 0
    rows_dropped_gender_null = 0
    rows_dropped_within_day_dedup = 0
    ts_mismatch_count = 0
    ts_unknown_count = 0
    ts_mismatch_examples = []
    for path in tqdm(files, desc=f"{year}-{month:02d} 帖子测量"):
        df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
        rows_in += len(df)

        filename_date_str = os.path.basename(path).replace(".parquet", "")
        filename_date = pd.to_datetime(filename_date_str).date()
        # 诊断：time_stamp 换算日期是否与文件名日期一致，只统计不改变任何行
        # （date 列仍然只取自文件名，见 diagnose_date_mismatch 的说明）
        diag = diagnose_date_mismatch(df["weibo_id"], df["time_stamp"], filename_date)
        ts_mismatch_count += diag["mismatch_count"]
        ts_unknown_count += diag["unknown_count"]
        if diag["example_weibo_ids"] and len(ts_mismatch_examples) < 5:
            ts_mismatch_examples.extend(
                diag["example_weibo_ids"][: 5 - len(ts_mismatch_examples)]
            )

        df, dropped_gender = _drop_gender_null(df)
        rows_dropped_gender_null += dropped_gender
        if len(df) == 0:
            continue
        df["date"] = filename_date_str
        rows_before_within_day_dedup = len(df)
        processed = process_frame(df, public_matcher, celebrity_matcher)
        # process_frame 内部按 weibo_id 去重（同一天内的重复帖），这里只是
        # 用前后行数差把丢弃数量单独记下来，不重复实现去重逻辑
        rows_dropped_within_day_dedup += rows_before_within_day_dedup - len(processed)
        parts.append(processed)

    if not parts:
        print(f"{year} 年 {month} 月没有可用数据")
        return None

    result = pd.concat(parts, ignore_index=True)
    result, rows_dropped_cross_file_dedup = _dedup_with_count(result, subset=["weibo_id"])
    print(f"跨文件去重：丢弃 {rows_dropped_cross_file_dedup} 行重复 weibo_id")

    out_path = os.path.join(out_dir, f"month={month:02d}.parquet")
    result.to_parquet(out_path, engine="pyarrow", index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"已保存: {out_path}（{len(result):,} 行, {size_mb:.1f} MB）")

    manifest = config.build_manifest(
        step=f"post_domain_measures_{year}_{month:02d}",
        inputs=[os.path.basename(p) for p in files],
        params={"year": year, "month": month},
        counts={
            # 逐步样本流：rows_in == 下面四项丢弃/写出之和，可据此核对流水线
            "rows_in": rows_in,
            "rows_dropped_gender_null": rows_dropped_gender_null,
            "rows_dropped_within_day_dedup": rows_dropped_within_day_dedup,
            "rows_dropped_cross_file_dedup": rows_dropped_cross_file_dedup,
            "rows_written": int(len(result)),
            "post_type": result["post_type"].value_counts().to_dict(),
            "public_hit_rate": float(result["public_hit"].mean()),
            "celebrity_hit_rate": float(result["celebrity_hit"].mean()),
            # 诊断：date（文件名口径）与 time_stamp 换算日期的偏差量级，
            # 不影响 date 列本身，仅供判断文件名日期是否可靠
            "date_timestamp_mismatch": {
                "mismatch_count": ts_mismatch_count,
                "unknown_count": ts_unknown_count,
                "example_weibo_ids": ts_mismatch_examples,
            },
        },
        fingerprints={
            "public_vocab": config.fingerprint_terms(public_terms),
            "celebrity_vocab": config.fingerprint_terms(celebrity_terms),
        },
    )
    config.write_manifest(manifest, os.path.join(out_dir, f"manifest_month_{month:02d}"))
    return out_path


def build_all(year=config.YEAR):
    """本地/单机顺序处理全年（服务器上优先用 SLURM array 按月并行）"""
    for month in range(1, 13):
        build_month(year, month)


if __name__ == "__main__":
    fire.Fire({"month": build_month, "all": build_all})
