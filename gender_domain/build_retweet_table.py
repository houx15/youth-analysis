"""
表 B：来源转发事件表（每条符合条件的转发一行）

对应研究设计 0.4 节 B 表。保留非正延迟记录但标记 delay_valid=False，
清理规则留给分析阶段，原始事件表不做删除。

使用方法:
    python -m gender_domain.build_retweet_table month 2020 5
    python -m gender_domain.build_retweet_table all 2020
"""

import glob
import os

import fire
import pandas as pd
from tqdm import tqdm

from gender_domain import config
from gender_domain import id_rules as ir

REQUIRED_COLUMNS = [
    "user_id",
    "weibo_id",
    "r_weibo_id",
    "r_user_id",
    "is_retweet",
    "gender",
    "time_stamp",
    "r_time_stamp",
]

OUTPUT_COLUMNS = [
    "user_id",
    "weibo_id",
    "r_weibo_id",
    "r_user_id",
    "date",
    "month",
    "gender",
    "source_domain",
    "source_category",
    "retweet_ts",
    "source_ts",
    "delay_seconds",
    "delay_valid",
]


def build_account_lookup(accounts):
    """{类别: {user_id}} -> {user_id: "类别1|类别2"}

    同一账号出现在多个类别时，类别名按字典序拼接，保证结果与输入顺序无关。
    """
    lookup = {}
    for category, ids in accounts.items():
        for uid in ids:
            uid = str(uid)
            if uid in lookup:
                lookup[uid] = "|".join(sorted(set(lookup[uid].split("|")) | {category}))
            else:
                lookup[uid] = category
    return lookup


def _drop_gender_null(df):
    """按 gender 是否为空过滤，返回 (过滤后的 df, 丢弃行数)

    单独抽出便于对丢弃计数做单元测试，与表 A 的同名函数保持一致的记账习惯。
    """
    before = len(df)
    filtered = df[df["gender"].notna()].copy()
    return filtered, before - len(filtered)


def _filter_retweets(df):
    """只保留转发记录（is_retweet 是字符串 "1"），返回 (筛选后的 df, 丢弃行数)"""
    before = len(df)
    filtered = df[df["is_retweet"].astype(str) == "1"].copy()
    return filtered, before - len(filtered)


def _exclude_source_self_retweets(df, known_sources):
    """排除来源账号自己发出的转发（自己转自己或转发另一个来源账号的行为
    不代表"某用户消费了该来源"），返回 (筛选后的 df, 丢弃行数)

    这里用 id_rules 归一化 user_id 而不是裸 astype(str)：user_id 列一旦
    当天文件里存在缺失值，会被 pandas 上转型为 float64，astype(str) 会把
    "123" 变成 "123.0"，导致下面的 isin(known_sources) 对整批行静默失配。
    """
    before = len(df)
    df = df.copy()
    df["user_id"] = ir.normalize_id_series(df["user_id"])
    filtered = df[~df["user_id"].isin(known_sources)]
    return filtered, before - len(filtered)


def _domain_rows(df, lookup, domain):
    """取出转发该领域来源账号的记录，标注领域与来源类别；无命中返回 None

    r_user_id 在服务器原始数据里按文档是字符串类型，但未针对真实文件核实
    过（与表 A 对 time_stamp 单位未核实、只能做防御性推断的情况相同）；
    process_frame 已经用 id_rules 统一归一化过，这里按归一化后的字符串匹配。
    """
    if not lookup:
        return None
    hit = df[df["r_user_id"].isin(lookup.keys())].copy()
    if len(hit) == 0:
        return None
    hit["source_domain"] = domain
    hit["source_category"] = hit["r_user_id"].map(lookup)
    return hit


def process_frame(df, public_lookup, celebrity_lookup):
    """对一批帖子提取来源转发事件，纯函数便于测试

    一条转发命中公共事务和明星两个领域的来源账号时（账号在两份名单中都
    出现），会各产出一行，即输出可能比"符合条件的转发数"更多。
    """
    df = df.copy()
    # 用 id_rules 而不是裸 astype(str)：ID 列一旦经过缺失值上转型为
    # float64，astype(str) 会产出 "123.0" 这种伪 ID，且不会报错，只是
    # 后续所有基于字符串的匹配（isin/表 A·表 B 的 user_id join）静默失效。
    # 两张表都必须用同一套归一化，才能保证 user_id 的字符串表示一致可 join。
    df["user_id"] = ir.normalize_id_series(df["user_id"])
    df["r_user_id"] = ir.normalize_id_series(df["r_user_id"])

    df, _ = _filter_retweets(df)
    known_sources = set(public_lookup.keys()) | set(celebrity_lookup.keys())
    df, _ = _exclude_source_self_retweets(df, known_sources)
    if len(df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    parts = []
    for lookup, domain in ((public_lookup, "public"), (celebrity_lookup, "celebrity")):
        rows = _domain_rows(df, lookup, domain)
        if rows is not None:
            parts.append(rows)

    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out["retweet_ts"] = pd.to_numeric(out["time_stamp"], errors="coerce")
    out["source_ts"] = pd.to_numeric(out["r_time_stamp"], errors="coerce")
    out["delay_seconds"] = out["retweet_ts"] - out["source_ts"]
    # 非正延迟（含缺失换算失败）是真实记录，必须保留，只标记 delay_valid=False，
    # 是否剔除留给分析阶段决定
    out["delay_valid"] = out["delay_seconds"].notna() & (out["delay_seconds"] > 0)
    out["month"] = pd.to_datetime(out["date"]).dt.month

    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def _month_files(year, month):
    pattern = os.path.join(config.DATA_DIR, str(year), f"{year}-{month:02d}-*.parquet")
    return sorted(glob.glob(pattern))


def build_month(year=config.YEAR, month=1):
    """处理一个月的日文件，写出一个分片"""
    files = _month_files(year, month)
    if not files:
        print(f"未找到 {year} 年 {month} 月的数据文件，跳过")
        return None

    public_accounts = config.load_source_accounts("public")
    celebrity_accounts = config.load_source_accounts("celebrity")
    public_lookup = build_account_lookup(public_accounts)
    celebrity_lookup = build_account_lookup(celebrity_accounts)
    known_sources = set(public_lookup) | set(celebrity_lookup)
    overlap = set(public_lookup) & set(celebrity_lookup)
    print(
        f"来源账号: 公共事务 {len(public_lookup)}，明星 {len(celebrity_lookup)}，"
        f"两侧重叠 {len(overlap)}"
    )

    out_dir = os.path.join(config.OUTPUT_DIR, f"retweet_domain_events_{year}")
    os.makedirs(out_dir, exist_ok=True)

    parts = []
    # 逐步样本流记账，manifest 里据此可以从 rows_in 逐级核对到 rows_written，
    # 不止是一个笼统的前后对比（约定见表 A 的记账方式）：
    #   rows_in
    #   - rows_dropped_gender_null      -> rows_after_gender_filter
    #   - rows_dropped_not_retweet      -> retweet_rows
    #   - rows_dropped_source_self_retweet -> eligible_retweet_rows
    #   - rows_dropped_unmatched_source -> matched_retweet_rows（去重后的原始转发行数）
    #   + rows_emitted_twice_overlap    -> events_written（一行命中两个领域各出一行）
    rows_in = 0
    rows_dropped_gender_null = 0
    rows_dropped_not_retweet = 0
    rows_dropped_source_self_retweet = 0
    rows_dropped_unmatched_source = 0
    eligible_retweet_rows = 0
    matched_retweet_rows = 0
    events_written = 0

    for path in tqdm(files, desc=f"{year}-{month:02d} 转发事件"):
        df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
        rows_in += len(df)

        df, dropped_gender = _drop_gender_null(df)
        rows_dropped_gender_null += dropped_gender
        if len(df) == 0:
            continue
        df["date"] = os.path.basename(path).replace(".parquet", "")

        retweet_only, dropped_not_retweet = _filter_retweets(df)
        rows_dropped_not_retweet += dropped_not_retweet

        eligible, dropped_self = _exclude_source_self_retweets(retweet_only, known_sources)
        rows_dropped_source_self_retweet += dropped_self
        eligible_retweet_rows += len(eligible)

        processed = process_frame(df, public_lookup, celebrity_lookup)
        day_events = len(processed)
        day_matched_rows = processed["weibo_id"].nunique() if day_events else 0
        rows_dropped_unmatched_source += len(eligible) - day_matched_rows
        matched_retweet_rows += day_matched_rows
        events_written += day_events

        parts.append(processed)

    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OUTPUT_COLUMNS)

    out_path = os.path.join(out_dir, f"month={month:02d}.parquet")
    result.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"已保存: {out_path}（{len(result):,} 行）")

    duplicated = int(result.duplicated(subset=["user_id", "r_weibo_id", "source_domain"]).sum())
    manifest = config.build_manifest(
        step=f"retweet_domain_events_{year}_{month:02d}",
        inputs=[os.path.basename(p) for p in files],
        params={"year": year, "month": month},
        counts={
            "rows_in": rows_in,
            "rows_dropped_gender_null": rows_dropped_gender_null,
            "rows_dropped_not_retweet": rows_dropped_not_retweet,
            "rows_dropped_source_self_retweet": rows_dropped_source_self_retweet,
            "eligible_retweet_rows": eligible_retweet_rows,
            "rows_dropped_unmatched_source": rows_dropped_unmatched_source,
            "matched_retweet_rows": matched_retweet_rows,
            "rows_emitted_twice_overlap": events_written - matched_retweet_rows,
            "rows_written": int(len(result)),
            "by_domain": result["source_domain"].value_counts().to_dict() if len(result) else {},
            "non_positive_delay": int((~result["delay_valid"]).sum()) if len(result) else 0,
            "duplicate_user_source_pairs": duplicated,
            "overlapping_accounts": len(overlap),
        },
        fingerprints={
            "public_accounts": config.fingerprint_terms(sorted(public_lookup.keys())),
            "celebrity_accounts": config.fingerprint_terms(sorted(celebrity_lookup.keys())),
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
