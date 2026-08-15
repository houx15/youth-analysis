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
from gender_domain import text_rules as tr

REQUIRED_COLUMNS = [
    "weibo_id",
    "user_id",
    "weibo_content",
    "is_retweet",
    "gender",
    "province",
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


def process_frame(df, public_matcher, celebrity_matcher):
    """对一天（或任意一批）帖子计算领域测量，纯函数便于测试"""
    df = df.drop_duplicates(subset=["weibo_id"], keep="first").copy()

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
    for path in tqdm(files, desc=f"{year}-{month:02d} 帖子测量"):
        df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
        rows_in += len(df)
        df = df[df["gender"].notna()]
        if len(df) == 0:
            continue
        df["date"] = os.path.basename(path).replace(".parquet", "")
        parts.append(process_frame(df, public_matcher, celebrity_matcher))

    if not parts:
        print(f"{year} 年 {month} 月没有可用数据")
        return None

    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates(subset=["weibo_id"], keep="first")

    out_path = os.path.join(out_dir, f"month={month:02d}.parquet")
    result.to_parquet(out_path, engine="pyarrow", index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"已保存: {out_path}（{len(result):,} 行, {size_mb:.1f} MB）")

    manifest = config.build_manifest(
        step=f"post_domain_measures_{year}_{month:02d}",
        inputs=[os.path.basename(p) for p in files],
        params={"year": year, "month": month},
        counts={
            "rows_in": rows_in,
            "rows_with_gender": int(len(result)),
            "post_type": result["post_type"].value_counts().to_dict(),
            "public_hit_rate": float(result["public_hit"].mean()),
            "celebrity_hit_rate": float(result["celebrity_hit"].mean()),
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
