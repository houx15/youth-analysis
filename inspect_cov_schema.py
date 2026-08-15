"""
cleaned_weibo_cov 数据字典生成脚本

用于正式分析开工前核对服务器上数据的真实结构，产出可下载的小体积报告，
不做任何重型计算。对应研究设计文档 0.5 节第 2 步。

功能：
1. schema  - 抽查若干天文件的列名与类型，检查跨文件 schema 是否一致
2. profile - 逐列缺失率、取值示例、关键字段取值分布、时间戳单位、帖子类型构成
3. users   - 全年唯一用户数与性别构成（只读 2 列，用于核对 225,339 的样本量）
4. sample  - 导出一天的小样本 parquet，供本地下载查看

使用方法:
    python inspect_cov_schema.py schema 2020 --n_files=5
    python inspect_cov_schema.py profile 2020 --n_files=3
    python inspect_cov_schema.py users 2020
    python inspect_cov_schema.py sample 2020 --n_rows=2000
"""

import os
import glob
import json

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "analysis_data/schema_report"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 关键字段：正式流水线依赖这些列，需要逐个确认类型与缺失
KEY_COLUMNS = [
    "user_id",
    "weibo_id",
    "weibo_content",
    "is_retweet",
    "r_user_id",
    "r_weibo_id",
    "time_stamp",
    "r_time_stamp",
    "gender",
    "province",
    "region",
]

# 取值分布需要枚举的低基数字段
CATEGORICAL_COLUMNS = ["is_retweet", "gender", "region", "user_type_x", "user_type_y"]


def _list_files(year):
    """列出指定年份的所有日文件"""
    pattern = os.path.join(DATA_DIR, str(year), "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 {year} 年的数据文件: {pattern}")
    return files


def _pick_files(files, n_files):
    """在全年文件中均匀抽取 n_files 个，避免只看年初数据"""
    if n_files >= len(files):
        return files
    idx = np.linspace(0, len(files) - 1, n_files).astype(int)
    return [files[i] for i in sorted(set(idx))]


def _write_json(obj, name):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    size_kb = os.path.getsize(path) / 1024
    print(f"  已保存: {path} ({size_kb:.1f} KB)")
    return path


def schema(year=2020, n_files=5):
    """抽查若干天文件的 schema，检查列名与类型是否跨文件一致"""
    files = _list_files(year)
    picked = _pick_files(files, n_files)

    print(f"{year} 年共 {len(files)} 个日文件，抽查 {len(picked)} 个")

    per_file = {}
    for path in picked:
        meta = pq.ParquetFile(path)
        arrow_schema = meta.schema_arrow
        per_file[os.path.basename(path)] = {
            "num_rows": meta.metadata.num_rows,
            "num_row_groups": meta.metadata.num_row_groups,
            "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "columns": {
                name: str(arrow_schema.field(name).type) for name in arrow_schema.names
            },
        }

    # 跨文件比较列集合
    col_sets = {fn: set(info["columns"].keys()) for fn, info in per_file.items()}
    all_cols = set().union(*col_sets.values())
    common_cols = set.intersection(*col_sets.values())
    inconsistent = sorted(all_cols - common_cols)

    # 跨文件比较类型
    type_conflicts = {}
    for col in sorted(common_cols):
        types = {info["columns"][col] for info in per_file.values()}
        if len(types) > 1:
            type_conflicts[col] = sorted(types)

    print(f"\n列总数: {len(all_cols)}，所有抽查文件共有: {len(common_cols)}")
    if inconsistent:
        print(f"⚠ 并非所有文件都有的列: {inconsistent}")
    else:
        print("✓ 抽查文件的列集合完全一致")
    if type_conflicts:
        print(f"⚠ 跨文件类型不一致的列: {type_conflicts}")
    else:
        print("✓ 抽查文件的列类型完全一致")

    missing_key = [c for c in KEY_COLUMNS if c not in common_cols]
    if missing_key:
        print(f"⚠ 缺少正式流水线依赖的关键列: {missing_key}")
    else:
        print(f"✓ {len(KEY_COLUMNS)} 个关键列齐备")

    total_rows = sum(info["num_rows"] for info in per_file.values())
    print(f"\n抽查文件平均行数: {total_rows / len(per_file):,.0f}")
    print(f"按此估算全年约: {total_rows / len(per_file) * len(files):,.0f} 行")

    report = {
        "year": year,
        "n_files_total": len(files),
        "n_files_checked": len(picked),
        "columns_all": sorted(all_cols),
        "columns_common": sorted(common_cols),
        "columns_inconsistent": inconsistent,
        "type_conflicts": type_conflicts,
        "key_columns_missing": missing_key,
        "estimated_total_rows": int(total_rows / len(per_file) * len(files)),
        "per_file": per_file,
    }
    _write_json(report, f"schema_{year}.json")
    return report


def profile(year=2020, n_files=3, n_examples=3):
    """逐列缺失率、取值示例、关键字段分布、时间戳单位与帖子类型构成"""
    files = _list_files(year)
    picked = _pick_files(files, n_files)
    print(f"抽查 {len(picked)} 个日文件做取值核查")

    frames = []
    for path in tqdm(picked, desc="读取抽查文件"):
        frames.append(pd.read_parquet(path))
    data = pd.concat(frames, ignore_index=True)
    del frames
    print(f"合计 {len(data):,} 行，{len(data.columns)} 列")

    # ---- 逐列缺失与示例 ----
    col_report = {}
    for col in data.columns:
        series = data[col]
        non_null = series.dropna()
        # 空字符串在本数据中等同缺失，单独统计
        empty_str = int((non_null.astype(str).str.strip() == "").sum())
        examples = [str(v)[:60] for v in non_null.head(n_examples).tolist()]
        col_report[col] = {
            "dtype": str(series.dtype),
            "null_ratio": round(float(series.isna().mean()), 4),
            "empty_string_count": empty_str,
            "n_unique": int(series.nunique(dropna=True)),
            "examples": examples,
        }

    # ---- 低基数字段取值分布 ----
    value_counts = {}
    for col in CATEGORICAL_COLUMNS:
        if col in data.columns:
            vc = data[col].value_counts(dropna=False).head(20)
            value_counts[col] = {str(k): int(v) for k, v in vc.items()}

    # ---- 唯一键检查 ----
    key_check = {}
    if "weibo_id" in data.columns:
        key_check["weibo_id_rows"] = int(len(data))
        key_check["weibo_id_unique"] = int(data["weibo_id"].nunique())
        key_check["weibo_id_is_unique"] = bool(
            data["weibo_id"].nunique() == len(data.dropna(subset=["weibo_id"]))
        )

    # ---- 时间戳单位与范围 ----
    time_check = {}
    for col in ["time_stamp", "r_time_stamp"]:
        if col not in data.columns:
            continue
        ts = pd.to_numeric(data[col], errors="coerce").dropna()
        if len(ts) == 0:
            time_check[col] = {"note": "无法转换为数值"}
            continue
        median = float(ts.median())
        unit = "seconds" if median < 1e11 else "milliseconds"
        divisor = 1 if unit == "seconds" else 1000
        time_check[col] = {
            "coerce_fail_ratio": round(
                float(pd.to_numeric(data[col], errors="coerce").isna().mean()), 4
            ),
            "inferred_unit": unit,
            "min_utc": str(pd.to_datetime(ts.min() / divisor, unit="s")),
            "median_utc": str(pd.to_datetime(median / divisor, unit="s")),
            "max_utc": str(pd.to_datetime(ts.max() / divisor, unit="s")),
        }

    # 转发延迟为负的比例（研究设计第十节要求）
    if {"time_stamp", "r_time_stamp", "is_retweet"} <= set(data.columns):
        rt = data[data["is_retweet"].astype(str) == "1"]
        delay = pd.to_numeric(rt["time_stamp"], errors="coerce") - pd.to_numeric(
            rt["r_time_stamp"], errors="coerce"
        )
        delay = delay.dropna()
        if len(delay) > 0:
            time_check["retweet_delay_raw"] = {
                "n": int(len(delay)),
                "non_positive_ratio": round(float((delay <= 0).mean()), 4),
                "median": float(delay.median()),
                "p95": float(delay.quantile(0.95)),
                "max": float(delay.max()),
            }

    # ---- 帖子类型构成（原创 / 转发加评论 / 纯转发）----
    post_type = {}
    if {"is_retweet", "weibo_content"} <= set(data.columns):
        content = data["weibo_content"].fillna("").astype(str).str.strip()
        is_rt = data["is_retweet"].astype(str) == "1"
        # 纯转发的常见形态：空文本或"转发微博"类占位
        placeholder = content.isin(["", "转发微博", "转发微博。", "轉發微博", "Repost"])
        post_type = {
            "original": int((~is_rt).sum()),
            "retweet_with_comment": int((is_rt & ~placeholder).sum()),
            "retweet_plain": int((is_rt & placeholder).sum()),
            "placeholder_examples": content[is_rt & placeholder].head(5).tolist(),
            "comment_examples": content[is_rt & ~placeholder].head(5).tolist(),
        }

    print("\n关键字段取值分布:")
    for col, vc in value_counts.items():
        print(f"  {col}: {vc}")
    print(f"\n唯一键检查: {key_check}")
    post_type_counts = {k: v for k, v in post_type.items() if isinstance(v, int)}
    print(f"帖子类型构成: {post_type_counts}")

    report = {
        "year": year,
        "files_checked": [os.path.basename(p) for p in picked],
        "rows_checked": int(len(data)),
        "columns": col_report,
        "value_counts": value_counts,
        "key_check": key_check,
        "time_check": time_check,
        "post_type": post_type,
    }
    _write_json(report, f"profile_{year}.json")
    return report


def users(year=2020):
    """全年唯一用户数与性别构成，用于核对 225,339 名匹配用户"""
    files = _list_files(year)
    print(f"扫描 {len(files)} 个日文件（只读 user_id 与 gender 两列）")

    gender_by_user = {}
    rows_total = 0
    for path in tqdm(files, desc="统计用户"):
        df = pd.read_parquet(path, columns=["user_id", "gender"])
        rows_total += len(df)
        df = df.dropna(subset=["user_id"])
        for uid, gender in zip(df["user_id"].astype(str), df["gender"]):
            if uid not in gender_by_user:
                gender_by_user[uid] = gender
            elif gender_by_user[uid] != gender and pd.notna(gender):
                # 同一用户跨天性别不一致，标记出来（研究设计 13.9 需要）
                gender_by_user[uid] = "CONFLICT"

    series = pd.Series(gender_by_user)
    counts = series.value_counts(dropna=False)
    with_gender = int(series.notna().sum() - int(counts.get("CONFLICT", 0)))

    print(f"\n总行数: {rows_total:,}")
    print(f"唯一用户数: {len(series):,}")
    print(f"有可用性别且跨天一致: {with_gender:,}")
    print("性别构成:")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")

    report = {
        "year": year,
        "n_files": len(files),
        "rows_total": rows_total,
        "users_total": int(len(series)),
        "users_with_consistent_gender": with_gender,
        "gender_counts": {str(k): int(v) for k, v in counts.items()},
        "design_doc_expectation": {"total": 225339, "female": 173503, "male": 51836},
    }
    _write_json(report, f"users_{year}.json")
    return report


def sample(year=2020, date=None, n_rows=2000):
    """导出一天的小样本，供本地下载查看真实取值"""
    files = _list_files(year)
    if date:
        matched = [p for p in files if os.path.basename(p).startswith(str(date))]
        if not matched:
            raise FileNotFoundError(f"未找到 {date} 的文件")
        path = matched[0]
    else:
        path = files[len(files) // 2]

    df = pd.read_parquet(path)
    print(f"源文件: {path}（{len(df):,} 行）")
    out = df.head(n_rows)

    out_path = os.path.join(OUTPUT_DIR, f"sample_{os.path.basename(path)}")
    out.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"已保存样本: {out_path}（{len(out):,} 行, {size_mb:.2f} MB）")
    print("下载后本地查看: pd.read_parquet(...)")
    return out_path


def all(year=2020, n_files=3):
    """schema + profile + sample（不含 users 全年扫描）"""
    schema(year, n_files=max(n_files, 5))
    profile(year, n_files=n_files)
    sample(year)
    print(f"\n报告目录: {OUTPUT_DIR}/")


if __name__ == "__main__":
    fire.Fire(
        {
            "schema": schema,
            "profile": profile,
            "users": users,
            "sample": sample,
            "all": all,
        }
    )
