"""
导出可视化所需的数据

在服务器上运行此脚本，生成parquet文件，下载到本地后用于可视化。

导出内容：
Section 1 (行为 - behavior):
  - retweet_overview_{year}.parquet: 转发行为概况（用户数、转发用户数、新闻/娱乐转发用户数）
  - retweet_counts_{source_type}_{year}.parquet: 所有转发用户的新闻/娱乐转发次数（含0值）
    列: user_id, gender, retweet_count
  - retweet_intervals_{source_type}_{year}.parquet: 转发间隔原始数据
    列: gender, retweet_interval

Section 2 (内容 - content):
  - density_summary_{year}.parquet: 新闻/娱乐密度汇总表
  - density_post_{type}_{year}.parquet: Post级别密度原始数据（density > 0）
    列: weibo_id, gender, density
  - density_user_{type}_{year}.parquet: User级别密度原始数据
    列: user_id, gender, avg_density

用法:
  python export_viz_data.py all 2020
  python export_viz_data.py behavior 2020
  python export_viz_data.py density 2020
"""

import os
import pandas as pd
import numpy as np
import glob
import fire
from tqdm import tqdm

DATA_DIR = "cleaned_weibo_cov"
ANALYSIS_DIR = "analysis_results"
VIZ_DIR = "viz_data"

os.makedirs(VIZ_DIR, exist_ok=True)


def _save_parquet(df, path):
    """保存parquet文件并打印信息"""
    df.to_parquet(path, engine="fastparquet", index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  已保存: {path} ({len(df):,} 行, {size_mb:.2f} MB)")


# ============================================================================
# Section 1: 转发行为数据导出
# ============================================================================


def export_behavior_data(year):
    """导出转发行为相关数据（原始数据）"""
    print(f"\n{'='*60}")
    print(f"导出 {year} 年转发行为数据")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: 扫描原始数据，统计所有用户和转发用户
    # ------------------------------------------------------------------
    print("\n扫描原始数据，统计用户...")
    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"未找到 {year} 年的数据目录: {year_dir}")
        return

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        print(f"未找到 {year} 年的parquet文件")
        return

    all_users = {}  # user_id(str) -> gender
    retweeters = set()  # user_ids with any retweet

    for f in tqdm(parquet_files, desc="扫描原始数据"):
        try:
            df = pd.read_parquet(f, columns=["user_id", "gender", "is_retweet"])
            df = df[df["gender"].notna()]
            df["user_id"] = df["user_id"].astype(str)

            for uid, gender in zip(df["user_id"], df["gender"]):
                if uid not in all_users:
                    all_users[uid] = gender

            retweet_uids = df[df["is_retweet"] == "1"]["user_id"]
            retweeters.update(retweet_uids)
        except Exception as e:
            print(f"  读取 {f} 失败: {e}")
            continue

    print(f"  总用户数: {len(all_users):,}")
    print(f"  有转发行为的用户数: {len(retweeters):,}")

    # ------------------------------------------------------------------
    # Step 2: 加载已有的新闻/娱乐转发数据
    # ------------------------------------------------------------------
    retweet_counts = {}  # source_type -> {user_id -> retweet_count}
    for source_type in ["news", "entertain"]:
        retweet_file = os.path.join(
            ANALYSIS_DIR, f"retweet_{source_type}_{year}.parquet"
        )
        if os.path.exists(retweet_file):
            rdf = pd.read_parquet(retweet_file, engine="fastparquet")
            rdf["user_id"] = rdf["user_id"].astype(str)
            retweet_counts[source_type] = dict(
                zip(rdf["user_id"], rdf["retweet_count"])
            )
            print(f"  已加载 {source_type} 转发数据: {len(retweet_counts[source_type]):,} 用户")
        else:
            print(f"  未找到 {source_type} 转发数据文件: {retweet_file}")
            retweet_counts[source_type] = {}

    # ------------------------------------------------------------------
    # Step 3: 导出概况表
    # ------------------------------------------------------------------
    overview_rows = []
    genders = sorted(set(all_users.values()))
    for gender in genders:
        gender_users = {uid for uid, g in all_users.items() if g == gender}
        gender_retweeters = gender_users & retweeters
        gender_news = gender_users & set(retweet_counts.get("news", {}).keys())
        gender_entertain = gender_users & set(retweet_counts.get("entertain", {}).keys())

        overview_rows.append(
            {
                "gender": gender,
                "total_users": len(gender_users),
                "users_with_retweet": len(gender_retweeters),
                "users_with_news_retweet": len(gender_news),
                "users_with_entertain_retweet": len(gender_entertain),
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    _save_parquet(overview_df, os.path.join(VIZ_DIR, f"retweet_overview_{year}.parquet"))
    print(overview_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 4: 导出转发次数原始数据（所有转发用户，含0值）
    # 列: user_id, gender, retweet_count
    # ------------------------------------------------------------------
    for source_type in ["news", "entertain"]:
        source_dict = retweet_counts.get(source_type, {})

        rows = []
        for uid in retweeters:
            gender = all_users.get(uid)
            if gender is None:
                continue
            count = source_dict.get(uid, 0)
            rows.append({"user_id": uid, "gender": gender, "retweet_count": count})

        if not rows:
            continue

        df = pd.DataFrame(rows)
        _save_parquet(
            df,
            os.path.join(VIZ_DIR, f"retweet_counts_{source_type}_{year}.parquet"),
        )

        # 打印摘要
        for gender in df["gender"].unique():
            g = df[df["gender"] == gender]
            zero = (g["retweet_count"] == 0).sum()
            print(
                f"  {source_type} {gender}: {len(g):,} 转发用户, "
                f"{zero:,} ({zero/len(g)*100:.1f}%) 未转发{source_type}"
            )

    # ------------------------------------------------------------------
    # Step 5: 导出转发间隔原始数据
    # 列: gender, retweet_interval
    # ------------------------------------------------------------------
    for source_type in ["news", "entertain"]:
        interval_file = os.path.join(
            ANALYSIS_DIR, f"retweet_{source_type}_intervals_{year}.parquet"
        )
        if not os.path.exists(interval_file):
            print(f"\n未找到 {source_type} 间隔数据文件，跳过")
            continue

        interval_df = pd.read_parquet(interval_file, engine="fastparquet")
        interval_df = interval_df[
            (interval_df["retweet_interval"] > 0)
            & (interval_df["retweet_interval"].notna())
        ].copy()

        if len(interval_df) == 0:
            continue

        # 只保留 gender + retweet_interval 两列
        interval_df = interval_df[["gender", "retweet_interval"]]
        _save_parquet(
            interval_df,
            os.path.join(VIZ_DIR, f"retweet_intervals_{source_type}_{year}.parquet"),
        )


# ============================================================================
# Section 2: 密度数据导出
# ============================================================================


def export_density_data(year):
    """导出新闻/娱乐密度相关数据（原始数据）"""
    print(f"\n{'='*60}")
    print(f"导出 {year} 年密度数据")
    print(f"{'='*60}")

    summary_rows = []

    for density_type in ["news", "entertain"]:
        prefix = "" if density_type == "news" else "entertain_"
        post_file = os.path.join(
            ANALYSIS_DIR, f"{prefix}post_density_{year}.parquet"
        )

        if not os.path.exists(post_file):
            print(f"\n未找到 {density_type} post density 文件: {post_file}")
            continue

        print(f"\n处理 {density_type} density...")
        post_df = pd.read_parquet(post_file, engine="fastparquet")
        print(f"  加载 {len(post_df):,} 条post数据")

        # === Post级别统计 ===
        for gender in sorted(post_df["gender"].unique()):
            g_data = post_df[post_df["gender"] == gender]
            densities = g_data["density"].values
            non_zero = densities[densities > 0]

            summary_rows.append(
                {
                    "type": density_type,
                    "level": "post",
                    "gender": gender,
                    "total_count": len(densities),
                    "zero_count": int(np.sum(densities == 0)),
                    "zero_ratio": float(np.mean(densities == 0)),
                    "non_zero_count": len(non_zero),
                    "non_zero_mean": float(np.mean(non_zero)) if len(non_zero) > 0 else 0,
                    "non_zero_median": float(np.median(non_zero)) if len(non_zero) > 0 else 0,
                    "non_zero_std": float(np.std(non_zero)) if len(non_zero) > 0 else 0,
                }
            )

        # === Post级别原始数据（density > 0）===
        post_non_zero = post_df[post_df["density"] > 0][
            ["weibo_id", "gender", "density"]
        ].copy()
        _save_parquet(
            post_non_zero,
            os.path.join(VIZ_DIR, f"density_post_{density_type}_{year}.parquet"),
        )

        # === User级别密度 ===
        user_density = (
            post_df.groupby(["user_id", "gender"])["density"]
            .mean()
            .reset_index()
        )
        user_density.columns = ["user_id", "gender", "avg_density"]

        # User级别统计
        for gender in sorted(user_density["gender"].unique()):
            g_data = user_density[user_density["gender"] == gender]
            densities = g_data["avg_density"].values
            non_zero = densities[densities > 0]

            summary_rows.append(
                {
                    "type": density_type,
                    "level": "user",
                    "gender": gender,
                    "total_count": len(densities),
                    "zero_count": int(np.sum(densities == 0)),
                    "zero_ratio": float(np.mean(densities == 0)),
                    "non_zero_count": len(non_zero),
                    "non_zero_mean": float(np.mean(non_zero)) if len(non_zero) > 0 else 0,
                    "non_zero_median": float(np.median(non_zero)) if len(non_zero) > 0 else 0,
                    "non_zero_std": float(np.std(non_zero)) if len(non_zero) > 0 else 0,
                }
            )

        # 导出user级别原始数据（全部用户，含0）
        _save_parquet(
            user_density,
            os.path.join(VIZ_DIR, f"density_user_{density_type}_{year}.parquet"),
        )

    # 保存汇总表
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        _save_parquet(
            summary_df,
            os.path.join(VIZ_DIR, f"density_summary_{year}.parquet"),
        )
        print(f"\n密度汇总表:")
        print(summary_df.to_string(index=False))


# ============================================================================
# 主入口
# ============================================================================


def export_all(year):
    """导出所有可视化数据"""
    export_behavior_data(year)
    export_density_data(year)

    print(f"\n{'='*60}")
    print(f"{year} 年所有可视化数据导出完成！")
    print(f"输出目录: {VIZ_DIR}/")
    print(f"{'='*60}")

    files = sorted(glob.glob(os.path.join(VIZ_DIR, f"*_{year}.parquet")))
    print(f"\n生成的文件列表:")
    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    fire.Fire(
        {
            "behavior": export_behavior_data,
            "density": export_density_data,
            "all": export_all,
        }
    )
