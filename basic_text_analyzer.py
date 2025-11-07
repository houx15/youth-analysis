"""
微博数据分析脚本

基于清理后的parquet数据进行分析，包括：
1. 用户设备更换频率分析
2. 转发官方媒体情况分析

>>> data
           user_id is_retweet   nick_name  ... demographic_gender demographic_province     region
0       6091236422          0    咕噜咕噜汽水砰砰  ...                  女                  内蒙古       West
1       2239189490          1  雨文_JoannaC  ...                  女                   广东       East
2       3623597887          0        晋城消防  ...                  男                   山西    Central
3       3623597887          0        晋城消防  ...                  男                   山西    Central
4       6511763400          0  _小猴子吃橙子yu_  ...                  女                   香港       None
...            ...        ...         ...  ...                ...                  ...        ...
116610  6013624378          0  Bxxdxicx__  ...                  女                   吉林  Northeast
116611  6008173602          0        簪默棠初  ...                  女                   重庆       West
116612  2267005925          0        土月土山  ...                  女                   浙江       East
116613  6495672754          0         阿鹬_  ...                  女                   上海       East
116614  6132944661          0    speed二号机  ...                  男                   北京       East

[116615 rows x 38 columns]
>>> data.columns
Index(['user_id', 'is_retweet', 'nick_name', 'user_type_x', 'weibo_id',
       'weibo_content', 'zhuan', 'ping', 'zan', 'device', 'locate',
       'time_stamp', 'r_user_id', 'r_nick_name', 'r_user_type', 'r_weibo_id',
       'r_weibo_content', 'r_zhuan', 'r_ping', 'r_zan', 'r_device',
       'r_location', 'r_time', 'r_time_stamp', 'src', 'tag', 'lat', 'lon',
       'user_type_y', 'gender', 'location', 'province', 'city', 'ip_location',
       'birthday', 'demographic_gender', 'demographic_province', 'region'],
      dtype='object')

"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import fire
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import seaborn as sns

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "analysis_results"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 官方媒体user_id列表（从配置文件加载）
OFFICIAL_MEDIA_IDS = set()


def load_official_media_ids():
    """从配置文件加载官方媒体ID"""
    global OFFICIAL_MEDIA_IDS

    try:
        from get_news_ids import load_news_user_ids

        OFFICIAL_MEDIA_IDS = load_news_user_ids()
        print(f"已加载 {len(OFFICIAL_MEDIA_IDS)} 个官方媒体账号ID")
    except ImportError:
        print("警告: 无法导入 get_news_ids 模块，请确保已生成新闻账号ID")
    except Exception as e:
        print(f"加载官方媒体ID时出错: {e}")


def get_date_range(year):
    """获取指定年份的日期范围"""
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return date_list


def load_data_for_year(year, analysis_type="all"):
    """加载指定年份的数据，只加载必要的列以节省内存"""
    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"未找到 {year} 年的数据目录: {year_dir}")
        return None

    # 查找该年份目录下的所有parquet文件
    import glob

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"未找到 {year} 年的parquet文件")
        return None

    print(f"找到 {len(parquet_files)} 个文件")

    # 根据分析类型确定需要的列
    if analysis_type == "device":
        required_columns = ["user_id", "device", "nick_name"]
    elif analysis_type == "retweet":
        required_columns = [
            "user_id",
            "is_retweet",
            "r_user_id",
            "gender",
            "time_stamp",
            "r_time_stamp",
        ]
    else:  # all
        required_columns = [
            "user_id",
            "device",
            "nick_name",
            "is_retweet",
            "r_user_id",
            "gender",
        ]

    all_data = []
    for file_path in parquet_files:
        try:
            # 只读取需要的列
            df = pd.read_parquet(file_path, columns=required_columns)
            # 从文件名提取日期
            filename = os.path.basename(file_path)
            date_str = filename.replace(".parquet", "")
            df["date"] = date_str  # 添加日期列
            all_data.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            continue

    if not all_data:
        print(f"未能加载 {year} 年的任何数据")
        return None

    result = pd.concat(all_data, ignore_index=True)
    print(f"共加载 {len(result)} 条数据，列数: {len(result.columns)}")
    return result


def analyze_device_changes(year):
    """分析用户设备更换频率，只分析occurrence>=10的用户"""
    print(f"\n开始分析 {year} 年用户设备更换情况...")

    # 加载数据，只加载设备分析需要的列
    data = load_data_for_year(year, "device")
    if data is None:
        return

    print(f"共加载 {len(data)} 条数据")

    # 统计每个用户的出现次数（不同日期的数量）
    user_occurrence = data.groupby("user_id")["date"].nunique().reset_index()
    user_occurrence.columns = ["user_id", "occurrence_count"]

    # 只保留occurrence>=10的用户
    active_users = user_occurrence[user_occurrence["occurrence_count"] >= 10]["user_id"]
    print(f"出现次数>=10的用户数: {len(active_users)}")

    # 过滤数据，只保留活跃用户
    data_filtered = data[data["user_id"].isin(active_users)]
    print(f"过滤后数据量: {len(data_filtered)} 条")

    # 创建用户每日设备表
    # 只取每天每个用户的最后一条微博的设备信息
    user_device_daily = (
        data_filtered.groupby(["date", "user_id"])
        .agg({"device": "last", "nick_name": "last"})  # 取最后一条
        .reset_index()
    )

    # 透视表：用户 x 日期（稀疏表格）
    device_pivot = user_device_daily.pivot_table(
        index="user_id",
        columns="date",
        values="device",
        aggfunc="last",
        fill_value="",  # 空值填充为空字符串
    )

    # 保存稀疏表格
    # 重置索引以便保存user_id
    device_pivot_reset = device_pivot.reset_index()
    output_file = os.path.join(OUTPUT_DIR, f"device_daily_{year}.parquet")
    device_pivot_reset.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"用户每日设备表已保存到: {output_file}")

    # 获取用户昵称映射
    user_names = user_device_daily.groupby("user_id")["nick_name"].first()

    # 分析设备切换模式
    device_changes = []

    for idx, row in device_pivot.iterrows():
        user_id = idx
        # 移除空值，只保留有设备信息的日期
        non_null_devices = row[row != ""].dropna()

        if len(non_null_devices) < 2:
            continue  # 至少需要2个数据点才能分析切换

        # 计算设备切换
        prev_device = None
        change_dates = []
        change_count = 0
        days_between_changes = []
        last_change_date = None

        for date, device in non_null_devices.items():
            if prev_device is not None and device != prev_device:
                change_count += 1
                change_dates.append(date)

                if last_change_date is not None:
                    days_diff = (
                        datetime.strptime(date, "%Y-%m-%d")
                        - datetime.strptime(last_change_date, "%Y-%m-%d")
                    ).days
                    days_between_changes.append(days_diff)

                last_change_date = date

            prev_device = device

        if change_count > 0:
            device_changes.append(
                {
                    "user_id": user_id,
                    "nick_name": user_names.get(user_id, ""),
                    "total_changes": change_count,
                    "change_dates": ",".join(change_dates),
                    "avg_days_between_changes": (
                        np.mean(days_between_changes) if days_between_changes else 0
                    ),
                    "min_days_between_changes": (
                        min(days_between_changes) if days_between_changes else 0
                    ),
                    "max_days_between_changes": (
                        max(days_between_changes) if days_between_changes else 0
                    ),
                }
            )

    # 保存设备切换分析结果
    changes_df = pd.DataFrame(device_changes)
    changes_output = os.path.join(OUTPUT_DIR, f"device_changes_{year}.parquet")
    changes_df.to_parquet(changes_output, engine="fastparquet", index=False)
    print(f"设备切换分析结果已保存到: {changes_output}")

    # 打印统计信息
    if len(changes_df) > 0:
        print(f"\n设备切换统计:")
        print(f"  有设备切换的用户数: {len(changes_df)}")
        print(f"  平均切换次数: {changes_df['total_changes'].mean():.2f}")
        print(f"  平均切换间隔: {changes_df['avg_days_between_changes'].mean():.2f} 天")
        print(f"  切换次数最多: {changes_df['total_changes'].max()} 次")
        print(f"  最短切换间隔: {changes_df['min_days_between_changes'].min()} 天")

    print(f"\n设备分析完成\n")


def analyze_retweet_media(year):
    """分析转发官方媒体情况，特别关注性别差异"""
    print(f"\n开始分析 {year} 年转发官方媒体情况...")

    # 如果ID列表为空，尝试从配置文件加载
    if not OFFICIAL_MEDIA_IDS:
        load_official_media_ids()

    if not OFFICIAL_MEDIA_IDS:
        print("警告: 官方媒体ID列表为空，请先运行 get_news_ids.py 生成新闻账号ID文件")
        return

    # 加载数据，只加载转发分析需要的列
    data = load_data_for_year(year, "retweet")
    if data is None:
        return

    # 检查是否有性别字段
    has_gender = "gender" in data.columns
    if has_gender:
        print(f"数据包含性别字段，将进行性别分析")
    else:
        print(f"数据不包含性别字段，将进行基本分析")

    # 排除官方媒体用户ID（避免官方媒体转发自己）
    data = data[~data["user_id"].isin(OFFICIAL_MEDIA_IDS)]
    print(f"排除官方媒体用户后，剩余 {len(data)} 条记录")

    # 只保留转发官方媒体的记录
    retweet_media = data[
        (data["is_retweet"] == "1") & (data["r_user_id"].isin(OFFICIAL_MEDIA_IDS))
    ]

    if len(retweet_media) == 0:
        print(f"未找到转发官方媒体的记录")
        return

    print(f"共找到 {len(retweet_media)} 条转发官方媒体的记录")

    # 计算转发间隔（转发时间 - 原微博时间）
    # 处理时间戳字段，转换为数值类型（秒）
    has_interval_data = False
    if (
        "time_stamp" in retweet_media.columns
        and "r_time_stamp" in retweet_media.columns
    ):
        # 尝试将时间戳转换为数值类型
        try:
            retweet_media = retweet_media.copy()
            retweet_media["time_stamp_num"] = pd.to_numeric(
                retweet_media["time_stamp"], errors="coerce"
            )
            retweet_media["r_time_stamp_num"] = pd.to_numeric(
                retweet_media["r_time_stamp"], errors="coerce"
            )
            # 计算转发间隔（秒）
            retweet_media["retweet_interval"] = (
                retweet_media["r_time_stamp_num"] - retweet_media["time_stamp_num"]
            )
            # 标记有效的转发间隔（大于0且不为NaN）
            valid_intervals = retweet_media[
                (retweet_media["retweet_interval"] > 0)
                & (retweet_media["retweet_interval"].notna())
            ]
            print(f"其中 {len(valid_intervals)} 条记录有有效的转发间隔数据")
            has_interval_data = len(valid_intervals) > 0
        except Exception as e:
            print(f"计算转发间隔时出错: {e}，将跳过转发间隔分析")
            retweet_media["retweet_interval"] = None
    else:
        retweet_media["retweet_interval"] = None

    # 基本统计：每个用户的转发次数（使用所有转发记录）
    user_retweet_count = (
        retweet_media.groupby("user_id").size().reset_index(name="retweet_count")
    )

    # 如果有性别信息，添加性别字段
    if has_gender:
        # 获取用户性别信息（取每个用户最常见的性别）
        user_gender = (
            retweet_media.groupby("user_id")["gender"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "未知")
            .reset_index()
        )
        user_gender.columns = ["user_id", "gender"]

        # 合并性别信息
        user_retweet_count = user_retweet_count.merge(
            user_gender, on="user_id", how="left"
        )

        # 计算总体性别分布（用于计算占比）
        total_gender_dist = data.groupby("gender").size()
        print(f"\n总体性别分布:")
        for gender in total_gender_dist.index:
            print(f"  {gender}: {total_gender_dist[gender]} 人")

        # 统计每个性别中有转发行为的用户数（不管是否转发官方媒体）
        users_with_retweet = (
            data[data["is_retweet"] == "1"]
            .groupby("gender")["user_id"]
            .nunique()
            .to_dict()
        )

        # 按性别统计转发情况
        gender_stats = []
        for gender in user_retweet_count["gender"].unique():
            if gender in user_retweet_count["gender"].values:
                gender_data = user_retweet_count[user_retweet_count["gender"] == gender]
                retweet_users = len(gender_data)
                total_retweets = gender_data["retweet_count"].sum()
                avg_retweets = gender_data["retweet_count"].mean()
                median_retweets = gender_data["retweet_count"].median()

                # 计算占比和人均次数
                total_users_of_gender = total_gender_dist.get(gender, 0)
                retweet_ratio = (
                    retweet_users / total_users_of_gender
                    if total_users_of_gender > 0
                    else 0
                )
                per_capita_retweets = (
                    total_retweets / total_users_of_gender
                    if total_users_of_gender > 0
                    else 0
                )

                # 计算有转发行为的用户数和转发官方媒体账号记录者占比
                users_with_retweet_behavior = users_with_retweet.get(gender, 0)
                retweet_media_ratio = (
                    retweet_users / users_with_retweet_behavior
                    if users_with_retweet_behavior > 0
                    else 0
                )

                # 计算平均转发间隔（秒）
                avg_retweet_interval = None
                median_retweet_interval = None
                if "retweet_interval" in retweet_media.columns:
                    gender_retweet_data = retweet_media[
                        retweet_media["gender"] == gender
                    ]
                    valid_intervals = gender_retweet_data[
                        gender_retweet_data["retweet_interval"].notna()
                    ]["retweet_interval"]
                    if len(valid_intervals) > 0:
                        avg_retweet_interval = valid_intervals.mean()
                        median_retweet_interval = valid_intervals.median()

                gender_stats.append(
                    {
                        "gender": gender,
                        "retweet_users": retweet_users,
                        "total_users": total_users_of_gender,
                        "retweet_ratio": retweet_ratio,
                        "total_retweets": total_retweets,
                        "avg_retweets_per_user": avg_retweets,
                        "median_retweets_per_user": median_retweets,
                        "per_capita_retweets": per_capita_retweets,
                        "users_with_retweet_behavior": users_with_retweet_behavior,
                        "retweet_media_ratio": retweet_media_ratio,
                        "avg_retweet_interval_seconds": avg_retweet_interval,
                        "median_retweet_interval_seconds": median_retweet_interval,
                    }
                )

                print(f"\n{gender}性转发统计:")
                print(f"  转发用户数: {retweet_users} 人")
                print(f"  总用户数: {total_users_of_gender} 人")
                print(f"  转发占比: {retweet_ratio:.4f} ({retweet_ratio*100:.2f}%)")
                print(f"  有转发行为的用户数: {users_with_retweet_behavior} 人")
                print(
                    f"  转发官方媒体账号记录者占比: {retweet_media_ratio:.4f} ({retweet_media_ratio*100:.2f}%)"
                )
                print(f"  总转发次数: {total_retweets} 次")
                print(f"  转发用户平均转发: {avg_retweets:.2f} 次")
                print(f"  转发用户中位数: {median_retweets:.2f} 次")
                print(f"  人均转发次数: {per_capita_retweets:.4f} 次")
                if avg_retweet_interval is not None:
                    # 将秒转换为更易读的格式
                    hours = avg_retweet_interval / 3600
                    minutes = (avg_retweet_interval % 3600) / 60
                    seconds = avg_retweet_interval % 60
                    if hours >= 1:
                        print(
                            f"  平均转发间隔: {avg_retweet_interval:.0f} 秒 ({hours:.1f} 小时)"
                        )
                    elif minutes >= 1:
                        print(
                            f"  平均转发间隔: {avg_retweet_interval:.0f} 秒 ({minutes:.1f} 分钟)"
                        )
                    else:
                        print(f"  平均转发间隔: {avg_retweet_interval:.0f} 秒")
                    if median_retweet_interval is not None:
                        median_hours = median_retweet_interval / 3600
                        median_minutes = (median_retweet_interval % 3600) / 60
                        if median_hours >= 1:
                            print(
                                f"  中位转发间隔: {median_retweet_interval:.0f} 秒 ({median_hours:.1f} 小时)"
                            )
                        elif median_minutes >= 1:
                            print(
                                f"  中位转发间隔: {median_retweet_interval:.0f} 秒 ({median_minutes:.1f} 分钟)"
                            )
                        else:
                            print(f"  中位转发间隔: {median_retweet_interval:.0f} 秒")
                else:
                    print(f"  平均转发间隔: 无数据")

        # 保存性别统计摘要
        gender_stats_df = pd.DataFrame(gender_stats)
        gender_summary_file = os.path.join(
            OUTPUT_DIR, f"retweet_media_gender_{year}.parquet"
        )
        gender_stats_df.to_parquet(
            gender_summary_file, engine="fastparquet", index=False
        )
        print(f"性别统计摘要已保存到: {gender_summary_file}")

        # 绘制转发次数和转发间隔的分布对比图
        if has_gender and len(user_retweet_count) > 0:
            # 准备转发次数数据（每个用户的转发次数，按性别分组）
            retweet_count_data = user_retweet_count[["gender", "retweet_count"]].copy()
            retweet_count_data = retweet_count_data[
                retweet_count_data["gender"].notna()
            ]

            # 准备转发间隔数据（每条转发记录的间隔，按性别分组）
            interval_data = None
            if "retweet_interval" in retweet_media.columns:
                interval_df = retweet_media[["gender", "retweet_interval"]].copy()
                interval_df = interval_df[
                    (interval_df["gender"].notna())
                    & (interval_df["retweet_interval"].notna())
                    & (interval_df["retweet_interval"] > 0)
                ]
                if len(interval_df) > 0:
                    interval_data = interval_df

            # 绘制转发次数分布对比图
            if len(retweet_count_data) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # 左图：箱线图
                ax1 = axes[0]
                genders = retweet_count_data["gender"].unique()
                box_data = [
                    retweet_count_data[retweet_count_data["gender"] == g][
                        "retweet_count"
                    ].values
                    for g in genders
                ]
                bp = ax1.boxplot(box_data, labels=genders, patch_artist=True)
                # 设置箱线图颜色
                colors = ["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181"]
                for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax1.set_xlabel("性别", fontsize=12)
                ax1.set_ylabel("转发次数", fontsize=12)
                ax1.set_title(
                    f"{year}年转发官方媒体次数分布对比（箱线图）",
                    fontsize=13,
                    fontweight="bold",
                )
                ax1.grid(True, alpha=0.3)

                # 右图：KDE图
                ax2 = axes[1]
                # 检查所有数据的最大值，决定是否使用对数变换
                max_count = retweet_count_data["retweet_count"].max()
                use_log = max_count > 100

                for gender in genders:
                    gender_data = retweet_count_data[
                        retweet_count_data["gender"] == gender
                    ]["retweet_count"]
                    # 对数据进行对数变换以便更好地展示分布（如果数据范围很大）
                    if use_log:
                        log_data = np.log1p(gender_data)
                        sns.kdeplot(log_data, ax=ax2, label=f"{gender}性", linewidth=2)
                    else:
                        sns.kdeplot(
                            gender_data, ax=ax2, label=f"{gender}性", linewidth=2
                        )
                if use_log:
                    ax2.set_xlabel("转发次数（对数变换）", fontsize=12)
                else:
                    ax2.set_xlabel("转发次数", fontsize=12)
                ax2.set_ylabel("密度", fontsize=12)
                ax2.set_title(
                    f"{year}年转发官方媒体次数分布对比（KDE）",
                    fontsize=13,
                    fontweight="bold",
                )
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                fig_path = os.path.join(
                    OUTPUT_DIR, f"retweet_count_distribution_{year}.pdf"
                )
                plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
                plt.close()
                print(f"转发次数分布对比图已保存到: {fig_path}")

            # 绘制转发间隔分布对比图
            if interval_data is not None and len(interval_data) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # 左图：箱线图（使用对数变换，因为间隔可能差异很大）
                ax1 = axes[0]
                genders = interval_data["gender"].unique()
                # 转换为小时以便更好地展示
                interval_hours = interval_data.copy()
                interval_hours["interval_hours"] = (
                    interval_hours["retweet_interval"] / 3600
                )
                box_data = [
                    interval_hours[interval_hours["gender"] == g][
                        "interval_hours"
                    ].values
                    for g in genders
                ]
                bp = ax1.boxplot(box_data, labels=genders, patch_artist=True)
                # 设置箱线图颜色
                colors = ["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181"]
                for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax1.set_xlabel("性别", fontsize=12)
                ax1.set_ylabel("转发间隔（小时）", fontsize=12)
                ax1.set_title(
                    f"{year}年转发间隔分布对比（箱线图）",
                    fontsize=13,
                    fontweight="bold",
                )
                ax1.grid(True, alpha=0.3)

                # 右图：KDE图（使用对数变换）
                ax2 = axes[1]
                for gender in genders:
                    gender_data = interval_hours[interval_hours["gender"] == gender][
                        "interval_hours"
                    ]
                    # 使用对数变换以便更好地展示分布
                    log_data = np.log1p(gender_data)
                    sns.kdeplot(log_data, ax=ax2, label=f"{gender}性", linewidth=2)
                ax2.set_xlabel("转发间隔（小时，对数变换）", fontsize=12)
                ax2.set_ylabel("密度", fontsize=12)
                ax2.set_title(
                    f"{year}年转发间隔分布对比（KDE）", fontsize=13, fontweight="bold"
                )
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                fig_path = os.path.join(
                    OUTPUT_DIR, f"retweet_interval_distribution_{year}.pdf"
                )
                plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
                plt.close()
                print(f"转发间隔分布对比图已保存到: {fig_path}")

    # 保存详细结果
    output_file = os.path.join(OUTPUT_DIR, f"retweet_media_{year}.parquet")
    user_retweet_count.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"转发官方媒体分析结果已保存到: {output_file}")

    # 打印总体统计信息
    print(f"\n总体转发统计:")
    print(f"  转发用户数: {len(user_retweet_count)}")
    print(f"  平均每人转发: {user_retweet_count['retweet_count'].mean():.2f} 次")
    print(f"  最多转发: {user_retweet_count['retweet_count'].max()} 次")
    print(f"  中位数: {user_retweet_count['retweet_count'].median():.2f} 次")

    print(f"\n转发分析完成\n")


def analyze_year(year: int, analysis_type: str = "all"):
    """
    分析指定年份的数据

    Args:
        year: 年份
        analysis_type: 分析类型，'device', 'retweet', 'all'（默认：all）
    """
    # 先加载官方媒体ID
    if analysis_type in ["retweet", "all"]:
        load_official_media_ids()

    print(f"开始分析 {year} 年数据，分析类型: {analysis_type}")

    if analysis_type in ["device", "all"]:
        analyze_device_changes(year)

    if analysis_type in ["retweet", "all"]:
        analyze_retweet_media(year)

    print(f"{year} 年分析完成")


def analyze_multiple_years(years: list, analysis_type: str = "all"):
    """
    分析多个年份的数据

    Args:
        years: 年份列表，例如 [2020, 2021, 2022]
        analysis_type: 分析类型，'device', 'retweet', 'all'（默认：all）
    """
    for year in years:
        analyze_year(year, analysis_type)

    print(f"\n所有年份分析完成")


if __name__ == "__main__":
    fire.Fire(
        {
            "year": analyze_year,
            "years": analyze_multiple_years,
        }
    )
