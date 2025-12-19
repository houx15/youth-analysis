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
from scipy.stats import mannwhitneyu, gaussian_kde

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "analysis_results"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 官方媒体user_id列表（从配置文件加载）
# 注意：现在不再使用全局变量，改为在函数中动态加载
OFFICIAL_MEDIA_IDS = set()

# 省份代码到名称的映射
PROVINCE_CODE_TO_NAME = {
    "11": "北京",
    "12": "天津",
    "13": "河北",
    "14": "山西",
    "15": "内蒙古",
    "21": "辽宁",
    "22": "吉林",
    "23": "黑龙江",
    "31": "上海",
    "32": "江苏",
    "33": "浙江",
    "34": "安徽",
    "35": "福建",
    "36": "江西",
    "37": "山东",
    "41": "河南",
    "42": "湖北",
    "43": "湖南",
    "44": "广东",
    "45": "广西",
    "46": "海南",
    "50": "重庆",
    "51": "四川",
    "52": "贵州",
    "53": "云南",
    "54": "西藏",
    "61": "陕西",
    "62": "甘肃",
    "63": "青海",
    "64": "宁夏",
    "65": "新疆",
    "71": "中国台湾",
    "81": "中国香港",
    "82": "中国澳门",
}

DISTRICT_MAP = {
    "东部": [
        "北京",
        "天津",
        "河北",
        "上海",
        "江苏",
        "浙江",
        "福建",
        "山东",
        "广东",
        "海南",
    ],
    "中部": ["山西", "安徽", "江西", "河南", "湖北", "湖南"],
    "西部": [
        "内蒙古",
        "广西",
        "重庆",
        "四川",
        "贵州",
        "云南",
        "西藏",
        "陕西",
        "甘肃",
        "青海",
        "宁夏",
        "新疆",
    ],
    "东北": ["辽宁", "吉林", "黑龙江"],
}


def load_source_ids(source_type="news"):
    """
    从配置文件加载账号ID（新闻或娱乐）

    Args:
        source_type: "news" 或 "entertain"，指定加载新闻账号还是娱乐账号ID

    Returns:
        set类型的user_id集合
    """
    if source_type == "news":
        try:
            from get_news_ids import load_news_user_ids

            ids = load_news_user_ids()
            print(f"已加载 {len(ids)} 个新闻账号ID")
            return ids
        except ImportError:
            print("警告: 无法导入 get_news_ids 模块，请确保已生成新闻账号ID")
            return set()
        except Exception as e:
            print(f"加载新闻账号ID时出错: {e}")
            return set()
    elif source_type == "entertain":
        try:
            from get_entertain_ids import load_entertain_user_ids

            ids = load_entertain_user_ids()
            print(f"已加载 {len(ids)} 个娱乐账号ID")
            return ids
        except ImportError:
            print("警告: 无法导入 get_entertain_ids 模块，请确保已生成娱乐账号ID")
            return set()
        except Exception as e:
            print(f"加载娱乐账号ID时出错: {e}")
            return set()
    else:
        print(f"错误: 未知的source_type: {source_type}，应为 'news' 或 'entertain'")
        return set()


def load_official_media_ids():
    """从配置文件加载官方媒体ID（保持向后兼容）"""
    global OFFICIAL_MEDIA_IDS
    OFFICIAL_MEDIA_IDS = load_source_ids("news")


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
            "province",
        ]
    else:  # all
        required_columns = [
            "user_id",
            "device",
            "nick_name",
            "is_retweet",
            "r_user_id",
            "gender",
            "province",
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


def analyze_retweet_media(year, force_reanalyze=False, source_type="news"):
    """分析转发官方媒体/娱乐账号情况，特别关注性别差异

    Args:
        year: 年份
        force_reanalyze: 如果为True，即使文件已存在也重新分析
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    source_label = "新闻" if source_type == "news" else "娱乐"
    print(f"\n开始分析 {year} 年转发{source_label}账号情况...")

    # 根据source_type生成不同的文件名
    file_prefix = f"retweet_{source_type}"
    output_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_{year}.parquet")
    gender_summary_file = os.path.join(
        OUTPUT_DIR, f"{file_prefix}_gender_{year}.parquet"
    )
    interval_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_intervals_{year}.parquet")

    if not force_reanalyze:
        # 只要主分析结果文件存在就跳过分析（其他文件可能因为数据特性而不存在）
        if os.path.exists(output_file):
            print(
                f"分析结果文件已存在，跳过分析。如需重新分析，请设置 force_reanalyze=True"
            )
            return

    # 加载对应的ID列表
    source_ids = load_source_ids(source_type)

    if not source_ids:
        source_name = "新闻账号" if source_type == "news" else "娱乐账号"
        print(f"警告: {source_name}ID列表为空，请先运行相应的脚本生成ID文件")
        if source_type == "news":
            print("  运行: python get_news_ids.py")
        else:
            print("  运行: python get_entertain_ids.py")
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

    # 排除对应类型的用户ID（避免自己转发自己）
    # 将user_id转换为字符串进行比较
    data = data[~data["user_id"].astype(str).isin(source_ids)]
    source_name = "新闻账号" if source_type == "news" else "娱乐账号"
    print(f"排除{source_name}用户后，剩余 {len(data)} 条记录")

    # 只保留转发对应类型账号的记录
    # r_user_id已经是字符串格式，user_id需要转换为字符串
    retweet_media = data[
        (data["is_retweet"] == "1") & (data["r_user_id"].astype(str).isin(source_ids))
    ]

    if len(retweet_media) == 0:
        print(f"未找到转发{source_label}账号的记录")
        return

    print(f"共找到 {len(retweet_media)} 条转发{source_label}账号的记录")

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
                retweet_media["time_stamp_num"] - retweet_media["r_time_stamp_num"]
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

    # 保存转发间隔详细数据（用于画图）
    if has_interval_data and has_gender:
        interval_df = retweet_media[["gender", "retweet_interval"]].copy()
        interval_df = interval_df[
            (interval_df["gender"].notna())
            & (interval_df["retweet_interval"].notna())
            & (interval_df["retweet_interval"] > 0)
        ]
        if len(interval_df) > 0:
            interval_df.to_parquet(interval_file, engine="fastparquet", index=False)
            print(f"转发间隔详细数据已保存到: {interval_file}")

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

        # 打印不同性别，retweet count最高的用户的user id，nick_name
        for gender in user_retweet_count["gender"].unique():
            gender_data = user_retweet_count[user_retweet_count["gender"] == gender]
            max_retweet_user = gender_data.sort_values(
                by="retweet_count", ascending=False
            ).iloc[0]
            print(
                f"{gender}: {max_retweet_user['user_id']}, {max_retweet_user['retweet_count']}"
            )  # {max_retweet_user['nick_name']},

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

                # 计算有转发行为的用户数和转发对应类型账号记录者占比
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
                    f"  转发{source_label}账号记录者占比: {retweet_media_ratio:.4f} ({retweet_media_ratio*100:.2f}%)"
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
        gender_stats_df.to_parquet(
            gender_summary_file, engine="fastparquet", index=False
        )
        print(f"性别统计摘要已保存到: {gender_summary_file}")

    # 保存详细结果
    user_retweet_count.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"转发{source_label}账号分析结果已保存到: {output_file}")

    # 打印总体统计信息
    print(f"\n总体转发统计:")
    print(f"  转发用户数: {len(user_retweet_count)}")
    print(f"  平均每人转发: {user_retweet_count['retweet_count'].mean():.2f} 次")
    print(f"  最多转发: {user_retweet_count['retweet_count'].max()} 次")
    print(f"  中位数: {user_retweet_count['retweet_count'].median():.2f} 次")

    print(f"\n转发{source_label}账号分析完成\n")


def convert_province_code(code):
    """将省份代码转换为省份名称"""
    if pd.isna(code):
        return None
    # 统一转换为字符串格式处理
    if isinstance(code, (int, float)):
        code_str = str(int(code))  # 去掉小数点
    else:
        code_str = str(code).strip()

    # 如果是编码，转换为名称
    if code_str in PROVINCE_CODE_TO_NAME:
        return PROVINCE_CODE_TO_NAME[code_str]
    # 如果已经是名称，直接返回
    return code_str


def get_district_from_province(province_name):
    """根据省份名称获取所属地区"""
    if pd.isna(province_name) or province_name is None:
        return None

    province_str = str(province_name).strip()

    # 遍历地区映射，查找省份所属地区
    for district, province_list in DISTRICT_MAP.items():
        if province_str in province_list:
            return district

    return None


def analyze_retweet_media_by_province(year, force_reanalyze=False, source_type="news"):
    """按省份分析转发官方媒体/娱乐账号情况的性别差异

    Args:
        year: 年份
        force_reanalyze: 如果为True，即使文件已存在也重新分析
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    source_label = "新闻" if source_type == "news" else "娱乐"
    print(f"\n开始分析 {year} 年按省份的转发{source_label}账号性别差异...")

    # 根据source_type生成不同的文件名
    file_prefix = f"retweet_{source_type}"
    output_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_province_{year}.parquet")

    if not force_reanalyze:
        if os.path.exists(output_file):
            print(
                f"分析结果文件已存在，跳过分析。如需重新分析，请设置 force_reanalyze=True"
            )
            return

    # 加载对应的ID列表
    source_ids = load_source_ids(source_type)

    if not source_ids:
        source_name = "新闻账号" if source_type == "news" else "娱乐账号"
        print(f"警告: {source_name}ID列表为空，请先运行相应的脚本生成ID文件")
        if source_type == "news":
            print("  运行: python get_news_ids.py")
        else:
            print("  运行: python get_entertain_ids.py")
        return

    # 加载数据
    data = load_data_for_year(year, "retweet")
    if data is None:
        return

    # 检查是否有性别和省份字段
    if "gender" not in data.columns:
        print("数据不包含性别字段，无法进行性别分析")
        return

    if "province" not in data.columns:
        print("数据不包含省份字段，无法进行省份分析")
        return

    # 转换省份代码为省份名称
    data = data.copy()
    data["province_name"] = data["province"].apply(convert_province_code)
    data = data.dropna(subset=["province_name", "gender"])

    # 排除对应类型的用户ID
    source_name = "新闻账号" if source_type == "news" else "娱乐账号"
    data = data[~data["user_id"].astype(str).isin(source_ids)]
    print(f"排除{source_name}用户后，剩余 {len(data)} 条记录")

    # 过滤有效省份（排除未知省份）
    valid_provinces = set(PROVINCE_CODE_TO_NAME.values())
    data = data[data["province_name"].isin(valid_provinces)]
    print(f"过滤有效省份后，剩余 {len(data)} 条记录")

    # 计算转发间隔（如果可用）
    has_interval_data = False
    if "time_stamp" in data.columns and "r_time_stamp" in data.columns:
        try:
            data = data.copy()
            data["time_stamp_num"] = pd.to_numeric(data["time_stamp"], errors="coerce")
            data["r_time_stamp_num"] = pd.to_numeric(
                data["r_time_stamp"], errors="coerce"
            )
            data["retweet_interval"] = data["time_stamp_num"] - data["r_time_stamp_num"]
            has_interval_data = True
        except Exception as e:
            print(f"计算转发间隔时出错: {e}")
            has_interval_data = False

    # 按省份和性别分析
    province_stats = []

    for province in sorted(data["province_name"].unique()):
        province_data = data[data["province_name"] == province]

        for gender in ["m", "f", "男", "女"]:
            gender_data = province_data[province_data["gender"] == gender]

            if len(gender_data) == 0:
                continue

            # 统计总用户数（去重）
            total_users = gender_data["user_id"].nunique()

            # 统计转发对应类型账号的用户
            retweet_media_users = gender_data[
                (gender_data["is_retweet"] == "1")
                & (gender_data["r_user_id"].astype(str).isin(source_ids))
            ]["user_id"].nunique()

            # 计算不转发对应类型账号帖子的比例
            non_retweet_ratio = (
                1 - (retweet_media_users / total_users) if total_users > 0 else 0
            )

            # 99%置信区间的z值（用于其他指标）
            z_99 = 2.576

            # 计算平均转发次数（只统计转发对应类型账号的用户）
            retweet_media_records = gender_data[
                (gender_data["is_retweet"] == "1")
                & (gender_data["r_user_id"].astype(str).isin(source_ids))
            ]

            if retweet_media_users > 0:
                user_retweet_counts = retweet_media_records.groupby("user_id").size()
                avg_retweet_count = user_retweet_counts.mean()

                # 计算平均转发次数的99%置信区间
                n_retweet = len(user_retweet_counts)
                if n_retweet > 1:
                    std_retweet = user_retweet_counts.std()
                    se_retweet = std_retweet / np.sqrt(n_retweet)
                    ci_lower_count = avg_retweet_count - z_99 * se_retweet
                    ci_upper_count = avg_retweet_count + z_99 * se_retweet
                else:
                    ci_lower_count = avg_retweet_count
                    ci_upper_count = avg_retweet_count
            else:
                avg_retweet_count = 0
                ci_lower_count = 0
                ci_upper_count = 0

            # 计算平均转发间隔（如果可用）
            avg_retweet_interval = None
            ci_lower_interval = None
            ci_upper_interval = None

            if has_interval_data and retweet_media_users > 0:
                valid_intervals = retweet_media_records[
                    (retweet_media_records["retweet_interval"].notna())
                    & (retweet_media_records["retweet_interval"] > 0)
                ]["retweet_interval"]

                if len(valid_intervals) > 0:
                    avg_retweet_interval = valid_intervals.mean()

                    # 计算平均转发间隔的99%置信区间
                    n_interval = len(valid_intervals)
                    if n_interval > 1:
                        std_interval = valid_intervals.std()
                        se_interval = std_interval / np.sqrt(n_interval)
                        ci_lower_interval = max(
                            0, avg_retweet_interval - z_99 * se_interval
                        )
                        ci_upper_interval = avg_retweet_interval + z_99 * se_interval
                    else:
                        ci_lower_interval = avg_retweet_interval
                        ci_upper_interval = avg_retweet_interval

            province_stats.append(
                {
                    "province": province,
                    "gender": gender,
                    "total_users": total_users,
                    "retweet_media_users": retweet_media_users,
                    "non_retweet_ratio": non_retweet_ratio,
                    "avg_retweet_count": avg_retweet_count,
                    "avg_retweet_count_ci_lower": ci_lower_count,
                    "avg_retweet_count_ci_upper": ci_upper_count,
                    "avg_retweet_interval": avg_retweet_interval,
                    "avg_retweet_interval_ci_lower": ci_lower_interval,
                    "avg_retweet_interval_ci_upper": ci_upper_interval,
                }
            )

    # 保存结果
    province_stats_df = pd.DataFrame(province_stats)
    province_stats_df.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"省份分析结果已保存到: {output_file}")
    print(f"共分析了 {len(province_stats_df)} 个省份-性别组合")

    print(f"\n按省份的转发{source_label}账号分析完成\n")


def analyze_retweet_media_by_district(year, force_reanalyze=False, source_type="news"):
    """按地区分析转发官方媒体/娱乐账号情况的性别差异

    Args:
        year: 年份
        force_reanalyze: 如果为True，即使文件已存在也重新分析
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    source_label = "新闻" if source_type == "news" else "娱乐"
    print(f"\n开始分析 {year} 年按地区的转发{source_label}账号性别差异...")

    # 根据source_type生成不同的文件名
    file_prefix = f"retweet_{source_type}"
    output_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_district_{year}.parquet")

    if not force_reanalyze:
        if os.path.exists(output_file):
            print(
                f"分析结果文件已存在，跳过分析。如需重新分析，请设置 force_reanalyze=True"
            )
            return

    # 加载对应的ID列表
    source_ids = load_source_ids(source_type)

    if not source_ids:
        source_name = "新闻账号" if source_type == "news" else "娱乐账号"
        print(f"警告: {source_name}ID列表为空，请先运行相应的脚本生成ID文件")
        if source_type == "news":
            print("  运行: python get_news_ids.py")
        else:
            print("  运行: python get_entertain_ids.py")
        return

    # 加载数据
    data = load_data_for_year(year, "retweet")
    if data is None:
        return

    # 检查是否有性别和省份字段
    if "gender" not in data.columns:
        print("数据不包含性别字段，无法进行性别分析")
        return

    if "province" not in data.columns:
        print("数据不包含省份字段，无法进行地区分析")
        return

    # 转换省份代码为省份名称，然后映射到地区
    data = data.copy()
    data["province_name"] = data["province"].apply(convert_province_code)
    data["district"] = data["province_name"].apply(get_district_from_province)
    data = data.dropna(subset=["district", "gender"])

    # 排除对应类型的用户ID
    source_name = "新闻账号" if source_type == "news" else "娱乐账号"
    data = data[~data["user_id"].astype(str).isin(source_ids)]
    print(f"排除{source_name}用户后，剩余 {len(data)} 条记录")

    # 过滤有效地区
    valid_districts = set(DISTRICT_MAP.keys())
    data = data[data["district"].isin(valid_districts)]
    print(f"过滤有效地区后，剩余 {len(data)} 条记录")

    # 计算转发间隔（如果可用）
    has_interval_data = False
    if "time_stamp" in data.columns and "r_time_stamp" in data.columns:
        try:
            data = data.copy()
            data["time_stamp_num"] = pd.to_numeric(data["time_stamp"], errors="coerce")
            data["r_time_stamp_num"] = pd.to_numeric(
                data["r_time_stamp"], errors="coerce"
            )
            data["retweet_interval"] = data["time_stamp_num"] - data["r_time_stamp_num"]
            has_interval_data = True
        except Exception as e:
            print(f"计算转发间隔时出错: {e}")
            has_interval_data = False

    # 按地区和性别分析
    district_stats = []

    for district in sorted(data["district"].unique()):
        district_data = data[data["district"] == district]

        for gender in ["m", "f", "男", "女"]:
            gender_data = district_data[district_data["gender"] == gender]

            if len(gender_data) == 0:
                continue

            # 统计总用户数（去重）
            total_users = gender_data["user_id"].nunique()

            # 统计转发对应类型账号的用户
            retweet_media_users = gender_data[
                (gender_data["is_retweet"] == "1")
                & (gender_data["r_user_id"].astype(str).isin(source_ids))
            ]["user_id"].nunique()

            # 计算不转发对应类型账号帖子的比例
            non_retweet_ratio = (
                1 - (retweet_media_users / total_users) if total_users > 0 else 0
            )

            # 99%置信区间的z值（用于其他指标）
            z_99 = 2.576

            # 计算平均转发次数（只统计转发对应类型账号的用户）
            retweet_media_records = gender_data[
                (gender_data["is_retweet"] == "1")
                & (gender_data["r_user_id"].astype(str).isin(source_ids))
            ]

            if retweet_media_users > 0:
                user_retweet_counts = retweet_media_records.groupby("user_id").size()
                avg_retweet_count = user_retweet_counts.mean()

                # 计算平均转发次数的99%置信区间
                n_retweet = len(user_retweet_counts)
                if n_retweet > 1:
                    std_retweet = user_retweet_counts.std()
                    se_retweet = std_retweet / np.sqrt(n_retweet)
                    ci_lower_count = avg_retweet_count - z_99 * se_retweet
                    ci_upper_count = avg_retweet_count + z_99 * se_retweet
                else:
                    ci_lower_count = avg_retweet_count
                    ci_upper_count = avg_retweet_count
            else:
                avg_retweet_count = 0
                ci_lower_count = 0
                ci_upper_count = 0

            # 计算平均转发间隔（如果可用）
            avg_retweet_interval = None
            ci_lower_interval = None
            ci_upper_interval = None

            if has_interval_data and retweet_media_users > 0:
                valid_intervals = retweet_media_records[
                    (retweet_media_records["retweet_interval"].notna())
                    & (retweet_media_records["retweet_interval"] > 0)
                ]["retweet_interval"]

                if len(valid_intervals) > 0:
                    avg_retweet_interval = valid_intervals.mean()

                    # 计算平均转发间隔的99%置信区间
                    n_interval = len(valid_intervals)
                    if n_interval > 1:
                        std_interval = valid_intervals.std()
                        se_interval = std_interval / np.sqrt(n_interval)
                        ci_lower_interval = max(
                            0, avg_retweet_interval - z_99 * se_interval
                        )
                        ci_upper_interval = avg_retweet_interval + z_99 * se_interval
                    else:
                        ci_lower_interval = avg_retweet_interval
                        ci_upper_interval = avg_retweet_interval

            district_stats.append(
                {
                    "district": district,
                    "gender": gender,
                    "total_users": total_users,
                    "retweet_media_users": retweet_media_users,
                    "non_retweet_ratio": non_retweet_ratio,
                    "avg_retweet_count": avg_retweet_count,
                    "avg_retweet_count_ci_lower": ci_lower_count,
                    "avg_retweet_count_ci_upper": ci_upper_count,
                    "avg_retweet_interval": avg_retweet_interval,
                    "avg_retweet_interval_ci_lower": ci_lower_interval,
                    "avg_retweet_interval_ci_upper": ci_upper_interval,
                }
            )

    # 保存结果
    district_stats_df = pd.DataFrame(district_stats)
    district_stats_df.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"地区分析结果已保存到: {output_file}")
    print(f"共分析了 {len(district_stats_df)} 个地区-性别组合")

    print(f"\n按地区的转发{source_label}账号分析完成\n")


def get_gender_label(gender):
    """将性别代码转换为中文标签"""
    gender_map = {"m": "男", "f": "女", "男": "男", "女": "女"}
    return gender_map.get(gender, gender)


def get_gender_color(gender):
    """获取性别对应的颜色"""
    color_map = {"m": "#20AEE6", "f": "#ff7333", "男": "#20AEE6", "女": "#ff7333"}
    return color_map.get(gender, "#808080")  # 默认灰色


def visualize_distribution_4_subplots(
    data_df, value_col, gender_col, title, output_path, xlabel, ylabel, log_scale=True
):
    """绘制4个子图的分布分析 (KDE, Boxplot, Mean+CI, ECDF)"""
    print(f"正在绘制: {title}")

    # 准备数据
    genders = data_df[gender_col].unique()
    # 过滤无效数据
    valid_df = data_df.dropna(subset=[value_col, gender_col])
    if log_scale:
        valid_df = valid_df[valid_df[value_col] > 0]

    if len(valid_df) == 0:
        print(f"没有有效数据用于绘制 {title}")
        return

    # 准备绘图数据字典
    all_data_dict = {}
    for gender in genders:
        vals = valid_df[valid_df[gender_col] == gender][value_col].values
        if len(vals) > 0:
            all_data_dict[gender] = vals

    if not all_data_dict:
        return

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight="bold")

    # 1. KDE (Log X)
    ax1 = axes[0, 0]
    all_values = valid_df[value_col].values

    if log_scale:
        x_min = np.log10(all_values.min())
        x_max = np.log10(all_values.max())
        x_plot_log = np.linspace(x_min, x_max, 1000)
        x_plot = 10**x_plot_log
    else:
        x_min = all_values.min()
        x_max = all_values.max()
        x_plot = np.linspace(x_min, x_max, 1000)

    for gender in genders:
        if gender not in all_data_dict:
            continue
        data = all_data_dict[gender]
        color = get_gender_color(gender)
        label = get_gender_label(gender)

        try:
            if log_scale:
                # 拟合 log10 数据
                kde = gaussian_kde(np.log10(data))
                kde_values = kde(x_plot_log)
            else:
                kde = gaussian_kde(data)
                kde_values = kde(x_plot)

            ax1.fill_between(x_plot, kde_values, alpha=0.3, color=color, label=label)
            ax1.plot(x_plot, kde_values, linewidth=2, color=color)
        except Exception as e:
            print(f"KDE绘制失败 ({gender}): {e}")

    if log_scale:
        ax1.set_xscale("log")
    ax1.set_title("1. 整体分布形态 (KDE)", fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("密度", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Boxplot
    ax2 = axes[0, 1]
    plot_df = valid_df.copy()
    plot_df["GenderLabel"] = plot_df[gender_col].apply(get_gender_label)

    palette = {get_gender_label(g): get_gender_color(g) for g in genders}

    sns.boxplot(
        data=plot_df,
        x="GenderLabel",
        y=value_col,
        palette=palette,
        notch=True,
        width=0.4,
        showfliers=False,
        ax=ax2,
    )
    if log_scale:
        ax2.set_yscale("log")
    ax2.set_title("2. 中位数差异检测 (Boxplot)", fontsize=14)
    ax2.set_xlabel("性别", fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Mean + CI
    ax3 = axes[1, 0]
    stats = (
        plot_df.groupby("GenderLabel")[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci99"] = 2.576 * stats["se"]

    x_coords = range(len(stats))
    ax3.errorbar(
        x=x_coords,
        y=stats["mean"],
        yerr=stats["ci99"],
        fmt="none",
        ecolor="black",
        capsize=8,
        elinewidth=2,
    )

    for i, row in stats.iterrows():
        ax3.scatter(
            x=i, y=row["mean"], s=150, color=palette[row["GenderLabel"]], zorder=5
        )

    ax3.set_xticks(x_coords)
    ax3.set_xticklabels(stats["GenderLabel"], fontsize=12)
    ax3.set_title("3. 均值差异对比 (Mean + 99% CI)", fontsize=14)
    ax3.set_xlabel("性别", fontsize=12)
    ax3.set_ylabel(f"平均 {ylabel}", fontsize=12)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. ECDF + Stats
    ax4 = axes[1, 1]
    for gender in genders:
        if gender not in all_data_dict:
            continue
        data = all_data_dict[gender]
        color = get_gender_color(gender)
        label = get_gender_label(gender)

        sorted_data = np.sort(data)
        ecdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax4.plot(sorted_data, ecdf_values, linewidth=2, color=color, label=label)

    if log_scale:
        ax4.set_xscale("log")
    ax4.set_title("4. 累积分布 (ECDF) 与 统计检验", fontsize=14)
    ax4.set_xlabel(xlabel, fontsize=12)
    ax4.set_ylabel("累积比例", fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Mann-Whitney U Test
    if len(genders) == 2:
        g1, g2 = genders[0], genders[1]
        if g1 in all_data_dict and g2 in all_data_dict:
            try:
                stat, p_val = mannwhitneyu(
                    all_data_dict[g1], all_data_dict[g2], alternative="two-sided"
                )
                p_text = "P < 0.001" if p_val < 0.001 else f"P = {p_val:.4f}"

                props = dict(
                    boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"
                )
                ax4.text(
                    0.05,
                    0.15,
                    f"Mann-Whitney U Test:\n结果: {p_text}",
                    transform=ax4.transAxes,
                    fontsize=11,
                    bbox=props,
                    verticalalignment="bottom",
                )
            except Exception as e:
                print(f"统计检验失败: {e}")

    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"图表已保存到: {output_path}")


def visualize_retweet_media(year, source_type="news"):
    """绘制转发官方媒体/娱乐账号情况的图表

    Args:
        year: 年份
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    source_label = "新闻" if source_type == "news" else "娱乐"
    print(f"\n开始绘制 {year} 年转发{source_label}账号图表...")

    # 根据source_type生成不同的文件名
    file_prefix = f"retweet_{source_type}"
    output_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_{year}.parquet")
    interval_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_intervals_{year}.parquet")

    if not os.path.exists(output_file):
        print(f"错误: 分析结果文件不存在: {output_file}")
        print(f"请先运行分析: analyze_retweet_media({year})")
        return

    # 加载用户转发次数数据
    user_retweet_count = pd.read_parquet(output_file, engine="fastparquet")
    print(f"已加载 {len(user_retweet_count)} 个用户的转发数据")

    # 检查是否有性别字段
    has_gender = "gender" in user_retweet_count.columns
    if not has_gender:
        print("数据不包含性别字段，无法绘制性别对比图")
        return

    # 1. 绘制转发次数分布对比图 (4 subplots)
    fig_path_count = os.path.join(
        OUTPUT_DIR, f"retweet_count_distribution_{source_type}_{year}.pdf"
    )
    visualize_distribution_4_subplots(
        user_retweet_count,
        "retweet_count",
        "gender",
        f"{year}年转发{source_label}账号次数分布对比",
        fig_path_count,
        "转发次数",
        "转发次数",
        log_scale=True,
    )

    # 加载转发间隔数据（如果存在）
    if os.path.exists(interval_file):
        try:
            interval_data = pd.read_parquet(interval_file, engine="fastparquet")
            print(f"已加载 {len(interval_data)} 条转发间隔数据")

            # 2. 绘制转发间隔分布对比图 (4 subplots)
            fig_path_interval = os.path.join(
                OUTPUT_DIR, f"retweet_interval_distribution_{source_type}_{year}.pdf"
            )
            visualize_distribution_4_subplots(
                interval_data,
                "retweet_interval",
                "gender",
                f"{year}年转发{source_label}账号间隔分布对比",
                fig_path_interval,
                "转发间隔 (秒)",
                "转发间隔 (秒)",
                log_scale=True,
            )
        except Exception as e:
            print(f"加载转发间隔数据时出错: {e}")
    else:
        print("没有转发间隔数据，跳过转发间隔分布图绘制")

    print(f"\n图表绘制完成\n")


def visualize_province_gender_gap(year, source_type="news"):
    """绘制按省份的性别差异图表（三个子图）

    Args:
        year: 年份
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    source_label = "新闻" if source_type == "news" else "娱乐"
    print(f"\n开始绘制 {year} 年按省份的转发{source_label}账号性别差异图表...")

    # 根据source_type生成不同的文件名
    file_prefix = f"retweet_{source_type}"
    output_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_province_{year}.parquet")

    if not os.path.exists(output_file):
        print(f"错误: 分析结果文件不存在: {output_file}")
        print(f"请先运行分析: analyze_retweet_media_by_province({year})")
        return

    # 加载数据
    province_stats_df = pd.read_parquet(output_file, engine="fastparquet")
    print(f"已加载 {len(province_stats_df)} 条省份-性别数据")

    # 统一性别标签（将m/f转换为男/女）
    province_stats_df = province_stats_df.copy()
    province_stats_df["gender_label"] = province_stats_df["gender"].apply(
        lambda x: "男" if x in ["m", "男"] else "女" if x in ["f", "女"] else x
    )

    # 创建三个子图（根据省份数量调整大小）
    num_provinces = province_stats_df["province"].nunique()
    fig_width = max(24, num_provinces * 0.8)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 8), constrained_layout=True)
    fig.suptitle(
        f"{year}年各省份转发{source_label}账号性别差异分析",
        fontsize=18,
        fontweight="bold",
    )

    # 按省份分组，计算性别差异
    provinces = sorted(province_stats_df["province"].unique())

    # 准备数据
    plot_data = []
    for province in provinces:
        province_data = province_stats_df[province_stats_df["province"] == province]
        male_data = province_data[province_data["gender_label"] == "男"]
        female_data = province_data[province_data["gender_label"] == "女"]

        if len(male_data) == 0 or len(female_data) == 0:
            continue

        male_row = male_data.iloc[0]
        female_row = female_data.iloc[0]

        plot_data.append(
            {
                "province": province,
                "male_non_retweet_ratio": male_row["non_retweet_ratio"],
                "female_non_retweet_ratio": female_row["non_retweet_ratio"],
                "male_avg_retweet_count": male_row["avg_retweet_count"],
                "female_avg_retweet_count": female_row["avg_retweet_count"],
                "male_count_ci_lower": male_row["avg_retweet_count_ci_lower"],
                "male_count_ci_upper": male_row["avg_retweet_count_ci_upper"],
                "female_count_ci_lower": female_row["avg_retweet_count_ci_lower"],
                "female_count_ci_upper": female_row["avg_retweet_count_ci_upper"],
                "male_avg_interval": (
                    male_row["avg_retweet_interval"]
                    if pd.notna(male_row["avg_retweet_interval"])
                    else None
                ),
                "female_avg_interval": (
                    female_row["avg_retweet_interval"]
                    if pd.notna(female_row["avg_retweet_interval"])
                    else None
                ),
                "male_interval_ci_lower": (
                    male_row["avg_retweet_interval_ci_lower"]
                    if pd.notna(male_row["avg_retweet_interval_ci_lower"])
                    else None
                ),
                "male_interval_ci_upper": (
                    male_row["avg_retweet_interval_ci_upper"]
                    if pd.notna(male_row["avg_retweet_interval_ci_upper"])
                    else None
                ),
                "female_interval_ci_lower": (
                    female_row["avg_retweet_interval_ci_lower"]
                    if pd.notna(female_row["avg_retweet_interval_ci_lower"])
                    else None
                ),
                "female_interval_ci_upper": (
                    female_row["avg_retweet_interval_ci_upper"]
                    if pd.notna(female_row["avg_retweet_interval_ci_upper"])
                    else None
                ),
            }
        )

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        print("没有足够的数据用于绘图")
        return

    # 1. 不转发媒体帖子比例
    ax1 = axes[0]
    x_pos = np.arange(len(plot_df))

    male_ratios = plot_df["male_non_retweet_ratio"].values
    female_ratios = plot_df["female_non_retweet_ratio"].values

    ax1.plot(
        x_pos,
        male_ratios,
        marker="o",
        label="男性",
        color=get_gender_color("m"),
        linewidth=3,
        markersize=6,
        alpha=0.8,
    )
    ax1.plot(
        x_pos,
        female_ratios,
        marker="s",
        label="女性",
        color=get_gender_color("f"),
        linewidth=3,
        markersize=6,
        alpha=0.8,
    )

    ax1.set_xlabel("省份", fontsize=12)
    ax1.set_ylabel("不转发媒体帖子比例", fontsize=12)
    ax1.set_title("1. 不转发媒体帖子比例", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(plot_df["province"].values, rotation=45, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim([0, 1.1])

    # 2. 平均转发次数
    ax2 = axes[1]

    male_counts = plot_df["male_avg_retweet_count"].values
    female_counts = plot_df["female_avg_retweet_count"].values

    # 计算置信区间
    male_ci_lower = plot_df["male_count_ci_lower"].values
    male_ci_upper = plot_df["male_count_ci_upper"].values
    female_ci_lower = plot_df["female_count_ci_lower"].values
    female_ci_upper = plot_df["female_count_ci_upper"].values

    male_color = get_gender_color("m")
    female_color = get_gender_color("f")

    # 计算误差范围（用于errorbar）
    male_err_lower = male_counts - male_ci_lower
    male_err_upper = male_ci_upper - male_counts
    female_err_lower = female_counts - female_ci_lower
    female_err_upper = female_ci_upper - female_counts

    # 绘制折线
    ax2.plot(
        x_pos,
        male_counts,
        marker="o",
        label="男性",
        color=male_color,
        linewidth=3,
        markersize=6,
        alpha=0.8,
    )
    ax2.plot(
        x_pos,
        female_counts,
        marker="s",
        label="女性",
        color=female_color,
        linewidth=3,
        markersize=6,
        alpha=0.8,
    )

    # 添加误差条
    valid_male_ci = pd.notna(male_ci_lower) & pd.notna(male_ci_upper)
    if valid_male_ci.any():
        ax2.errorbar(
            x_pos[valid_male_ci],
            male_counts[valid_male_ci],
            yerr=[
                male_err_lower[valid_male_ci],
                male_err_upper[valid_male_ci],
            ],
            fmt="none",
            color=male_color,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
        )

    valid_female_ci = pd.notna(female_ci_lower) & pd.notna(female_ci_upper)
    if valid_female_ci.any():
        ax2.errorbar(
            x_pos[valid_female_ci],
            female_counts[valid_female_ci],
            yerr=[
                female_err_lower[valid_female_ci],
                female_err_upper[valid_female_ci],
            ],
            fmt="none",
            color=female_color,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
        )

    ax2.set_xlabel("省份", fontsize=12)
    ax2.set_ylabel("平均转发次数", fontsize=12)
    ax2.set_title("2. 平均转发次数（99% CI）", fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plot_df["province"].values, rotation=45, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. 平均转发间隔（只显示有数据的省份）
    ax3 = axes[2]

    # 过滤出有间隔数据的省份
    interval_data = plot_df[
        plot_df["male_avg_interval"].notna() & plot_df["female_avg_interval"].notna()
    ].copy()

    if len(interval_data) > 0:
        x_pos_interval = np.arange(len(interval_data))

        male_intervals = interval_data["male_avg_interval"].values
        female_intervals = interval_data["female_avg_interval"].values

        # 转换为小时
        male_intervals_hours = male_intervals / 3600
        female_intervals_hours = female_intervals / 3600

        # 计算置信区间（转换为小时）
        male_interval_ci_lower = interval_data["male_interval_ci_lower"].values / 3600
        male_interval_ci_upper = interval_data["male_interval_ci_upper"].values / 3600
        female_interval_ci_lower = (
            interval_data["female_interval_ci_lower"].values / 3600
        )
        female_interval_ci_upper = (
            interval_data["female_interval_ci_upper"].values / 3600
        )

        male_color = get_gender_color("m")
        female_color = get_gender_color("f")

        # 计算误差范围（用于errorbar）
        male_err_lower = male_intervals_hours - male_interval_ci_lower
        male_err_upper = male_interval_ci_upper - male_intervals_hours
        female_err_lower = female_intervals_hours - female_interval_ci_lower
        female_err_upper = female_interval_ci_upper - female_intervals_hours

        # 绘制折线
        ax3.plot(
            x_pos_interval,
            male_intervals_hours,
            marker="o",
            label="男性",
            color=male_color,
            linewidth=3,
            markersize=6,
            alpha=0.8,
        )
        ax3.plot(
            x_pos_interval,
            female_intervals_hours,
            marker="s",
            label="女性",
            color=female_color,
            linewidth=3,
            markersize=6,
            alpha=0.8,
        )

        # 添加误差条
        valid_male_ci = pd.notna(interval_data["male_interval_ci_lower"]) & pd.notna(
            interval_data["male_interval_ci_upper"]
        )
        if valid_male_ci.any():
            ax3.errorbar(
                x_pos_interval[valid_male_ci],
                male_intervals_hours[valid_male_ci],
                yerr=[
                    male_err_lower[valid_male_ci],
                    male_err_upper[valid_male_ci],
                ],
                fmt="none",
                color=male_color,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.7,
            )

        valid_female_ci = pd.notna(
            interval_data["female_interval_ci_lower"]
        ) & pd.notna(interval_data["female_interval_ci_upper"])
        if valid_female_ci.any():
            ax3.errorbar(
                x_pos_interval[valid_female_ci],
                female_intervals_hours[valid_female_ci],
                yerr=[
                    female_err_lower[valid_female_ci],
                    female_err_upper[valid_female_ci],
                ],
                fmt="none",
                color=female_color,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.7,
            )

        ax3.set_xlabel("省份", fontsize=12)
        ax3.set_ylabel("平均转发间隔 (小时)", fontsize=12)
        ax3.set_title("3. 平均转发间隔（99% CI）", fontsize=14)
        ax3.set_xticks(x_pos_interval)
        ax3.set_xticklabels(
            interval_data["province"].values, rotation=45, ha="right", fontsize=8
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(
            0.5,
            0.5,
            "无转发间隔数据",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=14,
        )
        ax3.set_title("3. 平均转发间隔", fontsize=14)

    # 保存图表
    fig_path = os.path.join(OUTPUT_DIR, f"province_gender_gap_{source_type}_{year}.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"图表已保存到: {fig_path}")
    print(f"\n省份性别差异图表绘制完成\n")


def visualize_district_gender_gap(year, source_type="news"):
    """绘制按地区的性别差异图表（三个子图）

    Args:
        year: 年份
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    source_label = "新闻" if source_type == "news" else "娱乐"
    print(f"\n开始绘制 {year} 年按地区的转发{source_label}账号性别差异图表...")

    # 根据source_type生成不同的文件名
    file_prefix = f"retweet_{source_type}"
    output_file = os.path.join(OUTPUT_DIR, f"{file_prefix}_district_{year}.parquet")

    if not os.path.exists(output_file):
        print(f"错误: 分析结果文件不存在: {output_file}")
        print(f"请先运行分析: analyze_retweet_media_by_district({year})")
        return

    # 加载数据
    district_stats_df = pd.read_parquet(output_file, engine="fastparquet")
    print(f"已加载 {len(district_stats_df)} 条地区-性别数据")

    # 统一性别标签（将m/f转换为男/女）
    district_stats_df = district_stats_df.copy()
    district_stats_df["gender_label"] = district_stats_df["gender"].apply(
        lambda x: "男" if x in ["m", "男"] else "女" if x in ["f", "女"] else x
    )

    # 创建三个子图
    districts = sorted(district_stats_df["district"].unique())
    fig_width = max(16, len(districts) * 2)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 6), constrained_layout=True)
    fig.suptitle(
        f"{year}年各地区转发{source_label}账号性别差异分析",
        fontsize=18,
        fontweight="bold",
    )

    # 准备数据
    plot_data = []
    for district in districts:
        district_data = district_stats_df[district_stats_df["district"] == district]
        male_data = district_data[district_data["gender_label"] == "男"]
        female_data = district_data[district_data["gender_label"] == "女"]

        if len(male_data) == 0 or len(female_data) == 0:
            continue

        male_row = male_data.iloc[0]
        female_row = female_data.iloc[0]

        plot_data.append(
            {
                "district": district,
                "male_non_retweet_ratio": male_row["non_retweet_ratio"],
                "female_non_retweet_ratio": female_row["non_retweet_ratio"],
                "male_avg_retweet_count": male_row["avg_retweet_count"],
                "female_avg_retweet_count": female_row["avg_retweet_count"],
                "male_count_ci_lower": male_row["avg_retweet_count_ci_lower"],
                "male_count_ci_upper": male_row["avg_retweet_count_ci_upper"],
                "female_count_ci_lower": female_row["avg_retweet_count_ci_lower"],
                "female_count_ci_upper": female_row["avg_retweet_count_ci_upper"],
                "male_avg_interval": (
                    male_row["avg_retweet_interval"]
                    if pd.notna(male_row["avg_retweet_interval"])
                    else None
                ),
                "female_avg_interval": (
                    female_row["avg_retweet_interval"]
                    if pd.notna(female_row["avg_retweet_interval"])
                    else None
                ),
                "male_interval_ci_lower": (
                    male_row["avg_retweet_interval_ci_lower"]
                    if pd.notna(male_row["avg_retweet_interval_ci_lower"])
                    else None
                ),
                "male_interval_ci_upper": (
                    male_row["avg_retweet_interval_ci_upper"]
                    if pd.notna(male_row["avg_retweet_interval_ci_upper"])
                    else None
                ),
                "female_interval_ci_lower": (
                    female_row["avg_retweet_interval_ci_lower"]
                    if pd.notna(female_row["avg_retweet_interval_ci_lower"])
                    else None
                ),
                "female_interval_ci_upper": (
                    female_row["avg_retweet_interval_ci_upper"]
                    if pd.notna(female_row["avg_retweet_interval_ci_upper"])
                    else None
                ),
            }
        )

    plot_df = pd.DataFrame(plot_data)

    if len(plot_df) == 0:
        print("没有足够的数据用于绘图")
        return

    # 1. 不转发媒体帖子比例
    ax1 = axes[0]
    x_pos = np.arange(len(plot_df))

    male_ratios = plot_df["male_non_retweet_ratio"].values
    female_ratios = plot_df["female_non_retweet_ratio"].values

    ax1.plot(
        x_pos,
        male_ratios,
        marker="o",
        label="男性",
        color=get_gender_color("m"),
        linewidth=3,
        markersize=8,
        alpha=0.8,
    )
    ax1.plot(
        x_pos,
        female_ratios,
        marker="s",
        label="女性",
        color=get_gender_color("f"),
        linewidth=3,
        markersize=8,
        alpha=0.8,
    )

    ax1.set_xlabel("地区", fontsize=12)
    ax1.set_ylabel("不转发媒体帖子比例", fontsize=12)
    ax1.set_title("1. 不转发媒体帖子比例", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        plot_df["district"].values, rotation=0, ha="center", fontsize=10
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim([0, 1.1])

    # 2. 平均转发次数
    ax2 = axes[1]

    male_counts = plot_df["male_avg_retweet_count"].values
    female_counts = plot_df["female_avg_retweet_count"].values

    # 计算置信区间
    male_ci_lower = plot_df["male_count_ci_lower"].values
    male_ci_upper = plot_df["male_count_ci_upper"].values
    female_ci_lower = plot_df["female_count_ci_lower"].values
    female_ci_upper = plot_df["female_count_ci_upper"].values

    male_color = get_gender_color("m")
    female_color = get_gender_color("f")

    # 计算误差范围（用于errorbar）
    male_err_lower = male_counts - male_ci_lower
    male_err_upper = male_ci_upper - male_counts
    female_err_lower = female_counts - female_ci_lower
    female_err_upper = female_ci_upper - female_counts

    # 绘制折线
    ax2.plot(
        x_pos,
        male_counts,
        marker="o",
        label="男性",
        color=male_color,
        linewidth=3,
        markersize=8,
        alpha=0.8,
    )
    ax2.plot(
        x_pos,
        female_counts,
        marker="s",
        label="女性",
        color=female_color,
        linewidth=3,
        markersize=8,
        alpha=0.8,
    )

    # 添加误差条
    valid_male_ci = pd.notna(male_ci_lower) & pd.notna(male_ci_upper)
    if valid_male_ci.any():
        ax2.errorbar(
            x_pos[valid_male_ci],
            male_counts[valid_male_ci],
            yerr=[
                male_err_lower[valid_male_ci],
                male_err_upper[valid_male_ci],
            ],
            fmt="none",
            color=male_color,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
        )

    valid_female_ci = pd.notna(female_ci_lower) & pd.notna(female_ci_upper)
    if valid_female_ci.any():
        ax2.errorbar(
            x_pos[valid_female_ci],
            female_counts[valid_female_ci],
            yerr=[
                female_err_lower[valid_female_ci],
                female_err_upper[valid_female_ci],
            ],
            fmt="none",
            color=female_color,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
        )

    ax2.set_xlabel("地区", fontsize=12)
    ax2.set_ylabel("平均转发次数", fontsize=12)
    ax2.set_title("2. 平均转发次数（99% CI）", fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        plot_df["district"].values, rotation=0, ha="center", fontsize=10
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. 平均转发间隔（只显示有数据的地区）
    ax3 = axes[2]

    # 过滤出有间隔数据的地区
    interval_data = plot_df[
        plot_df["male_avg_interval"].notna() & plot_df["female_avg_interval"].notna()
    ].copy()

    if len(interval_data) > 0:
        x_pos_interval = np.arange(len(interval_data))

        male_intervals = interval_data["male_avg_interval"].values
        female_intervals = interval_data["female_avg_interval"].values

        # 转换为小时
        male_intervals_hours = male_intervals / 3600
        female_intervals_hours = female_intervals / 3600

        # 计算置信区间（转换为小时）
        male_interval_ci_lower = interval_data["male_interval_ci_lower"].values / 3600
        male_interval_ci_upper = interval_data["male_interval_ci_upper"].values / 3600
        female_interval_ci_lower = (
            interval_data["female_interval_ci_lower"].values / 3600
        )
        female_interval_ci_upper = (
            interval_data["female_interval_ci_upper"].values / 3600
        )

        male_color = get_gender_color("m")
        female_color = get_gender_color("f")

        # 计算误差范围（用于errorbar）
        male_err_lower = male_intervals_hours - male_interval_ci_lower
        male_err_upper = male_interval_ci_upper - male_intervals_hours
        female_err_lower = female_intervals_hours - female_interval_ci_lower
        female_err_upper = female_interval_ci_upper - female_intervals_hours

        # 绘制折线
        ax3.plot(
            x_pos_interval,
            male_intervals_hours,
            marker="o",
            label="男性",
            color=male_color,
            linewidth=3,
            markersize=8,
            alpha=0.8,
        )
        ax3.plot(
            x_pos_interval,
            female_intervals_hours,
            marker="s",
            label="女性",
            color=female_color,
            linewidth=3,
            markersize=8,
            alpha=0.8,
        )

        # 添加误差条
        valid_male_ci = pd.notna(interval_data["male_interval_ci_lower"]) & pd.notna(
            interval_data["male_interval_ci_upper"]
        )
        if valid_male_ci.any():
            ax3.errorbar(
                x_pos_interval[valid_male_ci],
                male_intervals_hours[valid_male_ci],
                yerr=[
                    male_err_lower[valid_male_ci],
                    male_err_upper[valid_male_ci],
                ],
                fmt="none",
                color=male_color,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.7,
            )

        valid_female_ci = pd.notna(
            interval_data["female_interval_ci_lower"]
        ) & pd.notna(interval_data["female_interval_ci_upper"])
        if valid_female_ci.any():
            ax3.errorbar(
                x_pos_interval[valid_female_ci],
                female_intervals_hours[valid_female_ci],
                yerr=[
                    female_err_lower[valid_female_ci],
                    female_err_upper[valid_female_ci],
                ],
                fmt="none",
                color=female_color,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                alpha=0.7,
            )

        ax3.set_xlabel("地区", fontsize=12)
        ax3.set_ylabel("平均转发间隔 (小时)", fontsize=12)
        ax3.set_title("3. 平均转发间隔（99% CI）", fontsize=14)
        ax3.set_xticks(x_pos_interval)
        ax3.set_xticklabels(
            interval_data["district"].values, rotation=0, ha="center", fontsize=10
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(
            0.5,
            0.5,
            "无转发间隔数据",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=14,
        )
        ax3.set_title("3. 平均转发间隔", fontsize=14)

    # 保存图表
    fig_path = os.path.join(OUTPUT_DIR, f"district_gender_gap_{source_type}_{year}.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"图表已保存到: {fig_path}")
    print(f"\n地区性别差异图表绘制完成\n")


def analyze_year(
    year: int,
    analysis_type: str = "all",
    visualize: bool = True,
    source_type: str = "news",
):
    """
    分析指定年份的数据

    Args:
        year: 年份
        analysis_type: 分析类型，'device', 'retweet', 'all'（默认：all）
        visualize: 是否在分析后自动绘制图表（默认：True）
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    print(f"开始分析 {year} 年数据，分析类型: {analysis_type}, 账号类型: {source_type}")

    if analysis_type in ["device", "all"]:
        analyze_device_changes(year)

    if analysis_type in ["retweet", "all"]:
        analyze_retweet_media(year, source_type=source_type)
        # 分析省份级别的性别差异
        analyze_retweet_media_by_province(year, source_type=source_type)
        # 分析地区级别的性别差异
        analyze_retweet_media_by_district(year, source_type=source_type)
        # 分析完成后自动画图
        if visualize:
            visualize_retweet_media(year, source_type=source_type)
            visualize_province_gender_gap(year, source_type=source_type)
            visualize_district_gender_gap(year, source_type=source_type)

    print(f"{year} 年分析完成")


def analyze_multiple_years(
    years: list,
    analysis_type: str = "all",
    visualize: bool = True,
    source_type: str = "news",
):
    """
    分析多个年份的数据

    Args:
        years: 年份列表，例如 [2020, 2021, 2022]
        analysis_type: 分析类型，'device', 'retweet', 'all'（默认：all）
        visualize: 是否在分析后自动绘制图表（默认：True）
        source_type: "news" 或 "entertain"，指定分析新闻账号还是娱乐账号（默认"news"）
    """
    for year in years:
        analyze_year(year, analysis_type, visualize, source_type)

    print(f"\n所有年份分析完成")


if __name__ == "__main__":
    fire.Fire(
        {
            "year": analyze_year,
            "years": analyze_multiple_years,
            "visualize": visualize_retweet_media,
            "province": analyze_retweet_media_by_province,
            "province_viz": visualize_province_gender_gap,
            "district": analyze_retweet_media_by_district,
            "district_viz": visualize_district_gender_gap,
        }
    )
