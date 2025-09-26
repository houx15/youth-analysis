from configs.configs import *
import pandas as pd
import glob
from collections import Counter
import os
import json
import time
import fire
import re
import matplotlib.pyplot as plt
import seaborn as sns


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def log(text):
    """Write log to device_analysis.log"""
    with open("device_analysis.log", "a") as f:
        f.write(f"{text}\n")


DEVICE_BRAND_MAP = {
    "Apple": ["IPHONE", "IPAD", "MAC", "IPOD"],
    "Huawei": ["HUAWEI", "华为"],
    "NOVA": ["NOVA"],
    "麦芒": ["麦芒"],
    "Honor": ["荣耀", "HONOR"],
    "Xiaomi": ["小米", "XIAOMI", "MI PHONE"],
    "Redmi": ["REDMI", "红米", "K30"],
    "BlackShark": ["黑鲨"],
    "Vivo": ["VIVO", "IQOO", "NEX", "X20PLUS"],
    "Oppo": ["OPPO", "RENO", "FIND"],
    "Realme": ["REALME", "真我"],
    "Samsung": ["三星", "SAMSUNG", "NOTE"],
    "OnePlus": ["ONEPLUS"],
    "Meizu": ["魅族", "MEIZU"],
    "魅蓝": ["魅蓝"],
    "Lenovo": ["联想", "ZUK"],
    "Nubia": ["努比亚", "NUBIA", "红魔"],
    "Smartisan": ["坚果", "SMARTISAN"],
    "360": ["360手机"],
    "Hisense": ["海信"],
    "Nokia": ["NOKIA", "诺基亚"],
    "ZTE": ["中兴智能", "ZTE", "中兴", "AXON"],
    "Sony": ["索尼", "XPERIA"],
    "Doov": ["DOOV", "朵唯"],
    "Gionee": ["金立", "GIONEE"],
    "Gree": ["格力"],
    "Coolpad": ["酷派", "COOL"],
    "LeEco": ["乐PRO", "乐MAX", "乐2", "乐S3", "乐视"],
    "BlackBerry": ["BLACKBERRY"],
    "Motorola": ["MOTOROLA", "摩托罗拉"],
    "HTC": ["HTC"],
    "AGM": ["AGM"],
    "ASUS": ["ROG"],
    "Sugar": ["SUGAR", "糖果翻译手机"],
    "Google": ["GOOGLE"],
    "国美": ["国美"],
}

device_to_brand = {}
for brand, devices in DEVICE_BRAND_MAP.items():
    for device in devices:
        device_to_brand[device] = brand

INTERESTED_DEVICE_TYPES = [
    "IPHONE",
    "HUAWEI",
    "华为",
    "VIVO",
    "OPPO",
    "NOVA",
    "REDMI",
    "红米",
    "IPAD",
    "荣耀",
    "HONOR",
    "小米",
    "XIAOMI",
    "MI PHONE",
    "三星",
    "SAMSUNG",
    "ONEPLUS",
    "魅族",
    "MEIZU",
    "联想",
    "麦芒",
    "REALME",
    "真我",
    "努比亚",
    "NUBIA",
    "坚果",
    "SMARTISAN",
    "IQOO",
    "360手机",
    "K30",
    "魅蓝",
    "RENO",
    "海信",
    "MAC",
    "NOKIA",
    "诺基亚",
    "中兴智能",
    "ZTE",
    "索尼",
    "XPERIA",
    "NEX",
    "DOOV",
    "朵唯",
    "金立",
    "GIONEE",
    "格力",
    "黑鲨",
    "小灵通",
    "X20PLUS",
    "NOTE",
    "GOOGLE",
    "酷派",
    "乐PRO",
    "乐MAX",
    "乐2",
    "乐S3",
    "乐视",
    "中兴",
    "BLACKBERRY",
    "MOTOROLA",
    "摩托罗拉",
    "ZUK",
    "HTC",
    "AGM",
    "IPOD",
    "AXON",
    "FIND",
    "COOL",
    "ROG",
    "红魔",
    "SUGAR",
    "糖果翻译手机",
    "X5MAX",
    "国美",
]

pad_signal = ["平板", "PAD"]
computer_signal = ["MAC", "网页"]


def device_handler(device_string):
    """
    处理device数据
    """
    device = None
    # 转为大写
    device_string = device_string.upper()
    for brand in INTERESTED_DEVICE_TYPES:
        if brand in device_string:
            device = brand
            break
    if device is None:
        return None

    brand = device_to_brand[device]

    # 获取字符串中的英文+数字作为型号 中间可能有空格隔开
    model_match = re.search(r"[A-Za-z0-9 ]+", device_string)
    if model_match:
        # 清理匹配结果，我要的是匹配中的最后一项
        model = model_match.group().split(" ")[-1]
    else:
        model = None
    return {
        "device": device,
        "model": model,
        "brand": brand,
    }


def analyze_device_basic(year, mode="youth"):
    """Basic analysis of tweet data"""
    log(f"analyzing tweet basic for {year}")
    # Load tweet data
    os.makedirs(f"figures/{year}", exist_ok=True)
    if mode == "youth":
        parquet_files = glob.glob(f"cleaned_youth_weibo/{year}/*.parquet")
    elif mode == "all":
        parquet_files = glob.glob(f"youth_weibo_stat/{year}-*.parquet")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not parquet_files:
        log(f"No parquet files found for year {year}")
        return

    # Only read necessary columns
    needed_columns = ["device", "user_id", "time_stamp"]
    df = pd.concat([pd.read_parquet(f, columns=needed_columns) for f in parquet_files])

    # 2. Device analysis
    # device_counts = df["device"].value_counts()
    # with open(f"figures/{year}/device_distribution.txt", "w", encoding="utf-8") as f:
    #     f.write("Device Distribution:\n")
    #     for device, count in device_counts.items():
    #         f.write(f"{device}: {count}\n")

    weibo_counts = df.groupby("user_id").size().rename("weibo_count")

    # 3. 统计每个用户最常用device
    frequent_device = (
        df.groupby(["user_id", "device"])
        .size()
        .reset_index(name="count")
        .sort_values(["user_id", "count"], ascending=[True, False])
        .drop_duplicates("user_id")
        .set_index("user_id")
    )

    # 4. 统计每个用户最近一次device
    df["time_stamp"] = pd.to_numeric(df["time_stamp"])
    df["beijing_time"] = pd.to_datetime(df["time_stamp"], unit="s") + pd.Timedelta(
        hours=8
    )
    df = df.sort_values(by="beijing_time", ascending=False)
    most_recent = df.groupby("user_id").head(1).set_index("user_id")

    # 5. 合并结果
    result = pd.DataFrame(index=frequent_device.index)
    result["frequent_device_original"] = frequent_device["device"]
    result["most_recent_device_original"] = most_recent["device"]

    # 6. 用device_handler处理
    def safe_device_handler(x):
        try:
            return device_handler(str(x))
        except Exception:
            return None

    freq_info = result["frequent_device_original"].apply(safe_device_handler)
    recent_info = result["most_recent_device_original"].apply(safe_device_handler)

    result["frequent_brand"] = freq_info.apply(lambda x: x["brand"] if x else None)
    result["frequent_device"] = freq_info.apply(lambda x: x["device"] if x else None)
    result["frequent_model"] = freq_info.apply(lambda x: x["model"] if x else None)

    result["most_recent_brand"] = recent_info.apply(lambda x: x["brand"] if x else None)
    result["most_recent_device"] = recent_info.apply(
        lambda x: x["device"] if x else None
    )
    result["most_recent_model"] = recent_info.apply(lambda x: x["model"] if x else None)

    # 7. 合并微博数量
    result = result.merge(weibo_counts, left_index=True, right_index=True, how="left")

    # 8. 重置索引，调整列顺序
    result.reset_index(inplace=True)
    cols = [
        "user_id",
        "weibo_count",
        "frequent_device_original",
        "frequent_brand",
        "frequent_device",
        "frequent_model",
        "most_recent_device_original",
        "most_recent_brand",
        "most_recent_device",
        "most_recent_model",
    ]
    result = result[cols]

    result.to_parquet(f"merged_profiles/device_analysis_{year}_{mode}.parquet")

    # 绘制品牌分布图
    brand_counts = result["frequent_brand"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=brand_counts.index, y=brand_counts.values)
    plt.title(f"{year} 用户最常用品牌分布 ({mode})")
    plt.xlabel("品牌")
    plt.ylabel("用户数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{year}/brand_distribution_{mode}.png")
    plt.close()

    # 统计未匹配人数
    unmatched_users = result[result["frequent_brand"].isna()].shape[0]
    log(f"Year {year}: {unmatched_users} users had no matching brand.")


def plot_brand_distribution(year, mode="youth"):
    """绘制品牌分布图并统计未匹配用户"""
    log(f"绘制 {year} 年品牌分布图")

    # 读取设备分析结果
    device_file = f"merged_profiles/device_analysis_{year}_{mode}.parquet"
    if not os.path.exists(device_file):
        log(f"设备分析文件不存在: {device_file}")
        return

    result = pd.read_parquet(device_file)

    # 统计品牌分布
    brand_counts = result["frequent_brand"].value_counts()

    # 统计未匹配用户
    unmatched_users = result[result["frequent_brand"].isna()].shape[0]
    total_users = len(result)
    matched_users = total_users - unmatched_users

    log(f"{year} 年统计结果:")
    log(f"  总用户数: {total_users}")
    log(f"  成功匹配品牌用户数: {matched_users}")
    log(f"  未匹配品牌用户数: {unmatched_users}")
    log(f"  匹配率: {matched_users/total_users*100:.2f}%")

    # 绘制品牌分布图
    plt.figure(figsize=(15, 8))

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左侧：品牌分布柱状图
    top_brands = brand_counts.head(15)  # 显示前15个品牌
    bars = ax1.bar(range(len(top_brands)), top_brands.values, color="skyblue")
    ax1.set_title(
        f"{year} 年用户最常用品牌分布 (前15名) ({mode})", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("品牌", fontsize=12)
    ax1.set_ylabel("用户数量", fontsize=12)
    ax1.set_xticks(range(len(top_brands)))
    ax1.set_xticklabels(top_brands.index, rotation=45, ha="right")

    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 右侧：匹配情况饼图
    labels = ["成功匹配", "未匹配"]
    sizes = [matched_users, unmatched_users]
    colors = ["lightgreen", "lightcoral"]

    ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title(f"{year} 年品牌匹配情况 ({mode})", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        f"figures/{year}/brand_analysis_{mode}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 保存详细统计结果到文本文件
    with open(
        f"figures/{year}/brand_statistics_{mode}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(f"{year} 年品牌分析统计结果 ({mode})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总用户数: {total_users:,}\n")
        f.write(f"成功匹配品牌用户数: {matched_users:,}\n")
        f.write(f"未匹配品牌用户数: {unmatched_users:,}\n")
        f.write(f"匹配率: {matched_users/total_users*100:.2f}%\n\n")

        f.write("品牌分布详情:\n")
        f.write("-" * 30 + "\n")
        for brand, count in brand_counts.items():
            percentage = count / total_users * 100
            f.write(f"{brand}: {count:,} 用户 ({percentage:.2f}%)\n")

        f.write(f"\n未匹配用户的设备信息 (前20个):\n")
        f.write("-" * 40 + "\n")
        unmatched_devices = (
            result[result["frequent_brand"].isna()]["frequent_device_original"]
            .value_counts()
            .head(20)
        )
        for device, count in unmatched_devices.items():
            f.write(f"{device}: {count} 用户\n")

    log(f"品牌分析图表已保存到 figures/{year}/brand_analysis_{mode}.png")
    log(f"详细统计结果已保存到 figures/{year}/brand_statistics_{mode}.txt")


def check(year, ratio=0.001):
    """
    检查发言数异常多的用户，抽样发言内容
    """
    stats = pd.read_parquet(f"merged_profiles/device_analysis_{year}.parquet")
    stats = stats.sort_values(by="weibo_count", ascending=False)
    stats = stats[stats["weibo_count"] > 10000]
    # 取前ratio
    # stats = stats.head(int(len(stats) * ratio))
    log(f"抽样用户数: {len(stats)}")

    # 打印前几个用户的信息
    for idx, row in stats.head(10).iterrows():
        log(f"user_id: {row['user_id']}, weibo_count: {row['weibo_count']}")

    # 挑最高的5个，看看都在发什么
    top_5_userids = []
    for idx, row in stats.iterrows():
        top_5_userids.append(int(row["user_id"]))
    with open(f"configs/too_many_weibo_userids.json", "w") as f:
        json.dump(top_5_userids, f)

    raise Exception("stop here")

    parquet_files = glob.glob(f"cleaned_youth_weibo/{year}/*.parquet")
    if not parquet_files:
        log(f"No parquet files found for year {year}")
        return

    # Only read necessary columns
    needed_columns = ["weibo_content", "user_id", "time_stamp"]
    printed_user = set()
    for f in parquet_files:
        data = pd.read_parquet(f, columns=needed_columns)
        print(data["user_id"].dtype)
        data = data[data["user_id"].isin(top_5_userids)]
        for user_id in top_5_userids:
            if user_id in printed_user:
                continue
            user_data = data[data["user_id"] == user_id]
            if user_data.shape[0] > 20:
                user_data = user_data.sample(20)
                printed_user.add(user_id)
                for i in range(user_data.shape[0]):
                    log(
                        f"file: {f}, user_id: {user_id}, weibo_content: {user_data.iloc[i]['weibo_content']}"
                    )
        if len(printed_user) >= 5:
            break


def process_device_analysis(year, mode="youth"):
    analyze_device_basic(year, mode)
    plot_brand_distribution(year, mode)


if __name__ == "__main__":
    fire.Fire(
        {
            "analyze": analyze_device_basic,
            "plot": plot_brand_distribution,
            "process": process_device_analysis,
            "check": check,
        }
    )
