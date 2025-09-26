#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词频分析模块 - 将词频计算和可视化分开
支持按省份、区域、性别等维度进行词频分析
"""

import pandas as pd
import glob
import os
import json
import pickle
from collections import Counter, defaultdict
import jieba.posseg as pseg
from utils.utils import sentence_cleaner, STOP_WORDS
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
from datetime import datetime

other_devices = [
    "魅蓝",
    "Meizu",
    "OnePlus",
    "Smartisan",
    "Realme",
    "麦芒",
    "Lenovo",
    "Nubia",
    "ZTE",
    "360",
    "Gionee",
    "Nokia",
    "LeEco",
    "Hisense",
    "BlackShark",
    "Sony",
    "Coolpad",
    "国美",
    "Google",
    "Motorola",
    "ASUS",
    "BlackBerry",
    "Gree",
    "Sugar",
]

# Region mapping (从user_profile_analysis.py复制)
region_to_province = {
    "East": [
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
    "Central": ["山西", "安徽", "江西", "河南", "湖北", "湖南"],
    "West": [
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
    "Northeast": ["辽宁", "吉林", "黑龙江"],
}

province_to_region = {}
for region, provinces in region_to_province.items():
    for province in provinces:
        province_to_region[province] = region


def get_region_from_location(location):
    """Get region from location string"""
    if pd.isna(location) or location == "":
        return None
    # Split by space and get first element
    province = location.split()[0]
    return province_to_region.get(province)


def get_province_from_location(location):
    """Get province from location string"""
    if pd.isna(location) or location == "":
        return None
    # Split by space and get first element
    province = location.split()[0]
    return province


def create_counter():
    """创建一个新的Counter对象，用于defaultdict"""
    return Counter()


def create_defaultdict_counter():
    """创建一个新的defaultdict(Counter)对象，用于嵌套defaultdict"""
    return defaultdict(Counter)


def extract_words_from_content(content):
    """从文本内容中提取有意义的词汇"""
    if pd.isna(content) or content == "":
        return []

    cleaned_content = sentence_cleaner(content)
    if cleaned_content == "" or cleaned_content == "转发微博":
        return []
    words = pseg.cut(cleaned_content)
    meaningful_words = [
        word
        for word, flag in words
        if flag.startswith(("n", "v")) and len(word) > 1 and word not in STOP_WORDS
    ]
    return meaningful_words


def calculate_word_frequencies(year, month=None, save_path=None):
    """
    计算词频，支持按不同维度分组

    Args:
        year: 年份
        month: 月份（可选）
        save_path: 保存路径（可选）

    Returns:
        dict: 包含不同维度的词频统计
    """
    print(f"开始计算 {year} 年词频...")

    # 确定要处理的月份
    month_list = [month] if month else range(1, 13)

    # 初始化词频统计器
    word_freqs = {
        "total": Counter(),  # 总词频
        "by_gender": defaultdict(Counter),  # 按性别分组
        "by_region": defaultdict(Counter),  # 按区域分组
        "by_province": defaultdict(Counter),  # 按省份分组
        "by_gender_region": defaultdict(create_defaultdict_counter),  # 按性别+区域分组
        "by_device": defaultdict(Counter),  # 按设备分组
    }

    total_files = 0
    total_records = 0

    device_data = pd.read_parquet(
        f"merged_profiles/device_analysis_{year}.parquet",
        columns=["user_id", "frequent_brand"],
    )

    # 替换frequent brand中other_devices为Other
    device_data["frequent_brand"] = device_data["frequent_brand"].apply(
        lambda x: "Other" if x in other_devices else x
    )

    for month in month_list:
        month_str = f"{month:02d}"
        pattern = f"cleaned_youth_weibo/{year}/{year}-{month_str}-*.parquet"
        parquet_files = glob.glob(pattern)

        if not parquet_files:
            print(f"未找到 {year} 年 {month} 月的数据文件")
            continue

        print(f"处理 {year} 年 {month} 月，共 {len(parquet_files)} 个文件")

        for file_path in parquet_files:
            # 读取文件，只读取需要的列
            needed_columns = ["user_id", "weibo_content", "gender", "location"]
            df = pd.read_parquet(file_path, columns=needed_columns)

            df = df.merge(device_data, on="user_id", how="left")

            total_files += 1
            total_records += len(df)

            # 处理每一行数据
            for _, row in df.iterrows():
                # 提取词汇
                words = extract_words_from_content(row["weibo_content"])
                if not words:
                    continue

                # 获取用户属性
                gender = row.get("gender", "unknown")
                location = row.get("location", "")
                region = get_region_from_location(location)
                province = get_province_from_location(location)
                device = row.get("frequent_brand", "unknown")

                # 更新总词频
                word_freqs["total"].update(words)

                # 更新按性别分组的词频
                word_freqs["by_gender"][gender].update(words)

                # 更新按区域分组的词频
                if region:
                    word_freqs["by_region"][region].update(words)

                # 更新按省份分组的词频
                if province:
                    word_freqs["by_province"][province].update(words)

                # 更新按性别+区域分组的词频
                if region:
                    word_freqs["by_gender_region"][gender][region].update(words)

                if device:
                    word_freqs["by_device"][device].update(words)

    print(f"词频计算完成！")
    print(f"处理文件数: {total_files}")
    print(f"处理记录数: {total_records:,}")
    print(f"按设备分组数: {len(word_freqs['by_device'])}")
    print(f"总词汇数: {len(word_freqs['total'])}")

    # 保存词频数据
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(word_freqs, f)
        print(f"词频数据已保存到: {save_path}")

    return word_freqs


def load_word_frequencies(file_path):
    """加载保存的词频数据"""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def remove_stop_words_from_freq(word_freq):
    """从词频字典中移除停用词"""
    filtered_freq = Counter()
    for word, count in word_freq.items():
        if word not in STOP_WORDS:
            filtered_freq[word] = count
    return filtered_freq


def convert_to_ratio(word_freq):
    """将绝对频率转换为相对频率（ratio）"""
    if not word_freq:
        return Counter()

    total_count = sum(word_freq.values())
    if total_count == 0:
        return Counter()

    ratio_freq = Counter()
    for word, count in word_freq.items():
        ratio_freq[word] = count / total_count

    return ratio_freq


def get_top_words(word_freq, top_n=50):
    """获取词频最高的词汇"""
    return dict(word_freq.most_common(top_n))


def create_word_cloud(word_freq, title, save_path, width=800, height=400):
    """创建词云图"""
    if not word_freq:
        print(f"警告: {title} 没有词频数据")
        return

    wordcloud = WordCloud(
        font_path="/gpfs/share/home/2401111059/.fonts/simhei/SimHei.ttf",
        width=width,
        height=height,
        background_color="white",
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(save_path, bbox_inches="tight", dpi=300, format="pdf")
    plt.close()
    print(f"词云图已保存: {save_path}")


def create_word_frequency_plots(word_freqs, output_dir, year):
    """创建各种词频可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 处理词频数据：去除停用词并转换为ratio
    processed_freqs = {}

    # 处理总词频
    filtered_total = remove_stop_words_from_freq(word_freqs["total"])
    processed_freqs["total"] = convert_to_ratio(filtered_total)

    # 处理按性别分组的词频
    processed_freqs["by_gender"] = {}
    for gender, freq in word_freqs["by_gender"].items():
        if freq:
            filtered_freq = remove_stop_words_from_freq(freq)
            processed_freqs["by_gender"][gender] = convert_to_ratio(filtered_freq)

    processed_freqs["by_device"] = {}
    for device, freq in word_freqs["by_device"].items():
        if freq:
            filtered_freq = remove_stop_words_from_freq(freq)
            processed_freqs["by_device"][device] = convert_to_ratio(filtered_freq)

    # 处理按区域分组的词频
    processed_freqs["by_region"] = {}
    for region, freq in word_freqs["by_region"].items():
        if freq:
            filtered_freq = remove_stop_words_from_freq(freq)
            processed_freqs["by_region"][region] = convert_to_ratio(filtered_freq)

    # 处理按性别+区域分组的词频
    processed_freqs["by_gender_region"] = {}
    for gender, region_freqs in word_freqs["by_gender_region"].items():
        processed_freqs["by_gender_region"][gender] = {}
        for region, freq in region_freqs.items():
            if freq:
                filtered_freq = remove_stop_words_from_freq(freq)
                processed_freqs["by_gender_region"][gender][region] = convert_to_ratio(
                    filtered_freq
                )

    # 1. 总词频词云
    create_word_cloud(
        get_top_words(processed_freqs["total"], 50),
        f"Total Word Frequency Ratio - {year}",
        f"{output_dir}/total_wordcloud.pdf",
    )

    # 2. 按性别分组的词云
    for gender, freq in processed_freqs["by_gender"].items():
        if freq:
            create_word_cloud(
                get_top_words(freq, 50),
                f"Word Frequency Ratio by Gender ({gender}) - {year}",
                f"{output_dir}/gender_{gender}_wordcloud.pdf",
            )

    # 2. 按设备分组的词云
    for device, freq in processed_freqs["by_device"].items():
        if freq:
            create_word_cloud(
                get_top_words(freq, 50),
                f"Word Frequency Ratio by Device ({device}) - {year}",
                f"{output_dir}/device_{device}_wordcloud.pdf",
            )

    # 3. 按区域分组的词云
    for region, freq in processed_freqs["by_region"].items():
        if freq:
            create_word_cloud(
                get_top_words(freq, 50),
                f"Word Frequency Ratio by Region ({region}) - {year}",
                f"{output_dir}/region_{region}_wordcloud.pdf",
            )

    # 4. 性别词频对比柱状图
    create_gender_comparison_plot(processed_freqs["by_gender"], output_dir, year)

    # 5. 区域词频对比柱状图
    create_region_comparison_plot(processed_freqs["by_region"], output_dir, year)

    # 6. 性别+区域组合词频热力图
    create_gender_region_heatmap(processed_freqs["by_gender_region"], output_dir, year)


def create_gender_comparison_plot(gender_freqs, output_dir, year):
    """创建性别词频对比图"""
    # 选择一些常见词汇进行对比
    common_words = set()
    for freq in gender_freqs.values():
        common_words.update(list(freq.keys())[:20])

    if not common_words:
        return

    # 创建对比数据
    comparison_data = []
    for word in list(common_words)[:20]:  # 取前20个词
        for gender, freq in gender_freqs.items():
            comparison_data.append(
                {"word": word, "gender": gender, "ratio": freq.get(word, 0)}
            )

    df = pd.DataFrame(comparison_data)

    # 创建柱状图
    plt.figure(figsize=(15, 8))
    pivot_df = df.pivot(index="word", columns="gender", values="ratio")
    pivot_df.plot(kind="bar", figsize=(15, 8))
    plt.title(f"Word Frequency Ratio Comparison by Gender - {year}")
    plt.xlabel("Words")
    plt.ylabel("Frequency Ratio")
    plt.xticks(rotation=45)
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gender_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def create_device_comparison_plot(device_freqs, output_dir, year):
    """创建设备词频对比图"""
    # 选择一些常见词汇进行对比
    common_words = set()
    for freq in device_freqs.values():
        common_words.update(list(freq.keys())[:20])

    if not common_words:
        return

    # 创建对比数据
    comparison_data = []
    for word in list(common_words)[:20]:
        for device, freq in device_freqs.items():
            comparison_data.append(
                {"word": word, "device": device, "ratio": freq.get(word, 0)}
            )

    df = pd.DataFrame(comparison_data)

    # 创建柱状图
    plt.figure(figsize=(15, 8))
    pivot_df = df.pivot(index="word", columns="device", values="ratio")
    pivot_df.plot(kind="bar", figsize=(15, 8))
    plt.title(f"Word Frequency Ratio Comparison by Device - {year}")
    plt.xlabel("Words")
    plt.ylabel("Frequency Ratio")
    plt.xticks(rotation=45)
    plt.legend(title="Device")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/device_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def create_region_comparison_plot(region_freqs, output_dir, year):
    """创建区域词频对比图"""
    # 选择一些常见词汇进行对比
    common_words = set()
    for freq in region_freqs.values():
        common_words.update(list(freq.keys())[:20])

    if not common_words:
        return

    # 创建对比数据
    comparison_data = []
    for word in list(common_words)[:20]:  # 取前20个词
        for region, freq in region_freqs.items():
            comparison_data.append(
                {"word": word, "region": region, "ratio": freq.get(word, 0)}
            )

    df = pd.DataFrame(comparison_data)

    # 创建柱状图
    plt.figure(figsize=(15, 8))
    pivot_df = df.pivot(index="word", columns="region", values="ratio")
    pivot_df.plot(kind="bar", figsize=(15, 8))
    plt.title(f"Word Frequency Ratio Comparison by Region - {year}")
    plt.xlabel("Words")
    plt.ylabel("Frequency Ratio")
    plt.xticks(rotation=45)
    plt.legend(title="Region")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/region_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def create_gender_region_heatmap(gender_region_freqs, output_dir, year):
    """创建性别+区域词频热力图"""
    # 选择一些常见词汇
    common_words = set()
    for gender_freqs in gender_region_freqs.values():
        for freq in gender_freqs.values():
            common_words.update(list(freq.keys())[:10])

    if not common_words:
        return

    # 创建热力图数据
    heatmap_data = []
    for word in list(common_words)[:10]:
        for gender, region_freqs in gender_region_freqs.items():
            for region, freq in region_freqs.items():
                heatmap_data.append(
                    {
                        "word": word,
                        "gender": gender,
                        "region": region,
                        "ratio": freq.get(word, 0),
                    }
                )

    df = pd.DataFrame(heatmap_data)

    # 创建热力图
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot_table(
        index="word", columns=["gender", "region"], values="ratio", aggfunc="sum"
    )

    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlOrRd")
    plt.title(f"Word Frequency Ratio Heatmap by Gender and Region - {year}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gender_region_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def analyze_word_frequencies(year, month=None, recalculate=False):
    """
    完整的词频分析流程

    Args:
        year: 年份
        month: 月份（可选）
        recalculate: 是否重新计算词频
    """
    # 设置文件路径
    word_freq_file = f"word_frequencies/word_freq_{year}.pkl"
    if month is not None:
        word_freq_file = f"word_frequencies/word_freq_{year}_{month:02d}.pkl"
    output_dir = f"figures/{year}/word_frequency"
    if month is not None:
        output_dir = f"figures/{year}/word_frequency/{month:02d}"

    # 计算或加载词频数据
    if recalculate or not os.path.exists(word_freq_file):
        word_freqs = calculate_word_frequencies(year, month, word_freq_file)
    else:
        print(f"加载已保存的词频数据: {word_freq_file}")
        word_freqs = load_word_frequencies(word_freq_file)

    # 创建可视化图表
    create_word_frequency_plots(word_freqs, output_dir, year)

    # 输出统计信息
    print(f"\n词频分析统计信息:")
    print(f"总词汇数: {len(word_freqs['total'])}")
    print(f"性别分组数: {len(word_freqs['by_gender'])}")
    print(f"区域分组数: {len(word_freqs['by_region'])}")
    print(f"省份分组数: {len(word_freqs['by_province'])}")

    return word_freqs


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "calculate": calculate_word_frequencies,
            "analyze": analyze_word_frequencies,
            "visualize": create_word_frequency_plots,
        }
    )
