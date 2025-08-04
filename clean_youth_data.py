#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗微博数据，只保留10-16岁用户的数据
"""

import pandas as pd
import glob
import os
import json
import fire


def get_youth_user_ids():
    """加载人口统计数据"""
    print("加载人口统计数据...")
    df = pd.read_parquet("merged_profiles/merged_user_profiles.parquet")

    # Create figures directory
    os.makedirs("figures", exist_ok=True)

    # 1. Calculate age and filter
    df["birthday"] = pd.to_datetime(df["birthday"])
    df["age"] = 2020 - df["birthday"].dt.year

    # Count before filtering
    total_users = len(df)

    # Filter age range
    youth_data = df[(df["age"] >= 10) & (df["age"] <= 16)]

    # 转换为集合，便于快速查找
    youth_user_ids = set(youth_data["user_id"].astype(int))

    return youth_user_ids


def clean_weibo_data(year, month=None):
    """
    清洗指定年份的微博数据，只保留10-16岁用户的数据

    Args:
        year: 年份
        month: 月份（可选），如果不指定则处理整年
    """
    print(f"开始清洗 {year} 年微博数据...")

    # 加载符合标准的用户ID
    youth_user_ids = get_youth_user_ids()

    # 确定要处理的月份
    if month is not None:
        months = [month]
    else:
        months = range(1, 13)

    # 创建输出目录
    output_dir = f"cleaned_youth_weibo/{year}"
    os.makedirs(output_dir, exist_ok=True)

    total_files_processed = 0
    total_records_before = 0
    total_records_after = 0

    for month in months:
        month_str = f"{month:02d}"
        print(f"\n处理 {year} 年 {month} 月数据...")

        # 查找该月的所有文件
        pattern = f"youth_weibo_stat/{year}-{month_str}-*.parquet"
        parquet_files = glob.glob(pattern)

        if not parquet_files:
            print(f"未找到 {year} 年 {month} 月的数据文件")
            continue

        print(f"找到 {len(parquet_files)} 个文件")

        for file_path in parquet_files:
            # 读取文件
            df = pd.read_parquet(file_path)
            total_records_before += len(df)

            # 确保user_id是字符串类型
            df["user_id"] = df["user_id"].astype(int)
            # drop na user ids
            df = df.dropna(subset=["user_id"])

            # 过滤只保留符合标准的用户
            filtered_df = df[df["user_id"].isin(youth_user_ids)]
            total_records_after += len(filtered_df)

            # 如果过滤后还有数据，保存到新文件
            if len(filtered_df) > 0:
                # 生成输出文件名
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, filename)

                # 保存过滤后的数据
                filtered_df.to_parquet(output_path, index=False)
                print(f"  {filename}: {len(df)} -> {len(filtered_df)} 条记录")

            total_files_processed += 1

    print(f"\n清洗完成！")
    print(f"处理文件数: {total_files_processed}")
    print(f"原始记录数: {total_records_before:,}")
    print(f"过滤后记录数: {total_records_after:,}")
    print(f"过滤比例: {(1 - total_records_after/total_records_before)*100:.2f}%")
    print(f"输出目录: {output_dir}")


def remove_shuijun(year, month=None):
    """
    删除水军
    configs/too_many_weibo_userids.json list里面单个元素是int，是水军user id
    需要遍历clean weibo data处理过了的parquet（每一个），删除这些user id发布的内容
    """
    with open("configs/too_many_weibo_userids.json", "r") as f:
        shuijun_user_ids = json.load(f)

    if month is not None:
        months = [month]
    else:
        months = range(1, 13)

    for month in months:
        month_str = f"{month:02d}"
        pattern = f"cleaned_youth_weibo/{year}/{year}-{month_str}-*.parquet"
        parquet_files = glob.glob(pattern)
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            before = len(df)
            df["user_id"] = df["user_id"].astype(int)
            df = df[~df["user_id"].isin(shuijun_user_ids)]
            after = len(df)
            df.to_parquet(file_path)
            print(f"处理 {file_path} 完成，删除前 {before} 条，删除后 {after} 条")
        print(f"处理 {year} 年 {month} 月数据完成")


def merge_demo_to_content(year, month=None):
    """
        合并demo数据到content数据

        demo 数据：merged_profiles/merged_user_profiles.parquet
        >>> data.columns
    Index(['date', 'user_id', 'nick_name', 'user_type', 'gender', 'verified_type',
           'verified_reason', 'description', 'fans_number', 'weibo_number', 'type',
           'friends_count', 'favourites_count', 'created_at', 'allow_all_comment',
           'bi_followers_count', 'location', 'province', 'city', 'ip_location',
           'birthday', 'demographic_gender', 'demographic_province', 'region'],
          dtype='object')
        youth_data = age 10-16
        合并 user_type, gender, location, province, city, ip_location, birthday, demographic_gender, demographic_province, region

        content 数据：cleaned_youth_weibo/{year}/{year}-{month_str}-*.parquet
    """
    demo_data = pd.read_parquet("merged_profiles/merged_user_profiles.parquet")
    demo_data["birthday"] = pd.to_datetime(demo_data["birthday"])
    demo_data["age"] = 2020 - demo_data["birthday"].dt.year
    demo_data = demo_data[demo_data["age"] >= 10]
    demo_data = demo_data[demo_data["age"] <= 16]
    demo_data = demo_data[
        [
            "user_id",
            "user_type",
            "gender",
            "location",
            "province",
            "city",
            "ip_location",
            "birthday",
            "demographic_gender",
            "demographic_province",
            "region",
        ]
    ]
    demo_data = demo_data.drop_duplicates(subset=["user_id"])
    demo_data = demo_data.dropna(subset=["user_id"])

    if month is not None:
        months = [month]
    else:
        months = range(1, 13)

    for month in months:
        month_str = f"{month:02d}"
        pattern = f"cleaned_youth_weibo/{year}/{year}-{month_str}-*.parquet"
        parquet_files = glob.glob(pattern)
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            df = df.merge(demo_data, on="user_id", how="left")
            df.to_parquet(file_path)
            print(f"处理 {file_path} 完成")
        print(f"处理 {year} 年 {month} 月数据完成")


if __name__ == "__main__":
    fire.Fire(
        {
            "clean": clean_weibo_data,
            "remove": remove_shuijun,
            "merge": merge_demo_to_content,
        }
    )
