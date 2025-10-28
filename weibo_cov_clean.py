#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗微博数据
"""

import pandas as pd
import glob
import os
import json
import fire


def clean_weibo_data(year, month=None):
    """
    清洗指定年份的微博数据

    Args:
        year: 年份
        month: 月份（可选），如果不指定则处理整年
    """
    print(f"开始清洗 {year} 年微博数据...")

    # 确定要处理的月份
    if month is not None:
        months = [month]
    else:
        months = range(1, 13)

    # 创建输出目
    output_dir = f"cleaned_weibo_cov/{year}"
    os.makedirs(output_dir, exist_ok=True)

    total_files_processed = 0

    demo_data = pd.read_parquet("merged_profiles/merged_user_profiles.parquet")
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
    demo_data["user_id"] = demo_data["user_id"].astype(int)

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

            # 确保user_id是字符串类型
            df["user_id"] = df["user_id"].astype(int)
            # drop na user ids
            df = df.dropna(subset=["user_id"])

            # 生成输出文件名
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            # 保存过滤后的数据
            df = df.merge(demo_data, on="user_id", how="left")
            df.to_parquet(output_path, index=False)

            total_files_processed += 1

    print(f"\n清洗完成！")
    print(f"处理文件数: {total_files_processed}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    fire.Fire(clean_weibo_data)
