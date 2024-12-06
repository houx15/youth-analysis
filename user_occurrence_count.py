import os
import pandas as pd
from datetime import datetime, timedelta
from user_id_counter import count_user_ids
from collections import defaultdict

OUTPUT_DIR = "user_count"
SOURCE_DIR = "youth_text_data_dedup"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 初始化日期
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# 存储每年 user_id 出现次数
yearly_counts = defaultdict(int)

# 遍历每一天
current_date = start_date
date_range = [
    start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)
]

for current_date in date_range:
    year_output_dir = os.path.join(OUTPUT_DIR, f"count-{str(current_date.year)}")
    if not os.path.exists(year_output_dir):
        os.makedirs(year_output_dir)
    file_path = os.path.join(SOURCE_DIR, f"{current_date.strftime('%Y-%m-%d')}.parquet")

    if os.path.exists(file_path):
        # 统计单个文件中各个 user_id 的出现次数
        daily_counts = count_user_ids(file_path)
        if daily_counts is None:
            continue

        # 将每日统计结果存储为 Parquet 文件
        daily_df = pd.DataFrame(
            [(current_date, user_id, count) for user_id, count in daily_counts.items()],
            columns=["date", "user_id", "count"],
        )
        daily_output_path = os.path.join(
            year_output_dir, f"{current_date.strftime('%Y-%m-%d')}.parquet"
        )
        daily_df.to_parquet(daily_output_path, engine="fastparquet", index=False)

        # 累加到 yearly_counts 中
        for user_id, count in daily_counts.items():
            yearly_counts[(current_date.year, user_id)] += count

# 转换 yearly_counts 为 DataFrame 并存储为 Parquet
yearly_df = pd.DataFrame(
    [(year, user_id, count) for (year, user_id), count in yearly_counts.items()],
    columns=["year", "user_id", "count"],
)

yearly_output_path = os.path.join(OUTPUT_DIR, "yearly_user_counts_2020.parquet")
yearly_df.to_parquet(yearly_output_path, engine="fastparquet", index=False)
