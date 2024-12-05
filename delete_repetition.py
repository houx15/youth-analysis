import pandas as pd
from deduplicate import deduplicate_by_weibo_id
import os
from datetime import datetime, timedelta

SOURCE_DIR = "youth_text_data"
OUTPUT_DIR = "youth_text_data_dedup"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 初始化日期
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# 生成日期范围
date_range = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]

for current_date in date_range:
    file_path = os.path.join(SOURCE_DIR, f"{current_date.strftime('%Y-%m-%d')}.parquet")
    
    if os.path.exists(file_path):
        # 读取 Parquet 文件
        df = pd.read_parquet(file_path)
        
        # 提取 line_binary 列并调用去重函数
        lines = df['line_binary'].tolist()
        unique_flags = deduplicate_by_weibo_id(lines)
        
        # 过滤掉重复的行
        df_unique = df[[flag == 1 for flag in unique_flags]]
        
        # 保存不重复的内容到新的 Parquet 文件
        output_path = os.path.join(OUTPUT_DIR, f"{current_date.strftime('%Y-%m-%d')}.parquet")
        df_unique.to_parquet(output_path)
