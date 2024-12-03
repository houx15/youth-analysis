import os
import time
import json

import pandas as pd

from extract_user_id import extract_user_ids
from extract_user_id_old import extract_user_ids_old
from collections import defaultdict
from datetime import datetime, timedelta

from utils.utils import extract_single_7z_file
from configs.configs import ORIGIN_DATA_DIR

def log(text, lid=None):
    output = f"logs/log_{lid}.txt" if lid is not None else "logs/log.txt"
    with open(output, "a") as f:
        f.write(f"{text}\n")

def get_zipped_fresh_data_file(year, date):
    """
    date should be yyyy-mm-dd format
    """
    return f"{ORIGIN_DATA_DIR}/{year}/freshdata/weibo_freshdata.{date}.7z"


def get_unzipped_fresh_data_folder(year):
    return f"text_working_data/{year}/"


def get_unzipped_fresh_data_file(year, date):
    return f"text_working_data/{year}/weibo_freshdata.{date}"


def delete_unzipped_fresh_data_file(year, date):
    """
    处理完毕之后需要删除文件
    """
    file_path = get_unzipped_fresh_data_file(year, date)
    # 删除文件
    try:
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除。")
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
    except Exception as e:
        print(f"删除文件时发生错误: {e}")

def unzip_one_fresh_data_file(year, date):
    """
    date should be yyyy-mm-dd format
    """
    unzipped_file_path = get_unzipped_fresh_data_file(year, date)
    
    # 检查文件是否已经解压
    if os.path.exists(unzipped_file_path):
        print(f"文件 {unzipped_file_path} 已经存在，跳过解压。")
        return unzipped_file_path

    # 如果文件不存在，则进行解压
    zipped_file_path = get_zipped_fresh_data_file(year, date)
    unzipped_dir = get_unzipped_fresh_data_folder(year)
    result = extract_single_7z_file(file_path=zipped_file_path, target_folder=unzipped_dir)
    
    if result == "success":
        return unzipped_file_path
    return None


def process_file(file_path, matched_ids, year):
    """
    从大文件中逐块提取 user_id 和 JSON 数据，直接存储 bytes 数据到 Parquet 文件。
    :param file_path: 输入文件路径
    :param matched_ids: 目标 user_id 集合（bytes 类型）
    :param output_parquet_path: 输出 Parquet 文件路径
    """
    # 定义每块的大小（行数）
    extract_function = extract_user_ids if year > 2019 else extract_user_ids_old
    chunk_size = 5000000
    all_userids = []
    all_results = []

    # 如果file_path不存在，直接返回空列表
    if not os.path.exists(file_path):
        return all_userids, all_results
    # 逐行读取文件
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            # 当块大小达到限制时，处理该块
            if len(chunk) == chunk_size:
                # 调用 Cython 函数
                userids, results = extract_function(chunk, matched_ids)
                # 直接存储 bytes 数据
                all_userids.extend(userids)
                all_results.extend(results)
                chunk = []  # 清空块

        # 处理最后一块（如果有剩余）
        if chunk:
            userids, results = extract_function(chunk, matched_ids)
            all_userids.extend(userids)
            all_results.extend(results)

    # 将结果整理为 DataFrame，直接存储 bytes 数据
    df = pd.DataFrame({
        "user_id_binary": all_userids,  # bytes 类型的 user_id
        "line_binary": all_results      # bytes 类型的 JSON 数据
    })

    return all_userids, all_results

def append_to_parquet(date, userids, results):
    """
    将数据追加到指定年份的 Parquet 文件中。
    :param year: 年份（如 2024）
    :param userids: 用户 ID 列表
    :param results: 结果数据列表
    """
    output_parquet_path = f"youth_text_data/{date}.parquet"
    df = pd.DataFrame({
        "user_id_binary": userids,  # bytes 类型的 user_id
        "line_binary": results      # bytes 类型的 JSON 数据
    })

    # # 检查文件是否存在
    # if not os.path.exists(output_parquet_path):
    #     # 如果文件不存在，直接写入（新建文件）
    df.to_parquet(output_parquet_path, engine="fastparquet", index=False)
    # else:
    #     # 如果文件存在，以追加模式写入
    #     df.to_parquet(output_parquet_path, engine="fastparquet", index=False, append=True)
    #     log(f"追加数据到文件：{output_parquet_path}")

def process_year(year, mode):
    with open(f"data/youth_user_ids_{year}.json", "r") as f:
        year_match_ids = json.load(f)
        all_matched_ids = {user_id.encode('utf-8') for user_id in year_match_ids}

    start_date_options = [datetime(year, 1, 1), datetime(year, 7, 1)]
    end_date_options = [datetime(year, 6, 30), datetime(year, 12, 31)]
    start_date = start_date_options[mode]
    end_date = end_date_options[mode]

    current_date = start_date

    date_range = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]
    # date_range = ["test"]
    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")

        file_path = unzip_one_fresh_data_file(year, date_str)
        # file_path = f"text_working_data/{year}/weibo_freshdata.test"
        if file_path is None:
            continue
        start_timestamp = int(time.time())
        userids, results = process_file(file_path, all_matched_ids, year)
        append_to_parquet(date_str, userids, results)

        log(f"处理 {date_str} 完成，耗时 {int(time.time()) - start_timestamp} 秒。", f"{year}_{mode}")

        delete_unzipped_fresh_data_file(year, date_str)


if __name__ == "__main__":
    # for y in [2016, 2017, 2018, 2019]:
    process_year(2019, 1)
    log("处理完毕。")
