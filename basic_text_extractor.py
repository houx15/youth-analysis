"""
清洗混乱的底层微博数据
- 只去重（重复的weibo_id）
- 不筛选关键词
- 不筛选user_id
- 将各年的微博数据整理成更方便使用的parquet格式
- 按天存储，存储到~/cleaned_weibo_data
"""

import os
import json
import time
import pandas as pd
import orjson
from datetime import datetime, timedelta
from collections import defaultdict

from utils.utils import extract_single_7z_file
from configs.configs import ORIGIN_DATA_DIR

OUTPUT_DIR = os.path.expanduser("~/cleaned_weibo_data")

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def log(text, lid=None):
    """记录日志"""
    os.makedirs("logs", exist_ok=True)
    output = (
        f"logs/clean_weibo_log_{lid}.txt"
        if lid is not None
        else "logs/clean_weibo_log.txt"
    )
    with open(output, "a") as f:
        f.write(f"{text}\n")


def get_zipped_fresh_data_file(year, date):
    """
    date should be yyyy-mm-dd format
    """
    return f"{ORIGIN_DATA_DIR}/{year}/freshdata/weibo_freshdata.{date}.7z"


def get_unzipped_fresh_data_folder(year):
    return f"text_working_data/{year}/testing"


def get_unzipped_fresh_data_file(year, date):
    if date == "2020-06-30":
        return f"text_working_data/{year}/testing/weibo_2020-06-30.csv"
    elif date in ["2017-01-11", "2016-07-24", "2016-08-09"]:
        return f"text_working_data/{year}/testing/weibo_log/weibo_freshdata.{date}.csv"
    return f"text_working_data/{year}/testing/weibo_freshdata.{date}"


def delete_unzipped_fresh_data_file(year, date):
    """
    处理完毕之后需要删除文件
    """
    file_path = get_unzipped_fresh_data_file(year, date)
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
    result = extract_single_7z_file(
        file_path=zipped_file_path, target_folder=unzipped_dir
    )

    if os.path.exists(unzipped_file_path):
        print(f"文件 {unzipped_file_path} 解压成功。")
        return unzipped_file_path
    else:
        print(f"文件 {unzipped_file_path} 解压失败。")
        return None


def print_first_3_lines(file_path):
    """
    打印文件的前3行，用于检查2023年及之后的数据格式
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for i, line in enumerate(file):
                if i >= 3:
                    break
                print(f"Line {i+1}: {line[:200]}")  # 只打印前200个字符
                print("-" * 50)
    except Exception as e:
        print(f"读取文件时出错: {e}")


def process_special_format_2020_06_30(chunk, seen_weibo_ids):
    """
    处理2020-06-30的特殊CSV格式
    seen_weibo_ids: set，用于实时去重
    """
    results = []
    for line in chunk:
        try:
            # CSV格式："46890032291","2020-06-30 00:12:36",...
            line_data = line.strip().strip('"').split('","')
            if len(line_data) < 24:
                continue

            # 先提取weibo_id进行去重检查
            weibo_id = line_data[8].strip('"')
            if not weibo_id or weibo_id in seen_weibo_ids:
                continue  # 跳过重复的

            # 添加到seen集合
            seen_weibo_ids.add(weibo_id)

            # 提取字段
            result = {
                "weibo_id": weibo_id,
                "user_id": line_data[4].strip('"'),
                "is_retweet": line_data[3].strip('"'),
                "nick_name": line_data[5].strip('"'),
                "user_type": line_data[7].strip('"'),
                "weibo_content": line_data[9].strip('"'),
                "zhuan": line_data[10].strip('"'),
                "ping": line_data[11].strip('"'),
                "zan": line_data[12].strip('"'),
                "device": "",
                "locate": "",
                "time_stamp": line_data[17].strip('"'),
                "r_user_id": line_data[13].strip('"'),
                "r_nick_name": line_data[14].strip('"'),
                "r_user_type": line_data[15].strip('"'),
                "r_weibo_id": line_data[16].strip('"'),
                "r_weibo_content": line_data[22].strip('"'),
                "r_zhuan": line_data[20].strip('"'),
                "r_ping": line_data[21].strip('"'),
                "r_zan": "",
                "r_device": "",
                "r_location": "",
                "r_time": line_data[19].strip('"'),
                "r_time_stamp": line_data[18].strip('"'),
                "src": "",
                "tag": "",
                "lat": "",
                "lon": "",
            }
            results.append(result)
        except Exception as e:
            continue

    return results


def process_old_format(chunk, seen_weibo_ids):
    """
    处理旧格式（2019年8月9日之前的\t分割格式）
    seen_weibo_ids: set，用于实时去重
    """
    results = []
    for line in chunk:
        try:
            line_data = line.strip().split("\t")
            if len(line_data) < 24:
                continue

            # 先提取weibo_id进行去重检查
            weibo_id = line_data[8]
            if not weibo_id or weibo_id in seen_weibo_ids:
                continue  # 跳过重复的

            # 添加到seen集合
            seen_weibo_ids.add(weibo_id)

            # 提取字段（基于旧格式的列位置）
            result = {
                "weibo_id": weibo_id,
                "user_id": line_data[4],
                "is_retweet": line_data[3],
                "nick_name": line_data[5],
                "user_type": line_data[7],
                "weibo_content": line_data[9],
                "zhuan": line_data[10],
                "ping": line_data[11],
                "zan": line_data[12],
                "device": "",
                "locate": "",
                "time_stamp": line_data[17],
                "r_user_id": line_data[13],
                "r_nick_name": line_data[14],
                "r_user_type": line_data[15],
                "r_weibo_id": line_data[16],
                "r_weibo_content": line_data[22],
                "r_zhuan": line_data[20],
                "r_ping": line_data[21],
                "r_zan": "",
                "r_device": "",
                "r_location": "",
                "r_time": line_data[19],
                "r_time_stamp": line_data[18],
                "src": "",
                "tag": "",
                "lat": "",
                "lon": "",
            }
            results.append(result)
        except Exception as e:
            continue

    return results


def process_json_format(chunk, seen_weibo_ids):
    """
    处理JSON格式（2019年8月9日及之后）
    seen_weibo_ids: set，用于实时去重
    """
    results = []
    for line in chunk:
        try:
            # 按\t分割，第二部分是JSON
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue

            # 解析JSON
            data = orjson.loads(parts[1])

            # 先提取weibo_id进行去重检查
            weibo_id = data.get("weibo_id", "")
            if not weibo_id or weibo_id in seen_weibo_ids:
                continue  # 跳过重复的

            # 添加到seen集合
            seen_weibo_ids.add(weibo_id)

            # 提取字段
            result = {
                "weibo_id": weibo_id,
                "user_id": data.get("user_id", ""),
                "is_retweet": data.get("is_retweet", ""),
                "nick_name": data.get("nick_name", ""),
                "user_type": data.get("user_type", ""),
                "weibo_content": data.get("weibo_content", ""),
                "zhuan": data.get("zhuan", ""),
                "ping": data.get("ping", ""),
                "zan": data.get("zhan", ""),
                "device": data.get("device", ""),
                "locate": data.get("locate", ""),
                "time_stamp": data.get("time_stamp", ""),
                "r_user_id": data.get("r_user_id", ""),
                "r_nick_name": data.get("r_nick_name", ""),
                "r_user_type": data.get("r_user_type", ""),
                "r_weibo_id": data.get("r_weibo_id", ""),
                "r_weibo_content": data.get("r_weibo_content", ""),
                "r_zhuan": data.get("r_zhuan", ""),
                "r_ping": data.get("r_ping", ""),
                "r_zan": data.get("r_zhan", ""),
                "r_device": data.get("r_device", ""),
                "r_location": data.get("r_location", ""),
                "r_time": data.get("r_time", ""),
                "r_time_stamp": data.get("r_time_stamp", ""),
                "src": data.get("src", ""),
                "tag": data.get("tag", ""),
                "lat": data.get("lat", ""),
                "lon": data.get("lon", ""),
            }
            results.append(result)
        except (orjson.JSONDecodeError, IndexError) as e:
            continue

    return results


def process_file(file_path, date):
    """
    处理单个文件，实时去重
    """
    # 确定处理函数
    if date == datetime(2020, 6, 30):
        process_function = process_special_format_2020_06_30
    elif date >= datetime(2019, 8, 9):
        process_function = process_json_format
    else:
        process_function = process_old_format

    chunk_size = 500000
    all_results = []
    seen_weibo_ids = set()  # 在文件处理级别维护seen集合

    # 如果文件不存在，直接返回空列表
    if not os.path.exists(file_path):
        return all_results

    # 逐行读取文件
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        chunk = []
        for line in file:
            chunk.append(line)

            # 当块大小达到限制时，处理该块
            if len(chunk) == chunk_size:
                results = process_function(chunk, seen_weibo_ids)
                all_results.extend(results)
                chunk = []  # 清空块

        # 处理最后一块（如果有剩余）
        if chunk:
            results = process_function(chunk, seen_weibo_ids)
            all_results.extend(results)

    return all_results


def save_to_parquet(date, results):
    """
    保存数据到Parquet文件
    """
    if not results:
        return

    output_parquet_path = f"{OUTPUT_DIR}/{date}.parquet"
    df = pd.DataFrame(results)

    df.to_parquet(output_parquet_path, engine="fastparquet", index=False)
    print(f"保存了 {len(results)} 条记录到 {output_parquet_path}")


def check_year_format(year):
    """
    检查指定年份1月1日的数据格式
    用于检查新年度数据格式是否变化
    """
    date_str = f"{year}-01-01"
    file_path = unzip_one_fresh_data_file(year, date_str)

    if file_path:
        print(f"\n开始检查 {year} 年1月1日文件格式: {file_path}")
        print_first_3_lines(file_path)
        print("\n格式检查完成\n")
        delete_unzipped_fresh_data_file(year, date_str)
    else:
        print(f"无法解压 {year} 年1月1日的数据文件")


def process_year(year, mode):
    """
    处理指定年份的数据
    mode: 0 = 上半年, 1 = 下半年
    """
    start_date_options = [datetime(year, 1, 1), datetime(year, 7, 1)]
    end_date_options = [datetime(year, 6, 30), datetime(year, 12, 31)]
    start_date = start_date_options[mode]
    end_date = end_date_options[mode]

    date_range = [
        start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)
    ]

    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")

        file_path = unzip_one_fresh_data_file(year, date_str)
        if file_path is None:
            continue

        start_timestamp = int(time.time())
        results = process_file(file_path, current_date)
        save_to_parquet(date_str, results)

        log(
            f"处理 {date_str} 完成，耗时 {int(time.time()) - start_timestamp} 秒，共 {len(results)} 条记录。",
            f"{year}_{mode}",
        )

        delete_unzipped_fresh_data_file(year, date_str)
        print(f"finished {date_str} with {len(results)} records")


def main(year: int, mode: int = 0, check: bool = False):
    """
    微博数据清洗主函数

    Args:
        year: 年份（必需）
        mode: 处理模式，0=上半年，1=下半年（默认：0）
        check: 只检查格式，不处理数据（默认：False）
    """
    if check:
        # 只检查格式模式
        check_year_format(year)
    else:
        # 正常处理模式
        print(f"开始处理 {year} 年，模式 {'下半年' if mode == 1 else '上半年'}")
        process_year(year, mode)
        log("处理完毕。")
        print("处理完毕。")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
