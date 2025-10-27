"""
通用用户profile提取脚本
提取所有用户的profile数据，不限制用户ID

profile location: /gpfs/share/home/1706188064/group/data/weibo-2020/{year}/user_profile

profile file format: weibo_user_profile.{date}.7z

after unzipped, several lines examples:
13512066647	{"id":"13512066647","crawler_date":"2023-10-04","crawler_time_stamp":"1696348800875","user_id":"7841605551","nick_name":"雨语昕馨",...}
"""

import os
import json
import time
import pandas as pd
import orjson
from datetime import datetime, timedelta

from utils.utils import extract_single_7z_file

OUTPUT_DIR = "cleaned_profile_data"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def log(text, lid=None):
    """记录日志"""
    os.makedirs("logs", exist_ok=True)
    output = (
        f"logs/profile_log_{lid}.txt" if lid is not None else "logs/profile_log.txt"
    )
    with open(output, "a") as f:
        f.write(f"{text}\n")


def get_zipped_profile_file(year, date):
    """Get the path of zipped profile file for a given date"""
    # TODO: 根据实际情况修改路径
    PROFILE_BASE_DIR = "/gpfs/share/home/1706188064/group/data/weibo-2020"
    return f"{PROFILE_BASE_DIR}/{year}/user_profile/weibo_user_profile.{date}.7z"


def get_unzipped_profile_folder(year):
    """Get the path of unzipped profile folder for a given year"""
    return f"profile_working_data/{year}"


def get_unzipped_profile_file(year, date):
    """Get the path of unzipped profile file for a given date"""
    return f"profile_working_data/{year}/weibo_user_profile.{date}"


def delete_unzipped_profile_file(year, date):
    """Delete the unzipped profile file for a given date"""
    file_path = get_unzipped_profile_file(year, date)
    try:
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除。")
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
    except Exception as e:
        print(f"删除文件时发生错误: {e}")


def unzip_one_profile_file(year, date):
    """Unzip the profile file for a given date"""
    unzipped_file_path = get_unzipped_profile_file(year, date)

    # 检查文件是否已经解压
    if os.path.exists(unzipped_file_path):
        print(f"文件 {unzipped_file_path} 已经存在，跳过解压。")
        return unzipped_file_path

    # 如果文件不存在，则进行解压
    zipped_file_path = get_zipped_profile_file(year, date)
    unzipped_dir = get_unzipped_profile_folder(year)
    result = extract_single_7z_file(
        file_path=zipped_file_path, target_folder=unzipped_dir
    )

    if os.path.exists(unzipped_file_path):
        print(f"文件 {unzipped_file_path} 解压成功。")
        return unzipped_file_path
    else:
        print(f"文件 {unzipped_file_path} 解压失败。")
        return None


def process_chunk(chunk):
    """处理chunk中的所有行，解析并格式化"""
    results = []
    for line in chunk:
        try:
            result = single_line_formatter(line)
            if result:
                results.append(result)
        except Exception as e:
            continue

    return results


def single_line_formatter(line):
    """
    格式化单行profile数据

    return dict: {
        "date": date,
        "timestamp": timestamp,
        "user_id": user_id,
        "nick_name": nick_name,
        ...
    }
    """
    try:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            return None

        data = orjson.loads(parts[1])

        # 解析ext字段获取ip_location
        ip_location = ""
        if "ext" in data and data["ext"]:
            try:
                ext_data = orjson.loads(data["ext"])
                ip_location = ext_data.get("ip_location", "")
            except:
                pass

        result = {
            "date": data.get("crawler_date", ""),
            "timestamp": data.get("crawler_time_stamp", ""),
            "user_id": data.get("user_id", ""),
            "nick_name": data.get("nick_name", ""),
            "user_type": data.get("user_type", ""),
            "gender": data.get("gender", ""),
            "verified_type": data.get("verified_type", ""),
            "verified_reason": data.get("verified_reason", ""),
            "description": data.get("description", ""),
            "fans_number": data.get("fans_number", ""),
            "weibo_number": data.get("weibo_number", ""),
            "type": data.get("type", ""),
            "friends_count": data.get("friends_count", ""),
            "favourites_count": data.get("favourites_count", ""),
            "created_at": data.get("created_at", ""),
            "allow_all_comment": data.get("allow_all_comment", ""),
            "bi_followers_count": data.get("bi_followers_count", ""),
            "location": data.get("location", ""),
            "province": data.get("province", ""),
            "city": data.get("city", ""),
            "ip_location": ip_location,
        }
        return result
    except (orjson.JSONDecodeError, IndexError) as e:
        return None


def save_to_parquet(date, results):
    """Save results to parquet file"""
    if not results:
        return

    output_parquet_path = f"{OUTPUT_DIR}/{date}.parquet"
    df = pd.DataFrame(results)
    df.to_parquet(output_parquet_path, engine="fastparquet", index=False)
    print(f"保存了 {len(results)} 条profile记录到 {output_parquet_path}")


def process_file(file_path):
    """处理单个profile文件"""
    all_results = []
    chunk_size = 500000  # 每块50万行

    if not os.path.exists(file_path):
        return all_results

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        chunk = []
        for line in file:
            chunk.append(line)

            # 当块大小达到限制时，处理该块
            if len(chunk) == chunk_size:
                results = process_chunk(chunk)
                all_results.extend(results)
                chunk = []  # 清空块

        # 处理最后一块（如果有剩余）
        if chunk:
            results = process_chunk(chunk)
            all_results.extend(results)

    return all_results


def process_year(year, mode):
    """处理指定年份的profile数据"""
    start_date_options = [datetime(year, 1, 1), datetime(year, 7, 1)]
    end_date_options = [datetime(year, 6, 30), datetime(year, 12, 31)]
    start_date = start_date_options[mode]
    end_date = end_date_options[mode]

    date_range = [
        start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)
    ]

    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")

        start_timestamp = int(time.time())
        file_path = unzip_one_profile_file(year, date_str)

        if file_path is None:
            continue

        results = process_file(file_path)
        save_to_parquet(date_str, results)

        log(
            f"处理 {date_str} 完成，耗时 {int(time.time()) - start_timestamp} 秒，共 {len(results)} 条记录。",
            f"{year}_{mode}",
        )

        delete_unzipped_profile_file(year, date_str)
        print(f"finished {date_str} with {len(results)} profiles")


def check_profile_format(year):
    """检查指定年份1月1日的profile数据格式"""
    date_str = f"{year}-01-01"
    file_path = unzip_one_profile_file(year, date_str)

    if file_path:
        print(f"\n开始检查 {year} 年1月1日profile文件格式: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for i, line in enumerate(file):
                if i >= 3:
                    break
                print(f"Line {i+1}: {line[:200]}")
                print("-" * 50)
        print("\n格式检查完成\n")
        delete_unzipped_profile_file(year, date_str)
    else:
        print(f"无法解压 {year} 年1月1日的profile文件")


def main(year: int, mode: int = 0, check: bool = False):
    """
    用户profile提取主函数

    Args:
        year: 年份（必需）
        mode: 处理模式，0=上半年，1=下半年（默认：0）
        check: 只检查格式，不处理数据（默认：False）
    """
    if check:
        # 只检查格式模式
        check_profile_format(year)
    else:
        # 正常处理模式
        print(
            f"开始处理 {year} 年profile数据，模式 {'下半年' if mode == 1 else '上半年'}"
        )
        process_year(year, mode)
        log("处理完毕。")
        print("处理完毕。")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
