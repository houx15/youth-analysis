"""
Find all profiles for each user, user id to match, keep the efficiency high by skipping json decoder but use text level matching

user id list: data/youth_user_ids_list_2020.json
["1231231", "123123123123"]

profile location: /gpfs/share/home/1706188064/group/data/weibo-2020/2020/user_profile

profile file format (examples):
weibo_user_profile.2020-05-25.7z  weibo_user_profile.2020-12-25.7z
weibo_user_profile.2020-05-26.7z  weibo_user_profile.2020-12-26.7z
weibo_user_profile.2020-05-27.7z  weibo_user_profile.2020-12-27.7z
weibo_user_profile.2020-05-28.7z  weibo_user_profile.2020-12-28.7z
weibo_user_profile.2020-05-29.7z  weibo_user_profile.2020-12-29.7z
weibo_user_profile.2020-05-30.7z  weibo_user_profile.2020-12-30.7z
weibo_user_profile.2020-05-31.7z  weibo_user_profile.2020-12-31.7z

after unzipped, several lines examples:
13512066647	{"id":"13512066647","crawler_date":"2023-10-04","crawler_time_stamp":"1696348800875","user_id":"7841605551","nick_name":"雨语昕馨","tou_xiang":"https:\/\/tvax4.sinaimg.cn\/crop.0.0.1040.1040.50\/008yGzOTly8hfjm9mv2hmj30sw0sw0tx.jpg?KID=imgbed,tva&Expires=1696359600&ssig=GUM0MPgbZv","user_type":"普通用户","gender":"m","verified_type":"-1","verified_reason":"","description":"","fans_number":"1","weibo_number":"11","type":"1","friends_count":"36","favourites_count":"0","created_at":"2023-05-27 20:41:15","allow_all_comment":"1","bi_followers_count":"0","location":"IP属地：山东","province":"100","city":"1000","domain":"","ext":"{\"ip_location\":\"山东\"}","d":"2023-10-04"}
13512066648	{"id":"13512066648","crawler_date":"2023-10-04","crawler_time_stamp":"1696348800876","user_id":"7025268325","nick_name":"别跑美嘉-永不失联版","tou_xiang":"https:\/\/tvax4.sinaimg.cn\/crop.0.0.914.914.50\/007Frjdrly8h5qf48g7dcj30pe0pemxy.jpg?KID=imgbed,tva&Expires=1696359600&ssig=W7JPYodKjD","user_type":"普通用户","gender":"f","verified_type":"-1","verified_reason":"","description":"去夏威夷吹海风","fans_number":"10","weibo_number":"1458","type":"1","friends_count":"458","favourites_count":"123","created_at":"2019-03-08 22:30:26","allow_all_comment":"1","bi_followers_count":"0","location":"IP属地：重庆","province":"50","city":"1000","domain":"","ext":"{\"ip_location\":\"重庆\"}","d":"2023-10-04"}

step1: extract all profile data if the profile data is in the user id list
step2: convert it to a parquet:
date, user_id, nick_name, user_type, gender, verified_type, verified_reason, description, fans_number, weibo_number, type, friends_count, favourites_count, created_at, allow_all_comment, bi_followers_count, location, province, city, ip_location
"""

import os
import json
import time
import pandas as pd
import py7zr
from datetime import datetime, timedelta
from io import StringIO
from ahocorasick import Automaton

from utils.utils import extract_single_7z_file
import orjson


def log(text, lid=None):
    """Write log to user_profile_match.log"""
    with open("user_profile_match.log", "a") as f:
        f.write(f"{text}\n")


def get_zipped_profile_file(year, date):
    """Get the path of zipped profile file for a given date"""
    return f"/gpfs/share/home/1706188064/group/data/weibo-2020/{year}/user_profile/weibo_user_profile.{date}.7z"


def get_unzipped_profile_folder(year):
    """Get the path of unzipped profile folder for a given year"""
    return f"user_profile_data/{year}"


def get_unzipped_profile_file(year, date):
    """Get the path of unzipped profile file for a given date"""
    return f"user_profile_data/{year}/weibo_user_profile.{date}"


def delete_unzipped_profile_file(year, date):
    """Delete the unzipped profile file for a given date"""
    file_path = get_unzipped_profile_file(year, date)
    if os.path.exists(file_path):
        os.remove(file_path)


def unzip_one_profile_file(year, date):
    """Unzip the profile file for a given date"""
    unzipped_file_path = get_unzipped_profile_file(year, date)
    if os.path.exists(unzipped_file_path):
        return unzipped_file_path
    zipped_file_path = get_zipped_profile_file(year, date)
    unzipped_dir = get_unzipped_profile_folder(year)
    result = extract_single_7z_file(
        file_path=zipped_file_path, target_folder=unzipped_dir
    )
    if os.path.exists(unzipped_file_path):
        return unzipped_file_path
    else:
        return None


def process_lines(list_of_lines, set_of_user_ids):
    results = []
    for line in list_of_lines:
        """
        13512066647	{"id":"13512066647","crawler_date":"2023-10-04","crawler_time_stamp":"1696348800875","user_id":"7841605551","nick_name":"雨语昕馨","tou_xiang":"https:\/\/tvax4.sinaimg.cn\/crop.0.0.1040.1040.50\/008yGzOTly8hfjm9mv2hmj30sw0sw0tx.jpg?KID=imgbed,tva&Expires=1696359600&ssig=GUM0MPgbZv","user_type":"普通用户","gender":"m","verified_type":"-1","verified_reason":"","description":"","fans_number":"1","weibo_number":"11","type":"1","friends_count":"36","favourites_count":"0","created_at":"2023-05-27 20:41:15","allow_all_comment":"1","bi_followers_count":"0","location":"IP属地：山东","province":"100","city":"1000","domain":"","ext":"{\"ip_location\":\"山东\"}","d":"2023-10-04"}
        """
        try:
            user_id = line.split('"user_id":"')[1].split('","')[0]
        except:
            continue
        if user_id in set_of_user_ids:
            results.append(line)
    return results


def single_line_formatter(line):
    """
    13512066647	{"id":"13512066647","crawler_date":"2023-10-04","crawler_time_stamp":"1696348800875","user_id":"7841605551","nick_name":"雨语昕馨","tou_xiang":"https:\/\/tvax4.sinaimg.cn\/crop.0.0.1040.1040.50\/008yGzOTly8hfjm9mv2hmj30sw0sw0tx.jpg?KID=imgbed,tva&Expires=1696359600&ssig=GUM0MPgbZv","user_type":"普通用户","gender":"m","verified_type":"-1","verified_reason":"","description":"","fans_number":"1","weibo_number":"11","type":"1","friends_count":"36","favourites_count":"0","created_at":"2023-05-27 20:41:15","allow_all_comment":"1","bi_followers_count":"0","location":"IP属地：山东","province":"100","city":"1000","domain":"","ext":"{\"ip_location\":\"山东\"}","d":"2023-10-04"}

    return dict {
    "timestamp": 1696348800875,
    "user_id": 7841605551,
    "nick_name": "雨语昕馨",
    "user_type": "普通用户",
    "gender": "m",
    "verified_type": "-1",
    "verified_reason": "",
    "description": "",
    "fans_number": 1,
    "weibo_number": 11,
    "type": 1,
    "friends_count": 36,
    "favourites_count": 0,
    "created_at": "2023-05-27 20:41:15",
    "allow_all_comment": 1,
    "bi_followers_count": 0,
    "location": "IP属地：山东",
    "province": 100,
    "city": 1000,
    "ip_location": "山东"
    }
    """
    try:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            return
        data = orjson.loads(parts[1])
        result = {
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
            "ip_location": data.get("ip_location", ""),
        }
        return result
    except (orjson.JSONDecodeError, IndexError) as e:
        return


def save_to_parquet(date, results):
    """Save results to parquet file"""
    if not results:
        return

    formatted_results = []
    for result in results:
        formatted_result = single_line_formatter(result)
        if formatted_result:
            formatted_results.append(formatted_result)

    if formatted_results:
        df = pd.DataFrame(formatted_results)
        output_parquet_path = f"youth_profile_data/{date}.parquet"
        df.to_parquet(output_parquet_path, engine="fastparquet", index=False)
        log(f"Saved {len(formatted_results)} profiles to {output_parquet_path}")


def process_file(file_path, matched_ids):
    all_results = []
    chunk_size = 50000
    if not os.path.exists(file_path):
        return all_results
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                results = process_lines(chunk, matched_ids)
                all_results.extend(results)
                chunk = []
        if chunk:
            results = process_lines(chunk, matched_ids)
            all_results.extend(results)
        return all_results


def process_year(year):
    """Process all profile files for a given year"""
    # Load user IDs to match
    with open(f"data/youth_user_ids_{year}.json", "r") as f:
        year_match_ids = json.load(f)
        # Convert to set for faster lookups
        all_matched_ids = set(year_match_ids)

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        log(f"Processing date: {date_str}")

        start_timestamp = int(time.time())

        file_path = unzip_one_profile_file(year, date_str)
        if file_path is None:
            log(f"File not found: {file_path}")
            current_date += timedelta(days=1)
            continue

        results = process_file(file_path, all_matched_ids)
        save_to_parquet(date_str, results)

        log(
            f"处理 {date_str} 完成，耗时 {int(time.time()) - start_timestamp} 秒。",
            f"{year}",
        )
        delete_unzipped_profile_file(year, date_str)

        current_date += timedelta(days=1)


if __name__ == "__main__":
    process_year(2020)
    log("处理完毕。")
