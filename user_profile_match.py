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
import multiprocessing as mp
from itertools import islice
import orjson


def log(text, lid=None):
    output = f"logs/log_{lid}.txt" if lid is not None else "logs/log.txt"
    with open(output, "a") as f:
        f.write(f"{text}\n")


def get_zipped_profile_file(year, date):
    """Get the path of zipped profile file for a given date"""
    return f"/gpfs/share/home/1706188064/group/data/weibo-2020/{year}/user_profile/weibo_user_profile.{date}.7z"


def create_automaton(user_ids):
    """Create Aho-Corasick automaton for fast string matching"""
    A = Automaton()
    for user_id in user_ids:
        # Add the pattern with quotes to match the JSON format
        pattern = user_id
        A.add_word(pattern, user_id)
    A.make_automaton()
    return A


def process_chunk(chunk, automaton):
    """Process a chunk of lines using the automaton"""
    results = []
    for line in chunk:
        # Use Aho-Corasick to find matches
        matches = []
        for end_index, user_id in automaton.iter(line):
            matches.append(user_id)

        if matches:  # If we found any matches
            try:
                # Get the JSON part (after the tab)
                json_part = line.split("\t")[1]
                profile_data = orjson.loads(json_part)

                # Extract required fields
                result = {
                    "date": profile_data.get("crawler_date", ""),
                    "user_id": profile_data.get("user_id", ""),
                    "nick_name": profile_data.get("nick_name", ""),
                    "user_type": profile_data.get("user_type", ""),
                    "gender": profile_data.get("gender", ""),
                    "verified_type": profile_data.get("verified_type", ""),
                    "verified_reason": profile_data.get("verified_reason", ""),
                    "description": profile_data.get("description", ""),
                    "fans_number": profile_data.get("fans_number", ""),
                    "weibo_number": profile_data.get("weibo_number", ""),
                    "type": profile_data.get("type", ""),
                    "friends_count": profile_data.get("friends_count", ""),
                    "favourites_count": profile_data.get("favourites_count", ""),
                    "created_at": profile_data.get("created_at", ""),
                    "allow_all_comment": profile_data.get("allow_all_comment", ""),
                    "bi_followers_count": profile_data.get("bi_followers_count", ""),
                    "location": profile_data.get("location", ""),
                    "province": profile_data.get("province", ""),
                    "city": profile_data.get("city", ""),
                    "ip_location": orjson.loads(profile_data.get("ext", "{}")).get(
                        "ip_location", ""
                    ),
                }
                results.append(result)
            except (orjson.JSONDecodeError, IndexError):
                continue
    return results


def chunk_generator(file_content, chunk_size=100000):
    """Generate chunks of lines from file content"""
    lines = file_content.splitlines()
    for i in range(0, len(lines), chunk_size):
        yield lines[i : i + chunk_size]


def process_profile_file(file_path, matched_ids):
    """Process profile file and extract matching profiles directly from 7z file"""
    if not os.path.exists(file_path):
        return []

    # Create automaton for fast matching
    automaton = create_automaton(matched_ids)
    results = []

    try:
        with py7zr.SevenZipFile(file_path, mode="r") as z:
            # Get the first file in the archive (there should be only one)
            file_name = z.getnames()[0]
            # Read the file content
            file_obj = z.read([file_name])[file_name]
            # 如果 file_obj 是 BytesIO，需要 getvalue
            if hasattr(file_obj, "getvalue"):
                content = file_obj.getvalue().decode("utf-8", errors="replace")
            else:
                content = file_obj.decode("utf-8", errors="replace")

            # Create a pool of workers
            num_cores = mp.cpu_count()
            chunk_size = 100000  # Process 100k lines at a time

            # Process chunks in parallel
            with mp.Pool(num_cores) as pool:
                # Create chunks and process them
                chunks = chunk_generator(content, chunk_size)
                # Process chunks in parallel
                for chunk_results in pool.imap_unordered(
                    lambda chunk: process_chunk(chunk, automaton), chunks
                ):
                    results.extend(chunk_results)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []

    return results


def save_to_parquet(results, year, date):
    """Save results to parquet file by date"""
    if not results:
        return

    output_dir = f"youth_profile_data/{year}"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    output_path = f"{output_dir}/user_profiles_{date}.parquet"

    df.to_parquet(output_path, engine="fastparquet", index=False)
    print(f"Saved {len(results)} profiles to {output_path}")


def process_year(year):
    """Process all profile files for a given year"""
    # Load user IDs to match
    with open(f"data/youth_user_ids_list_{year}.json", "r") as f:
        year_match_ids = json.load(f)
        # Convert to set for faster lookups
        all_matched_ids = set(year_match_ids)

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Processing date: {date_str}")

        file_path = get_zipped_profile_file(year, date_str)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            current_date += timedelta(days=1)
            continue

        start_timestamp = int(time.time())
        results = process_profile_file(file_path, all_matched_ids)
        save_to_parquet(results, year, date_str)

        log(
            f"处理 {date_str} 完成，耗时 {int(time.time()) - start_timestamp} 秒。",
            f"{year}",
        )

        print(f"Finished {date_str} with {len(results)} records")
        current_date += timedelta(days=1)


if __name__ == "__main__":
    process_year(2020)
    log("处理完毕。")
