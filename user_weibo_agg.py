"""
format the data to a parquet file, user_id, is_retweet, nick_name, user_type, weibo_id, weibo_content, zhuan, ping, zan, device, locate, time_stamp, r_user_id, r_nick_name, r_user_type, r_weibo_id, r_weibo_content, r_zhuan, r_ping, r_zan, r_device, r_location, r_time, r_time_stamp, src, tag, lat, lon
"""

import os
import json
import pandas as pd
import orjson
from datetime import datetime


def process_file(input_file, output_file):
    """Process the input parquet file and save to output parquet file"""
    print(f"Processing file: {input_file}")

    # Read the input parquet file
    df = pd.read_parquet(input_file)

    results = []
    for line in df["line_binary"]:
        try:
            # Split the line into id and json parts
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue

            # Parse the JSON part
            data = orjson.loads(parts[1])

            # Extract required fields
            result = {
                "user_id": data.get("user_id", ""),
                "is_retweet": data.get("is_retweet", ""),
                "nick_name": data.get("nick_name", ""),
                "user_type": data.get("user_type", ""),
                "weibo_id": data.get("weibo_id", ""),
                "weibo_content": data.get("weibo_content", ""),
                "zhuan": data.get("zhuan", ""),
                "ping": data.get("ping", ""),
                "zan": data.get("zhan", ""),  # Note: original field is "zhan"
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
                "r_zan": data.get("r_zhan", ""),  # Note: original field is "r_zhan"
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

    # Convert results to DataFrame and save
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_parquet(output_file, engine="fastparquet", index=False)
        print(f"Saved {len(results)} records to {output_file}")
    else:
        print(f"No valid records found in {input_file}")


def process_directory(input_dir, output_dir):
    """Process all parquet files in the input directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all parquet files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]

    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file)
        process_file(input_path, output_path)


if __name__ == "__main__":
    input_dir = "youth_text_data_dedup"
    output_dir = "youth_weibo_stat"
    process_directory(input_dir, output_dir)
