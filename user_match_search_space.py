"""
match weibo_cov user to original user
solution:
1. scan text data and get a minimal day-window for the whole dataset
2. Get a search space for each user (5 blogs)

The following steps are in another script:
3. unzip the relevant original text data
4. use create_time to match a set of text
5. use the text to match the user
"""
import os
import json
import time

import pandas as pd
from configs.configs import COV_WEIBO_DIR

from collections import defaultdict


user_path = os.path.join(COV_WEIBO_DIR, "user.csv")

user_df = pd.read_csv(user_path, usecols=["user_id"])
target_users = set(user_df["user_id"])

del user_df

# 1. scan text data and get a minimal day-window

# text_suffix = ["2019-12.csv"]
text_suffix = []
for i in range(1, 13):
    if i == 10:
        # 没有10月的数据
        continue
    text_suffix.append(f"2020-{i:02}.csv")


date_user_map = defaultdict(set)
user_id_to_text_id = defaultdict(dict)

for suffix in text_suffix:
    text_path = os.path.join(COV_WEIBO_DIR, suffix)

    start = int(time.time())
    text_df = pd.read_csv(text_path, usecols=["_id", "user_id", "created_at", "content"])
    print(f"Read {suffix} in {int(time.time()) - start} seconds")

    text_df["day"] = pd.to_datetime(text_df["created_at"], errors='coerce').dt.date
    text_df.dropna(subset=["day"], inplace=True)
    print(f"processed datetime in {int(time.time()) - start} seconds")

    for date, group in text_df.groupby("day"):
        # 如果日期是9月7日-11月6日（包含边界）之间，则不加入处理
        if date >= pd.to_datetime("2020-09-07").date() and date <= pd.to_datetime("2020-11-06").date():
            continue
        unique_users = group.drop_duplicates(subset=["user_id"], keep="first")
        group_user_set = unique_users["user_id"]
        text_id_set = unique_users["_id"]

        date_user_map[date].update(group_user_set)
        user_id_to_text_id[date.strftime('%Y-%m-%d')]= dict(zip(group_user_set, text_id_set))
    print(f"date group in {int(time.time()) - start} seconds")


del text_df

covered_users = set()
selected_dates = set()

date_covered_users = defaultdict(set)

# 用贪心算法进行初步优化
while covered_users != target_users:
    print("searching")
    best_date = None
    best_coverage = set()
    
    for date, users in date_user_map.items():
        new_covered = users - covered_users
        date_user_map[date] = new_covered
        if len(new_covered) > len(best_coverage):
            best_date = date
            best_coverage = new_covered
    
    if best_date is None:
        break  # 无法覆盖更多用户

    date_covered_users[best_date.strftime('%Y-%m-%d')] = best_coverage
    
    selected_dates.add(best_date.strftime('%Y-%m-%d'))
    covered_users.update(best_coverage)

    del date_user_map[best_date]

    print(f"round in {int(time.time()) - start} seconds, added {len(best_coverage)} users")


del date_user_map

print(f"Covered {len(covered_users)} users with {len(selected_dates)} days")

# save covered_users, selected_dates, and date_covered_users as json
OUTPUT_DIR = "match_working_data"

with open(os.path.join(OUTPUT_DIR, "covered_users.json"), "w") as f:
    json.dump(list(covered_users), f)

with open(os.path.join(OUTPUT_DIR, "user_to_text.json"), "w") as f:
    json.dump(user_id_to_text_id, f)

date_covered_users_serializable = {str(k): list(v) for k, v in date_covered_users.items()}

with open(os.path.join(OUTPUT_DIR, "date_covered_users.json"), "w") as f:
    json.dump(date_covered_users_serializable, f)


# 2. Get a search space for each user (3-5 blogs)

"""
更新：不要一个个用户找资料了
直接打开日期文件，按行匹配即可
"""

# months = set()
# month_dates = defaultdict(set)

# user_content = defaultdict(list)

# for date_str in date_covered_users.keys():
#     months.add(date_str[:7])
#     month_dates[date_str[:7]].add(date_str)


# for month in months:
#     text_df = pd.read_csv(os.path.join(COV_WEIBO_DIR, f"{month}.csv"), usecols=["user_id", "created_at", "content"])

#     text_df["day"] = pd.to_datetime(text_df["created_at"], errors='coerce').dt.date

#     for date_str in month_dates[month]:
#         target_date = pd.to_datetime(date_str).date()
#         user_set = date_covered_users[date_str]

#         filtered_date_df = text_df[text_df["day"] == target_date]

#         for user_id in user_set:
#             user_rows = filtered_date_df[filtered_date_df["user_id"] == user_id]
#             formatted_content = [f"{row['created_at']},{row['content']}" for _, row in user_rows.iterrows()]
#             user_content[user_id].extend(formatted_content)

# # save user_content as json
# with open(os.path.join(OUTPUT_DIR, "user_content.json"), "w") as f:
#     json.dump(user_content, f)