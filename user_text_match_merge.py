import json
import os

# merge json files (dict) into one

suffix = ["", "_1", "_2", "_3", "_4", "_5", "_7", "_8", "_10"]
last_ids = []


with open(f"data/weibo_cov_user_to_original_id.json", "r") as f:
    result = json.load(f)
    print(len(result.keys()))


for s in suffix:
    with open(f"user_id_to_original_id{s}.txt", "r") as rfile:
        # 最后一行，用,分割的第一个元素加入last ids, 考虑最后的空行
        for line in rfile.readlines():
            user_id, original_id = line.strip().split(",")
            result[user_id] = original_id
        print(len(result.keys()))
        # last_ids.append(rfile.readlines()[-1].split(",")[0])

OUTPUT_DIR = "match_working_data"
with open(os.path.join(OUTPUT_DIR, "date_covered_users.json"), "r") as f:
    date_covered_users_serializable = json.load(f)

for date, user_list in date_covered_users_serializable.items():
    for idx, uid in enumerate(last_ids):
        if uid in user_list:
            print(f"Found {uid} in {date}-from file {suffix[idx]}")

with open("data/weibo_cov_user_to_original_id.json", "w") as f:
    json.dump(result, f)

print(len(result.keys()))

