import pandas as pd

# data = pd.read_parquet("youth_text_data/2020-06-30.parquet")
# print(data.shape)
# # data = data.sample(30)
# # for _, row in data.iterrows():
# #     print(row["line_binary"])

# data = pd.read_parquet("youth_text_data_dedup/2020-06-30.parquet")
# print(data.shape)

data = pd.read_csv("/gpfs/share/home/2401111059/cov-weibo-2/COV-Weibo2.0/user.csv", usecols=["user_id", "gender", "birthday", "province"])
print(data["gender"].value_counts())
data['birthday'] = pd.to_datetime(data['birthday'], errors='coerce', format='%Y-%m-%d')

data.dropna(subset=['birthday'], inplace=True)

data = data[data['birthday'].dt.year >= 2000]

print(data.shape, data[data["gender"] == "女"].shape, data[data["gender"] == "男"].shape, data["gender"].value_counts())