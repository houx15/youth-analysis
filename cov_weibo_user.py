import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件，并选择 "birthday" 和 "province" 列
data = pd.read_csv("/gpfs/share/home/2401111059/cov-weibo-2/COV-Weibo2.0/user.csv", usecols=["user_id", "gender", "birthday", "province"])

with open(f"data/weibo_cov_user_to_original_id.json", "r") as f:
    weibo_cov_user_to_original_id = json.load(f)
    matched_ids = list(weibo_cov_user_to_original_id.keys())

print("总用户数目：", data.shape)


# 删除缺少 "birthday" 和 "province" 的行
data = data.dropna(subset=['birthday'])

# 剔除出生年份开头不是19/20的数据
data['birthday'] = pd.to_datetime(data['birthday'], errors='coerce', format='%Y-%m-%d')
data = data.dropna(subset=['birthday'])
data = data[data['birthday'].dt.year >= 1920]
data = data[data['birthday'].dt.year <= 2020]
print("无生日缺失值或异常值：", data.shape)

# data 只保留匹配的用户
data = data[data['user_id'].isin(matched_ids)]
print("匹配的用户数目：", data.shape)

birthday_data = data[data['birthday'].dt.year >= 1949]
# 根据出生年份绘制分布图
plt.figure(figsize=(12, 8))
birthday_data['birthday'].dt.year.value_counts().sort_index().plot(kind='bar')
plt.title('用户出生年份分布')
plt.xlabel('出生年份')
plt.ylabel('用户数量')
# xticks太过于拥挤
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("birthday_count.pdf", format="pdf")


def year_youth_keys(year):
    target_year = year - 16
    filtered_df = data[data['birthday'].dt.year >= target_year]
    # 筛选weibo_cov_user_to_original_id中key在filtered_df的user_id中的数据，保存values
    year_youth_users =  [weibo_cov_user_to_original_id[user_id] for user_id in filtered_df['user_id'] if user_id in weibo_cov_user_to_original_id]
    # 存储到文件, json格式
    with open(f"data/youth_user_ids_{year}.json", "w") as f:
        json.dump(year_youth_users, f)

for year in range(2004, 2024):
    year_youth_keys(year)

# 过滤出生日在 2004 年 1 月 1 日及之后的用户
filtered_df = data[data['birthday'] >= '2004-01-01']

print("青少年用户数量，按照生日再2004年1月1日之后计算", filtered_df.shape)

non_missing_data = filtered_df.dropna(subset=['gender', 'province'])
print("无性别缺失值和省份缺失值：", non_missing_data.shape)


# 定义要保留的省份列表
provinces_to_keep = [
    '北京', '上海', '广东', '台湾', '辽宁', '福建', '浙江', '四川', '河南', '吉林', '海南', 
    '河北', '陕西', '江苏', '山东', '湖北', '湖南', '重庆', '香港', '山西', '天津', '安徽', '广西', 
    '内蒙古', '宁夏', '云南', '新疆', '黑龙江', '青海', '江西', '西藏', '澳门', '甘肃', '贵州'
]

# 筛选出 'province' 列值属于指定列表的行
filtered_by_province = non_missing_data[non_missing_data['province'].isin(provinces_to_keep)]
print("去除海外、其他等，有有效省份数据等青少年数目：", filtered_by_province.shape)


# 统计各省份的用户数量
province_counts = filtered_by_province['province'].value_counts()

# 设置 matplotlib 的字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False    # 用于正常显示负号

# 绘制柱状图
plt.figure(figsize=(12, 8))
province_counts.plot(kind='bar')
plt.title('各省份青少年微博用户分布')
plt.xlabel('省份')
plt.ylabel('用户数量')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("province_count.pdf", format="pdf")