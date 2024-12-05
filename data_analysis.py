import os
import json

import pandas as pd

from collections import defaultdict

from datetime import datetime, timedelta

import time

import matplotlib.pyplot as plt
import seaborn as sns


start_time = int(time.time())

def log(text):
    with open("log.txt", "a") as f:
        f.write(f"{text}\n")

class AddDemographic:
    def __init__(self, ):
        self.load_demographic_mapping()

    def load_demographic_mapping(self, ):
        log(f"Loading demographic data, time: {int(time.time()) - start_time}")

        with open("data/weibo_cov_user_to_original_id.json", "r") as f:
            weibo_cov_user_to_original_id = json.load(f)
        
        # with open("data/youth_user_ids_2016.json", "r") as f:
        #     youth_user_ids = json.load(f)
        
        # youth_cov_to_original = {k: v for k, v in weibo_cov_user_to_original_id.items() if v in youth_user_ids}

        # weibo_cov_user_ids = set(youth_cov_to_original.keys())

        # del youth_user_ids, weibo_cov_user_to_original_id

        data = pd.read_csv("/gpfs/share/home/2401111059/cov-weibo-2/COV-Weibo2.0/user.csv", usecols=["user_id", "gender", "birthday", "province"])
        data['birthday'] = pd.to_datetime(data['birthday'], errors='coerce', format='%Y-%m-%d')
        # 删除缺少birthday的行
        data.dropna(subset=['birthday'], inplace=True)
        # 只保留birthday在2000及之后的数据
        data = data[data['birthday'].dt.year >= 2000]
        # data = data[data['user_id'].isin(weibo_cov_user_ids)]
        log(f"Data loaded after filtering, time: {int(time.time()) - start_time}")

        data['original_id'] = data['user_id'].map(weibo_cov_user_to_original_id)
        data.dropna(subset=['original_id'], inplace=True)

        log(f"Data loaded after mapping, time: {int(time.time()) - start_time}")

        # 将 DataFrame 转换为字典
        self.demographic_data = data.set_index('original_id').to_dict('index')

        log(f"Loading demographic data finished, time: {int(time.time()) - start_time}")

        # demographic_data = defaultdict(dict)

        # for _, row in data.iterrows():
        #     user_id = row['user_id']
        #     demographic_data[youth_cov_to_original[user_id]] = {
        #         "gender": row["gender"],
        #         "province": row["province"],
        #         "birthday": datetime.strptime(row["birthday"], "%Y-%m-%d")
        #     }
        
        # self.demographic_data = demographic_data
    

    def add_demographic(self, row):
        user_id = row["user_id"]
        if user_id not in self.demographic_data:
            return None, None, None
        single_demo = self.demographic_data[user_id]
        # 计算年龄，根据row的date或者year（需要先判断存在哪一个）减去birthday获得年龄
        # 注意birthday是字符串，需要转换为datetime
        age = None
        birthday = single_demo["birthday"]

        if "date" in row:
            age = row["date"].year - birthday.year
        elif "year" in row:
            age = row["year"] - birthday.year
        return single_demo["gender"], age, single_demo["province"]


def add_demographic_for_all():
    demographic_processor = AddDemographic()

    start_date = datetime(2016, 1, 1)
    end_date = datetime(2023, 12, 31)

    date_range = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]
    for current_date in date_range:
        date_start_time = int(time.time())
        log(f"Processing date: {current_date}")

        file_path = os.path.join(f"user_count/count-{current_date.year}", f"{current_date.strftime('%Y-%m-%d')}.parquet")

        output_folder = f"user_count2/count-{current_date.year}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file_path = os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}.parquet")

        if os.path.exists(file_path):
            daily_df = pd.read_parquet(file_path)
            daily_df[["gender", "age", "province"]] = daily_df.apply(demographic_processor.add_demographic, axis=1, result_type='expand')
            daily_df.to_parquet(output_file_path, engine='fastparquet', index=False)
        
        log(f"Processing date: {current_date} finished, time: {int(time.time()) - date_start_time}")
    
    yearly = ["2016", "2020"]
    year_dfs = [] # 合并两个年份数据
    for year in yearly:
        file_path = os.path.join("user_count", f"yearly_user_counts_{year}.parquet")
        if os.path.exists(file_path):
            year_df = pd.read_parquet(file_path)
            year_df[["gender", "age", "province"]] = year_df.apply(demographic_processor.add_demographic, axis=1, result_type='expand')
            year_dfs.append(year_df)
    
    if year_dfs:
        yearly_df = pd.concat(year_dfs)
        yearly_df.to_parquet("user_count2/yearly_user_counts.parquet", engine='fastparquet', index=False)


def gender_analysis():
    yearly_df = pd.read_parquet("user_count2/yearly_user_counts.parquet", engine='fastparquet')
    year_range = range(2016, 2024)

    year_data = []
    year_data_with_missing = []
    absolute_counts = []

    for year in year_range:
        year_data_year = yearly_df[yearly_df["year"] == year]

        # 计算性别比例，包括缺失值
        female_count = year_data_year[year_data_year["gender"] == "女"].shape[0]
        male_count = year_data_year[year_data_year["gender"] == "男"].shape[0]
        total_count = year_data_year.shape[0]
        missing_count = total_count - female_count - male_count
        
        female_ratio = female_count / total_count
        male_ratio = male_count / total_count
        missing_ratio = missing_count / total_count

        year_data_with_missing.append({
            "year": year,
            "male": male_ratio,
            "female": female_ratio,
            "missing": missing_ratio
        })

        # 计算性别比例，不包括缺失值
        year_data_year = year_data_year.dropna(subset=["gender"])
        female_ratio = year_data_year[year_data_year["gender"] == "女"].shape[0] / year_data_year.shape[0]
        male_ratio = 1 - female_ratio
        year_data.append({
            "year": year,
            "male": male_ratio,
            "female": female_ratio
        })

        # 记录绝对数量
        absolute_counts.append({
            "year": year,
            "male": male_count,
            "female": female_count,
            "missing": missing_count
        })
    
    year_data = pd.DataFrame(year_data)
    year_data_with_missing = pd.DataFrame(year_data_with_missing)
    absolute_counts = pd.DataFrame(absolute_counts)

    # 绘制面积图
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # 包含缺失值的比例面积图
    axes[0].stackplot(year_data_with_missing['year'], 
                      year_data_with_missing['female'], 
                      year_data_with_missing['male'], 
                      year_data_with_missing['missing'], 
                      labels=['Female', 'Male', 'Missing'])
    axes[0].set_title('Gender Ratio Over Years (Including Missing)')
    axes[0].legend(loc='upper left')

    # 不包含缺失值的比例面积图
    axes[1].stackplot(year_data['year'], 
                      year_data['female'], 
                      year_data['male'], 
                      labels=['Female', 'Male'])
    axes[1].set_title('Gender Ratio Over Years (Excluding Missing)')
    axes[1].legend(loc='upper left')

    # 绝对数量变化的面积图
    axes[2].stackplot(absolute_counts['year'], 
                      absolute_counts['female'], 
                      absolute_counts['male'], 
                      absolute_counts['missing'], 
                      labels=['Female', 'Male', 'Missing'])
    axes[2].set_title('Absolute Gender Counts Over Years')
    axes[2].legend(loc='upper left')

    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig("images/gender_trend.pdf", format='pdf')



def province_analysis():
    yearly_df = pd.read_parquet("user_count2/yearly_user_counts.parquet", engine='fastparquet')
    year_range = range(2016, 2024)

    # 定义省份到区域的映射
    region_to_province = {
        'East': ['北京', '天津', '河北', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南'],
        'Central': ['山西', '安徽', '江西', '河南', '湖北', '湖南'],
        'West': ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'],
        'Northeast': ['辽宁', '吉林', '黑龙江']
    }

    province_to_region = {}

    for region, provinces in region_to_province.items():
        for province in provinces:
            province_to_region[province] = region
    
    year_data = []

    for year in year_range:
        year_data_year = yearly_df[yearly_df["year"] == year]
        region_counts = year_data_year['province'].map(province_to_region).dropna().value_counts(normalize=True)
        
        year_data.append({
            "year": year,
            "East": region_counts.get('East', 0),
            "West": region_counts.get('West', 0),
            "Central": region_counts.get('Central', 0),
            "Northeast": region_counts.get('Northeast', 0)
        })

    year_data = pd.DataFrame(year_data)

    # 绘制面积图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(year_data['year'], 
                 year_data['East'], 
                 year_data['West'], 
                 year_data['Central'], 
                 year_data['Northeast'], 
                 labels=['East', 'West', 'Central', 'Northeast'])
    ax.set_title('Province Distribution Over Years')
    ax.legend(loc='upper left')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig("images/province_trend.pdf", format='pdf')


def age_analysis():
    yearly_df = pd.read_parquet("user_count2/yearly_user_counts.parquet", engine='fastparquet')
    year_range = range(2016, 2024)

    # 定义年龄段
    bins = [0, 4, 8, 12, 16]
    labels = ['0-4', '4-8', '8-12', '12-16']

    year_data = []

    for year in year_range:
        year_data_year = yearly_df[yearly_df["year"] == year]
        age_groups = pd.cut(year_data_year['age'], bins=bins, labels=labels, right=False)
        age_counts = age_groups.value_counts(normalize=True, sort=False)
        
        year_data.append({
            "year": year,
            "0-4": age_counts.get('0-4', 0),
            "4-8": age_counts.get('4-8', 0),
            "8-12": age_counts.get('8-12', 0),
            "12-16": age_counts.get('12-16', 0)
        })

    year_data = pd.DataFrame(year_data)

    # 绘制面积图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(year_data['year'], 
                 year_data['0-4'], 
                 year_data['4-8'], 
                 year_data['8-12'], 
                 year_data['12-16'], 
                 labels=labels)
    ax.set_title('Age Distribution Over Years')
    ax.legend(loc='upper left')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig("images/age_trend.pdf", format='pdf')



if __name__ == "__main__":
    add_demographic_for_all()