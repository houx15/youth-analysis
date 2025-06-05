from configs.configs import *
import pandas as pd
import glob
from collections import Counter
import os
import json
import time
import fire
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import jieba.posseg as pseg
from wordcloud import WordCloud
import numpy as np
from utils.utils import sentence_cleaner
import geopandas as gpd
from shapely.geometry import Point

# Region mapping
region_to_province = {
    "East": [
        "北京",
        "天津",
        "河北",
        "上海",
        "江苏",
        "浙江",
        "福建",
        "山东",
        "广东",
        "海南",
    ],
    "Central": ["山西", "安徽", "江西", "河南", "湖北", "湖南"],
    "West": [
        "内蒙古",
        "广西",
        "重庆",
        "四川",
        "贵州",
        "云南",
        "西藏",
        "陕西",
        "甘肃",
        "青海",
        "宁夏",
        "新疆",
    ],
    "Northeast": ["辽宁", "吉林", "黑龙江"],
}

province_to_region = {}
for region, provinces in region_to_province.items():
    for province in provinces:
        province_to_region[province] = region


def get_region_from_province(province):
    """Get region from province name"""
    if pd.isna(province):
        return None
    return province_to_region.get(province)


def get_region_from_location(location):
    """Get region from location string"""
    if pd.isna(location) or location == "":
        return None
    # Split by space and get first element
    province = location.split()[0]
    return province_to_region.get(province)


def get_province_from_location(location):
    """Get province from location string"""
    if pd.isna(location) or location == "":
        return None
    # Split by space and get first element
    province = location.split()[0]
    return province


def load_demographic_data():
    """Load demographic data from COV-Weibo dataset"""
    print("Loading demographic data...")

    # Load user ID mapping
    with open("data/weibo_cov_user_to_original_id.json", "r") as f:
        weibo_cov_user_to_original_id = json.load(f)

    # Load user data
    data = pd.read_csv(
        "/gpfs/share/home/2401111059/cov-weibo-2/COV-Weibo2.0/user.csv",
        usecols=["user_id", "gender", "birthday", "province"],
    )

    # Process birthday
    data["birthday"] = pd.to_datetime(
        data["birthday"], errors="coerce", format="%Y-%m-%d"
    )
    data.dropna(subset=["birthday"], inplace=True)
    data = data[data["birthday"].dt.year >= 2000]

    # Map to original IDs
    data["original_id"] = data["user_id"].map(weibo_cov_user_to_original_id)
    data.dropna(subset=["original_id"], inplace=True)

    # Convert to dictionary
    demographic_data = data.set_index("original_id").to_dict("index")

    print("Demographic data loaded successfully")
    return demographic_data


def analyze_profiles():
    """Analyze merged user profiles"""
    # Load merged profiles
    log("analyzing profiles")
    df = pd.read_parquet("merged_profiles/merged_user_profiles.parquet")

    # Create figures directory
    os.makedirs("figures", exist_ok=True)

    # 1. Calculate age and filter
    df["birthday"] = pd.to_datetime(df["birthday"])
    df["age"] = 2020 - df["birthday"].dt.year

    # Count before filtering
    total_users = len(df)

    # Filter age range
    df_filtered = df[(df["age"] >= 10) & (df["age"] <= 16)]
    filtered_users = len(df_filtered)

    log(f"\n1. Age Filtering Results:")
    log(f"Total users: {total_users}")
    log(f"Users in age range 10-16: {filtered_users}")
    log(
        f"Percentage filtered out: {((total_users - filtered_users) / total_users * 100):.2f}%"
    )

    # 2. Age distribution
    plt.figure(figsize=(10, 6))
    # 统计每个年龄，一共有10-16，的数目
    sns.histplot(data=df_filtered, x="age", bins=7)
    plt.title("Age Distribution (10-16 years)")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig("figures/age_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Gender ratio (pie chart)
    gender_counts = df_filtered["gender"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%")
    plt.title("Gender Distribution")
    plt.savefig("figures/gender_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # 4. Age-Gender stacked bar chart
    age_gender = (
        pd.crosstab(df_filtered["age"], df_filtered["gender"], normalize="index") * 100
    )
    plt.figure(figsize=(12, 6))
    age_gender.plot(kind="bar", stacked=True)
    plt.title("Gender Distribution by Age")
    plt.xlabel("Age")
    plt.ylabel("Percentage")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig("figures/age_gender_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # 5. Region analysis
    df_filtered["region"] = df_filtered["location"].apply(get_region_from_location)
    region_counts = df_filtered["region"].value_counts()

    # Calculate percentages
    total_valid = region_counts.sum()
    other_count = len(
        df_filtered[df_filtered["location"].str.contains("其他|海外", na=False)]
    )
    other_percentage = (other_count / len(df_filtered)) * 100

    log(f"\n5. Region Analysis:")
    log(f"Region distribution:")
    for region, count in region_counts.items():
        log(f"{region}: {count} ({count/total_valid*100:.2f}%)")
    log(f"Other/Overseas: {other_count} ({other_percentage:.2f}%)")

    # Region pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(region_counts, labels=region_counts.index, autopct="%1.1f%%")
    plt.title("Region Distribution")
    plt.savefig("figures/region_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    # 6. Province analysis
    df_filtered["province"] = df_filtered["location"].apply(get_province_from_location)
    province_counts = df_filtered["province"].value_counts()
    log(f"\n6. Province Analysis:")
    log(f"Province distribution:")
    for province, count in province_counts.items():
        log(f"{province}: {count} ({count/total_valid*100:.2f}%)")

    # bar plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_filtered, x="province", bins=province_counts.shape[0])
    plt.title("Province Distribution")
    plt.xlabel("Province")
    plt.ylabel("Count")
    plt.savefig("figures/province_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def merge_user_profiles():
    """
    Merge user profiles from all parquet files, keeping the latest values for most fields
    and using mode (most frequent value) for location-related fields.
    Also adds birthday and region information from demographic data.
    """
    # Get all parquet files
    parquet_files = glob.glob("youth_profile_data/2020/*.parquet")

    if not parquet_files:
        print("No parquet files found in youth_profile_data directory")
        return

    # Load demographic data
    demographic_data = load_demographic_data()

    # Read and concatenate all parquet files
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    all_profiles = pd.concat(dfs, ignore_index=True)

    # Convert timestamp to datetime for sorting
    # all_profiles["timestamp"] = pd.to_datetime(all_profiles["timestamp"], unit="ms")
    # crawler_date 2023-10-04
    all_profiles["date"] = pd.to_datetime(all_profiles["date"])

    # Group by user_id
    grouped = all_profiles.groupby("user_id")

    # Initialize lists to store results
    merged_profiles = []

    # Process each user's profiles
    for user_id, group in grouped:
        # Sort by timestamp to get the latest values
        group = group.sort_values("date", ascending=False)

        # Get the latest values for most fields
        latest_profile = group.iloc[0].copy()

        # For location-related fields, get the mode
        location_fields = ["location", "province", "city", "ip_location"]
        for field in location_fields:
            if field in group.columns:
                # Get non-empty values
                values = group[field].dropna().astype(str)
                if not values.empty:
                    # Get the most common value
                    mode_value = Counter(values).most_common(1)[0][0]
                    latest_profile[field] = mode_value

        # Add demographic information
        if user_id in demographic_data:
            demo = demographic_data[user_id]
            latest_profile["birthday"] = demo["birthday"]
            latest_profile["demographic_gender"] = demo["gender"]
            latest_profile["demographic_province"] = demo["province"]
            latest_profile["region"] = get_region_from_province(demo["province"])

        merged_profiles.append(latest_profile)

    # Convert to DataFrame
    merged_df = pd.DataFrame(merged_profiles)

    # Drop the timestamp column as it's no longer needed
    # if "timestamp" in merged_df.columns:
    #     merged_df = merged_df.drop("timestamp", axis=1)

    # Create output directory if it doesn't exist
    os.makedirs("merged_profiles", exist_ok=True)

    # Save to parquet
    output_path = "merged_profiles/merged_user_profiles.parquet"
    merged_df.to_parquet(output_path, engine="fastparquet", index=False)
    print(f"Saved merged profiles to {output_path}")
    print(f"Total unique users: {len(merged_df)}")
    print(f"Users with demographic data: {merged_df['birthday'].notna().sum()}")


def log(text):
    """Write log to user_profile_analysis.log"""
    with open("user_profile_analysis.log", "a") as f:
        f.write(f"{text}\n")


def get_month_files(year, month):
    """Get all parquet files for a specific month"""
    month_str = f"{month:02d}"
    pattern = f"youth_weibo_stat/{year}-{month_str}-*.parquet"
    return sorted(glob.glob(pattern))


def analyze_tweet_basic(year):
    """Basic analysis of tweet data"""
    log(f"analyzing tweet basic for {year}")
    # Load tweet data
    os.makedirs(f"figures/{year}", exist_ok=True)
    parquet_files = glob.glob(f"youth_weibo_stat/{year}-*.parquet")
    if not parquet_files:
        log(f"No parquet files found for year {year}")
        return

    # Only read necessary columns
    needed_columns = ["lat", "lon", "device"]
    df = pd.concat([pd.read_parquet(f, columns=needed_columns) for f in parquet_files])

    # 1. Location data analysis
    total_tweets = len(df)
    # lat lon不为空，不是na
    tweets_with_location = df[df["lat"].notna() & df["lon"].notna()]
    # 不为空
    tweets_with_location = tweets_with_location[
        (tweets_with_location["lat"].astype(str) != "")
        & (tweets_with_location["lon"].astype(str) != "")
    ]
    location_percentage = (tweets_with_location.shape[0] / total_tweets) * 100

    log(f"\nLocation Data Analysis:")
    log(f"Total tweets: {total_tweets}")
    log(f"Tweets with location: {tweets_with_location}")
    log(f"Percentage with location: {location_percentage:.2f}%")

    # 2. Device analysis
    device_counts = df["device"].value_counts()
    with open(f"figures/{year}/device_distribution.txt", "w", encoding="utf-8") as f:
        f.write("Device Distribution:\n")
        for device, count in device_counts.items():
            f.write(f"{device}: {count}\n")


def analyze_tweet_temporal(year):
    """Analyze temporal patterns in tweets"""
    # Load tweet data
    os.makedirs(f"figures/{year}", exist_ok=True)

    # Process each month separately
    for month in range(1, 13):
        log(f"analyzing tweet temporal for {year} {month}")
        month_str = f"{month:02d}"
        parquet_files = glob.glob(f"youth_weibo_stat/{year}-{month_str}-*.parquet")
        if not parquet_files:
            log(f"No parquet files found for year {year} {month}")
            continue

        # Initialize hour counts
        hour_counts = np.zeros(24)

        # Process each file
        for file in parquet_files:
            df = pd.read_parquet(file, columns=["time_stamp"])
            # Convert timestamp to hour
            hours = pd.to_datetime(df["time_stamp"], unit="ms").dt.hour
            # Update hour counts
            for hour in range(24):
                hour_counts[hour] += (hours == hour).sum()

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(24), hour_counts)
        plt.title(f"Using Time Distribution - {month} ({year})")
        plt.xlabel("Time (24 hour clock)")
        plt.ylabel("Count")
        plt.savefig(
            f"figures/{year}/hour_distribution_{month:02d}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def analyze_tweet_content(year):
    """Analyze tweet content and generate word clouds"""
    # Process each month
    for month in range(1, 13):
        month_files = get_month_files(year, month)
        if not month_files:
            continue

        # Only read necessary columns
        needed_columns = ["weibo_content"]
        all_words = []

        for file in month_files:
            df = pd.read_parquet(file, columns=needed_columns)
            for content in df["weibo_content"].dropna():
                cleaned_content = sentence_cleaner(content)
                words = pseg.cut(cleaned_content)
                meaningful_words = [
                    word
                    for word, flag in words
                    if flag.startswith(("n", "v")) and len(word) > 1
                ]
                all_words.extend(meaningful_words)

        if not all_words:
            continue

        # Generate word cloud
        word_freq = Counter(all_words)
        wordcloud = WordCloud(
            font_path="/gpfs/share/home/2401111059/.fonts/simhei/SimHei.ttf",
            width=800,
            height=400,
            background_color="white",
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {month} ({year})")
        plt.savefig(
            f"figures/{year}/wordcloud_{month:02d}.pdf", bbox_inches="tight", dpi=300
        )
        plt.close()


def analyze_tweet_profile_merge(year):
    """Merge tweet analysis with user profiles"""
    # Load data
    log(f"analyzing tweet profile merge for {year}")
    parquet_files = glob.glob(f"youth_weibo_stat/{year}-*.parquet")
    if not parquet_files:
        log(f"No parquet files found for year {year}")
        return

    # Load profiles once
    profiles_df = pd.read_parquet("merged_profiles/merged_user_profiles.parquet")

    # Initialize counters
    tweet_counts = Counter()
    late_night_counts = Counter()
    total_tweets = Counter()

    # Process files in chunks
    for file in parquet_files:
        df = pd.read_parquet(file, columns=["user_id", "time_stamp"])

        # Update tweet counts
        tweet_counts.update(df["user_id"].value_counts().to_dict())

        # Update late night counts
        hours = pd.to_datetime(df["time_stamp"], unit="ms").dt.hour
        late_night_mask = hours.between(0, 8)
        late_night_counts.update(
            df.loc[late_night_mask, "user_id"].value_counts().to_dict()
        )
        total_tweets.update(df["user_id"].value_counts().to_dict())

    # Convert counters to DataFrame
    tweet_stats = pd.DataFrame(
        {
            "user_id": list(tweet_counts.keys()),
            "tweet_count": list(tweet_counts.values()),
            "late_night_count": [
                late_night_counts.get(uid, 0) for uid in tweet_counts.keys()
            ],
            "total_tweets": [total_tweets.get(uid, 0) for uid in tweet_counts.keys()],
        }
    )

    # Calculate late night ratio
    tweet_stats["late_night_ratio"] = (
        tweet_stats["late_night_count"] / tweet_stats["total_tweets"]
    )

    # Merge with profiles
    merged_df = profiles_df.merge(tweet_stats, on="user_id", how="left")
    merged_df["tweet_count"] = merged_df["tweet_count"].fillna(0)
    merged_df["late_night_ratio"] = merged_df["late_night_ratio"].fillna(0)

    # Gender analysis
    gender_tweet_stats = merged_df.groupby("gender")["tweet_count"].agg(
        ["mean", "std", "count"]
    )
    log("\nTweet Count by Gender:")
    log(str(gender_tweet_stats))

    # Region analysis
    region_tweet_stats = merged_df.groupby("region")["tweet_count"].agg(
        ["mean", "std", "count"]
    )
    log("\nTweet Count by Region:")
    log(str(region_tweet_stats))

    # Gender analysis for late night tweets
    gender_late_night = merged_df.groupby("gender")["late_night_ratio"].mean()
    log("\nLate Night Tweet Ratio by Gender:")
    log(str(gender_late_night))

    # Region analysis for late night tweets
    region_late_night = merged_df.groupby("region")["late_night_ratio"].mean()
    log("\nLate Night Tweet Ratio by Region:")
    log(str(region_late_night))

    # Create visualizations
    # 1. Tweet count by gender
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged_df, x="gender", y="tweet_count")
    plt.title("Tweet Count Distribution by Gender")
    plt.savefig(
        f"figures/{year}/tweet_count_by_gender.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()

    # 2. Tweet count by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_df, x="region", y="tweet_count")
    plt.title("Tweet Count Distribution by Region")
    plt.xticks(rotation=45)
    plt.savefig(
        f"figures/{year}/tweet_count_by_region.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()

    # 3. Late night tweet ratio by gender
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged_df, x="gender", y="late_night_ratio")
    plt.title("Late Night Tweet Ratio by Gender")
    plt.savefig(
        f"figures/{year}/late_night_by_gender.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()

    # 4. Late night tweet ratio by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=merged_df, x="region", y="late_night_ratio")
    plt.title("Late Night Tweet Ratio by Region")
    plt.xticks(rotation=45)
    plt.savefig(
        f"figures/{year}/late_night_by_region.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()


def analyze_all(year):
    analyze_profiles()
    analyze_tweet_basic(year)
    analyze_tweet_temporal(year)
    # analyze_tweet_content(year)
    analyze_tweet_profile_merge(year)


if __name__ == "__main__":
    fire.Fire(
        {
            "merge_profile": merge_user_profiles,
            "analyze_profile": analyze_profiles,
            "analyze_tweet_basic": analyze_tweet_basic,
            "analyze_tweet_temporal": analyze_tweet_temporal,
            "analyze_tweet_content": analyze_tweet_content,
            "analyze_tweet_profile": analyze_tweet_profile_merge,
            "analyze_all": analyze_all,
        }
    )
