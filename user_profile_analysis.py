from configs.configs import *
import pandas as pd
import glob
from collections import Counter
import os
import json
import time

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


def merge_user_profiles():
    """
    Merge user profiles from all parquet files, keeping the latest values for most fields
    and using mode (most frequent value) for location-related fields.
    Also adds birthday and region information from demographic data.
    """
    # Get all parquet files
    parquet_files = glob.glob("youth_profile_data/*.parquet")

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
    all_profiles["timestamp"] = pd.to_datetime(all_profiles["timestamp"], unit="ms")

    # Group by user_id
    grouped = all_profiles.groupby("user_id")

    # Initialize lists to store results
    merged_profiles = []

    # Process each user's profiles
    for user_id, group in grouped:
        # Sort by timestamp to get the latest values
        group = group.sort_values("timestamp", ascending=False)

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
    if "timestamp" in merged_df.columns:
        merged_df = merged_df.drop("timestamp", axis=1)

    # Create output directory if it doesn't exist
    os.makedirs("merged_profiles", exist_ok=True)

    # Save to parquet
    output_path = "merged_profiles/merged_user_profiles.parquet"
    merged_df.to_parquet(output_path, engine="fastparquet", index=False)
    print(f"Saved merged profiles to {output_path}")
    print(f"Total unique users: {len(merged_df)}")
    print(f"Users with demographic data: {merged_df['birthday'].notna().sum()}")


if __name__ == "__main__":
    # unzip_year_huati_bang_files(2023)
    merge_user_profiles()
