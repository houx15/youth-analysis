"""
从"merged_profiles/merged_user_profiles.parquet"中获取新闻账号的id
nick_name列
导入configs/cn_news_sources_lists.py
分别看nick_name是否属于其中一个列表
如果是，就把user id存下来，存为一个json，configs/news_user_ids.json
按类别分类存储
"""

import pandas as pd
import json
import os
from configs.cn_news_sources_lists import (
    CENTRAL_THEORY_WEBSITE,
    CENTRAL_NEWS,
    ORG_NEWS,
    OTHER_ORG_NEWS,
    GOV_RELEASE_WEIBO,
    LOCAL_NEWS_WEBSITES,
    LOCAL_NEWS_UNITS,
    PROV_RELEASE_WEIBO,
)


def get_all_news_names():
    """合并所有新闻源列表为一个集合"""
    all_lists = [
        CENTRAL_THEORY_WEBSITE,
        CENTRAL_NEWS,
        ORG_NEWS,
        OTHER_ORG_NEWS,
        GOV_RELEASE_WEIBO,
        LOCAL_NEWS_WEBSITES,
        LOCAL_NEWS_UNITS,
        PROV_RELEASE_WEIBO,
    ]

    # 合并所有列表为一个集合
    news_names = set()
    for news_list in all_lists:
        news_names.update(news_list)

    return news_names


def load_news_user_ids(
    json_file="configs/news_user_ids.json", categories=None, merge=True
):
    """
    从JSON文件中加载新闻账号的用户ID

    Args:
        json_file: JSON文件路径
        categories: 要加载的类别列表，如果为None则加载所有类别
        merge: 是否合并所有类别的ID（True返回所有ID的列表，False返回分类字典）

    Returns:
        如果merge=True，返回set类型的user_id集合
        如果merge=False，返回dict类型 {category: [user_ids]}
    """
    if not os.path.exists(json_file):
        print(f"警告: 未找到新闻账号ID文件 {json_file}")
        return set() if merge else {}

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if merge:
            # 合并所有类别的ID
            all_ids = set()
            categories_to_load = categories if categories else data.keys()

            for category in categories_to_load:
                if category in data:
                    all_ids.update(data[category])

            print(
                f"从 {json_file} 加载了 {len(all_ids)} 个新闻账号ID (来自 {len(categories_to_load)} 个类别)"
            )
            return all_ids
        else:
            # 返回分类字典
            if categories:
                return {k: v for k, v in data.items() if k in categories}
            return data

    except Exception as e:
        print(f"读取新闻账号ID文件失败: {e}")
        return set() if merge else {}


def get_news_user_ids():
    """从 merged_profiles.parquet 中提取新闻账号的用户 ID，按类别分类存储"""
    # 输入文件路径
    input_file = "merged_profiles/merged_user_profiles.parquet"

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 未找到文件 {input_file}")
        return

    print(f"正在读取文件: {input_file}")

    # 读取 parquet 文件
    try:
        df = pd.read_parquet(input_file)
        print(f"成功读取 {len(df)} 条用户记录")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 检查必要的列是否存在
    if "nick_name" not in df.columns:
        print("错误: 文件中没有 'nick_name' 列")
        return

    if "user_id" not in df.columns:
        print("错误: 文件中没有 'user_id' 列")
        return

    # 定义新闻类别和对应的列表
    news_categories = {
        "central_theory_website": CENTRAL_THEORY_WEBSITE,
        "central_news": CENTRAL_NEWS,
        "org_news": ORG_NEWS,
        "other_org_news": OTHER_ORG_NEWS,
        "gov_release_weibo": GOV_RELEASE_WEIBO,
        "local_news_websites": LOCAL_NEWS_WEBSITES,
        "local_news_units": LOCAL_NEWS_UNITS,
        "prov_release_weibo": PROV_RELEASE_WEIBO,
    }

    # 过滤出新闻账号
    # 移除 nick_name 为空的值
    df = df.dropna(subset=["nick_name"])
    df["nick_name"] = df["nick_name"].astype(str).str.strip()

    # 用于存储每个类别的用户 ID
    categorized_user_ids = {}

    # 遍历每个类别
    for category_name, news_list in news_categories.items():
        # 判断该行的 nick_name 是否属于当前类别
        df["is_in_category"] = df["nick_name"].isin(news_list)

        # 筛选出当前类别的用户
        category_users = df[df["is_in_category"]].copy()

        # 提取用户 ID 并转换为字符串列表
        category_ids = [str(uid) for uid in category_users["user_id"].tolist()]

        categorized_user_ids[category_name] = category_ids

        print(f"{category_name}: 找到 {len(category_ids)} 个账号")

    # 计算统计信息
    total_news_users = len(df[df["nick_name"].isin(get_all_news_names())])

    # 确保输出目录存在
    output_dir = "configs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存为 JSON 文件
    output_file = "configs/news_user_ids.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(categorized_user_ids, f, ensure_ascii=False, indent=2)

    # 计算总数和去重后的总数
    all_ids = []
    for ids in categorized_user_ids.values():
        all_ids.extend(ids)
    unique_ids = list(set(all_ids))

    print(f"\n成功保存到 {output_file}")
    print("\n新闻账号统计:")
    print(f"  - 总用户数: {len(df)}")
    print(f"  - 新闻账号数 (去重前): {len(all_ids)}")
    print(f"  - 新闻账号数 (去重后): {len(unique_ids)}")
    print(f"  - 匹配率: {len(unique_ids)/len(df)*100:.2f}%")

    # 显示各类别统计
    print("\n各类别统计:")
    for category_name, ids in categorized_user_ids.items():
        print(f"  - {category_name}: {len(ids)} 个账号")


if __name__ == "__main__":
    get_news_user_ids()
