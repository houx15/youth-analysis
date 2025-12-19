"""
从"merged_profiles/merged_user_profiles.parquet"中获取娱乐账号的id
通过verified_reason列包含特定关键词来识别
关键词：演员，歌手，艺人，主唱，偶像，乐队，歌星，唱作人，导演，童星，音乐人
如果是，就把user id存下来，存为一个json，configs/entertain_user_ids.json
按关键词分类存储
"""

import pandas as pd
import json
import os


# 定义娱乐行业关键词列表
ENTERTAIN_KEYWORDS = [
    "演员",
    "歌手",
    "艺人",
    "主唱",
    "偶像",
    "乐队",
    "歌星",
    "唱作人",
    "导演",
    "童星",
    "音乐人",
]


def load_entertain_user_ids(
    json_file="configs/entertain_user_ids.json", keywords=None, merge=True
):
    """
    从JSON文件中加载娱乐账号的用户ID

    Args:
        json_file: JSON文件路径
        keywords: 要加载的关键词列表，如果为None则加载所有关键词
        merge: 是否合并所有关键词的ID（True返回所有ID的列表，False返回分类字典）

    Returns:
        如果merge=True，返回set类型的user_id集合
        如果merge=False，返回dict类型 {keyword: [user_ids]}
    """
    if not os.path.exists(json_file):
        print(f"警告: 未找到娱乐账号ID文件 {json_file}")
        return set() if merge else {}

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if merge:
            # 合并所有关键词的ID
            all_ids = set()
            keywords_to_load = keywords if keywords else data.keys()

            for keyword in keywords_to_load:
                if keyword in data:
                    all_ids.update(data[keyword])

            print(
                f"从 {json_file} 加载了 {len(all_ids)} 个娱乐账号ID (来自 {len(keywords_to_load)} 个关键词)"
            )
            return all_ids
        else:
            # 返回分类字典
            if keywords:
                return {k: v for k, v in data.items() if k in keywords}
            return data

    except Exception as e:
        print(f"读取娱乐账号ID文件失败: {e}")
        return set() if merge else {}


def get_entertain_user_ids():
    """从 merged_profiles.parquet 中提取娱乐账号的用户 ID，按关键词分类存储"""
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
    if "verified_reason" not in df.columns:
        print("错误: 文件中没有 'verified_reason' 列")
        return

    if "user_id" not in df.columns:
        print("错误: 文件中没有 'user_id' 列")
        return

    # 过滤出有verified_reason的记录
    # 移除 verified_reason 为空的值
    df = df.dropna(subset=["verified_reason"])
    df["verified_reason"] = df["verified_reason"].astype(str).str.strip()

    # 用于存储每个关键词的用户 ID
    categorized_user_ids = {}

    # 遍历每个关键词
    for keyword in ENTERTAIN_KEYWORDS:
        # 判断该行的 verified_reason 是否包含当前关键词
        df["contains_keyword"] = df["verified_reason"].str.contains(
            keyword, na=False, regex=False
        )

        # 筛选出包含当前关键词的用户
        keyword_users = df[df["contains_keyword"]].copy()

        # 提取用户 ID 并转换为字符串列表
        keyword_ids = [str(uid) for uid in keyword_users["user_id"].tolist()]

        categorized_user_ids[keyword] = keyword_ids

        print(f"{keyword}: 找到 {len(keyword_ids)} 个账号")

    # 计算所有匹配娱乐关键词的用户（去重）
    # 创建一个包含所有关键词的正则表达式
    all_keywords_pattern = "|".join(ENTERTAIN_KEYWORDS)
    df["is_entertain"] = df["verified_reason"].str.contains(
        all_keywords_pattern, na=False, regex=True
    )
    total_entertain_users = df[df["is_entertain"]].copy()

    # 确保输出目录存在
    output_dir = "configs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存为 JSON 文件
    output_file = "configs/entertain_user_ids.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(categorized_user_ids, f, ensure_ascii=False, indent=2)

    # 计算总数和去重后的总数
    all_ids = []
    for ids in categorized_user_ids.values():
        all_ids.extend(ids)
    unique_ids = list(set(all_ids))

    print(f"\n成功保存到 {output_file}")
    print("\n娱乐账号统计:")
    print(f"  - 总用户数: {len(df)}")
    print(f"  - 娱乐账号数 (去重前): {len(all_ids)}")
    print(f"  - 娱乐账号数 (去重后): {len(unique_ids)}")
    print(f"  - 匹配率: {len(unique_ids)/len(df)*100:.2f}%")

    # 显示各关键词统计
    print("\n各关键词统计:")
    for keyword, ids in categorized_user_ids.items():
        print(f"  - {keyword}: {len(ids)} 个账号")


if __name__ == "__main__":
    get_entertain_user_ids()

