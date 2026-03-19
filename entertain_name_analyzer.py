"""
娱乐人名提及分析脚本

分析每个娱乐人名在男性/女性微博中的提及情况：
1. 每个名字被男/女用户提及的帖子数
2. 提及该名字的帖子的平均字符数（清理后）
3. 提及该名字的帖子中，字符数>10的比例（过滤极短提及）

输出: analysis_results/entertain_name_mentions_{year}.parquet
  列: name, gender, post_count, avg_char_count, ratio_gt_10_chars

用法:
  python entertain_name_analyzer.py analyze 2020
  python entertain_name_analyzer.py analyze 2020 --force_recalculate
"""

import os
import pandas as pd
import numpy as np
import glob
import fire
from tqdm import tqdm
from collections import defaultdict

DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "analysis_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 停用词和清理函数
try:
    from utils.utils import sentence_cleaner
except ImportError:

    def sentence_cleaner(sentence, src_type="weibo"):
        import re

        if pd.isna(sentence) or sentence == "":
            return ""
        sentence = str(sentence)
        sentence = sentence.replace("\u201c", "").replace("\u201d", "")
        sentence = sentence.replace("\u2026", "")
        sentence = sentence.replace("点击链接查看更多->", "")
        results = re.compile(r"[a-zA-Z0-9.?/&=:_%,-~#《》]", re.S)
        sentence = re.sub(results, "", sentence)
        results2 = re.compile(r"[//@].*?[:]", re.S)
        sentence = re.sub(results2, "", sentence)
        sentence = sentence.replace("\n", " ").strip()
        return sentence


def load_entertain_names(year):
    """加载娱乐人名列表"""
    # 尝试多个路径
    for dir_name in ["configs", "wordlists"]:
        path = os.path.join(dir_name, f"entertainment_nouns_{year}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
            print(f"已加载 {len(names)} 个娱乐人名 (from {path})")
            return names

    raise FileNotFoundError(
        f"未找到娱乐人名文件: configs/entertainment_nouns_{year}.txt"
    )


def build_name_index(names):
    """按字符长度对名字分组，用于高效滑动窗口匹配

    Returns:
        dict: {char_length: set_of_names}
    """
    index = defaultdict(set)
    for name in names:
        index[len(name)].add(name)
    print(f"  名字长度分布: {', '.join(f'{k}字={len(v)}个' for k, v in sorted(index.items()))}")
    return dict(index)


def find_names_in_text(text, name_index):
    """在文本中查找所有出现的娱乐人名（滑动窗口）

    Args:
        text: 清理后的文本
        name_index: {char_length: set_of_names}

    Returns:
        set: 在文本中出现的名字集合
    """
    if not text:
        return set()

    found = set()
    text_len = len(text)

    for name_len, name_set in name_index.items():
        if name_len > text_len:
            continue
        for i in range(text_len - name_len + 1):
            substr = text[i : i + name_len]
            if substr in name_set:
                found.add(substr)

    return found


def analyze_name_mentions(year, force_recalculate=False):
    """分析每个娱乐人名在男/女微博中的提及情况"""
    print(f"\n{'='*60}")
    print(f"分析 {year} 年娱乐人名提及情况")
    print(f"{'='*60}")

    output_file = os.path.join(
        OUTPUT_DIR, f"entertain_name_mentions_{year}.parquet"
    )

    if not force_recalculate and os.path.exists(output_file):
        print(f"结果文件已存在: {output_file}")
        print("如需重新计算，请设置 force_recalculate=True")
        df = pd.read_parquet(output_file, engine="fastparquet")
        print(f"共 {len(df)} 条记录")
        return

    # 加载人名列表
    names = load_entertain_names(year)
    name_index = build_name_index(names)

    # 扫描所有parquet文件
    year_dir = os.path.join(DATA_DIR, str(year))
    if not os.path.exists(year_dir):
        print(f"未找到 {year} 年的数据目录: {year_dir}")
        return

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        print(f"未找到 {year} 年的parquet文件")
        return

    print(f"找到 {len(parquet_files)} 个文件")

    # 累加器: (name, gender) -> [post_count, total_char_count, gt_10_count]
    stats = defaultdict(lambda: [0, 0, 0])

    total_posts = 0
    matched_posts = 0

    for file_path in tqdm(parquet_files, desc="分析人名提及"):
        try:
            df = pd.read_parquet(
                file_path, columns=["weibo_content", "gender"]
            )
            df = df[df["gender"].notna()]

            if len(df) == 0:
                continue

            total_posts += len(df)

            for content, gender in zip(df["weibo_content"], df["gender"]):
                cleaned = sentence_cleaner(content)
                if not cleaned or cleaned == "转发微博":
                    continue

                char_count = len(cleaned)
                found_names = find_names_in_text(cleaned, name_index)

                if found_names:
                    matched_posts += 1
                    for name in found_names:
                        key = (name, gender)
                        stats[key][0] += 1  # post_count
                        stats[key][1] += char_count  # total_char_count
                        if char_count > 10:
                            stats[key][2] += 1  # gt_10_count

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    print(f"\n扫描完成:")
    print(f"  总帖子数: {total_posts:,}")
    print(f"  提及娱乐人名的帖子数: {matched_posts:,} ({matched_posts/total_posts*100:.2f}%)")

    # 构建结果DataFrame
    rows = []
    for (name, gender), (post_count, total_chars, gt_10_count) in stats.items():
        rows.append(
            {
                "name": name,
                "gender": gender,
                "post_count": post_count,
                "avg_char_count": total_chars / post_count,
                "ratio_gt_10_chars": gt_10_count / post_count,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        ["name", "gender"], ignore_index=True
    )

    # 保存
    result_df.to_parquet(output_file, engine="fastparquet", index=False)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n结果已保存: {output_file} ({len(result_df)} 行, {size_mb:.2f} MB)")

    # 打印Top 20热门人名
    print(f"\n{'='*60}")
    print("Top 20 被提及最多的娱乐人名:")
    print(f"{'='*60}")

    name_total = (
        result_df.groupby("name")["post_count"]
        .sum()
        .sort_values(ascending=False)
    )
    for i, (name, count) in enumerate(name_total.head(20).items()):
        name_data = result_df[result_df["name"] == name]
        gender_counts = dict(zip(name_data["gender"], name_data["post_count"]))
        m_count = gender_counts.get("m", gender_counts.get("男", 0))
        f_count = gender_counts.get("f", gender_counts.get("女", 0))
        ratio = m_count / f_count if f_count > 0 else float("inf")
        print(
            f"  {i+1:2d}. {name}: {count:,} 次 "
            f"(男 {m_count:,}, 女 {f_count:,}, 男/女={ratio:.2f})"
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "analyze": analyze_name_mentions,
        }
    )
