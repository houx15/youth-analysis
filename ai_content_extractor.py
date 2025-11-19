"""
从清洗后的微博数据中提取包含AI关键词的内容
筛选条件：weibo_content转换为小写后，至少包含AI_KEYWORDS中的一个词
"""

import os
import re
import glob
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

AI_KEYWORDS = [
    "ai",
    "人工智能",
    "transformer",
    "agi",
    "llm",
    "大模型",
    "大语言模型",
    "openai",
    "deepmind",
    "anthropic",
    "nvidia",
    "英伟达",
    "cursor",
    "mistral",
    "perplexity",
    "gpt",
    "claude",
    "gemini",
    "grok",
    "llama",
    "moonshot",
    "月之暗面",
    "deepseek",
    "sora",
    "通义千问",
    "火山引擎",
    "豆包",
    "腾讯元宝",
    "可灵",
    "qwen",
    "qwq",
    "doubao",
    "kimi",
    "stable diffusion",
    "midjourney",
    "codex",
    "characterai",
    "replika",
    "chatgpt",
    "openrouter",
    "辛顿",
    "copilot",
    "奥特曼",
    "李飞飞",
    "andrej karpathy",
    "lecun",
    "hinton",
    "杨立坤",
    "rlhf",
    "agent",
    "vibe coding",
    "RAG",
    "prompt",
    "即梦",
    "智谱",
    "文心一言",
    "黄仁勋",
    "梁文锋",
    "智能体",
    "提示词",
    "GLM",
]

# 输入数据目录（basic_text_extractor处理后的数据）
INPUT_DIR = os.path.expanduser("~/cleaned_weibo_data")
# 输出数据目录
OUTPUT_DIR = "ai_weibo_text"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def log(text, lid=None):
    """记录日志"""
    os.makedirs("logs", exist_ok=True)
    output = (
        f"logs/ai_content_extractor_log_{lid}.txt"
        if lid is not None
        else "logs/ai_content_extractor_log.txt"
    )
    with open(output, "a", encoding="utf-8") as f:
        f.write(f"{text}\n")
    # 同时输出到控制台
    print(text)


def build_keyword_pattern():
    """
    构建正则表达式模式字符串，用于高效匹配所有AI关键词
    转义特殊字符并构建 OR 模式
    """
    # 转义特殊字符并转换为小写
    escaped_keywords = [re.escape(kw.lower()) for kw in AI_KEYWORDS]
    # 使用 | 连接所有关键词
    pattern = "|".join(escaped_keywords)
    return pattern


# 预编译正则表达式模式字符串（全局变量，只编译一次）
KEYWORD_PATTERN = build_keyword_pattern()


def process_single_temp_file(
    temp_file_path: str, date_str: str
) -> tuple[pd.DataFrame, int]:
    """
    处理单个临时文件，返回筛选后的DataFrame和记录数

    Args:
        temp_file_path: 临时文件路径
        date_str: 日期字符串

    Returns:
        (筛选后的DataFrame, 记录数)
    """
    try:
        # 定义需要读取的列（一次性读取，避免重复读取文件）
        columns_to_read = [
            "weibo_id",
            "user_id",
            "is_retweet",
            "weibo_content",
            "zhuan",
            "ping",
            "zan",
            "time_stamp",
            "r_weibo_content",
            "lat",
            "lon",
        ]

        # 一次性读取所有需要的列（包括weibo_content用于筛选）
        df = pd.read_parquet(
            temp_file_path, columns=columns_to_read, engine="fastparquet"
        )

        # 如果weibo_content列不存在，直接返回
        if "weibo_content" not in df.columns:
            return pd.DataFrame(), 0

        # 使用向量化的str.contains()方法筛选（比apply快得多）
        mask = (
            df["weibo_content"]
            .astype(str)
            .str.lower()
            .str.contains(KEYWORD_PATTERN, na=False, regex=True)
        )

        # 如果没有匹配的记录，直接返回
        if not mask.any():
            return pd.DataFrame(), 0

        # 筛选匹配的记录
        filtered_df = df[mask].copy()

        # 确保所有列都存在（如果不存在则填充空值）
        for col in columns_to_read:
            if col not in filtered_df.columns:
                filtered_df[col] = ""

        # 只保留需要的列
        result_df = filtered_df[columns_to_read].copy()

        return result_df, len(result_df)

    except Exception as e:
        error_msg = f"  处理文件 {temp_file_path} 时出错: {e}"
        log(error_msg)
        import traceback

        traceback.print_exc()
        return pd.DataFrame(), 0


def process_single_date(date_str: str) -> int:
    """
    处理单个日期的所有临时文件（.temp_x格式）

    Args:
        date_str: 日期字符串，格式为 yyyy-mm-dd

    Returns:
        筛选出的记录数量
    """
    start_time = time.time()

    # 查找该日期的所有临时文件
    pattern = os.path.join(INPUT_DIR, f"{date_str}.parquet.temp_*")
    temp_files = sorted(glob.glob(pattern))

    if not temp_files:
        log(f"未找到 {date_str} 的临时文件")
        return 0

    log(f"正在处理: {date_str} (找到 {len(temp_files)} 个临时文件)")

    all_results = []
    total_records = 0

    # 处理每个临时文件
    for temp_file in temp_files:
        result_df, count = process_single_temp_file(temp_file, date_str)
        if count > 0:
            all_results.append(result_df)
            total_records += count
            log(f"  {os.path.basename(temp_file)}: 筛选出 {count} 条记录")

    if total_records == 0:
        elapsed_time = int(time.time() - start_time)
        log(f"  {date_str}: 未找到包含AI关键词的记录，耗时 {elapsed_time} 秒")
        return 0

    # 合并所有结果
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # 保存到parquet文件
        output_file = os.path.join(OUTPUT_DIR, f"{date_str}.parquet")
        final_df.to_parquet(
            output_file, engine="fastparquet", index=False, compression="gzip"
        )

        elapsed_time = int(time.time() - start_time)
        log(
            f"  {date_str}: 共筛选出 {total_records} 条记录，已保存到 {output_file}，耗时 {elapsed_time} 秒"
        )

    return total_records


def process_date_range(start_date: str, end_date: str):
    """
    处理日期范围内的所有数据

    Args:
        start_date: 开始日期，格式为 yyyy-mm-dd
        end_date: 结束日期，格式为 yyyy-mm-dd
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current_date = start
    total_records = 0

    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        records = process_single_date(date_str)
        total_records += records
        current_date += timedelta(days=1)

    log(f"\n处理完成！共筛选出 {total_records} 条记录")


def process_year(year: int, mode: int = 0):
    """
    处理指定年份的数据

    Args:
        year: 年份
        mode: 处理模式，0=上半年，1=下半年（默认：0）
    """
    start_date_options = [datetime(year, 1, 1), datetime(year, 7, 1)]
    end_date_options = [datetime(year, 6, 30), datetime(year, 12, 31)]
    start_date = start_date_options[mode]
    end_date = end_date_options[mode]

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    log(f"开始处理 {year} 年，模式 {'下半年' if mode == 1 else '上半年'}")
    log(f"日期范围: {start_str} 到 {end_str}")

    process_date_range(start_str, end_str)


def main(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
    mode: int = 0,
):
    """
    主函数

    Args:
        start_date: 开始日期，格式为 yyyy-mm-dd（与end_date一起使用）
        end_date: 结束日期，格式为 yyyy-mm-dd（与start_date一起使用）
        year: 年份（与mode一起使用）
        mode: 处理模式，0=上半年，1=下半年（默认：0，与year一起使用）
    """
    if start_date and end_date:
        # 使用日期范围模式
        process_date_range(start_date, end_date)
    elif year:
        # 使用年份模式
        process_year(year, mode)
    else:
        log("请提供 start_date 和 end_date，或者提供 year 参数")
        log("示例:")
        log(
            "  python ai_content_extractor.py --start_date 2024-01-01 --end_date 2024-12-31"
        )
        log("  python ai_content_extractor.py --year 2024 --mode 0")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
