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

# 链接匹配正则表达式（用于识别URL）
URL_PATTERN = re.compile(
    r"(https?://[^\s]+|t\.cn/[^\s]+|http://[^\s]+|www\.[^\s]+)", re.IGNORECASE
)


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


def extract_original_content(content: str) -> str:
    """
    从微博内容中提取原始内容，去除转发部分
    使用//分割，保留第0个部分（原始内容）

    Args:
        content: 微博内容

    Returns:
        提取的原始内容
    """
    if pd.isna(content) or not content:
        return ""

    content_str = str(content)
    # 使用//分割，保留第一部分（原始内容）
    parts = content_str.split("//")
    if parts:
        return parts[0].strip()
    return content_str


def keyword_in_url(content: str) -> bool:
    """
    检查关键词是否出现在链接中

    Args:
        content: 微博内容

    Returns:
        如果关键词出现在链接中返回True，否则返回False
    """
    if pd.isna(content) or not content:
        return False

    content_str = str(content)
    # 查找所有链接
    urls = URL_PATTERN.findall(content_str)

    if not urls:
        return False

    # 检查链接中是否包含关键词（只检查链接部分，不检查整个内容）
    for url in urls:
        url_lower = url.lower()
        # 检查这个链接中是否包含任何关键词
        for keyword in AI_KEYWORDS:
            keyword_lower = keyword.lower()
            # 确保关键词是作为完整单词或子串出现在URL中
            if keyword_lower in url_lower:
                return True

    return False


def clean_single_date(date_str: str) -> int:
    """
    清洗单个日期的数据，去除误匹配的记录

    Args:
        date_str: 日期字符串，格式为 yyyy-mm-dd

    Returns:
        清洗后保留的记录数量
    """
    start_time = time.time()

    input_file = os.path.join(OUTPUT_DIR, f"{date_str}.parquet")

    if not os.path.exists(input_file):
        log(f"未找到 {date_str} 的数据文件: {input_file}")
        return 0

    try:
        log(f"正在清洗: {date_str}")

        # 读取数据
        df = pd.read_parquet(input_file, engine="fastparquet")
        original_count = len(df)

        if original_count == 0:
            log(f"  {date_str}: 数据为空，跳过清洗")
            return 0

        # 1. 提取原始内容（去除转发部分）
        df["original_content"] = df["weibo_content"].apply(extract_original_content)

        # 2. 检查原始内容中是否包含关键词（去除转发后的内容）
        mask_has_keyword = (
            df["original_content"]
            .astype(str)
            .str.lower()
            .str.contains(KEYWORD_PATTERN, na=False, regex=True)
        )

        # 3. 检查关键词是否在链接中（如果是，则剔除）
        mask_keyword_in_url = df["weibo_content"].apply(keyword_in_url)

        # 最终筛选：原始内容包含关键词 且 关键词不在链接中
        final_mask = mask_has_keyword & (~mask_keyword_in_url)

        # 筛选数据
        cleaned_df = df[final_mask].copy()

        # 删除临时列
        cleaned_df = cleaned_df.drop(columns=["original_content"], errors="ignore")

        removed_count = original_count - len(cleaned_df)

        if len(cleaned_df) == 0:
            log(f"  {date_str}: 清洗后无有效记录，删除原文件")
            os.remove(input_file)
            elapsed_time = int(time.time() - start_time)
            log(
                f"  {date_str}: 原始 {original_count} 条，剔除 {removed_count} 条，耗时 {elapsed_time} 秒"
            )
            return 0

        # 保存清洗后的数据（覆盖原文件）
        cleaned_df.to_parquet(
            input_file, engine="fastparquet", index=False, compression="gzip"
        )

        elapsed_time = int(time.time() - start_time)
        log(
            f"  {date_str}: 原始 {original_count} 条，剔除 {removed_count} 条，保留 {len(cleaned_df)} 条，耗时 {elapsed_time} 秒"
        )

        return len(cleaned_df)

    except Exception as e:
        error_msg = f"清洗 {date_str} 时出错: {e}"
        log(error_msg)
        import traceback

        traceback.print_exc()
        return 0


def clean_date_range(start_date: str, end_date: str):
    """
    清洗日期范围内的所有数据

    Args:
        start_date: 开始日期，格式为 yyyy-mm-dd
        end_date: 结束日期，格式为 yyyy-mm-dd
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current_date = start
    total_original = 0
    total_cleaned = 0

    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        cleaned_count = clean_single_date(date_str)
        total_cleaned += cleaned_count
        current_date += timedelta(days=1)

    log(f"\n清洗完成！共保留 {total_cleaned} 条记录")


def clean_year(year: int, mode: int = 0):
    """
    清洗指定年份的数据

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

    log(f"开始清洗 {year} 年，模式 {'下半年' if mode == 1 else '上半年'}")
    log(f"日期范围: {start_str} 到 {end_str}")

    clean_date_range(start_str, end_str)


def main(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
    mode: int = 0,
    clean: bool = False,
):
    """
    主函数

    Args:
        start_date: 开始日期，格式为 yyyy-mm-dd（与end_date一起使用）
        end_date: 结束日期，格式为 yyyy-mm-dd（与start_date一起使用）
        year: 年份（与mode一起使用）
        mode: 处理模式，0=上半年，1=下半年（默认：0，与year一起使用）
        clean: 是否执行清洗模式（默认：False）
    """
    if clean:
        # 清洗模式
        if start_date and end_date:
            clean_date_range(start_date, end_date)
        elif year:
            clean_year(year, mode)
        else:
            log("清洗模式：请提供 start_date 和 end_date，或者提供 year 参数")
            log("示例:")
            log(
                "  python ai_content_extractor.py --clean --start_date 2024-01-01 --end_date 2024-12-31"
            )
            log("  python ai_content_extractor.py --clean --year 2024 --mode 0")
    else:
        # 正常提取模式
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
            log("\n清洗模式:")
            log("  python ai_content_extractor.py --clean --year 2024 --mode 0")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
