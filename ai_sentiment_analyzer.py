"""
AI相关内容情感分析脚本

从ai_content_extractor.py提取的AI相关微博数据中随机抽取样本，
使用大模型API分析用户对AI技术的情感倾向。

使用方法:
    # 第一步：采样微博数据
    python ai_sentiment_analyzer.py sample --sample_size 1000

    # 第二步：分析采样后的数据
    python ai_sentiment_analyzer.py analyze --sampled_file ai_attitudes/sampled_data/sampled_weibos_20240101_120000.parquet

    # 采样时指定输入文件
    python ai_sentiment_analyzer.py sample --input_file ai_attitudes/ai_weibo_text/2024-01-01.parquet

    # 采样时指定输出文件
    python ai_sentiment_analyzer.py sample --output_file my_sample.parquet

    # 分析时指定输出目录
    python ai_sentiment_analyzer.py analyze --sampled_file sample.parquet --output_dir my_results
"""

import json
import logging
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import fire

from configs.configs import OPENAI_API_KEY, OPENAI_BASE_URL

BASE_DIR = Path("ai_attitudes")
# 默认配置
DEFAULT_INPUT_DIR = BASE_DIR / "ai_weibo_text"
DEFAULT_OUTPUT_DIR = BASE_DIR / "ai_sentiment_results"
DEFAULT_SAMPLE_DIR = BASE_DIR / "sampled_data"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_MODEL = "kimi-k2-0905-preview"  # 可以根据需要修改
DEFAULT_DELAY = 0.1  # API调用间隔（秒）
DEFAULT_SAMPLED_FILE = DEFAULT_SAMPLE_DIR / "sampled_weibos.parquet"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AISentimentAnalyzer:
    """使用OpenAI API分析AI相关微博内容的情感倾向"""

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        初始化情感分析器

        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL（可选，默认从configs读取）
            model: 使用的模型名称
        """
        if not api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量，或在代码中直接配置API密钥")

        # 如果没有提供base_url，使用configs中的默认值
        if base_url is None:
            base_url = OPENAI_BASE_URL

        # 构建OpenAI客户端参数
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        logger.info(f"初始化情感分析器，使用模型: {model}, base_url: {base_url}")

    def _build_prompt(self, row: pd.Series) -> tuple[str, str]:
        """
        构建中文分析提示词

        Args:
            row: 微博数据行

        Returns:
            (system_prompt, user_content) 元组
        """

        # System prompt: 使用新的 prompt 内容
        system_prompt = """
你将阅读一段用户在微博上发布的文本。请分析该文本对AI技术的观点，为其分配一个意见标签。你的输出必须是一个 JSON 对象，只包含字段 "opinion"，取值为 -2、-1、0、1、2 或 "cannot tell"。
 
标签定义：
2 = 对AI技术表达强烈的积极态度，明确主张AI技术利大于弊，并认为其对社会带来显著正面影响（如明显提升生活便利性、改善健康与医疗、带来经济机会、提高学习和工作效率、改善安全性、为研究和创新提供帮助等，或表达对使用AI的依赖或信任）。
1 = 整体倾向正面，但态度温和或带有一定保留（表达支持但语气不强；认为“总体有益”但同时提到风险或局限；表达期待、兴趣或积极看法，但无强烈赞美）。
0 = 客观中立或难以判断倾向（同时提到利弊，但无明确倾向）。
-1 = 整体倾向负面，但态度温和或带有一定保留（表达担忧或反对但未完全否定AI；认为“有风险或者有弊端”但承认某些益处；表达谨慎、不安或负面看法，但无强烈否定）。
-2 = 对AI技术表达强烈的消极态度，明确主张AI技术弊大于利，并认为其对社会带来显著负面影响（如加剧失业、经济增长泡沫、隐私风险、边缘群体偏见、错误信息或谣言、安全威胁等，或表达对AI的抗拒或不信任）。
"cannot tell" = 文本未表达对AI的任何态度（如内容与AI无关，或未反映观点）。
请仅返回一个JSON对象，例如：
{
"opinion": 1
}
""".strip()

        weibo_content = str(row.get("weibo_content", "") or "")
        is_retweet = row.get("is_retweet", False)
        time_stamp = row.get("time_stamp", None)

        # 构建内容部分
        content_parts = []

        if weibo_content:
            content_parts.append(f"微博内容: {weibo_content}")

        # 将timestamp转为年月日时分秒
        if time_stamp:
            try:
                # 尝试将timestamp转换为数字
                if isinstance(time_stamp, str):
                    time_stamp = float(time_stamp)
                date_time = datetime.fromtimestamp(time_stamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                content_parts.append(f"发布时间: {date_time}")
            except (ValueError, TypeError, OSError):
                # 如果转换失败，使用原始值
                content_parts.append(f"发布时间: {time_stamp}")

        content_parts.append(
            f"是否转发: {'是 (//后面为转发内容)' if is_retweet else '否'}"
        )

        user_prompt = "\n".join(content_parts)

        return system_prompt, user_prompt

    def analyze_single(self, row: pd.Series, max_retries: int = 3) -> Dict[str, Any]:
        """
        分析单条微博的情感倾向（带重试逻辑）

        Args:
            row: 微博数据行
            max_retries: 最大重试次数（默认3次）

        Returns:
            分析结果字典，包含 sentiment/opinion 和 token 使用统计
        """
        system_prompt, user_prompt = self._build_prompt(row)
        weibo_id = str(row.get("weibo_id", "") or "")

        # 重试逻辑
        for attempt in range(max_retries + 1):  # 总共尝试 max_retries + 1 次
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                response_text = response.choices[0].message.content.strip()

                # 提取 token 使用统计
                usage = response.usage
                token_stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                    "cached_tokens": getattr(usage, "cached_tokens", 0) if usage else 0,
                }

                # 尝试提取JSON
                opinion = None
                try:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        analysis = json.loads(json_text)
                        opinion = analysis.get("opinion")
                except json.JSONDecodeError:
                    opinion = None

                # 成功返回结果
                if attempt > 0:
                    logger.info(f"weibo_id {weibo_id} 在第 {attempt + 1} 次尝试后成功")
                return {
                    "opinion": opinion,
                    **token_stats,
                }

            except Exception as e:
                # 如果是最后一次尝试，记录错误并返回失败结果
                if attempt == max_retries:
                    logger.error(
                        f"分析单条微博 weibo_id {weibo_id} 时出错（已重试 {max_retries} 次）: {e}"
                    )
                    return {
                        "opinion": None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cached_tokens": 0,
                    }
                else:
                    # 重试前等待，等待时间递增（指数退避）
                    wait_time = (attempt + 1) * 2  # 2秒, 4秒, 6秒...
                    logger.warning(
                        f"分析 weibo_id {weibo_id} 失败（第 {attempt + 1} 次尝试），{wait_time} 秒后重试: {e}"
                    )
                    time.sleep(wait_time)

        # 理论上不会到达这里，但为了安全起见
        return {
            "opinion": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }

    def analyze_batch(
        self,
        df: pd.DataFrame,
        delay: float = DEFAULT_DELAY,
    ) -> List[Dict[str, Any]]:
        """
        批量分析微博情感倾向

        Args:
            df: 包含微博数据的DataFrame
            delay: API调用间隔（秒）

        Returns:
            分析结果列表
        """
        logger.info(f"开始分析 {len(df)} 条微博的情感倾向...")

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="分析微博"):
            try:
                analysis = self.analyze_single(row)

                result = {
                    "weibo_id": row.get("weibo_id", ""),
                    "weibo_content": row.get("weibo_content", ""),
                    "is_retweet": row.get("is_retweet", False),
                    "time_stamp": row.get("time_stamp", ""),
                    "opinion": analysis["opinion"],
                    "prompt_tokens": analysis.get("prompt_tokens", 0),
                    "completion_tokens": analysis.get("completion_tokens", 0),
                    "total_tokens": analysis.get("total_tokens", 0),
                    "cached_tokens": analysis.get("cached_tokens", 0),
                }
                results.append(result)

                if len(results) % 50 == 0:
                    logger.info(f"已分析 {len(results)}/{len(df)} 条微博")

                # 添加延迟以避免API速率限制
                time.sleep(delay)

            except Exception as e:
                logger.error(f"分析索引 {idx} 的微博时出错: {e}")
                results.append(
                    {
                        "weibo_id": row.get("weibo_id", ""),
                        "weibo_content": row.get("weibo_content", ""),
                        "is_retweet": row.get("is_retweet", False),
                        "time_stamp": row.get("time_stamp", ""),
                        "opinion": None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cached_tokens": 0,
                    }
                )

        logger.info(f"分析完成: 共 {len(results)} 条结果")
        return results


def load_parquet_files(
    input_dir: Union[str, Path] = DEFAULT_INPUT_DIR,
    input_file: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    从parquet文件加载微博数据

    Args:
        input_dir: 包含parquet文件的目录
        input_file: 可选的特定文件路径

    Returns:
        包含所有微博数据的DataFrame
    """
    if input_file:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        logger.info(f"从 {input_file} 加载数据")
        df = pd.read_parquet(
            input_path,
            columns=[
                "weibo_id",
                "weibo_content",
                "is_retweet",
                "time_stamp",
            ],
            engine="fastparquet",
        )
        return df

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    parquet_files = sorted(input_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到parquet文件")

    logger.info(f"找到 {len(parquet_files)} 个parquet文件")

    dfs = []
    for pf in tqdm(parquet_files, desc="加载parquet文件"):
        try:
            df = pd.read_parquet(
                pf,
                columns=[
                    "weibo_id",
                    "weibo_content",
                    "is_retweet",
                    "time_stamp",
                ],
                engine="fastparquet",
            )
            dfs.append(df)
        except Exception as e:
            logger.warning(f"加载文件 {pf} 时出错: {e}，跳过")

    if not dfs:
        raise ValueError("没有成功加载任何数据文件")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"共加载 {len(combined)} 条微博数据")

    return combined


def sample_weibos(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    从DataFrame中随机采样微博

    Args:
        df: 包含微博数据的DataFrame
        sample_size: 采样数量
        seed: 随机种子

    Returns:
        采样后的DataFrame
    """
    if len(df) <= sample_size:
        logger.warning(
            f"DataFrame有 {len(df)} 行，少于采样数量 {sample_size}，返回全部数据"
        )
        return df

    sampled = df.sample(n=sample_size, random_state=seed)
    logger.info(f"从 {len(df)} 条微博中采样了 {len(sampled)} 条")

    return sampled


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从分析结果生成统计摘要

    Args:
        results: 分析结果列表

    Returns:
        摘要字典
    """
    opinions = [r.get("opinion") for r in results if r.get("opinion") is not None]

    summary = {
        "total_analyzed": len(results),
        "opinion_2": opinions.count("2"),
        "opinion_1": opinions.count("1"),
        "opinion_0": opinions.count("0"),
        "opinion_-1": opinions.count("-1"),
        "opinion_-2": opinions.count("-2"),
        "cannot_tell": opinions.count("cannot tell"),
    }

    # 计算百分比
    total = summary["total_analyzed"]
    if total > 0:
        summary["opinion_2_pct"] = round(summary["opinion_2"] / total * 100, 2)
        summary["opinion_1_pct"] = round(summary["opinion_1"] / total * 100, 2)
        summary["opinion_0_pct"] = round(summary["opinion_0"] / total * 100, 2)
        summary["opinion_-1_pct"] = round(summary["opinion_-1"] / total * 100, 2)
        summary["opinion_-2_pct"] = round(summary["opinion_-2"] / total * 100, 2)
        summary["cannot_tell_pct"] = round(summary["cannot_tell"] / total * 100, 2)

    # 统计 token 使用情况
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    total_cached_tokens = sum(r.get("cached_tokens", 0) for r in results)

    summary["token_usage"] = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_cached_tokens": total_cached_tokens,
        "avg_prompt_tokens": round(total_prompt_tokens / total, 2) if total > 0 else 0,
        "avg_completion_tokens": (
            round(total_completion_tokens / total, 2) if total > 0 else 0
        ),
        "avg_total_tokens": round(total_tokens / total, 2) if total > 0 else 0,
        "avg_cached_tokens": round(total_cached_tokens / total, 2) if total > 0 else 0,
    }

    return summary


def sample(
    input_dir: Union[str, Path] = DEFAULT_INPUT_DIR,
    sample_dir: Union[str, Path] = DEFAULT_SAMPLE_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
    output_file: Optional[Union[str, Path]] = None,
):
    """
    从AI相关微博数据中随机采样

    Args:
        input_dir: 包含parquet文件的目录
        sample_dir: 采样结果保存目录
        sample_size: 采样数量
        seed: 随机种子
        output_file: 可选的输出文件路径（如果不指定，将使用默认文件名）
    """
    logger.info("=== 开始采样微博数据 ===")

    # 创建采样目录
    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)

    # 加载数据（不指定input_file，从input_dir加载所有文件）
    df = load_parquet_files(input_dir=input_dir, input_file=None)

    # 采样微博
    sampled_df = sample_weibos(df, sample_size=sample_size, seed=seed)

    # 保存采样结果
    if output_file:
        output_path = Path(output_file)
        # 确保输出文件的父目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = sample_path / "sampled_weibos.parquet"

    sampled_df.to_parquet(
        output_path, engine="fastparquet", index=False, compression="gzip"
    )
    logger.info(f"采样结果已保存到 {output_path}")
    logger.info(f"共采样 {len(sampled_df)} 条微博")

    print(f"\n=== 采样完成 ===")
    print(f"采样数量: {len(sampled_df)}")
    print(f"采样结果已保存到: {output_path}")

    logger.info("=== 采样完成 ===")
    return str(output_path)


def analyze(
    sampled_file: Union[str, Path] = DEFAULT_SAMPLED_FILE,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    delay: float = DEFAULT_DELAY,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    对已采样的微博进行情感分析

    Args:
        sampled_file: 采样后的parquet文件路径（默认使用DEFAULT_SAMPLED_FILE）
        output_dir: 输出结果目录
        delay: API调用间隔（秒）
        model: 使用的模型名称
        api_key: OpenAI API密钥（默认从configs.configs读取）
        base_url: API基础URL（默认从configs.configs读取）
    """
    logger.info("=== 开始AI内容情感分析 ===")

    # 处理sampled_file路径：如果是相对路径，尝试相对于DEFAULT_SAMPLE_DIR解析
    sampled_path = Path(sampled_file)
    if not sampled_path.is_absolute() and not sampled_path.exists():
        # 尝试相对于DEFAULT_SAMPLE_DIR
        potential_path = DEFAULT_SAMPLE_DIR / sampled_path
        if potential_path.exists():
            sampled_path = potential_path
        # 如果还是不存在，使用原始路径（会在后面检查）

    if not sampled_path.exists():
        raise FileNotFoundError(f"采样文件不存在: {sampled_path}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载采样数据
    logger.info(f"从 {sampled_path} 加载采样数据")
    sampled_df = pd.read_parquet(
        sampled_path,
        columns=[
            "weibo_id",
            "weibo_content",
            "is_retweet",
            "time_stamp",
        ],
        engine="fastparquet",
    )
    logger.info(f"加载了 {len(sampled_df)} 条采样微博")

    # 初始化分析器（使用默认值或提供的参数）
    api_key_to_use = api_key if api_key is not None else OPENAI_API_KEY
    analyzer = AISentimentAnalyzer(
        api_key=api_key_to_use, base_url=base_url, model=model
    )

    # 运行分析
    results = analyzer.analyze_batch(sampled_df, delay=delay)

    # 生成摘要
    summary = generate_summary(results)

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 保存详细结果为CSV
    results_df = pd.DataFrame(results)
    results_file = output_path / f"sentiment_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
    logger.info(f"详细结果已保存到 {results_file}")

    # 保存摘要为JSON
    summary_file = output_path / f"sentiment_summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"摘要已保存到 {summary_file}")

    # 打印摘要
    print("\n=== AI内容情感分析摘要 ===")
    print(f"总分析数量: {summary['total_analyzed']}")
    print(
        f"Opinion 2 (强烈积极): {summary['opinion_2']} ({summary.get('opinion_2_pct', 0)}%)"
    )
    print(
        f"Opinion 1 (温和积极): {summary['opinion_1']} ({summary.get('opinion_1_pct', 0)}%)"
    )
    print(
        f"Opinion 0 (中性): {summary['opinion_0']} ({summary.get('opinion_0_pct', 0)}%)"
    )
    print(
        f"Opinion -1 (温和消极): {summary['opinion_-1']} ({summary.get('opinion_-1_pct', 0)}%)"
    )
    print(
        f"Opinion -2 (强烈消极): {summary['opinion_-2']} ({summary.get('opinion_-2_pct', 0)}%)"
    )
    print(f"无法判断: {summary['cannot_tell']} ({summary.get('cannot_tell_pct', 0)}%)")

    # 打印 token 使用统计
    token_usage = summary.get("token_usage", {})
    print(f"\n=== Token 使用统计 ===")
    print(f"总 Prompt Tokens: {token_usage.get('total_prompt_tokens', 0)}")
    print(f"总 Completion Tokens: {token_usage.get('total_completion_tokens', 0)}")
    print(f"总 Tokens: {token_usage.get('total_tokens', 0)}")
    print(f"总 Cached Tokens: {token_usage.get('total_cached_tokens', 0)}")
    print(f"平均 Prompt Tokens: {token_usage.get('avg_prompt_tokens', 0)}")
    print(f"平均 Completion Tokens: {token_usage.get('avg_completion_tokens', 0)}")
    print(f"平均 Total Tokens: {token_usage.get('avg_total_tokens', 0)}")
    print(f"平均 Cached Tokens: {token_usage.get('avg_cached_tokens', 0)}")

    print(f"\n结果已保存到: {output_path}")

    logger.info("=== AI内容情感分析完成 ===")


def get_group_date_range(group: int) -> tuple[str, str]:
    """
    根据月份组获取日期范围

    Args:
        group: 月份组编号 (1-5)

    Returns:
        target dates list, format: ["2024-03-01", "2024-03-31", "2024-04-01", "2024-04-30", ...]
    """
    group_ranges = {
        1: ["2024-02-29", "2024-03-10", "2024-03-20"],
        2: ["2024-08-01", "2024-08-10", "2024-08-20"],
        3: ["2025-01-01", "2025-01-10", "2025-01-20"],
        4: ["2025-02-01"],
        5: ["2025-03-01"],
        6: ["2024-04-01", "2024-04-10", "2024-04-20"],
        7: ["2024-05-01", "2024-05-10", "2024-05-20"],
        8: ["2024-06-01", "2024-06-10", "2024-06-20"],
        9: ["2024-07-01", "2024-07-10", "2024-07-20"],
        10: ["2024-09-01", "2024-09-10", "2024-09-20"],
        11: ["2024-10-02", "2024-10-10", "2024-10-20"],
        12: ["2024-11-01", "2024-11-10", "2024-11-20"],
        13: ["2024-12-01", "2024-12-10", "2024-12-20"],
        14: ["2025-02-10"],
        15: ["2025-02-20"],
        16: ["2025-03-10"],
        17: ["2025-03-20"],
    }

    if group not in group_ranges:
        raise ValueError(f"月份组编号必须在 1-5 之间，当前值: {group}")

    return group_ranges[group]


def get_target_dates(
    start_date_str: str = "2024-03-01", end_date_str: str = "2025-03-20"
) -> List[str]:
    """
    生成指定日期范围内每个月1日、10日、20日的日期列表

    Args:
        start_date_str: 开始日期，格式为 YYYY-MM-DD
        end_date_str: 结束日期，格式为 YYYY-MM-DD

    Returns:
        日期字符串列表，格式为 YYYY-MM-DD
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    target_dates = []
    current_date = start_date

    while current_date <= end_date:
        # 检查是否是1日、10日或20日
        if current_date.day in [1, 10, 20]:
            target_dates.append(current_date.strftime("%Y-%m-%d"))
        # 移动到下一天
        current_date += timedelta(days=1)

    print(f"目标日期列表: {target_dates}")
    return target_dates


def get_analyzed_weibo_ids(results_file: Path) -> set[str]:
    """
    从结果文件中读取已分析的weibo_id

    Args:
        results_file: 结果文件路径

    Returns:
        已分析的weibo_id集合
    """
    if not results_file.exists():
        return set()

    try:
        df = pd.read_csv(results_file, encoding="utf-8-sig")
        if "weibo_id" in df.columns:
            # 过滤掉空值
            analyzed_ids = df["weibo_id"].dropna().astype(str).unique()
            return set(analyzed_ids)
        return set()
    except Exception as e:
        logger.warning(f"读取已分析weibo_id时出错: {e}")
        return set()


def generate_final_summary(
    output_dir: Union[str, Path],
    summary_file_name: str = "analyze_all_summary.json",
    merge_all_groups: bool = True,
    groups: Optional[List[int]] = None,
):
    """
    从结果文件生成最终摘要并保存（支持合并多个组的结果）

    Args:
        output_dir: 输出目录
        summary_file_name: 摘要文件名
        merge_all_groups: 是否合并所有组的结果（默认True，合并1-5组）
        groups: 要合并的组列表（如果指定，则只合并这些组；如果不指定且merge_all_groups=True，则合并1-5组）
    """
    logger.info("=== 生成最终摘要 ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 确定要合并的组
    if groups is None:
        if merge_all_groups:
            groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        else:
            logger.warning("未指定要合并的组，且 merge_all_groups=False")
            return

    # 收集所有组的结果文件
    all_results_dfs = []
    existing_groups = []

    for group in groups:
        results_file = output_path / f"analyze_all_results_group_{group}.csv"
        if results_file.exists():
            try:
                df = pd.read_csv(results_file, encoding="utf-8-sig")
                if len(df) > 0:
                    all_results_dfs.append(df)
                    existing_groups.append(group)
                    logger.info(f"加载组 {group} 的结果: {len(df)} 条记录")
            except Exception as e:
                logger.warning(f"读取组 {group} 的结果文件时出错: {e}")
        else:
            logger.warning(f"组 {group} 的结果文件不存在: {results_file}")

    if not all_results_dfs:
        logger.warning("没有找到任何结果文件，无法生成摘要")
        return

    try:
        # 合并所有组的结果
        all_results_df = pd.concat(all_results_dfs, ignore_index=True)

        # 去重（基于weibo_id）
        original_count = len(all_results_df)
        all_results_df = all_results_df.drop_duplicates(
            subset=["weibo_id"], keep="first"
        )
        if len(all_results_df) < original_count:
            logger.info(f"去重后: {original_count} -> {len(all_results_df)} 条记录")

        if len(all_results_df) == 0:
            logger.warning("合并后的结果为空，无法生成摘要")
            return

        # 映射列名以匹配 generate_summary 的期望格式
        if "input_token" in all_results_df.columns:
            all_results_df = all_results_df.rename(
                columns={
                    "input_token": "prompt_tokens",
                    "output_token": "completion_tokens",
                    "cached_token": "cached_tokens",
                }
            )
        all_results = all_results_df.to_dict("records")

        summary = generate_summary(all_results)

        # 添加组信息到摘要
        summary["groups_merged"] = existing_groups
        summary["groups_count"] = len(existing_groups)

        # 保存摘要
        summary_file = output_path / summary_file_name
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"摘要已保存到 {summary_file}")

        # 打印摘要
        print("\n=== 批量分析摘要（合并所有组） ===")
        print(f"合并的组: {existing_groups}")
        print(f"总分析数量: {summary['total_analyzed']}")
        print(
            f"Opinion 2 (强烈积极): {summary['opinion_2']} ({summary.get('opinion_2_pct', 0)}%)"
        )
        print(
            f"Opinion 1 (温和积极): {summary['opinion_1']} ({summary.get('opinion_1_pct', 0)}%)"
        )
        print(
            f"Opinion 0 (中性): {summary['opinion_0']} ({summary.get('opinion_0_pct', 0)}%)"
        )
        print(
            f"Opinion -1 (温和消极): {summary['opinion_-1']} ({summary.get('opinion_-1_pct', 0)}%)"
        )
        print(
            f"Opinion -2 (强烈消极): {summary['opinion_-2']} ({summary.get('opinion_-2_pct', 0)}%)"
        )
        print(
            f"无法判断: {summary['cannot_tell']} ({summary.get('cannot_tell_pct', 0)}%)"
        )

        # 打印 token 使用统计
        token_usage = summary.get("token_usage", {})
        print(f"\n=== Token 使用统计 ===")
        print(f"总 Prompt Tokens: {token_usage.get('total_prompt_tokens', 0)}")
        print(f"总 Completion Tokens: {token_usage.get('total_completion_tokens', 0)}")
        print(f"总 Tokens: {token_usage.get('total_tokens', 0)}")
        print(f"总 Cached Tokens: {token_usage.get('total_cached_tokens', 0)}")
        print(f"平均 Prompt Tokens: {token_usage.get('avg_prompt_tokens', 0)}")
        print(f"平均 Completion Tokens: {token_usage.get('avg_completion_tokens', 0)}")
        print(f"平均 Total Tokens: {token_usage.get('avg_total_tokens', 0)}")
        print(f"平均 Cached Tokens: {token_usage.get('avg_cached_tokens', 0)}")

    except Exception as e:
        logger.error(f"生成摘要时出错: {e}")


def append_single_result_to_csv(
    weibo_id: str,
    opinion: Any,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    results_file: Path,
):
    """
    将单条分析结果追加到CSV文件（逐条写入）

    Args:
        weibo_id: 微博ID
        opinion: 观点标签
        prompt_tokens: 输入token数
        completion_tokens: 输出token数
        cached_tokens: 缓存token数
        results_file: 结果文件路径
    """
    # 检查文件是否存在，决定是否需要写入表头
    file_exists = results_file.exists()
    write_header = not file_exists

    if file_exists:
        try:
            # 尝试读取文件，检查是否有表头
            with open(results_file, "r", encoding="utf-8-sig") as f:
                first_line = f.readline().strip()
                # 如果第一行是表头，则不需要再写
                if first_line.startswith("weibo_id") or first_line.startswith(
                    "weibo_id,opinion"
                ):
                    write_header = False
        except Exception:
            write_header = True

    # 打开文件，追加写入
    with open(results_file, "a", encoding="utf-8-sig", newline="") as f:
        if write_header:
            f.write("weibo_id,opinion,input_token,output_token,cached_token\n")
        # 写入数据行
        f.write(
            f"{weibo_id},{opinion},{prompt_tokens},{completion_tokens},{cached_tokens}\n"
        )


def analyze_all(
    input_dir: Union[str, Path] = DEFAULT_INPUT_DIR,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    delay: float = DEFAULT_DELAY,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    group: int = 1,
):
    """
    分析指定月份组的数据（支持并发）
    按weibo_id判断是否已分析，每条结果立即写入

    Args:
        input_dir: 输入数据目录（包含日期命名的parquet文件）
        output_dir: 输出结果目录
        delay: API调用间隔（秒）
        model: 使用的模型名称
        api_key: OpenAI API密钥（默认从configs.configs读取）
        base_url: API基础URL（默认从configs.configs读取）
        group: 月份组编号 (1-5)
            - 1: 2024-03至2024-07
            - 2: 2024-08至2024-12
            - 3: 2025-01
            - 4: 2025-02
            - 5: 2025-03
    """
    logger.info(f"=== 开始批量分析月份组 {group} ===")

    # 根据组获取日期范围
    # start_date, end_date = get_group_date_range(group)
    # logger.info(f"月份组 {group} 日期范围: {start_date} 到 {end_date}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 结果文件路径（每个组有独立的文件）
    results_file = output_path / f"analyze_all_results_group_{group}.csv"

    # 获取已分析的weibo_id
    analyzed_weibo_ids = get_analyzed_weibo_ids(results_file)
    logger.info(f"已分析 {len(analyzed_weibo_ids)} 条微博")

    # 获取目标日期列表（用于筛选输入文件）
    target_dates = get_group_date_range(group)  # get_target_dates(start_date, end_date)
    # logger.info(f"目标日期范围: {start_date} 到 {end_date}")
    logger.info(f"共 {len(target_dates)} 个目标日期：{target_dates}")

    # 初始化分析器
    api_key_to_use = api_key if api_key is not None else OPENAI_API_KEY
    analyzer = AISentimentAnalyzer(
        api_key=api_key_to_use, base_url=base_url, model=model
    )

    input_path = Path(input_dir)

    total_analyzed = 0
    total_skipped = 0

    # 遍历每个目标日期
    for date_str in tqdm(target_dates, desc="处理日期"):
        # 构建输入文件路径
        input_file = input_path / f"{date_str}.parquet"

        if not input_file.exists():
            logger.warning(f"文件不存在: {input_file}，跳过")
            continue

        try:
            # 加载数据
            logger.info(f"正在处理 {date_str} 的数据...")
            df = pd.read_parquet(
                input_file,
                columns=[
                    "weibo_id",
                    "weibo_content",
                    "is_retweet",
                    "time_stamp",
                ],
                engine="fastparquet",
            )

            # 去重
            df = df.drop_duplicates(subset=["weibo_id"])
            # 去除analyzed
            df = df[~df["weibo_id"].isin(analyzed_weibo_ids)]

            if len(df) == 0:
                logger.warning(f"{date_str} 的数据为空，跳过")
                continue

            logger.info(f"加载了 {len(df)} 条数据")

            # 逐条处理每条微博
            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc=f"分析{date_str}", leave=False
            ):
                weibo_id = str(row.get("weibo_id", "") or "")

                # 检查是否已分析
                # if weibo_id in analyzed_weibo_ids:
                #     total_skipped += 1
                #     continue

                try:
                    # 分析单条微博
                    analysis = analyzer.analyze_single(row)

                    # 立即写入结果
                    append_single_result_to_csv(
                        weibo_id=weibo_id,
                        opinion=analysis.get("opinion"),
                        prompt_tokens=analysis.get("prompt_tokens", 0),
                        completion_tokens=analysis.get("completion_tokens", 0),
                        cached_tokens=analysis.get("cached_tokens", 0),
                        results_file=results_file,
                    )

                    # 更新已分析集合（避免重复分析）
                    analyzed_weibo_ids.add(weibo_id)
                    total_analyzed += 1

                    # 每50条记录输出一次进度
                    if total_analyzed % 50 == 0:
                        logger.info(
                            f"已分析 {total_analyzed} 条，跳过 {total_skipped} 条"
                        )

                    # 添加延迟以避免API速率限制
                    time.sleep(delay)

                except Exception as e:
                    logger.error(f"分析 weibo_id {weibo_id} 时出错: {e}")
                    continue

            logger.info(
                f"{date_str} 处理完成，本次分析 {total_analyzed} 条，跳过 {total_skipped} 条"
            )

        except Exception as e:
            logger.error(f"处理 {date_str} 时出错: {e}")
            continue

    logger.info("=== 批量分析完成 ===")
    print(f"\n结果已保存到: {results_file}")
    print(f"本次运行：分析 {total_analyzed} 条，跳过 {total_skipped} 条")


if __name__ == "__main__":
    fire.Fire(
        {
            "sample": sample,
            "analyze": analyze,
            "analyze_all": analyze_all,
            "generate_summary": generate_final_summary,
        }
    )
