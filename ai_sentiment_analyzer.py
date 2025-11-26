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
from datetime import datetime
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
DEFAULT_DELAY = 0.5  # API调用间隔（秒）
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

    def analyze_single(self, row: pd.Series) -> Dict[str, Any]:
        """
        分析单条微博的情感倾向

        Args:
            row: 微博数据行

        Returns:
            分析结果字典，包含 sentiment/opinion 和 token 使用统计
        """
        system_prompt, user_prompt = self._build_prompt(row)

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

            return {
                "opinion": opinion,
                **token_stats,
            }
        except Exception as e:
            logger.error(f"分析单条微博时出错: {e}")
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
        "opinion_2": opinions.count(2),
        "opinion_1": opinions.count(1),
        "opinion_0": opinions.count(0),
        "opinion_-1": opinions.count(-1),
        "opinion_-2": opinions.count(-2),
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


if __name__ == "__main__":
    fire.Fire(
        {
            "sample": sample,
            "analyze": analyze,
        }
    )
