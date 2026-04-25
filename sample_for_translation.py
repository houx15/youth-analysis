"""
从已分析的微博情感数据中抽样，用于跨语言验证（DeepL 翻译后由 GPT 重新打分）。

按 opinion (-2/-1/0/1/2) 比例分层抽样，输出单个 parquet 文件，
之后可通过 scp 转移到 twitterapi-io 所在的服务器进行下游处理。

输出列：weibo_id, weibo_content, original_opinion, time_stamp

使用：
    python sample_for_translation.py --sample_size 200
    python sample_for_translation.py --output_file ai_attitudes/weibo_translation_sample.parquet
"""

import logging
from pathlib import Path
from typing import Union

import fire
import pandas as pd

from ai_sentiment_analyzer import BASE_DIR, get_group_date_range

DEFAULT_RESULTS_DIR = BASE_DIR / "ai_sentiment_results"
DEFAULT_WEIBO_DIR = BASE_DIR / "ai_weibo_text"
DEFAULT_OUTPUT_FILE = BASE_DIR / "weibo_translation_sample.parquet"
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_analyzed_results(results_dir: Path) -> tuple[pd.DataFrame, list[int]]:
    """加载所有 analyze_all_results_group_*.csv 并返回去重后的 df 与组列表。"""
    csv_files = sorted(results_dir.glob("analyze_all_results_group_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"在 {results_dir} 下未找到 analyze_all_results_group_*.csv"
        )

    dfs, groups = [], []
    for f in csv_files:
        name = f.stem
        try:
            group_num = int(name.rsplit("_", 1)[-1])
        except ValueError:
            logger.warning(f"无法从 {name} 解析组号，跳过")
            continue
        groups.append(group_num)
        df = pd.read_csv(f, encoding="utf-8-sig", dtype={"weibo_id": str})
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["weibo_id"], keep="first")
    logger.info(
        f"从 {len(csv_files)} 个 CSV 加载 {len(combined)} 条去重记录 (组: {sorted(set(groups))})"
    )
    return combined, sorted(set(groups))


def _filter_valid_opinions(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 opinion ∈ {-2,-1,0,1,2} 的行。"""
    df = df[df["opinion"].notna()].copy()
    df["opinion"] = pd.to_numeric(df["opinion"], errors="coerce")
    df = df.dropna(subset=["opinion"])
    df["opinion"] = df["opinion"].astype(int)
    df = df[df["opinion"].isin([-2, -1, 0, 1, 2])]
    return df


def _load_weibo_content(weibo_dir: Path, groups: list[int]) -> pd.DataFrame:
    """根据组号反推日期，加载对应的微博内容 parquet。"""
    target_dates: set[str] = set()
    for g in groups:
        try:
            target_dates.update(get_group_date_range(g))
        except ValueError:
            logger.warning(f"组 {g} 不在已知映射中，跳过")

    dfs = []
    for date_str in sorted(target_dates):
        path = weibo_dir / f"{date_str}.parquet"
        if not path.exists():
            logger.warning(f"缺失 {path}，跳过")
            continue
        df = pd.read_parquet(
            path,
            columns=["weibo_id", "weibo_content", "time_stamp"],
            engine="fastparquet",
        )
        df["weibo_id"] = df["weibo_id"].astype(str)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"组 {groups} 下没有找到任何微博内容文件")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["weibo_id"], keep="first")
    logger.info(f"加载 {len(combined)} 条微博内容记录")
    return combined


def _stratified_proportional_sample(
    df: pd.DataFrame, n: int, seed: int
) -> pd.DataFrame:
    """按 opinion 类别比例分层抽样，总数为 n。"""
    counts = df["opinion"].value_counts().sort_index()
    total = counts.sum()
    raw = {op: cnt / total * n for op, cnt in counts.items()}
    allocations = {op: int(round(v)) for op, v in raw.items()}

    diff = n - sum(allocations.values())
    if diff != 0:
        fracs = sorted(
            raw.items(), key=lambda x: x[1] - int(x[1]), reverse=(diff > 0)
        )
        for op, _ in fracs[: abs(diff)]:
            allocations[op] += 1 if diff > 0 else -1

    logger.info(f"分层分配（按比例）：{dict(sorted(allocations.items()))}")

    samples = []
    for op, k in allocations.items():
        if k <= 0:
            continue
        cls_df = df[df["opinion"] == op]
        if len(cls_df) <= k:
            logger.warning(f"类别 {op} 仅有 {len(cls_df)} 条 < 需要 {k}，全部纳入")
            samples.append(cls_df)
        else:
            samples.append(cls_df.sample(n=k, random_state=seed))
    return pd.concat(samples, ignore_index=True)


def main(
    results_dir: Union[str, Path] = DEFAULT_RESULTS_DIR,
    weibo_dir: Union[str, Path] = DEFAULT_WEIBO_DIR,
    output_file: Union[str, Path] = DEFAULT_OUTPUT_FILE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
):
    """
    抽取用于跨语言验证的微博样本（一次抽样、一次翻译，下游对同一份样本做多轮 LLM 重打分）。

    Args:
        results_dir: 含 analyze_all_results_group_*.csv 的目录
        weibo_dir: 含每日微博 parquet 的目录
        output_file: 输出 parquet 路径
        sample_size: 样本量（按 opinion 比例分层）
        seed: 随机种子
    """
    results_dir = Path(results_dir)
    weibo_dir = Path(weibo_dir)
    output_file = Path(output_file)

    results_df, groups = _load_analyzed_results(results_dir)
    results_df = _filter_valid_opinions(results_df)
    logger.info(f"过滤合法 opinion 后剩余 {len(results_df)} 条")
    logger.info(
        f"opinion 分布：\n{results_df['opinion'].value_counts().sort_index()}"
    )

    content_df = _load_weibo_content(weibo_dir, groups)
    merged = results_df.merge(content_df, on="weibo_id", how="inner")
    logger.info(f"与内容合并后 {len(merged)} 条")

    sampled = _stratified_proportional_sample(merged, n=sample_size, seed=seed)
    sampled = sampled.rename(columns={"opinion": "original_opinion"})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out = sampled[["weibo_id", "weibo_content", "original_opinion", "time_stamp"]]
    out.to_parquet(
        output_file, engine="fastparquet", index=False, compression="gzip"
    )
    logger.info(f"已保存 {len(out)} 条样本到 {output_file}")
    print("\n=== 样本类别分布 ===")
    print(out["original_opinion"].value_counts().sort_index())
    print(f"\n输出: {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
