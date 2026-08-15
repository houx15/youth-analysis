"""
新分析表与既有 PPT 基线的对照

不追求数字完全一致：帖子类型拆分、不重叠匹配和修正后的转发链正则都会
带来差异。目标是让每一处差异都有明确来源，并留下书面记录。已知的三个
差异来源：
    1) 旧 density 覆盖全部帖子（纯转发计为 0），稀释了用户内容命中率；
       新表主口径只用表达帖（原创 + 带评论转发），数字预期偏高。
    2) 旧 density 逐词 str.count() 累加，嵌套词（如"新冠"出现在"新冠肺炎"
       里）会被重复计入字符数；新表按最左最长不重叠匹配，数字预期偏高。
    3) 旧文本清洗正则会误伤正常的 @提及，同时未能剥净部分转发链；
       新清洗器把 //@ 到帖尾整段截断。

另外，旧基线发布的"来源转发进入率"分母是"该性别全部转发者"，本研究
还需要"该性别全部用户"这个分母，两者不能混用，本模块把两种口径都
算出来并显式标注各自分母，避免下游用错分母。

使用方法:
    python -m gender_domain.reconcile_baseline run 2020
"""

import json
import os

import fire
import pandas as pd

from gender_domain import config

VIZ_DIR = "viz_data"

# 对照只需要用到表 C 的这几列，显式声明便于按需读取
USER_TABLE_COLUMNS = [
    "gender",
    "n_retweets",
    "public_source_entered",
    "public_topical_share",
    "celebrity_source_entered",
    "celebrity_topical_share",
]

DOMAINS = ("public", "celebrity")

DENOMINATOR_LABELS = {
    "entered_rate_all_users": "分子=进入该领域来源转发的用户数；分母=该性别全部用户数（不论是否转发过）",
    "entered_rate_among_retweeters": (
        "分子=进入该领域来源转发的用户数；分母=该性别全部转发者数（与旧基线 PPT 口径一致）"
    ),
}


def summarize_new_table(users):
    """基于表 C 计算新口径的性别对照指标，纯函数，不做任何文件 IO

    users 为空表时所有 groupby 结果自然退化为空字典，不会报错，
    对应"本地没有服务器产出的 analysis_data"这一常见场景。
    """
    by_gender = users.groupby("gender")
    summary = {"users": by_gender.size().to_dict()}

    for domain in DOMAINS:
        entered_col = f"{domain}_source_entered"
        summary[f"{domain}_source_entered"] = by_gender[entered_col].sum().to_dict()
        # 分母=全部同性别用户
        summary[f"{domain}_entered_rate_all_users"] = by_gender[entered_col].mean().to_dict()
        # 分母=该性别全部转发者，与旧基线 PPT 口径一致
        retweeters = users[users["n_retweets"] > 0]
        summary[f"{domain}_entered_rate_among_retweeters"] = (
            retweeters.groupby("gender")[entered_col].mean().to_dict()
        )
        summary[f"{domain}_topical_share_mean"] = (
            by_gender[f"{domain}_topical_share"].mean().to_dict()
        )

    # 两种分母的进入率极易被混用，这里显式写出各自定义，而不是只靠字段名隐含
    summary["retweet_ratio_denominators"] = DENOMINATOR_LABELS
    return summary


def build_report(users, overview, density, year, overview_path=None, density_path=None):
    """组装完整对照报告，不做任何文件 IO，便于用合成数据测试

    Args:
        users: 表 C（或至少含 USER_TABLE_COLUMNS 列的子集）
        overview: 旧转发总览基线 DataFrame，缺失时传 None
        density: 旧 density 基线 DataFrame，缺失时传 None
        year: 年份，仅用于写入报告
        overview_path/density_path: 仅用于缺失时在 notes 里报出具体路径

    Returns:
        dict，结构为 {"year", "new", "baseline", "notes"}
    """
    report = {"year": year, "new": summarize_new_table(users), "baseline": {}, "notes": []}

    if overview is not None:
        report["baseline"]["retweet_overview"] = overview.to_dict(orient="records")
        report["notes"].append(
            "旧基线的来源转发比例分母是'所有转发者'，新表同时给出'全部同性别用户'口径，"
            "两者不应直接相等"
        )
    else:
        report["notes"].append(f"未找到 {overview_path or '(未提供路径)'}，跳过转发基线对照")

    # 三条 density 差异来源说明都只在 density 基线真的存在时才写：
    # 它们描述的是"新表相对旧 density 基线为什么会不同"，没有基线可比时
    # 写出来只会让报告看起来做过一次并不存在的对照。
    if density is not None:
        report["baseline"]["density_summary"] = density.to_dict(orient="records")
        report["notes"].append(
            "旧 density 覆盖全部帖子（纯转发计为 0），新表主口径只用表达帖，差异应向上"
        )
        report["notes"].append(
            "旧 density 逐词累加会让嵌套词重复计字符，新表为最左最长不重叠，差异应向上"
        )
        report["notes"].append(
            "旧文本清洗正则会误伤正常的 @提及、且未能剥净部分转发链，新清洗器把 //@ 到帖尾整段截断"
        )
    else:
        report["notes"].append(
            f"未找到 {density_path or '(未提供路径)'}，跳过 density 基线对照"
            "（三条 density 差异来源说明一并省略）"
        )

    return report


def reconcile(year=config.YEAR):
    """对照用户数、来源进入人数和内容命中率，写出对照报告

    user_domain 表是本次对照的主表，缺失时直接抛出异常（pandas 读取
    不存在的 parquet 天然会 FileNotFoundError），不应被静默当成空表；
    两个 viz_data 基线文件则允许缺失，缺失时优雅降级并记入 notes。
    """
    user_path = os.path.join(config.OUTPUT_DIR, f"user_domain_{year}.parquet")
    users = pd.read_parquet(user_path, columns=USER_TABLE_COLUMNS)

    overview_path = os.path.join(VIZ_DIR, f"retweet_overview_{year}.parquet")
    overview = pd.read_parquet(overview_path) if os.path.exists(overview_path) else None

    density_path = os.path.join(VIZ_DIR, f"density_summary_{year}.parquet")
    density = pd.read_parquet(density_path) if os.path.exists(density_path) else None

    report = build_report(
        users, overview, density, year, overview_path=overview_path, density_path=density_path
    )

    out_path = os.path.join(config.OUTPUT_DIR, f"reconciliation_{year}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"已保存对照报告: {out_path}")

    # 与其它每一步一致地写出 manifest：对照报告同样是正式产出，
    # 也需要记录代码版本、实际读到的输入（哪些基线在、哪些缺）和样本数
    inputs = [os.path.basename(user_path)]
    if overview is not None:
        inputs.append(os.path.join(VIZ_DIR, os.path.basename(overview_path)))
    if density is not None:
        inputs.append(os.path.join(VIZ_DIR, os.path.basename(density_path)))
    manifest = config.build_manifest(
        step=f"reconciliation_{year}",
        inputs=inputs,
        params={
            "year": year,
            "user_table_columns": USER_TABLE_COLUMNS,
            "baseline_present": {
                "retweet_overview": overview is not None,
                "density_summary": density is not None,
            },
        },
        counts={
            "users": int(len(users)),
            "gender": users["gender"].value_counts().to_dict(),
            "notes": len(report["notes"]),
        },
    )
    config.write_manifest(manifest, os.path.join(config.OUTPUT_DIR, f"reconciliation_{year}"))

    print(json.dumps(report["new"], ensure_ascii=False, indent=2, default=str))
    return report


if __name__ == "__main__":
    fire.Fire({"run": reconcile})
