"""
表 C：用户级母表；表 D：用户—月份面板

对应研究设计 0.4 节 C/D 表。输入为表 A、表 B 的分片目录。

内容指标的主口径分母为"表达帖"（原创 + 转发新增评论），
同时并行输出以全部帖子为分母的 _allposts 版本，供 13.1 分母稳健性使用。

使用方法:
    python -m gender_domain.build_user_tables build 2020
"""

import glob
import os

import fire
import numpy as np
import pandas as pd

from gender_domain import config
from gender_domain import id_rules as ir

DOMAINS = ("public", "celebrity")
EXPRESSIVE_TYPES = ("original", "retweet_comment")


def _safe_divide(numerator, denominator):
    """分母为 0 时返回 NaN，避免把"没有分母"写成 0"""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def aggregate_posts(post_df):
    """表 A -> 用户级内容与活动指标"""
    df = post_df.copy()
    # 用 id_rules 而不是裸 astype(str)：user_id 列一旦因缺失值被 pandas
    # 上转型为 float64，裸 astype(str) 会产出 "123.0" 这种伪 ID，导致后续
    # 与表 B 聚合结果 join 时静默对不上（见 id_rules.py 顶部说明）。
    df["user_id"] = ir.normalize_id_series(df["user_id"])
    df["is_expressive"] = df["post_type"].isin(EXPRESSIVE_TYPES)
    df["is_retweet_post"] = df["post_type"].isin(("retweet_plain", "retweet_comment"))

    base = df.groupby("user_id").agg(
        gender=("gender", "first"),
        n_posts=("weibo_id", "count"),
        n_retweets=("is_retweet_post", "sum"),
        n_expressive_posts=("is_expressive", "sum"),
        n_active_days=("date", "nunique"),
        n_active_months=("month", "nunique"),
    )

    expressive = df[df["is_expressive"]]
    for domain in DOMAINS:
        hit_col = f"{domain}_hit"
        # 主口径：表达帖
        agg = expressive.groupby("user_id").agg(
            topical_posts=(hit_col, "sum"),
            chars=("n_chars", "sum"),
            chars_hit=(f"{domain}_chars_hit", "sum"),
            n_hits=(f"{domain}_n_hits", "sum"),
            post_mean_density=(f"{domain}_density", "mean"),
        )
        agg = agg.reindex(base.index)
        # 计数类列（帖数、字符数）在"该用户没有表达帖"时填 0 是有定义的事实
        # （0 条表达帖，自然 0 次命中、0 个字符），下游 _safe_divide 会把
        # 0/0 这种分母为 0 的情形转成 NaN。但 post_mean_density 是逐帖密度
        # 的算术平均，不经过 _safe_divide；如果在这里也 fillna(0)，会把
        # "无表达帖、均值无定义"错误地写成"均值恰好为 0"，违反"零分母
        # 必须是 NaN"的规则，所以这一列刻意不填 0，保留 NaN。
        # fillna(0) 之后必须显式转回整数类型：reindex 引入的 NaN 会先把
        # 整列上转型为 float64，fillna(0) 只是把值变成 0.0，不会把 dtype
        # 转回来，如果不加 astype(int)，只要出现过一个"该用户没有表达帖"
        # 的场景（真实数据里必然出现），topical_posts 这类计数列在最终
        # 表 C 里就会一直是浮点数（1.0 而不是 1）。
        count_cols = ["topical_posts", "chars", "chars_hit", "n_hits"]
        agg[count_cols] = agg[count_cols].fillna(0).astype(int)

        base[f"{domain}_topical_posts"] = agg["topical_posts"]
        base[f"{domain}_topical_share"] = _safe_divide(
            agg["topical_posts"], base["n_expressive_posts"]
        )
        base[f"{domain}_char_density"] = _safe_divide(agg["chars_hit"], agg["chars"])
        base[f"{domain}_hits_per_1k"] = _safe_divide(agg["n_hits"], agg["chars"]) * 1000
        base[f"{domain}_post_mean_density"] = agg["post_mean_density"]

        # 并行口径：全部帖子为分母
        all_hits = df.groupby("user_id")[hit_col].sum().reindex(base.index).fillna(0)
        base[f"{domain}_topical_share_allposts"] = _safe_divide(all_hits, base["n_posts"])

        # 出现该领域内容的月份数（计数，0 是有定义的事实，填 0 没问题）
        months = (
            df[df[hit_col]].groupby("user_id")["month"].nunique().reindex(base.index).fillna(0)
        )
        base[f"{domain}_content_months"] = months.astype(int)

    return base.reset_index()


def aggregate_events(event_df):
    """表 B -> 用户级来源传播指标"""
    df = event_df.copy()
    df["user_id"] = ir.normalize_id_series(df["user_id"])

    users = pd.Index(df["user_id"].unique(), name="user_id")
    out = pd.DataFrame(index=users)

    for domain in DOMAINS:
        sub = df[df["source_domain"] == domain]
        counts = sub.groupby("user_id").size().reindex(users).fillna(0)
        months = sub.groupby("user_id")["month"].nunique().reindex(users).fillna(0)
        valid = sub[sub["delay_valid"]]
        delay_median = valid.groupby("user_id")["delay_seconds"].median().reindex(users)
        delay_p90 = valid.groupby("user_id")["delay_seconds"].quantile(0.9).reindex(users)

        out[f"{domain}_source_count"] = counts.astype(int)
        out[f"{domain}_source_entered"] = counts > 0
        out[f"{domain}_source_months"] = months.astype(int)
        out[f"{domain}_delay_median"] = delay_median
        out[f"{domain}_delay_p90"] = delay_p90

    return out.reset_index()


def aggregate_user_month(post_df, event_df):
    """表 D：用户—月份面板，帖子或转发事件任一存在即为活跃月"""
    posts = post_df.copy()
    posts["user_id"] = ir.normalize_id_series(posts["user_id"])
    posts["is_expressive"] = posts["post_type"].isin(EXPRESSIVE_TYPES)
    posts["is_retweet_post"] = posts["post_type"].isin(("retweet_plain", "retweet_comment"))

    monthly = posts.groupby(["user_id", "month"]).agg(
        gender=("gender", "first"),
        n_posts=("weibo_id", "count"),
        n_retweets=("is_retweet_post", "sum"),
        n_expressive_posts=("is_expressive", "sum"),
        n_active_days=("date", "nunique"),
    )
    for domain in DOMAINS:
        hits = posts[posts[f"{domain}_hit"]].groupby(["user_id", "month"]).size()
        chars = posts.groupby(["user_id", "month"])["n_chars"].sum()
        chars_hit = posts.groupby(["user_id", "month"])[f"{domain}_chars_hit"].sum()
        monthly[f"{domain}_topical_posts"] = hits.reindex(monthly.index).fillna(0).astype(int)
        monthly[f"{domain}_topical_share"] = _safe_divide(
            monthly[f"{domain}_topical_posts"], monthly["n_expressive_posts"]
        )
        monthly[f"{domain}_char_density"] = _safe_divide(
            chars_hit.reindex(monthly.index), chars.reindex(monthly.index)
        )

    events = event_df.copy()
    events["user_id"] = ir.normalize_id_series(events["user_id"])
    event_counts = (
        events.groupby(["user_id", "month", "source_domain"]).size().unstack(fill_value=0)
    )

    # 外连接：事件表里存在、但帖子表里完全没有出现的 (user_id, month)
    # 组合（该月只转发、没有留下任何帖子）也必须在面板里占一行，否则
    # 该用户"只转发不发帖"的月份会从活跃月分母里消失。
    panel = monthly.join(event_counts, how="outer")
    for domain in DOMAINS:
        col = domain if domain in panel.columns else None
        values = panel[col] if col else 0
        panel[f"{domain}_source_count"] = pd.Series(values, index=panel.index).fillna(0).astype(int)
        panel[f"{domain}_source_entered"] = panel[f"{domain}_source_count"] > 0
        if col:
            panel = panel.drop(columns=[col])

    # 计数类列：外连接后，只在事件表中出现的月份在这些列上是 NaN，
    # 但"这个月没有帖子"本身是有定义的事实（0 帖），填 0 是正确的。
    count_cols = ["n_posts", "n_retweets", "n_expressive_posts", "n_active_days"]
    panel[count_cols] = panel[count_cols].fillna(0).astype(int)
    # 同理，命中计数（topical_posts）也是有定义的事实：这个月没有帖子，
    # 自然 0 次命中。char_density/topical_share 这些比例类列则保持 NaN
    # （0 分母 -> 无法计算），不在这里填充。
    topical_posts_cols = [f"{domain}_topical_posts" for domain in DOMAINS]
    panel[topical_posts_cols] = panel[topical_posts_cols].fillna(0).astype(int)
    # gender 是本研究的自变量，绝不能跨用户借值。
    # SeriesGroupBy.ffill() 返回的已经是普通 Series（不再是分组对象），
    # 链式 .bfill() 会在整张面板（跨所有用户）上无分组地回填，把相邻
    # 用户（按 (user_id, month) 排序后紧邻的下一行）的性别错误地抄给
    # 这个用户——尤其容易发生在某用户的所有月份都只来自事件表（该月
    # 只转发没发帖），这类行在 monthly 里从未出现过 gender，必须完全
    # 依赖同一用户组内其它行的 ffill/bfill 才能补上。用 groupby().transform()
    # 把 ffill 和 bfill 都锁在组内执行，两个方向都不越出 user_id 边界。
    # 如果某用户的所有行 gender 都缺失（不应该发生，因为上游表 A/B 都已
    # 过滤掉 gender 为空的行，但这里不假设上游一定做到），组内 ffill/bfill
    # 找不到任何非空值可借，会原样保留 NaN，不会被强行赋值。
    panel["gender"] = panel.groupby(level="user_id")["gender"].transform(
        lambda s: s.ffill().bfill()
    )

    return panel.reset_index()


def _source_combo(row):
    public = row["public_source_entered"]
    celebrity = row["celebrity_source_entered"]
    if public and celebrity:
        return "both"
    if public:
        return "public_only"
    if celebrity:
        return "celebrity_only"
    return "neither"


def diagnose_event_only_users(post_agg, event_agg, max_examples=5):
    """诊断：出现在表 B 聚合结果、但完全没出现在表 A 聚合结果里的用户数

    combine_user_table 用左连接把 event_agg 拼到 post_agg 上（以 post_agg
    的用户为准）。这隐含一个假设：出现在表 B 的用户一定也出现在表 A——
    因为每条转发事件的 weibo_id 本身就是一条帖子记录，且两张表用同一套
    gender 非空过滤，理论上不应该有用户只在事件表出现。但这从未在真实
    数据上验证过，所以这里不做任何删除/报错（不阻断流程），只计数写入
    manifest：如果正式跑出来这个数不是 0，说明左连接正在悄悄丢弃这些
    用户（他们仍会出现在表 D 的用户—月份面板里，因为那里是外连接），
    需要人工核实，而不是继续信任这个假设。
    """
    post_users = set(post_agg["user_id"])
    event_users = set(event_agg["user_id"])
    missing = sorted(event_users - post_users)
    return {
        "event_only_user_count": len(missing),
        "example_user_ids": missing[:max_examples],
    }


def combine_user_table(post_agg, event_agg, month_panel):
    """合并用户级指标，补齐没有来源转发的用户并生成参与组合

    活跃月份数以面板为准：只在某月转发、当月没有留下帖子的用户，
    该月同样算作活跃月，否则持续性比例的分母会偏小。
    """
    combined = post_agg.merge(event_agg, on="user_id", how="left")

    panel_months = (
        month_panel.groupby("user_id")["month"].nunique().rename("n_active_months_panel")
    )
    combined = combined.merge(panel_months, on="user_id", how="left")
    combined["n_active_months_panel"] = (
        combined["n_active_months_panel"].fillna(combined["n_active_months"]).astype(int)
    )

    for domain in DOMAINS:
        combined[f"{domain}_source_count"] = combined[f"{domain}_source_count"].fillna(0).astype(int)
        combined[f"{domain}_source_entered"] = combined[f"{domain}_source_entered"].fillna(False)
        combined[f"{domain}_source_months"] = combined[f"{domain}_source_months"].fillna(0).astype(int)
        # 配置比例：该领域来源转发数 ÷ 用户全部转发帖数，上限截到 1
        share = _safe_divide(combined[f"{domain}_source_count"], combined["n_retweets"])
        combined[f"{domain}_source_share"] = share.clip(upper=1.0)
        # 持续性：参与月份数 ÷ 活跃月份数（分母含只转发不发帖的月份）
        combined[f"{domain}_source_month_share"] = _safe_divide(
            combined[f"{domain}_source_months"], combined["n_active_months_panel"]
        )

    combined["source_combo"] = combined.apply(_source_combo, axis=1)
    return combined


def _read_shards(name, year):
    pattern = os.path.join(config.OUTPUT_DIR, f"{name}_{year}", "month=*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到分片: {pattern}")
    print(f"读取 {len(files)} 个 {name} 分片")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build(year=config.YEAR):
    """从表 A/B 分片生成表 C 与表 D"""
    posts = _read_shards("post_domain_measures", year)
    events = _read_shards("retweet_domain_events", year)
    print(f"帖子 {len(posts):,} 行，转发事件 {len(events):,} 行")

    post_agg = aggregate_posts(posts)
    event_agg = aggregate_events(events)
    panel = aggregate_user_month(posts, events)
    # 诊断：combine_user_table 的左连接假设"表 B 的用户都在表 A 出现"，
    # 只计数不阻断，具体见 diagnose_event_only_users 的说明
    event_only = diagnose_event_only_users(post_agg, event_agg)
    if event_only["event_only_user_count"] > 0:
        print(
            f"警告: {event_only['event_only_user_count']} 个用户只出现在转发事件表，"
            f"不在帖子表聚合结果中，示例: {event_only['example_user_ids']}"
        )
    user_table = combine_user_table(post_agg, event_agg, panel)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    user_path = os.path.join(config.OUTPUT_DIR, f"user_domain_{year}.parquet")
    panel_path = os.path.join(config.OUTPUT_DIR, f"user_month_domain_{year}.parquet")
    user_table.to_parquet(user_path, engine="pyarrow", index=False)
    panel.to_parquet(panel_path, engine="pyarrow", index=False)
    print(f"已保存: {user_path}（{len(user_table):,} 用户）")
    print(f"已保存: {panel_path}（{len(panel):,} 用户-月）")

    manifest = config.build_manifest(
        step=f"user_tables_{year}",
        inputs=[f"post_domain_measures_{year}", f"retweet_domain_events_{year}"],
        params={"year": year, "expressive_types": list(EXPRESSIVE_TYPES)},
        counts={
            "users": int(len(user_table)),
            "user_months": int(len(panel)),
            "gender": user_table["gender"].value_counts().to_dict(),
            "source_combo": user_table["source_combo"].value_counts().to_dict(),
            "event_only_users": event_only,
        },
    )
    config.write_manifest(manifest, os.path.join(config.OUTPUT_DIR, f"user_tables_{year}"))
    return user_path, panel_path


if __name__ == "__main__":
    fire.Fire({"build": build})
