"""
表 C：用户级母表；表 D：用户—月份面板

对应研究设计 0.4 节 C/D 表。输入为表 A、表 B 的分片目录。

内容指标的主口径分母为"表达帖"（原创 + 转发新增评论），
同时并行输出以全部帖子为分母的 _allposts 版本，供 13.1 分母稳健性使用。

使用方法:
    python -m gender_domain.build_user_tables build 2020
"""

import glob
import json
import os

import fire
import numpy as np
import pandas as pd

from gender_domain import config
from gender_domain import id_rules as ir
from gender_domain import text_rules as tr

DOMAINS = ("public", "celebrity")
# 表达帖口径的唯一来源是 text_rules，本模块不另立定义
EXPRESSIVE_TYPES = tr.EXPRESSIVE_TYPES

# 表 A 分片里表 C/表 D 真正用到的列。parquet 必须显式指定 columns：
# 表 A 还带着 public_term_counts / celebrity_term_counts 两列"词:次数"
# 拼接字符串（供 §13.3 词表重采样重新聚合用），全年一次性读进内存却
# 一次都用不到，是纯粹的浪费。
#
# province：表 A 已经透传的省级行政区划代码列（见 build_post_table.py），
# 这里显式读入供 aggregate_posts 聚合到表 C。region 不在这份列表里——
# 表 A 目前并不写出 region 列（上游 cleaned_weibo_cov 合并画像时其实已经
# 带了这个字段，但 build_post_table.py 尚未透传），所以 region 由
# aggregate_posts 在聚合前从 province 现场推导，不作为要从 parquet 读取
# 的原始列；一旦表 A 补上 region 直传，_ensure_province_region 会优先
# 采用表 A 自带的取值，不会重复推导。
POST_SHARD_COLUMNS = [
    "weibo_id",
    "user_id",
    "date",
    "month",
    "gender",
    "post_type",
    "n_chars",
    "is_expressive",
    "province",
] + [
    f"{domain}_{suffix}"
    for domain in DOMAINS
    for suffix in ("hit", "n_hits", "chars_hit", "density")
]

# 表 B 分片里表 C/表 D 真正用到的列
EVENT_SHARD_COLUMNS = [
    "user_id",
    "weibo_id",
    "month",
    "source_domain",
    "delay_seconds",
    "delay_valid",
]


def _safe_divide(numerator, denominator):
    """分母为 0 时返回 NaN，避免把"没有分母"写成 0"""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


# 国家统计局四大区域划分（东部/中部/西部/东北），键为两位省级行政区划
# 代码（GB/T 2260），与本仓库 word_frequency_analysis.py / data_analysis.py /
# user_profile_analysis.py 里以中文省份名为键的 province_to_region 使用
# 同一套四区域方案，只是换成表 A province 列本身的编码格式，避免再多
# 维护一份中文省份名映射、也避免两处方案不一致。
PROVINCE_CODE_TO_REGION = {
    # East 东部
    "11": "East", "12": "East", "13": "East", "31": "East", "32": "East",
    "33": "East", "35": "East", "37": "East", "44": "East", "46": "East",
    # Central 中部
    "14": "Central", "34": "Central", "36": "Central", "41": "Central",
    "42": "Central", "43": "Central",
    # West 西部
    "15": "West", "45": "West", "50": "West", "51": "West", "52": "West",
    "53": "West", "54": "West", "61": "West", "62": "West", "63": "West",
    "64": "West", "65": "West",
    # Northeast 东北
    "21": "Northeast", "22": "Northeast", "23": "Northeast",
}


def _ensure_province_region(df):
    """保证聚合前的帖子表带有 province 与 region 两列（就地补齐并返回同一个 df）

    - province 缺失（旧夹具/旧分片没有这一列）：两列都填缺失并打印警告，
      不阻断聚合——与 _ensure_is_expressive 处理旧分片的方式一致。
    - province 存在、region 缺失（表 A 当前的真实状态）：按
      PROVINCE_CODE_TO_REGION 从 province 现场推导 region。
    - region 已经存在（表 A 未来某天补上直传的 region 列）：直接采用表 A
      自带的取值，不重复推导，避免两处方案不一致时被这里的映射覆盖。
    未匹配到映射表的省份代码（如港澳台或异常代码）留空为 NaN，不强行归类。
    """
    if "province" not in df.columns:
        print("警告: 帖子表缺少 province 列，province/region 填为缺失（可能是旧夹具或旧分片）")
        df["province"] = np.nan
    if "region" not in df.columns:
        df["region"] = df["province"].map(PROVINCE_CODE_TO_REGION)
    return df


def _ensure_is_expressive(df):
    """保证帖子表带有权威的 is_expressive 列（就地补齐并返回同一个 df）

    表 A 从 v2 起自带 is_expressive 列（由 text_rules.is_expressive_series
    唯一定义）。表 C 与表 D 两条聚合路径都只读这一列，谁都不许自己再按
    post_type 推一遍——修复前正是因为两边各推各的，才出现同名列
    ({d}_char_density、{d}_topical_share) 在两张表里用了不同分母。

    只有读到旧版分片（没有这一列）时才现场补，且仍然调用同一个定义函数，
    不复制口径；同时打印中文警告，让"分片是旧版"这件事不会静默通过。
    """
    if "is_expressive" not in df.columns:
        print("警告: 表 A 分片缺少 is_expressive 列（旧版分片），按 text_rules 同一口径现场推导")
        df["is_expressive"] = tr.is_expressive_series(df["post_type"], df["n_chars"])
    else:
        # astype(bool) 会把 NaN 静默变成 True，那等于凭空把一批帖子塞进分母。
        # 缺失说明这一列本身不可信，宁可显式失败也不能接着算。
        n_missing = int(df["is_expressive"].isna().sum())
        if n_missing:
            raise ValueError(
                f"表 A 的 is_expressive 列有 {n_missing} 行缺失，表达帖口径不可信，拒绝继续聚合"
            )
        df["is_expressive"] = df["is_expressive"].astype(bool)
    return df


def aggregate_posts(post_df):
    """表 A -> 用户级内容与活动指标"""
    df = post_df.copy()
    # 用 id_rules 而不是裸 astype(str)：user_id 列一旦因缺失值被 pandas
    # 上转型为 float64，裸 astype(str) 会产出 "123.0" 这种伪 ID，导致后续
    # 与表 B 聚合结果 join 时静默对不上（见 id_rules.py 顶部说明）。
    df["user_id"] = ir.normalize_id_series(df["user_id"])
    df = _ensure_is_expressive(df)
    df = _ensure_province_region(df)
    df["is_retweet_post"] = df["post_type"].isin(("retweet_plain", "retweet_comment"))

    base = df.groupby("user_id").agg(
        gender=("gender", "first"),
        n_posts=("weibo_id", "count"),
        n_retweets=("is_retweet_post", "sum"),
        n_expressive_posts=("is_expressive", "sum"),
        n_active_days=("date", "nunique"),
        n_active_months=("month", "nunique"),
        # province/region 取该用户在表 A 里首次出现的取值，不取众数：
        # 众数在票数相同时不确定、计算成本也更高，而绝大多数用户全年
        # 地理位置不变，首值已经足够代表；少数确有搬家的用户，取首值
        # 而非众数是刻意的简化，认领归属期外的偏差留给后续按需评估，
        # 不在这里用众数悄悄"多数表决"掉搬家这件事。
        province=("province", "first"),
        region=("region", "first"),
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

        # 并行口径：全部帖子为分母（13.1 分母稳健性用）。
        # 这个口径刻意保持"字面上的全部帖子"，不采纳零字符帖排除规则——
        # 它存在的意义就是与旧流水线口径可比，一旦跟着主口径改就失去比较价值。
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
    posts = _ensure_is_expressive(posts)
    posts["is_retweet_post"] = posts["post_type"].isin(("retweet_plain", "retweet_comment"))

    monthly = posts.groupby(["user_id", "month"]).agg(
        gender=("gender", "first"),
        n_posts=("weibo_id", "count"),
        n_retweets=("is_retweet_post", "sum"),
        n_expressive_posts=("is_expressive", "sum"),
        n_active_days=("date", "nunique"),
    )
    # 内容指标必须与表 C（aggregate_posts）用同一个表达帖子集：
    # 修复前这里的分子分母都取自全部帖子，而分母列名却是 n_expressive_posts，
    # 于是同名的 {d}_char_density / {d}_topical_share 在表 C 和表 D 里口径不同
    # （纯转发的"转发微博"有 4 个真实字符，会实打实地稀释表 D 的字符分母），
    # 且 topical_share 可能大于 1（分子数全部帖子、分母只数表达帖）。
    expressive = posts[posts["is_expressive"]]
    for domain in DOMAINS:
        hits = expressive[expressive[f"{domain}_hit"]].groupby(["user_id", "month"]).size()
        chars = expressive.groupby(["user_id", "month"])["n_chars"].sum()
        chars_hit = expressive.groupby(["user_id", "month"])[f"{domain}_chars_hit"].sum()
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


def _source_combo_series(public_entered, celebrity_entered):
    """向量化生成参与组合，避免在千万行量级上逐行 apply

    逐行 apply 会把每一行都拉回 Python 解释器执行一次函数调用；这里改成
    两个布尔掩码的组合，语义与原来的逐行判断完全一致。
    """
    # 用 eq(True) 而不是 fillna(False).astype(bool)：左连接留下的 NaN 在
    # object 列上 fillna 会触发 pandas 的静默降型（未来版本行为会变），
    # eq(True) 直接把 NaN/None 判成 False，语义相同且结果稳定是纯 bool
    public = public_entered.eq(True).to_numpy()
    celebrity = celebrity_entered.eq(True).to_numpy()
    combo = np.select(
        [public & celebrity, public & ~celebrity, ~public & celebrity],
        ["both", "public_only", "celebrity_only"],
        default="neither",
    )
    return pd.Series(combo, index=public_entered.index, dtype=object)


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
        # 左连接后没有转发事件的用户在这一列上是 NaN，等价于"没有进入"；
        # eq(True) 一步得到纯 bool 列，避免 object 列 fillna 的静默降型
        combined[f"{domain}_source_entered"] = combined[f"{domain}_source_entered"].eq(True)
        combined[f"{domain}_source_months"] = combined[f"{domain}_source_months"].fillna(0).astype(int)
        # 配置比例：该领域来源转发数 ÷ 用户全部转发帖数，上限截到 1
        share = _safe_divide(combined[f"{domain}_source_count"], combined["n_retweets"])
        combined[f"{domain}_source_share"] = share.clip(upper=1.0)
        # 持续性：参与月份数 ÷ 活跃月份数（分母含只转发不发帖的月份）
        combined[f"{domain}_source_month_share"] = _safe_divide(
            combined[f"{domain}_source_months"], combined["n_active_months_panel"]
        )

    combined["source_combo"] = _source_combo_series(
        combined["public_source_entered"], combined["celebrity_source_entered"]
    )
    return combined


def _read_shards(name, year, columns):
    """读取某一步的全年分片，返回 (合并后的 DataFrame, 分片文件列表)

    必须显式传 columns：表 A 的 {domain}_term_counts 是"词:次数"拼接字符串，
    表 C/表 D 一次都用不到，全年读进来只是白白占内存。
    """
    pattern = os.path.join(config.OUTPUT_DIR, f"{name}_{year}", "month=*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到分片: {pattern}")
    print(f"读取 {len(files)} 个 {name} 分片，列 {len(columns)} 个")
    frame = pd.concat(
        [pd.read_parquet(f, columns=columns) for f in files], ignore_index=True
    )
    return frame, files


def collect_shard_fingerprints(shard_dir, files):
    """从表 A/表 B 的分月 manifest 里收集词表/账号名单指纹

    表 C 和表 D 是论文数字的直接来源，却是全流程里唯一不记指纹的一步：
    它不自己加载词表和账号名单，指纹只能从上游分片的 manifest 继承过来。

    每个指纹项的值是"全年各月出现过的不同取值"排序去重后的列表：正常情况
    下应该只有一个元素（全年用同一份词表）；出现多个元素就说明不同月份用的
    词表/名单不一样，必须人工核实，而不是被一个取值悄悄代表。
    分月 manifest 缺失时显式记入 missing_manifests，绝不静默省略指纹——
    否则"没有指纹"和"指纹一致"在文件里长得一模一样。
    """
    collected = {}
    missing = []
    for path in files:
        shard_name = os.path.basename(path).replace(".parquet", "")
        month_part = shard_name.split("=")[-1]
        manifest_path = os.path.join(shard_dir, f"manifest_month_{month_part}", "manifest.json")
        if not os.path.exists(manifest_path):
            missing.append(os.path.basename(shard_dir) + f"/manifest_month_{month_part}")
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            shard_manifest = json.load(f)
        for key, value in (shard_manifest.get("fingerprints") or {}).items():
            collected.setdefault(key, set()).add(value)

    out = {key: sorted(values) for key, values in collected.items()}
    if missing:
        print(f"警告: {len(missing)} 个分片缺少 manifest，指纹无法继承: {missing[:5]}")
    # 即使为空也显式写出这个键，让"没有缺失"是一个被记录的事实
    out["missing_manifests"] = missing
    return out


def build(year=config.YEAR):
    """从表 A/B 分片生成表 C 与表 D"""
    posts, post_files = _read_shards("post_domain_measures", year, POST_SHARD_COLUMNS)
    events, event_files = _read_shards("retweet_domain_events", year, EVENT_SHARD_COLUMNS)
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

    post_dir = os.path.join(config.OUTPUT_DIR, f"post_domain_measures_{year}")
    event_dir = os.path.join(config.OUTPUT_DIR, f"retweet_domain_events_{year}")
    manifest = config.build_manifest(
        step=f"user_tables_{year}",
        # inputs 记录实际 glob 到的分片文件，而不是目录名：目录名说不清
        # 这次到底读到了哪几个月（少跑一个月和跑全了在目录名上看不出差别）
        inputs=[os.path.relpath(p, config.OUTPUT_DIR) for p in post_files + event_files],
        params={
            "year": year,
            "expressive_types": list(EXPRESSIVE_TYPES),
            "expressive_requires_nonzero_chars": True,
        },
        counts={
            "users": int(len(user_table)),
            "user_months": int(len(panel)),
            "gender": user_table["gender"].value_counts().to_dict(),
            "source_combo": user_table["source_combo"].value_counts().to_dict(),
            "event_only_users": event_only,
        },
        # 表 C/表 D 自己不加载词表和账号名单，指纹从上游分片 manifest 继承，
        # 让这两张（论文数字直接来源的）表也能自证用的是哪一版词表/名单
        fingerprints={
            "post_shards": collect_shard_fingerprints(post_dir, post_files),
            "retweet_shards": collect_shard_fingerprints(event_dir, event_files),
        },
    )
    config.write_manifest(manifest, os.path.join(config.OUTPUT_DIR, f"user_tables_{year}"))
    return user_path, panel_path


if __name__ == "__main__":
    fire.Fire({"build": build})
