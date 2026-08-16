"""
表 C 画像控制变量拼接：把 M2 模型（研究设计 §6.5）需要的账号画像变量
（认证状态、账号类型、粉丝数、关注数）左连接到表 C，province/region
在 build_user_tables.aggregate_posts 里已经从表 A 带过来，本模块不重复
处理，只是原样保留在输出里。

背景：表 C 目前只有性别与活动指标，M0/M1 两层模型够用，但 M2（加入认证
状态、账号类型、粉丝/关注数、地区）与 §13 的地区异质性分析都做不了。
这些控制变量来自 merged_profiles/merged_user_profiles.parquet，与表 C
不是同一条流水线产出的，天然存在"表 C 有的用户，画像表未必有"的缺口。

非负原则（研究协议 §11.4，两条不可谈判的约束）：
1. 左连接必须以表 C 为准（left-from-Table-C），绝不能因为在画像表里
   找不到而丢用户——M0/M1 依赖表 C 保持完整样本（225,339 用户，见
   §11.4），只有 M2 才通过 profile_complete 收窄样本。
2. 必须报告因收窄而损失的样本量，且要按性别拆开报告，不能笼统地给一个
   总数——这正是 attach_profile_controls 第二个返回值（loss report）
   存在的意义。

verified_flag 的取值来源与假设（判断，需要留痕）：
verified_type 在原始画像表里是一个编码（categorical code），不是布尔值，
但 M2 模型层（见 build_covariate_sets.py 的规划）需要的是一个可以直接
进回归的二值哑变量。本项目目前没有拿到微博官方对 verified_type 各取值
的权威码表，只能做保守假设：数值 -1（含各种能转成 -1 的形式）在微博的
常见编码习惯里代表"未认证"，其余非空取值（无论是个人认证的 "0" 还是
机构认证的其他正数编码）一律二值化为"已认证"。这是刻意选择的保守二值
方案，而不是尝试猜测并区分"个人认证/机构认证/大V"等多级含义——如果
后续拿到官方码表且发现与此假设不符（例如某些正数编码其实也表示未
认证），需要回来修正这里的映射，并重新跑一遍下游 M2 结果。
"""

import os

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gender_domain import config
from gender_domain import id_rules as ir

# 从画像表 merged_profiles/merged_user_profiles.parquet 里，本模块唯一
# 需要读取的列。该文件还带着 nick_name/location/ip_location 等本模块用
# 不到的列，parquet 读取必须显式指定 columns，不能整表读入。
PROFILE_COLUMNS = [
    "user_id",
    "verified_type",
    "user_type",
    "fans_number",
    "friends_count",
]

# 判定"该用户 M2 可用"的画像来源控制变量：即 verified_flag/log_fans/
# log_friends 三项全部非缺失。region 不在这里面——它来自表 C 自身（表 A
# 聚合而来），不是画像表拼接的产物，缺失与否由聚合环节负责，本模块不
# 重复校验，避免同一件事在两处判断出不一致的结果。
M2_PROFILE_CONTROLS = ("verified_flag", "log_fans", "log_friends")

# 报告里逐控制变量统计"有多少用户该项非缺失"时使用的列名清单，比
# M2_PROFILE_CONTROLS 多了 user_type：user_type 目前只是透传的描述性
# 字段，不进 M2 回归方程，但仍然值得在归因报告里单独看一眼缺失情况。
_CONTROLS_FOR_PRESENCE_REPORT = ("verified_flag", "user_type", "fans_number", "friends_count")


def _clean_nonneg_count(series):
    """清洗粉丝数/关注数：非数值或负数一律变成 NaN，绝不写成 0

    0 是合法的真实取值（粉丝数确实可以是 0），必须与"异常/未知"区分开；
    如果把负数或非数值填成 0，会把"缺失"悄悄伪装成"该用户没有粉丝"，
    后续 log1p(0)=0 会被当成一个真实观测值参与回归，而不是被当成缺失
    在 M2 完整性判断里显式排除。
    """
    if series is None:
        return pd.Series(dtype="float64")
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric >= 0)


def _verified_flag_series(verified_type):
    """把 verified_type 编码二值化为 verified_flag（含义与假设见模块顶部说明）

    - 缺失（用户没有匹配到画像行，或画像行本身 verified_type 为空）：
      结果为 <NA>，不是 False——"未知"和"确认未认证"是两件不同的事,
      不能混为一谈。
    - 能转成数值、且数值等于 -1：判定为未认证（False）。
    - 能转成数值、且数值不等于 -1：判定为已认证（True），不区分认证
      类型（个人/机构/大V 等）。
    - 不能转成数值但非空（预留给未来可能出现的字符串枚举）：保守判定
      为已认证（True），因为"有认证信息"本身通常意味着不是普通未认证
      账号；如与真实码表不符需要回来修正。
    """
    if verified_type is None:
        return pd.Series(dtype="boolean")
    numeric = pd.to_numeric(verified_type, errors="coerce")
    flag = pd.Series(pd.NA, index=verified_type.index, dtype="boolean")
    is_numeric = numeric.notna()
    flag[is_numeric] = numeric[is_numeric] != -1
    is_nonempty_non_numeric = (
        verified_type.notna() & ~is_numeric & (verified_type.astype(str).str.strip() != "")
    )
    flag[is_nonempty_non_numeric] = True
    return flag


def _build_loss_report(merged, matched):
    """构造按性别拆分的样本损失报告

    两个字段名不能弄混，这正是本报告存在的意义（§11.4 要求的归因报告）：
    - "profile_matched"：该用户是否在画像表里匹配到了记录（左连接是否
      命中），刻意取一个宽松的名字——不代表这条画像记录里每个字段都是
      干净可用的数值。用户即使匹配到画像，其中某个数值字段（比如
      friends_count 为负）异常时仍然算"matched"，因为我们确实拿到了这个
      用户的画像行，只是其中一项数值不可信、不能直接进回归。
    - "m2_ready"：严格判定，要求 verified_flag/log_fans/log_friends 三项
      M2 控制变量全部非缺失（M2_PROFILE_CONTROLS），即这一行是否真的能
      被 M2 回归使用，与 DataFrame 行级 profile_complete 列共用同一个
      严格定义。研究协议 §11.4 要求报告的"因收窄而损失的样本量"必须
      引用这一列，不能引用 profile_matched——后者会低估 M2 的实际损失
      （有画像但数值异常的用户，profile_matched 算"匹配"，只有 m2_ready
      会正确地把它算作不可用；写论文缺失数据小节时，"profile_matched: 1"
      不能被误读成"1 个用户可用于校正后的模型"）。
    """
    matched = pd.Series(matched, index=merged.index)
    m2_ready = merged["profile_complete"]

    def _stats_for(frame, frame_matched, frame_m2_ready):
        return {
            "users_total": int(len(frame)),
            "profile_matched": int(frame_matched.sum()),
            "controls_present": {
                ctrl: int(frame[ctrl].notna().sum()) for ctrl in _CONTROLS_FOR_PRESENCE_REPORT
            },
            "m2_ready": int(frame_m2_ready.sum()),
        }

    report = {
        "users_total": int(len(merged)),
        **_stats_for(merged, matched, m2_ready),
        "by_gender": {},
    }
    for gender, group in merged.groupby("gender", dropna=False):
        report["by_gender"][gender] = _stats_for(
            group, matched.loc[group.index], m2_ready.loc[group.index]
        )
    return report


def attach_profile_controls(user_df, profile_df):
    """左连接画像控制变量到表 C，返回 (拼接后的表, 按性别拆分的损失报告)

    左连接以 user_df（表 C）为准：user_df 里的每一个用户都会保留在结果
    里，即使在 profile_df 里完全找不到匹配记录——绝不因为缺画像而丢用户
    （见模块顶部的非负原则 1）。

    user_id 在两张表里都先经过 id_rules.normalize_id_series 归一化再
    join：profile_df 的 user_id 常见是整数列（见测试夹具），裸 astype(str)
    在有缺失值时会产出 "123.0" 这种伪 ID，导致 join 静默失效。
    """
    left = user_df.copy()
    left["user_id"] = ir.normalize_id_series(left["user_id"])

    # reindex 到 PROFILE_COLUMNS：profile_df 缺少某一列时（例如上游少给了
    # 一列）用全缺失列补齐，而不是直接 KeyError 中断整个拼接——这条画像
    # 表本来就不保证每个字段都存在，本函数的职责就是容忍这种不完整。
    right = profile_df.reindex(columns=PROFILE_COLUMNS).copy()
    right["user_id"] = ir.normalize_id_series(right["user_id"])
    # 画像表里同一 user_id 出现多次时只取第一条，避免左连接后行数膨胀
    # （表 C 本身已经是用户级唯一，拼接后不应该出现重复行）。
    right = right.drop_duplicates(subset="user_id", keep="first")

    merged = left.merge(right, on="user_id", how="left", indicator="_merge_indicator")
    matched = (merged["_merge_indicator"] == "both").to_numpy()
    merged = merged.drop(columns=["_merge_indicator"])

    merged["fans_number"] = _clean_nonneg_count(merged.get("fans_number"))
    merged["friends_count"] = _clean_nonneg_count(merged.get("friends_count"))
    merged["log_fans"] = np.log1p(merged["fans_number"])
    merged["log_friends"] = np.log1p(merged["friends_count"])
    merged["verified_flag"] = _verified_flag_series(merged.get("verified_type"))

    # profile_complete（这一列是严格定义：M2 三项画像控制变量全部非缺失，
    # 且这条记录必须真的在画像表里匹配到——未匹配的用户三项自然全是
    # 缺失，这里的 matched 检查是双保险，不依赖三项 NaN 判断本身）。
    # 这一列与下面 loss report 里的 "m2_ready" 共用同一个严格定义；loss
    # report 里另一个字段 "profile_matched" 是刻意更宽松的"是否匹配到
    # 画像行"，二者不是同一件事，见 _build_loss_report 的说明。
    complete = pd.Series(True, index=merged.index)
    for ctrl in M2_PROFILE_CONTROLS:
        complete &= merged[ctrl].notna()
    merged["profile_complete"] = matched & complete.to_numpy()

    report = _build_loss_report(merged, matched)
    return merged, report


def build(year=config.YEAR):
    """从表 C 与画像表拼接 M2 控制变量，落盘为 *_with_profile.parquet

    刻意不覆盖 user_domain_{year}.parquet 本体：表 C 原表仍然只有表 A/表
    B 聚合出的活动与领域参与指标，下游任何还依赖旧表 C 结构（不含画像
    列）的代码不会被这次改动破坏；新增列只出现在这份新文件里。
    """
    user_path = os.path.join(config.OUTPUT_DIR, f"user_domain_{year}.parquet")
    profile_path = os.path.join("merged_profiles", "merged_user_profiles.parquet")

    # 表 C 的列集合由 build_user_tables.combine_user_table 动态决定，这里
    # 没有一份现成的列清单可以复用；先读 schema 再把全部列名显式传给
    # columns=，既满足"parquet 读取必须显式指定 columns"的约定，又不需要
    # 在这里另外硬编码一份、迟早跟表 C 真实结构脱节的列清单。
    user_columns = pq.ParquetFile(user_path).schema.names
    user_df = pd.read_parquet(user_path, columns=user_columns)
    profile_df = pd.read_parquet(profile_path, columns=PROFILE_COLUMNS)
    print(f"读取表 C {len(user_df):,} 用户，画像表 {len(profile_df):,} 行")

    out, report = attach_profile_controls(user_df, profile_df)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, f"user_domain_{year}_with_profile.parquet")
    out.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"已保存: {out_path}（{len(out):,} 用户）")
    print(
        f"画像匹配 {report['profile_matched']:,} / {report['users_total']:,}，"
        f"M2 可用 {report['m2_ready']:,} / {report['users_total']:,}"
    )

    manifest = config.build_manifest(
        step=f"profile_join_{year}",
        inputs=[
            os.path.relpath(user_path, config.OUTPUT_DIR),
            profile_path,
        ],
        params={"year": year, "profile_columns": PROFILE_COLUMNS},
        # counts 里完整包含按性别拆分的损失报告，满足研究协议 §11.4：
        # 展示 M2 模型前必须能说清损失了多少用户、按性别是否有差异。
        counts=report,
    )
    config.write_manifest(manifest, os.path.join(config.OUTPUT_DIR, f"profile_join_{year}"))
    return out_path


if __name__ == "__main__":
    fire.Fire({"build": build})
