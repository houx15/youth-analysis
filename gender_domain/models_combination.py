"""
§7 来源/内容分解与 §8 参与组合：扩散与表达是不是同一件事，用户是否在
两个域之间"分工"。

前面几个模块（models_core / models_interaction）回答的都是"某个域上的
性别差有多大"。本模块回答两个它们回答不了的问题：

- **§7：转发来源账号（扩散）与自己写出带该领域词汇的帖子（表达）是不是
  同一种行为？** 两者可以各自发生，因此每个域上都有四种参与类型：
  两者都没有（neither）、只转发（source_only）、只表达（content_only）、
  两者都有（both）。
- **§8：用户是在两个域之间分工，还是"活跃的人什么都多做一点"？**
  四类 source_combo（neither / public_only / celebrity_only / both）上的
  多项 logit 给出离散版本的答案；两个域 topical_share 的相关（原始的与
  控制活跃度之后的）给出连续版本的答案。

必须写清楚的口径与裁定（每一条都是"换一种写法就会得到另一个数"的选择）：

1. **内容参与的门槛不是一个可以随手定的数。** 公共事务词表的命中在真实
   数据上接近饱和：全年有超过 93% 的用户至少命中过一个词。用"命中过就算
   参与"来定义内容参与，几乎所有人都是内容参与者，四分类会退化成"来源
   参与的两分类"，§7 的表就没有信息量了。因此本模块对每个域同时给出两类
   变体：`any_hit`（topical_share > 0）与若干个 `topical_share >= 阈值`
   的变体。阈值是**事先定好的一把梯子**（CONTENT_THRESHOLDS），全部跑、
   全部写进结果表，且阈值本身同时写进 `model` 列与 `note` 列——这样读者
   能自己看出结论对阈值的敏感性，而不是只能相信"我们选了 0.1"这句话。
   （这是 §7.1 明确要求的：阈值的选择必须是可见的、不是为了产出结论而挑的。）

2. **没有表达帖的用户，内容参与是"无定义"，不是"没有参与"。**
   topical_share 的分母是表达帖数，分母为 0 时表 C 里就是 NaN
   （项目全局约定：NaN is not zero）。这类用户在本模块的所有四分类与相关
   里一律剔除并计入 n_dropped（原因 REASON_NO_EXPRESSIVE），不当成
   "内容参与 = 否"。理由：把"一条表达帖都没写"和"写了很多但没写这个域"
   合并成同一格，会把一个纯粹的活跃度事实伪装成一个内容偏好事实。
   代价是 any_hit 变体与阈值变体共用同一个分母（都只用有定义的用户），
   这是刻意的：变体之间的差异必须只来自阈值本身，不能同时混进分母的变化。

3. **多项 logit 报告的是各类别上的性别 AME（概率尺度），不是系数。**
   多项 logit 的系数是"相对基准类别的对数发生比"，既依赖基准类别的选择，
   又在 M0/M1/M2 之间不可比（与 models_core 模块文档第 1 条同一个理由）。
   本模块对每个类别用反事实预测算 AME：把样本里每个人分别设成男性、
   女性，各自预测四类概率，逐类取均值之差。
   **四类 AME 相加必须约等于 0**：一个用户被推进某一格，必然从另一格里被
   推出来。这条恒等式是本模块最重要的内部一致性检查——如果 AME 是从系数
   里读出来的、或者某一类漏算了，它立刻破。结果表里专门有一行
   TERM_AME_SUM 把这个和写出来，代码里同时硬断言（超过 AME_SUM_TOL 直接
   报错），两处都不省。

4. **相关只能用两个量都有定义的用户来算，且必须报告剔除了多少人。**
   没有表达帖的用户两个域的 topical_share 同时是 NaN，把它们当成 0 会
   凭空制造一个"两个域都低"的点簇，从而制造出正相关。本模块所有相关
   （Spearman、偏 Spearman、phi）都只用两侧都有定义的用户，剔除数写进
   n_dropped 与 drop_reason。

5. **"控制活跃度之后"的相关用秩上的偏相关实现。**
   做法：把两个结果变量与全部活跃度控制变量都做秩变换，再把两个结果变量
   的秩对控制变量的秩做线性回归，取残差的 Pearson 相关。控制变量也做秩
   变换是必要的：活跃度与占比之间的关系是单调但非线性的，直接对原始尺度
   的活跃度做线性残差化会残留一大截曲率，看起来像"控制之后仍有相关"，
   其实只是没控制干净。
   这个量回答的正是 §8.2 的问题：如果原始相关为正、控制活跃度之后明显
   衰减，说明"做得多的人什么都多做一点"，而不是用户在两个域之间分工。

6. **§8.3 是一条禁令：不构造任何"新闻 vs 娱乐"的一维偏好分数。**
   不做差值指标、不做比值指标、不做任何"倾向性"变量，输出里没有，中间列
   里也没有。两个域并不穷尽微博内容、两个占比也不相加为常数，一个一维
   分数会把数据里根本不存在的"此消彼长"强加进来。需要同时看两个域时，
   本模块给的是二维的东西（四分类、四类别 AME、两个占比的相关），
   而不是把它们压成一个数。测试里有一条扫描输出与源码命名的守卫。

7. **失败必须留痕，样本量必须自洽。** 沿用 models_core 的约定：任何一层
   拟合不收敛/退化，都产出 estimate=NaN 且 note 写明原因的行，绝不少一行；
   n_obs + n_dropped 恒等于该行对应的输入用户数。

使用方法:
    python -m gender_domain.models_combination build --year=2020
"""

import os

import fire
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

from gender_domain import config
from gender_domain import describe
from gender_domain import models_core as mc
from gender_domain import models_interaction as mi
from gender_domain import stats_utils as su

DOMAINS = mc.DOMAINS
FOCAL_COLUMN = mc.FOCAL_COLUMN            # "male"，与 models_core 同一口径
DOMAIN_BOTH = mi.DOMAIN_BOTH              # 跨域的行在 domain 列上取 "both"

# §7.1 的四种参与类型（顺序固定，输出行按它排列）
PARTICIPATION_TYPES = ("neither", "source_only", "content_only", "both")
# §8.1 的四类参与组合，与 build_user_tables._source_combo_series 一致；
# 顺序同时决定多项 logit 的基准类别（第一个，即 "neither"）
COMBO_CATEGORIES = ("neither", "public_only", "celebrity_only", "both")
COMBO_COLUMN = "source_combo"

# 内容参与的两类口径：any_hit 与"占比 >= 阈值"。
# 阈值是事先定好的梯子（见模块文档第 1 条），不是看完结果再挑的：
# 0.05 / 0.10 / 0.20 覆盖"偶尔写到"到"该领域占了自己表达帖的五分之一"，
# 跨越了公共事务域上 topical_share 的中位数附近，足以让读者看出结论
# 是随阈值连续变化还是只在某一个阈值上成立。
VARIANT_ANY_HIT = "any_hit"
CONTENT_THRESHOLDS = (0.05, 0.10, 0.20)

REASON_NO_EXPRESSIVE = "no_expressive_posts"
REASON_UNKNOWN_COMBO = "unknown_source_combo"
REASON_MISSING_ACTIVITY = "missing_activity_control"

# 活跃度控制变量：就是 M1 相对 M0 多出来的那几个，不另立一套
ACTIVITY_CONTROLS = tuple(c for c in mc.M1 if c != "gender")

# 结果表里的 outcome / term 命名
OUTCOME_PARTICIPATION = "participation_type"
OUTCOME_CROSS = "source_content_cross"
OUTCOME_CONTENT_ON_SOURCE = "content_participation"
OUTCOME_COMBO = "source_combo"
OUTCOME_CONTENT_CORR = "topical_share_correlation"

TERM_GENDER_AME = mc.TERM_AME                       # "gender_male"
TERM_SOURCE_AME = "source_entered_ame"
TERM_SOURCE_X_GENDER = "source_x_gender_did"
TERM_SOURCE_X_GENDER_COEF = "source_x_gender_coef"
TERM_AME_SUM = "gender_male_ame_sum"

MODEL_RAW = "raw"
MODEL_ACTIVITY_PARTIAL = "activity_partial"

_CONFIDENCE = 0.95
_Z = float(norm.ppf(1 - (1 - _CONFIDENCE) / 2))
# 相关系数与 phi 的百分位 bootstrap 次数。实测在 22.5 万用户上，一次
# Spearman 约 30 毫秒、一次秩偏相关约 60 毫秒，500 次重抽样对单个统计量
# 约 15~30 秒；本模块最多有 (3 组 × 2 种相关) × (2 域 + 跨域) 个这样的量，
# 合计约 10 分钟，在 4 小时的作业里是可以接受的开销。
_N_BOOT = 500
_SEED = 20200101

# 四类 AME 相加的容差（见模块文档第 3 条）。理论上恒等于 0（每行四类
# 预测概率相加为 1），留一点数值容差只是给浮点求和，超过就是实现错了。
AME_SUM_TOL = 1e-6


# ---------------------------------------------------------------------------
# §7.1 四分类
# ---------------------------------------------------------------------------

def variant_label(content_threshold=None):
    """内容参与口径的标签，写进结果表的 model 列

    用 model 列而不是 note 列承载口径，是因为下游按
    (outcome, domain, model, term) 取行必须唯一：几个阈值变体的 term 完全
    相同，如果口径只写在 note 里，同一个 key 会对应多行。
    """
    if content_threshold is None:
        return VARIANT_ANY_HIT
    return "share_ge_{:g}".format(float(content_threshold))


def variant_note(content_threshold=None):
    """把阈值本身写进 note，使结果表脱离代码也能自证口径（§7.1 的要求）"""
    if content_threshold is None:
        return "content_threshold=any_hit"
    return "content_threshold={:g}".format(float(content_threshold))


def content_participation(user_df, domain, content_threshold=None):
    """内容参与指示列：float 的 1.0/0.0，无定义时为 NaN

    定义在 `{domain}_topical_share` 上（分子分母都在表达帖上计算）：
        content_threshold is None -> share > 0（any_hit）
        否则                       -> share >= 阈值
    分母为 0（没有表达帖）时表 C 里 share 是 NaN，这里原样保留 NaN，
    不折成 0（模块文档第 2 条）。
    """
    column = f"{domain}_topical_share"
    if column not in user_df.columns:
        raise KeyError(f"表 C 缺少列 {column}，无法定义 {domain} 域的内容参与")
    share = pd.to_numeric(user_df[column].astype("object"), errors="coerce").astype(float)
    # 占比越界只可能是上游分子分母算错了，必须在这里炸掉而不是继续算
    su.check_proportion_range(share, column)
    if content_threshold is None:
        flag = share > 0
    else:
        flag = share >= float(content_threshold)
    return pd.Series(np.where(share.isna(), np.nan, flag.astype(float)),
                     index=user_df.index, dtype=float)


def source_participation(user_df, domain):
    """来源参与指示列：float 的 1.0/0.0

    `{domain}_source_entered` 在表 C 里是纯 bool（左连接后的缺失已经被
    eq(True) 判成 False），对每个用户都有定义，因此这里不会产生 NaN。
    但本模块的调用方有两种：直接拿表 C（bool 列）和拿
    models_core.prepare_model_frame 的输出（同一列已经被转成 0/1 的 float）。
    所以这里统一走"转数值再判正"，两种输入都得到同一个 0/1 float 列，
    而不是依赖 bool 与 float 在 `.eq(True)` 下恰好等价这件事。
    """
    column = f"{domain}_source_entered"
    if column not in user_df.columns:
        raise KeyError(f"表 C 缺少列 {column}")
    numeric = pd.to_numeric(user_df[column].astype("object"), errors="coerce")
    return (numeric.fillna(0) > 0).astype(float)


def classify_participation(user_df, domain, content_threshold=None):
    """§7.1 四分类：neither / source_only / content_only / both

    Returns:
        与 user_df 同索引的 Series；内容参与无定义的用户取 NaN
        （不归入任何一格，见模块文档第 2 条）。非缺失部分互斥且穷尽。
    """
    source = source_participation(user_df, domain).to_numpy() > 0
    content_raw = content_participation(user_df, domain, content_threshold)
    defined = content_raw.notna().to_numpy()
    content = content_raw.fillna(0).to_numpy() > 0

    labels = np.select(
        [source & content, source & ~content, ~source & content],
        ["both", "source_only", "content_only"],
        default="neither",
    ).astype(object)
    labels[~defined] = np.nan
    return pd.Series(labels, index=user_df.index, dtype=object)


def _gender_masks(user_df):
    """(标签, 掩码) 列表：male / female

    表 C 上游已经过滤掉 gender 缺失的行，这里仍然只按 'm'/'f' 取子集、
    不做 else 分支——出现其它取值时它不会被静默算进任何一组。
    """
    gender = user_df["gender"].astype("object")
    return [("male", gender == "m"), ("female", gender == "f")]


def participation_types(user_df, domain, content_threshold=None):
    """§7.1 四种参与类型在各性别下的占比，以及性别差（百分点）

    每个 (类型 × 性别) 一行（Wilson 区间），外加每个类型一行男女差
    （Newcombe 区间）。同一个 domain 上 any_hit 与各阈值变体各跑一次，
    口径写在 model 列与 note 列里。
    """
    labels = classify_participation(user_df, domain, content_threshold)
    model = variant_label(content_threshold)
    note = variant_note(content_threshold)
    rows = []

    grouped = {}
    for gender_label, mask in _gender_masks(user_df):
        sub = labels[mask]
        valid = sub.dropna()
        grouped[gender_label] = {
            "n_total": int(mask.sum()),
            "n_obs": int(len(valid)),
            "counts": valid.value_counts(),
        }

    for ptype in PARTICIPATION_TYPES:
        for gender_label in ("male", "female"):
            info = grouped[gender_label]
            n_obs = info["n_obs"]
            n_dropped = info["n_total"] - n_obs
            successes = int(info["counts"].get(ptype, 0))
            if n_obs:
                estimate = successes / n_obs
                low, high = su.proportion_ci(successes, n_obs)
            else:
                estimate, low, high = np.nan, np.nan, np.nan
            rows.append(su.tidy_result(
                outcome=OUTCOME_PARTICIPATION, domain=domain, model=model,
                term=f"{ptype}_{gender_label}", estimate=estimate, se=np.nan,
                ci_low=low, ci_high=high, scale="probability",
                n_obs=n_obs, n_dropped=n_dropped,
                drop_reason=mi.format_drop_reason(
                    [(REASON_NO_EXPRESSIVE, n_dropped)] if n_dropped else []
                ),
                note=mc._join_notes(note, "wilson"),
            ))

        m, f = grouped["male"], grouped["female"]
        if m["n_obs"] and f["n_obs"]:
            diff, low, high = su.proportion_diff_ci(
                int(m["counts"].get(ptype, 0)), m["n_obs"],
                int(f["counts"].get(ptype, 0)), f["n_obs"],
            )
        else:
            diff, low, high = np.nan, np.nan, np.nan
        n_dropped = (m["n_total"] - m["n_obs"]) + (f["n_total"] - f["n_obs"])
        rows.append(su.tidy_result(
            outcome=OUTCOME_PARTICIPATION, domain=domain, model=model,
            term=f"{ptype}_gap_pp", estimate=diff, se=np.nan, ci_low=low, ci_high=high,
            scale="probability", n_obs=m["n_obs"] + f["n_obs"], n_dropped=n_dropped,
            drop_reason=mi.format_drop_reason(
                [(REASON_NO_EXPRESSIVE, n_dropped)] if n_dropped else []
            ),
            note=mc._join_notes(note, "newcombe"),
        ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# §7.2 交叉关系：条件占比、phi、（偏）Spearman
# ---------------------------------------------------------------------------

def phi_coefficient(a, b, c, d):
    """2×2 表的 phi 系数

        phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))

    其中 a=两者都有、b=只有行变量、c=只有列变量、d=两者都没有。
    任一边际为 0 时分母为 0，phi 无定义，返回 NaN（这是一个真实的数据
    事实——某一行为在样本里从未发生——不该让整张表算不出来）。
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom <= 0:
        return np.nan
    return (a * d - b * c) / np.sqrt(denom)


def _phi_from_frame(frame):
    """在一份两列（source / content）的 0/1 表上算 phi，供 bootstrap 调用"""
    source = frame["source"].to_numpy() > 0
    content = frame["content"].to_numpy() > 0
    a = int(np.sum(source & content))
    b = int(np.sum(source & ~content))
    c = int(np.sum(~source & content))
    d = int(np.sum(~source & ~content))
    return phi_coefficient(a, b, c, d)


def _rank(values):
    return pd.Series(np.asarray(values, dtype=float)).rank().to_numpy()


def spearman(x, y):
    """Spearman 秩相关（用秩上的 Pearson 相关实现，与 scipy 定义一致）

    自己实现而不是每次调用 scipy.stats.spearmanr，是因为 bootstrap 里要
    调用几十万次，scipy 版本每次还会顺带算 p 值（本研究根本不报告 p 值）。
    两者在无并列时完全相同，有并列时也一致——scipy 的定义就是"秩上的
    Pearson 相关"。
    """
    rx, ry = _rank(x), _rank(y)
    if len(rx) < 3 or np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, controls):
    """控制活跃度之后的秩偏相关（见模块文档第 5 条）

    x、y 与全部控制变量都先做秩变换，再把 rank(x)、rank(y) 分别对控制
    变量的秩做线性回归，取残差的 Pearson 相关。控制变量也做秩变换是关键：
    活跃度与占比之间是单调但非线性的关系，直接在原始尺度上做线性残差化
    会残留曲率，看起来像"控制之后仍然相关"，其实只是没控制干净。
    """
    rx, ry = _rank(x), _rank(y)
    ctrl = np.asarray(controls, dtype=float)
    if ctrl.ndim == 1:
        ctrl = ctrl[:, None]
    if len(rx) < ctrl.shape[1] + 3:
        return np.nan
    ranked = np.column_stack([_rank(ctrl[:, j]) for j in range(ctrl.shape[1])])
    design = np.column_stack([np.ones(len(rx)), ranked])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def _bootstrap_row(frame, statistic, outcome, domain, model, term, scale,
                   n_obs, n_dropped, drop_reason, note, n_boot=_N_BOOT, seed=_SEED):
    """点估计 + 百分位 bootstrap 区间的一行；样本为空时留一行 NaN"""
    if len(frame) == 0:
        return su.tidy_result(
            outcome=outcome, domain=domain, model=model, term=term,
            estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, scale=scale,
            n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
            note=mc._join_notes(note, "empty_sample"),
        )
    est, low, high = su.bootstrap_ci(
        frame, statistic, n_boot=n_boot, seed=seed, confidence=_CONFIDENCE
    )
    return su.tidy_result(
        outcome=outcome, domain=domain, model=model, term=term,
        # 百分位 bootstrap 的区间不对称，反推 se 会丢掉区间形状
        # （stats_utils 模块文档第 5 条），因此 se 留 NaN
        estimate=est, se=np.nan, ci_low=low, ci_high=high, scale=scale,
        n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
        note=mc._join_notes(note, "bootstrap"),
    )


def _activity_matrix(frame):
    """活跃度控制矩阵（只取本样本里确实有变化的列）"""
    cols = [c for c in ACTIVITY_CONTROLS
            if c in frame.columns and frame[c].nunique(dropna=True) > 1]
    if not cols:
        return None, []
    return frame[cols].astype(float).to_numpy(), cols


def cross_relations(user_df, domain, content_threshold=None, include_correlations=True,
                    n_boot=_N_BOOT, seed=_SEED):
    """§7.2 扩散与表达的交叉关系

    每个性别组（male / female / all）产出：
        content_given_source_*   来源参与者里有内容参与的占比（Wilson）
        source_given_content_*   内容参与者里有来源参与的占比（Wilson）
        phi_*                    2×2 的 phi 系数（bootstrap 区间）
        spearman_source_count_topical_share_*          来源转发条数与
                                 topical_share 的 Spearman 相关
        spearman_source_count_topical_share_partial_*  同上，控制活跃度

    Args:
        include_correlations: 两条 Spearman 行只依赖连续变量，与内容参与
            的二分口径无关，跑多个阈值变体时它们每次都会得到同一个数。
            build() 因此只在 any_hit 变体上把它们算出来，其余变体传 False，
            避免把同一个 bootstrap 重复跑五遍。
    """
    model = variant_label(content_threshold)
    note = variant_note(content_threshold)
    source = source_participation(user_df, domain)
    content = content_participation(user_df, domain, content_threshold)
    count_col = f"{domain}_source_count"
    share_col = f"{domain}_topical_share"

    groups = _gender_masks(user_df) + [("all", pd.Series(True, index=user_df.index))]
    rows = []
    for gender_label, mask in groups:
        n_total = int(mask.sum())
        defined = mask & content.notna()
        n_obs_defined = int(defined.sum())
        n_dropped = n_total - n_obs_defined
        drop_reason = mi.format_drop_reason(
            [(REASON_NO_EXPRESSIVE, n_dropped)] if n_dropped else []
        )

        src = source[defined].to_numpy() > 0
        cnt = content[defined].to_numpy() > 0
        a = int(np.sum(src & cnt))
        b = int(np.sum(src & ~cnt))
        c = int(np.sum(~src & cnt))
        d = int(np.sum(~src & ~cnt))

        # 来源参与者里有内容参与的占比：分母是来源参与者
        n_source = a + b
        est = a / n_source if n_source else np.nan
        low, high = su.proportion_ci(a, n_source) if n_source else (np.nan, np.nan)
        rows.append(su.tidy_result(
            outcome=OUTCOME_CROSS, domain=domain, model=model,
            term=f"content_given_source_{gender_label}", estimate=est, se=np.nan,
            ci_low=low, ci_high=high, scale="probability",
            n_obs=n_source, n_dropped=n_dropped, drop_reason=drop_reason,
            note=mc._join_notes(note, "wilson", "denominator=source_participants"),
        ))

        # 内容参与者里有来源参与的占比：分母是内容参与者
        n_content = a + c
        est = a / n_content if n_content else np.nan
        low, high = su.proportion_ci(a, n_content) if n_content else (np.nan, np.nan)
        rows.append(su.tidy_result(
            outcome=OUTCOME_CROSS, domain=domain, model=model,
            term=f"source_given_content_{gender_label}", estimate=est, se=np.nan,
            ci_low=low, ci_high=high, scale="probability",
            n_obs=n_content, n_dropped=n_dropped, drop_reason=drop_reason,
            note=mc._join_notes(note, "wilson", "denominator=content_participants"),
        ))

        # phi：2×2 的相关，bootstrap 区间（每个用户一行，无需整簇重抽样）
        two_by_two = pd.DataFrame({"source": src.astype(float),
                                   "content": cnt.astype(float)})
        # bootstrap_ci 的点估计是在**原始未重抽样样本**上算出来的，因此
        # 这一行的 estimate 与读者按 2×2 计数手算的 phi 完全一致，
        # bootstrap 只负责区间
        rows.append(_bootstrap_row(
            two_by_two, _phi_from_frame, OUTCOME_CROSS, domain, model,
            f"phi_{gender_label}", "correlation", n_obs_defined, n_dropped,
            drop_reason, note, n_boot=n_boot, seed=seed,
        ))

        if not include_correlations:
            continue

        corr_frame = pd.DataFrame({
            "x": pd.to_numeric(user_df.loc[defined, count_col], errors="coerce").astype(float).to_numpy(),
            "y": pd.to_numeric(user_df.loc[defined, share_col], errors="coerce").astype(float).to_numpy(),
        })
        controls, control_cols = _activity_matrix(user_df.loc[defined])
        for j, name in enumerate(control_cols):
            corr_frame[f"ctrl_{j}"] = controls[:, j]
        ctrl_names = [f"ctrl_{j}" for j in range(len(control_cols))]

        rows.append(_bootstrap_row(
            corr_frame, lambda f: spearman(f["x"], f["y"]),
            OUTCOME_CROSS, domain, model,
            f"spearman_source_count_topical_share_{gender_label}", "correlation",
            n_obs_defined, n_dropped, drop_reason, note, n_boot=n_boot, seed=seed,
        ))

        if ctrl_names:
            rows.append(_bootstrap_row(
                corr_frame,
                lambda f: partial_spearman(f["x"], f["y"], f[ctrl_names].to_numpy()),
                OUTCOME_CROSS, domain, model,
                f"spearman_source_count_topical_share_partial_{gender_label}",
                "correlation", n_obs_defined, n_dropped, drop_reason,
                mc._join_notes(note, "controls:" + ",".join(control_cols)),
                n_boot=n_boot, seed=seed,
            ))
        else:
            rows.append(su.tidy_result(
                outcome=OUTCOME_CROSS, domain=domain, model=model,
                term=f"spearman_source_count_topical_share_partial_{gender_label}",
                estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan,
                scale="correlation", n_obs=n_obs_defined, n_dropped=n_dropped,
                drop_reason=drop_reason,
                note=mc._join_notes(note, "no_activity_controls_available"),
            ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# 反事实预测的公共件（本模块内两处用到：§7.3 的交互、§8.1 的多项 AME）
# ---------------------------------------------------------------------------

def _forced_exog(design_info, data, assignments):
    """把若干列强制取定值后，用拟合时的 design_info 重建设计矩阵

    与 stats_utils._design_matrices / models_interaction._cell_exog 同一思路
    （复用 design_info 保证列顺序与哑变量编码完全一致），区别只是这里允许
    一次强制任意多列，服务"来源参与 × 性别"的四格反事实。
    """
    frame = data.copy()
    for column, value in assignments.items():
        frame[column] = float(value)
    return np.asarray(patsy.dmatrix(design_info, frame, return_type="dataframe"))


def _delta_method_se(flat_params, cov, func, eps=1e-6):
    """对返回向量的函数做 delta method，返回逐分量的标准误

    数值中心差分求梯度（对 link function 不敏感，换模型不用重写），
    Var_c = grad_c' cov grad_c。方差非有限或为负时报错而不是悄悄换个
    公式——与 stats_utils.average_marginal_effect 同一条原则。
    """
    base = np.atleast_1d(np.asarray(func(flat_params), dtype=float))
    grad = np.zeros((len(flat_params), base.size))
    for j in range(len(flat_params)):
        step = eps * max(1.0, abs(flat_params[j]))
        p_plus = flat_params.copy()
        p_plus[j] += step
        p_minus = flat_params.copy()
        p_minus[j] -= step
        grad[j] = (np.atleast_1d(func(p_plus)) - np.atleast_1d(func(p_minus))) / (2 * step)
    var = np.einsum("jc,jk,kc->c", grad, cov, grad)
    if not np.isfinite(var).all() or (var < 0).any():
        raise ValueError(
            f"delta method 得到无效方差 var={var!r}：协方差矩阵可能含 NaN 或"
            "不是半正定矩阵，这是模型本身有问题，必须先修好，不允许静默退化"
            "成别的区间来掩盖。"
        )
    return base, np.sqrt(var)


# ---------------------------------------------------------------------------
# §7.3 内容参与 ~ 来源参与 × 性别
# ---------------------------------------------------------------------------

def _content_frame(user_df, domain, content_threshold):
    """建模帧：在 models_core.prepare_model_frame 之上补内容/来源参与列"""
    frame = mc.prepare_model_frame(user_df)
    frame["content_flag"] = content_participation(frame, domain, content_threshold)
    frame["source_flag"] = source_participation(frame, domain)
    return frame


def _blocked_note(sample, width_terms, outcome_col=None):
    """拟合之前就能判定"这层没法估"的情形（与 models_core._blocked_note 同规则）

    单独写一份而不是直接调用 models_core 的版本，只是因为这里的设计矩阵
    宽度要把交互项与来源参与列一起算进去；判定规则本身完全一致。
    """
    if len(sample) == 0:
        return "empty_sample"
    if sample[FOCAL_COLUMN].nunique() < 2:
        return "no_gender_variation"
    if outcome_col is not None and sample[outcome_col].nunique(dropna=True) < 2:
        return "no_outcome_variation"
    if len(sample) <= mc._design_width(width_terms, sample):
        return "insufficient_observations"
    return None


def fit_content_on_source(user_df, domain, content_threshold=None):
    """§7.3 内容参与 ~ 来源参与 × 性别 + 活跃度/画像控制（logistic，M0/M1/M2）

    每层四行：
        TERM_SOURCE_AME        来源参与的平均边际效应（概率尺度）——
                               "转发过该领域来源的人，写该领域内容的概率高多少"
        TERM_GENDER_AME        性别的平均边际效应（概率尺度）
        TERM_SOURCE_X_GENDER   来源参与效应的男女之差（概率尺度的差中差），
                               四格反事实预测 + delta method
        TERM_SOURCE_X_GENDER_COEF  交互项系数（log_odds，附加参考，不是头条）

    交互不读系数而是算概率尺度的差中差，理由与 models_interaction 模块
    文档第 2 条完全一致：logit 尺度的交互系数不是"概率差之差"，也不在
    M0/M1/M2 之间可比。
    """
    n_input = len(user_df)
    frame = _content_frame(user_df, domain, content_threshold)
    defined = frame["content_flag"].notna()
    model_note = variant_note(content_threshold)
    rows = []

    for layer, covariates in mc.MODEL_LAYERS:
        sample, n_obs, n_dropped, drop_reason = mc._layer_sample(
            frame, layer, n_input, base_mask=defined, base_reason=REASON_NO_EXPRESSIVE,
        )
        terms, dropped = mc._formula_terms(covariates, sample)
        others = [t for t in terms if t != FOCAL_COLUMN]
        width_terms = [FOCAL_COLUMN, "source_flag", "male_x_source"] + others
        base_note = mc._join_notes(
            model_note,
            "dropped_constant:" + ",".join(dropped) if dropped else None,
        )

        blocked = _blocked_note(sample, width_terms, outcome_col="content_flag")
        if blocked is None and sample["source_flag"].nunique() < 2:
            blocked = "no_source_variation"
        if blocked is not None:
            rows.extend(_content_nan_rows(domain, layer, n_obs, n_dropped, drop_reason,
                                          mc._join_notes(base_note, blocked)))
            continue

        rhs = " + ".join(["{} * source_flag".format(FOCAL_COLUMN)] + others)
        formula = "content_flag ~ {}".format(rhs)
        label = f"{OUTCOME_CONTENT_ON_SOURCE}/{domain}/{layer}"
        result, fail = mc._safe_fit(
            lambda: smf.logit(formula, data=sample).fit(
                cov_type="HC1", disp=False, maxiter=100
            ),
            label,
        )
        if result is None:
            rows.extend(_content_nan_rows(domain, layer, n_obs, n_dropped, drop_reason,
                                          mc._join_notes(base_note, fail)))
            continue

        n_obs, n_dropped, drop_reason = mc._reconcile_nobs(
            result, n_input, n_obs, n_dropped, drop_reason
        )
        note = base_note

        for term, focal in ((TERM_GENDER_AME, FOCAL_COLUMN),
                            (TERM_SOURCE_AME, "source_flag")):
            ame, se, low, high = su.average_marginal_effect(
                result, focal, sample, confidence=_CONFIDENCE
            )
            rows.append(su.tidy_result(
                outcome=OUTCOME_CONTENT_ON_SOURCE, domain=domain, model=layer,
                term=term, estimate=ame, se=se, ci_low=low, ci_high=high,
                scale="probability", n_obs=n_obs, n_dropped=n_dropped,
                drop_reason=drop_reason, note=note,
            ))

        did, se, low, high = _source_by_gender_contrast(result, sample)
        rows.append(su.tidy_result(
            outcome=OUTCOME_CONTENT_ON_SOURCE, domain=domain, model=layer,
            term=TERM_SOURCE_X_GENDER, estimate=did, se=se, ci_low=low, ci_high=high,
            scale="probability", n_obs=n_obs, n_dropped=n_dropped,
            drop_reason=drop_reason,
            note=mc._join_notes(note, "counterfactual_four_cells"),
        ))

        coef_name = "{}:source_flag".format(FOCAL_COLUMN)
        coef = float(result.params[coef_name])
        bse = float(result.bse[coef_name])
        rows.append(su.tidy_result(
            outcome=OUTCOME_CONTENT_ON_SOURCE, domain=domain, model=layer,
            term=TERM_SOURCE_X_GENDER_COEF, estimate=coef, se=bse,
            ci_low=coef - _Z * bse, ci_high=coef + _Z * bse, scale="log_odds",
            n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
            note=mc._join_notes(note, "not_the_headline"),
        ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


def _content_nan_rows(domain, layer, n_obs, n_dropped, drop_reason, note):
    """一层拟合不成立时的四行留痕（与 models_core 第 7 条同一原则）"""
    specs = (
        (TERM_GENDER_AME, "probability"),
        (TERM_SOURCE_AME, "probability"),
        (TERM_SOURCE_X_GENDER, "probability"),
        (TERM_SOURCE_X_GENDER_COEF, "log_odds"),
    )
    return [
        mc._nan_row(OUTCOME_CONTENT_ON_SOURCE, domain, layer, term, scale,
                    n_obs, n_dropped, drop_reason, note)
        for term, scale in specs
    ]


def _source_by_gender_contrast(fit, sample):
    """来源参与效应的男女之差（概率尺度），四格反事实 + delta method

        [(P|来源=1) - (P|来源=0)]_男 - [(P|来源=1) - (P|来源=0)]_女
    """
    design_info = getattr(fit.model.data, "design_info", None)
    if design_info is None:
        raise ValueError("logit 结果里没有 design_info，无法构造反事实设计矩阵")
    exogs = {
        (male, src): _forced_exog(design_info, sample,
                                  {FOCAL_COLUMN: male, "source_flag": src})
        for male in (0, 1) for src in (0, 1)
    }
    params = np.asarray(fit.params, dtype=float)

    def _contrast(p):
        preds = {k: float(np.mean(fit.model.predict(p, e))) for k, e in exogs.items()}
        return np.array([(preds[(1, 1)] - preds[(1, 0)]) - (preds[(0, 1)] - preds[(0, 0)])])

    cov = su._get_cov_params_or_none(fit)
    if cov is None:
        print("警告: 来源参与 × 性别的差中差拿不到协方差矩阵，区间记为 NaN")
        return float(_contrast(params)[0]), np.nan, np.nan, np.nan
    est, se = _delta_method_se(params, cov, _contrast)
    return float(est[0]), float(se[0]), float(est[0] - _Z * se[0]), float(est[0] + _Z * se[0])


# ---------------------------------------------------------------------------
# §8.1 四类参与组合的多项 logit
# ---------------------------------------------------------------------------

def combo_ame_term(category):
    """某一类参与组合上的性别 AME 的 term 名"""
    return f"gender_male_ame_{category}"


def combo_ame_terms(categories=COMBO_CATEGORIES):
    return tuple(combo_ame_term(c) for c in categories)


def _multinomial_ames(result, design_info, sample, categories):
    """多项 logit 的性别 AME（逐类别）+ delta method 标准误

    做法与 stats_utils.average_marginal_effect 完全同源，只是结果是向量：
    把样本里每个人分别设成男性、女性，各自预测各类别概率，逐类取均值之差。
    **不是**读系数（多项 logit 的系数是相对基准类别的对数发生比，既依赖
    基准类别、又在 M0/M1/M2 之间不可比）。

    statsmodels 的 MNLogit 参数是 (k_exog, J-1) 的二维数组，而 cov_params()
    是按列优先（order="F"）展平后的方阵——本函数内部一律用展平向量，
    只在调用 predict 之前 reshape 回去，两处顺序必须一致，否则 delta method
    会把不同方程的方差张冠李戴。
    """
    params_2d = np.asarray(result.params, dtype=float)
    k_exog, n_eq = params_2d.shape
    flat = params_2d.ravel(order="F")

    exog1 = _forced_exog(design_info, sample, {FOCAL_COLUMN: 1})
    exog0 = _forced_exog(design_info, sample, {FOCAL_COLUMN: 0})

    def _ame(p):
        reshaped = np.asarray(p, dtype=float).reshape(k_exog, n_eq, order="F")
        pred1 = result.model.predict(reshaped, exog=exog1)
        pred0 = result.model.predict(reshaped, exog=exog0)
        return np.asarray(pred1).mean(axis=0) - np.asarray(pred0).mean(axis=0)

    cov = su._get_cov_params_or_none(result)
    if cov is None:
        ames = _ame(flat)
        ses = np.full(len(categories), np.nan)
        print("警告: 多项 logit 拿不到 cov_params()，各类别 AME 的区间记为 NaN")
    else:
        ames, ses = _delta_method_se(flat, cov, _ame)

    ame_sum = float(np.sum(ames))
    if abs(ame_sum) > AME_SUM_TOL:
        raise ValueError(
            f"四类参与组合上的性别 AME 相加为 {ame_sum!r}，超出容差 {AME_SUM_TOL}："
            "一个用户被推进某一格必然从另一格里被推出来，这个和在数学上恒等于 0。"
            "出现非零值说明边际效应算错了（例如读了系数、漏了某一类、或者参数"
            "展平顺序与协方差矩阵不一致），不允许把这样的数字写进结果表。"
        )
    return np.asarray(ames, dtype=float), np.asarray(ses, dtype=float), ame_sum


def _combo_nan_rows(layer, n_obs, n_dropped, drop_reason, note,
                    categories=COMBO_CATEGORIES):
    rows = [
        mc._nan_row(OUTCOME_COMBO, DOMAIN_BOTH, layer, combo_ame_term(c),
                    "probability", n_obs, n_dropped, drop_reason, note)
        for c in categories
    ]
    rows.append(mc._nan_row(OUTCOME_COMBO, DOMAIN_BOTH, layer, TERM_AME_SUM,
                            "probability", n_obs, n_dropped, drop_reason,
                            mc._join_notes(note, "internal_consistency_check")))
    return rows


def fit_combination_multinomial(user_df):
    """§8.1 四类 source_combo 上的多项 logit，报告各类别的性别 AME（M0/M1/M2）

    每层五行：四个类别各一行 AME（概率尺度），外加一行 TERM_AME_SUM 写出
    四者之和——它在数学上恒等于 0，写进结果表是为了让读者能一眼确认这批
    边际效应是"同一次移动的四个去向"，而不是四个各自算出来的数字
    （代码里同时硬断言，见 _multinomial_ames）。
    """
    n_input = len(user_df)
    frame = mc.prepare_model_frame(user_df)
    if COMBO_COLUMN not in frame.columns:
        raise KeyError(
            f"表 C 缺少 {COMBO_COLUMN} 列：§8.1 的四类参与组合由 build_user_tables"
            "已经算好，本模块不重新推导它（全局约束：不在 results 层重新造度量）。"
        )
    combo = frame[COMBO_COLUMN].astype("object")
    known = combo.isin(COMBO_CATEGORIES)

    rows = []
    for layer, covariates in mc.MODEL_LAYERS:
        sample, n_obs, n_dropped, drop_reason = mc._layer_sample(
            frame, layer, n_input, base_mask=known, base_reason=REASON_UNKNOWN_COMBO,
        )
        terms, dropped = mc._formula_terms(covariates, sample)
        base_note = "dropped_constant:" + ",".join(dropped) if dropped else None

        categories = [c for c in COMBO_CATEGORIES
                      if c in set(sample[COMBO_COLUMN].astype("object"))]
        if len(categories) < len(COMBO_CATEGORIES):
            base_note = mc._join_notes(
                base_note,
                "absent_categories:" + ",".join(
                    c for c in COMBO_CATEGORIES if c not in categories
                ),
            )
        blocked = _blocked_note(sample, terms)
        if blocked is None and len(categories) < 2:
            blocked = "insufficient_categories"
        # 多项 logit 每多一个类别就多一整套系数：真正要估的参数量是
        # 设计矩阵宽度 ×(类别数-1)，用单方程的宽度判断"观测够不够估"
        # 会在四类别下宽松三倍。
        if blocked is None and len(sample) <= mc._design_width(terms, sample) * (
                len(categories) - 1):
            blocked = "insufficient_observations"
        if blocked is not None:
            rows.extend(_combo_nan_rows(layer, n_obs, n_dropped, drop_reason,
                                        mc._join_notes(base_note, blocked)))
            continue

        # MNLogit 不接受字符串因变量，因此显式编码：类别顺序固定为
        # COMBO_CATEGORIES，编码 0（neither）自动成为基准类别。
        # 基准类别的选择不影响 AME（AME 是概率尺度的量），但固定下来能让
        # 附带的系数在不同层之间可读。
        exog = patsy.dmatrix(" + ".join(terms), sample, return_type="dataframe")
        design_info = exog.design_info
        # patsy 会静默丢掉协变量含 NaN 的行，用它保留下来的索引对齐因变量，
        # 否则 endog 与 exog 会错位（这是一个不会报错的错误）
        aligned = sample.loc[exog.index]
        codes = pd.Categorical(
            aligned[COMBO_COLUMN].astype("object"), categories=categories
        ).codes

        label = f"{OUTCOME_COMBO}/multinomial/{layer}"
        result, fail = mc._safe_fit(
            lambda: sm.MNLogit(codes, np.asarray(exog)).fit(
                cov_type="HC1", disp=False, maxiter=200
            ),
            label,
        )
        if result is None:
            rows.extend(_combo_nan_rows(layer, n_obs, n_dropped, drop_reason,
                                        mc._join_notes(base_note, fail), categories))
            continue

        n_obs, n_dropped, drop_reason = mc._reconcile_nobs(
            result, n_input, n_obs, n_dropped, drop_reason
        )
        ames, ses, ame_sum = _multinomial_ames(result, design_info, aligned, categories)
        note = mc._join_notes(base_note, "mnlogit_base:" + categories[0])

        for category, ame, se in zip(categories, ames, ses):
            low = ame - _Z * se if np.isfinite(se) else np.nan
            high = ame + _Z * se if np.isfinite(se) else np.nan
            rows.append(su.tidy_result(
                outcome=OUTCOME_COMBO, domain=DOMAIN_BOTH, model=layer,
                term=combo_ame_term(category), estimate=float(ame),
                se=float(se) if np.isfinite(se) else np.nan,
                ci_low=low, ci_high=high, scale="probability",
                n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason, note=note,
            ))
        rows.append(su.tidy_result(
            outcome=OUTCOME_COMBO, domain=DOMAIN_BOTH, model=layer, term=TERM_AME_SUM,
            estimate=ame_sum, se=np.nan, ci_low=np.nan, ci_high=np.nan,
            scale="probability", n_obs=n_obs, n_dropped=n_dropped,
            drop_reason=drop_reason,
            note=mc._join_notes(note, "internal_consistency_check"),
        ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# §8.2 两个域 topical_share 的相关
# ---------------------------------------------------------------------------

def content_correlation(user_df, n_boot=_N_BOOT, seed=_SEED):
    """§8.2 公共事务域与娱乐域 topical_share 的 Spearman 相关

    两种口径并列给出（用 model 列区分）：
        MODEL_RAW               原始相关
        MODEL_ACTIVITY_PARTIAL  控制活跃度之后的秩偏相关

    这正是"用户是否在两个域之间分工"的连续版本答案：如果原始相关为正、
    控制活跃度之后明显衰减，说明"做得多的人什么都多做一点"；如果控制之后
    仍然为负，才谈得上分工。**本函数刻意不构造任何"新闻 vs 娱乐"的一维
    偏好分数**（§8.3 的禁令，见模块文档第 6 条）：报告的是两个变量之间的
    相关，不是把它们压成一个数。

    只用两个域的 topical_share 都有定义的用户（两者共用同一个分母
    n_expressive_posts，因此实际上同时有定义或同时缺失），剔除数写进
    n_dropped 与 drop_reason——把缺失当成 0 会凭空制造相关。
    """
    public = pd.to_numeric(
        user_df["public_topical_share"].astype("object"), errors="coerce"
    ).astype(float)
    celebrity = pd.to_numeric(
        user_df["celebrity_topical_share"].astype("object"), errors="coerce"
    ).astype(float)
    su.check_proportion_range(public, "public_topical_share")
    su.check_proportion_range(celebrity, "celebrity_topical_share")
    both_defined = public.notna() & celebrity.notna()

    groups = _gender_masks(user_df) + [("all", pd.Series(True, index=user_df.index))]
    rows = []
    for gender_label, mask in groups:
        n_total = int(mask.sum())
        keep = mask & both_defined
        n_obs = int(keep.sum())
        n_dropped = n_total - n_obs
        drop_reason = mi.format_drop_reason(
            [(REASON_NO_EXPRESSIVE, n_dropped)] if n_dropped else []
        )

        frame = pd.DataFrame({
            "x": public[keep].to_numpy(),
            "y": celebrity[keep].to_numpy(),
        })
        controls, control_cols = _activity_matrix(user_df.loc[keep])
        for j in range(len(control_cols)):
            frame[f"ctrl_{j}"] = controls[:, j]
        ctrl_names = [f"ctrl_{j}" for j in range(len(control_cols))]

        rows.append(_bootstrap_row(
            frame, lambda f: spearman(f["x"], f["y"]),
            OUTCOME_CONTENT_CORR, DOMAIN_BOTH, MODEL_RAW,
            f"spearman_{gender_label}", "correlation", n_obs, n_dropped,
            drop_reason, "both_domains_defined_only", n_boot=n_boot, seed=seed,
        ))
        if ctrl_names:
            rows.append(_bootstrap_row(
                frame,
                lambda f: partial_spearman(f["x"], f["y"], f[ctrl_names].to_numpy()),
                OUTCOME_CONTENT_CORR, DOMAIN_BOTH, MODEL_ACTIVITY_PARTIAL,
                f"spearman_{gender_label}", "correlation", n_obs, n_dropped,
                drop_reason, "controls:" + ",".join(control_cols),
                n_boot=n_boot, seed=seed,
            ))
        else:
            rows.append(su.tidy_result(
                outcome=OUTCOME_CONTENT_CORR, domain=DOMAIN_BOTH,
                model=MODEL_ACTIVITY_PARTIAL, term=f"spearman_{gender_label}",
                estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan,
                scale="correlation", n_obs=n_obs, n_dropped=n_dropped,
                drop_reason=drop_reason, note="no_activity_controls_available",
            ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DECOMPOSITION_FILE = "decomposition_source_content.parquet"
COMBINATION_FILE = "combination_multinomial.parquet"


def _write_partial(frames, path, out_dir, source_path, year, n_users, completed):
    """把当前已完成的阶段落盘，并同步更新 manifest（与 models_interaction 同一约定）

    **增量落盘不是优化，是防丢**：§7 的几个阶段在全样本上要跑十几分钟
    （bootstrap 的相关行是大头），中途任何一步报错或作业撞墙钟被杀，
    如果只在最后写文件，前面算好的全都消失。manifest 的 completed_stages
    记录这份文件到底包含哪几个阶段，避免读者把被截断的结果当成完整结果。
    """
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    n_failed = int(frame["estimate"].isna().sum())
    print(f"已保存（增量）: {path}（{len(frame)} 行，其中 {n_failed} 行为拟合失败留痕）")

    manifest = config.build_manifest(
        step=f"models_combination_{year}",
        inputs=[os.path.relpath(source_path, config.OUTPUT_DIR)],
        params={
            "year": year,
            "M0": mc.M0, "M1": mc.M1, "M2": mc.M2,
            "content_thresholds": [float(t) for t in CONTENT_THRESHOLDS],
            "participation_variants": [VARIANT_ANY_HIT]
            + [variant_label(t) for t in CONTENT_THRESHOLDS],
            "activity_controls": list(ACTIVITY_CONTROLS),
            "estimator": "logit(HC1) / MNLogit(HC1) + counterfactual AME",
            "confidence": _CONFIDENCE,
            "n_boot": _N_BOOT,
            "no_single_preference_index": True,
        },
        counts={
            "n_users": n_users,
            "completed_stages": list(completed),
            "n_completed_stages": len(completed),
            "rows": int(len(frame)),
            "failed_rows": n_failed,
        },
    )
    config.write_manifest(manifest, os.path.join(out_dir, "manifest_combination"))
    return frame


def build(year=config.YEAR):
    """跑完 §7 与 §8 并写出两张结果表 + manifest

    §7 的三组结果写进 decomposition_source_content.parquet，§8 的多项 logit
    与两域相关写进 combination_multinomial.parquet。每完成一个阶段就落盘
    一次（见 _write_partial）。

    内容参与口径：any_hit 与 CONTENT_THRESHOLDS 里的每个阈值都跑一遍
    （§7.1 的要求：阈值的选择必须是可见的）。§7.2 里两条 Spearman 行与
    二分口径无关，只在 any_hit 变体上算一次，其余变体跳过，避免把同一个
    bootstrap 重复跑五遍。
    """
    user_df, source_path = describe._load_user_table(year)
    out_dir = os.path.join(config.OUTPUT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    n_users = int(len(user_df))
    variants = [None] + list(CONTENT_THRESHOLDS)

    decomposition_path = os.path.join(out_dir, DECOMPOSITION_FILE)
    frames, completed = [], []

    for domain in DOMAINS:
        for threshold in variants:
            label = variant_label(threshold)
            print(f"§7.1 参与类型: {domain} / {label}")
            frames.append(participation_types(user_df, domain, threshold))
            completed.append(f"participation_types/{domain}/{label}")
            _write_partial(frames, decomposition_path, out_dir, source_path,
                           year, n_users, completed)

            print(f"§7.2 交叉关系: {domain} / {label}")
            frames.append(cross_relations(
                user_df, domain, threshold, include_correlations=(threshold is None)
            ))
            completed.append(f"cross_relations/{domain}/{label}")
            _write_partial(frames, decomposition_path, out_dir, source_path,
                           year, n_users, completed)

            print(f"§7.3 内容参与 ~ 来源参与 × 性别: {domain} / {label}")
            frames.append(fit_content_on_source(user_df, domain, threshold))
            completed.append(f"content_on_source/{domain}/{label}")
            _write_partial(frames, decomposition_path, out_dir, source_path,
                           year, n_users, completed)

    combination_path = os.path.join(out_dir, COMBINATION_FILE)
    combo_frames = []
    print("§8.1 四类参与组合的多项 logit")
    combo_frames.append(fit_combination_multinomial(user_df))
    completed.append("combination_multinomial")
    _write_partial(combo_frames, combination_path, out_dir, source_path,
                   year, n_users, completed)

    print("§8.2 两域 topical_share 的相关（原始与控制活跃度）")
    combo_frames.append(content_correlation(user_df))
    completed.append("content_correlation")
    _write_partial(combo_frames, combination_path, out_dir, source_path,
                   year, n_users, completed)

    return [decomposition_path, combination_path]


if __name__ == "__main__":
    fire.Fire({"build": build})
