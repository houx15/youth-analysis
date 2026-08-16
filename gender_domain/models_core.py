"""
§6.2–6.5 核心模型：进入（entry）、强度（intensity）、占比（share）、
持续性（persistence），每一个都跑 M0/M1/M2 三层协变量。

只读表 C（`user_domain_{year}_with_profile.parquet`，缺失时回退到
`user_domain_{year}.parquet`），不重新推导任何度量——四个结果变量都已经
在 build_user_tables.combine_user_table 里按正确的 NaN 语义算好。本模块
唯一新造的量是 log_posts / log_retweets（log1p 变换），它们是被单测覆盖的
显式函数，不写在模型调用里。

统计口径与实现取舍（写在这里，供论文方法部分与后续维护者直接引用）：

1. **报告的量是概率/比例尺度的平均边际效应（AME），不是回归系数。**
   论文要把 M0/M1/M2 三层并排比较，而发生比（odds ratio）在协变量集合
   不同的模型之间不可比——非线性模型里加入与自变量无关的控制变量也会
   把发生比推大（unobserved heterogeneity / noncollapsibility），于是
   "加了控制之后性别效应反而变大"这种纯属尺度假象的结论就会被写进论文。
   AME 在概率尺度上定义，没有这个问题。发生比作为 `odds_ratio` 尺度的
   附加行同时给出，只是"可以放在一起看"，绝不取代 AME。
   AME 一律调用 stats_utils.average_marginal_effect（反事实预测 + delta
   method），本模块不自己实现边际效应。

2. **三层协变量的意义是让"性别差异有多少只是平台活跃度"可见。**
   M0 只有性别；M1 加发帖量/转发量/活跃天数/活跃月数；M2 再加账号画像。
   论文的论证依赖于读者能看到性别效应从 M0 到 M1 缩小（或没有缩小），
   所以三层对每个结果变量都必须出现，哪怕某一层本身没有故事。

3. **M2 收窄样本，M0/M1 不收窄。** M2 只用画像完整的用户
   （profile_complete 且 region 非缺失——region 也是 M2 的控制变量，
   缺 region 同样无法进模型，一并计入 `incomplete_profile`），
   这一层的 `n_obs` / `n_dropped` / `drop_reason="incomplete_profile"`
   必须写进结果表：M2 的样本流失是论文要交代的事实（§11.4），
   不是可以省略的实现细节。

4. **计数用两部分（hurdle）结构，不用单一 Poisson/OLS。**
   来源转发数是"绝大多数人是 0、少数人非常大"的分布，单一 Poisson 会
   同时低估零膨胀和过离散，单一 OLS 会被长尾拖走。因此：
   第一部分是进入模型（有没有参与），第二部分只在参与者上估计正计数，
   用零截断负二项（TruncatedLFNegativeBinomialP, truncation=0, p=2），
   报告发生率比（IRR）。log(1+x) 的 OLS 只作为稳健性行给出
   （note="log1p_ols"），它的系数不是 IRR，尺度另记为 log1p_count，
   刻意不与 IRR 混在同一个 scale 里被下游当成同一种量。
   **IRR 的口径必须读准**：零截断负二项的 IRR 是"潜在未截断均值之比"，
   不是"参与者条件均值 E[y|y>0] 之比"。模拟核对：真实潜在比 2.0 时本
   模块估出 1.989，而观测到的 E[y|y>0] 之比只有 1.783。读者很容易把
   表里的 2.0 读成"参与的男性转发量是女性的两倍"，所以每一行 IRR 的
   note 都带上 NOTE_IRR_LATENT，让结果表自证口径。

5. **占比型结果变量用分数 logit（fractional logit）。**
   statsmodels 没有独立的 QuasiBinomial family；Papke–Wooldridge 的
   分数 logit 本来就是"Binomial family + logit link 的拟极大似然 +
   稳健协方差"，所以这里用 GLM(Binomial) 配 cov_type="HC1"，
   这就是准二项（quasi-binomial）的正确实现，不是退而求其次的近似。
   持续性模型的分母是活跃月数（每个用户的"试验次数"不同），因此额外
   传 var_weights=活跃月数——但**权重只作用于拟合，不改变报告口径**：
   报告的 AME 依然在用户上等权平均，与表 2 的用户层面均值同一个总体
   单位（详见 fit_persistence_models 的 docstring）。
   GLM 的代价是它的 IRLS 在（准）完全分离时会照样报告 converged=True，
   参数飙到 20~30 也自称收敛、标准误还异常地小——进入模型改用 Logit
   正是为了避开这一点，占比/持续性没得选，因此补了 _glm_separation_note
   这道事后检查（系数量级 / 拟合值贴边），命中就在 note 里点名，
   但不删除该行。

5b. **n_obs 记的是真正进入拟合的观测数**（result.nobs），不是我们挑出来
   的子样本行数：patsy 会静默丢掉任何一个协变量为 NaN 的行。两者不一致
   时差额补进 n_dropped 并追加 missing_covariate 原因，_reconcile_nobs
   还会断言 n_obs + n_dropped 恒等于输入用户数——结果表的样本量记账
   不应该依赖上游模块的内部约定才成立。

6. **NaN 不是 0。** 分母为 0 的用户（没有转发 / 没有表达帖 / 没有活跃月）
   结果变量是 NaN，一律剔除并计入 n_dropped，绝不 fillna(0)。

7. **拟合失败必须留痕，不能少一行。** 任何一层模型不收敛、协变量退化、
   抛异常，都会产出一行 estimate=NaN 且 note 写明失败原因的结果行。
   结果表里少一行，读者会理解成"这个假设没人检验过"，那比一个显式的
   失败标记糟糕得多。

8. **AME 的 se 必须真的算出来。** stats_utils.average_marginal_effect 在
   cov_params() 确实不可用时会返回 NaN（那是刻意的：宁可没有区间，也不
   给一个偏窄的区间）。本模块用到的四种估计量（Logit、GLM Binomial、
   OLS、零截断负二项）都提供 cov_params()，因此正常路径不会落到那个
   分支——test_every_successful_estimate_carries_a_finite_interval 就是
   守这条线的。万一将来换了某个不提供协方差的拟合方式，正确做法是在
   本模块里对估计样本做"每次重抽样重新 .fit()"的 cluster bootstrap，
   而不是接受 NaN，也不是在 stats_utils 里放宽那条规则。

使用方法:
    python -m gender_domain.models_core build --year=2020
"""

import os

import fire
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP

from gender_domain import config
from gender_domain import describe
from gender_domain import stats_utils as su

DOMAINS = ("public", "celebrity")

# 性别在模型里以 0/1 数值列 male 进入（patsy 对字符串列会自己造哑变量，
# 但 stats_utils.average_marginal_effect 需要把 focal 列直接置 1/0 做
# 反事实预测，因此必须是一个真正的 0/1 数值列）。协变量常量里仍然写
# "gender"，_formula_terms 负责把它翻译成 male。
FOCAL_COLUMN = "male"

M0 = ["gender"]
M1 = M0 + ["log_posts", "log_retweets", "n_active_days", "n_active_months"]
M2 = M1 + ["verified_flag", "log_fans", "log_friends", "region"]
MODEL_LAYERS = (("M0", M0), ("M1", M1), ("M2", M2))

# 结果行的 term 命名：一个 term 对应一个明确的量，不靠 scale 列去区分，
# 这样下游按 (outcome, domain, model, term) 取行永远唯一。
TERM_AME = "gender_male"
TERM_ODDS_RATIO = "gender_male_odds_ratio"
TERM_IRR = "gender_male_irr"
TERM_LOG1P = "gender_male_log1p_coef"

NOTE_PART1 = "two_part_part1"
NOTE_LOG1P_OLS = "log1p_ols"
NOTE_LOG_SCALE = "log_scale"
# 零截断负二项的 IRR 是"潜在未截断均值之比"，不是"参与者条件均值之比"。
# 模拟核对过：真实潜在比 2.0 时本模块估出 1.989，而观测到的
# E[y|y>0] 之比只有 1.783。读者很容易把表里的 2.0 读成"参与的男性转发
# 量是女性的两倍"，所以这一条必须写在行内的 note 里，让结果表自证口径。
NOTE_IRR_LATENT = "irr_of_latent_untruncated_mean"

# 占比型结果变量 -> 分母为 0 时的缺失原因（写进 drop_reason）。
# 两个结果变量的分母不同（全部转发 / 全部表达帖），所以缺失原因也必须
# 不同——统一写成 "missing" 会让读者无法判断被剔除的是哪一群用户。
SHARE_OUTCOMES = ("source_share", "topical_share")
SHARE_DROP_REASONS = {
    "source_share": "no_retweets",
    "topical_share": "no_expressive_posts",
}

# 零截断负二项的优化器重试顺序，见 _fit_ztnb 的说明。
# 刻意不含 "nm"（Nelder–Mead）：实测在 44 参数的 M2 上跑满 1000 次迭代
# 要 25 秒且仍然 converged=False，救不回任何一层；更要命的是 NM 的收敛
# 标志判定的是"单纯形塌缩"而不是"梯度为零"，它报告的 converged=True
# 可能落在一个非驻点上——一个救不了场、又会给出虚假收敛信号的档位，
# 留着只有坏处。
ZTNB_METHODS = ("newton", "bfgs")
# 非首选优化器的结果必须额外过一道梯度检验：max|score| / nobs 低于此值
# 才承认它落在驻点上（score 是对数似然的梯度，MLE 处应为 0）
ZTNB_SCORE_TOL = 1e-3

# GLM（分数 logit）的准分离告警阈值，见 _glm_separation_note。
# 15 是 logit 尺度上的"荒谬大"：exp(15) ≈ 3.3e6 的发生比，真实社会数据
# 里不会出现，出现了就是设计矩阵把某些观测完美预测掉了。
GLM_EXTREME_COEF = 15.0
GLM_BOUNDARY_EPS = 1e-6
GLM_BOUNDARY_SHARE = 0.01

_CONFIDENCE = 0.95
_Z = float(norm.ppf(1 - (1 - _CONFIDENCE) / 2))


# ---------------------------------------------------------------------------
# 派生变量：全部是显式的、被单测覆盖的函数，不写在模型调用里
# ---------------------------------------------------------------------------

def add_log_activity(user_df):
    """派生 log_posts / log_retweets（log1p，而不是 log）

    用 log1p 是因为 n_posts / n_retweets 都可能取 0（表 C 里 0 是有定义的
    事实，不是缺失），log(0) = -inf 会直接毁掉设计矩阵。log1p(0)=0，
    在"完全没有该行为"这个点上取 0 也符合直觉。
    """
    out = user_df.copy()
    out["log_posts"] = np.log1p(out["n_posts"].astype(float))
    out["log_retweets"] = np.log1p(out["n_retweets"].astype(float))
    return out


def add_male_indicator(user_df):
    """派生 0/1 的 male 列（男性=1，女性=0）

    方向固定为"男性 - 女性"，与表 2 的 gap_pp 口径一致，这样描述表与
    模型表里"正号代表男性更高"是同一个含义，不需要读者逐表确认。
    """
    out = user_df.copy()
    out[FOCAL_COLUMN] = (out["gender"] == "m").astype(int)
    return out


def prepare_model_frame(user_df):
    """把表 C 整理成可以直接进 patsy 的建模帧

    做四件事，都刻意做在这里而不是散落在各个 fit_* 里：
    1) 只保留 gender 为 'm'/'f' 的用户——性别缺失/其它取值的用户没有
       办法进入以性别为焦点变量的模型，剔除数量由调用方计入 n_dropped；
    2) 派生 male / log_posts / log_retweets；
    3) 把 bool 与 pandas 可空布尔列转成 float：patsy 会把 bool 列当成
       分类变量处理，而可空布尔（"boolean" dtype）在 astype(float) 时
       遇到 pd.NA 会直接抛错，两者都必须在进模型之前统一成 float；
    4) 派生 log1p 计数列，供强度模型的稳健性行使用。
    """
    frame = user_df[user_df["gender"].isin(("m", "f"))].copy()
    frame = add_male_indicator(frame)
    frame = add_log_activity(frame)

    if "verified_flag" in frame.columns:
        frame["verified_flag"] = pd.to_numeric(
            frame["verified_flag"].astype("object"), errors="coerce"
        ).astype(float)
    if "profile_complete" in frame.columns:
        frame["profile_complete"] = (
            frame["profile_complete"].astype("object").fillna(False).astype(bool)
        )

    for domain in DOMAINS:
        entered = f"{domain}_source_entered"
        if entered in frame.columns:
            frame[entered] = frame[entered].astype("object").fillna(False).astype(float)
        count_col = f"{domain}_source_count"
        if count_col in frame.columns:
            frame[f"{domain}_log1p_source_count"] = np.log1p(
                frame[count_col].astype(float)
            )
    return frame


def active_months_column(frame):
    """持续性模型的分母列名

    优先用面板口径的 n_active_months_panel（只转发不发帖的月份也算活跃月，
    见 build_user_tables.combine_user_table），缺失时退回 n_active_months。
    """
    return "n_active_months_panel" if "n_active_months_panel" in frame.columns else "n_active_months"


# ---------------------------------------------------------------------------
# 公式与样本
# ---------------------------------------------------------------------------

def _formula_terms(covariates, sample):
    """把协变量常量翻译成 patsy 右侧项，并剔除在本样本里退化的变量

    - "gender" -> male（0/1 数值列，见 FOCAL_COLUMN 的说明）
    - "region" -> C(region)（分类变量）
    - 在本样本里取值唯一的协变量直接剔除：留着它只会让设计矩阵奇异，
      拟合要么报错要么给出无意义的巨大系数。剔除了哪些变量写进 note，
      不静默处理——"这一层其实没控制住 verified_flag"是读者需要知道的。

    Returns:
        (terms, dropped)：patsy 项列表，与被剔除的协变量名列表
    """
    terms = []
    dropped = []
    for name in covariates:
        if name == "gender":
            terms.append(FOCAL_COLUMN)
            continue
        if name not in sample.columns:
            dropped.append(name)
            continue
        if sample[name].nunique(dropna=True) < 2:
            dropped.append(name)
            continue
        terms.append("C(region)" if name == "region" else name)
    return terms, dropped


def _layer_sample(frame, layer, n_input, base_mask=None, base_reason=None):
    """构造某一层模型的估计样本，并把样本流失如实记账

    Args:
        frame: prepare_model_frame 的输出（已只含 m/f）
        layer: "M0"/"M1"/"M2"
        n_input: 调用方传进来的原始用户数（含性别缺失者），n_dropped 以它为基准
        base_mask: 该结果变量自身的有效性掩码（例如占比非 NaN、计数为正）
        base_reason: 上述掩码对应的缺失原因字符串

    Returns:
        (sample, n_obs, n_dropped, drop_reason)
    """
    mask = pd.Series(True, index=frame.index)
    reasons = []
    if n_input > len(frame):
        reasons.append("missing_gender")
    if base_mask is not None:
        mask &= base_mask.reindex(frame.index).fillna(False).astype(bool)
        if int((~mask).sum()) > 0 and base_reason:
            reasons.append(base_reason)
    if layer == "M2":
        complete = frame["profile_complete"] if "profile_complete" in frame.columns \
            else pd.Series(False, index=frame.index)
        if "region" in frame.columns:
            complete = complete & frame["region"].notna()
        before = int(mask.sum())
        mask &= complete.astype(bool)
        if int(mask.sum()) < before:
            reasons.append("incomplete_profile")

    sample = frame[mask]
    n_obs = len(sample)
    n_dropped = n_input - n_obs
    drop_reason = "+".join(reasons) if reasons else None
    return sample, n_obs, n_dropped, drop_reason


def _join_notes(*parts):
    """把若干条 note 片段用 + 连起来，None/空串直接跳过"""
    kept = [p for p in parts if p]
    return "+".join(kept) if kept else None


# ---------------------------------------------------------------------------
# 拟合与失败留痕
# ---------------------------------------------------------------------------

def _convergence_note(result):
    """返回 None 表示这次拟合可用，否则返回失败原因字符串"""
    retvals = getattr(result, "mle_retvals", None)
    if retvals is not None and not retvals.get("converged", True):
        return "not_converged"
    converged = getattr(result, "converged", None)
    if converged is not None and not bool(converged):
        return "not_converged"
    params = np.asarray(result.params, dtype=float)
    if not np.isfinite(params).all():
        return "non_finite_params"
    bse = np.asarray(getattr(result, "bse", np.array([])), dtype=float)
    if bse.size and not np.isfinite(bse).all():
        return "non_finite_se"
    return None


def _safe_fit(fit_callable, label):
    """执行一次拟合，把异常与不收敛统一翻译成 (result, note)

    异常在这里被捕获成一行结果，而不是让整张表跑不出来：某个领域某一层
    模型退化，不应该连带抹掉其它 11 个已经算好的模型。捕获到的异常类型
    与消息原样写进 note，读者能看出到底是奇异矩阵还是完全分离。
    """
    try:
        result = fit_callable()
    except Exception as exc:  # noqa: BLE001 —— 失败必须留痕，见模块文档第 7 条
        note = "{}: {}".format(type(exc).__name__, exc)
        print(f"警告: {label} 拟合失败（{note}），已写入一行 NaN 结果并注明原因")
        return None, note[:200]
    note = _convergence_note(result)
    if note is not None:
        print(f"警告: {label} 未收敛或估计不可用（{note}），已写入一行 NaN 结果并注明原因")
        return None, note
    return result, None


def _design_width(terms, sample):
    """估算设计矩阵的列数（含截距）

    不能用 len(covariates)+1：C(region) 一项在真实数据里会展开成 30+ 个
    哑变量，用协变量个数当参数个数会把 M2 的参数量少算约 30，
    "观测数够不够估"这个判断因此形同虚设。这里按分类变量的水平数
    （减去参照水平）逐项累加。
    """
    width = 1
    for term in terms:
        if term.startswith("C(") and term.endswith(")"):
            name = term[2:-1]
            width += max(int(sample[name].nunique(dropna=True)) - 1, 0)
        else:
            width += 1
    return width


def _blocked_note(sample, terms):
    """拟合之前就能判定"这层没法估"的情形，返回原因字符串，否则 None"""
    if len(sample) == 0:
        return "empty_sample"
    if sample[FOCAL_COLUMN].nunique() < 2:
        return "no_gender_variation"
    if len(sample) <= _design_width(terms, sample):
        return "insufficient_observations"
    return None


def _glm_separation_note(result):
    """分数 logit 的准分离/完全分离告警，返回 note 片段或 None

    GLM 的 IRLS 在（准）完全分离时会照样报告 converged=True，参数飙到
    20~30 也自称收敛，标准误还异常地小——实测系数 27.79、se 0.0045、
    note 干净，在结果表里看起来和一行正常估计毫无区别。进入模型选用
    Logit 而非 GLM 正是为了避开这一点，但占比/持续性模型必须用 GLM
    （分数 logit 只能这么实现），所以这里补一道便宜的事后检查：
    - logit 尺度上出现 |系数| > GLM_EXTREME_COEF 的项；或
    - 超过 GLM_BOUNDARY_SHARE 比例的拟合值贴在 0/1 边界上。
    命中任一条就在 note 里写明，**不删除这一行**——把估计连同告警一起
    交给读者，比悄悄丢掉或悄悄放行都更诚实。
    """
    parts = []
    params = np.asarray(result.params, dtype=float)
    max_abs = float(np.max(np.abs(params))) if params.size else 0.0
    if max_abs > GLM_EXTREME_COEF:
        parts.append("max_coef={:.1f}".format(max_abs))
    fitted = np.asarray(getattr(result, "fittedvalues", np.array([])), dtype=float)
    if fitted.size:
        boundary = np.mean((fitted < GLM_BOUNDARY_EPS) | (fitted > 1 - GLM_BOUNDARY_EPS))
        if boundary > GLM_BOUNDARY_SHARE:
            parts.append("boundary_share={:.3f}".format(float(boundary)))
    if not parts:
        return None
    return "possible_separation:" + ",".join(parts)


def _reconcile_nobs(result, n_input, n_obs, n_dropped, drop_reason):
    """用真正进入拟合的观测数覆盖 n_obs，并守住样本量恒等式

    此前 n_obs 记的是"我们挑出来的子样本行数"，而 patsy 会在拟合时静默
    丢掉任何一个协变量为 NaN 的行——两者不一致时，结果表会声称模型用了
    4000 个观测，实际只用了 3500，而且 n_dropped 还是 0。目前上游
    profile_join 让 profile_complete 蕴含三项控制变量非缺失，所以这在
    真实数据上尚未发生，但结果表不该依赖另一个模块的内部约定来保证
    自己的数字是对的。这里一律以 result.nobs 为准，差额补进 n_dropped
    并追加 missing_covariate 原因，最后断言 n_obs + n_dropped == n_input。
    """
    fitted = int(result.nobs)
    if fitted != n_obs:
        reasons = [drop_reason] if drop_reason else []
        reasons.append("missing_covariate")
        drop_reason = "+".join(reasons)
        n_obs = fitted
        n_dropped = n_input - fitted
    if n_obs + n_dropped != n_input:
        raise ValueError(
            f"样本量记账不自洽: n_obs={n_obs} + n_dropped={n_dropped} "
            f"!= 输入用户数 {n_input}；结果表的 n_obs/n_dropped 必须始终"
            "相加等于同一个总数，否则读者无法核对样本流失。"
        )
    return n_obs, n_dropped, drop_reason


def _nan_row(outcome, domain, model, term, scale, n_obs, n_dropped, drop_reason, note):
    return su.tidy_result(
        outcome=outcome, domain=domain, model=model, term=term,
        estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan,
        scale=scale, n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
        note=note,
    )


def _ame_row(result, sample, outcome, domain, model, scale, n_obs, n_dropped,
             drop_reason, note):
    """概率/比例尺度的平均边际效应行（本模块的主报告量）"""
    ame, se, low, high = su.average_marginal_effect(
        result, FOCAL_COLUMN, sample, confidence=_CONFIDENCE
    )
    if not np.isfinite(se):
        # 见模块文档第 8 条：正常路径不会走到这里；真走到了，说明换了
        # 一种不提供协方差的拟合方式，必须在本模块实现重拟合的 cluster
        # bootstrap，而不是让一个没有区间的点估计混进论文表格。
        print(
            f"警告: {outcome}/{domain}/{model} 的 AME 拿不到 delta method 标准误，"
            "结果表里这一行没有置信区间。这不是可以接受的终态——请在本模块中"
            "对该模型实现'每次重抽样重新拟合'的 cluster bootstrap 区间。"
        )
        note = _join_notes(note, "ame_se_unavailable")
    return su.tidy_result(
        outcome=outcome, domain=domain, model=model, term=TERM_AME,
        estimate=ame, se=se, ci_low=low, ci_high=high, scale=scale,
        n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason, note=note,
    )


def _exponentiated_row(result, outcome, domain, model, term, scale, n_obs, n_dropped,
                       drop_reason, note):
    """发生比 / 发生率比行：在对数尺度做正态近似再指数变换回来

    与 stats_utils.risk_ratio_ci 同一思路——比值本身右偏，对数尺度的
    正态近似准确得多。se 这一列刻意留 NaN：指数变换后的区间不对称，
    反推一个 se 会丢掉区间形状（stats_utils 模块文档第 5 条），
    note 里的 "log_scale" 就是在说明这一点。
    """
    coef = float(result.params[FOCAL_COLUMN])
    bse = float(result.bse[FOCAL_COLUMN])
    return su.tidy_result(
        outcome=outcome, domain=domain, model=model, term=term,
        estimate=float(np.exp(coef)), se=np.nan,
        ci_low=float(np.exp(coef - _Z * bse)),
        ci_high=float(np.exp(coef + _Z * bse)),
        scale=scale, n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
        note=_join_notes(note, NOTE_LOG_SCALE),
    )


# ---------------------------------------------------------------------------
# §6.2 进入模型
# ---------------------------------------------------------------------------

def fit_entry_models(user_df, domain, note_prefix=None):
    """§6.2 是否曾经转发过该领域来源账号：二值 logistic，M0/M1/M2

    每一层产出两行：概率尺度的性别 AME（主报告量）与同一模型的发生比
    （附加参考）。发生比永远不单独出现。
    """
    outcome = "source_entered"
    outcome_col = f"{domain}_source_entered"
    n_input = len(user_df)
    frame = prepare_model_frame(user_df)

    rows = []
    for model, covariates in MODEL_LAYERS:
        sample, n_obs, n_dropped, drop_reason = _layer_sample(frame, model, n_input)
        base_note = note_prefix
        terms, dropped = _formula_terms(covariates, sample)

        blocked = _blocked_note(sample, terms)
        if blocked is None and sample[outcome_col].nunique() < 2:
            blocked = "no_outcome_variation"
        if blocked is not None:
            note = _join_notes(base_note, blocked)
            rows.append(_nan_row(outcome, domain, model, TERM_AME, "probability",
                                 n_obs, n_dropped, drop_reason, note))
            rows.append(_nan_row(outcome, domain, model, TERM_ODDS_RATIO, "odds_ratio",
                                 n_obs, n_dropped, drop_reason, note))
            continue

        formula = "{} ~ {}".format(outcome_col, " + ".join(terms))
        label = f"{outcome}/{domain}/{model}"
        result, fail = _safe_fit(
            lambda: smf.logit(formula, data=sample).fit(
                cov_type="HC1", disp=False, maxiter=100
            ),
            label,
        )
        note = _join_notes(
            base_note,
            "dropped_constant:" + ",".join(dropped) if dropped else None,
            fail,
        )
        if result is None:
            rows.append(_nan_row(outcome, domain, model, TERM_AME, "probability",
                                 n_obs, n_dropped, drop_reason, note))
            rows.append(_nan_row(outcome, domain, model, TERM_ODDS_RATIO, "odds_ratio",
                                 n_obs, n_dropped, drop_reason, note))
            continue

        n_obs, n_dropped, drop_reason = _reconcile_nobs(
            result, n_input, n_obs, n_dropped, drop_reason
        )
        rows.append(_ame_row(result, sample, outcome, domain, model, "probability",
                             n_obs, n_dropped, drop_reason, note))
        rows.append(_exponentiated_row(result, outcome, domain, model, TERM_ODDS_RATIO,
                                       "odds_ratio", n_obs, n_dropped, drop_reason, note))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# §6.3 强度模型（两部分 / hurdle）
# ---------------------------------------------------------------------------

def _scaled_score_max(result):
    """max|对数似然梯度| / nobs，用来独立检验参数是否真的落在驻点上

    取不到梯度时返回 NaN（调用方按"没通过检验"处理）——一个拿不到梯度
    的结果没办法证明自己收敛了，就不该被当作收敛。
    """
    try:
        score = np.asarray(
            result.model.score(np.asarray(result.params, dtype=float)), dtype=float
        )
    except Exception:  # noqa: BLE001 —— 拿不到梯度就是没通过检验
        return np.nan
    if score.size == 0 or not np.isfinite(score).all():
        return np.nan
    return float(np.max(np.abs(score))) / max(float(result.nobs), 1.0)


def _fit_ztnb(formula, sample, label):
    """零截断负二项的拟合，按优化器顺序重试

    statsmodels 对这个模型的默认优化器是 bfgs，实测在"计数分布比较平、
    协变量几乎不解释计数"的样本上会停在 warnflag=2（梯度未收敛）——
    这不是模型不适用，只是这个优化器走不到极值点。因此按 ZTNB_METHODS
    的顺序（newton -> bfgs）依次尝试，并把实际用到的优化器写进 note
    （不是第一顺位时才写，避免正常情况下的噪声）。

    **非首选优化器的结果必须额外过一道梯度检验**：不同优化器的
    converged 标志判定的东西并不一样（Newton 看的是梯度/步长，单纯形类
    方法看的是单纯形塌缩），一个"自称收敛"的备用档位完全可能停在非驻点
    上，而结果表里只会留下一句"换了个优化器"。因此这里对第二顺位以后的
    结果独立计算 max|score|/nobs，超过 ZTNB_SCORE_TOL 就当作没收敛，
    并把实测值写进 note，让"为什么这一层是 NaN"可以追溯。

    Returns:
        (result, method_note, fail_note)；成功时 fail_note 为 None
    """
    model = TruncatedLFNegativeBinomialP.from_formula(
        formula, data=sample, truncation=0, p=2
    )
    last_fail = None
    for i, method in enumerate(ZTNB_METHODS):
        result, fail = _safe_fit(
            lambda: model.fit(method=method, cov_type="HC1", disp=False, maxiter=1000),
            f"{label}[{method}]",
        )
        if result is None:
            last_fail = fail
            continue
        if i == 0:
            return result, None, None
        score_max = _scaled_score_max(result)
        if np.isnan(score_max) or score_max > ZTNB_SCORE_TOL:
            last_fail = "score_not_stationary:{:.2e}".format(score_max)
            print(f"警告: {label}[{method}] 自称收敛但梯度检验未通过（{last_fail}），不予采信")
            continue
        return result, "ztnb_method:{}+score_max:{:.1e}".format(method, score_max), None
    return None, f"ztnb_methods_tried:{','.join(ZTNB_METHODS)}", last_fail


def fit_intensity_models(user_df, domain):
    """§6.3 来源转发强度：第一部分进入模型 + 第二部分零截断负二项

    第二部分只在"至少转发过一次该领域来源"的用户上估计，报告发生率比
    （IRR）。log(1+x) 的 OLS 作为稳健性行同时给出，它的估计量是
    log1p 尺度上的系数，scale 记为 log1p_count，与 IRR 刻意分开。
    """
    outcome = "source_count_positive"
    count_col = f"{domain}_source_count"
    log_col = f"{domain}_log1p_source_count"
    n_input = len(user_df)

    # 第一部分就是进入模型本身，原样带进强度表，note 标明它是第一部分，
    # 这样 models_intensity.parquet 单独拿出来也是完整的两部分结果。
    part1 = fit_entry_models(user_df, domain, note_prefix=NOTE_PART1)

    frame = prepare_model_frame(user_df)
    participants = frame[count_col].astype(float) > 0

    rows = []
    for model, covariates in MODEL_LAYERS:
        sample, n_obs, n_dropped, drop_reason = _layer_sample(
            frame, model, n_input, base_mask=participants,
            base_reason="no_source_retweets",
        )
        terms, dropped = _formula_terms(covariates, sample)
        blocked = _blocked_note(sample, terms)
        if blocked is not None:
            rows.append(_nan_row(outcome, domain, model, TERM_IRR, "irr",
                                 n_obs, n_dropped, drop_reason,
                                 _join_notes(NOTE_IRR_LATENT, blocked)))
            rows.append(_nan_row(outcome, domain, model, TERM_LOG1P, "log1p_count",
                                 n_obs, n_dropped, drop_reason,
                                 _join_notes(NOTE_LOG1P_OLS, blocked)))
            continue

        dropped_note = "dropped_constant:" + ",".join(dropped) if dropped else None

        # --- 第二部分：零截断负二项（NB2），报告 IRR ---
        formula = "{} ~ {}".format(count_col, " + ".join(terms))
        result, method_note, fail = _fit_ztnb(
            formula, sample, f"{outcome}/{domain}/{model}/ztnb"
        )
        note = _join_notes("ztnb", NOTE_IRR_LATENT, method_note, dropped_note, fail)
        if result is None:
            rows.append(_nan_row(outcome, domain, model, TERM_IRR, "irr",
                                 n_obs, n_dropped, drop_reason, note))
        else:
            irr_obs, irr_dropped, irr_reason = _reconcile_nobs(
                result, n_input, n_obs, n_dropped, drop_reason
            )
            rows.append(_exponentiated_row(result, outcome, domain, model, TERM_IRR,
                                           "irr", irr_obs, irr_dropped, irr_reason, note))

        # --- 稳健性：log(1+x) OLS，报告 log1p 尺度上的系数 ---
        ols_formula = "{} ~ {}".format(log_col, " + ".join(terms))
        ols_result, ols_fail = _safe_fit(
            lambda: smf.ols(ols_formula, data=sample).fit(cov_type="HC1"),
            f"{outcome}/{domain}/{model}/log1p_ols",
        )
        if ols_result is None:
            rows.append(_nan_row(outcome, domain, model, TERM_LOG1P, "log1p_count",
                                 n_obs, n_dropped, drop_reason,
                                 _join_notes(NOTE_LOG1P_OLS, dropped_note, ols_fail)))
        else:
            coef = float(ols_result.params[FOCAL_COLUMN])
            bse = float(ols_result.bse[FOCAL_COLUMN])
            ols_obs, ols_dropped, ols_reason = _reconcile_nobs(
                ols_result, n_input, n_obs, n_dropped, drop_reason
            )
            rows.append(su.tidy_result(
                outcome=outcome, domain=domain, model=model, term=TERM_LOG1P,
                estimate=coef, se=bse, ci_low=coef - _Z * bse, ci_high=coef + _Z * bse,
                scale="log1p_count", n_obs=ols_obs, n_dropped=ols_dropped,
                drop_reason=ols_reason,
                # 稳健性行的 note 必须恰好是 log1p_ols（下游按它筛行），
                # 额外信息追加在后面
                note=_join_notes(NOTE_LOG1P_OLS, dropped_note) if dropped_note
                else NOTE_LOG1P_OLS,
            ))

    part2 = pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))
    return pd.concat([part1, part2], ignore_index=True)


# ---------------------------------------------------------------------------
# §6.3 占比模型（分数 logit）
# ---------------------------------------------------------------------------

def _fit_fractional_logit(sample, outcome_col, terms, label, var_weights=None):
    formula = "{} ~ {}".format(outcome_col, " + ".join(terms))

    def _fit():
        kwargs = {"family": sm.families.Binomial()}
        if var_weights is not None:
            kwargs["var_weights"] = var_weights
        return smf.glm(formula, data=sample, **kwargs).fit(cov_type="HC1")

    return _safe_fit(_fit, label)


def fit_share_models(user_df, domain, outcome):
    """§6.3 占比型结果变量的分数 logit（quasi-binomial），AME 在比例尺度上

    Args:
        outcome: "source_share" 或 "topical_share"

    分母为 0 的用户（没有转发 / 没有表达帖）该结果变量是 NaN，剔除并
    计入 n_dropped，drop_reason 用 SHARE_DROP_REASONS 里的具体原因，
    不用笼统的 "missing"。
    """
    if outcome not in SHARE_DROP_REASONS:
        raise ValueError(f"未知的占比结果变量: {outcome!r}，只支持 {sorted(SHARE_DROP_REASONS)}")
    outcome_col = f"{domain}_{outcome}"
    n_input = len(user_df)
    frame = prepare_model_frame(user_df)
    # 分数 logit 以"取值落在 [0,1] 内"为前提（见模块文档第 5 条）。越界值
    # 不会让拟合报错，只会悄悄把估计推走，所以在拟合之前先硬校验一次。
    su.check_proportion_range(frame[outcome_col], outcome_col)
    defined = frame[outcome_col].notna()

    rows = []
    for model, covariates in MODEL_LAYERS:
        sample, n_obs, n_dropped, drop_reason = _layer_sample(
            frame, model, n_input, base_mask=defined,
            base_reason=SHARE_DROP_REASONS[outcome],
        )
        terms, dropped = _formula_terms(covariates, sample)
        blocked = _blocked_note(sample, terms)
        if blocked is not None:
            rows.append(_nan_row(outcome, domain, model, TERM_AME, "proportion",
                                 n_obs, n_dropped, drop_reason, blocked))
            continue

        result, fail = _fit_fractional_logit(
            sample, outcome_col, terms, f"{outcome}/{domain}/{model}"
        )
        note = _join_notes(
            "dropped_constant:" + ",".join(dropped) if dropped else None, fail
        )
        if result is None:
            rows.append(_nan_row(outcome, domain, model, TERM_AME, "proportion",
                                 n_obs, n_dropped, drop_reason, note))
            continue
        note = _join_notes(note, _glm_separation_note(result))
        n_obs, n_dropped, drop_reason = _reconcile_nobs(
            result, n_input, n_obs, n_dropped, drop_reason
        )
        rows.append(_ame_row(result, sample, outcome, domain, model, "proportion",
                             n_obs, n_dropped, drop_reason, note))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# §6.4 持续性模型
# ---------------------------------------------------------------------------

def fit_persistence_models(user_df, domain):
    """§6.4 参与月数 ÷ 活跃月数：分数 logit + 以活跃月数为 var_weights

    **估计量（estimand）与权重是两件事，这里必须说清楚，否则读者会读错
    这一行。**

    - 拟合用 var_weights=活跃月数：每个用户的"试验次数"不同，只活跃
      1 个月的用户对自己那个比例的信息量确实比活跃 12 个月的用户少，
      var_weights 是二项比例数据的标准权重设定，解决的是估计效率与
      异方差问题。
    - 报告的 AME 则是**在用户上等权平均**的（每个用户算一次反事实
      差值，再取算术平均，见 stats_utils.average_marginal_effect）。
      也就是说本行回答的是"随机抽一个用户，ta 是男性而不是女性会让
      ta 的参与月份占比变化多少"，单位是人，不是人月。

    选这个估计量的理由：表 2 的 source_month_share 报的就是用户层面
    比例的均值（每个用户一票），模型表与描述表必须回答同一个总体上的
    同一个问题，否则读者拿两张表对照时会得到两个不同的"性别差距"却
    找不到原因。若改成月加权 AME，本行就变成"随机抽一个活跃人月"的
    效应，与表 2 不可比。

    注意 M0 层两种平均恰好相等（M0 的拟合值在性别内是常数，逐用户差值
    处处相同，怎么平均都一样），所以 M0 的精确性测试无法区分这两种口径
    ——区分它们的是 test_persistence_ame_averages_over_users_not_over_months
    里的 M1 fixture。
    """
    outcome = "source_month_share"
    outcome_col = f"{domain}_source_month_share"
    n_input = len(user_df)
    frame = prepare_model_frame(user_df)
    su.check_proportion_range(frame[outcome_col], outcome_col)
    months_col = active_months_column(frame)
    defined = frame[outcome_col].notna() & (frame[months_col].astype(float) > 0)

    rows = []
    for model, covariates in MODEL_LAYERS:
        sample, n_obs, n_dropped, drop_reason = _layer_sample(
            frame, model, n_input, base_mask=defined, base_reason="no_active_months",
        )
        terms, dropped = _formula_terms(covariates, sample)
        blocked = _blocked_note(sample, terms)
        if blocked is not None:
            rows.append(_nan_row(outcome, domain, model, TERM_AME, "proportion",
                                 n_obs, n_dropped, drop_reason, blocked))
            continue

        weights = sample[months_col].astype(float).to_numpy()
        result, fail = _fit_fractional_logit(
            sample, outcome_col, terms, f"{outcome}/{domain}/{model}",
            var_weights=weights,
        )
        # note 同时写明拟合权重与报告口径：拟合是月加权的，报告的 AME
        # 是在用户上等权平均的，两者不是同一件事（见函数 docstring）
        note = _join_notes(
            "months_weighted_fit",
            "ame_averaged_over_users",
            "dropped_constant:" + ",".join(dropped) if dropped else None,
            fail,
        )
        if result is None:
            rows.append(_nan_row(outcome, domain, model, TERM_AME, "proportion",
                                 n_obs, n_dropped, drop_reason, note))
            continue
        note = _join_notes(note, _glm_separation_note(result))
        n_obs, n_dropped, drop_reason = _reconcile_nobs(
            result, n_input, n_obs, n_dropped, drop_reason
        )
        rows.append(_ame_row(result, sample, outcome, domain, model, "proportion",
                             n_obs, n_dropped, drop_reason, note))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build(year=config.YEAR):
    """跑完四组核心模型并写出四张结果表 + manifest"""
    # 表 C 的读取口径（优先带画像的版本、显式 columns）只在 describe 里
    # 实现一次，这里直接复用，避免两个模块对"读哪张表"各有一套判断。
    user_df, source_path = describe._load_user_table(year)

    entry = pd.concat([fit_entry_models(user_df, d) for d in DOMAINS], ignore_index=True)
    intensity = pd.concat([fit_intensity_models(user_df, d) for d in DOMAINS],
                          ignore_index=True)
    share = pd.concat(
        [fit_share_models(user_df, d, o) for d in DOMAINS for o in SHARE_OUTCOMES],
        ignore_index=True,
    )
    persistence = pd.concat([fit_persistence_models(user_df, d) for d in DOMAINS],
                            ignore_index=True)

    out_dir = os.path.join(config.OUTPUT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, frame in (
        ("models_entry", entry),
        ("models_intensity", intensity),
        ("models_share", share),
        ("models_persistence", persistence),
    ):
        path = os.path.join(out_dir, f"{name}.parquet")
        frame.to_parquet(path, engine="pyarrow", index=False)
        n_failed = int(frame["estimate"].isna().sum())
        print(f"已保存: {path}（{len(frame)} 行，其中 {n_failed} 行为拟合失败留痕）")
        paths.append(path)

    manifest = config.build_manifest(
        step=f"models_core_{year}",
        inputs=[os.path.relpath(source_path, config.OUTPUT_DIR)],
        params={
            "year": year,
            "M0": M0, "M1": M1, "M2": M2,
            "cov_type": "HC1",
            "confidence": _CONFIDENCE,
        },
        counts={
            "n_users": int(len(user_df)),
            "entry_rows": int(len(entry)),
            "intensity_rows": int(len(intensity)),
            "share_rows": int(len(share)),
            "persistence_rows": int(len(persistence)),
            "failed_rows": {
                "entry": int(entry["estimate"].isna().sum()),
                "intensity": int(intensity["estimate"].isna().sum()),
                "share": int(share["estimate"].isna().sum()),
                "persistence": int(persistence["estimate"].isna().sum()),
            },
        },
    )
    # 自己的 manifest 子目录：write_manifest 会覆盖同目录的 manifest.json，
    # results/ 根目录是所有 results 层模块共用的，不能各写各的 manifest.json
    # （与 describe 的 manifest_describe/ 同一约定）。
    config.write_manifest(manifest, os.path.join(out_dir, "manifest_models_core"))
    return paths


if __name__ == "__main__":
    fire.Fire({"build": build})
