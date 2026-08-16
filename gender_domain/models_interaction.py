"""
§6.6 Gender × Domain 交互：全篇论文的核心论断。

论文要说的不是"男女在某个域上有差异"，而是"性别差异的方向和大小在两个
信息域之间发生了改变"。两个各自显著的性别效应并不能证明这件事——两个
95% 区间不重叠只是交互显著的充分条件之一，重叠也完全可能交互显著；反过来
两个都显著、方向还相同的情形（本模块的 test_did_interval_contains_zero_
without_an_interaction 就是这种数据）根本不构成交互。因此这里直接估交互
本身，并把它报告成一个**可解释尺度上的差中差**。

数据结构与估计策略（每一条都是"换一种写法就会得到另一个数"的选择）：

1. **长表：每个用户两行（公共事务域 + 娱乐域各一行）。**
   同一个人的两行必然相关，因此模型用 GEE，工作相关结构取 exchangeable，
   聚类变量是 user_id；任何 bootstrap 也必须以**用户**为单位整簇重抽样，
   把这个用户的两行一起带走。按行重抽样会把"同一个人两次观测"当成两条
   独立信息，恰好在论文最重要的那个量上系统性低估标准误。

2. **报告的头条是概率/比例尺度上的差中差（DiD）**：
       DiD = (男 - 女)_公共事务 - (男 - 女)_娱乐
   用四个 gender × domain 反事实格子预测出来，而不是读交互项系数。
   logit 尺度的交互系数不是"概率差之差"，它既不能直接读成百分点，也在
   M0/M1/M2 之间不可比（与 models_core 模块文档第 1 条同一个理由：
   非线性模型的系数尺度随协变量集合漂移）。交互系数作为附加行给出
   （term=TERM_COEF，scale="log_odds"），只是"可以放在一起看"。

3. **M0/M1/M2 三层照跑。** 读者需要看到交互在控制掉平台总体活跃度之后
   是否还在。这里有一个容易误解的点：本模块所有控制变量都是用户级的，
   在一个用户的两行上取值完全相同，因此它们不可能混淆"域内对比"这件事
   本身；但模型是非线性的，控制变量会通过 link function 的曲率改变概率
   尺度上的 DiD。所以 M1/M2 的 DiD 与 M0 不同是正常的，不是 bug。

4. **单域缺失：整个用户剔除（本模块的裁定）。**
   一个用户在一个域上结果变量是 NaN、另一个域有效时，有两种处理：
   (a) 剔除整个用户；(b) 保留有效的那一行。两者给出不同的数字，所以必须
   显式选一个。本模块选 (a)，理由：
   - 估计量本身是一个**域间对比**。保留单行会让"公共事务域的样本"和
     "娱乐域的样本"变成两群不同的人，DiD 里就混进了人群构成差异，
     而论文想说的恰恰是"同一批人在两个域上表现不同"。
   - GEE 的 exchangeable 工作相关结构是在成对簇上定义的，混入大量只有
     一行的簇会让相关参数由一批根本没有配对信息的簇共同估计。
   - 缺失机制与域相关（结果变量的分母就是用户自己的行为量），保留单行
     等于让缺失机制直接决定各域的样本构成。
   代价是样本更小，这个代价被如实写进 n_dropped / drop_reason
   （REASON_INCOMPLETE_PAIR）。
   顺带说明：按目前表 C 的定义，两个域的 topical_share 共用同一个分母
   （n_expressive_posts），source_entered 又永远非缺失，所以这条规则在
   当前数据上命中数为 0。它仍然被实现并被测试覆盖，是为了将来有人改动
   分母口径时，这个选择不会被悄悄改掉——"两域都缺"与"只缺一个域"在
   drop_reason 里也刻意分开记，读者才能看出配对规则本身丢了多少人。

5. **中心数字不允许没有置信区间。**
   默认路径是 delta method：GEE 的默认协方差就是按簇（用户）算的稳健
   三明治协方差，所以 delta method 得到的区间本身已经是聚类稳健的，且
   在 22.5 万用户 / 45 万行上只需拟合一次。只有当 cov_params() 确实不可用
   时才退回到"每次重抽样重新 .fit()"的 cluster bootstrap——这正是
   stats_utils.average_marginal_effect 文档里说的"留给各个 models_*.py
   自己实现"的那条路径。两条路径都不通时本模块直接报错，而不是返回一个
   没有区间的头条数字（见 difference_in_differences 的说明）。
   bootstrap 里某次重抽样重拟合失败时，本模块**不会**把这次重抽样悄悄
   丢掉再用剩下的算百分位：能收敛的重抽样是参数空间里被挑选过的一部分，
   丢掉失败的那些会让区间偏窄。失败次数如实计入 note，区间随之变成 NaN，
   这是一个必须被人看到的信号。

6. **占比型结果变量用 GEE + Binomial family（准二项 / 分数 logit）。**
   GEE 本身就是拟似然估计，配上默认的稳健三明治协方差，这与
   models_core 里 "GLM(Binomial) + HC1" 是同一个 Papke–Wooldridge 估计量
   的面板版本，不是退而求其次的近似。表 C 的 topical_share 分子分母都在
   表达帖上计算，取值必然落在 [0,1]，Binomial family 适用。

7. **n_obs / n_dropped 记的是用户数，不是行数。** 长表有 2 倍的行，但
   结果表要与 models_core 的其它表并排看，样本量的单位必须一致。
   note 里的 NOTE_GEE 会写明"每用户两行"这件事，避免读者以为样本量少了
   一半。与 models_core._reconcile_nobs 同样的恒等式在这里以用户为单位
   守住：n_obs + n_dropped == 输入用户数。

使用方法:
    python -m gender_domain.models_interaction build --year=2020
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
from gender_domain import stats_utils as su

# 长表的列名：这些名字同时出现在公式、反事实预测和 bootstrap 里，
# 全部提成常量，避免三处各写一个字符串字面量而其中一处拼错。
USER_COLUMN = "user_id"
CLUSTER_COLUMN = "cluster_id"
DOMAIN_COLUMN = "domain"
DOMAIN_INDICATOR = "is_public"
OUTCOME_COLUMN = "outcome"
OUTCOME_KIND_COLUMN = "outcome_kind"
FOCAL_COLUMN = mc.FOCAL_COLUMN          # "male"，与 models_core 同一个口径

# 跨域的行（DiD 与交互系数）在 domain 列上取这个值：它们不属于任何单个域。
# 下游画图按 domain == DOMAIN_BOTH 取头条行，不要按 "public" 去筛。
DOMAIN_BOTH = "both"

DOMAINS = mc.DOMAINS                     # ("public", "celebrity")

TERM_DID = "gender_male_x_public_did"
TERM_GAP_PUBLIC = "gender_male_gap_public"
TERM_GAP_CELEBRITY = "gender_male_gap_celebrity"
TERM_COEF = "gender_male_x_public_coef"

NOTE_GEE = "gee_exchangeable_cluster_user_2rows_per_user"
METHOD_DELTA = "delta"
METHOD_BOOTSTRAP = "cluster_bootstrap"

REASON_INCOMPLETE_PAIR = "incomplete_domain_pair"

# 结果变量口径：列名模板、结果表里的 outcome 名、报告尺度、两域都缺时的原因。
# source_entered 在表 C 里是纯 bool（永远非缺失），因此"两域都缺"的原因
# 只是一个形式上的占位，真实数据里不会命中。
OUTCOME_KINDS = {
    "source_entry": {
        "column": "{domain}_source_entered",
        "outcome": "source_entered",
        "scale": "probability",
        "missing_reason": "no_retweet_events",
        "binary": True,
    },
    "topical_share": {
        "column": "{domain}_topical_share",
        "outcome": "topical_share",
        "scale": "proportion",
        "missing_reason": "no_expressive_posts",
        "binary": False,
    },
}

_CONFIDENCE = 0.95
_Z = float(norm.ppf(1 - (1 - _CONFIDENCE) / 2))
_N_BOOT = 500
_SEED = 20200101


# ---------------------------------------------------------------------------
# 长表构造
# ---------------------------------------------------------------------------

def _male_indicator(gender):
    """性别 -> 0/1 浮点列，无法使用的性别取值记 NaN 而不是 0

    models_core.add_male_indicator 用的是 (gender == "m").astype(int)，
    那里可以这么写是因为它的调用方 prepare_model_frame 已经先把非 m/f
    的用户整体剔除了。本模块的长表刻意不剔除任何用户（见 to_long_format
    的说明），如果照抄那一行，性别缺失的用户会被静默算成女性——这正是
    "让 pandas 顺手决定"最危险的一类情形，所以这里记 NaN。
    """
    gender = pd.Series(gender).astype("object")
    return pd.Series(
        np.where(gender == "m", 1.0, np.where(gender == "f", 0.0, np.nan)),
        index=gender.index, dtype=float,
    )


def _prepare_wide(user_df):
    """把表 C 整理成长表所需的宽表：派生 male / log 活跃度，统一 dtype

    没有直接调用 models_core.prepare_model_frame，是因为那个函数会先按
    gender in {m,f} 过滤掉一部分用户；本模块的样本量记账以 len(long_df)//2
    为分母，长表必须包含全部用户，过滤要发生在 _select_sample 里并被计入
    n_dropped。除此之外的处理（log1p 活跃度、bool/可空布尔转 float）与
    prepare_model_frame 完全一致，log 变换直接复用 mc.add_log_activity。
    """
    frame = mc.add_log_activity(user_df)
    frame[FOCAL_COLUMN] = _male_indicator(frame["gender"]).to_numpy()

    if "verified_flag" in frame.columns:
        frame["verified_flag"] = pd.to_numeric(
            frame["verified_flag"].astype("object"), errors="coerce"
        ).astype(float)
    if "profile_complete" in frame.columns:
        # 缺失一律当作"画像不完整"。用 .where 而不是 .fillna：后者在
        # object 列上会触发 pandas 的静默降型（未来版本行为还会变），
        # 这里的语义只是"把缺失换成 False"，不需要也不应该依赖降型。
        flag = frame["profile_complete"]
        frame["profile_complete"] = flag.astype("object").where(flag.notna(), False).astype(bool)
    return frame


def to_long_format(user_df, outcome_kind):
    """表 C -> 长表：每个用户恰好两行（public + celebrity）

    Args:
        user_df: 表 C（用户级宽表）
        outcome_kind: "source_entry" 或 "topical_share"

    **本函数不做任何剔除**，哪怕结果变量是 NaN、性别无法使用。理由：
    长表的行数是后续样本流失记账的唯一分母（n_input = len(long_df)//2），
    一旦在这里静默丢行，结果表里的 n_dropped 就失去了参照物。剔除规则
    集中在 _select_sample 里，并逐条写进 drop_reason。

    返回的行按 (user_id, domain) 排序，**同一用户的两行永远相邻**——
    这是 _cluster_bootstrap_did 重标簇号时依赖的硬不变量，改动排序前请
    先读那个函数。

    Returns:
        DataFrame：原表 C 的全部列 + user_id / cluster_id / domain /
        is_public / outcome / outcome_kind / male
    """
    if outcome_kind not in OUTCOME_KINDS:
        raise ValueError(
            f"未知的结果变量口径: {outcome_kind!r}，只支持 {sorted(OUTCOME_KINDS)}"
        )
    spec = OUTCOME_KINDS[outcome_kind]
    wide = _prepare_wide(user_df)

    blocks = []
    for domain in DOMAINS:
        column = spec["column"].format(domain=domain)
        if column not in wide.columns:
            raise KeyError(f"表 C 缺少结果变量列 {column}，无法构造 {outcome_kind} 长表")
        block = wide.copy()
        block[DOMAIN_COLUMN] = domain
        block[DOMAIN_INDICATOR] = 1.0 if domain == "public" else 0.0
        # bool 列（source_entered）先过 object 再转 float：可空布尔直接
        # astype(float) 遇到 pd.NA 会抛错，与 prepare_model_frame 同一处理
        block[OUTCOME_COLUMN] = pd.to_numeric(
            block[column].astype("object"), errors="coerce"
        ).astype(float)
        block[OUTCOME_KIND_COLUMN] = outcome_kind
        blocks.append(block)

    long_df = pd.concat(blocks, ignore_index=True)
    long_df[CLUSTER_COLUMN] = long_df[USER_COLUMN].astype(str)
    long_df = long_df.sort_values([USER_COLUMN, DOMAIN_COLUMN]).reset_index(drop=True)

    n_users = int(long_df[USER_COLUMN].nunique())
    if len(long_df) != 2 * n_users:
        raise ValueError(
            f"长表行数 {len(long_df)} 不等于 2 × 用户数 {n_users}："
            "表 C 里出现了重复的 user_id，长表的配对结构不成立，"
            "必须先修好上游的用户表，而不是在这里去重掩盖。"
        )
    return long_df


# ---------------------------------------------------------------------------
# 样本选择与记账（单位一律是用户，不是行）
# ---------------------------------------------------------------------------

def _select_sample(long_df, layer, outcome_kind=None):
    """按层构造估计样本，并以用户为单位记账

    剔除顺序与原因：
        1) 性别无法使用（male 为 NaN）-> missing_gender
        2) 两个域的结果变量都缺 -> 该结果变量自己的缺失原因
        3) 只缺一个域的结果变量 -> REASON_INCOMPLETE_PAIR（整个用户剔除，
           见模块文档第 4 条）
        4) M2 层画像不完整（profile_complete 为假或 region 缺失）
           -> incomplete_profile，与 models_core._layer_sample 同一口径

    Returns:
        (sample, n_input, n_obs, n_dropped, drop_reason)，其中 n_input /
        n_obs / n_dropped 都是**用户数**
    """
    if outcome_kind is None:
        outcome_kind = str(long_df[OUTCOME_KIND_COLUMN].iloc[0])
    spec = OUTCOME_KINDS[outcome_kind]

    n_input = int(long_df[USER_COLUMN].nunique())
    reasons = []

    keep = long_df[FOCAL_COLUMN].notna()
    if int((~keep).sum()) > 0:
        reasons.append("missing_gender")

    valid = long_df[OUTCOME_COLUMN].notna()
    n_valid = valid.groupby(long_df[USER_COLUMN]).transform("sum")
    if int(((n_valid == 0) & keep).sum()) > 0:
        reasons.append(spec["missing_reason"])
    if int(((n_valid == 1) & keep).sum()) > 0:
        reasons.append(REASON_INCOMPLETE_PAIR)
    keep &= (n_valid == 2)

    if layer == "M2":
        complete = (
            long_df["profile_complete"] if "profile_complete" in long_df.columns
            else pd.Series(False, index=long_df.index)
        )
        if "region" in long_df.columns:
            complete = complete & long_df["region"].notna()
        before = int(keep.sum())
        keep &= complete.astype(bool)
        if int(keep.sum()) < before:
            reasons.append("incomplete_profile")

    sample = long_df[keep].reset_index(drop=True)
    n_obs = int(sample[USER_COLUMN].nunique())
    n_dropped = n_input - n_obs
    drop_reason = "+".join(reasons) if reasons else None
    return sample, n_input, n_obs, n_dropped, drop_reason


def _reconcile_users(result, n_input, n_obs, n_dropped, drop_reason):
    """用真正进入拟合的用户数覆盖 n_obs，并守住样本量恒等式

    与 models_core._reconcile_nobs 同一职责，但那边的 result.nobs 就是
    用户数，这里的 result.nobs 是**行数**（每个用户两行），不能直接套用，
    否则 n_obs 会凭空翻倍。patsy 静默丢掉协变量含 NaN 的行时，本模块要
    求它成对地丢——真出现奇数行，说明有用户只剩一行进了模型，配对结构
    已经被破坏，此时报错而不是四舍五入。
    """
    n_rows = int(result.nobs)
    if n_rows % 2 != 0:
        raise ValueError(
            f"进入 GEE 的行数 {n_rows} 是奇数：某个用户只剩一行进了模型，"
            "长表的配对结构已被破坏（通常是某个协变量只在一个域的行上缺失）。"
            "必须先查清是哪一列，而不是让一个半配对的样本产出论文头条数字。"
        )
    fitted_users = n_rows // 2
    if fitted_users != n_obs:
        parts = [drop_reason] if drop_reason else []
        parts.append("missing_covariate")
        drop_reason = "+".join(parts)
        n_obs = fitted_users
        n_dropped = n_input - fitted_users
    if n_obs + n_dropped != n_input:
        raise ValueError(
            f"样本量记账不自洽: n_obs={n_obs} + n_dropped={n_dropped} "
            f"!= 输入用户数 {n_input}"
        )
    return n_obs, n_dropped, drop_reason


# ---------------------------------------------------------------------------
# 公式与拟合
# ---------------------------------------------------------------------------

def _build_formula(sample, layer):
    """构造 GEE 公式，返回 (formula, dropped)

    右侧固定以 "male * is_public" 开头（patsy 展开成 male + is_public +
    male:is_public，交互项因此永远在模型里），其余协变量沿用
    models_core._formula_terms 的翻译规则与"本样本内取值唯一就剔除"的
    处理，保证两个模块对同一层协变量的理解不会分叉。
    """
    covariates = dict(mc.MODEL_LAYERS)[layer]
    terms, dropped = mc._formula_terms(covariates, sample)
    others = [t for t in terms if t != FOCAL_COLUMN]
    rhs = " + ".join(["{} * {}".format(FOCAL_COLUMN, DOMAIN_INDICATOR)] + others)
    return "{} ~ {}".format(OUTCOME_COLUMN, rhs), dropped


def _family(outcome_kind):
    """两种结果变量都用 Binomial family + logit link

    - source_entry 是 0/1，就是标准的二值 GEE；
    - topical_share 是 [0,1] 上的占比，GEE 本身是拟似然估计、默认协方差
      又是稳健三明治，因此 "Binomial family + logit link" 在这里就是准二项
      （分数 logit）的面板版本，与 models_core 里 GLM(Binomial)+HC1 同源
      （见模块文档第 6 条）。
    """
    return sm.families.Binomial()


def _fit_gee(sample, formula, outcome_kind):
    """拟合一次 GEE：exchangeable 工作相关，按 user 聚类

    聚类变量用 CLUSTER_COLUMN 而不是 USER_COLUMN：bootstrap 重抽样时同一个
    用户可能被抽中多次，那几份副本必须是不同的簇，否则 GEE 会把 4 行当成
    一个簇（见 _cluster_bootstrap_did）。原始样本上两者取值相同。
    """
    model = smf.gee(
        formula, groups=CLUSTER_COLUMN, data=sample,
        family=_family(outcome_kind), cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit()


# ---------------------------------------------------------------------------
# 差中差：四个反事实格子
# ---------------------------------------------------------------------------

def _cell_exog(fit, data, male, is_public):
    """构造 (male, is_public) 被强制取定值的反事实设计矩阵

    复用拟合时的 design_info 重新生成，保证列顺序与哑变量编码和原模型
    完全一致——与 stats_utils._design_matrices 同一思路，只是这里要同时
    强制两个变量，而那个函数只处理单个 focal 变量。
    """
    frame = data.copy()
    frame[FOCAL_COLUMN] = float(male)
    frame[DOMAIN_INDICATOR] = float(is_public)
    design_info = getattr(fit.model.data, "design_info", None)
    if design_info is None:
        raise ValueError(
            "GEE 结果里没有 patsy 的 design_info，无法构造反事实设计矩阵；"
            "本模块的模型必须用公式接口拟合。"
        )
    return patsy.dmatrix(design_info, frame, return_type="dataframe").values


def _did_at_params(fit, params, exogs):
    """给定一组参数，算四格反事实预测的均值并组合成差中差

    四个格子都在**同一批行**上预测，而每个用户恰好贡献两行、协变量在两行
    上完全相同，因此"按行取均值"与"按用户取均值"是同一个数——报告口径是
    每个用户一票，与 models_core 的 AME、describe 的表 2 同一个总体单位。
    """
    preds = {}
    for key, exog in exogs.items():
        preds[key] = float(np.mean(fit.model.predict(params, exog)))
    gap_public = preds[(1, 1)] - preds[(0, 1)]
    gap_celebrity = preds[(1, 0)] - preds[(0, 0)]
    return gap_public - gap_celebrity, gap_public, gap_celebrity, preds


def _cov_params_or_none(fit):
    """取参数协方差矩阵；只有"确实不可用"时返回 None

    判定标准与 stats_utils._get_cov_params_or_none 严格一致：没有
    cov_params 方法，或调用时抛 NotImplementedError。其它异常一律向上抛，
    不允许在这里被吞掉后让头条数字悄悄换成另一条路径算出来的区间。
    """
    fn = getattr(fit, "cov_params", None)
    if fn is None:
        return None
    try:
        cov = fn()
    except NotImplementedError:
        return None
    return np.asarray(cov, dtype=float)


def _did_point(fit, data):
    """只算差中差的点估计（bootstrap 每次重抽样调用的就是它）"""
    exogs = {
        (male, is_public): _cell_exog(fit, data, male, is_public)
        for male in (0, 1) for is_public in (0, 1)
    }
    params = np.asarray(fit.params, dtype=float)
    did, _, _, _ = _did_at_params(fit, params, exogs)
    return did


def _assert_paired(frame):
    """长表配对的硬检查：相邻两行必须同一个用户、且覆盖两个域

    这个检查放在 bootstrap 的每一次重抽样上，是因为"配对被打散"不会让
    程序崩溃，只会让论文头条数字的区间悄悄变窄——那是一个不报错的错误，
    必须自己长出一个报错来。
    """
    if len(frame) % 2 != 0:
        raise ValueError(
            f"长表行数 {len(frame)} 不是偶数：存在只剩一行的用户，配对结构已被破坏"
        )
    users = frame[USER_COLUMN].to_numpy()
    domains = frame[DOMAIN_COLUMN].to_numpy()
    if not (users[0::2] == users[1::2]).all():
        raise ValueError("相邻两行不属于同一个用户：cluster bootstrap 的配对已被打散")
    if not (domains[0::2] != domains[1::2]).all():
        raise ValueError("同一个用户的两行落在同一个域上：长表构造有误")


def _relabel_clusters(frame):
    """给重抽样后的每一对行分配唯一簇号

    有放回抽样里同一个用户可能被抽中多次，如果两份副本共用原始 user_id，
    GEE 会把它们当成一个 4 行的簇，exchangeable 相关参数因此被算错。
    这里按"每两行一簇"重新编号（配对连续由 _assert_paired 保证）。
    """
    n_pairs = len(frame) // 2
    out = frame.copy()
    out[CLUSTER_COLUMN] = np.repeat(np.arange(n_pairs), 2).astype(str)
    return out


def _cluster_bootstrap_did(sample, statistic, n_boot=_N_BOOT, seed=_SEED,
                           confidence=_CONFIDENCE):
    """按用户整簇重抽样的百分位 bootstrap

    重抽样本身完全交给 stats_utils.bootstrap_ci（cluster 参数会把一个被
    抽中的簇连同其全部行整体带走，那段逻辑有自己的单测，本模块不重写）。
    本函数在它外面加两件事：
        1) 每次重抽样先检查配对完整（_assert_paired），再重标簇号
           （_relabel_clusters），然后才交给 statistic；
        2) 统计量抛异常（例如重拟合不收敛）时计一次失败并返回 NaN。
           **不丢弃失败的重抽样再用剩下的算百分位**：能收敛的重抽样是
           参数空间里被挑选过的一部分，丢掉失败的那些会让区间偏窄。
           失败次数返回给调用方写进 note，区间随之为 NaN——一个必须被人
           看到的信号，而不是一个悄悄变窄的区间。

    Args:
        sample: 长表样本，行按 (user_id, domain) 排序、同一用户两行相邻
        statistic: 接收一份重抽样长表、返回一个标量的可调用对象

    Returns:
        (est, low, high, n_failed)
    """
    _assert_paired(sample)
    failures = {"n": 0}

    def _wrapped(frame):
        _assert_paired(frame)
        relabeled = _relabel_clusters(frame)
        try:
            return float(statistic(relabeled))
        except Exception as exc:  # noqa: BLE001 —— 失败要计数，不能被丢掉
            failures["n"] += 1
            print(f"警告: cluster bootstrap 的某次重抽样失败（{type(exc).__name__}: {exc}）")
            return np.nan

    est, low, high = su.bootstrap_ci(
        sample, _wrapped, n_boot=n_boot, seed=seed,
        cluster=sample[USER_COLUMN].to_numpy(), confidence=confidence,
    )
    return est, low, high, failures["n"]


def difference_in_differences(fit, long_df, fit_fn=None, method="auto",
                              n_boot=_N_BOOT, seed=_SEED, confidence=_CONFIDENCE):
    """论文真正报告的量：概率/比例尺度上的差中差

        DiD = (男 - 女)_公共事务 - (男 - 女)_娱乐

    做法是四格反事实预测：把估计样本的每一行分别按 (male, is_public) 的
    四种组合强制取值，各自预测再取均值，然后做两次差。**不是**读交互项
    系数（logit 尺度的系数不是概率差之差），**也不是**只在"代表性个体"
    上算一次（那是 MEM，非线性模型里与逐行平均不是同一个数）。

    区间的两条路径：
    - method="delta"（默认 auto 下的首选）：把 DiD 看作参数 beta 的函数，
      中心差分求数值梯度，Var = grad' cov_params() grad。GEE 的默认协方差
      就是按簇（用户）的稳健三明治矩阵，所以这个区间本身已经是聚类稳健的，
      而且只需拟合一次——在 22.5 万用户上这是唯一现实的选择。
    - method="bootstrap"：以用户为簇整簇重抽样，每次重抽样上**重新拟合**
      GEE 再重算 DiD，取百分位区间。需要调用方给出 fit_fn。

    auto 的规则：cov_params() 可用且算得出有限的 se 就用 delta，否则退到
    bootstrap。cov_params() 不可用而调用方又没给 fit_fn 时直接报错——
    论文的中心数字不接受"没有区间"这个终态（模块文档第 5 条），也不接受
    用别的公式补一个看起来自信的数字上去。

    Args:
        fit: 已拟合的 GEE 结果（或提供 .params / .model.predict /
            .model.data.design_info / 可选 .cov_params() 的对象）
        long_df: 估计样本（进入拟合的那一份长表）
        fit_fn: 可调用对象，接收一份重抽样长表、返回重新拟合的结果对象

    Returns:
        dict：did / se / ci_low / ci_high / method / gap_public /
        gap_celebrity / cells / n_users / n_boot_failed
    """
    if method not in ("auto", METHOD_DELTA, METHOD_BOOTSTRAP):
        raise ValueError(f"未知的区间方法: {method!r}")

    exogs = {
        (male, is_public): _cell_exog(fit, long_df, male, is_public)
        for male in (0, 1) for is_public in (0, 1)
    }
    params = np.asarray(fit.params, dtype=float)
    did, gap_public, gap_celebrity, cells = _did_at_params(fit, params, exogs)

    out = {
        "did": did,
        "gap_public": gap_public,
        "gap_celebrity": gap_celebrity,
        "cells": cells,
        "n_users": int(long_df[USER_COLUMN].nunique()),
        "n_boot_failed": 0,
    }

    cov = None if method == METHOD_BOOTSTRAP else _cov_params_or_none(fit)
    if cov is not None:
        grad = np.zeros_like(params)
        for j in range(len(params)):
            step = 1e-6 * max(1.0, abs(params[j]))
            p_plus = params.copy()
            p_plus[j] += step
            p_minus = params.copy()
            p_minus[j] -= step
            grad[j] = (
                _did_at_params(fit, p_plus, exogs)[0]
                - _did_at_params(fit, p_minus, exogs)[0]
            ) / (2 * step)
        var = float(grad @ cov @ grad)
        if np.isfinite(var) and var >= 0:
            se = float(np.sqrt(var))
            z = float(norm.ppf(1 - (1 - confidence) / 2))
            out.update({
                "se": se, "ci_low": did - z * se, "ci_high": did + z * se,
                "method": METHOD_DELTA,
            })
            return out
        print(
            f"警告: delta method 得到无效方差 var={var!r}，改用重拟合的 cluster bootstrap"
        )

    if method == METHOD_DELTA:
        raise ValueError(
            "method='delta' 但 cov_params() 不可用或方差无效：这个模型拿不到"
            "delta method 区间，请改用 method='bootstrap' 并提供 fit_fn。"
        )
    if fit_fn is None:
        raise ValueError(
            "GEE 的 cov_params() 不可用，delta method 无法计算，而调用方没有提供"
            " fit_fn，无法做重拟合的 cluster bootstrap。论文的中心数字不接受"
            "一个没有置信区间的点估计——请提供 fit_fn（接收重抽样长表、返回"
            "重新拟合的结果对象）。"
        )

    def _statistic(frame):
        return _did_point(fit_fn(frame), frame)

    est, low, high, n_failed = _cluster_bootstrap_did(
        long_df, _statistic, n_boot=n_boot, seed=seed, confidence=confidence
    )
    if n_failed:
        print(
            f"警告: cluster bootstrap 有 {n_failed}/{n_boot} 次重抽样重拟合失败，"
            "本模块不丢弃失败的重抽样（丢弃会让区间偏窄），因此这一行没有区间。"
        )
    out.update({
        # 百分位 bootstrap 的区间不对称，反推一个 se 会丢掉区间形状
        # （stats_utils 模块文档第 5 条），因此 se 留 NaN
        "did": est, "se": np.nan, "ci_low": low, "ci_high": high,
        "method": METHOD_BOOTSTRAP, "n_boot_failed": n_failed,
    })
    return out


# ---------------------------------------------------------------------------
# 单层结果表
# ---------------------------------------------------------------------------

def _nan_rows(outcome, model, scale, n_obs, n_dropped, drop_reason, note):
    """一层拟合不成立时的四行留痕（与 models_core 第 7 条同一原则）"""
    rows = []
    for term, domain, row_scale in (
        (TERM_DID, DOMAIN_BOTH, scale),
        (TERM_GAP_PUBLIC, "public", scale),
        (TERM_GAP_CELEBRITY, "celebrity", scale),
        (TERM_COEF, DOMAIN_BOTH, "log_odds"),
    ):
        rows.append(mc._nan_row(outcome, domain, model, term, row_scale,
                                n_obs, n_dropped, drop_reason, note))
    return rows


def fit_interaction(long_df, model_layer, outcome_kind=None, did_method="auto",
                    n_boot=_N_BOOT, seed=_SEED):
    """某一层协变量下的 Gender × Domain 交互，返回共享 schema 的结果行

    每层四行：
        TERM_DID           跨域（domain=DOMAIN_BOTH）——**头条**，概率/比例尺度
        TERM_GAP_PUBLIC    公共事务域内的性别 AME
        TERM_GAP_CELEBRITY 娱乐域内的性别 AME
        TERM_COEF          交互项系数（log_odds 尺度，附加参考，不是头条）

    两个域内的性别 AME 直接调用 stats_utils.average_marginal_effect（把
    is_public 强制成该域后对 male 做反事实预测），因此它们与 models_core
    里各表的 AME 是同一个函数算出来的。DiD 不能用"两个 AME 相减 + 方差
    相加"来得到区间：两者高度相关，忽略协方差会得到一个明显偏宽的区间，
    所以 DiD 的不确定性单独用 delta method / bootstrap 算（见
    difference_in_differences）。点估计上 DiD 恒等于两者之差，这条恒等式
    被 test_did_equals_the_difference_of_the_two_domain_gaps 守着。
    """
    if outcome_kind is None:
        outcome_kind = str(long_df[OUTCOME_KIND_COLUMN].iloc[0])
    spec = OUTCOME_KINDS[outcome_kind]
    outcome, scale = spec["outcome"], spec["scale"]

    sample, n_input, n_obs, n_dropped, drop_reason = _select_sample(
        long_df, model_layer, outcome_kind=outcome_kind
    )
    formula, dropped = _build_formula(sample, model_layer)
    base_note = mc._join_notes(
        NOTE_GEE, "dropped_constant:" + ",".join(dropped) if dropped else None
    )

    # 设计矩阵宽度按展开后的三项（male / is_public / 交互）计，
    # 直接把 "male * is_public" 交给 _design_width 会少算两列
    width_terms = [FOCAL_COLUMN, DOMAIN_INDICATOR, "male_x_is_public"] + [
        t for t in mc._formula_terms(dict(mc.MODEL_LAYERS)[model_layer], sample)[0]
        if t != FOCAL_COLUMN
    ]
    blocked = mc._blocked_note(sample, width_terms)
    if blocked is None and sample[DOMAIN_INDICATOR].nunique() < 2:
        blocked = "no_domain_variation"
    if blocked is None and spec["binary"] and sample[OUTCOME_COLUMN].nunique() < 2:
        blocked = "no_outcome_variation"
    if blocked is not None:
        note = mc._join_notes(base_note, blocked)
        return pd.DataFrame(
            _nan_rows(outcome, model_layer, scale, n_obs, n_dropped, drop_reason, note),
            columns=list(su.RESULT_SCHEMA),
        )

    label = f"{outcome}/interaction/{model_layer}"
    fit, fail = mc._safe_fit(lambda: _fit_gee(sample, formula, outcome_kind), label)
    if fit is None:
        note = mc._join_notes(base_note, fail)
        return pd.DataFrame(
            _nan_rows(outcome, model_layer, scale, n_obs, n_dropped, drop_reason, note),
            columns=list(su.RESULT_SCHEMA),
        )

    n_obs, n_dropped, drop_reason = _reconcile_users(
        fit, n_input, n_obs, n_dropped, drop_reason
    )
    # 准分离检查沿用 models_core 的那道事后检查：GEE 与 GLM 一样会在
    # （准）完全分离时给出自称收敛、系数却荒谬大的结果，命中就在 note 里
    # 点名，但不删除该行
    note = mc._join_notes(base_note, mc._glm_separation_note(fit))

    did = difference_in_differences(
        fit, sample,
        fit_fn=lambda frame: _fit_gee(frame, formula, outcome_kind),
        method=did_method, n_boot=n_boot, seed=seed,
    )
    did_note = mc._join_notes(
        note, "did_ci:" + did["method"],
        "boot_refit_failed:{}".format(did["n_boot_failed"]) if did["n_boot_failed"] else None,
    )

    rows = [su.tidy_result(
        outcome=outcome, domain=DOMAIN_BOTH, model=model_layer, term=TERM_DID,
        estimate=did["did"], se=did["se"], ci_low=did["ci_low"], ci_high=did["ci_high"],
        scale=scale, n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
        note=did_note,
    )]

    # 两个域内的性别 AME：把 is_public 固定成该域后复用共享的 AME 函数
    for domain, is_public, term in (
        ("public", 1.0, TERM_GAP_PUBLIC), ("celebrity", 0.0, TERM_GAP_CELEBRITY)
    ):
        forced = sample.copy()
        forced[DOMAIN_INDICATOR] = is_public
        ame, se, low, high = su.average_marginal_effect(
            fit, FOCAL_COLUMN, forced, confidence=_CONFIDENCE
        )
        rows.append(su.tidy_result(
            outcome=outcome, domain=domain, model=model_layer, term=term,
            estimate=ame, se=se, ci_low=low, ci_high=high, scale=scale,
            n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
            note=mc._join_notes(note, "counterfactual_domain"),
        ))

    # 交互项系数（log_odds 尺度）：附加参考行，永远不单独出现
    coef_name = "{}:{}".format(FOCAL_COLUMN, DOMAIN_INDICATOR)
    coef = float(fit.params[coef_name])
    bse = float(fit.bse[coef_name])
    rows.append(su.tidy_result(
        outcome=outcome, domain=DOMAIN_BOTH, model=model_layer, term=TERM_COEF,
        estimate=coef, se=bse, ci_low=coef - _Z * bse, ci_high=coef + _Z * bse,
        scale="log_odds", n_obs=n_obs, n_dropped=n_dropped, drop_reason=drop_reason,
        note=mc._join_notes(note, "not_the_headline"),
    ))
    return pd.DataFrame(rows, columns=list(su.RESULT_SCHEMA))


def interaction_table(user_df, did_method="auto", n_boot=_N_BOOT, seed=_SEED):
    """两个结果变量 × 三层协变量的完整交互结果表"""
    frames = []
    for outcome_kind in OUTCOME_KINDS:
        long_df = to_long_format(user_df, outcome_kind)
        for layer, _ in mc.MODEL_LAYERS:
            print(f"拟合 Gender × Domain 交互: {outcome_kind} / {layer}")
            frames.append(fit_interaction(
                long_df, layer, outcome_kind=outcome_kind,
                did_method=did_method, n_boot=n_boot, seed=seed,
            ))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build(year=config.YEAR):
    """跑完 §6.6 交互模型并写出 interaction_gender_domain.parquet + manifest"""
    user_df, source_path = describe._load_user_table(year)
    frame = interaction_table(user_df)

    out_dir = os.path.join(config.OUTPUT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "interaction_gender_domain.parquet")
    frame.to_parquet(path, engine="pyarrow", index=False)
    n_failed = int(frame["estimate"].isna().sum())
    print(f"已保存: {path}（{len(frame)} 行，其中 {n_failed} 行为拟合失败留痕）")

    did_rows = frame[frame["term"] == TERM_DID]
    manifest = config.build_manifest(
        step=f"models_interaction_{year}",
        inputs=[os.path.relpath(source_path, config.OUTPUT_DIR)],
        params={
            "year": year,
            "M0": mc.M0, "M1": mc.M1, "M2": mc.M2,
            "estimator": "GEE(Binomial, logit) + exchangeable, cluster=user_id",
            "headline": TERM_DID,
            "confidence": _CONFIDENCE,
            "n_boot": _N_BOOT,
        },
        counts={
            "n_users": int(len(user_df)),
            "rows": int(len(frame)),
            "failed_rows": n_failed,
            # 头条那几行的区间是 delta method 还是重拟合 bootstrap 算的，
            # 必须写进 manifest：这决定了读者怎么理解那个区间
            "did_ci_method": did_rows["note"].dropna().tolist(),
        },
    )
    config.write_manifest(manifest, os.path.join(out_dir, "manifest_interaction"))
    return path


if __name__ == "__main__":
    fire.Fire({"build": build})
