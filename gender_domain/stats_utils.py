"""
共享统计工具：置信区间、bootstrap、边际效应、结果表统一 schema。

纯函数模块，不做文件 IO、不读 parquet、不联网，与 text_rules.py /
id_rules.py 同一思路，方便脱离服务器环境单测。后续所有 results 层模块
（describe.py、models_core.py 等）都直接调用本模块，不得各自重新实现
同一类估计逻辑——这是本模块存在的唯一理由。

统计口径背景（研究负责人裁定，写在这里方便下游模块的作者理解"为什么
这样写"而不是照抄语法）：
1. 样本量约 22.5 万，在这个量级上任何微小差异都会"统计显著"，因此论文
   以效应量 + 置信区间为主，p 值退居次要。本模块里每一个函数存在的目的
   都是产出一个区间，而不是一个"是否显著"的结论。
2. 比例的置信区间一律用 Wilson score interval，而不是正态近似区间：
   本研究里若干结果的比例非常接近 0 或 1（例如某些来源参与率），正态
   近似区间在这种情况下会越过 [0,1] 边界，这种明显错误的区间一旦印到
   论文表格里是灾难性的。
3. bootstrap 的 cluster 参数是本模块最容易被以为写对、实则写错的一处：
   后续模型建立在长表数据上，每个用户贡献两行（公共事务域 + 娱乐域各
   一行）。如果按行重抽样，会把同一用户两行之间的相关性直接当作独立
   信息，系统性低估标准误。正确做法是整簇（用户）连同其名下所有行一起
   放回抽样，再把抽中的簇拼起来算统计量——不是"抽簇内的行"，也不是
   "只抽簇标签、不带走对应的行"。
4. 边际效应一律报告在概率/比例尺度上，而不是发生比（odds ratio）：
   论文要把 M0/M1/M2 三层协变量集合的模型并排比较，发生比在协变量集合
   不同的模型之间不可比，概率尺度的平均边际效应（AME）才可比。
5. 本模块的区间型函数（proportion_ci、proportion_diff_ci、risk_ratio_ci）
   只返回 (low, high)，故意不附带一个 se 字段：Wilson/Newcombe/对数尺度
   区间本来就不是对称于点估计的，(high-low)/(2z) 这种反推只在正态近似
   下才有意义，用它给这些非对称区间"补"一个 se 会丢失区间形状携带的
   信息，还可能让下游模块以为拿到了一个可以自由做加减法的正态近似量。
   下游模块如果确实需要 se，应该直接使用区间本身，或者改用 bootstrap_ci
   拿到的百分位区间，不要在这些函数外自己反推一个数字。
6. 缺置信区间比给一个偏窄的置信区间更安全：某个数量真的没办法算出
   不确定性时（例如协方差矩阵不可用），本模块的原则是返回 NaN 而不是
   退化成一个看起来自信、实际低估了不确定性的区间——一个缺失的区间在
   论文里是可以说明原因的，一个悄悄偏窄 10 倍的区间不是。
"""

import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm

# 共享结果表 schema：所有 models_*.parquet 都必须恰好是这些列，
# 这样一套绘图函数就能服务所有结果表。列的含义见方案文档
# docs/superpowers/plans/2026-08-16-domain-participation-results.md
# 的 "Shared result schema" 一节。
RESULT_SCHEMA = (
    "outcome",
    "domain",
    "model",
    "term",
    "estimate",
    "se",
    "ci_low",
    "ci_high",
    "scale",
    "n_obs",
    "n_dropped",
    "drop_reason",
    "note",
)

# `scale` 是一个**封闭词表**，不是自由文本。它唯一的用途是告诉绘图层
# "这一行该画在什么轴上、参照线画在哪里"，所以每多一个拼写变体，就多一个
# 会被下游 if/else 漏掉的分支：方案文档只声明了四个取值，实现里一度长到
# 十个，其中三个还是同一件事的三种写法。因此这里把允许的取值写死，由
# tidy_result 在构造每一行时就地拒绝表外的取值——与它拒绝未知字段名
# 完全同一个理由：静默接受一个没人认识的取值，比报错更容易一路混进论文。
#
# 分组说明（新增取值前先想清楚它属于哪一组，再往对应的元组里加）：
#
# 1) 落在 [0,1] 上的水平量，线性轴，无参照线（或参照线在 0，差值型时）：
#    probability   二值事件的概率（进入率、预测格子）
#    proportion    用户层面的占比（话题占比、参与月份占比）
# 2) 对称的差值/系数量，线性轴，参照线在 0：
#    log_odds      logit 尺度的回归系数（含交互项系数）
#    log1p_count   log(1+计数) 尺度的 OLS 系数
#    correlation   相关系数（phi / Spearman），天然落在 [-1,1]
# 3) 比值量，**对数轴，参照线在 1**——三者共用同一种画法，但**不是同一个
#    估计量**，因此刻意不合并成一个名字：
#    risk_ratio    两个比例（或两组用户层面均值）之比。describe.py 原来把
#                  它写成含糊的 "ratio"，与下面两个混在一起看不出区别，
#                  这里改名为 risk_ratio，与 stats_utils.risk_ratio_ci 同名。
#    odds_ratio    两个发生比之比（logit 系数取指数），在协变量集合不同的
#                  模型之间不可比（见 models_core 模块文档第 1 条），
#                  与 risk_ratio 数值上也不相等（罕见事件下才近似）。
#    irr           零截断负二项的发生率比，且是"潜在未截断均值之比"
#                  （models_core 模块文档第 4 条），既不是 risk_ratio 也
#                  不是 odds_ratio。
#    需要"这三个都画成对数轴 + 参照线 1"时用 RATIO_SCALES，不要在下游
#    各自硬编码一份名字清单。
# 4) 带单位的量，线性轴，无参照线：
#    hours              小时（转发延迟）
#    share_of_behavior  头部集中度份额（前 q 比例用户占行为总量的份额）；
#                       它是一个 [0,1] 的份额，但分母是"行为总量"而不是
#                       "用户总数"，与 proportion 不是同一个总体单位，
#                       所以单列一个名字而不是并进 proportion。
RATIO_SCALES = ("risk_ratio", "odds_ratio", "irr")

RESULT_SCALES = (
    "probability",
    "proportion",
    "log_odds",
    "log1p_count",
    "correlation",
) + RATIO_SCALES + (
    "hours",
    "share_of_behavior",
)

_DEFAULT_CONFIDENCE = 0.95


def _z_value(confidence):
    """双侧置信区间对应的正态分位数"""
    return norm.ppf(1 - (1 - confidence) / 2)


# ---------------------------------------------------------------------------
# 比例的置信区间
# ---------------------------------------------------------------------------

def proportion_ci(successes, n, method="wilson", confidence=_DEFAULT_CONFIDENCE):
    """单个比例的置信区间，默认 Wilson score interval

    Wilson 区间闭式解（Wilson, 1927）：
        z = 置信水平对应的正态分位数
        p_hat = successes / n
        center = (p_hat + z^2/(2n)) / (1 + z^2/n)
        margin = z * sqrt(p_hat(1-p_hat)/n + z^2/(4n^2)) / (1 + z^2/n)
        区间 = [center - margin, center + margin]
    该公式天然落在 [0,1] 内（是对二项分布似然的一个二次方程求根得到的），
    但仍显式 clip 一次防御浮点误差，保证"绝不越界"这个承诺不依赖运气。

    Args:
        successes: 成功次数（可以是数组，逐元素计算）
        n: 试验次数
        method: 目前只实现 "wilson"，预留参数是为了未来可能加正态近似区间
            用于对比，但不应该被下游当作默认选项使用
        confidence: 置信水平，默认 0.95

    Returns:
        (low, high)，与输入同形状
    """
    if method != "wilson":
        raise ValueError(f"未实现的区间方法: {method!r}，目前只支持 'wilson'")
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = _z_value(confidence)
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    low = np.clip(center - margin, 0.0, 1.0)
    high = np.clip(center + margin, 0.0, 1.0)
    if low.ndim == 0:
        return float(low), float(high)
    return low, high


def proportion_diff_ci(s1, n1, s2, n2, confidence=_DEFAULT_CONFIDENCE):
    """两个独立比例之差的置信区间：Newcombe（1998）方法 10

    做法是把两条 Wilson 区间"拼"成一条差值区间，而不是用正态近似的
    合并标准误——这样在 p 接近 0/1 时依然稳健。设 p_i = s_i/n_i，
    [l_i, u_i] 为各自的 Wilson 区间，则：
        diff = p1 - p2
        low  = diff - sqrt((p1-l1)^2 + (u2-p2)^2)
        high = diff + sqrt((u1-p1)^2 + (p2-l2)^2)

    交换 (s1,n1) 与 (s2,n2) 时，diff 变号，low/high 互换后各自取反
    （low' = -high, high' = -low）——区间宽度不变，只是围绕新的差值
    对称地重新定位，这是 Newcombe 方法本身的代数性质，不是巧合。

    与 `proportion_ci` 不同，本函数只接受标量 s1/n1/s2/n2，不做向量化；
    需要批量计算请显式循环。另见模块文档第 5 条：不返回 se。

    Returns:
        (diff, low, high)
    """
    p1, p2 = s1 / n1, s2 / n2
    l1, u1 = proportion_ci(s1, n1, confidence=confidence)
    l2, u2 = proportion_ci(s2, n2, confidence=confidence)
    diff = p1 - p2
    low = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, low, high


def risk_ratio_ci(s1, n1, s2, n2, confidence=_DEFAULT_CONFIDENCE):
    """两个独立比例之比（risk ratio）的对数尺度置信区间

    标准做法：在 log(rr) 尺度上做正态近似再指数变换回来，因为 rr 本身
    是右偏的，对数尺度的正态近似远比原始尺度准确：
        rr = (s1/n1) / (s2/n2)
        SE[log(rr)] = sqrt(1/s1 - 1/n1 + 1/s2 - 1/n2)
        区间 = exp(log(rr) ± z * SE)

    rr 用交叉相乘 (s1*n2)/(n1*s2) 计算，避免两次浮点除法各自舍入后
    相除引入误差（例如 20/100 和 10/100 各自算出的浮点数相除不一定
    恰好等于 2.0，交叉相乘可以保证整数输入时得到精确值）。

    任一分子为 0 时，log(rr) 未定义（0 或除以 0），区间没有意义；
    这里返回 NaN 边界而不是抛异常——描述性表格里出现"某组这项行为
    从未发生"是完全正常的数据事实，不该让整张表的计算因此中断。
    s2=0 时 rr 本身统一记为 NaN（即便 s1>0，"分母组从未发生"也已经
    让"比率"这个概念本身失去意义，不应该返回一个看似有信息量的 inf，
    那会在下游排序/汇总时造成误导）。

    本函数只接受标量 s1/n1/s2/n2（与 `proportion_ci` 不同，后者对
    `successes`/`n` 做了逐元素向量化）。这里不做向量化是因为分子为 0
    时的分支处理天然是逐对进行的；调用方如果有一批 (s1,n1,s2,n2)
    需要计算，请显式循环或用 `np.vectorize` 包一层，不要假设本函数
    能直接接受数组输入。
    另见模块文档第 5 条：本函数不返回 se，只返回对数尺度正态近似
    换算回原始尺度后的 (low, high)。

    Returns:
        (rr, low, high)；任一分子为 0 时 rr 记为 NaN，low/high 恒为 NaN。
    """
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    if s2 == 0:
        rr = np.nan
    else:
        rr = (s1 * n2) / (n1 * s2)
    if s1 == 0 or s2 == 0:
        return rr, np.nan, np.nan
    log_rr = np.log(rr)
    se = np.sqrt(1 / s1 - 1 / n1 + 1 / s2 - 1 / n2)
    z = _z_value(confidence)
    low = np.exp(log_rr - z * se)
    high = np.exp(log_rr + z * se)
    return rr, low, high


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def check_proportion_range(values, label, lower=0.0, upper=1.0, max_examples=5):
    """校验占比型变量确实落在 [0,1] 内，越界立即报错

    本项目所有占比型结果变量都用二项族的拟似然（分数 logit / 准二项，
    见 models_core 模块文档第 5 条与 models_interaction 第 6 条）来估计，
    而"取值落在 [0,1] 内"是这个族适用的前提，不是一句客套话：越界的值
    照样能让 IRLS 收敛、照样给出一个干净的 note 和一个看起来正常的估计，
    只是把结果悄悄推走——实测在 22.5 万用户里注入 5 个 1.7 的占比，
    交互项的头条数字就从 0.1994 变成 0.1796，而结果表上没有任何痕迹。

    占比越界只可能是上游算错了分子分母（例如分母用了表达帖、分子却数了
    全部帖子），属于必须有人去修的 bug，因此这里抛异常而不是 clip、也不是
    记一行 note：clip 会把一个上游 bug 变成一个永远不会被发现的小偏差。

    NaN 是合法的（"该用户此项无定义"，见项目全局约定 NaN is not zero），
    不参与校验。

    Args:
        values: 待校验的数值序列
        label: 出错信息里用来指认这一列的名字（用列名，方便直接去查上游）
        lower / upper: 合法闭区间，默认 [0, 1]
        max_examples: 报错信息里最多列出几个越界样例

    Returns:
        校验通过时返回有效（非 NaN）观测数，方便调用方顺手记账
    """
    arr = np.asarray(pd.Series(values).astype(float))
    valid = arr[~np.isnan(arr)]
    bad = valid[(valid < lower) | (valid > upper)]
    if bad.size:
        examples = ", ".join("{:.6g}".format(v) for v in bad[:max_examples])
        raise ValueError(
            f"占比型变量 {label} 有 {bad.size} 个观测落在 [{lower}, {upper}] 之外"
            f"（最小 {valid.min():.6g}，最大 {valid.max():.6g}，例如 {examples}）。"
            "二项族拟似然（分数 logit / 准二项）以取值落在该区间为前提，越界值"
            "会在不报错的情况下把估计值推走。这只可能是上游分子/分母口径算错了，"
            "请回到构表环节修正，不要在这里 clip 或忽略。"
        )
    return int(valid.size)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _take_rows(values, idx):
    """按整数下标数组取行，兼容 ndarray / pandas Series / DataFrame"""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values.iloc[idx].reset_index(drop=True)
    return np.asarray(values)[idx]


def bootstrap_ci(values, statistic, n_boot=1000, seed=0, cluster=None,
                  confidence=_DEFAULT_CONFIDENCE):
    """百分位 bootstrap 置信区间

    Args:
        values: 观测值，ndarray / pandas Series / DataFrame 均可，
            长度为样本量；`statistic` 直接消费重抽样后的同类型对象
        statistic: 可调用对象，接收与 `values` 同类型的重抽样结果，
            返回一个标量
        n_boot: 重抽样次数
        seed: 随机种子，固定后两次调用结果完全一致（内部每次调用都用
            这个种子重新构造一个 `numpy.random.default_rng`，不依赖
            任何外部随机状态）
        cluster: 可选，长度与 `values` 相同的簇标签数组。给定时按簇
            整体重抽样（见下），而不是按行重抽样——这是本函数最容易
            被写错的地方，务必读完下面的说明再改动这段逻辑。
        confidence: 置信水平

    簇重抽样的正确做法：
        1) 先按簇标签把行分组，得到 {簇标签: 该簇所有行的下标数组}；
        2) 每次重抽样，从"簇标签的全集"里有放回抽出与簇数相同个数的
           标签（同一个簇可能被抽中 0 次、1 次或多次）；
        3) 把抽中的每个簇标签对应的**全部原始行**依次拼接起来，作为
           这一次重抽样的样本，再对这份样本调用 `statistic`。
    这与两种似是而非的错误实现的区别：
        - 错误 A："按簇重抽样标签，但每个簇只取一行/取簇的汇总值"——
          丢失了簇内的行级信息，且没有真正"连同该簇的所有行"重抽样；
        - 错误 B："在每个簇内部对行做独立重抽样"——簇与簇之间倒是没
          抽错，但簇内那几行的组合被打散了，等价于放大了自由度，
          还是会低估真实的组间方差。
        本实现每次重抽样时，一个被抽中的簇，其名下的行永远整体出现，
        顺序和组合与原始数据完全一致，不会被打散，也不会被替换成汇总量。

    Returns:
        (est, low, high)；est 是在原始（未重抽样）样本上算出的点估计
    """
    n = len(values)
    est = statistic(values)
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)

    if cluster is None:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_stats[b] = statistic(_take_rows(values, idx))
    else:
        cluster_arr = np.asarray(cluster)
        if len(cluster_arr) != n:
            raise ValueError("cluster 长度必须与 values 一致")
        unique_clusters = np.unique(cluster_arr)
        n_clusters = len(unique_clusters)
        # 预先建好 簇标签 -> 行下标数组 的映射，重抽样时直接拼接，
        # 保证每个被抽中的簇，其全部行整体出现、内部顺序不变
        cluster_to_rows = {
            c: np.where(cluster_arr == c)[0] for c in unique_clusters
        }
        for b in range(n_boot):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            idx = np.concatenate([cluster_to_rows[c] for c in sampled_clusters])
            boot_stats[b] = statistic(_take_rows(values, idx))

    alpha = 1 - confidence
    low, high = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(low), float(high)


# ---------------------------------------------------------------------------
# Top share（§6.1 头部集中度）
# ---------------------------------------------------------------------------

def top_share(values, q):
    """按用户排序后，贡献量最大的前 q 比例用户占总量的份额

    Args:
        values: 每个用户的非负数量（例如转发条数），可含 NaN——NaN
            视为"该用户此项未定义"，不计入用户数也不计入总量，而不是
            当作 0（§项目全局约定：NaN is not zero）
        q: 头部用户占比，取值 (0, 1]

    q<=0 时直接返回 0.0（"前 0 个用户的份额"就是 0，不做特殊照顾）；
    q>0 时头部用户数 k 取 ceil(q * n) 并至少为 1（q 很小时也至少纳入
    贡献最大的 1 人，否则"前一点点用户的份额"会因为四舍五入到 0 个人
    而无法计算）；q>=1 时 k 被 min 到 n，份额恒为 1.0。

    Returns:
        float，份额；若没有任何有效观测或总量为 0，返回 NaN
    """
    if q <= 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan
    total = values.sum()
    if total == 0:
        return np.nan
    k = int(np.ceil(q * n))
    k = min(max(k, 1), n)
    sorted_vals = np.sort(values)[::-1]
    return float(sorted_vals[:k].sum() / total)


# ---------------------------------------------------------------------------
# 共享结果表 schema
# ---------------------------------------------------------------------------

def tidy_result(**kwargs):
    """构造共享结果 schema 的一行，缺省的可选字段一律填 None

    这样无论调用方传了哪些字段，返回的 dict 的 key 集合永远恰好等于
    `RESULT_SCHEMA`，后续所有 models_*.parquet 才能共用同一套绘图和
    汇总代码。传入 schema 之外的字段视为调用方的笔误，直接报错，而不是
    静默忽略——静默丢弃一个字段，比报错更容易让人以为它已经写进结果表了。
    """
    unknown = set(kwargs) - set(RESULT_SCHEMA)
    if unknown:
        raise ValueError(
            f"tidy_result 收到未知字段 {sorted(unknown)}，"
            f"超出共享结果 schema {RESULT_SCHEMA} 的范围"
        )
    scale = kwargs.get("scale")
    if scale is not None and scale not in RESULT_SCALES:
        raise ValueError(
            f"tidy_result 收到未知的 scale={scale!r}，超出封闭词表 {RESULT_SCALES}。"
            "scale 决定绘图层把这一行画在什么轴上、参照线画在哪里，多一个拼写"
            "变体就多一个会被下游漏掉的分支——真的需要一种新尺度时，请先在"
            "stats_utils.RESULT_SCALES 里按分组说明加上它，并在绘图层处理它，"
            "不要在调用处就地新造一个名字。"
        )
    return {col: kwargs.get(col) for col in RESULT_SCHEMA}


# ---------------------------------------------------------------------------
# 样本流失原因的统一写法
# ---------------------------------------------------------------------------

def format_drop_reason(counts):
    """把 [(原因, 观测数), ...] 拼成 "reason=count+reason=count" 形式

    只写原因名而不写数量是不够的：读者看到
    "missing_gender+no_expressive_posts+incomplete_profile" 和一个合计
    n_dropped=3，无法知道每条规则各自丢了多少人——而"某条规则的代价有
    多大"恰恰是各模块必须让读者能自己核对的东西。因此这里逐条记数量，
    且各条数量按定义互不重叠、相加恰好等于 n_dropped（顺序归因）。

    这个格式在整套结果表里只有一处实现，就是本函数：一张
    decomposition_source_content.parquet 里同时出现两种 drop_reason 写法
    （一半带计数、一半只有名字），会让"样本流程能不能重建"这件事变成
    逐行掷骰子。models_interaction.format_drop_reason 是本函数的别名，
    保留是因为 models_temporal 与测试都按那个名字引用它。
    """
    if not counts:
        return None
    return "+".join("{}={}".format(reason, n) for reason, n in counts)


# ---------------------------------------------------------------------------
# 平均边际效应（AME）
# ---------------------------------------------------------------------------

def _design_matrices(model_result, focal_var, data):
    """构造 focal_var 恒为 1 / 恒为 0 两份反事实设计矩阵

    优先复用模型拟合时（patsy 公式接口）保存下来的 design_info 重新
    生成设计矩阵，保证列顺序、哑变量编码方式与原模型完全一致；如果
    模型不是用公式接口拟合的（没有 design_info），退化为直接从 `data`
    里按 `model_result.model.exog_names` 取列——这种情况下调用方必须
    保证 `data` 已经是与拟合时同结构的设计矩阵（含常数项列）。
    """
    data1 = data.copy()
    data1[focal_var] = 1
    data0 = data.copy()
    data0[focal_var] = 0

    design_info = getattr(model_result.model.data, "design_info", None)
    if design_info is not None:
        exog1 = patsy.dmatrix(design_info, data1, return_type="dataframe").values
        exog0 = patsy.dmatrix(design_info, data0, return_type="dataframe").values
    else:
        cols = list(model_result.model.exog_names)
        exog1 = data1[cols].values
        exog0 = data0[cols].values
    return exog1, exog0


def _ame_at_params(model_result, params, exog1, exog0):
    """给定一组参数，反事实预测两侧结果并取均值之差"""
    pred1 = model_result.model.predict(params, exog1)
    pred0 = model_result.model.predict(params, exog0)
    return float(np.mean(pred1 - pred0))


def _get_cov_params_or_none(model_result):
    """尝试取模型的参数协方差矩阵；只有在协方差"确实不可用"时才返回 None

    "不可用"严格限定为两种情况：模型对象根本没有 `cov_params` 这个
    方法，或者调用它时抛出 `NotImplementedError`（statsmodels 里部分
    拟合方式——例如某些正则化拟合——就是用这个异常表示"这次拟合没有
    协方差矩阵"，这是它们自己声明的、明确的"不可用"信号）。

    除此之外的任何异常——协方差矩阵形状与参数维度不匹配、含 NaN、
    不是数值等——都不是"协方差不可用"，而是调用方传入的模型本身有
    问题，必须原样向上抛出，绝不能在这里被吞掉后让 `average_marginal_
    effect` 悄悄退化成一个更窄但错误的区间。这正是本函数只窄窄地捕获
    `NotImplementedError`、不用 `except Exception` 的原因。
    """
    cov_params_fn = getattr(model_result, "cov_params", None)
    if cov_params_fn is None:
        return None
    try:
        cov = cov_params_fn()
    except NotImplementedError:
        return None
    return np.asarray(cov, dtype=float)


def average_marginal_effect(model_result, focal_var, data, confidence=_DEFAULT_CONFIDENCE,
                             delta_eps=1e-6):
    """二值 focal 变量的平均边际效应（AME），反事实预测法

    做法（必须是这三步，不能替换成读系数或替换成在协变量均值处求边际
    效应——三者数值不同，本研究只报告下面这一种）：
        1) 把 `data` 的 focal_var 全部设为 1，对每一行预测结果；
        2) 把 `data` 的 focal_var 全部设为 0，对每一行预测结果；
        3) 两份预测逐行相减，取平均。
    这就是 AME 的定义：对样本里每个个体分别问"如果 ta 是 1 组/0 组，
    模型预测的结果会是多少"，再把差值平均——而不是先对协变量取平均、
    只算一个"代表性个体"的边际效应（那是 MEM，在非线性模型里与 AME
    不是同一个数：模型是非线性的，"先平均协变量再算一次" 和 "先逐行算
    再平均结果" 一般不可交换），也不是直接读回归系数（系数是 log-odds
    尺度，本研究一律报告概率/比例尺度，两者不可以互相替代）。

    标准误用 delta method：把 AME 看作模型参数 beta 的函数 g(beta)，
    在 `model_result.params` 处用中心差分数值求梯度（对每个参数分量
    单独扰动 ±delta_eps*max(1,|beta_j|)），再用
    Var(AME) = grad' * cov_params() * grad 得到方差。之所以用数值梯度
    而不是针对某一种 link function 手写解析梯度，是因为本模块要同时
    服务 logit/probit/分数 logit 等多种模型，数值梯度对 link function
    不敏感，换模型不用重写这段代码。

    如果协方差矩阵确实不可用（见 `_get_cov_params_or_none` 的严格判定），
    本函数**不会**退化成任何形式的区间——之前的版本在协方差不可用时
    用"固定参数、只重抽样预测数据"的 bootstrap 顶替，那个区间完全不
    反映参数估计的不确定性（只重抽样了协变量分布，参数本身纹丝不动），
    实测比真实 delta method 区间窄一个数量级以上，这比"没有区间"更
    危险：一个缺失的区间在论文里可以如实说明原因，一个自信但错误的
    窄区间会被当真。因此这种情况下 se/ci_low/ci_high 一律返回 NaN，
    只保留可以独立算出的点估计 ame，并打印中文警告说明原因。真正需要
    在这种模型上拿到有效区间，唯一正确的路径是对估计数据重新拟合模型
    做 cluster bootstrap（即在每次重抽样上重新 `.fit()`，而不是复用
    这里已经点估计好的参数）——那需要知道具体的拟合方式（公式、
    family、cluster 结构等），因人而异，不属于本共享工具函数的职责，
    留给各个 models_*.py 在拟合模型的地方自己实现。

    Args:
        model_result: 已拟合的 statsmodels 结果对象（或提供同样接口的
            对象：`.params`、`.model.predict(params, exog)`、可选的
            `.cov_params()`）
        focal_var: `data` 中代表 focal 变量的列名，取值应为 0/1
        data: 用于反事实预测的协变量数据（不要求是拟合时用的那份数据，
            但列必须能通过模型的 design_info 或 exog_names 对齐）

    Returns:
        (ame, se, low, high)；协方差不可用时 se/low/high 为 NaN
    """
    exog1, exog0 = _design_matrices(model_result, focal_var, data)
    params = np.asarray(model_result.params, dtype=float)
    ame = _ame_at_params(model_result, params, exog1, exog0)

    cov = _get_cov_params_or_none(model_result)
    if cov is None:
        print(
            "警告: average_marginal_effect 未能取得 cov_params()"
            "（模型没有该方法，或拟合方式明确声明协方差不可用），"
            "delta method 无法计算。为避免返回一个低估不确定性的区间，"
            "se/ci_low/ci_high 一律记为 NaN，只保留点估计 ame——"
            "下游写结果表时必须把这种情况当作缺置信区间处理，"
            "不能自己用别的公式补一个数字上去。"
        )
        return ame, np.nan, np.nan, np.nan

    grad = np.zeros_like(params)
    for j in range(len(params)):
        step = delta_eps * max(1.0, abs(params[j]))
        p_plus = params.copy()
        p_plus[j] += step
        p_minus = params.copy()
        p_minus[j] -= step
        grad[j] = (
            _ame_at_params(model_result, p_plus, exog1, exog0)
            - _ame_at_params(model_result, p_minus, exog1, exog0)
        ) / (2 * step)
    # grad @ cov @ grad：cov 形状与 grad 维度不匹配时 numpy 会直接抛
    # ValueError——不捕获，原样向上抛出，这就是"形状错误必须暴露"的
    # 实现方式，不允许被更外层的 except 悄悄吞掉
    var = float(grad @ cov @ grad)
    if not np.isfinite(var) or var < 0:
        raise ValueError(
            f"delta method 得到无效方差 var={var!r}：cov_params() 返回的协方差"
            "矩阵可能含 NaN、不是半正定矩阵，这是调用方传入的模型本身有问题，"
            "必须先修好模型的协方差矩阵，不允许静默退化成别的区间来掩盖。"
        )
    se = float(np.sqrt(var))
    z = _z_value(confidence)
    low, high = ame - z * se, ame + z * se
    return ame, se, low, high
