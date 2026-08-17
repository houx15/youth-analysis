"""
§13.3 词表稳健性：结论有多依赖那一份人工定稿的词表。

本模块产出两类变体，全部通过 harness.estimate_all 估计同样的六个量：

1. **随机保留 80% 的重采样**（默认 200 次）：按"词长 × 语料频次"分层抽样，
   频次取自帖子×词矩阵实测，不是假设的；
2. **词表构成变体**：留一类别、只留 3 字及以上、剔除嵌套词、明星词表只留
   人名。

--------------------------------------------------------------------------
本模块存在的真正难点：重聚合偏差必须被测量，而不是被继承
--------------------------------------------------------------------------
按存量逐词计数重新聚合，只有在"被剔除的词没有掩盖住任何被保留的词"时才
与重扫原文相等，否则**恒为低估**（方向单向：存量里出现过的保留词一定
真实出现在正文里，所以重聚合判命中的帖子重扫必然也命中，反过来不然）。
掩盖有两种方式，见 incidence.at_risk_pairs 上方的说明：子串嵌套（公共
事务 112 词）与边界重叠（公共事务 1502 对有序词对）。**只看子串会漏掉
这一类错误里绝大部分的来源**，所以本模块一律用完整口径
（incidence.reaggregation_exposure / at_risk_terms）。

这件事的量级与 §13.3 本身想检验的效应是同一个数量级，而且方向恒定——也
就是说，一个未经测量的重聚合偏差完全可能被读成"结论对词表敏感"。所以
本模块做三件事，缺一不可：

- **逐 replicate 记录暴露面**：保留词数、处于风险中的保留词数（完整口径
  与只看子串的口径并列写出，让读者看得见后者小多少）、受影响帖子数。
  这些数字写进 `vocabulary_diagnostics.parquet`，不是只写进报告：一个
  风险词数很高的 replicate 本身就该被少信一点。
- **对子样本做重扫校准**：默认 200 次里抽 20 次（`n_calibration` 参数），
  把"重聚合判为不命中、但存量计数里含有风险剔除词"的那批帖子回到
  `cleaned_weibo_cov` 重扫原文，得到这批帖子上的真实命中判定，再与重聚合
  的估计配对写进结果表（variant_family="vocabulary_calibration"，
  variant_label 以 `_reaggregated` / `_rescanned` 结尾）。
- **一个不依赖上面那套推理的独立检验**：从**全部**"重聚合判为不命中的
  表达帖"里随机抽样重扫（`random_nonhit_probe`），报告它们实际命中的
  比例。风险集是按"我们理解的匹配失效方式"推出来的，可能仍有没想到的
  失效方式；随机抽样不看词表关系，因此能兜住风险集之外的部分，是这里
  唯一一个在"理论想错了"时还站得住的证据。

**本模块绝不用校准结果去偷偷修正主结果**：一个被测量、被报告的偏差是
一条发现，一个被悄悄调过的数字不是。

为什么只重扫那一批帖子（而不是重扫全年）：唯一可能出错的方向是"重聚合
判不命中、重扫却命中"，而在风险集之内它只可能发生在存量计数里含有风险
剔除词的帖子上。把这批帖子（`at_risk_affected_posts`）逐条重扫，其余
帖子按重聚合的结论照单全收，代价是读几个月的日文件、扫几万条正文，而不是
重扫三千多万条。**必须说清楚：这样得到的数是"在风险集之内修正过"的值，
不是精确值**——如果风险集本身漏了某类失效方式，被修正过的那一侧同样偏低，
于是配对差值只是重聚合误差的**下界**，不是上界。随机抽样检验存在的意义
正是给这个下界配上一个不依赖风险集定义的量。

--------------------------------------------------------------------------
两处刻意的"做不到就说做不到"
--------------------------------------------------------------------------
`configs/news_vocabulary_2020.txt` 与 `configs/entertainment_nouns_2020.txt`
都是逐行纯词，**没有任何类别信息**。因此：

- 留一类别（`run_leave_one_category_out`）在真实词表上无法定义，本模块
  输出一行注明原因的行，并支持调用方显式传入 `categories` 映射；
- 明星词表"只留人名 vs 人名加作品"（`run_celebrity_person_only`）同样
  无法定义。**绝不从字符串上猜哪个是人名、哪个是作品名**——猜出来的类别
  会造出一条看起来像结论、实际上是编的结果。

--------------------------------------------------------------------------
诊断为什么单独一张表，以及它怎么与结果表对上
--------------------------------------------------------------------------
`harness.append_rows` 按 `ROBUSTNESS_SCHEMA` 显式读列，往结果帧上多挂几列
诊断字段，会在下一次追加读旧文件时被静默丢掉。共享结果 schema 是整套
稳健性套件的公共契约，不该为了一个 variant family 去改它，所以诊断走
并排的一张 `vocabulary_diagnostics.parquet`。它是输出的一部分，不是报告
里的一段话。

**连接键是 (variant_family, variant_label)**，每个键下每个领域一行（因此
一个 variant 两行）。三件事必须说在前面：

1. 校准家族的两个标签（`..._reaggregated` / `..._rescanned`）各自都写了
   自己的诊断行——它们恰恰是论文最可能引用的行，不能要求读者去掉后缀才
   找得到自己的诊断；
2. 结果表里 domain="both" 的差中差行（`did_entry` / `did_topical`）**按
   设计没有单独的诊断行**：它同时依赖两个领域的词表改动，对应的是该标签
   下的两行诊断（public 与 celebrity），不是其中某一行；
3. "做不到"的那几行注明行（留一类别、明星只留人名）按设计没有诊断行——
   它们没有跑过任何词表子集，没有暴露面可记。

`append_diagnostics` 与 `harness.append_rows` 一样是纯追加、不去重：同一个
year 跑两次 `build()` 会让两张表都翻倍，这是本套件一贯的行为（重跑前先
删掉旧文件），不是这里额外给的唯一性保证。

使用方法:
    python -m gender_domain.robustness.vocabulary build --year 2020 \
        --n_replicates 200 --n_calibration 20
"""

import glob
import os
import tempfile
from collections import namedtuple

import fire
import numpy as np
import pandas as pd
from scipy import stats as sps

from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import id_rules as ir
from gender_domain import stats_utils as su
from gender_domain import text_rules as tr
from gender_domain.robustness import harness
from gender_domain.robustness import incidence as inc

DOMAINS = but.DOMAINS

# 结果表里的两个 variant family：主家族是全部变体，校准家族只放"同一个
# replicate 的重聚合 vs 精确重扫"这对配对行，让下游能一眼把它们配上，
# 又不会把主家族的 replicate 分布掺进多余的行。
VARIANT_FAMILY = "vocabulary"
CALIBRATION_FAMILY = "vocabulary_calibration"

# 默认校准多少个 replicate（简报给的默认值：200 抽 20）
DEFAULT_N_CALIBRATION = 20

# 表 C 里六个量真正用得到的列。显式列名而不是整表读入：表 C 有几十列，
# 本模块只改 topical_share，其余列原样透传给 harness。
USER_TABLE_COLUMNS = [
    "user_id",
    "gender",
    "n_posts",
    "n_retweets",
    "n_expressive_posts",
    "n_active_days",
    "n_active_months",
    "n_active_months_panel",
    "region",
    "public_topical_posts",
    "public_topical_share",
    "celebrity_topical_posts",
    "celebrity_topical_share",
    "public_source_entered",
    "celebrity_source_entered",
    "public_source_count",
    "celebrity_source_count",
]

# 精确重扫时从 cleaned_weibo_cov 日文件里读的列。命中与否只取决于清洗后
# 的正文，所以只要这两列——表达帖口径一律沿用表 A 写好的 is_expressive，
# 本模块不自己再推一遍（与 incidence 同一条纪律）。
RAW_POST_COLUMNS = ["weibo_id", "weibo_content"]

# 每个被校准的 replicate、每个领域，随机抽多少条"重聚合判不命中的表达帖"
# 做独立检验。500 条在 flip 率为 1% 时的 95% 区间约 ±1 个百分点，足够回答
# "风险集之外还有没有成规模的漏判"这个问题，而代价只是多读几千条正文。
DEFAULT_N_PROBE = 500

# 诊断表 schema。逐 (variant_family, variant_label, domain) 一行。分五组：
# 1) 身份；2) 这次剔了多少词；3) 风险集规模（完整口径 at_risk_* 与只看
# 子串的 shadowed_* 并列，让读者看得见后者小多少）；4) 风险集之内的重扫
# 校准实测偏差；5) 不依赖风险集定义的随机抽样检验。第 4、5 组在未校准的
# replicate 上全部是 NaN，而不是 0——"没测过"和"测出来是 0"必须能分开。
DIAGNOSTIC_SCHEMA = (
    "variant_family",
    "variant_label",
    "replicate",
    "seed",
    "domain",
    "n_vocab_terms",
    "n_retained_terms",
    "retained_fraction",
    # 完整风险口径（子串嵌套 + 边界重叠）
    "n_at_risk_terms",
    "at_risk_share_of_retained",
    "n_at_risk_dropped_terms",
    "n_posts_with_at_risk_term",
    "n_expressive_posts_possibly_lost",
    # 只看子串的那一部分，用来显示完整口径比它大多少
    "n_shadowed_terms",
    "n_posts_with_shadowing_term",
    "n_expressive_posts_possibly_lost_substring_only",
    "calibrated",
    "n_posts_rescanned",
    "n_expressive_posts_recovered",
    "reagg_topical_posts",
    "rescan_topical_posts",
    "mean_delta_topical_share",
    "max_abs_delta_topical_share",
    "n_users_with_delta",
    # 独立检验：从全部"重聚合判不命中的表达帖"里随机抽样重扫
    "n_nonhit_expressive_posts",
    "n_probe_sampled",
    "n_probe_flipped",
    "n_probe_flipped_outside_at_risk",
    "probe_flip_rate",
    "probe_flip_rate_ci_low",
    "probe_flip_rate_ci_high",
    "probe_implied_missed_posts",
    "note",
)

# risk_pairs：每个领域的 {可能掩盖者: {被掩盖词}}，按词表建一次（O(n^2)），
# 之后每个 replicate 只做集合运算。没有它，200 个 replicate × 2 个领域
# 会把同一份两两判断重算 400 遍。
VocabularyContext = namedtuple(
    "VocabularyContext", ["year", "user_table", "vocab", "incidence", "risk_pairs"]
)

# 校准一个 (replicate, 领域) 需要事先准备好的两批帖子：风险集之内要逐条
# 重扫的那批，以及不依赖风险集定义、随机抽出来的那批。两批的正文在
# run_resampling 里合并成一次 IO 取回。
CalibrationPlan = namedtuple(
    "CalibrationPlan", ["affected", "probe", "probe_at_risk", "n_nonhit"]
)


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def robustness_dir():
    """稳健性层唯一允许写入的目录"""
    return os.path.join(config.OUTPUT_DIR, "robustness")


def results_path():
    return os.path.join(robustness_dir(), "vocabulary.parquet")


def diagnostics_path():
    return os.path.join(robustness_dir(), "vocabulary_diagnostics.parquet")


def manifest_dir(year):
    return os.path.join(robustness_dir(), "vocabulary_{}".format(year))


# ---------------------------------------------------------------------------
# 词表加载与类别
# ---------------------------------------------------------------------------

def load_vocabulary(domain, year=config.YEAR):
    """按领域加载定稿词表，并按 VocabMatcher 的口径归一化"""
    if domain == "public":
        terms = config.load_public_vocabulary(year)
    elif domain == "celebrity":
        terms = config.load_celebrity_vocabulary(year)
    else:
        raise ValueError("未知的 domain: {}，只支持 {}".format(domain, DOMAINS))
    return inc.normalize_vocabulary(terms)


def load_vocabulary_categories(domain, year=config.YEAR):
    """词表的类别划分；两份词表文件都是逐行纯词，因此恒为 None

    保留这个函数而不是直接在调用处写死 None，是为了让"类别信息从哪里来"
    有一个明确的位置：将来词表文件真的加上类别列时，只改这里。**绝不
    在这里按字符串规则猜类别**——猜出来的"人名 / 作品名"会让一条编出来的
    划分长得和一条真实的划分一模一样。
    """
    return None


# ---------------------------------------------------------------------------
# 分层重采样
# ---------------------------------------------------------------------------

def replicate_seeds(seed, n_replicates):
    """由一个主 seed 派生出 n_replicates 个互不相同、可复现的子 seed

    用 SeedSequence 而不是 seed+i：后者在不同主 seed 之间会产生重叠的子
    seed 序列（seed=0 的第 5 个和 seed=5 的第 0 个是同一个），跨批次比较
    时那是一种看不见的重复抽样。
    """
    if n_replicates <= 0:
        return []
    states = np.random.SeedSequence(seed).generate_state(n_replicates)
    return [int(s) for s in states]


def _domain_seed(replicate_seed, domain, stream=0):
    """同一个 replicate 下、各领域各条用途各自的随机流

    两个领域的词表互不相干，必须各抽各的；但只有 replicate_seed 会被写进
    结果表（一个 replicate 一个 seed），领域 seed 由它确定性派生，因此
    结果仍然完全可复现。

    stream 把"同一个 (replicate, 领域) 下不同用途"的随机数彻底分开：
    0 是抽词表子集，1 是随机抽样检验抽帖子。两者共用同一个整数种子不会
    产生系统性偏差（population 不同、抽样形状也不同），但随机抽样检验的
    全部意义就在于"它与风险集那套推理无关"，让它和抽词共用一个种子是白白
    削弱这句话，没有任何好处。
    """
    return int(
        np.random.SeedSequence(
            [int(replicate_seed), DOMAINS.index(domain), int(stream)]
        ).generate_state(1)[0]
    )


# _domain_seed 的 stream 取值：抽词表子集 / 随机抽样检验抽帖子
_STREAM_TERMS = 0
_STREAM_PROBE = 1


def term_document_frequency(incidence, terms):
    """每个词的语料频次（含该词的帖子数），实测自帖子×词矩阵

    用"含该词的帖子数"而不是"出现总次数"：分层想控制的是"这个词有多常
    被用到"，一条帖子里刷十次同一个词不该让它跳两个频次层。矩阵里没有
    的词（全年一次没出现）频次为 0——那是一个真实的事实，不是缺失值。
    """
    counts = {}
    if incidence.matrix.shape[1] == 0:
        return {term: 0 for term in inc.normalize_vocabulary(terms)}
    binary = incidence.matrix.copy()
    binary.data = np.ones_like(binary.data)
    per_column = np.asarray(binary.sum(axis=0)).ravel()
    for term in inc.normalize_vocabulary(terms):
        col = incidence.term_index.get(term)
        counts[term] = int(per_column[col]) if col is not None else 0
    return counts


def _length_bin(term):
    """词长分层：1、2、3、4、5+（真实词表的词长绝大多数落在 2-4）"""
    return min(len(term), 5)


def term_strata(incidence, terms, n_frequency_bins=3):
    """分层键：(词长层, 频次层)，频次层按语料频次的分位数切

    频次层用分位数而不是固定阈值：真实词表的频次分布极偏（少数词占了
    绝大多数命中），固定阈值会把几乎所有词塞进同一层，分层就形同虚设。
    频次为 0 的词单独成层（bin = 0），不与真实出现过的低频词混在一起。
    """
    frequencies = term_document_frequency(incidence, terms)
    positive = np.array([f for f in frequencies.values() if f > 0], dtype=np.float64)
    if positive.size and n_frequency_bins > 1:
        quantiles = np.quantile(
            positive, np.linspace(0, 1, n_frequency_bins + 1)[1:-1]
        )
    else:
        quantiles = np.array([])

    strata = {}
    for term, freq in frequencies.items():
        if freq == 0:
            freq_bin = 0
        else:
            freq_bin = int(np.searchsorted(quantiles, freq, side="right")) + 1
        strata[term] = (_length_bin(term), freq_bin)
    return strata


def stratified_resample(terms, keep=0.8, strata=None, seed=0):
    """按分层保留 keep 比例的词，返回排序后的保留词列表

    Args:
        terms: 完整词表
        keep: 保留比例
        strata: {词: 分层键}；缺省时只按词长分层（频次要靠矩阵实测，
            调用方应当用 term_strata 先算好再传进来）
        seed: 这一次抽样的随机种子

    每一层各自取 round(keep * 该层词数) 个词，因此每一层的保留比例都等于
    keep（而不是"整体上等于 keep、某些层被整批留下"）。层内用一次
    permutation 取前 k 个，层间按分层键排序遍历，保证同一个 seed 逐词
    可复现。
    """
    cleaned = inc.normalize_vocabulary(terms)
    if not 0.0 <= keep <= 1.0:
        raise ValueError("keep 必须落在 [0, 1]，收到 {}".format(keep))
    rng = np.random.default_rng(seed)

    grouped = {}
    for term in cleaned:
        key = strata.get(term, (_length_bin(term),)) if strata else (_length_bin(term),)
        grouped.setdefault(key, []).append(term)

    retained = []
    for key in sorted(grouped, key=lambda k: tuple(str(x) for x in k)):
        group = sorted(grouped[key])
        n_keep = int(round(keep * len(group)))
        if n_keep <= 0:
            continue
        order = rng.permutation(len(group))
        retained.extend(group[i] for i in order[:n_keep])
    return sorted(retained)


def drop_short_terms(terms, min_len=3):
    """只保留长度 >= min_len 的词

    真实词表上 min_len=3 会剔掉公共事务 816 词里的 337 个、明星 535 词里
    的 182 个——三分之一到一半，是一次很大的干预，不是脚注。保留比例因此
    必须写进诊断表（run_drop_short_terms 会写）。
    """
    return sorted(t for t in inc.normalize_vocabulary(terms) if len(t) >= min_len)


def drop_nested_terms(terms):
    """剔除"是另一个词的子串"的那些词

    这只是重聚合偏差的一个来源，不是唯一来源：边界重叠（见
    incidence.at_risk_pairs）在真实词表上比嵌套多一个数量级，剔完嵌套词
    之后它仍然在。
    """
    nested = inc.nested_terms(terms)
    return sorted(t for t in inc.normalize_vocabulary(terms) if t not in nested)


# ---------------------------------------------------------------------------
# 上下文：矩阵和表 C 建一次，所有变体复用
# ---------------------------------------------------------------------------

def build_context(year=config.YEAR, domains=DOMAINS):
    """加载表 C、两份词表，并为每个领域建一次帖子×词矩阵"""
    user_path = os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year))
    if not os.path.exists(user_path):
        raise FileNotFoundError("未找到表 C: {}".format(user_path))
    user_table = pd.read_parquet(user_path, columns=USER_TABLE_COLUMNS)
    user_table["user_id"] = ir.normalize_id_series(user_table["user_id"])
    print("表 C {:,} 个用户".format(len(user_table)))

    vocab = {}
    incidences = {}
    risk_pairs = {}
    for domain in domains:
        vocab[domain] = load_vocabulary(domain, year)
        incidences[domain] = inc.build_post_term_incidence(year, domain)
        risk_pairs[domain] = inc.at_risk_pairs(vocab[domain])
        n_pairs = sum(len(v) for v in risk_pairs[domain].values())
        print(
            "{} 词表 {} 个词：子串嵌套词 {} 个，可能互相掩盖的有序词对 {} 对".format(
                domain, len(vocab[domain]), len(inc.nested_terms(vocab[domain])), n_pairs
            )
        )
    return VocabularyContext(
        year=year, user_table=user_table, vocab=vocab, incidence=incidences,
        risk_pairs=risk_pairs,
    )


def user_frame_with_topical(user_table, topical_by_domain):
    """把重算出来的逐用户 topical 指标贴回表 C，其余列原样保留

    只覆盖 {domain}_topical_posts / {domain}_topical_share 两列：词表变体
    不改样本、不改分母口径、更不改任何一个量的定义，它只改"内容参与是
    怎么测出来的"。
    """
    frame = user_table.copy()
    for domain, measures in topical_by_domain.items():
        merged = frame[["user_id"]].merge(
            measures[["user_id", "topical_posts", "topical_share"]],
            on="user_id", how="left",
        )
        frame["{}_topical_posts".format(domain)] = merged["topical_posts"].to_numpy()
        frame["{}_topical_share".format(domain)] = merged["topical_share"].to_numpy()
    return frame


def topical_for_subset(context, domain, term_subset):
    """某个领域、某个词表子集下的逐用户 topical 指标（存量重聚合口径）"""
    incidence = context.incidence[domain]
    return inc.topical_by_user(incidence, term_subset, incidence.posts)


# ---------------------------------------------------------------------------
# 重扫校准（风险集之内）与随机抽样检验（不依赖风险集）
# ---------------------------------------------------------------------------

_EMPTY_POSTS = ("weibo_id", "user_id", "month")


def _empty_post_frame():
    return pd.DataFrame({
        "weibo_id": pd.Series(dtype="object"),
        "user_id": pd.Series(dtype="object"),
        "month": pd.Series(dtype="int64"),
    })


def _post_frame(posts, mask):
    """按逐帖布尔掩码取出 weibo_id / user_id / month 三列"""
    if not mask.any():
        return _empty_post_frame()
    out = posts.loc[mask, list(_EMPTY_POSTS)].copy()
    out["weibo_id"] = out["weibo_id"].astype(str)
    out["user_id"] = out["user_id"].astype(str)
    out["month"] = out["month"].astype("int64")
    return out.reset_index(drop=True)


def _reaggregated_hit_mask(incidence, term_subset):
    """逐帖：本子集下按存量重聚合判定是否命中"""
    if incidence.matrix.shape[1] == 0:
        return np.zeros(len(incidence.posts), dtype=bool)
    keep = inc.term_subset_vector(incidence, term_subset, warn_unrecognized=False)
    return inc._rows_to_posts(incidence.posts, incidence.matrix.dot(keep) > 0)


def at_risk_affected_posts(incidence, term_subset, vocab, pairs=None):
    """重聚合**可能**判错的那批帖子（重扫只需要读这些）

    条件三条同时成立：
    1) 是表达帖（非表达帖不进任何内容指标的分子分母）；
    2) 在本子集下按存量重聚合判定为**不命中**；
    3) 存量计数里含有至少一个"可能掩盖住某个保留词"的被剔除词——完整口径
       （子串嵌套 **加上** 边界重叠，见 incidence.at_risk_pairs），不是
       只看子串的那一小半。

    第 2 条是重聚合唯一可能出错的方向（重聚合判命中的帖子，其保留词必然
    真实出现在正文里，重扫不可能翻案）。第 3 条把范围收到"按我们理解的
    匹配失效方式，有可能翻案"的那些帖子上——**这一条是一个理论假设，不是
    事实**：如果还有没想到的失效方式，符合第 1、2 条却不符合第 3 条的帖子
    同样可能翻案，而本函数不会把它们交出去。random_nonhit_probe 存在的
    意义正是去测这一块。

    Returns:
        DataFrame，列为 weibo_id / user_id / month。行数恰好等于
        incidence.reaggregation_exposure 报的 n_expressive_posts_possibly_lost。
    """
    posts = incidence.posts
    dropped = inc.at_risk_dropped_terms(vocab, term_subset, pairs=pairs)
    if not dropped or incidence.matrix.shape[1] == 0:
        return _empty_post_frame()

    risk_vec = np.zeros(incidence.matrix.shape[1], dtype=np.int32)
    for term in dropped:
        col = incidence.term_index.get(term)
        if col is not None:
            risk_vec[col] = 1

    post_risk = inc._rows_to_posts(posts, incidence.matrix.dot(risk_vec) > 0)
    post_hit = _reaggregated_hit_mask(incidence, term_subset)
    expressive = posts["is_expressive"].to_numpy(dtype=bool)
    return _post_frame(posts, post_risk & (~post_hit) & expressive)


def sample_nonhit_posts(incidence, term_subset, n_probe=DEFAULT_N_PROBE, seed=0):
    """从**全部**"重聚合判不命中的表达帖"里等概率随机抽 n_probe 条

    与 at_risk_affected_posts 的区别是关键：那一个只交出"按理论会翻案"的
    帖子，这一个完全不看词表关系。因此在这批帖子上量到的翻案率，是对
    "重聚合到底漏判了多少"的**无偏**估计，哪怕我们对匹配失效方式的理解
    是错的、漏的。

    Returns:
        (抽样帧, 该子集下不命中表达帖的总数)
    """
    posts = incidence.posts
    post_hit = _reaggregated_hit_mask(incidence, term_subset)
    expressive = posts["is_expressive"].to_numpy(dtype=bool)
    candidates = np.flatnonzero((~post_hit) & expressive)
    n_total = int(candidates.size)
    if n_total == 0 or n_probe <= 0:
        return _empty_post_frame(), n_total

    rng = np.random.default_rng(seed)
    take = min(int(n_probe), n_total)
    picked = rng.choice(candidates, size=take, replace=False)
    picked.sort()
    mask = np.zeros(len(posts), dtype=bool)
    mask[picked] = True
    return _post_frame(posts, mask), n_total


def _clopper_pearson(n_success, n_total, confidence=0.95):
    """二项比例的 Clopper-Pearson 精确区间

    抽样检验的意义全在区间上：抽 500 条一条都没翻案，结论不是"漏判率是
    0"，而是"漏判率的 95% 上限约 0.7%"。只报点估计会把后者说成前者。
    """
    if n_total <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    low = 0.0 if n_success == 0 else float(
        sps.beta.ppf(alpha / 2, n_success, n_total - n_success + 1)
    )
    high = 1.0 if n_success == n_total else float(
        sps.beta.ppf(1 - alpha / 2, n_success + 1, n_total - n_success)
    )
    return low, high


def random_nonhit_probe(context, domain, term_subset, plan=None, text_cache=None,
                        n_probe=DEFAULT_N_PROBE, seed=0):
    """独立检验：随机抽的"不命中表达帖"里，实际有多大比例其实命中

    这是本模块唯一一个**不依赖"风险集定义对不对"**的证据。返回的
    probe_implied_missed_posts 是按抽样翻案率推回全体的漏判帖数，可以直接
    与 n_expressive_posts_recovered（风险集之内实际找回的帖数）比较：后者
    明显小于前者，就说明风险集漏掉了成规模的失效方式。
    """
    incidence = context.incidence[domain]
    if plan is None:
        plan = plan_calibration(context, domain, term_subset, n_probe=n_probe, seed=seed)
    probe, n_nonhit = plan.probe, plan.n_nonhit
    if len(probe) == 0:
        return {
            "n_nonhit_expressive_posts": int(n_nonhit),
            "n_probe_sampled": 0, "n_probe_flipped": 0,
            "n_probe_flipped_outside_at_risk": 0,
            "probe_flip_rate": np.nan,
            "probe_flip_rate_ci_low": np.nan, "probe_flip_rate_ci_high": np.nan,
            "probe_implied_missed_posts": np.nan,
        }

    hits = rescan_hits(context.year, probe, term_subset, text_cache=text_cache)
    flipped = hits["hit"].to_numpy(dtype=bool)
    outside = ~np.isin(hits["weibo_id"].to_numpy(), plan.probe_at_risk)
    n_flipped = int(flipped.sum())
    n_sampled = int(len(hits))
    rate = n_flipped / n_sampled
    low, high = _clopper_pearson(n_flipped, n_sampled)
    print(
        "{} 随机抽样检验: 抽 {} 条不命中表达帖，实际命中 {} 条"
        "（其中风险集之外 {} 条），翻案率 {:.3%}（95% CI {:.3%}-{:.3%}）".format(
            domain, n_sampled, n_flipped, int((flipped & outside).sum()),
            rate, low, high,
        )
    )
    return {
        "n_nonhit_expressive_posts": int(n_nonhit),
        "n_probe_sampled": n_sampled,
        "n_probe_flipped": n_flipped,
        "n_probe_flipped_outside_at_risk": int((flipped & outside).sum()),
        "probe_flip_rate": float(rate),
        "probe_flip_rate_ci_low": float(low),
        "probe_flip_rate_ci_high": float(high),
        "probe_implied_missed_posts": float(rate * n_nonhit),
    }


def plan_calibration(context, domain, term_subset, n_probe=DEFAULT_N_PROBE, seed=0):
    """一次校准需要读回正文的两批帖子（风险集之内 + 随机抽样），纯矩阵运算"""
    incidence = context.incidence[domain]
    affected = at_risk_affected_posts(
        incidence, term_subset, context.vocab[domain],
        pairs=context.risk_pairs.get(domain),
    )
    probe, n_nonhit = sample_nonhit_posts(
        incidence, term_subset, n_probe=n_probe, seed=seed
    )
    at_risk_ids = set(affected["weibo_id"]) if len(affected) else set()
    probe_at_risk = np.array(
        [wid for wid in probe["weibo_id"]] if len(probe) else [], dtype=object
    )
    probe_at_risk = np.array(
        [wid for wid in probe_at_risk if wid in at_risk_ids], dtype=object
    )
    return CalibrationPlan(
        affected=affected, probe=probe, probe_at_risk=probe_at_risk,
        n_nonhit=n_nonhit,
    )


def load_post_texts(year, affected):
    """把这批帖子的正文从 cleaned_weibo_cov 读回来，清洗后放进 {weibo_id: 文本}

    只读受影响帖子所在的那几个月的日文件、只读 weibo_id 与正文两列
    （表 A 不存正文，`posts.weibo_id` 就是为了这一步能回溯到原帖）。

    **这一步是整个校准里唯一昂贵的 IO，因此必须只做一次。** 20 个被校准
    的 replicate 各自去扫一遍全年日文件，是同一份正文被读 20 遍；调用方
    （run_resampling）先把 20 次的受影响帖子并集算出来，一次读完存进这份
    字典，之后每个 replicate 只在内存里换一个 VocabMatcher 重新判定。
    并集本身是纯矩阵运算，不需要读任何正文。

    找不到的 weibo_id 直接报错，不当作"没命中"：静默按不命中处理会让
    校准结果朝"重聚合没有偏差"的方向偏，那恰恰是这一步要检验的东西。
    """
    if len(affected) == 0:
        return {}

    wanted = set(affected["weibo_id"].astype(str))
    months = sorted({int(m) for m in affected["month"].unique()})
    print("精确重扫取原文: {} 条帖子，涉及 {} 个月".format(len(wanted), len(months)))

    texts = {}
    for month in months:
        pattern = os.path.join(
            config.DATA_DIR, str(year), "{}-{:02d}-*.parquet".format(year, month)
        )
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError("精确重扫未找到原始日文件: {}".format(pattern))
        for path in files:
            frame = pd.read_parquet(path, columns=RAW_POST_COLUMNS)
            frame["weibo_id"] = frame["weibo_id"].astype(str)
            frame = frame[frame["weibo_id"].isin(wanted)]
            for weibo_id, content in zip(frame["weibo_id"], frame["weibo_content"]):
                # 同一个 weibo_id 在原始层重复出现时保留先读到的那一条，
                # 与 build_post_table.process_frame 的 keep="first" 一致
                texts.setdefault(weibo_id, tr.clean_text(content))

    missing = wanted - set(texts)
    if missing:
        raise ValueError(
            "精确重扫有 {} 条帖子在原始层找不到（示例 {}）。"
            "把它们当作不命中会让校准朝'重聚合没有偏差'的方向偏，"
            "所以这里直接报错，而不是静默跳过。".format(
                len(missing), sorted(missing)[:5]
            )
        )
    return texts


def rescan_hits(year, affected, term_subset, text_cache=None):
    """精确重扫：这批帖子在该词表子集下到底命中没有

    Args:
        text_cache: load_post_texts 预取好的 {weibo_id: 清洗后正文}。
            为 None 时本函数自己去读（单次校准的路径）；批量校准时由
            调用方预取一次全部并集，避免同一份日文件被读很多遍。
    """
    columns = ["weibo_id", "hit"]
    if len(affected) == 0:
        return pd.DataFrame({"weibo_id": pd.Series(dtype="object"),
                             "hit": pd.Series(dtype=bool)}, columns=columns)

    weibo_ids = affected["weibo_id"].astype(str).to_numpy()
    if text_cache is None:
        text_cache = load_post_texts(year, affected)
    missing = [wid for wid in weibo_ids if wid not in text_cache]
    if missing:
        raise ValueError(
            "精确重扫的正文缓存里缺 {} 条帖子（示例 {}），"
            "缓存与受影响帖子集合已经对不上".format(len(missing), missing[:5])
        )

    matcher = tr.VocabMatcher(term_subset)
    hits = [
        tr.measure_text(text_cache[wid], matcher)["hit"] for wid in weibo_ids
    ]
    return pd.DataFrame(
        {"weibo_id": weibo_ids, "hit": np.array(hits, dtype=bool)}, columns=columns
    )


def exposure_pair(context, domain, term_subset):
    """(完整口径, 只看子串) 两份暴露面，一个 (replicate, 领域) 只算一次

    每算一份暴露面就是两次稀疏矩阵-向量乘法，在真实数据上是两次三千多万行
    的整表扫描。同一个 (replicate, 领域) 的同一个词表子集会被
    calibrate_topical 与三次 diagnostics_row（基础标签 + 两个校准标签）
    重复用到，各自现算就是十来遍白扫。所以算一次装进这个 dict，往下传。
    """
    vocab = context.vocab[domain]
    retained = inc.normalize_vocabulary(term_subset)
    return {
        "full": inc.reaggregation_exposure(
            context.incidence[domain], retained, vocab,
            pairs=context.risk_pairs.get(domain),
        ),
        "narrow": inc.shadowing_exposure(context.incidence[domain], retained, vocab),
    }


def calibrate_topical(context, domain, term_subset, plan=None, text_cache=None,
                      n_probe=DEFAULT_N_PROBE, probe_seed=0, exposure=None):
    """对一个词表子集做重扫校准（风险集之内）加一次随机抽样检验

    Args:
        exposure: exposure_pair 的返回值；缺省时自己算（单次调用的路径）。

    Returns:
        (reagg, corrected, stats)
        - reagg: 存量重聚合的逐用户 topical 指标
        - corrected: **在风险集之内修正过**的口径（重聚合 + 重扫翻案的
          那批帖子）。刻意不叫"精确值"：它只修正了 at_risk_affected_posts
          交出来的那批帖子，如果风险集本身漏了某类失效方式，这一侧同样
          偏低，于是 stats 里的 delta 是重聚合误差的**下界**而不是上界。
        - stats: 实测偏差 + 随机抽样检验。delta 一律定义为
          **重聚合 - 修正值**，因此恒 <= 0（低估）；一条都没翻案时是 0
          而不是 NaN。
    """
    vocab = context.vocab[domain]
    reagg = topical_for_subset(context, domain, term_subset)
    if plan is None:
        plan = plan_calibration(
            context, domain, term_subset, n_probe=n_probe, seed=probe_seed
        )
    affected = plan.affected

    # 一致性检查：受影响帖子集合与暴露面必须同一口径。注意它只能发现
    # "本模块与 incidence 的两处实现分叉了"，**发现不了"两处用的是同一个
    # 不完整的定义"**——那件事只有随机抽样检验才看得见。
    if exposure is None:
        exposure = exposure_pair(context, domain, term_subset)
    full_exposure = exposure["full"]
    if len(affected) != full_exposure["n_expressive_posts_possibly_lost"]:
        raise ValueError(
            "受影响帖子数 {} 与 reaggregation_exposure 报的 {} 不一致，"
            "两处口径已经分叉，校准结果不可信".format(
                len(affected), full_exposure["n_expressive_posts_possibly_lost"]
            )
        )

    hits = rescan_hits(context.year, affected, term_subset, text_cache=text_cache)
    recovered = affected.merge(hits, on="weibo_id", how="left")
    recovered["hit"] = recovered["hit"].fillna(False).astype(bool)
    recovered = recovered[recovered["hit"]]

    corrected = reagg.copy()
    if len(recovered):
        delta = (
            recovered.groupby("user_id").size().rename("recovered").reset_index()
        )
        merged = corrected[["user_id"]].merge(delta, on="user_id", how="left")
        corrected["topical_posts"] = (
            corrected["topical_posts"].to_numpy(dtype=np.int64)
            + merged["recovered"].fillna(0).to_numpy(dtype=np.int64)
        )
        # 份额的除法只能有一个来源：与 incidence.topical_by_user 一样复用
        # 主流水线的 _safe_divide，零分母必须是 NaN 而不是 0
        corrected["topical_share"] = but._safe_divide(
            corrected["topical_posts"], corrected["n_expressive_posts"]
        ).to_numpy(dtype=np.float64)

    diff = reagg["topical_share"].to_numpy(dtype=np.float64) - corrected[
        "topical_share"
    ].to_numpy(dtype=np.float64)
    finite = diff[np.isfinite(diff)]
    stats = {
        "n_expressive_posts_possibly_lost": int(
            full_exposure["n_expressive_posts_possibly_lost"]
        ),
        "n_posts_rescanned": int(len(affected)),
        "n_expressive_posts_recovered": int(len(recovered)),
        "reagg_topical_posts": int(reagg["topical_posts"].sum()),
        "rescan_topical_posts": int(corrected["topical_posts"].sum()),
        "mean_delta_topical_share": float(finite.mean()) if finite.size else 0.0,
        "max_abs_delta_topical_share": (
            float(np.abs(finite).max()) if finite.size else 0.0
        ),
        "n_users_with_delta": int((finite != 0).sum()),
    }
    print(
        "{} 校准: 风险集内重扫 {} 条，翻案 {} 条，"
        "逐用户 topical_share 平均偏差 {:.4f}，最差 {:.4f}".format(
            domain, stats["n_posts_rescanned"], stats["n_expressive_posts_recovered"],
            stats["mean_delta_topical_share"], stats["max_abs_delta_topical_share"],
        )
    )
    # 独立检验与上面的修正是两件事：它只测量、不参与修正 corrected，
    # 因为抽样只能给出比例，落不到具体是哪几条帖子上。
    stats.update(random_nonhit_probe(
        context, domain, term_subset, plan=plan, text_cache=text_cache
    ))
    return reagg, corrected, stats


# ---------------------------------------------------------------------------
# 诊断行
# ---------------------------------------------------------------------------

def diagnostics_row(context, domain, term_subset, variant_label, replicate=0,
                    seed=None, variant_family=VARIANT_FAMILY, calibration=None,
                    note=None, exposure=None):
    """一个 (变体, 领域) 的诊断行

    这一行回答三件事：这次剔了多少词；风险集有多大（完整口径与只看子串的
    口径并列）；以及校准过的话，风险集之内实测错了多少、风险集之外的随机
    抽样又看到了什么。未校准的 replicate 的后两组字段一律是 NaN，不是 0——
    "没测过"与"测出来是 0"必须能分开。

    exposure 传 exposure_pair 的返回值可以省掉两次整表扫描；同一个词表
    子集要写好几行诊断（基础标签 + 两个校准标签）时必须传，否则每行都
    重扫一遍矩阵。
    """
    vocab = context.vocab[domain]
    retained = inc.normalize_vocabulary(term_subset)
    if exposure is None:
        exposure = exposure_pair(context, domain, term_subset)
    full = exposure["full"]
    narrow = exposure["narrow"]
    row = {
        "variant_family": variant_family,
        "variant_label": variant_label,
        "replicate": int(replicate),
        "seed": float(seed) if seed is not None else np.nan,
        "domain": domain,
        "n_vocab_terms": len(vocab),
        "n_retained_terms": len(retained),
        "retained_fraction": len(retained) / len(vocab) if vocab else np.nan,
        "n_at_risk_terms": int(full["n_shadowed_terms"]),
        "at_risk_share_of_retained": float(full["shadowed_share_of_retained"]),
        "n_at_risk_dropped_terms": int(full["n_shadowing_dropped_terms"]),
        "n_posts_with_at_risk_term": int(full["n_posts_with_shadowing_term"]),
        "n_expressive_posts_possibly_lost": int(
            full["n_expressive_posts_possibly_lost"]
        ),
        "n_shadowed_terms": int(narrow["n_shadowed_terms"]),
        "n_posts_with_shadowing_term": int(narrow["n_posts_with_shadowing_term"]),
        "n_expressive_posts_possibly_lost_substring_only": int(
            narrow["n_expressive_posts_possibly_lost"]
        ),
        "calibrated": calibration is not None,
        "note": note,
    }
    for key in _CALIBRATION_FIELDS:
        row[key] = np.nan
    if calibration is not None:
        for key in _CALIBRATION_FIELDS:
            value = calibration.get(key, np.nan)
            row[key] = float(value) if value is not None else np.nan
    return row


# 诊断表里"只有校准过才有值"的那些列
_CALIBRATION_FIELDS = (
    "n_posts_rescanned",
    "n_expressive_posts_recovered",
    "reagg_topical_posts",
    "rescan_topical_posts",
    "mean_delta_topical_share",
    "max_abs_delta_topical_share",
    "n_users_with_delta",
    "n_nonhit_expressive_posts",
    "n_probe_sampled",
    "n_probe_flipped",
    "n_probe_flipped_outside_at_risk",
    "probe_flip_rate",
    "probe_flip_rate_ci_low",
    "probe_flip_rate_ci_high",
    "probe_implied_missed_posts",
)


def append_diagnostics(rows, path):
    """诊断行增量落盘，与 harness.append_rows 同一套原子改名做法

    单独实现是因为它的 schema 不是 ROBUSTNESS_SCHEMA：harness.append_rows
    读旧文件时显式点名共享 schema 的列，用它写诊断表会把诊断列全丢掉。

    与 harness.append_rows 一样**不去重**：同一个 year 跑第二次 build()
    会让表翻倍，(variant_family, variant_label, domain) 也就不再唯一。
    这是本套件一贯的行为（重跑前先删旧文件），不是这里额外保证的不变量。
    """
    if path is None:
        return None
    frame = pd.DataFrame(list(rows), columns=list(DIAGNOSTIC_SCHEMA))
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path, engine="pyarrow", columns=list(DIAGNOSTIC_SCHEMA))
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.reset_index(drop=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".append_diag_tmp_", suffix=".parquet", dir=out_dir
    )
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print("已增量写入诊断: {}（新增 {} 行，累计 {} 行）".format(
        path, len(frame), len(combined)))
    return path


# ---------------------------------------------------------------------------
# 估计与落盘的公共动作
# ---------------------------------------------------------------------------

def _estimate_and_append(user_df, variant_label, replicate, seed, out_path,
                         variant_family=VARIANT_FAMILY):
    """跑一次六个量并立刻落盘，返回这一批行"""
    rows = harness.estimate_all(
        user_df, variant_family=variant_family, variant_label=variant_label,
        replicate=replicate, seed=seed,
    )
    if out_path is not None:
        harness.append_rows(rows, out_path)
    return rows


def _note_only_rows(variant_label, note, outcome, domain, out_path=None,
                    variant_family=VARIANT_FAMILY):
    """做不到的变体：留一行注明原因，而不是编一个结果，也不是整个消失

    行仍然走 su.tidy_result + harness.attach_variant_identity，列集合与
    其它结果行完全一致——下游按同一套 schema 读得到它，才可能在汇总时
    看见"这个变体没能做"。
    """
    row = su.tidy_result(
        outcome=outcome, domain=domain, model=None, term=None,
        estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, scale=None,
        n_obs=0, n_dropped=0, drop_reason=None, note=note,
    )
    frame = pd.DataFrame(
        [harness.attach_variant_identity(row, variant_family, variant_label, 0, None)],
        columns=list(harness.ROBUSTNESS_SCHEMA),
    )
    print("变体 {} 无法定义: {}".format(variant_label, note))
    if out_path is not None:
        harness.append_rows(frame, out_path)
    return frame


def _annotate_note(frame, text):
    """在结果行的 note 上追加一段说明，不覆盖已有内容

    确定性的词表构成变体（例如只留 3 字以上的词）必须在结果行上自己说清
    "剔的是什么、剩下多少"，否则读者拿到 vocabulary.parquet 一张表时，
    `drop_short_terms_min3` 这个标签背后是剔了 5 个词还是 337 个词完全看
    不出来。已有的 note 是拟合失败的留痕，绝不能被覆盖，所以是追加。
    """
    if not text:
        return frame
    frame = frame.copy()
    frame["note"] = [
        text if existing is None or (isinstance(existing, float) and existing != existing)
        else "{};{}".format(existing, text)
        for existing in frame["note"]
    ]
    return frame


def _subset_note(context, subsets, prefix=None):
    """"这次每个领域保留了多少词"的紧凑说明，供 _annotate_note 用"""
    parts = [] if not prefix else [prefix]
    for domain in sorted(subsets):
        retained = len(inc.normalize_vocabulary(subsets[domain]))
        total = len(context.vocab[domain])
        parts.append("{}_retained={}/{}({:.1%})".format(
            domain, retained, total, retained / total if total else float("nan")))
    return ";".join(parts)


def _run_subset_variant(context, subsets, variant_label, out_path, diag_path,
                        replicate=0, seed=None, note=None):
    """一个确定性词表变体：两个领域各换一份词表，估一次六个量，写两行诊断"""
    topical = {
        domain: topical_for_subset(context, domain, subset)
        for domain, subset in subsets.items()
    }
    user_df = user_frame_with_topical(context.user_table, topical)
    rows = harness.estimate_all(
        user_df, variant_family=VARIANT_FAMILY, variant_label=variant_label,
        replicate=replicate, seed=seed,
    )
    rows = _annotate_note(rows, _subset_note(context, subsets, prefix=note))
    if out_path is not None:
        harness.append_rows(rows, out_path)
    append_diagnostics(
        [
            diagnostics_row(context, domain, subset, variant_label,
                            replicate=replicate, seed=seed, note=note)
            for domain, subset in subsets.items()
        ],
        diag_path,
    )
    return rows


# ---------------------------------------------------------------------------
# 变体一：随机保留 keep 比例的重采样（含精确重扫校准）
# ---------------------------------------------------------------------------

def calibration_replicates(n_replicates, n_calibration):
    """挑哪些 replicate 做精确重扫校准：在 0..n-1 上等距取点

    等距而不是取前 n_calibration 个：重采样的 replicate 之间本来就是独立
    同分布的，但等距取点在"作业被中途杀掉"时仍然能保证已完成的部分里有
    校准样本，取前几个则会让校准全挤在最前面。
    """
    if n_calibration <= 0 or n_replicates <= 0:
        return set()
    n = min(int(n_calibration), int(n_replicates))
    picks = np.unique(np.linspace(0, n_replicates - 1, n).round().astype(int))
    return {int(p) for p in picks}


def _prepare_calibration(context, subsets_by_replicate, calibrate, seeds,
                         n_probe=DEFAULT_N_PROBE):
    """为全部被校准的 replicate 备好两批帖子，并一次把正文读回来

    两批帖子（风险集之内要重扫的、随机抽样检验要重扫的）都只靠矩阵运算
    得到，不需要读任何正文；把它们的并集一次读完，全年日文件就只被扫一遍。
    每个被校准的 replicate 各扫一遍，在真实数据上是几个小时的纯 IO。

    calibrate 为空时返回 ({}, None)：完全不碰原始层，没有原始数据的机器
    照样能跑重采样。
    """
    if not calibrate:
        return {}, None
    plans = {}
    frames = []
    for replicate in sorted(calibrate):
        plans[replicate] = {}
        for domain, subset in subsets_by_replicate[replicate].items():
            # 抽样种子挂在 replicate 种子上：同一次运行可复现，不同
            # replicate 抽到的又不是同一批帖子。**stream 必须与抽词的那条
            # 分开**：随机抽样检验的意义就是独立于词表子集是怎么抽出来的。
            plan = plan_calibration(
                context, domain, subset, n_probe=n_probe,
                seed=_domain_seed(seeds[replicate], domain, stream=_STREAM_PROBE),
            )
            plans[replicate][domain] = plan
            frames.extend([plan.affected, plan.probe])
    union = pd.concat(frames, ignore_index=True) if frames else _empty_post_frame()
    if len(union):
        union = union.drop_duplicates(subset=["weibo_id"], keep="first")
    print("{} 个被校准的 replicate 合计需要原文 {:,} 条（并集），一次读回".format(
        len(calibrate), len(union)))
    return plans, load_post_texts(context.year, union)


def run_resampling(year=config.YEAR, n_replicates=200, keep=0.8, seed=0,
                   n_calibration=DEFAULT_N_CALIBRATION, n_probe=DEFAULT_N_PROBE,
                   context=None, out_path=None, diag_path=None):
    """随机保留 keep 比例的词表重采样，逐 replicate 落盘

    每个 replicate：两个领域各抽一份词表子集 -> 重聚合逐用户 topical ->
    估六个量 -> 立刻落盘 -> 写两行诊断。被抽中校准的 replicate 额外做一次
    风险集内重扫（成对写进 CALIBRATION_FAMILY）和一次随机抽样检验
    （只写进诊断表）。

    **偏差不是一个常数。** 它随这个 replicate 剔掉了多少"可能掩盖保留词"
    的词而变（实测：k=0/5/15/80/167 个容器词对应 0 / -0.0010 / -0.0040 /
    -0.0093 / -0.0233），因此 keep 越低偏差越大、离散度也越大。所以诊断表
    逐 replicate 记 n_at_risk_terms，让下游能把偏差与它对上，而不是套用
    某一次运行里的一个平均数。
    """
    context = context or build_context(year)
    seeds = replicate_seeds(seed, n_replicates)
    calibrate = calibration_replicates(n_replicates, n_calibration)
    strata = {
        domain: term_strata(context.incidence[domain], context.vocab[domain])
        for domain in context.vocab
    }
    print(
        "词表重采样: {} 次，保留 {:.0%}，其中 {} 次做精确重扫校准（replicate {}）".format(
            n_replicates, keep, len(calibrate), sorted(calibrate)
        )
    )

    # 全部 replicate 的词表子集先一次性抽好（纯随机数运算，代价可以忽略）：
    # 这样被校准的那些 replicate 的"受影响帖子并集"可以在读任何正文之前
    # 就算出来，全年日文件只需要扫一遍，而不是每个被校准的 replicate 扫
    # 一遍（20 次校准就是 20 遍全年正文，那是几个小时的纯 IO）。
    subsets_by_replicate = [
        {
            domain: stratified_resample(
                context.vocab[domain], keep=keep, strata=strata[domain],
                seed=_domain_seed(seeds[replicate], domain, stream=_STREAM_TERMS),
            )
            for domain in context.vocab
        }
        for replicate in range(n_replicates)
    ]
    plans, text_cache = _prepare_calibration(
        context, subsets_by_replicate, calibrate, seeds, n_probe=n_probe
    )

    collected = []
    for replicate in range(n_replicates):
        replicate_seed = seeds[replicate]
        label = "keep{:g}_rep{}".format(keep, replicate)
        subsets = subsets_by_replicate[replicate]
        topical = {
            domain: topical_for_subset(context, domain, subset)
            for domain, subset in subsets.items()
        }
        user_df = user_frame_with_topical(context.user_table, topical)
        rows = _estimate_and_append(
            user_df, label, replicate, replicate_seed, out_path
        )
        collected.append(rows)

        # 暴露面一个 (replicate, 领域) 只算一次：下面校准一次、诊断最多三次
        # 都用同一份，各自现算就是十来遍整表扫描
        exposures = {
            domain: exposure_pair(context, domain, subsets[domain])
            for domain in subsets
        }

        calibration_stats = {}
        diagnostic_rows = []
        if replicate in calibrate:
            corrected = {}
            for domain, subset in subsets.items():
                _, corrected_domain, stats = calibrate_topical(
                    context, domain, subset, plan=plans[replicate][domain],
                    text_cache=text_cache, exposure=exposures[domain],
                )
                corrected[domain] = corrected_domain
                calibration_stats[domain] = stats
            corrected_df = user_frame_with_topical(context.user_table, corrected)
            # 重聚合那一半直接复用刚才那次拟合的结果，只换身份标签——不是
            # 重新拟合一遍：配对比较要求两边除了"测量是怎么来的"以外
            # 完全相同，重跑一次反而多引入一次拟合的不确定性。
            paired_reagg = rows.copy()
            paired_reagg["variant_family"] = CALIBRATION_FAMILY
            paired_reagg["variant_label"] = label + "_reaggregated"
            if out_path is not None:
                harness.append_rows(paired_reagg, out_path)
            paired_rescan = _estimate_and_append(
                corrected_df, label + "_rescanned", replicate, replicate_seed,
                out_path, variant_family=CALIBRATION_FAMILY,
            )
            collected.extend([paired_reagg, paired_rescan])
            # 校准家族的两个标签各自也要有诊断行：它们是最可能被引用的行，
            # 不能要求读者剥掉 "_reaggregated" 后缀才找得到自己的诊断。
            for suffix in ("_reaggregated", "_rescanned"):
                diagnostic_rows.extend(
                    diagnostics_row(
                        context, domain, subsets[domain], label + suffix,
                        replicate=replicate, seed=replicate_seed,
                        variant_family=CALIBRATION_FAMILY,
                        calibration=calibration_stats.get(domain),
                        note="paired_with:{}".format(label),
                        exposure=exposures[domain],
                    )
                    for domain in sorted(subsets)
                )

        diagnostic_rows.extend(
            diagnostics_row(
                context, domain, subsets[domain], label, replicate=replicate,
                seed=replicate_seed, calibration=calibration_stats.get(domain),
                exposure=exposures[domain],
            )
            for domain in sorted(subsets)
        )
        append_diagnostics(diagnostic_rows, diag_path)

    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(
        columns=list(harness.ROBUSTNESS_SCHEMA)
    )


# ---------------------------------------------------------------------------
# 变体二至五：词表构成
# ---------------------------------------------------------------------------

def run_leave_one_category_out(year=config.YEAR, categories=None, context=None,
                               out_path=None, diag_path=None):
    """留一类别：每次剔掉某个领域的一整类词

    Args:
        categories: {领域: {类别: [词]}}。缺省时向
            load_vocabulary_categories 要，而它对当前这两份纯词表文件恒
            返回 None——此时每个领域输出一行注明"没有类别信息"的行，
            **不从字符串上猜类别**。
    """
    context = context or build_context(year)
    if categories is None:
        categories = {
            domain: load_vocabulary_categories(domain, year)
            for domain in context.vocab
        }

    frames = []
    for domain in context.vocab:
        domain_categories = (categories or {}).get(domain)
        if not domain_categories:
            frames.append(_note_only_rows(
                "{}_leave_one_category_out_unavailable".format(domain),
                "no_category_information_in_vocabulary_file:"
                "leave_one_category_out_not_defined",
                outcome="topical_share", domain=domain, out_path=out_path,
            ))
            continue
        for category, terms in sorted(domain_categories.items()):
            removed = set(inc.normalize_vocabulary(terms))
            subsets = {
                d: (
                    [t for t in context.vocab[d] if t not in removed]
                    if d == domain else list(context.vocab[d])
                )
                for d in context.vocab
            }
            label = "{}_without_{}".format(domain, category)
            frames.append(_run_subset_variant(
                context, subsets, label, out_path, diag_path,
                note="left_out_category:{}".format(category),
            ))
    return pd.concat(frames, ignore_index=True)


def run_drop_short_terms(year=config.YEAR, min_len=3, context=None,
                         out_path=None, diag_path=None):
    """只保留 min_len 字及以上的词

    真实词表上 min_len=3 剔掉 337/816 公共事务词、182/535 明星词。诊断行
    里的 retained_fraction 就是给读者看这次干预有多大的——59% / 66% 的词
    还在，不是"微调"。
    """
    context = context or build_context(year)
    subsets = {
        domain: drop_short_terms(context.vocab[domain], min_len)
        for domain in context.vocab
    }
    for domain, subset in subsets.items():
        print("{}: 保留 {} / {} 个词（>= {} 字）".format(
            domain, len(subset), len(context.vocab[domain]), min_len))
    return _run_subset_variant(
        context, subsets, "drop_short_terms_min{}".format(min_len),
        out_path, diag_path, note="min_len={}".format(min_len),
    )


def run_drop_nested_terms(year=config.YEAR, context=None, out_path=None,
                          diag_path=None):
    """剔除全部子串嵌套词

    注意：这**不能**让重聚合变得精确。它只消掉了嵌套这一类掩盖方式，
    边界重叠（真实公共事务词表 1502 对有序词对）原封不动地留着，诊断行
    里的 n_at_risk_terms 会照实反映这一点。
    """
    context = context or build_context(year)
    subsets = {
        domain: drop_nested_terms(context.vocab[domain]) for domain in context.vocab
    }
    return _run_subset_variant(
        context, subsets, "drop_nested_terms", out_path, diag_path,
        note="removed_terms_that_are_substrings_of_another_term",
    )


def run_celebrity_person_only(year=config.YEAR, categories=None, context=None,
                              out_path=None, diag_path=None):
    """明星词表"只留人名" vs "人名加作品"

    词表文件里没有任何类别信息，这个区分**做不出来**。这里输出一行注明
    原因的行，而不是按"看起来像不像人名"去猜——猜出来的划分会产出一条
    读起来像结论、实际上是编的结果。
    """
    context = context or build_context(year)
    person_terms = (categories or {}).get("person")
    if not person_terms:
        return _note_only_rows(
            "celebrity_person_only_unavailable",
            "no_category_information_in_celebrity_vocabulary:"
            "person_vs_works_distinction_not_defined",
            outcome="topical_share", domain="celebrity", out_path=out_path,
        )
    subsets = {
        d: (
            sorted(set(inc.normalize_vocabulary(person_terms)) & set(context.vocab[d]))
            if d == "celebrity" else list(context.vocab[d])
        )
        for d in context.vocab
    }
    return _run_subset_variant(
        context, subsets, "celebrity_person_only", out_path, diag_path,
        note="person_terms_supplied_by_caller",
    )


# ---------------------------------------------------------------------------
# 全流程
# ---------------------------------------------------------------------------

def build(year=config.YEAR, n_replicates=200, keep=0.8, seed=0,
          n_calibration=DEFAULT_N_CALIBRATION, n_probe=DEFAULT_N_PROBE, min_len=3):
    """§13.3 全部词表变体，逐 replicate 增量落盘，最后写 manifest"""
    context = build_context(year)
    out_path = results_path()
    diag_path = diagnostics_path()
    os.makedirs(robustness_dir(), exist_ok=True)

    run_resampling(year, n_replicates=n_replicates, keep=keep, seed=seed,
                   n_calibration=n_calibration, n_probe=n_probe, context=context,
                   out_path=out_path, diag_path=diag_path)
    run_leave_one_category_out(year, context=context, out_path=out_path,
                               diag_path=diag_path)
    run_drop_short_terms(year, min_len=min_len, context=context,
                         out_path=out_path, diag_path=diag_path)
    run_drop_nested_terms(year, context=context, out_path=out_path,
                          diag_path=diag_path)
    run_celebrity_person_only(year, context=context, out_path=out_path,
                              diag_path=diag_path)

    results = pd.read_parquet(out_path, engine="pyarrow",
                              columns=list(harness.ROBUSTNESS_SCHEMA))
    diagnostics = pd.read_parquet(diag_path, engine="pyarrow",
                                  columns=list(DIAGNOSTIC_SCHEMA))
    calibrated = diagnostics[diagnostics["calibrated"]]
    manifest = config.build_manifest(
        step="robustness_vocabulary_{}".format(year),
        inputs=[
            os.path.relpath(
                os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year)),
                config.OUTPUT_DIR,
            ),
            "post_domain_measures_{}".format(year),
        ],
        params={
            "year": year,
            "n_replicates": n_replicates,
            "keep": keep,
            "seed": seed,
            "seeds": replicate_seeds(seed, n_replicates),
            "n_calibration": n_calibration,
            "calibration_replicates": sorted(
                calibration_replicates(n_replicates, n_calibration)
            ),
            "n_probe": n_probe,
            "min_len": min_len,
            # 逐 replicate 的诊断表另有一份，这里只记它在哪
            "diagnostics_file": os.path.basename(diag_path),
        },
        counts={
            "result_rows": int(len(results)),
            "diagnostic_rows": int(len(diagnostics)),
            "variant_labels": int(results["variant_label"].nunique()),
            # 校准的实测偏差摘要：这是"重聚合到底靠不靠得住"的证据，
            # 不是一句"影响很小"的断言。摘要是**这一次运行**的，不是常数：
            # 偏差随每个 replicate 剔掉多少风险词而变。
            "calibration": {
                "n_calibrated_rows": int(len(calibrated)),
                "mean_delta_topical_share": (
                    float(calibrated["mean_delta_topical_share"].mean())
                    if len(calibrated) else None
                ),
                "sd_delta_topical_share": (
                    float(calibrated["mean_delta_topical_share"].std())
                    if len(calibrated) > 1 else None
                ),
                "worst_delta_topical_share": (
                    float(calibrated["max_abs_delta_topical_share"].max())
                    if len(calibrated) else None
                ),
                "posts_recovered": (
                    int(calibrated["n_expressive_posts_recovered"].sum())
                    if len(calibrated) else 0
                ),
            },
            # 不依赖风险集定义的那一条证据：随机抽样看到的翻案率，以及
            # 其中落在风险集之外的部分（> 0 就说明风险集还漏了失效方式）
            "random_probe": {
                "n_sampled": (
                    int(calibrated["n_probe_sampled"].sum()) if len(calibrated) else 0
                ),
                "n_flipped": (
                    int(calibrated["n_probe_flipped"].sum()) if len(calibrated) else 0
                ),
                "n_flipped_outside_at_risk": (
                    int(calibrated["n_probe_flipped_outside_at_risk"].sum())
                    if len(calibrated) else 0
                ),
                "implied_missed_posts": (
                    float(calibrated["probe_implied_missed_posts"].sum())
                    if len(calibrated) else 0.0
                ),
            },
        },
        fingerprints={
            domain: config.fingerprint_terms(context.vocab[domain])
            for domain in context.vocab
        },
    )
    config.write_manifest(manifest, manifest_dir(year))
    return {"results_path": out_path, "diagnostics_path": diag_path}


if __name__ == "__main__":
    fire.Fire({
        "build": build,
        "resampling": run_resampling,
        "drop_short_terms": run_drop_short_terms,
        "drop_nested_terms": run_drop_nested_terms,
        "leave_one_category_out": run_leave_one_category_out,
        "celebrity_person_only": run_celebrity_person_only,
    })
