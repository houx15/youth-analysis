"""
§13.10 的综合层：把五个稳健性族的几万行估计压缩成"作者据以下判断的那几个
数字"，外加 §11.3 的 FDR 与 §12.7 的规格曲线数据。

--------------------------------------------------------------------------
这个模块唯一的纪律：**judge 报告，不裁定**
--------------------------------------------------------------------------
方案文档 §13.10 明确写了：稳健不等于"每一个版本都显著"。一个把四条准则
压成一个 `robust=True` 的布尔列，等于把作者该做的判断替他做完，而且做得
比他差——它看不见"这个量根本没被某一族检验过"，也看不见"方向一致率
是 4/4 但那 4 个变体全来自同一族"。因此本模块：

- 输出准则本身与准则背后的数字（方向一致率与它的 Wilson 区间、M0→M1 的
  衰减、单个账号/用户群/词表/月份能把估计推开多远、交互项区间是否越过 0）；
- **绝不输出任何布尔"稳健"列**，测试逐列扫描钉死这一条；
- 把"这个量缺了整整一族"标成 `completeness` 的一个取值，而不是拿手头
  恰好有的那几行给它下结论。§13.2 的 untreated vs log(1+x) 已知做不到，
  这条路径在真实数据上一定会被走到。

--------------------------------------------------------------------------
上游四件事，本模块必须原样接住
--------------------------------------------------------------------------
1. **没有任何一族产出基线行。** vocabulary 与 accounts 是有意不产出的，
   samples 的 `untreated` / `full_sample_reference` 也**不是**基线（前者
   是 §13.2 的一个变体，后者是一行只讲样本长相的参照行）。因此
   `load_all` 显式去取参照：先看数据里有没有 `variant_family="baseline"`
   的行，没有就读主结果层的三张表，再没有才考虑从表 C 重算，最后退化成
   一组注明"参照不可得"的 NaN 行。**取的是哪一种，写进每一行的
   `baseline_source` 列**，下游每一张表都带着它——"和什么比的"这件事
   不该只存在于某个人的记忆里。

2. **有些 NaN 是刻意的。** measures 的帖子类型/替代测量变体按构造碰不到
   三个 entry 量，逐行写成 NaN 并在 note 里注明；samples 的 log(1+x)
   对照整个做不到，只留一行 model/term 全空的行。方向一致率的分母因此
   只能数**真的有估计的行**——把 NaN 算进分子会报出一个从没被检验过的
   一致，把 NaN 算进分母会低报一致率。两者都是错的，前者更危险。

3. **note 是追加的，不是覆盖的。** `voc._annotate_note` 把新说明拼在已有
   note 后面，所以任何"这一行是不是那类刻意 NaN"的判断只能用子串匹配，
   绝不能用相等比较。

4. **身份键是四元组** `(variant_family, variant_label, replicate, domain)`。
   `model` 列在某些族里装的是层名以外的东西，`variant_label` 里编码着
   分母与阈值；`domain` 有三个取值，DiD 行是 `"both"`。

--------------------------------------------------------------------------
刻意不做的事
--------------------------------------------------------------------------
- **不把 vocabulary_calibration 当独立变体。** 那一族的 `_reaggregated`
  行是对应 replicate 的**逐字拷贝**（vocabulary.py 有意复用同一次拟合），
  把它算进方向一致率就是同一个数字投两次票；`_rescanned` 与它是一对
  配对测量，只有作为**差值**才有意义。因此这一族整族退出变体池，改由
  `calibration_pairs` 单独报告配对差值。
- **不对六个预先设定量做 FDR。** §11.3 写死了这一条，`apply_fdr` 逐行
  按 `harness.QUANTITY_META` 的三元组认出它们并跳过。
- **不重算任何模型。** 本模块只读各族落盘的结果表；唯一一处可能触发拟合
  的是"参照实在取不到、又允许重算"时调用 `harness.baseline`，而且默认
  排在读主结果表之后。

用法：
    python3 -m gender_domain.robustness.synthesis build --year=2020
"""

import os
from collections import OrderedDict

import fire
import numpy as np
import pandas as pd
from scipy.stats import norm

from gender_domain import config
from gender_domain import models_interaction as mi
from gender_domain import stats_utils as su
from gender_domain.robustness import accounts as acc
from gender_domain.robustness import harness
from gender_domain.robustness import measures as mea
from gender_domain.robustness import samples as smp
from gender_domain.robustness import vocabulary as voc

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BASELINE_FAMILY = "baseline"

# 参照来源的取值。写成常量而不是散落的字面量：它会被写进每一行、写进
# manifest，也会被测试引用。
BASELINE_SOURCE_ROWS = "robustness_rows:variant_family=baseline"
BASELINE_SOURCE_MAIN_RESULTS = "analysis_data/results:{}"
BASELINE_SOURCE_RECOMPUTED = "recomputed_by_harness.estimate_all_from:{}"
BASELINE_SOURCE_UNAVAILABLE = "unavailable:no_baseline_rows_no_main_results"

# 参照取不到时，占位行的 note 前缀（下游按子串识别）
BASELINE_UNAVAILABLE_NOTE = (
    "baseline_unavailable:"
    "no_variant_family=baseline_rows_and_no_readable_main_result_tables;"
    "every_comparison_against_this_quantity_is_undefined"
)
BASELINE_MISSING_ROW_NOTE = "baseline_row_missing_in_main_results"

# 五个族各自的落盘文件。键是 SLURM 数组任务的那个"族"，值是结果表文件名。
# 一个文件里可以有多个 variant_family（samples 里有四个），两者刻意分开。
FAMILY_FILES = OrderedDict([
    ("vocabulary", "vocabulary.parquet"),
    ("accounts", "accounts.parquet"),
    ("samples", "samples.parquet"),
    ("measures", "measures.parquet"),
])

# 一次完整运行**应当**出现的全部 variant_family。少了哪一个，受影响的量
# 就该被标成没测全，而不是拿剩下的行给它下结论。
EXPECTED_VARIANT_FAMILIES = (
    voc.VARIANT_FAMILY,
    acc.VARIANT_FAMILY,
    acc.BOOTSTRAP_FAMILY,
    smp.VARIANT_FAMILY_EXTREME,
    smp.VARIANT_FAMILY_EQUAL_SIZE,
    smp.VARIANT_FAMILY_BALANCE,
    smp.VARIANT_FAMILY_USER_TYPE,
    mea.VARIANT_FAMILY_DENOMINATOR,
    mea.VARIANT_FAMILY_POST_TYPE,
    mea.VARIANT_FAMILY_TEMPORAL,
)

# 配对族：不进变体池（见模块文档"刻意不做的事"）
PAIRED_FAMILIES = (voc.CALIBRATION_FAMILY,)

# M2 层只有 §13.9 那一族会产出（方案文档 §11.4：M2 会收窄样本，混进其它
# 比较会混淆结论）。因此"这一层缺了哪些族"在 M2 上只能拿这一族去比——
# 按全套十个族去比，M2 会永远显示缺九个族，那不是一条信息，是噪声。
M2_LAYER = "M2"
M2_ONLY_FAMILIES = (smp.VARIANT_FAMILY_USER_TYPE,)

# judge 里那块 §13.9 指路牌上写的话。**它是指路牌，不是第五条准则。**
M2_POINTER_NOTE = (
    "M2(account_profile_controls,13.9)_is_reported_separately_and_is_NOT_folded_"
    "into_any_criterion_on_this_row:11.4_restricts_the_M2_sample,so_M2_estimates_"
    "are_not_comparable_to_the_M0/M1_numbers_beside_them;"
    "per-variant_M2_rows_live_in_synthesis_direction.parquet_and_"
    "synthesis_specification_curve.parquet(model=='M2')"
)

# "这个变体按构造碰不到这个量"的 note 标记。**只能用子串匹配**：note 是
# 追加的，前面还挂着这个变体自己的说明。
NOT_APPLICABLE_NOTE_MARKERS = (
    "entry_not_estimated_in_this_variant",
)

# 影响力单位：§13.10 第三条准则问的是"是不是被少数账号 / 用户群 / 词 /
# 月份推着走"。每个 variant_family 归到一个单位；账号族内部还要再分
# "单个账号"与"一批账号"——"剔掉一个账号结论就变了"和"剔掉一整个类别
# 结论才变"是两件完全不同的事。
UNIT_ACCOUNT = "single_account"
UNIT_ACCOUNT_SET = "account_set"
UNIT_TERM_SET = "term_set"
UNIT_USER_GROUP = "user_group"
UNIT_MONTH = "month"
UNIT_MEASUREMENT = "measurement"
UNIT_UNCLASSIFIED = "unclassified"

# 简报点名的四个单位，judge 逐个单独成列
NAMED_UNITS = OrderedDict([
    (UNIT_ACCOUNT, "single_account"),
    (UNIT_USER_GROUP, "user_group"),
    (UNIT_TERM_SET, "term_set"),
    (UNIT_MONTH, "month"),
])

_UNIT_BY_FAMILY = {
    voc.VARIANT_FAMILY: UNIT_TERM_SET,
    acc.BOOTSTRAP_FAMILY: UNIT_ACCOUNT_SET,
    smp.VARIANT_FAMILY_EXTREME: UNIT_USER_GROUP,
    smp.VARIANT_FAMILY_EQUAL_SIZE: UNIT_USER_GROUP,
    smp.VARIANT_FAMILY_BALANCE: UNIT_USER_GROUP,
    smp.VARIANT_FAMILY_USER_TYPE: UNIT_USER_GROUP,
    mea.VARIANT_FAMILY_TEMPORAL: UNIT_MONTH,
    mea.VARIANT_FAMILY_DENOMINATOR: UNIT_MEASUREMENT,
    mea.VARIANT_FAMILY_POST_TYPE: UNIT_MEASUREMENT,
}

# accounts 族里"动的**恰好是一个**账号"的标签前缀（见 accounts.py 的标签
# 构造：`loo_{domain}_rank{NN}_{account_id}`，逐个剔除转发量最高的 top_n 个
# 账号，每次只剔一个）。
#
# **`{domain}_exclude_top{k}` 刻意不在这里面。** 它的 k 取 (1, 5, 10)，
# `public_exclude_top10` 一次剔掉十个账号；把它算成"单个账号"，就意味着
# judge 里回答 §13.5 那个核心问题（"这个结论是不是一个账号撑起来的"）的
# 数字，可能是由删掉十个账号得到的——那是另一个问题的答案。因此它落到
# `account_set`。这样不会丢掉任何信息：k=1 那一条与 `loo_..._rank01_...`
# 剔的是同一个账号（都是转发量第一名），单账号的情形已经被 loo_ 精确、
# 且穷尽地覆盖了。
_SINGLE_ACCOUNT_MARKERS = ("loo_",)

# 影响力阈值：**一个说出来的分数**，不是一个藏在代码里的判断。默认 0.5，
# 即"某个单位把估计推开了基线一半以上"，写进每一行与 manifest。
DEFAULT_INFLUENCE_THRESHOLD = 0.5

# 变体行数不变量没守住时的原因
REASON_NOTE_ONLY = "note_only_row:variant_could_not_run"
REASON_MISSING_ROWS = "missing_rows:fewer_than_len(QUANTITIES)*len(layers)"

# completeness 的取值（**不是**一个布尔裁定）
COMPLETENESS_COMPLETE = "complete"
COMPLETENESS_MISSING_FAMILIES = "incomplete:missing_variant_families"
COMPLETENESS_NO_LIVE = "incomplete:no_live_estimates"
COMPLETENESS_NO_BASELINE = "incomplete:baseline_unavailable"

# judge 每一行都带着的那句话：判断是作者的
JUDGE_NOTE = (
    "this_table_reports_the_four_criteria_of_13.10_and_the_numbers_behind_them;"
    "it_deliberately_emits_no_boolean_robustness_verdict:"
    "13.10_states_robustness_is_not_every_version_significant;"
    "the_judgement_is_the_author's"
)

# §11.3 的 FDR 只作用于次要分析。这些是主结果层里可能装着次要分析的文件；
# 其中的六个预先设定量由 apply_fdr 逐行认出并跳过。
SECONDARY_RESULT_FILES = (
    "models_entry.parquet",
    "models_intensity.parquet",
    "models_share.parquet",
    "models_persistence.parquet",
    "interaction_gender_domain.parquet",
    "decomposition_source_content.parquet",
    "combination_multinomial.parquet",
    "monthly_rates.parquet",
    "leave_one_month_out.parquet",
    "delay_quantiles.parquet",
)

# 主结果层里六个量各自在哪张表
_BASELINE_RESULT_FILES = (
    "models_entry.parquet",
    "models_share.parquet",
    "interaction_gender_domain.parquet",
)

DEFAULT_LAYERS = ("M0", "M1")

# (outcome, domain, term) -> 六个量的 key。唯一来源是 harness.QUANTITY_META，
# 本模块只做反查，不另抄一份定义。
_QUANTITY_BY_TRIPLE = {
    (meta["outcome"], meta["domain"], meta["term"]): key
    for key, meta in harness.QUANTITY_META.items()
}

# 本模块在共享 schema 之上追加的列
SYNTHESIS_EXTRA_COLUMNS = ("quantity", "source_file", "baseline_source")


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def robustness_dir():
    """稳健性层唯一允许写入的目录"""
    return os.path.join(config.OUTPUT_DIR, "robustness")


def results_dir():
    """主结果层目录。本模块对它**只读**"""
    return os.path.join(config.OUTPUT_DIR, "results")


def manifest_dir(year):
    return os.path.join(robustness_dir(), "synthesis_{}".format(year))


def output_path(name):
    return os.path.join(robustness_dir(), "{}.parquet".format(name))


# ---------------------------------------------------------------------------
# 行的分类：量、影响力单位、刻意 NaN
# ---------------------------------------------------------------------------

def quantity_of(outcome, domain, term):
    """(outcome, domain, term) 反查六个量的 key，认不出来返回 None

    认不出来是正常的：注明原因的行（model/term 全空）与主结果层的次要
    分析行都落在这里，它们不该被硬塞进某个量。
    """
    return _QUANTITY_BY_TRIPLE.get((outcome, domain, term))


def influence_unit(variant_family, variant_label):
    """这一行动的是哪一类东西：单个账号 / 一批账号 / 词表 / 用户群 / 月份"""
    if variant_family == acc.VARIANT_FAMILY:
        label = "" if variant_label is None else str(variant_label)
        if any(marker in label for marker in _SINGLE_ACCOUNT_MARKERS):
            return UNIT_ACCOUNT
        return UNIT_ACCOUNT_SET
    return _UNIT_BY_FAMILY.get(variant_family, UNIT_UNCLASSIFIED)


def is_not_applicable(note):
    """这一行的 NaN 是不是"按构造碰不到"，靠**子串**判断

    note 由 voc._annotate_note 追加拼接，前面一定还挂着这个变体自己的
    说明，因此相等比较必然失效——上游两次复核抓到的就是这一类错误。
    """
    if note is None or (isinstance(note, float) and note != note):
        return False
    text = str(note)
    return any(marker in text for marker in NOT_APPLICABLE_NOTE_MARKERS)


# ---------------------------------------------------------------------------
# 参照的显式取得
# ---------------------------------------------------------------------------

def _nan_baseline_rows(layers, note, source):
    """参照取不到时的占位行：六个量 × 各层，全部 NaN 并注明原因"""
    rows = []
    for model in layers:
        for quantity in harness.QUANTITIES:
            meta = harness.QUANTITY_META[quantity]
            row = su.tidy_result(
                outcome=meta["outcome"], domain=meta["domain"], model=model,
                term=meta["term"], estimate=np.nan, se=np.nan, ci_low=np.nan,
                ci_high=np.nan, scale=meta["scale"], n_obs=0, n_dropped=0,
                drop_reason=None, note=note,
            )
            row = harness.attach_variant_identity(
                row, BASELINE_FAMILY, BASELINE_FAMILY, 0, None)
            row["quantity"] = quantity
            row["source_file"] = source
            rows.append(row)
    return pd.DataFrame(
        rows, columns=list(harness.ROBUSTNESS_SCHEMA) + ["quantity", "source_file"])


def _baseline_from_main_results(layers, directory):
    """从主结果层的三张表里按 (outcome, domain, term, model) 取六个量

    这是默认的参照来源：论文正文报告的就是这三张表里的数字，稳健性检验
    该跟它比，而不是跟一个在稳健性层里现算的、可能用了不同表 C 版本的
    数字比。
    """
    frames = []
    used = []
    for name in _BASELINE_RESULT_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        frames.append(pd.read_parquet(path, engine="pyarrow",
                                      columns=list(su.RESULT_SCHEMA)))
        used.append(name)
    if not frames:
        return None, None
    pooled = pd.concat(frames, ignore_index=True)
    rows = []
    n_missing = 0
    for model in layers:
        for quantity in harness.QUANTITIES:
            meta = harness.QUANTITY_META[quantity]
            mask = (
                (pooled["outcome"] == meta["outcome"])
                & (pooled["domain"] == meta["domain"])
                & (pooled["term"] == meta["term"])
                & (pooled["model"] == model)
            )
            sub = pooled[mask]
            if len(sub) != 1:
                n_missing += 1
                row = su.tidy_result(
                    outcome=meta["outcome"], domain=meta["domain"], model=model,
                    term=meta["term"], estimate=np.nan, se=np.nan, ci_low=np.nan,
                    ci_high=np.nan, scale=meta["scale"], n_obs=0, n_dropped=0,
                    drop_reason=None,
                    note="{}:{}/{}(found {} rows)".format(
                        BASELINE_MISSING_ROW_NOTE, quantity, model, len(sub)),
                )
            else:
                row = {col: sub.iloc[0][col] for col in su.RESULT_SCHEMA}
            row = harness.attach_variant_identity(
                row, BASELINE_FAMILY, BASELINE_FAMILY, 0, None)
            row["quantity"] = quantity
            row["source_file"] = "+".join(used)
            rows.append(row)
    if n_missing:
        print("警告: 主结果表里有 {} 个 (量, 层) 组合找不到唯一一行，"
              "这些参照写成 NaN 并注明原因".format(n_missing))
    frame = pd.DataFrame(
        rows, columns=list(harness.ROBUSTNESS_SCHEMA) + ["quantity", "source_file"])
    return frame, BASELINE_SOURCE_MAIN_RESULTS.format("+".join(used))


def _baseline_recomputed(year, layers):
    """最后一条退路：从表 C 现场重算一次基线

    默认排在读主结果表之后，因为重算出来的数字与论文正文报告的数字之间
    再无任何东西保证一致（表 C 可能已经重建过）。真的走到这一步时，
    `baseline_source` 会写明"这是重算的"。
    """
    candidates = [
        os.path.join(config.OUTPUT_DIR,
                     "user_domain_{}_with_profile.parquet".format(year)),
        os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year)),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            user_df = pd.read_parquet(path, engine="pyarrow",
                                      columns=list(voc.USER_TABLE_COLUMNS))
            frame = harness.estimate_all(
                user_df, variant_family=BASELINE_FAMILY,
                variant_label=BASELINE_FAMILY, replicate=0, seed=None,
                layers=tuple(layers),
            )
        except Exception as exc:  # noqa: BLE001 —— 取不到参照必须留痕
            print("警告: 从 {} 重算基线失败（{}: {}），继续退化".format(
                os.path.basename(path), type(exc).__name__, exc))
            continue
        frame = frame.copy()
        frame["quantity"] = [
            quantity_of(o, d, t)
            for o, d, t in zip(frame["outcome"], frame["domain"], frame["term"])
        ]
        frame["source_file"] = os.path.basename(path)
        return frame, BASELINE_SOURCE_RECOMPUTED.format(os.path.basename(path))
    return None, None


def resolve_baseline(variants, year=config.YEAR, main_results_dir=None,
                     allow_recompute=True):
    """显式取得参照，并返回 (参照行, 来源说明)

    顺序刻意如此：
      1. 数据里已有 `variant_family="baseline"` 的行——最直接；
      2. 主结果层的三张表——论文正文报告的就是它们，默认走这一条；
      3. 从表 C 现场重算（可关）；
      4. 都没有：一组注明"参照不可得"的 NaN 行。**不编造参照。**
    """
    layers = _layers_of(variants)
    existing = variants[variants["variant_family"] == BASELINE_FAMILY]
    if len(existing):
        print("参照来源: 数据里已有 {} 行 variant_family=baseline".format(len(existing)))
        return existing.copy(), BASELINE_SOURCE_ROWS

    frame, source = _baseline_from_main_results(
        layers, main_results_dir if main_results_dir is not None else results_dir())
    if frame is not None:
        print("参照来源: {}".format(source))
        return frame, source

    if allow_recompute:
        frame, source = _baseline_recomputed(year, layers)
        if frame is not None:
            print("参照来源: {}".format(source))
            return frame, source

    print("警告: 参照不可得——既没有 baseline 行，也读不到主结果表。"
          "六个量的参照写成 NaN 并注明原因，所有比较都会随之为 NaN")
    return (_nan_baseline_rows(layers, BASELINE_UNAVAILABLE_NOTE,
                               BASELINE_SOURCE_UNAVAILABLE),
            BASELINE_SOURCE_UNAVAILABLE)


def _layers_of(frame):
    """变体行里出现过的模型层。§13.9 的 user_type 族会带上 M2"""
    if "model" not in frame.columns or not len(frame):
        return tuple(DEFAULT_LAYERS)
    layers = sorted({m for m in frame["model"].dropna().unique()})
    return tuple(layers) if layers else tuple(DEFAULT_LAYERS)


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------

def load_all(year=config.YEAR, directory=None, main_results_dir=None,
             allow_recompute=True):
    """把五个族的结果行读成一张表，并显式补上参照行

    Args:
        year: 年份，只用于重算参照时定位表 C
        directory: 稳健性结果目录，默认 analysis_data/robustness
        main_results_dir: 主结果目录，默认 analysis_data/results（只读）
        allow_recompute: 参照实在取不到时，允不允许从表 C 现场重算

    Returns:
        DataFrame，列 = ROBUSTNESS_SCHEMA + quantity + source_file +
        baseline_source。`quantity` 认不出来的行（注明原因的行、schema
        对得上但不属于六个量的行）如实留空，不硬塞。`baseline_source`
        是整张表的常量，因此任何子集都还带着"和什么比的"这条信息。
    """
    directory = directory if directory is not None else robustness_dir()
    frames = []
    missing = []
    for family, filename in FAMILY_FILES.items():
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            missing.append(family)
            print("警告: 找不到 {} 族的结果表 {}，这一族在下游会被标成缺失".format(
                family, path))
            continue
        frame = pd.read_parquet(path, engine="pyarrow",
                                columns=list(harness.ROBUSTNESS_SCHEMA))
        frame = frame.copy()
        frame["source_file"] = filename
        frames.append(frame)
        print("已读入 {}: {} 行".format(filename, len(frame)))

    if frames:
        variants = pd.concat(frames, ignore_index=True)
    else:
        variants = pd.DataFrame(
            columns=list(harness.ROBUSTNESS_SCHEMA) + ["source_file"])

    variants["quantity"] = [
        quantity_of(o, d, t)
        for o, d, t in zip(variants["outcome"], variants["domain"], variants["term"])
    ]

    baseline, source = resolve_baseline(
        variants, year=year, main_results_dir=main_results_dir,
        allow_recompute=allow_recompute)
    baseline = baseline.copy()
    if "source_file" not in baseline.columns:
        baseline["source_file"] = source
    if "quantity" not in baseline.columns:
        baseline["quantity"] = [
            quantity_of(o, d, t)
            for o, d, t in zip(baseline["outcome"], baseline["domain"],
                               baseline["term"])
        ]

    columns = list(harness.ROBUSTNESS_SCHEMA) + ["quantity", "source_file"]
    if len(variants) and (variants["variant_family"] == BASELINE_FAMILY).any():
        out = variants[columns]
    else:
        out = pd.concat([variants[columns], baseline[columns]], ignore_index=True)
    out = out.reset_index(drop=True)
    out["baseline_source"] = source
    print("综合层共读入 {} 行（其中参照 {} 行），缺失的族: {}".format(
        len(out), int((out["variant_family"] == BASELINE_FAMILY).sum()),
        missing or "无"))
    return out


def _prepare(df):
    """补齐 quantity / baseline_source 两列，让手工构造的帧也能直接进来"""
    out = df.copy()
    if "quantity" not in out.columns:
        out["quantity"] = [
            quantity_of(o, d, t)
            for o, d, t in zip(out["outcome"], out["domain"], out["term"])
        ]
    if "baseline_source" not in out.columns:
        has_rows = (out["variant_family"] == BASELINE_FAMILY).any()
        out["baseline_source"] = (
            BASELINE_SOURCE_ROWS if has_rows else BASELINE_SOURCE_UNAVAILABLE)
    return out


def variant_pool(df):
    """真正参与比较的那些行：去掉参照行、配对族与认不出量的行

    配对族（vocabulary_calibration）为什么退出：见模块文档。认不出量的
    行为什么退出：它们没有估计对象，无法与任何参照比较——它们由
    `incomplete_variants` 单独列出，不是被丢掉。
    """
    prepared = _prepare(df)
    excluded = {BASELINE_FAMILY} | set(PAIRED_FAMILIES)
    mask = (~prepared["variant_family"].isin(excluded)) & prepared["quantity"].notna()
    return prepared[mask].reset_index(drop=True)


def baseline_map(df):
    """(quantity, model) -> 参照估计值"""
    prepared = _prepare(df)
    rows = prepared[prepared["variant_family"] == BASELINE_FAMILY]
    return {
        (row["quantity"], row["model"]): row["estimate"]
        for _, row in rows.iterrows()
        if row["quantity"] is not None and row["quantity"] == row["quantity"]
    }


# ---------------------------------------------------------------------------
# 行数不变量：谁没跑完
# ---------------------------------------------------------------------------

def incomplete_variants(df):
    """按 estimate_all 的行数不变量找出"没真的跑出来"的变体

    `harness.estimate_all` 恒产出 `len(QUANTITIES) × len(layers)` 行，失败
    也写 NaN 行。因此一个变体的行数少于这个数，只可能是它压根没跑完——
    这正是那条不变量存在的理由。注明原因的行（model/term 全空）会被单独
    标成 `note_only`，它是"这个变体做不到"的既定写法，不是事故。
    """
    prepared = _prepare(df)
    rows = []
    key = ["variant_family", "variant_label", "replicate"]
    for (family, label, replicate), group in prepared.groupby(key, dropna=False):
        if family == BASELINE_FAMILY:
            continue
        # 这个变体实际跑了几层。注明原因的行 model 是空的，此时按一层算，
        # 期望行数仍然是六个量——它就是要显示成"只写出来 1 行"。
        n_models = int(group["model"].nunique())
        expected = len(harness.QUANTITIES) * max(n_models, 1)
        if len(group) == expected and group["quantity"].notna().all():
            continue
        if group["model"].isna().all():
            reason = REASON_NOTE_ONLY
        else:
            reason = REASON_MISSING_ROWS
        rows.append({
            "variant_family": family,
            "variant_label": label,
            "replicate": replicate,
            "n_rows": int(len(group)),
            "n_rows_expected": int(expected),
            "n_models": int(n_models),
            "reason": reason,
            "note": _first_note(group),
        })
    return pd.DataFrame(rows, columns=[
        "variant_family", "variant_label", "replicate", "n_rows",
        "n_rows_expected", "n_models", "reason", "note",
    ])


def _first_note(group):
    notes = [n for n in group["note"].tolist() if isinstance(n, str) and n]
    return notes[0] if notes else None


# ---------------------------------------------------------------------------
# §13.10 准则一：方向一致率
# ---------------------------------------------------------------------------

def expected_families_for(model, families=EXPECTED_VARIANT_FAMILIES):
    """这一层上**应当**出现的 variant_family（M2 只有 §13.9 那一族）"""
    return tuple(M2_ONLY_FAMILIES) if model == M2_LAYER else tuple(families)


def _family_status(group_all, families):
    """逐 family 判定这个量在这一层上到底被测了没有

    四种状态互斥：
      tested          有活着的估计
      not_applicable  有行、但全是"按构造碰不到"的 NaN（子串匹配）
      failed          有行、全是 NaN，且没有那个标记 —— 这是真的失败
      missing         一行都没有
    """
    tested, not_applicable, failed, missing = [], [], [], []
    for family in families:
        sub = group_all[group_all["variant_family"] == family]
        if not len(sub):
            missing.append(family)
        elif sub["estimate"].notna().any():
            tested.append(family)
        elif sub["note"].map(is_not_applicable).any():
            not_applicable.append(family)
        else:
            failed.append(family)
    return tested, not_applicable, failed, missing


def direction_consistency(df, families=EXPECTED_VARIANT_FAMILIES):
    """逐 (量, 层)：与参照同号的变体占**活着的变体**的比例，以及估计的跨度

    分母 `n_live` 只数真的有估计的行。刻意 NaN 的行单独记在 `n_nan` 里，
    既不进分子也不进分母——这是本模块最容易写错、也最要命的一处：把它们
    算进分子会报出一个从没被检验过的一致。
    """
    prepared = _prepare(df)
    pool = variant_pool(prepared)
    base = baseline_map(prepared)
    source = str(prepared["baseline_source"].iloc[0]) if len(prepared) else None

    models = _layers_of(pool) if len(pool) else _layers_of(prepared)
    rows = []
    for quantity in harness.QUANTITIES:
        for model in models:
            group = pool[(pool["quantity"] == quantity) & (pool["model"] == model)]
            live = group[group["estimate"].notna()]
            baseline_estimate = base.get((quantity, model), np.nan)
            baseline_sign = (
                float(np.sign(baseline_estimate))
                if baseline_estimate == baseline_estimate else np.nan)
            if len(live) and baseline_sign == baseline_sign:
                n_agree = int((np.sign(live["estimate"].astype(float))
                               == baseline_sign).sum())
                share = n_agree / float(len(live))
                ci_low, ci_high = su.proportion_ci(n_agree, len(live))
            else:
                # 参照拿不到、或者一个活着的估计都没有：这次比较**根本
                # 没能做**。n_agree 必须也写 NaN，不能写 0——写 0 会让
                # 一个从没做过的比较读起来像"0 / n_live 个变体同号"，
                # 即一次全面的不一致。这与本模块在方向一致率分母上守的
                # 是同一条纪律：做不到的事不能渲染成一条负面事实。
                n_agree, share, ci_low, ci_high = (
                    np.nan, np.nan, np.nan, np.nan)

            tested, not_applicable, failed, missing = _family_status(
                group, expected_families_for(model, families))
            incomplete = bool(missing or failed or not len(live)
                              or baseline_estimate != baseline_estimate)
            rows.append({
                "quantity": quantity,
                "model": model,
                "baseline_source": source,
                "baseline_estimate": float(baseline_estimate)
                if baseline_estimate == baseline_estimate else np.nan,
                "baseline_sign": baseline_sign,
                "n_rows": int(len(group)),
                "n_live": int(len(live)),
                "n_nan": int(len(group) - len(live)),
                "n_agree": float(n_agree) if n_agree == n_agree else np.nan,
                "share_agree": float(share) if share == share else np.nan,
                "share_ci_low": float(ci_low) if ci_low == ci_low else np.nan,
                "share_ci_high": float(ci_high) if ci_high == ci_high else np.nan,
                "estimate_min": float(live["estimate"].min()) if len(live) else np.nan,
                "estimate_max": float(live["estimate"].max()) if len(live) else np.nan,
                "estimate_median": float(live["estimate"].median())
                if len(live) else np.nan,
                "n_families_live": int(len(tested)),
                "families_tested": "+".join(tested) or None,
                "families_not_applicable": "+".join(not_applicable) or None,
                "families_failed": "+".join(failed) or None,
                "families_missing": "+".join(missing) or None,
                "incompletely_tested": incomplete,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §13.10 准则二：活动量调整带来的衰减
# ---------------------------------------------------------------------------

def _attenuation(m0, m1):
    """1 - |M1| / |M0|：正数表示调整之后估计变小了"""
    if m0 != m0 or m1 != m1 or float(m0) == 0.0:
        return np.nan
    return 1.0 - abs(float(m1)) / abs(float(m0))


def activity_attenuation(df, layer_from="M0", layer_to="M1"):
    """逐量：参照的 M0→M1 衰减，以及各变体自己配对算出来的衰减分布

    配对是按 `(variant_family, variant_label, replicate)` 做的——两层必须
    来自同一个变体，拿 A 变体的 M0 去比 B 变体的 M1 没有任何意义。
    """
    prepared = _prepare(df)
    pool = variant_pool(prepared)
    base = baseline_map(prepared)
    source = str(prepared["baseline_source"].iloc[0]) if len(prepared) else None

    rows = []
    for quantity in harness.QUANTITIES:
        group = pool[pool["quantity"] == quantity]
        # values 是**两层都活着**的配对；其中 M0 恰好等于 0 的那些配对算不出
        # 衰减（分母为 0），进 finite 不了。两个计数必须分开报，否则
        # n_variant_pairs 说"有 N 对"、而中位数/分位数只用了其中一部分，
        # 读者无从知道差在哪里。
        values, flips = [], 0
        key = ["variant_family", "variant_label", "replicate"]
        if len(group):
            for _, sub in group.groupby(key, dropna=False):
                first = sub[sub["model"] == layer_from]["estimate"]
                second = sub[sub["model"] == layer_to]["estimate"]
                if len(first) != 1 or len(second) != 1:
                    continue
                m0, m1 = float(first.iloc[0]), float(second.iloc[0])
                if m0 != m0 or m1 != m1:
                    continue
                values.append(_attenuation(m0, m1))
                if np.sign(m0) != np.sign(m1):
                    flips += 1
        finite = [v for v in values if v == v]
        b0 = base.get((quantity, layer_from), np.nan)
        b1 = base.get((quantity, layer_to), np.nan)
        rows.append({
            "quantity": quantity,
            "layer_from": layer_from,
            "layer_to": layer_to,
            "baseline_source": source,
            "baseline_estimate_M0": float(b0) if b0 == b0 else np.nan,
            "baseline_estimate_M1": float(b1) if b1 == b1 else np.nan,
            "baseline_attenuation": _attenuation(b0, b1),
            # n_variant_pairs 与下面的分位数**口径一致**：都只数算得出衰减
            # 的那些配对
            "n_variant_pairs": int(len(finite)),
            "n_matched_pairs": int(len(values)),
            "n_pairs_attenuation_undefined": int(len(values) - len(finite)),
            "attenuation_median": float(np.median(finite)) if finite else np.nan,
            "attenuation_p10": float(np.percentile(finite, 10)) if finite else np.nan,
            "attenuation_p90": float(np.percentile(finite, 90)) if finite else np.nan,
            "attenuation_min": float(np.min(finite)) if finite else np.nan,
            "attenuation_max": float(np.max(finite)) if finite else np.nan,
            "n_sign_flip_M0_to_M1": int(flips),
            "note": "attenuation = 1 - |{}| / |{}|; positive means the estimate "
                    "shrinks after activity adjustment".format(layer_to, layer_from),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §13.10 准则三：是不是被少数账号 / 用户群 / 词 / 月份推着走
# ---------------------------------------------------------------------------

def influence_summary(df, threshold=DEFAULT_INFLUENCE_THRESHOLD):
    """逐 (量, 层, 影响力单位)：这个单位最多能把估计推开基线的多少

    `threshold` 是一个**说出来的分数**，写进每一行；`exceeds_threshold`
    是一条事实（有没有越过这个分数），不是"稳不稳健"的裁定。
    """
    prepared = _prepare(df)
    pool = variant_pool(prepared)
    base = baseline_map(prepared)
    source = str(prepared["baseline_source"].iloc[0]) if len(prepared) else None
    if not len(pool):
        return pd.DataFrame(columns=[
            "quantity", "model", "influence_unit", "baseline_source",
            "baseline_estimate", "n_live", "max_abs_relative_shift",
            "worst_variant_family", "worst_variant_label", "worst_estimate",
            "threshold", "n_exceeding", "exceeds_threshold",
        ])

    pool = pool.copy()
    pool["influence_unit"] = [
        influence_unit(f, l)
        for f, l in zip(pool["variant_family"], pool["variant_label"])
    ]
    rows = []
    for (quantity, model, unit), group in pool.groupby(
            ["quantity", "model", "influence_unit"], dropna=False):
        live = group[group["estimate"].notna()]
        baseline_estimate = base.get((quantity, model), np.nan)
        if not len(live) or baseline_estimate != baseline_estimate \
                or float(baseline_estimate) == 0.0:
            shift = pd.Series(dtype=float)
        else:
            shift = (live["estimate"].astype(float) - float(baseline_estimate)).abs() \
                / abs(float(baseline_estimate))
        if len(shift):
            position = int(shift.values.argmax())
            worst = live.iloc[position]
            max_shift = float(shift.iloc[position])
            n_exceeding = int((shift > float(threshold)).sum())
        else:
            # 参照拿不到（或这一格没有活着的估计）时，"有没有越过阈值"
            # 这个问题**没被问过**。n_exceeding 与 exceeds_threshold 都写
            # 空值，不写 0 / False——后者读起来是"问过了，没越过"。
            worst, max_shift, n_exceeding = None, np.nan, np.nan
        rows.append({
            "quantity": quantity,
            "model": model,
            "influence_unit": unit,
            "baseline_source": source,
            "baseline_estimate": float(baseline_estimate)
            if baseline_estimate == baseline_estimate else np.nan,
            "n_live": int(len(live)),
            "max_abs_relative_shift": max_shift,
            "worst_variant_family": None if worst is None else worst["variant_family"],
            "worst_variant_label": None if worst is None else worst["variant_label"],
            "worst_estimate": np.nan if worst is None else float(worst["estimate"]),
            "threshold": float(threshold),
            "n_exceeding": n_exceeding,
            "exceeds_threshold": (
                bool(max_shift > float(threshold))
                if max_shift == max_shift else None),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §11.3 FDR：只作用于次要分析
# ---------------------------------------------------------------------------

def is_prespecified(outcome, domain, term):
    """这一行是不是六个预先设定量之一——是的话永远不进 BH 校正"""
    return quantity_of(outcome, domain, term) is not None


def apply_fdr(df, alpha=0.05):
    """对**次要分析**行做 Benjamini–Hochberg 校正，六个预先设定量一律跳过

    §11.3 写死了这一条：预先设定的量不做多重比较校正，做了反而是错的
    （它们不是"从一堆结果里挑出来的"）。因此本函数逐行按
    `harness.QUANTITY_META` 的 (outcome, domain, term) 三元组认出它们，
    给它们 `q_value = NaN`、`fdr_rejected = None`，永不参与排序。

    p 值由 Wald 正态近似（estimate / se）现算。se 缺失或为 0 的行（例如
    区间来自 bootstrap 的行）无法算 p 值，如实标成未检验，而不是拿区间
    去反推一个假的 se。
    """
    out = df.copy()
    out["is_prespecified"] = [
        is_prespecified(o, d, t)
        for o, d, t in zip(out["outcome"], out["domain"], out["term"])
    ]

    p_values, sources = [], []
    for estimate, se in zip(out["estimate"], out["se"]):
        usable = (estimate == estimate and se == se and se is not None
                  and float(se) > 0.0)
        if usable:
            z = abs(float(estimate) / float(se))
            p_values.append(float(2.0 * norm.sf(z)))
            sources.append("wald_normal_from_estimate_and_se")
        else:
            p_values.append(np.nan)
            sources.append("unavailable:se_missing_or_zero")
    out["p_value"] = p_values
    out["p_value_source"] = sources

    testable = (~out["is_prespecified"]) & out["p_value"].notna()
    n_tested = int(testable.sum())
    q_values = pd.Series(np.nan, index=out.index, dtype=float)
    rejected = pd.Series([None] * len(out), index=out.index, dtype=object)
    if n_tested:
        ordered = out.loc[testable, "p_value"].sort_values()
        ranks = np.arange(1, n_tested + 1, dtype=float)
        raw = ordered.values * n_tested / ranks
        # BH 的阶梯校正：从最大的 p 往回取累积最小值，保证 q 单调不减
        adjusted = np.minimum.accumulate(raw[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        q_values.loc[ordered.index] = adjusted
        rejected.loc[ordered.index] = [bool(v <= float(alpha)) for v in adjusted]
    out["q_value"] = q_values
    out["fdr_rejected"] = rejected
    out["fdr_alpha"] = float(alpha)
    out["fdr_n_tested"] = int(n_tested)
    print("FDR: {} 行中 {} 行是预先设定量（跳过），{} 行进入 BH 校正，"
          "alpha={}".format(len(out), int(out["is_prespecified"].sum()),
                            n_tested, alpha))
    return out


# ---------------------------------------------------------------------------
# §12.7 规格曲线数据
# ---------------------------------------------------------------------------

def specification_curve_data(df, threshold=DEFAULT_INFLUENCE_THRESHOLD):
    """逐 (量, 层) 把全部变体按估计值排序，带上足以画图的身份标签

    估不出来的行**保留**（`estimate_available=False`、`rank` 为空、note
    原样带着）：一张只画得出结果的规格曲线会让读者以为那些变体没跑过。
    注明原因的行（连量都认不出来）不在这里，它们由 `incomplete_variants`
    列出——它们没有估计对象，摆进某个量的曲线里只会误导。
    """
    prepared = _prepare(df)
    pool = variant_pool(prepared)
    base = baseline_map(prepared)
    source = str(prepared["baseline_source"].iloc[0]) if len(prepared) else None
    baseline_rows = prepared[
        (prepared["variant_family"] == BASELINE_FAMILY)
        & prepared["quantity"].notna()
    ]
    frame = pd.concat([pool, baseline_rows], ignore_index=True)
    if not len(frame):
        return pd.DataFrame(columns=[
            "quantity", "model", "variant_family", "variant_label", "replicate",
            "seed", "influence_unit", "is_baseline", "estimate", "se", "ci_low",
            "ci_high", "estimate_available", "rank", "n_specifications",
            "crosses_zero", "agrees_with_baseline", "relative_shift",
            "baseline_estimate", "baseline_source", "note",
        ])

    frame = frame.copy()
    frame["influence_unit"] = [
        influence_unit(f, l)
        for f, l in zip(frame["variant_family"], frame["variant_label"])
    ]
    frame["is_baseline"] = frame["variant_family"] == BASELINE_FAMILY
    frame["estimate_available"] = frame["estimate"].notna()
    frame["baseline_estimate"] = [
        base.get((q, m), np.nan) for q, m in zip(frame["quantity"], frame["model"])
    ]
    frame["baseline_source"] = source
    frame["crosses_zero"] = [
        bool(low == low and high == high and low <= 0.0 <= high)
        for low, high in zip(frame["ci_low"], frame["ci_high"])
    ]
    frame["agrees_with_baseline"] = [
        bool(e == e and b == b and np.sign(e) == np.sign(b))
        for e, b in zip(frame["estimate"], frame["baseline_estimate"])
    ]
    frame["relative_shift"] = [
        (float(e) - float(b)) / abs(float(b))
        if (e == e and b == b and float(b) != 0.0) else np.nan
        for e, b in zip(frame["estimate"], frame["baseline_estimate"])
    ]

    # 排序：先按量与层分组，组内可用的按估计值升序排在前、rank 从 1 起，
    # 估不出来的排在组尾、rank 留空
    frame = frame.sort_values(
        ["quantity", "model", "estimate_available", "estimate"],
        ascending=[True, True, False, True], kind="mergesort",
    ).reset_index(drop=True)
    ranks, counts = [], []
    for (_, _), group in frame.groupby(["quantity", "model"], sort=False):
        n_available = int(group["estimate_available"].sum())
        position = 0
        for available in group["estimate_available"]:
            if available:
                position += 1
                ranks.append(float(position))
            else:
                ranks.append(np.nan)
            counts.append(n_available)
    frame["rank"] = ranks
    frame["n_specifications"] = counts

    columns = [
        "quantity", "model", "variant_family", "variant_label", "replicate",
        "seed", "influence_unit", "is_baseline", "estimate", "se", "ci_low",
        "ci_high", "estimate_available", "rank", "n_specifications",
        "crosses_zero", "agrees_with_baseline", "relative_shift",
        "baseline_estimate", "baseline_source", "note",
    ]
    return frame[columns]


# ---------------------------------------------------------------------------
# 配对校准：被排除出变体池的那一族，单独报告
# ---------------------------------------------------------------------------

def calibration_pairs(df):
    """vocabulary_calibration 的 `_reaggregated` 与 `_rescanned` 配对差值

    这一族不进变体池（它的 `_reaggregated` 行是对应 replicate 的逐字
    拷贝），但把它整族丢掉同样不行——重聚合与重扫之间的差就是 §13.3
    自己声明要测的那个测量误差。因此在这里单独成表。
    """
    prepared = _prepare(df)
    rows = prepared[prepared["variant_family"].isin(PAIRED_FAMILIES)]
    out = []
    for _, row in rows.iterrows():
        label = str(row["variant_label"])
        for suffix in ("_reaggregated", "_rescanned"):
            if label.endswith(suffix):
                out.append({
                    "quantity": row["quantity"],
                    "model": row["model"],
                    "pair_key": label[: -len(suffix)],
                    "arm": suffix.lstrip("_"),
                    "estimate": row["estimate"],
                    "note": row["note"],
                })
                break
    frame = pd.DataFrame(out, columns=[
        "quantity", "model", "pair_key", "arm", "estimate", "note"])
    if not len(frame):
        return frame.assign(delta=pd.Series(dtype=float))
    wide = frame.pivot_table(index=["quantity", "model", "pair_key"],
                             columns="arm", values="estimate",
                             aggfunc="first").reset_index()
    for arm in ("reaggregated", "rescanned"):
        if arm not in wide.columns:
            wide[arm] = np.nan
    wide["delta"] = wide["rescanned"] - wide["reaggregated"]
    return wide


# ---------------------------------------------------------------------------
# judge：把四条准则与它们背后的数字摆在一起
# ---------------------------------------------------------------------------

def _share_ci_excludes_zero(pool, quantity, model):
    """这一层上，区间不越过 0 的变体占活着的变体多少（DiD 行即"交互还在不在"）"""
    group = pool[(pool["quantity"] == quantity) & (pool["model"] == model)]
    live = group[group["estimate"].notna()
                 & group["ci_low"].notna() & group["ci_high"].notna()]
    if not len(live):
        return np.nan, 0
    excludes = int(((live["ci_low"] > 0) | (live["ci_high"] < 0)).sum())
    return excludes / float(len(live)), int(len(live))


def judge(df, threshold=DEFAULT_INFLUENCE_THRESHOLD, layers=DEFAULT_LAYERS):
    """§13.10 的四条准则，逐量一行——**只报告数字，不给出裁定**

    每一行回答：方向在多少比例的变体里守住了（带 Wilson 区间）；活动量
    调整让估计缩了多少；单个账号 / 用户群 / 词表 / 月份最多能把它推开多远，
    哪个变体推得最远；区间不越过 0 的变体占多少（对 did_* 两行就是"交互
    还在不在"）。外加一列 `completeness`：这个量到底有没有被测全。

    **刻意不输出的东西**：任何 `robust` / `verdict` / `passes` 之类的布尔
    结论。方案文档 §13.10 明确写了稳健不等于"每个版本都显著"，把四条准则
    压成一个布尔值等于替作者做完判断，而且做得比他差——它看不见"一致率
    4/4 但四个变体全来自同一族"这种局面。

    `layers` 默认只有 M0/M1：M2 只有 §13.9 那一族会产出（方案文档 §11.4，
    M2 会收窄样本），把它摆进跨族比较里只会混淆结论——"变体同号比例"由
    一个族算出来根本不是 §13.10 的准则一。**四条准则一律按 `layers` 过滤**，
    包括准则三：一个 M2 变体不允许把它在受限样本上的偏移贡献给一行其余
    数字全是 M0/M1 的记录。

    但 §13.9 本身是论文欠读者的一个交代，judge 不能对它只字不提。因此这里
    放一块**指路牌**（不是第五条准则）：`n_M2_variants`、
    `m2_direction_share_profile_family` 与 `m2_pointer`，告诉读者 M2 的答案
    在 `synthesis_direction.parquet` 与 `synthesis_specification_curve.parquet`
    的 `model == "M2"` 行里，并写明它与同一行的 M0/M1 数字不来自同一个样本、
    不能并排读。
    """
    prepared = _prepare(df)
    pool = variant_pool(prepared)
    direction = direction_consistency(prepared)
    attenuation = activity_attenuation(prepared)
    influence = influence_summary(prepared, threshold=threshold)
    incomplete = incomplete_variants(prepared)
    source = str(prepared["baseline_source"].iloc[0]) if len(prepared) else None

    rows = []
    for quantity in harness.QUANTITIES:
        row = {"quantity": quantity, "baseline_source": source}

        # --- 准则一：方向 ---
        missing_all, not_applicable_all = set(), set()
        n_live_total = 0
        baseline_available = False
        for model in layers:
            sub = direction[(direction["quantity"] == quantity)
                            & (direction["model"] == model)]
            if not len(sub):
                # 这一层压根没出现在数据里（例如只跑了 M0 的一次局部运行）。
                # 如实写 NaN，不省略这几列——省略会让不同次运行的 judge 表
                # 列集合不一致，下游拼不起来。
                for name in ("direction_share", "direction_n_live",
                             "direction_n_nan", "direction_ci_low",
                             "direction_ci_high", "estimate_min",
                             "estimate_max", "baseline_estimate"):
                    row["{}_{}".format(name, model)] = np.nan
                continue
            item = sub.iloc[0]
            row["direction_share_{}".format(model)] = item["share_agree"]
            row["direction_n_live_{}".format(model)] = int(item["n_live"])
            row["direction_n_nan_{}".format(model)] = int(item["n_nan"])
            row["direction_ci_low_{}".format(model)] = item["share_ci_low"]
            row["direction_ci_high_{}".format(model)] = item["share_ci_high"]
            row["estimate_min_{}".format(model)] = item["estimate_min"]
            row["estimate_max_{}".format(model)] = item["estimate_max"]
            row["baseline_estimate_{}".format(model)] = item["baseline_estimate"]
            n_live_total += int(item["n_live"])
            baseline_available = baseline_available or (
                item["baseline_estimate"] == item["baseline_estimate"])
            if item["families_missing"]:
                missing_all |= set(str(item["families_missing"]).split("+"))
            if item["families_not_applicable"]:
                not_applicable_all |= set(
                    str(item["families_not_applicable"]).split("+"))

        # --- 准则二：活动量调整的衰减 ---
        att = attenuation[attenuation["quantity"] == quantity]
        if len(att):
            item = att.iloc[0]
            row["baseline_attenuation"] = item["baseline_attenuation"]
            row["attenuation_median"] = item["attenuation_median"]
            row["attenuation_p10"] = item["attenuation_p10"]
            row["attenuation_p90"] = item["attenuation_p90"]
            row["n_variant_pairs"] = int(item["n_variant_pairs"])
            row["n_matched_pairs"] = int(item["n_matched_pairs"])
            row["n_pairs_attenuation_undefined"] = int(
                item["n_pairs_attenuation_undefined"])
            row["n_sign_flip_M0_to_M1"] = int(item["n_sign_flip_M0_to_M1"])

        # --- 准则三：少数账号 / 用户群 / 词表 / 月份 ---
        n_exceeding_units = 0
        for unit, name in NAMED_UNITS.items():
            sub = influence[(influence["quantity"] == quantity)
                            & (influence["influence_unit"] == unit)
                            # **必须按 layers 过滤**：准则一与准则四都是逐层
                            # 取的，如果这里不过滤，一个只在 M2 上跑的
                            # §13.9 变体（user_type 是唯一产出 M2 的族）会
                            # 把它在受限样本上的偏移塞进一行"其余数字全是
                            # M0/M1"的记录里——两个数字不来自同一个样本，
                            # 摆在一行里读只会得出错的结论。
                            & (influence["model"].isin(list(layers)))]
            sub = sub[sub["max_abs_relative_shift"].notna()]
            if len(sub):
                # 在 layers 之内跨层取最大：一个量在 M0 或 M1 任一层上被单个
                # 账号推开，都是 §13.10 第三条准则要报告的事。哪一层推得最远
                # 写进 worst_*_layer，读者不必反查。
                position = int(np.argmax(sub["max_abs_relative_shift"].values))
                worst = sub.iloc[position]
                row["max_relative_shift_{}".format(name)] = float(
                    worst["max_abs_relative_shift"])
                row["worst_{}_variant".format(name)] = worst["worst_variant_label"]
                row["worst_{}_layer".format(name)] = worst["model"]
                if float(worst["max_abs_relative_shift"]) > float(threshold):
                    n_exceeding_units += 1
            else:
                row["max_relative_shift_{}".format(name)] = np.nan
                row["worst_{}_variant".format(name)] = None
                row["worst_{}_layer".format(name)] = None
        row["influence_threshold"] = float(threshold)
        row["n_units_exceeding_threshold"] = int(n_exceeding_units)

        # --- 准则四：区间还在不在 0 的一侧（did_* 两行即"交互存活") ---
        for model in layers:
            share, n_live = _share_ci_excludes_zero(pool, quantity, model)
            row["share_ci_excludes_zero_{}".format(model)] = share
            row["n_ci_evaluable_{}".format(model)] = int(n_live)
        row["is_interaction_quantity"] = quantity in ("did_entry", "did_topical")

        # --- 测全了没有：一个标记，不是一个裁定 ---
        if not baseline_available:
            completeness = COMPLETENESS_NO_BASELINE
        elif n_live_total == 0:
            completeness = COMPLETENESS_NO_LIVE
        elif missing_all:
            completeness = COMPLETENESS_MISSING_FAMILIES
        else:
            completeness = COMPLETENESS_COMPLETE
        row["completeness"] = completeness
        row["families_missing"] = "+".join(sorted(missing_all)) or None
        row["families_not_applicable"] = "+".join(sorted(not_applicable_all)) or None
        row["n_incomplete_variants"] = int(len(incomplete))

        # --- §13.9 的指路牌：**不是**第五条准则 ---
        # M2 不进上面任何一条准则（理由见本函数 docstring 与 §11.4），但
        # "加进账号画像控制之后结论还在不在"是这篇论文欠读者的一个交代，
        # judge 不能对它只字不提。因此这里只放一个指路牌：M2 上有几个变体、
        # 它们的方向一致率是多少、去哪张表看逐行的细节。它与同一行里的
        # M0/M1 数字**不来自同一个样本**，不能并排读——这句话写在 note 里。
        m2 = direction[(direction["quantity"] == quantity)
                       & (direction["model"] == M2_LAYER)]
        if len(m2):
            row["n_M2_variants"] = int(m2["n_live"].iloc[0])
            row["m2_direction_share_profile_family"] = m2["share_agree"].iloc[0]
        else:
            row["n_M2_variants"] = 0
            row["m2_direction_share_profile_family"] = np.nan
        row["m2_pointer"] = M2_POINTER_NOTE

        row["note"] = JUDGE_NOTE
        rows.append(row)

    out = pd.DataFrame(rows)
    # `is_interaction_quantity` 是一条身份事实，不是裁定；其余布尔列一律
    # 不允许存在（测试逐列扫描）。这里把它转成字符串，彻底不留布尔裁定的
    # 形状。
    out["is_interaction_quantity"] = out["is_interaction_quantity"].map(
        {True: "yes", False: "no"})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def unlisted_result_files(directory):
    """主结果目录里既不在次要清单、也没被认领的 parquet

    SECONDARY_RESULT_FILES 是一份**手写的**清单。以后有人往
    analysis_data/results/ 里加一张新的次要分析表而忘了登记，它就会静悄悄
    地不进 FDR——一个漏做的多重比较校正不会报错，只会让一批 p 值看起来
    比它们该有的样子更好看。因此这里主动扫一遍目录，把没登记的文件点名
    警告出来，并写进 manifest，让"漏了一张表"这件事至少是可见的。
    """
    if not os.path.isdir(directory):
        return []
    known = set(SECONDARY_RESULT_FILES)
    return sorted(
        name for name in os.listdir(directory)
        if name.endswith(".parquet") and name not in known
    )


def _read_secondary_results(directory):
    """读主结果层里可能装着次要分析的结果表（只读，读不到就跳过）"""
    unlisted = unlisted_result_files(directory)
    if unlisted:
        print("警告: {} 里有 {} 张结果表不在 SECONDARY_RESULT_FILES 清单上，"
              "**不会进 FDR**: {}。如果其中有次要分析，请把文件名登记进"
              "synthesis.SECONDARY_RESULT_FILES".format(
                  directory, len(unlisted), "+".join(unlisted)))
    frames = []
    used = []
    for name in SECONDARY_RESULT_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        try:
            frame = pd.read_parquet(path, engine="pyarrow",
                                    columns=list(su.RESULT_SCHEMA))
        except Exception as exc:  # noqa: BLE001 —— 读不动就说读不动
            print("警告: {} 读取失败（{}: {}），跳过".format(
                name, type(exc).__name__, exc))
            continue
        frame = frame.copy()
        frame["source_file"] = name
        frames.append(frame)
        used.append(name)
    if not frames:
        print("警告: 主结果层里一张次要分析表都没读到，FDR 表为空")
        return pd.DataFrame(columns=list(su.RESULT_SCHEMA) + ["source_file"]), used
    return pd.concat(frames, ignore_index=True), used


def build(year=config.YEAR, threshold=DEFAULT_INFLUENCE_THRESHOLD, alpha=0.05,
          allow_recompute=True):
    """读五个族的结果 -> 四条准则 + FDR + 规格曲线数据 -> 落盘 + manifest"""
    os.makedirs(robustness_dir(), exist_ok=True)
    df = load_all(year, allow_recompute=allow_recompute)
    source = str(df["baseline_source"].iloc[0]) if len(df) else None

    tables = OrderedDict([
        ("synthesis", judge(df, threshold=threshold)),
        ("synthesis_direction", direction_consistency(df)),
        ("synthesis_attenuation", activity_attenuation(df)),
        ("synthesis_influence", influence_summary(df, threshold=threshold)),
        ("synthesis_specification_curve", specification_curve_data(df)),
        ("synthesis_incomplete_variants", incomplete_variants(df)),
        ("synthesis_calibration_pairs", calibration_pairs(df)),
    ])

    secondary, secondary_files = _read_secondary_results(results_dir())
    fdr = apply_fdr(secondary, alpha=alpha)
    tables["synthesis_fdr"] = fdr

    paths = OrderedDict()
    for name, frame in tables.items():
        path = output_path(name)
        frame.to_parquet(path, engine="pyarrow", index=False)
        print("已保存: {}（{} 行）".format(path, len(frame)))
        paths[name] = path

    pool = variant_pool(df)
    judged = tables["synthesis"]
    manifest = config.build_manifest(
        step="robustness_synthesis_{}".format(year),
        inputs=[os.path.join("robustness", filename)
                for filename in FAMILY_FILES.values()]
        + [os.path.join("results", name) for name in secondary_files],
        params={
            "year": year,
            # "和什么比的"——这是本层最该被引用的一条参数
            "baseline_source": source,
            "influence_threshold": float(threshold),
            "fdr_alpha": float(alpha),
            "fdr_scope": "secondary_analyses_only;"
                         "the_six_prespecified_quantities_are_never_corrected(11.3)",
            # 目录里没被 FDR 认领的结果表：漏登记一张次要分析表不会报错，
            # 只会让一批 p 值看起来更好看，所以把它记进 manifest
            "result_files_not_in_fdr_scope": unlisted_result_files(results_dir()),
            "expected_variant_families": list(EXPECTED_VARIANT_FAMILIES),
            "paired_families_excluded_from_the_pool": list(PAIRED_FAMILIES),
            "not_applicable_note_markers": list(NOT_APPLICABLE_NOTE_MARKERS),
            "layers": list(_layers_of(pool)),
            "emits_boolean_robustness_verdict": False,
        },
        counts={
            "quantities": len(harness.QUANTITIES),
            "rows_loaded": int(len(df)),
            "variant_rows": int(len(pool)),
            "live_variant_rows": int(pool["estimate"].notna().sum()),
            "variant_labels": int(pool["variant_label"].nunique()),
            "incomplete_variants": int(len(tables["synthesis_incomplete_variants"])),
            "calibration_pairs": int(len(tables["synthesis_calibration_pairs"])),
            "fdr_n_tested": int(fdr["fdr_n_tested"].iloc[0]) if len(fdr) else 0,
            "completeness": {
                str(value): int(count)
                for value, count in judged["completeness"].value_counts().items()
            },
        },
    )
    config.write_manifest(manifest, manifest_dir(year))
    return paths


if __name__ == "__main__":
    fire.Fire({
        "build": build,
        "load_all": load_all,
        "direction_consistency": direction_consistency,
        "activity_attenuation": activity_attenuation,
        "influence_summary": influence_summary,
        "specification_curve": specification_curve_data,
        "judge": judge,
    })
