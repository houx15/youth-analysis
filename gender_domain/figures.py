"""
§12 论文图：只读 analysis_data/figure_data/，只写 figures/。

这一层跑在本地笔记本上，不在服务器上：分析数据搬不动，但图要反复改。
服务器那边由 export_figure_data.py 把结果表和三份分布抽样写成一小撮
parquet，下载下来之后本模块画图，与旧流水线 export_viz_data.py /
visualize.py 的分工完全一致。

本模块只做"读一个数、画在图上"，不做任何估计：图里出现的每一个点、
每一条区间都必须已经在结果表里。三条容易踩的坑写在这里，具体实现见
各函数的 docstring：

1. **domain 有三个取值，不是两个**。models_interaction 的差中差写在
   domain="both" 的行上——那是全篇的核心结果。任何按 domain 分面/分组的
   代码都必须显式声明只画 public/celebrity（domain_facets），跨域行单独
   取出来（did_rows），否则要么多出一个空面板，要么最重要的一行被
   groupby 静默吞掉。
2. **model 里塞了变体，不只有层**。Tasks 6/7 把变体标签折进了 model
   （"M1/share_ge_0.1"、"record_level/male/max=720h"、"month=03"），
   所以筛选一律用精确匹配或显式前缀解析，绝不假设 model ∈ {M0,M1,M2}。
3. **NaN 是拟合失败留下的痕迹，不是"没有这一行"**。结果表里估计值为 NaN
   的行带着 note 说明原因。图不许把它们静默跳过：画不出点，就在图上写出
   缺了哪一格、note 是什么。

配色与命名沿用项目既有约定：男 #20AEE6、女 #ff7333，文件名日期前缀，
中日韩字体列表与 basic_text_analyzer.py 一致。

用法（本地）：
  python3 -m gender_domain.figures all --year=2020
  python3 -m gender_domain.figures fig3 --year=2020
"""

import os
import re
from datetime import datetime

import fire
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")          # 只出 PDF，不开窗口；无头环境也能跑
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gender_domain import config  # noqa: E402
from gender_domain import stats_utils as su  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

FIG_DIR = "figures"
FIGURE_DATA_DIRNAME = "figure_data"

MALE_COLOR = "#20AEE6"
FEMALE_COLOR = "#ff7333"
GENDER_COLORS = {"male": MALE_COLOR, "female": FEMALE_COLOR,
                 "m": MALE_COLOR, "f": FEMALE_COLOR}
GENDER_MARKERS = {"male": "o", "female": "^", "m": "o", "f": "^"}
GENDER_LABELS = {"male": "Male", "female": "Female", "m": "Male", "f": "Female"}
GENDER_ORDER = ("male", "female")
DIST_GENDER_ORDER = ("m", "f")

DOMAIN_ORDER = ("public", "celebrity")
DOMAIN_BOTH = "both"
DOMAIN_LABELS = {"public": "Public affairs", "celebrity": "Celebrity culture",
                 "both": "Cross-domain"}
MODEL_LAYERS = ("M0", "M1", "M2")
LAYER_MARKERS = {"M0": "o", "M1": "s", "M2": "D"}
LAYER_LABELS = {"M0": "M0 (gender only)", "M1": "M1 (+ activity)",
                "M2": "M2 (+ profile)"}
HEADLINE_LAYER = "M1"

TERM_DID = "gender_male_x_public_did"
TERM_GAP = {"public": "gender_male_gap_public",
            "celebrity": "gender_male_gap_celebrity"}
TERM_AME = "gender_male"

PARTICIPATION_TYPES = ("neither", "source_only", "content_only", "both")
PARTICIPATION_LABELS = {"neither": "Neither", "source_only": "Source only",
                        "content_only": "Content only", "both": "Both"}
PARTICIPATION_COLORS = ("#d9d9d9", "#9ecae1", "#fdd0a2", "#7f7f7f")
CONTENT_VARIANT = "any_hit"        # §7.1 主口径；阈值变体不画进正文图

COMBO_CATEGORIES = ("neither", "public_only", "celebrity_only", "both")
COMBO_LABELS = {"neither": "Neither domain", "public_only": "Public only",
                "celebrity_only": "Celebrity only", "both": "Both domains"}

# 附录分布
DIST_FILES = {
    "retweet_count": "dist_retweet_count.parquet",
    "delay_hours": "dist_delay_hours.parquet",
    "char_density": "dist_char_density.parquet",
}
SUMMARY_FILE = "distribution_summary.parquet"
COMBO_FILE = "combo_distribution.parquet"
DIST_COLUMNS = ["measure", "gender", "domain", "value"]
DIST_LABELS = {
    "retweet_count": "Source retweet count (per user, > 0 shown on log axis)",
    "delay_hours": "Retweet delay (hours, log axis)",
    "char_density": "User-level character density (> 0 shown on log axis)",
}
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def quantile_column(level, kind):
    """分位数列名，与 export_figure_data.quantile_column 必须完全一致

    两个模块各写一份看似冗余，但 figures 是本地跑的、export 是服务器跑的，
    本地这边不该为了一个字符串格式去 import 一个会去读服务器路径的模块。
    列名一旦不一致，测试里的 quantile_column 断言会立刻失败。
    """
    return "q{:g}_{}".format(level * 100, kind)


RESULT_COLUMNS = list(su.RESULT_SCHEMA)
TABLE2_COLUMNS = RESULT_COLUMNS + ["denominator"]

# 图 1 / 图 3 的四个核心结果，以及各自的主分母口径（§13.1 的其它分母
# 只进稳健性表，不进正文图；正文图必须写清自己用的是哪个分母）
CORE_CELLS = (("source_entered", "public"), ("source_entered", "celebrity"),
              ("topical_share", "public"), ("topical_share", "celebrity"))
PRIMARY_DENOMINATOR = {
    "source_entered": "all_same_gender_users",
    "topical_share": "users_with_expressive_posts",
}
DENOMINATOR_LABELS = {
    "all_same_gender_users": "denominator: all users of that gender",
    "retweeters_only": "denominator: users with ≥1 retweet",
    "users_with_expressive_posts": "denominator: users with ≥1 expressive post",
    "users_with_retweets": "denominator: users with ≥1 retweet",
    "users_with_active_months": "denominator: users with ≥1 active month",
}
OUTCOME_LABELS = {
    "source_entered": "Entered an institutional / celebrity source",
    "topical_share": "Share of expressive posts on the domain's topics",
    "source_share": "Domain retweets as a share of all retweets",
    "source_month_share": "Share of active months with domain participation",
}

# 异常月份的判定阈值：稳健 z 分数 |x - median| / (1.4826 * MAD) 超过它就标记。
# 3 是事前固定的常数，不按图形调整——见 fig6_monthly 的 docstring。
UNUSUAL_MONTH_MAD = 3.0

_MONTH_PATTERN = re.compile(r"^month=(\d{2})$")


# ---------------------------------------------------------------------------
# 读文件与保存
# ---------------------------------------------------------------------------

def figure_data_dir():
    """默认的图数据目录：analysis_data/figure_data/（下载到本地后的位置）"""
    return os.path.join(config.OUTPUT_DIR, FIGURE_DATA_DIRNAME)


def _read(data_dir, name, columns):
    """读一份图数据；缺文件必须当场报错，绝不画一张空图

    空坐标轴是这层最危险的失败模式：图看起来"画出来了"，只是里面没有点，
    很容易被当成"这一格确实没有数据"。所以缺文件一律抛异常，并把下一步
    该做什么写进报错信息。
    """
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到图数据文件 {}。请先在服务器上运行 "
            "`python -m gender_domain.export_figure_data build --year=YYYY`，"
            "再把 analysis_data/figure_data/ 整个目录下载到本地。".format(path)
        )
    # 列集合不固定的表（分布汇总表的分位数列随 QUANTILE_LEVELS 变）先读
    # schema 拿到列名，仍然显式传 columns=，不走"默认读全部"
    if columns is None:
        columns = pq.ParquetFile(path).schema.names
    return pd.read_parquet(path, columns=columns)


def _date_prefix():
    return datetime.now().strftime("%Y%m%d")


def _save_fig(fig, name, fig_dir):
    """按 {YYYYMMDD}_{name}.pdf 保存，与 visualize.py 同一约定"""
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, "{}_{}.pdf".format(_date_prefix(), name))
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  已保存: {}".format(path))
    return path


# ---------------------------------------------------------------------------
# 选行工具：全部是显式筛选，没有一处 groupby/dropna
# ---------------------------------------------------------------------------

def select(frame, **filters):
    """按列值精确筛选。传 None 表示这一列不筛"""
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        if value is None:
            continue
        mask &= frame[column].astype("object") == value
    return frame[mask]


def domain_facets(frame):
    """按领域分面时该画哪几个面板——永远只有 public / celebrity

    跨域行（domain="both"）不是"第三个领域"，它是两个领域之差。
    如果这里图省事写成 sorted(frame["domain"].unique())，图 3 会平白
    多出一个 "both" 面板，而那一格里其实只有差中差一个数字。
    """
    present = set(frame["domain"].astype("object"))
    return [d for d in DOMAIN_ORDER if d in present]


def did_rows(frame, outcome=None, model=None):
    """取出差中差行（domain="both" 且 term=TERM_DID）

    单独一个函数，是为了让"这一行必须显式取"这件事有一个可测的落点：
    §6.6 的差中差是全文的核心结果，一旦被按领域分组的代码丢掉，图 3
    会安静地少掉论文的头条。
    """
    rows = select(frame, domain=DOMAIN_BOTH, term=TERM_DID)
    rows = select(rows, outcome=outcome, model=model)
    return rows


def did_annotation(frame, outcome, model=HEADLINE_LAYER):
    """差中差的标注文字：有估计就写估计与 95% 区间，没有就写缺失原因"""
    rows = did_rows(frame, outcome=outcome, model=model)
    if rows.empty:
        return "DiD ({}): 结果表里没有这一行".format(model)
    row = rows.iloc[0]
    if pd.isna(row["estimate"]):
        return "DiD ({}): 无估计（{}）".format(model, row["note"] or "note 缺失")
    return "DiD ({}) = {:+.3f}  95% CI [{:+.3f}, {:+.3f}]".format(
        model, row["estimate"], row["ci_low"], row["ci_high"])


def missing_estimate_rows(frame):
    """估计值为 NaN 的行——拟合失败留下的痕迹，图必须把它们标出来"""
    return frame[pd.to_numeric(frame["estimate"], errors="coerce").isna()]


def missing_estimate_labels(frame):
    """把 NaN 行写成一行行可以直接印在图上的说明"""
    labels = []
    for _, row in missing_estimate_rows(frame).iterrows():
        labels.append("{}/{}/{}/{}: {}".format(
            row["outcome"], row["domain"], row["model"], row["term"],
            row["note"] or "no note"))
    return labels


def _annotate_missing(ax, labels, prefix="missing estimates"):
    """把缺失说明写在坐标轴左下角，最多列 4 条，其余折叠成计数"""
    if not labels:
        return
    shown = labels[:4]
    text = "{}:\n".format(prefix) + "\n".join("· " + s for s in shown)
    if len(labels) > len(shown):
        text += "\n· … {} more".format(len(labels) - len(shown))
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=6,
            color="#b22222", va="bottom", ha="left")


def monthly_rows(frame, outcome, domain):
    """取出按月的描述行，并把 model="month=NN" 解析成整数 month 列

    monthly_rates.parquet 里同时躺着月份固定效应的行
    （model="M0/month_fixed_effects"），它们不是按月的点，必须按 model
    的形状排除，而不是"除了 M0/M1/M2 之外都当成月份"。
    """
    rows = select(frame, outcome=outcome, domain=domain).copy()
    months = rows["model"].astype(str).str.extract(_MONTH_PATTERN, expand=False)
    rows = rows[months.notna()].copy()
    rows["month"] = months[months.notna()].astype(int)
    return rows.sort_values("month")


def flag_unusual_months(values, n_mad=UNUSUAL_MONTH_MAD):
    """标记异常月份：|x - median| > n_mad × 1.4826 × MAD

    规则是事前固定的，且只依赖数据本身的稳健离散度，不依赖任何事件列表：
    §12.6 明确禁止"看图之后再去挑一个事件来解释这个峰"。因此这里用中位数
    与 MAD（而不是均值与标准差）——离群点会把标准差自己撑大，用它做阈值
    等于让异常点参与决定"什么算异常"。1.4826 是让 MAD 在正态下与标准差
    同尺度的常数。MAD 恰为 0 时（多数月份取值完全相同）退化为"凡是与
    中位数不同的月份都标记"。NaN 月份不标记。
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    flags = np.zeros(len(values), dtype=bool)
    if finite.sum() < 3:
        return flags
    median = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - median)))
    scaled = 1.4826 * mad
    if scaled == 0:
        flags[finite] = values[finite] != median
        return flags
    flags[finite] = np.abs(values[finite] - median) > n_mad * scaled
    return flags


def _n_label(row):
    """样本量标注：n_obs 与流失原因都写出来"""
    n_obs = row.get("n_obs")
    if n_obs is None or (isinstance(n_obs, float) and np.isnan(n_obs)):
        return ""
    return "n={:,}".format(int(n_obs))


def _errorbar(ax, x, y, low, high, color, marker, label=None, horizontal=True):
    """画一个点 + 95% 区间；估计值缺失时不画点（缺失另行标注）"""
    if pd.isna(x if horizontal else y):
        return
    if horizontal:
        err = [[x - low if pd.notna(low) else 0.0],
               [high - x if pd.notna(high) else 0.0]]
        ax.errorbar([x], [y], xerr=err, fmt=marker, color=color, markersize=6,
                    capsize=3, linewidth=1.2, label=label)
    else:
        err = [[y - low if pd.notna(low) else 0.0],
               [high - y if pd.notna(high) else 0.0]]
        ax.errorbar([x], [y], yerr=err, fmt=marker, color=color, markersize=6,
                    capsize=3, linewidth=1.2, label=label)


# ---------------------------------------------------------------------------
# 图 1：四项核心参与结果（§12.1）
# ---------------------------------------------------------------------------

def fig1_core_outcomes(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """点区间图：四项核心结果的男女原始水平（§12.1）

    刻意不用柱状图：柱子从 0 起画会把"0.30 与 0.22"的差距视觉上放大成
    柱高之比，而这四个量本来就都是 0–1 之间的比例，读者要看的是点的位置
    与区间的宽度。

    两个面板按 scale 分开：左边是进入概率（probability），右边是内容比例
    （proportion）。它们都落在 0–1 区间里，但不是同一个量，共用一根横轴
    会暗示"公共事务进入率 0.30 比公共事务议题比例 0.09 高"这种没有意义的
    比较。每个面板的横轴标签写明自己的分母。
    """
    data_dir = data_dir or figure_data_dir()
    table2 = _read(data_dir, "table2_raw_gender_gaps.parquet", TABLE2_COLUMNS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = (("source_entered", "probability"), ("topical_share", "proportion"))
    for ax, (outcome, _scale) in zip(axes, panels):
        denominator = PRIMARY_DENOMINATOR[outcome]
        rows = select(table2, outcome=outcome, model="raw", denominator=denominator)
        y_ticks, y_labels, missing = [], [], []
        for i, domain in enumerate(DOMAIN_ORDER):
            y_ticks.append(i)
            y_labels.append(DOMAIN_LABELS[domain])
            for j, gender in enumerate(GENDER_ORDER):
                cell = select(rows, domain=domain, term=gender)
                if cell.empty:
                    missing.append("{}/{}: row absent".format(domain, gender))
                    continue
                row = cell.iloc[0]
                y = i + (0.14 if gender == "male" else -0.14)
                if pd.isna(row["estimate"]):
                    missing.append("{}/{}: {}".format(
                        domain, gender, row["note"] or "no note"))
                    continue
                _errorbar(ax, row["estimate"], y, row["ci_low"], row["ci_high"],
                          GENDER_COLORS[gender], GENDER_MARKERS[gender],
                          label=GENDER_LABELS[gender] if i == 0 else None)
                ax.annotate(_n_label(row), (row["estimate"], y),
                            textcoords="offset points", xytext=(6, 6), fontsize=6,
                            color="#555555")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(-0.6, len(DOMAIN_ORDER) - 0.4)
        ax.set_xlabel("{}\n({}, 95% CI)".format(
            OUTCOME_LABELS[outcome], DENOMINATOR_LABELS[denominator]), fontsize=8)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=9)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7, loc="lower right")
        _annotate_missing(ax, missing)

    fig.suptitle("Figure 1. Core participation outcomes by gender, {}".format(year),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save_fig(fig, "fig1_core_outcomes", fig_dir)


# ---------------------------------------------------------------------------
# 图 2：调整后的性别差异（§12.2）
# ---------------------------------------------------------------------------

def fig2_adjusted_effects(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """森林图：性别平均边际效应在 M0 → M1 → M2 之间怎么移动（§12.2）

    这张图的全部意义在于让读者看着系数缩水：M0 只有性别，M1 加了一般
    活动量，M2 再加画像。三层并排画在同一行上，缩水（或不缩水）一眼可见。

    横轴按 scale 分成两个面板：概率尺度的进入模型与比例尺度的内容/持续
    模型不共用一根轴。整层拟合失败时（估计值 NaN）不画点，改在面板里
    列出缺了哪一层、note 是什么——安静地少画一个点会被读成"这一层与
    上一层重合"。
    """
    data_dir = data_dir or figure_data_dir()
    entry = _read(data_dir, "models_entry.parquet", RESULT_COLUMNS)
    share = _read(data_dir, "models_share.parquet", RESULT_COLUMNS)
    persistence = _read(data_dir, "models_persistence.parquet", RESULT_COLUMNS)

    panels = [
        ("probability", "Gender AME on entry probability (male − female)",
         [(entry, "source_entered", d) for d in DOMAIN_ORDER]),
        ("proportion", "Gender AME on shares (male − female)",
         [(share, "topical_share", d) for d in DOMAIN_ORDER]
         + [(persistence, "source_month_share", d) for d in DOMAIN_ORDER]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (_scale, xlabel, specs) in zip(axes, panels):
        y_ticks, y_labels, missing = [], [], []
        y = 0
        for frame, outcome, domain in specs:
            rows = select(frame, outcome=outcome, domain=domain, term=TERM_AME)
            y_ticks.append(y)
            y_labels.append("{}\n{}".format(OUTCOME_LABELS[outcome].split(",")[0],
                                            DOMAIN_LABELS[domain]))
            for k, layer in enumerate(MODEL_LAYERS):
                cell = select(rows, model=layer)
                offset = 0.22 * (1 - k)
                if cell.empty:
                    missing.append("{}/{}/{}: row absent".format(
                        outcome, domain, layer))
                    continue
                row = cell.iloc[0]
                if pd.isna(row["estimate"]):
                    missing.append("{}/{}/{}: {}".format(
                        outcome, domain, layer, row["note"] or "no note"))
                    continue
                color = MALE_COLOR if row["estimate"] >= 0 else FEMALE_COLOR
                _errorbar(ax, row["estimate"], y + offset, row["ci_low"],
                          row["ci_high"], color, LAYER_MARKERS[layer])
                ax.annotate(layer, (row["estimate"], y + offset),
                            textcoords="offset points", xytext=(7, -2),
                            fontsize=6, color="#555555")
            y += 1
        ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_ylim(-0.6, len(specs) - 0.4)
        ax.set_xlabel(xlabel + "\n(95% CI; denominators as in Table 2)", fontsize=8)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        _annotate_missing(ax, missing)

    handles = [Line2D([], [], color="#555555", marker=LAYER_MARKERS[layer],
                      linestyle="none", label=LAYER_LABELS[layer])
               for layer in MODEL_LAYERS]
    handles += [Line2D([], [], color=MALE_COLOR, marker="s", linestyle="none",
                       label="Male higher"),
                Line2D([], [], color=FEMALE_COLOR, marker="s", linestyle="none",
                       label="Female higher")]
    fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=5,
               frameon=False)
    fig.suptitle(
        "Figure 2. Gender average marginal effects across model layers, {}".format(
            year), fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    return _save_fig(fig, "fig2_adjusted_effects", fig_dir)


# ---------------------------------------------------------------------------
# 图 3：Gender × Domain 交互（§12.3）
# ---------------------------------------------------------------------------

def fig3_interaction(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """四个 性别 × 领域 格子的取值，并显式标出差中差（§12.3）

    两处口径必须写清楚，否则这张图会被误读：

    1. 四个格子画的是**观测**比例（表 2，各自的主分母口径），不是模型
       预测值。结果层的 interaction_gender_domain.parquet 只写出了差与
       差中差，没有写出四个格子的模型预测水平；图这一层不许自己反推
       （拿观测的女性水平加上调整后的性别差，等于在图里造一个结果表里
       没有的数字）。所以格子用观测值，调整后的量以标注形式给出。
    2. 差中差与两个领域内的调整后性别差来自 GEE 交互模型，写在图右侧的
       标注框里，并注明层次（M0/M1/M2）。差中差那一行的 domain 是
       "both"，用 did_rows 显式取出——它不属于任何一个面板。
    """
    data_dir = data_dir or figure_data_dir()
    table2 = _read(data_dir, "table2_raw_gender_gaps.parquet", TABLE2_COLUMNS)
    inter = _read(data_dir, "interaction_gender_domain.parquet", RESULT_COLUMNS)

    outcomes = ("source_entered", "topical_share")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, outcome in zip(axes, outcomes):
        denominator = PRIMARY_DENOMINATOR[outcome]
        observed = select(table2, outcome=outcome, model="raw",
                          denominator=denominator)
        facets = domain_facets(observed) or list(DOMAIN_ORDER)
        missing = []
        for i, domain in enumerate(facets):
            cell_values = {}
            for gender in GENDER_ORDER:
                cell = select(observed, domain=domain, term=gender)
                if cell.empty or pd.isna(cell.iloc[0]["estimate"]):
                    note = "row absent" if cell.empty else (
                        cell.iloc[0]["note"] or "no note")
                    missing.append("{}/{}: {}".format(domain, gender, note))
                    continue
                row = cell.iloc[0]
                cell_values[gender] = row["estimate"]
                _errorbar(ax, i + (0.12 if gender == "male" else -0.12),
                          row["estimate"], row["ci_low"], row["ci_high"],
                          GENDER_COLORS[gender], GENDER_MARKERS[gender],
                          label=GENDER_LABELS[gender] if i == 0 else None,
                          horizontal=False)
                ax.annotate(_n_label(row),
                            (i + (0.12 if gender == "male" else -0.12),
                             row["estimate"]), textcoords="offset points",
                            xytext=(6, 4), fontsize=6, color="#555555")
            if len(cell_values) == 2:
                ax.plot([i - 0.12, i + 0.12],
                        [cell_values["female"], cell_values["male"]],
                        color="#999999", linewidth=0.8, linestyle=":")
                ax.annotate("gap {:+.3f}".format(
                    cell_values["male"] - cell_values["female"]),
                    (i, (cell_values["male"] + cell_values["female"]) / 2),
                    textcoords="offset points", xytext=(12, 0), fontsize=7,
                    color="#333333")

        # 调整后的量：两个领域内的性别差 + 跨域差中差（domain="both"）
        lines = ["Model-adjusted (GEE, cluster-robust by user):"]
        for layer in MODEL_LAYERS:
            lines.append("  " + did_annotation(inter, outcome, layer))
        for domain in DOMAIN_ORDER:
            gap = select(inter, outcome=outcome, domain=domain,
                         term=TERM_GAP[domain], model=HEADLINE_LAYER)
            if gap.empty:
                continue
            row = gap.iloc[0]
            if pd.isna(row["estimate"]):
                lines.append("  {} gap ({}): 无估计（{}）".format(
                    DOMAIN_LABELS[domain], HEADLINE_LAYER,
                    row["note"] or "no note"))
            else:
                lines.append("  {} gap ({}) = {:+.3f} [{:+.3f}, {:+.3f}]".format(
                    DOMAIN_LABELS[domain], HEADLINE_LAYER, row["estimate"],
                    row["ci_low"], row["ci_high"]))
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="#f5f5f5",
                          edgecolor="#cccccc", linewidth=0.5))

        ax.set_xticks(range(len(facets)))
        ax.set_xticklabels([DOMAIN_LABELS[d] for d in facets])
        ax.set_xlim(-0.6, len(facets) - 0.4)
        ax.set_ylabel("{}\n(observed, {}, 95% CI)".format(
            OUTCOME_LABELS[outcome], DENOMINATOR_LABELS[denominator]), fontsize=7)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7, loc="lower right")
        _annotate_missing(ax, missing)

    fig.suptitle(
        "Figure 3. Gender × domain cells with the difference-in-differences, "
        "{}".format(year), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "fig3_interaction", fig_dir)


# ---------------------------------------------------------------------------
# 图 4：来源传播与内容表达（§12.4）
# ---------------------------------------------------------------------------

def fig4_source_content(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """四类参与（均未 / 只来源 / 只内容 / 两者）的 100% 堆积条形图（§12.4）

    §12.4 明确禁止 Sankey：桑基图的流向会被读成"先有来源传播、再转化为
    内容表达"，而这里的四类只是同一时点上的交叉分类，没有任何时间或
    因果方向。堆积条形只表达"占比构成"，正好是要说的东西。

    只画主口径变体 any_hit（model 列里还躺着 share_ge_0.05/0.1/0.2 三个
    阈值变体，它们属于稳健性表）。某一格缺失或为 NaN 时不补 0，改在图上
    标注，并把该性别各段之和写出来——四段之和明显不等于 1 就说明有格子
    没画上，这一点必须让读者直接看见。
    """
    data_dir = data_dir or figure_data_dir()
    decomposition = _read(data_dir, "decomposition_source_content.parquet",
                          RESULT_COLUMNS)
    rows = select(decomposition, outcome="participation_type",
                  model=CONTENT_VARIANT)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars, labels, missing = [], [], []
    for domain in DOMAIN_ORDER:
        for gender in GENDER_ORDER:
            values, n_obs = [], None
            for ptype in PARTICIPATION_TYPES:
                cell = select(rows, domain=domain,
                              term="{}_{}".format(ptype, gender))
                if cell.empty:
                    missing.append("{}/{}/{}: row absent".format(
                        domain, gender, ptype))
                    values.append(np.nan)
                    continue
                row = cell.iloc[0]
                n_obs = row["n_obs"]
                if pd.isna(row["estimate"]):
                    missing.append("{}/{}/{}: {}".format(
                        domain, gender, ptype, row["note"] or "no note"))
                values.append(row["estimate"])
            bars.append(values)
            labels.append("{}\n{}\n{}".format(
                DOMAIN_LABELS[domain], GENDER_LABELS[gender],
                "n={:,}".format(int(n_obs)) if n_obs is not None
                and pd.notna(n_obs) else "n=?"))

    x = np.arange(len(bars))
    bottom = np.zeros(len(bars))
    for k, ptype in enumerate(PARTICIPATION_TYPES):
        heights = np.array([b[k] for b in bars], dtype=float)
        drawn = np.where(np.isfinite(heights), heights, 0.0)
        ax.bar(x, drawn, bottom=bottom, width=0.6,
               color=PARTICIPATION_COLORS[k], edgecolor="white", linewidth=0.6,
               label=PARTICIPATION_LABELS[ptype])
        for xi, (h, b0) in enumerate(zip(drawn, bottom)):
            if h >= 0.06:
                ax.text(xi, b0 + h / 2, "{:.0%}".format(h), ha="center",
                        va="center", fontsize=7, color="#222222")
        bottom = bottom + drawn

    for xi, total in enumerate(bottom):
        if abs(total - 1.0) > 0.005:
            ax.text(xi, total + 0.01, "Σ={:.3f}".format(total), ha="center",
                    fontsize=6, color="#b22222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of users (denominator: users with a defined content "
                  "measure,\ni.e. ≥1 expressive post; variant = any_hit)",
                  fontsize=7)
    ax.legend(fontsize=7, ncol=4, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, -0.16))
    _annotate_missing(ax, missing)
    fig.suptitle(
        "Figure 4. Source diffusion and content expression, four participation "
        "types, {}".format(year), fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save_fig(fig, "fig4_source_content", fig_dir)


# ---------------------------------------------------------------------------
# 图 5：领域参与组合（§12.5）
# ---------------------------------------------------------------------------

def fig5_combinations(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """四类来源组合的性别分布 + 多项 logit 的性别边际效应（§12.5）

    左：各性别在四类组合上的观测占比（Wilson 区间），来自导出层的
    combo_distribution.parquet——表 C 已有的 source_combo 列做一次分组占比。
    右：多项 logit 各类别的性别平均边际效应（男−女，概率尺度），M0/M1/M2
    并排。这两半都写在 domain="both" 的行上：来源组合是跨领域的分类，
    因此这张图里没有按领域分面这回事。

    结果表里没有"各性别的多项模型预测概率"这一量（models_combination 只
    写出了两者之差），所以右半边画的是差，不是两条预测曲线——图注必须
    这样写，不能把左半边的观测占比说成模型预测值。
    """
    data_dir = data_dir or figure_data_dir()
    observed = _read(data_dir, COMBO_FILE, RESULT_COLUMNS)
    multinomial = _read(data_dir, "combination_multinomial.parquet", RESULT_COLUMNS)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    missing = []
    for i, category in enumerate(COMBO_CATEGORIES):
        for gender in GENDER_ORDER:
            cell = select(observed, term="{}_{}".format(category, gender))
            if cell.empty:
                missing.append("{}/{}: row absent".format(category, gender))
                continue
            row = cell.iloc[0]
            if pd.isna(row["estimate"]):
                missing.append("{}/{}: {}".format(
                    category, gender, row["note"] or "no note"))
                continue
            _errorbar(ax, row["estimate"], i + (0.14 if gender == "male" else -0.14),
                      row["ci_low"], row["ci_high"], GENDER_COLORS[gender],
                      GENDER_MARKERS[gender],
                      label=GENDER_LABELS[gender] if i == 0 else None)
            ax.annotate(_n_label(row),
                        (row["estimate"], i + (0.14 if gender == "male" else -0.14)),
                        textcoords="offset points", xytext=(6, 5), fontsize=6,
                        color="#555555")
    ax.set_yticks(range(len(COMBO_CATEGORIES)))
    ax.set_yticklabels([COMBO_LABELS[c] for c in COMBO_CATEGORIES])
    ax.set_ylim(-0.6, len(COMBO_CATEGORIES) - 0.4)
    ax.set_xlabel("Observed share of users (denominator: all users of that gender\n"
                  "with a known source combination; 95% Wilson CI)", fontsize=7)
    ax.set_title("Observed distribution", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, loc="lower right")
    _annotate_missing(ax, missing)

    ax = axes[1]
    missing = []
    combo_rows = select(multinomial, outcome="source_combo", domain=DOMAIN_BOTH)
    for i, category in enumerate(COMBO_CATEGORIES):
        for k, layer in enumerate(MODEL_LAYERS):
            cell = select(combo_rows, model=layer,
                          term="gender_male_ame_{}".format(category))
            offset = 0.22 * (1 - k)
            if cell.empty:
                missing.append("{}/{}: row absent".format(category, layer))
                continue
            row = cell.iloc[0]
            if pd.isna(row["estimate"]):
                missing.append("{}/{}: {}".format(
                    category, layer, row["note"] or "no note"))
                continue
            color = MALE_COLOR if row["estimate"] >= 0 else FEMALE_COLOR
            _errorbar(ax, row["estimate"], i + offset, row["ci_low"],
                      row["ci_high"], color, LAYER_MARKERS[layer])
            ax.annotate(layer, (row["estimate"], i + offset),
                        textcoords="offset points", xytext=(7, -2), fontsize=6,
                        color="#555555")
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(COMBO_CATEGORIES)))
    ax.set_yticklabels([COMBO_LABELS[c] for c in COMBO_CATEGORIES])
    ax.set_ylim(-0.6, len(COMBO_CATEGORIES) - 0.4)
    ax.set_xlabel("Multinomial logit gender AME (male − female), probability scale\n"
                  "(denominator: users with a known source combination; 95% CI)",
                  fontsize=7)
    ax.set_title("Model-adjusted gender difference", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    _annotate_missing(ax, missing)

    handles = [Line2D([], [], color="#555555", marker=LAYER_MARKERS[layer],
                      linestyle="none", label=LAYER_LABELS[layer])
               for layer in MODEL_LAYERS]
    fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=3,
               frameon=False)
    fig.suptitle("Figure 5. Source-combination categories by gender, {}".format(year),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    return _save_fig(fig, "fig5_combinations", fig_dir)


# ---------------------------------------------------------------------------
# 图 6：月度趋势（§12.6）
# ---------------------------------------------------------------------------

def fig6_monthly(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """两个领域的月度参与率与议题比例，并按事前规则标出异常月份（§12.6）

    **异常月份的标注规则是事前固定的，不做事后挑选。** 规则只有一条：
    某个月的取值偏离全年中位数超过 3 × 1.4826 × MAD（稳健 z 分数）就画一个
    空心圈。规则不看任何事件列表，也不因为某个月"看起来像有大事"而调整
    阈值；图上也不写任何事件名称。§12.6 允许标注异常月份，但明确禁止
    "看图之后再去挑一个事件来解释这个峰"——那是用同一批数据既生成假设
    又检验假设。要给某个月一个实质解释，必须在别处、用别的证据做。

    月份来自 monthly_rates.parquet 里 model="month=NN" 的描述行；同一份
    文件里 model="M0/month_fixed_effects" 的行是全年固定效应估计，不是
    按月的点，由 monthly_rows 按 model 形状排除。
    """
    data_dir = data_dir or figure_data_dir()
    monthly = _read(data_dir, "monthly_rates.parquet", RESULT_COLUMNS)

    outcomes = (("monthly_source_entered_rate",
                 "Monthly entry rate\n(denominator: users active that month)"),
                ("monthly_topical_share_mean",
                 "Monthly mean topical share\n(denominator: users with expressive "
                 "posts that month)"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for r, (outcome, ylabel) in enumerate(outcomes):
        for c, domain in enumerate(DOMAIN_ORDER):
            ax = axes[r, c]
            rows = monthly_rows(monthly, outcome, domain)
            missing = []
            for gender in GENDER_ORDER:
                series = select(rows, term=gender).sort_values("month")
                if series.empty:
                    missing.append("{}: rows absent".format(gender))
                    continue
                months = series["month"].to_numpy()
                values = pd.to_numeric(series["estimate"],
                                       errors="coerce").to_numpy(dtype=float)
                low = pd.to_numeric(series["ci_low"],
                                    errors="coerce").to_numpy(dtype=float)
                high = pd.to_numeric(series["ci_high"],
                                     errors="coerce").to_numpy(dtype=float)
                color = GENDER_COLORS[gender]
                ax.plot(months, values, color=color, marker=GENDER_MARKERS[gender],
                        markersize=4, linewidth=1.3, label=GENDER_LABELS[gender])
                ax.fill_between(months, low, high, color=color, alpha=0.15,
                                linewidth=0)
                flags = flag_unusual_months(values)
                if flags.any():
                    ax.scatter(months[flags], values[flags], s=90,
                               facecolors="none", edgecolors="#b22222",
                               linewidths=1.2, zorder=5)
                for month, value, row_note in zip(months, values, series["note"]):
                    if not np.isfinite(value):
                        missing.append("{} month={:02d}: {}".format(
                            gender, month, row_note or "no note"))
            ax.set_xticks(range(1, 13))
            ax.set_title("{}".format(DOMAIN_LABELS[domain]), fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=7)
            if r == 1:
                ax.set_xlabel("Month of {}".format(year), fontsize=8)
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.legend(fontsize=7, loc="best")
            _annotate_missing(ax, missing)

    fig.suptitle(
        "Figure 6. Monthly trends by gender, {} (open circles: months beyond "
        "3 × robust MAD from the yearly median;\nthe rule is fixed ex ante and no "
        "events are selected post hoc)".format(year), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save_fig(fig, "fig6_monthly", fig_dir)


# ---------------------------------------------------------------------------
# 附录图：分布（§12.7）
# ---------------------------------------------------------------------------

def _ecdf(values):
    """经验分布函数的 (x, y)，x 已排序"""
    values = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def appendix_distributions(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """附录：转发次数、转发延迟、字符 density 的 ECDF（§12.7）

    三份分布都是导出层按 性别 × 领域 抽样后的样本，不是全量：每个面板
    的标题写出该格的全量样本量与抽样点数，图注写出 seed 与每格上限，
    于是这张图可以被逐点复现。

    横轴一律取对数：三个量都是重尾的，线性轴上除了贴着 0 的一团什么都
    看不出来。对数轴画不了 0，因此每个面板单独写出 P(value = 0)——把
    零值悄悄丢掉会让 ECDF 的起点凭空抬高。
    """
    data_dir = data_dir or figure_data_dir()
    summary = _read(data_dir, SUMMARY_FILE, None)

    measures = ("retweet_count", "delay_hours", "char_density")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for r, measure in enumerate(measures):
        frame = _read(data_dir, DIST_FILES[measure], DIST_COLUMNS)
        for c, domain in enumerate(DOMAIN_ORDER):
            ax = axes[r, c]
            notes = []
            for gender in DIST_GENDER_ORDER:
                cell = frame[(frame["gender"].astype("object") == gender)
                             & (frame["domain"].astype("object") == domain)]
                values = pd.to_numeric(cell["value"],
                                       errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values) < 2:
                    notes.append("{}: <2 sampled points".format(
                        GENDER_LABELS[gender]))
                    continue
                zero_share = float(np.mean(values <= 0))
                positive = values[values > 0]
                if len(positive) < 2:
                    notes.append("{}: all sampled values are 0".format(
                        GENDER_LABELS[gender]))
                    continue
                x, y = _ecdf(positive)
                ax.plot(x, y, color=GENDER_COLORS[gender], linewidth=1.4,
                        label="{} (P(=0)={:.1%})".format(
                            GENDER_LABELS[gender], zero_share))
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel(DIST_LABELS[measure], fontsize=7)
            ax.set_ylabel("ECDF", fontsize=8)
            info = summary[(summary["measure"].astype("object") == measure)
                           & (summary["domain"].astype("object") == domain)]
            n_total = int(info["n_total"].sum()) if len(info) else 0
            n_sampled = int(info["n_sampled"].sum()) if len(info) else 0
            ax.set_title("{} — {}\n(sampled {:,} of {:,} valid observations)".format(
                DIST_LABELS[measure].split("(")[0].strip(), DOMAIN_LABELS[domain],
                n_sampled, n_total), fontsize=8)
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.legend(fontsize=7, loc="lower right")
            _annotate_missing(ax, notes, prefix="not plotted")

    seeds = "/".join(str(s) for s in sorted(set(int(s) for s in summary["seed"]))) \
        if len(summary) else "?"
    caps = "/".join("{:,}".format(c)
                    for c in sorted(set(int(c) for c in summary["cap"]))) \
        if len(summary) else "?"
    fig.suptitle(
        "Appendix Figure A1. Distributions behind the main results, {}\n"
        "(random sample, seed={}, at most {} points per gender × domain cell)".format(
            year, seeds, caps), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_fig(fig, "figA1_distributions", fig_dir)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def all(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """按顺序生成全部图，返回文件路径列表"""
    print("=" * 60)
    print("生成 {} 年的 §12 图，数据目录 {}".format(
        year, data_dir or figure_data_dir()))
    print("=" * 60)
    paths = [
        fig1_core_outcomes(year, data_dir, fig_dir),
        fig2_adjusted_effects(year, data_dir, fig_dir),
        fig3_interaction(year, data_dir, fig_dir),
        fig4_source_content(year, data_dir, fig_dir),
        fig5_combinations(year, data_dir, fig_dir),
        fig6_monthly(year, data_dir, fig_dir),
        appendix_distributions(year, data_dir, fig_dir),
    ]
    print("\n共生成 {} 张图，输出目录 {}".format(len(paths), fig_dir))
    return paths


if __name__ == "__main__":
    fire.Fire({
        "all": all,
        "fig1": fig1_core_outcomes,
        "fig2": fig2_adjusted_effects,
        "fig3": fig3_interaction,
        "fig4": fig4_source_content,
        "fig5": fig5_combinations,
        "fig6": fig6_monthly,
        "appendix": appendix_distributions,
    })
