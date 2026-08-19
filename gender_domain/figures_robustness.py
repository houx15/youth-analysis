"""§12.7 附录图（稳健性部分）：只读 analysis_data/robustness/，只写 figures/。

与 gender_domain.figures 同一分工：这一层跑在本地，不做任何估计，图里的
每一个点都必须已经在 robustness/ 的表里。区别只是数据源——正文图读
figure_data/ 的导出，本模块读综合层（synthesis_*）与各族原始结果。

设计文档 §12.7 点名要的附录图在这里各占一张：词表重采样的系数分布
（figR5）、leave-one-account-out（figR6）、不同样本与指标定义的
specification curve（figR1）、top-share 集中度（figR6 下半）。另外三张
对应 §13.10 的四条判定准则：逐族方向一致率（figR2）、少数账号/用户/词条/
月份的影响力（figR3）、控制活动量后的衰减（figR4）。

三条与正文图相同的纪律：

1. **NaN 不是"没有这一行"**。变体跑不出估计的行带着 note，图上必须写出
   缺了多少格，不能静默跳过。
2. **行池口径与逐族口径是两个数**。两个重抽样族各 200 个 replicate，按行
   汇总时它们占掉七成话语权；figR2 画的是逐族，figR1 的副标题同时写出两
   个数，谁进正文由作者定。
3. **分母写在图上**。一致率的分母是"跑出了估计的变体行数"，不是"变体总
   数"——NaN 行单独标注。

用法（本地）：
  python3 -m gender_domain.figures_robustness all --year=2020
  python3 -m gender_domain.figures_robustness figR1 --year=2020
"""

import os

import fire
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gender_domain import config  # noqa: E402
from gender_domain import figures as fg  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei",
                                          "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FIG_DIR = fg.FIG_DIR
ROBUSTNESS_DIRNAME = "robustness"

# 综合层固定报到 M0/M1 两层；M2 只有 profile 族产出，不进这批图的主口径
HEADLINE_LAYER = "M1"

QUANTITY_ORDER = ("entry_public", "entry_celebrity",
                  "topical_public", "topical_celebrity",
                  "did_entry", "did_topical")
QUANTITY_LABELS = {
    "entry_public": "Source entry\npublic affairs",
    "entry_celebrity": "Source entry\ncelebrity culture",
    "topical_public": "Topical share\npublic affairs",
    "topical_celebrity": "Topical share\ncelebrity culture",
    "did_entry": "Gender × Domain\n(entry, DiD)",
    "did_topical": "Gender × Domain\n(topical share, DiD)",
}

# FDR 图的对数纵轴地板。n=225,339 下大量 p 值小到 1e-240，照实画会把
# 坐标轴拉到 240 个数量级，唯一还看得出形状的那一段反而被压平。
P_VALUE_FLOOR = 1e-20

AGREE_COLOR = "#4c72b0"
DISAGREE_COLOR = "#b22222"
BASELINE_COLOR = "#000000"
NEUTRAL = "#999999"
BAND_COLOR = "#c7d5e8"

# 影响力单位：§13.10 第三条准则问的"结论是否由少数账号/用户/词条/月份推动"
INFLUENCE_UNITS = ("single_account", "account_set", "user_group",
                   "term_set", "month", "measurement")
INFLUENCE_LABELS = {
    "single_account": "One source account\n(leave-one-out)",
    "account_set": "Account set\n(subsets, bootstrap)",
    "user_group": "User group\n(verified, ordinary, …)",
    "term_set": "Term set\n(vocabulary rules)",
    "month": "Month\n(leave-one-month-out)",
    "measurement": "Measure / denominator",
}


# ---------------------------------------------------------------------------
# 读数
# ---------------------------------------------------------------------------

def robustness_dir():
    return os.path.join(config.OUTPUT_DIR, ROBUSTNESS_DIRNAME)


def _read(data_dir, name, columns=None):
    """读一份稳健性结果表；缺文件当场报错，绝不画一张空图

    与 figures._read 同一理由：空坐标轴是这层最危险的失败模式，看上去
    "画出来了"，只是里面没有点。
    """
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到稳健性结果文件 {}。请先在服务器上跑完 run_robustness / "
            "run_robustness_light，再跑 `python -m "
            "gender_domain.robustness.synthesis build --year=YYYY`，"
            "然后把 analysis_data/robustness/ 下载到本地。".format(path)
        )
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def dedup_variants(frame):
    """按变体身份去重，与 synthesis.drop_rerun_duplicates 同一口径

    各族的落盘是纯追加写，重跑一族会把它整份再写一遍。本模块直接读族表
    （figR5/figR6），必须和综合层用同一把尺子，否则同一批数据在两处画出
    不同的点数。
    """
    subset = [c for c in ("variant_family", "variant_label", "replicate",
                          "seed", "outcome", "domain", "model", "term")
              if c in frame.columns]
    return frame.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)


def _as_bool(series):
    """把可能含 NaN 的对象列转成 bool 数组：NaN 读作 False

    直接 .fillna(False).astype(bool) 会触发 pandas 的降级告警，而这里的
    语义很明确——"没标记"就是"不是"。
    """
    return np.array([bool(v) if v == v and v is not None else False
                     for v in series], dtype=bool)


def _pct(value):
    return "n/a" if pd.isna(value) else "{:.1f}%".format(100.0 * float(value))


def _note_missing(ax, lines, corner="top"):
    """把"这张图缺了什么"写在图上，而不是让读者以为本来就没有

    默认放左上角：本模块里带这个标注的几张图画的都是单调递增的排序曲线，
    左上角是唯一保证空着的地方。放左下角会正好压在曲线起点上。
    """
    if not lines:
        return
    y, va = (0.98, "top") if corner == "top" else (0.02, "bottom")
    ax.text(0.01, y, "\n".join(lines), transform=ax.transAxes, fontsize=5.5,
            va=va, ha="left", color="#b22222",
            bbox=dict(facecolor="white", edgecolor="#b22222", linewidth=0.4,
                      alpha=0.85, pad=2))


# ---------------------------------------------------------------------------
# 图 R1：specification curve（§12.7）
# ---------------------------------------------------------------------------

def figR1_specification_curve(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR,
                              model=HEADLINE_LAYER):
    """六个量各一格：把所有变体的估计从小到大排开，基线画成横线

    读法是"这条曲线有没有穿过 0"，不是"哪一个变体最极端"。穿过 0 的点标红，
    与基线同号的标蓝；基线自己用黑色菱形单独画出来，位置就是它在这条排序
    里的名次。
    """
    data_dir = data_dir if data_dir is not None else robustness_dir()
    curve = _read(data_dir, "synthesis_specification_curve.parquet")
    summary = _read(data_dir, "synthesis.parquet")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for ax, quantity in zip(axes.ravel(), QUANTITY_ORDER):
        sub = curve[(curve["quantity"] == quantity)
                    & (curve["model"] == model)].copy()
        sub = sub.sort_values("estimate", na_position="last")
        live = sub[sub["estimate"].notna()]
        n_missing = int(len(sub) - len(live))

        x = np.arange(len(live))
        agrees = _as_bool(live["agrees_with_baseline"])
        colors = np.where(agrees, AGREE_COLOR, DISAGREE_COLOR)
        ax.scatter(x, live["estimate"].values, s=4, c=colors, linewidths=0,
                   zorder=3)

        base_row = summary[summary["quantity"] == quantity]
        baseline = (float(base_row["baseline_estimate_{}".format(model)].iloc[0])
                    if len(base_row) else np.nan)
        if not pd.isna(baseline):
            ax.axhline(baseline, color=BASELINE_COLOR, linewidth=1.0,
                       linestyle="--", zorder=4)
            rank = int((live["estimate"].values < baseline).sum())
            ax.plot([rank], [baseline], marker="D", markersize=5,
                    color=BASELINE_COLOR, zorder=5)
        ax.axhline(0.0, color=NEUTRAL, linewidth=0.8, zorder=1)

        n_disagree = int((~agrees).sum())
        ax.set_title("{}\n{} specifications, {} against baseline sign".format(
            QUANTITY_LABELS[quantity].replace("\n", " "), len(live),
            n_disagree), fontsize=7.5)
        ax.set_xlabel("specification (sorted by estimate)", fontsize=6.5)
        ax.set_ylabel("AME / DiD (probability scale)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        if n_missing:
            _note_missing(ax, ["{} specification(s) produced no estimate "
                               "(shown nowhere on this curve)".format(n_missing)])

    handles = [
        Line2D([], [], color=AGREE_COLOR, marker="o", linestyle="none",
               markersize=4, label="Agrees with baseline sign"),
        Line2D([], [], color=DISAGREE_COLOR, marker="o", linestyle="none",
               markersize=4, label="Opposite sign"),
        Line2D([], [], color=BASELINE_COLOR, marker="D", linestyle="--",
               markersize=5, linewidth=1.0, label="Main specification"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Specification curve, {} layer, {} Weibo".format(model, year),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fg._save_fig(fig, "figR1_specification_curve", fig_dir)


# ---------------------------------------------------------------------------
# 图 R2：逐族方向一致率（§13.10 第一条准则）
# ---------------------------------------------------------------------------

def figR2_direction_by_family(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR,
                              model=HEADLINE_LAYER):
    """族 × 量的方向一致率热力图，格子里同时写出这一族活着的行数

    为什么必须逐族画：两个重抽样族各 200 个 replicate，按行汇总时它们占掉
    七成话语权，把只有 3~6 个变体的 denominators / post_types / user_type 压
    到看不见。§13.10 的第一条准则问的正是"换一批账号、换一个月、换一个分母
    之后方向还在不在"，那是逐族的读法。
    """
    data_dir = data_dir if data_dir is not None else robustness_dir()
    by_family = _read(data_dir, "synthesis_direction_by_family.parquet")
    sub = by_family[by_family["model"] == model]

    families = [f for f in sub["variant_family"].dropna().unique()]
    grid = np.full((len(families), len(QUANTITY_ORDER)), np.nan)
    labels = np.empty((len(families), len(QUANTITY_ORDER)), dtype=object)
    labels[:] = ""
    for i, family in enumerate(families):
        for j, quantity in enumerate(QUANTITY_ORDER):
            row = sub[(sub["variant_family"] == family)
                      & (sub["quantity"] == quantity)]
            if not len(row):
                labels[i, j] = "—"
                continue
            share = row["share_agree"].iloc[0]
            n_live = row["n_live"].iloc[0]
            if pd.isna(share):
                # 这一族对这个量一个估计都没跑出来：不是 0%，是没测
                labels[i, j] = "not\ntested"
                continue
            grid[i, j] = float(share)
            labels[i, j] = "{:.0f}%\nn={:,}".format(100 * float(share),
                                                    int(n_live))

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    cmap = matplotlib.colormaps["RdYlBu"].copy()
    # 没测过的格子不能借配色表的任何一档颜色：那会读成"一致率很低"。
    # 单独给一个中性灰，并在标题里说明它的含义
    cmap.set_bad("#e8e8e8")
    ax.set_facecolor("#e8e8e8")
    mesh = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.5, vmax=1.0,
                     aspect="auto")
    for i in range(len(families)):
        for j in range(len(QUANTITY_ORDER)):
            value = grid[i, j]
            # 深蓝底上的深灰字在 6pt 下读不出来：按格子的明暗选字色
            text_color = "#ffffff" if (not np.isnan(value) and value >= 0.86) \
                else "#222222"
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=6,
                    color=text_color)
    ax.set_xticks(range(len(QUANTITY_ORDER)))
    ax.set_xticklabels([QUANTITY_LABELS[q] for q in QUANTITY_ORDER], fontsize=6.5)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=7)
    ax.set_title("Direction agreement with the main specification, by variant "
                 "family ({} layer, {})\n"
                 "denominator: variant rows that produced an estimate; "
                 "grey cells were never estimable".format(model, year),
                 fontsize=9)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("share of variants agreeing on sign", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fg._save_fig(fig, "figR2_direction_by_family", fig_dir)


# ---------------------------------------------------------------------------
# 图 R3：少数单位的影响力（§13.10 第三条准则）
# ---------------------------------------------------------------------------

def figR3_influence(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR,
                    model=HEADLINE_LAYER):
    """每个量在六类"影响单位"下的最大相对位移，阈值画成竖线

    相对位移 = |变体估计 − 基线估计| / |基线估计|。超过阈值只说明"这一类单位
    里存在一个能把估计推走这么多的版本"，不等于结论不成立——最右边那根柱子
    旁边写着推走它的那个变体叫什么，判断留给读者。
    """
    data_dir = data_dir if data_dir is not None else robustness_dir()
    influence = _read(data_dir, "synthesis_influence.parquet")
    sub = influence[influence["model"] == model]
    threshold = (float(sub["threshold"].dropna().iloc[0])
                 if sub["threshold"].notna().any() else np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharex=True)
    missing_cells = []
    for ax, quantity in zip(axes.ravel(), QUANTITY_ORDER):
        rows = sub[sub["quantity"] == quantity]
        values, names, worst = [], [], []
        for unit in INFLUENCE_UNITS:
            row = rows[rows["influence_unit"] == unit]
            if not len(row) or pd.isna(row["max_abs_relative_shift"].iloc[0]):
                values.append(np.nan)
                worst.append("")
                names.append(INFLUENCE_LABELS[unit])
                missing_cells.append("{} / {}".format(quantity, unit))
                continue
            values.append(float(row["max_abs_relative_shift"].iloc[0]))
            worst.append(str(row["worst_variant_label"].iloc[0])[:34])
            names.append(INFLUENCE_LABELS[unit])

        y = np.arange(len(INFLUENCE_UNITS))
        finite = np.array([0.0 if pd.isna(v) else v for v in values])
        colors = [DISAGREE_COLOR if (not pd.isna(v) and not pd.isna(threshold)
                                     and v > threshold) else AGREE_COLOR
                  for v in values]
        ax.barh(y, finite, color=colors, height=0.62)
        for i, (value, label) in enumerate(zip(values, worst)):
            if pd.isna(value):
                ax.text(0.01, i, "not tested", va="center", fontsize=5.5,
                        color="#b22222")
            elif label:
                ax.text(min(value, 1.45) + 0.02, i, label, va="center",
                        fontsize=5.0, color="#444444")
        if not pd.isna(threshold):
            ax.axvline(threshold, color=BASELINE_COLOR, linestyle="--",
                       linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=5.8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.6)
        ax.set_title(QUANTITY_LABELS[quantity].replace("\n", " "), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("max |estimate − baseline| / |baseline|", fontsize=6.5)

    fig.suptitle("Is the result driven by a few accounts, users, terms or "
                 "months? ({} layer, {}; dashed line = {:.0%} threshold)".format(
                     model, year, threshold if not pd.isna(threshold) else 0),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fg._save_fig(fig, "figR3_influence", fig_dir)


# ---------------------------------------------------------------------------
# 图 R4：控制活动量后的衰减（§13.10 第二条准则）
# ---------------------------------------------------------------------------

def figR4_attenuation(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """M0 -> M1 的斜线图，右侧标出各变体衰减率的 p10–p90

    衰减率 = 1 − M1/M0，正值代表控制活动量之后差异缩小。负值不是错误，它
    的意思是控制之后差异反而变大——entry_public 就是这样，必须原样画出来。
    """
    data_dir = data_dir if data_dir is not None else robustness_dir()
    att = _read(data_dir, "synthesis_attenuation.parquet")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    flags = []
    for i, quantity in enumerate(QUANTITY_ORDER):
        row = att[att["quantity"] == quantity]
        if not len(row):
            flags.append("{}: absent from the attenuation table".format(quantity))
            continue
        m0 = row["baseline_estimate_M0"].iloc[0]
        m1 = row["baseline_estimate_M1"].iloc[0]
        if pd.isna(m0) or pd.isna(m1):
            flags.append("{}: no M0/M1 pair".format(quantity))
            continue
        color = AGREE_COLOR if float(m0) > 0 else DISAGREE_COLOR
        ax.plot([0, 1], [float(m0), float(m1)], marker="o", markersize=5,
                color=color, linewidth=1.3)
        ax.annotate(QUANTITY_LABELS[quantity].replace("\n", " "),
                    (1, float(m1)), textcoords="offset points", xytext=(6, 0),
                    fontsize=6, va="center")
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["M0\n(gender only)", "M1\n(+ activity)"], fontsize=7)
    ax.set_xlim(-0.12, 1.75)
    ax.set_ylabel("AME / DiD (probability scale)", fontsize=7.5)
    ax.set_title("Main specification: M0 → M1", fontsize=9)
    ax.tick_params(labelsize=6.5)
    _note_missing(ax, flags)

    y = np.arange(len(QUANTITY_ORDER))
    for i, quantity in enumerate(QUANTITY_ORDER):
        row = att[att["quantity"] == quantity]
        if not len(row):
            continue
        p10 = row["attenuation_p10"].iloc[0]
        p90 = row["attenuation_p90"].iloc[0]
        base = row["baseline_attenuation"].iloc[0]
        if not (pd.isna(p10) or pd.isna(p90)):
            ax2.plot([float(p10), float(p90)], [i, i], color=BAND_COLOR,
                     linewidth=6, solid_capstyle="butt", zorder=2)
        if not pd.isna(base):
            ax2.plot([float(base)], [i], marker="D", markersize=5,
                     color=BASELINE_COLOR, zorder=3)
    ax2.axvline(0.0, color=NEUTRAL, linewidth=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([QUANTITY_LABELS[q].replace("\n", " ")
                         for q in QUANTITY_ORDER], fontsize=6.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("attenuation  1 − M1/M0   (negative = effect grows)",
                   fontsize=7)
    ax2.set_title("Across variants: p10–p90 band, ◆ = main specification",
                  fontsize=9)
    ax2.tick_params(labelsize=6.5)

    fig.suptitle("How much does controlling for general activity shrink the "
                 "gender gap? ({})".format(year), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fg._save_fig(fig, "figR4_attenuation", fig_dir)


# ---------------------------------------------------------------------------
# 图 R5：词表重采样的系数分布（§12.7 / §13.3）
# ---------------------------------------------------------------------------

def figR5_vocabulary_resampling(year=config.YEAR, data_dir=None,
                                fig_dir=FIG_DIR, model=HEADLINE_LAYER):
    """keep-0.8 词表重采样下四个内容类量的估计分布，基线画成竖线

    只画内容类的量（topical_* 与 did_topical）与来源进入的对照：词表只进
    内容测量，来源进入是表 B 的转发事件，重采样对它没有作用——那条"分布退化
    成一根线"本身就是一条要被看见的事实，所以照画不误。
    """
    data_dir = data_dir if data_dir is not None else robustness_dir()
    vocab = dedup_variants(_read(data_dir, "vocabulary.parquet"))
    summary = _read(data_dir, "synthesis.parquet")

    resample = vocab[vocab["variant_label"].astype(str).str.match(
        r"^keep0\.8_rep\d+$", na=False)]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.8))
    for ax, quantity in zip(axes.ravel(), QUANTITY_ORDER):
        meta = _quantity_filter(quantity)
        sub = resample[(resample["outcome"] == meta["outcome"])
                       & (resample["domain"] == meta["domain"])
                       & (resample["term"] == meta["term"])
                       & (resample["model"] == model)]
        values = sub["estimate"].dropna().values
        n_nan = int(sub["estimate"].isna().sum())
        if len(values):
            spread = float(np.nanmax(values) - np.nanmin(values))
            if spread == 0:
                # 分布退化成一根线：直方图会画成一根无限高的柱子，
                # 直接说出来比画出来更诚实
                point = float(values[0])
                ax.axvline(point, color=AGREE_COLOR, linewidth=2.0)
                ax.text(0.5, 0.55, "all {} replicates identical\n"
                                   "(vocabulary does not enter this measure)"
                        .format(len(values)), transform=ax.transAxes,
                        ha="center", fontsize=6, color="#444444")
                # 只有两条竖线时坐标轴会退化成 [0, 1]，把一个 0.036 的估计
                # 画得像贴在原点上。手动给一个以估计值为中心的窄窗口，
                # 让这一格与相邻几格的读法一致
                half = max(abs(point) * 0.35, 1e-4)
                ax.set_xlim(min(point - half, -half * 0.2),
                            max(point + half, half * 0.2))
            else:
                ax.hist(values, bins=24, color=AGREE_COLOR, alpha=0.75)
        base_row = summary[summary["quantity"] == quantity]
        baseline = (float(base_row["baseline_estimate_{}".format(model)].iloc[0])
                    if len(base_row) else np.nan)
        if not pd.isna(baseline):
            ax.axvline(baseline, color=BASELINE_COLOR, linestyle="--",
                       linewidth=1.2)
        ax.axvline(0.0, color=NEUTRAL, linewidth=0.8)
        ax.set_title("{}\n{} replicates, {:.0f}% of the vocabulary kept".format(
            QUANTITY_LABELS[quantity].replace("\n", " "), len(values), 80),
            fontsize=7.5)
        ax.set_xlabel("AME / DiD", fontsize=6.5)
        ax.set_ylabel("replicates", fontsize=6.5)
        ax.tick_params(labelsize=6)
        if n_nan:
            _note_missing(ax, ["{} replicate(s) produced no estimate".format(n_nan)])

    handles = [
        Line2D([], [], color=AGREE_COLOR, linewidth=6, alpha=0.75,
               label="Replicate estimates"),
        Line2D([], [], color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
               label="Main specification (full vocabulary)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Vocabulary resampling: keep a random 80% of the term list, "
                 "{} layer, {}".format(model, year), fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fg._save_fig(fig, "figR5_vocabulary_resampling", fig_dir)


def _quantity_filter(quantity):
    """把 quantity 翻回 (outcome, domain, term) —— 与 harness.QUANTITY_META 同源"""
    from gender_domain.robustness import harness
    return harness.QUANTITY_META[quantity]


# ---------------------------------------------------------------------------
# 图 R6：leave-one-account-out 与来源账号集中度（§12.7 / §13.5）
# ---------------------------------------------------------------------------

def figR6_accounts(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR,
                   model=HEADLINE_LAYER):
    """上：逐个删掉一个来源账号后的估计；下：top-k 账号占了多少转发事件

    上半张回答"结论会不会被某一个账号推着走"，下半张说明为什么必须问这个
    问题——公共事务这一侧 top-5 账号占了近九成事件。
    """
    data_dir = data_dir if data_dir is not None else robustness_dir()
    accounts = dedup_variants(_read(data_dir, "accounts.parquet"))
    concentration = _read(data_dir, "account_concentration.parquet")
    concentration = concentration.drop_duplicates()
    summary = _read(data_dir, "synthesis.parquet")

    loo = accounts[accounts["variant_label"].astype(str).str.startswith("loo_")]
    entry_quantities = ("entry_public", "entry_celebrity", "did_entry")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for ax, quantity in zip(axes[0], entry_quantities):
        meta = _quantity_filter(quantity)
        sub = loo[(loo["outcome"] == meta["outcome"])
                  & (loo["domain"] == meta["domain"])
                  & (loo["term"] == meta["term"])
                  & (loo["model"] == model)].copy()
        sub = sub.sort_values("estimate", na_position="last")
        live = sub[sub["estimate"].notna()]
        ax.scatter(np.arange(len(live)), live["estimate"].values, s=6,
                   color=AGREE_COLOR, zorder=3)
        base_row = summary[summary["quantity"] == quantity]
        baseline = (float(base_row["baseline_estimate_{}".format(model)].iloc[0])
                    if len(base_row) else np.nan)
        if not pd.isna(baseline):
            ax.axhline(baseline, color=BASELINE_COLOR, linestyle="--",
                       linewidth=1.0)
        ax.axhline(0.0, color=NEUTRAL, linewidth=0.8)
        ax.set_title("{}\n{} accounts dropped one at a time".format(
            QUANTITY_LABELS[quantity].replace("\n", " "), len(live)),
            fontsize=7.5)
        ax.set_xlabel("account dropped (sorted by estimate)", fontsize=6.5)
        ax.set_ylabel("AME / DiD", fontsize=6.5)
        ax.tick_params(labelsize=6)
        n_nan = int(len(sub) - len(live))
        if n_nan:
            _note_missing(ax, ["{} leave-one-out fit(s) produced no "
                               "estimate".format(n_nan)])

    ks = sorted(concentration["k"].unique())
    width = 0.26
    for ax, domain in zip(axes[1], ("public", "celebrity")):
        for offset, gender in zip((-width, 0.0, width), ("all", "m", "f")):
            shares, n_accounts = [], None
            for k in ks:
                row = concentration[(concentration["domain"] == domain)
                                    & (concentration["gender"] == gender)
                                    & (concentration["k"] == k)]
                shares.append(float(row["top_k_share"].iloc[0]) if len(row)
                              else np.nan)
                if len(row):
                    n_accounts = int(row["n_accounts"].iloc[0])
            color = {"all": NEUTRAL, "m": fg.MALE_COLOR,
                     "f": fg.FEMALE_COLOR}[gender]
            label = {"all": "All users", "m": "Male", "f": "Female"}[gender]
            ax.bar(np.arange(len(ks)) + offset, shares, width=width,
                   color=color, label=label)
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels(["top {}".format(int(k)) for k in ks], fontsize=6.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("share of source-retweet events", fontsize=6.5)
        ax.set_title("{} sources ({} accounts in the list)".format(
            fg.DOMAIN_LABELS[domain], n_accounts), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)

    axes[1][2].axis("off")
    axes[1][2].text(0.0, 0.5,
                    "Top-share is computed within each gender's own event\n"
                    "pool, so the male and female bars answer\n"
                    "\"how concentrated is this gender's retweeting\",\n"
                    "not \"who retweets the same accounts\".\n\n"
                    "The pooled-ranking column in the table answers the\n"
                    "second question and is reported in the appendix table.",
                    transform=axes[1][2].transAxes, fontsize=6.5, va="center")

    fig.suptitle("Source accounts: leave-one-account-out and concentration "
                 "({} layer, {})".format(model, year), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fg._save_fig(fig, "figR6_accounts", fig_dir)


# ---------------------------------------------------------------------------
# 图 R7：次要分析的 FDR（§11.3）
# ---------------------------------------------------------------------------

def figR7_fdr(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """次要分析的 BH 图：p 值按名次排开，斜线是 BH 阈值

    六个预先设定的量**不进**这张图——它们不做多重比较校正（§11.3），
    把它们混进来会让校正看起来更严格，实际上改变的是别人的 q 值。
    """
    data_dir = data_dir if data_dir is not None else robustness_dir()
    fdr = _read(data_dir, "synthesis_fdr.parquet")

    prespecified = pd.Series(_as_bool(fdr["is_prespecified"]), index=fdr.index)
    tested = fdr[(~prespecified) & fdr["p_value"].notna()].copy()
    untested = fdr[(~prespecified) & fdr["p_value"].isna()]
    tested = tested.sort_values("p_value").reset_index(drop=True)
    alpha = (float(fdr["fdr_alpha"].dropna().iloc[0])
             if fdr["fdr_alpha"].notna().any() else 0.05)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    rank = np.arange(1, len(tested) + 1)
    rejected = _as_bool(tested["fdr_rejected"])
    # n=225,339 下大量 p 值小到 1e-240：照实画会把对数轴拉到 240 个数量级，
    # 于是唯一还看得出形状的那一段（BH 阈值附近）被压成一条线。把小于
    # P_FLOOR 的点画在地板上，并数出来有多少个——地板是画法，不是数据。
    drawn = np.maximum(tested["p_value"].values, P_VALUE_FLOOR)
    n_at_floor = int((tested["p_value"].values < P_VALUE_FLOOR).sum())
    ax.scatter(rank[rejected], drawn[rejected], s=8,
               color=AGREE_COLOR, label="Survives BH at q < {:.2f}".format(alpha))
    ax.scatter(rank[~rejected], drawn[~rejected], s=8,
               color=NEUTRAL, label="Does not survive")
    ax.plot(rank, alpha * rank / max(len(tested), 1), color=BASELINE_COLOR,
            linewidth=1.0, linestyle="--", label="BH threshold")
    ax.set_yscale("log")
    ax.set_ylim(P_VALUE_FLOOR / 3, 2.0)
    ax.set_xlabel("rank of p-value among secondary analyses", fontsize=7)
    ax.set_ylabel("p-value (log scale, floored at {:g})".format(P_VALUE_FLOOR),
                  fontsize=7)
    ax.set_title("Benjamini–Hochberg on {} secondary tests".format(len(tested)),
                 fontsize=9)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    if n_at_floor:
        ax.text(0.02, 0.06, "{} test(s) have p < {:g} and are drawn on the "
                            "floor".format(n_at_floor, P_VALUE_FLOOR),
                transform=ax.transAxes, fontsize=5.5, color="#444444")

    counts = [int(rejected.sum()), int((~rejected).sum()),
              int(len(untested)), int(prespecified.sum())]
    names = ["Survives BH", "Does not survive", "No usable SE\n(not tested)",
             "Prespecified\n(never corrected)"]
    ax2.bar(np.arange(len(counts)), counts,
            color=[AGREE_COLOR, NEUTRAL, "#d9b38c", "#7f7f7f"])
    for i, value in enumerate(counts):
        ax2.text(i, value, "{:,}".format(value), ha="center", va="bottom",
                 fontsize=7)
    ax2.set_xticks(np.arange(len(counts)))
    ax2.set_xticklabels(names, fontsize=6)
    ax2.set_ylabel("result rows", fontsize=7)
    ax2.set_title("What the correction covers", fontsize=9)
    ax2.tick_params(labelsize=6.5)

    fig.suptitle("Multiple-comparison correction, secondary analyses only "
                 "({})".format(year), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fg._save_fig(fig, "figR7_fdr", fig_dir)


# ---------------------------------------------------------------------------

FIGURES = (
    ("figR1", figR1_specification_curve),
    ("figR2", figR2_direction_by_family),
    ("figR3", figR3_influence),
    ("figR4", figR4_attenuation),
    ("figR5", figR5_vocabulary_resampling),
    ("figR6", figR6_accounts),
    ("figR7", figR7_fdr),
)


def all(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR):
    """按顺序生成全部稳健性附录图"""
    print("=" * 60)
    print("生成 {} 年的稳健性附录图，数据目录 {}".format(
        year, data_dir if data_dir is not None else robustness_dir()))
    print("=" * 60)
    paths = []
    for _, builder in FIGURES:
        paths.append(builder(year=year, data_dir=data_dir, fig_dir=fig_dir))
    print("\n共生成 {} 张图，输出目录 {}".format(len(paths), fig_dir))
    return paths


if __name__ == "__main__":
    commands = {name: builder for name, builder in FIGURES}
    commands["all"] = all
    fire.Fire(commands)
