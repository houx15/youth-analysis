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

import json
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
# 与 export_figure_data.FIGURE_MANIFEST_FILE 必须同名：本模块跑在本地，
# 不 import 那个会去读服务器路径的模块（与 quantile_column 同一处理）
FIGURE_MANIFEST_FILE = "manifest.json"

MALE_COLOR = "#20AEE6"
FEMALE_COLOR = "#ff7333"
GENDER_COLORS = {"male": MALE_COLOR, "female": FEMALE_COLOR,
                 "m": MALE_COLOR, "f": FEMALE_COLOR}
GENDER_MARKERS = {"male": "o", "female": "^", "m": "o", "f": "^"}
# 观测点专用的记号：与任何一个预测点的记号都不同（图 3 里两种点同时出现，
# 只靠"填充与否"区分，在 6pt 的正文尺寸下会被读错端点）
OBSERVED_MARKER = "s"
OBSERVED_COLOR = "#6d6d6d"
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
# 四个 性别 × 领域 反事实格子的预测水平（models_interaction.PRED_CELLS）。
# 这些行的 domain 是真实领域，不是 "both"——只有差中差与交互系数写在
# domain="both" 上。
TERM_PRED = {
    ("male", "public"): "pred_male_public",
    ("female", "public"): "pred_female_public",
    ("male", "celebrity"): "pred_male_celebrity",
    ("female", "celebrity"): "pred_female_celebrity",
}
NOTE_PRED_CLIPPED = "pred_ci_clipped_to_unit_interval"
NOTE_PRED_UNAVAILABLE = "pred_ci_unavailable"

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
# 表 2 里 model 列的取值。source_entered 有两个分母口径，describe.py 把它们
# 折成了两个 model 变体（"raw/all_users" / "raw/retweeters_only"），这样
# (outcome, domain, model, term) 才在整张表里唯一；其余结果变量分母单一，
# model 仍然是 "raw"。图只画主口径，因此这里只需要主口径对应的那个变体。
PRIMARY_TABLE2_MODEL = {
    "source_entered": "raw/all_users",
    "topical_share": "raw",
    "source_share": "raw",
    "source_month_share": "raw",
}
# 图 2 每一行自己写分母：这张图把三个结果变量画在一起，分母各不相同，
# 写一句"与表 2 相同"只是把说明推给别处
AME_DENOMINATOR = {
    "source_entered": "all_same_gender_users",
    "topical_share": "users_with_expressive_posts",
    "source_month_share": "users_with_active_months",
    "source_share": "users_with_retweets",
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


def read_manifest(data_dir):
    """读 figure_data/manifest.json；缺文件当场报错

    这份清单由 export_figure_data.build 写在 figure_data/ 里，与结果表
    一起被下载到本地。缺了它就说明这批图数据是旧版导出层写的（或者只
    拷了部分文件），不能拿来画图。
    """
    path = os.path.join(data_dir, FIGURE_MANIFEST_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "未找到图数据清单 {}。这批 figure_data/ 是旧版导出层写的，或者"
            "只拷了一部分文件。请在服务器上重新运行 "
            "`python -m gender_domain.export_figure_data build --year=YYYY`，"
            "再把 analysis_data/figure_data/ 整个目录（含 manifest.json）"
            "下载到本地。".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_manifest(data_dir, year):
    """核对这批图数据确实是为 `year` 导出的，返回 manifest

    为什么必须核对：figure_data/ 里的文件名不带年份，`--year` 在本模块里
    只进标题和文件名前缀。也就是说"拿 2019 年的导出画一张标题写着 2020
    的图"在此之前完全无法察觉——图能正常画出来，每个数字都是真的，只是
    属于另一年。清单里同时带着 run_id 与 git_sha，出问题时能直接指认是
    哪一次运行的产物。
    """
    manifest = read_manifest(data_dir)
    exported = manifest.get("params", {}).get("year")
    if exported is not None and int(exported) != int(year):
        raise ValueError(
            "图数据是 {} 年导出的（{}/{} 的 params.year），但这次画图要的是 {} 年。"
            "figure_data/ 的文件名不带年份，--year 只进标题，所以这种错配"
            "画出来的图看不出任何异常——每个数字都是真的，只是属于另一年。"
            "请下载对应年份的导出目录，或用 --year={} 重画。"
            "（该批导出 run_id={}, git_sha={}）".format(
                exported, data_dir, FIGURE_MANIFEST_FILE, year, exported,
                manifest.get("run_id"), manifest.get("git_sha"))
        )
    return manifest


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
    """按列值精确筛选。传 None 表示这一列不筛

    刻意**不**在这里断言"只剩一行"：本函数同样用于多行筛选
    （按 outcome + denominator 取出一整张子表、按月份取出 12 行等），
    一刀切的断言会把这些正当用法一起打死。要"恰好一行"的地方一律走
    `select_one`——它才是 .iloc[0] 的唯一合法入口。
    """
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        if value is None:
            continue
        mask &= frame[column].astype("object") == value
    return frame[mask]


def select_one(frame, **filters):
    """按列值精确筛选并取出**唯一**一行；没有匹配时返回 None

    存在的理由：整套结果表共用"(outcome, domain, model, term) 逐行唯一"
    这条不变量，而本模块在十几处按这个键筛完就直接 `.iloc[0]`。键一旦
    撞上（表 2 的两个分母口径就曾经都写 model="raw"），`.iloc[0]` 会
    按行序任取一行——不报错、不留痕，只是图上某个点悄悄变成了另一个
    口径的估计。这里把"多于一行"变成当场失败：宁可画不出图，也不要画
    一张看起来正常、数字却随行序漂移的图。

    没有匹配时返回 None 而不是报错：调用方本来就要区分"这一格没有行"
    （在图上标注 row absent）与"这一格有多行"（是数据的 bug）。
    """
    rows = select(frame, **filters)
    assert len(rows) <= 1, (
        "按 {} 筛出了 {} 行，但这个键在结果表里必须唯一。撞键时 .iloc[0] "
        "会按行序任取一行，图上的数字就再也说不清是哪一个模型/口径的估计。"
        "重复的行是：\n{}".format(
            {k: v for k, v in filters.items() if v is not None},
            len(rows), rows.head(8).to_string(index=False))
    )
    return rows.iloc[0] if len(rows) == 1 else None


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
    """差中差的标注文字

    三种情形分别写清楚，绝不出现 "95% CI [+nan, +nan]"：拟合失败（无估计）、
    只有点估计没有区间（bootstrap 分支重拟合全失败时会这样），以及正常的
    点估计 + 区间。
    """
    rows = did_rows(frame, outcome=outcome, model=model)
    if rows.empty:
        return "DiD ({}): 结果表里没有这一行".format(model)
    assert len(rows) == 1, (
        "差中差行 outcome={} model={} 有 {} 行；这一行是全篇的头条数字，"
        "撞键时任取一行等于随机挑一个头条".format(outcome, model, len(rows)))
    row = rows.iloc[0]
    status = estimate_status(row)
    if status == "missing":
        return "DiD ({}): 无估计（{}）".format(model, row["note"] or "note 缺失")
    if status == "ci_unavailable":
        return "DiD ({}) = {:+.3f}  95% CI unavailable（{}）".format(
            model, row["estimate"], row["note"] or "note 缺失")
    text = "DiD ({}) = {:+.3f}  95% CI [{:+.3f}, {:+.3f}]".format(
        model, row["estimate"], row["ci_low"], row["ci_high"])
    if status == "clipped":
        text += "（CI clipped to [0,1]）"
    return text


def predicted_cells(frame, outcome, model=HEADLINE_LAYER):
    """取出四个 性别 × 领域 反事实格子的预测水平行

    这些行由 models_interaction 与差中差在同一次拟合、同一批反事实设计
    矩阵上算出（格子之差恒等于域内性别差，两者再相减恒等于差中差，
    模块里守到 1e-10），所以图 3 画格子、标注差中差时，两者天然自洽。
    它们的 domain 是真实领域，因此按领域分面时会自然落进正确的面板，
    与 domain="both" 的差中差行互不干扰。
    """
    rows = select(frame, outcome=outcome, model=model)
    rows = rows[rows["term"].astype("object").isin(list(TERM_PRED.values()))]
    return rows[rows["domain"].astype("object").isin(list(DOMAIN_ORDER))]


def _note_has(row, token):
    note = row.get("note")
    return isinstance(note, str) and token in note


def estimate_status(row):
    """任何一行估计的成色：ok / clipped / ci_unavailable / missing

    四种状态在图上长得不一样，是因为它们说的是四件不同的事：
      ok              点估计与区间都算出来了；
      clipped         区间是正态近似区间，被显式截回 [0,1]，那一端不是
                      真实的置信界，画成实心端点会骗人；
      ci_unavailable  只有点估计、没有区间——必须画成一个没有区间的点，
                      并写明区间为什么没有；
      missing         整行没估出来（估计值 NaN）——必须在该位置留一个显式的
                      缺口标记，让读者看见"这里本该有一个估计"。

    `ci_unavailable` 不是假想情形，也不限于图 3 的预测格子：
    `stats_utils.average_marginal_effect` 在协方差不可用时、
    `models_interaction.difference_in_differences` 的 bootstrap 分支在重拟合
    全部失败时，都会**故意**返回"有点估计、区间为 NaN"的一行（AME 的
    docstring 明确要求下游按"没有区间"处理）。如果画图时把缺失的界当成 0
    宽度的区间，这一行会变成整张图上看起来最精确的估计——一个没有任何
    不确定性信息的数字，反而成了最有说服力的那个点。所以判定只看区间在不在，
    不依赖 note 里有没有写 `pred_ci_unavailable`：note 是解释，不是判据。
    """
    if pd.isna(row["estimate"]):
        return "missing"
    if pd.isna(row["ci_low"]) or pd.isna(row["ci_high"]) \
            or _note_has(row, NOTE_PRED_UNAVAILABLE):
        return "ci_unavailable"
    # 截断只按 note 判定，不按"界是否贴着 0/1"反推：差值类的量（AME、
    # 性别差）本来就可能取到负值，按边界反推会把一条完全正常的 AME 区间
    # 误标成被截断。哪一端被截断由 clipped_ends 在画箭头时单独判断。
    if _note_has(row, NOTE_PRED_CLIPPED):
        return "clipped"
    return "ok"


def cell_status(row):
    """预测格子的成色——与任何一行估计同一条规则（见 estimate_status）"""
    return estimate_status(row)


def clipped_ends(row, low_bound=0.0, high_bound=1.0):
    """区间的哪一端被截回了取值范围，返回 ("low",)/("high",)/两者/空

    用不等式而不是 `bound == 0.0` 判定：生产端目前用 np.clip 得到恰好的
    0.0/1.0，但换一种写法（例如先截再做一次浮点运算）就可能落在
    -1e-18 或 1+1e-18 上，等号判定会让箭头静默消失，而这一行仍然被列进
    flagged cells——图上标注与图下说明互相矛盾是最难查的一类错误。
    只对概率/比例尺度有意义，其它尺度的调用方不要传。
    """
    ends = []
    if pd.notna(row.get("ci_low")) and float(row["ci_low"]) <= low_bound:
        ends.append("low")
    if pd.notna(row.get("ci_high")) and float(row["ci_high"]) >= high_bound:
        ends.append("high")
    return tuple(ends)


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


def _annotate_missing(ax, labels, prefix="missing estimates", y=0.02,
                      va="bottom"):
    """把缺失说明写在坐标轴上，最多列 4 条，其余折叠成计数

    y < 0 时写在坐标轴下方（图 3 用这个位置，因为面板内部已经被格子、
    连线和差中差标注框占满了）；保存时用 bbox_inches="tight"，轴外的
    文字不会被裁掉。
    """
    if not labels:
        return
    shown = labels[:4]
    text = "{}:\n".format(prefix) + "\n".join("· " + s for s in shown)
    if len(labels) > len(shown):
        text += "\n· … {} more".format(len(labels) - len(shown))
    ax.text(0.02, y, text, transform=ax.transAxes, fontsize=6,
            color="#b22222", va=va, ha="left")


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


def _n_range_label(values):
    """把一组样本量写成 "n=52,111 / 174,222" 或 "n=226,333"

    图 3 的两种点各自来自不同的总体，样本量必须分别写出来；同一类点内部
    如果各格样本量不同（男女各按自己的分母），就把它们并列写出，而不是
    只写一个数字让读者以为四个点同源。
    """
    if not values:
        return "n unknown"
    return "n=" + " / ".join("{:,}".format(v) for v in sorted(values))


def _n_label(row):
    """样本量标注：n_obs 与流失原因都写出来"""
    n_obs = row.get("n_obs")
    if n_obs is None or (isinstance(n_obs, float) and np.isnan(n_obs)):
        return ""
    return "n={:,}".format(int(n_obs))


def _errorbar(ax, x, y, low, high, color, marker, label=None, horizontal=True):
    """画一个点 + 95% 区间。两端都必须存在——缺界的行不走这条路径

    这里刻意不再对缺失的界做任何兜底：以前用 0.0 兜底，等价于把"没有区间"
    画成"区间宽度为 0"，那是图上最自信的一种画法，正好与事实相反。
    缺界的行由 draw_estimate 分流到 ci_unavailable 分支。
    """
    if pd.isna(x if horizontal else y):
        return
    if pd.isna(low) or pd.isna(high):
        raise ValueError(
            "_errorbar 收到缺失的区间端点；没有区间的行必须走 draw_estimate 的 "
            "ci_unavailable 分支，不能画成宽度为 0 的区间"
        )
    if horizontal:
        err = [[x - low], [high - x]]
        ax.errorbar([x], [y], xerr=err, fmt=marker, color=color, markersize=6,
                    capsize=3, linewidth=1.2, label=label)
    else:
        err = [[y - low], [high - y]]
        ax.errorbar([x], [y], yerr=err, fmt=marker, color=color, markersize=6,
                    capsize=3, linewidth=1.2, label=label)


def _gender_legend_handles(suffix="", with_observed=False):
    """用固定的代理句柄拼图例，不从画出来的 artist 里取

    从 artist 取图例有一个安静的坑：如果某一格恰好是 ci_unavailable，
    那一格画的是空心点，图例键就会变成空心的，与图上其余实心点自相矛盾。
    代理句柄描述的是"这类点应该长什么样"，与某一格的成色无关。
    """
    handles = [
        Line2D([], [], color=GENDER_COLORS[gender], marker=GENDER_MARKERS[gender],
               linestyle="none", markersize=6,
               label=GENDER_LABELS[gender] + suffix)
        for gender in GENDER_ORDER
    ]
    if with_observed:
        handles.append(Line2D([], [], color=OBSERVED_COLOR,
                              marker=OBSERVED_MARKER, linestyle="none",
                              markersize=6, markerfacecolor="none",
                              label="Observed (raw)"))
    handles.append(Line2D([], [], color="#555555", marker="o", linestyle="none",
                          markersize=6, markerfacecolor="none",
                          markeredgewidth=1.6, label="No CI available"))
    handles.append(Line2D([], [], color="#b22222", marker="x", linestyle="none",
                          markersize=7, label="No estimate"))
    return handles


def draw_estimate(ax, row, pos, color, marker, orientation="h", label=None,
                  flags=None, tag="", annotate_n=False, unit_scale=False,
                  gap_frac=0.06):
    """按 estimate_status 画一行估计，返回状态字符串

    这是图 1/2/3/5 唯一的画点入口，存在的理由是四张图必须对"没有区间"和
    "没有估计"给出同一种、且不骗人的画法：

      ok              实心点 + 区间；
      clipped         实心点 + 区间，并在被截回 [0,1] 的那一端画箭头端点
                      （unit_scale=True 时才判断，差值类的量不适用）；
      ci_unavailable  空心点、**不画区间**，旁注 "CI n/a"，并写进 flags；
      missing         不画点，在该行/该列的位置画一个红色 ✕ 缺口标记，
                      并写进 flags——缺口必须看得见，不能让一行凭空消失。

    Args:
        pos: 类别方向的坐标（orientation="h" 时是 y，"v" 时是 x）
        orientation: "h" 横向区间（点区间图/森林图），"v" 纵向区间
        flags: 收集缺陷说明的列表，最后交给 _annotate_missing 印在图上
        tag: 缺陷说明里用来指认是哪一格的前缀
        unit_scale: 该量是否落在 [0,1]（概率/比例的水平量），决定要不要
            判断截断端
        gap_frac: missing 时缺口标记放在另一坐标轴的哪个位置（axes 分数）
    """
    horizontal = orientation == "h"
    status = estimate_status(row)
    note = row.get("note") or "no note"
    if flags is None:
        flags = []

    if status == "missing":
        flags.append("{}: no estimate ({})".format(tag or row.get("term"), note))
        blended = matplotlib.transforms.blended_transform_factory(
            ax.transAxes if horizontal else ax.transData,
            ax.transData if horizontal else ax.transAxes)
        xy = ([gap_frac], [pos]) if horizontal else ([pos], [gap_frac])
        ax.plot(xy[0], xy[1], marker="x", markersize=9, color="#b22222",
                linestyle="none", transform=blended, zorder=4)
        ax.annotate("no estimate", (xy[0][0], xy[1][0]), xycoords=blended,
                    textcoords="offset points", xytext=(8, 0), fontsize=5.5,
                    color="#b22222",
                    va="center", ha="left")
        return status

    est = float(row["estimate"])
    x, y = (est, pos) if horizontal else (pos, est)

    if status == "ci_unavailable":
        flags.append("{}: CI unavailable ({})".format(
            tag or row.get("term"), note))
        ax.plot([x], [y], marker=marker, markersize=7, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.6, linestyle="none",
                color=color, zorder=3, label=label)
        ax.annotate("CI n/a", (x, y), textcoords="offset points",
                    xytext=(6, -8), fontsize=5.5, color="#b22222")
        return status

    _errorbar(ax, x, y, row["ci_low"], row["ci_high"], color, marker,
              label=label, horizontal=horizontal)
    if status == "clipped":
        ends = clipped_ends(row) if unit_scale else ()
        flags.append("{}: CI clipped to [0,1] at {} end ({})".format(
            tag or row.get("term"), "/".join(ends) if ends else "unknown", note))
        for end in ends:
            bound = float(row["ci_low"] if end == "low" else row["ci_high"])
            if horizontal:
                ax.plot([bound], [pos], marker="<" if end == "low" else ">",
                        markersize=5, color=color, linestyle="none", zorder=4)
            else:
                ax.plot([pos], [bound], marker="v" if end == "low" else "^",
                        markersize=5, color=color, linestyle="none", zorder=4)
    if annotate_n:
        ax.annotate(_n_label(row), (x, y), textcoords="offset points",
                    xytext=(6, 5), fontsize=6, color="#555555")
    return status


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
    assert_manifest(data_dir, year)
    table2 = _read(data_dir, "table2_raw_gender_gaps.parquet", TABLE2_COLUMNS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = (("source_entered", "probability"), ("topical_share", "proportion"))
    for ax, (outcome, _scale) in zip(axes, panels):
        denominator = PRIMARY_DENOMINATOR[outcome]
        rows = select(table2, outcome=outcome,
                      model=PRIMARY_TABLE2_MODEL[outcome], denominator=denominator)
        y_ticks, y_labels, flags = [], [], []
        for i, domain in enumerate(DOMAIN_ORDER):
            y_ticks.append(i)
            y_labels.append(DOMAIN_LABELS[domain])
            for gender in GENDER_ORDER:
                cell = select_one(rows, domain=domain, term=gender)
                y = i + (0.14 if gender == "male" else -0.14)
                if cell is None:
                    flags.append("{}/{}: row absent".format(domain, gender))
                    continue
                draw_estimate(
                    ax, cell, y, GENDER_COLORS[gender],
                    GENDER_MARKERS[gender], orientation="h",
                    label=GENDER_LABELS[gender] if i == 0 else None,
                    flags=flags, tag="{}/{}".format(domain, gender),
                    annotate_n=True, unit_scale=True)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(-0.6, len(DOMAIN_ORDER) - 0.4)
        ax.set_xlabel("{}\n({}, 95% CI)".format(
            OUTCOME_LABELS[outcome], DENOMINATOR_LABELS[denominator]), fontsize=8)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=9)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.legend(handles=_gender_legend_handles(), fontsize=7, loc="lower right")
        _annotate_missing(ax, flags, prefix="flagged rows")

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
    assert_manifest(data_dir, year)
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (_scale, xlabel, specs) in zip(axes, panels):
        y_ticks, y_labels, flags = [], [], []
        y = 0
        for frame, outcome, domain in specs:
            rows = select(frame, outcome=outcome, domain=domain, term=TERM_AME)
            y_ticks.append(y)
            # 每一行自己写出分母：这张图混着三个结果变量，分母各不相同，
            # 只在横轴上写一句"与表 2 相同"是把分母推给别处，不是说明分母
            y_labels.append("{}\n{}\n({})".format(
                OUTCOME_LABELS[outcome].split(",")[0], DOMAIN_LABELS[domain],
                DENOMINATOR_LABELS[AME_DENOMINATOR[outcome]]))
            for k, layer in enumerate(MODEL_LAYERS):
                row = select_one(rows, model=layer)
                offset = 0.22 * (1 - k)
                if row is None:
                    flags.append("{}/{}/{}: row absent".format(
                        outcome, domain, layer))
                    continue
                # 颜色编码"谁更高"；估计值缺失时按中性灰画缺口标记
                color = "#777777" if pd.isna(row["estimate"]) else (
                    MALE_COLOR if row["estimate"] >= 0 else FEMALE_COLOR)
                draw_estimate(ax, row, y + offset, color, LAYER_MARKERS[layer],
                              orientation="h", flags=flags,
                              tag="{}/{}/{}".format(outcome, domain, layer))
                if pd.notna(row["estimate"]):
                    ax.annotate(layer, (row["estimate"], y + offset),
                                textcoords="offset points", xytext=(7, -2),
                                fontsize=6, color="#555555")
            y += 1
        ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=6.5)
        ax.set_ylim(-0.6, len(specs) - 0.4)
        ax.set_xlabel(xlabel + "\n(95% CI; each row's denominator is printed "
                               "with its label)", fontsize=8)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        _annotate_missing(ax, flags, prefix="flagged rows")

    handles = [Line2D([], [], color="#555555", marker=LAYER_MARKERS[layer],
                      linestyle="none", label=LAYER_LABELS[layer])
               for layer in MODEL_LAYERS]
    handles += [Line2D([], [], color=MALE_COLOR, marker="s", linestyle="none",
                       label="Male higher"),
                Line2D([], [], color=FEMALE_COLOR, marker="s", linestyle="none",
                       label="Female higher"),
                Line2D([], [], color="#555555", marker="o", linestyle="none",
                       markerfacecolor="none", markeredgewidth=1.6,
                       label="No CI available"),
                Line2D([], [], color="#b22222", marker="x", linestyle="none",
                       label="No estimate")]
    fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=4,
               frameon=False)
    fig.suptitle(
        "Figure 2. Gender average marginal effects across model layers, {}".format(
            year), fontsize=11)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    return _save_fig(fig, "fig2_adjusted_effects", fig_dir)


# ---------------------------------------------------------------------------
# 图 3：Gender × Domain 交互（§12.3）
# ---------------------------------------------------------------------------

def fig3_interaction(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR,
                     model=HEADLINE_LAYER):
    """四个 性别 × 领域 格子的**模型预测值**，并显式标出差中差（§12.3）

    画的是什么：

    1. 实心点 = 交互模型（GEE，按用户聚类稳健）在四个反事实格子上的预测
       水平（`pred_{male,female}_{public,celebrity}`，默认 M1 层，可用
       `model=` 改成 M0/M2）。这四行与两个域内性别差、与差中差来自同一次
       拟合、同一批反事实设计矩阵，模块里用 1e-10 的恒等式断言守着：
       格子之差 == 域内性别差，两者再相减 == 差中差。因此图上画的格子与
       标注的头条数字天然自洽，读者可以自己在图上把差中差量出来。
    2. 空心灰点 = 同一格的**观测**比例（表 2，主分母口径），只画点不画区间，
       作为对照。保留它是因为"调整后与调整前差多少"本身就是要看的东西
       （M1 控制了一般活动量，M2 还控制画像）；但两者是不同的量，因此
       在图例、纵轴标签里分别标明 predicted / observed，不让读者去猜。
    3. 标注框 = 差中差（M0/M1/M2 三层都写）与两个域内的调整后性别差。
       差中差那一行的 domain 是 "both"，由 did_rows 显式取出——它不属于
       任何一个面板；四个格子的 domain 是真实领域，因此自然落进对应面板，
       两者互不干扰（domain_facets 永远只返回 public/celebrity）。

    格子的三种"不干净"状态在图上长得不一样（见 cell_status）：
    区间被截回 [0,1] 的格子在被截的那一端画一个箭头端点并在图上列出；
    协方差不可用的格子只画一个空心点、不画区间；整格 NaN 的格子在该位置
    留一个灰色 ✕ 标记——缺口必须看得见，不能让格子凭空消失。
    """
    data_dir = data_dir or figure_data_dir()
    assert_manifest(data_dir, year)
    table2 = _read(data_dir, "table2_raw_gender_gaps.parquet", TABLE2_COLUMNS)
    inter = _read(data_dir, "interaction_gender_domain.parquet", RESULT_COLUMNS)

    outcomes = ("source_entered", "topical_share")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, outcome in zip(axes, outcomes):
        denominator = PRIMARY_DENOMINATOR[outcome]
        observed = select(table2, outcome=outcome,
                          model=PRIMARY_TABLE2_MODEL[outcome],
                          denominator=denominator)
        cells = predicted_cells(inter, outcome, model)
        facets = domain_facets(cells) or list(DOMAIN_ORDER)
        flags = []
        n_predicted, n_observed = set(), set()
        for i, domain in enumerate(facets):
            for gender in GENDER_ORDER:
                x = i + (0.13 if gender == "male" else -0.13)
                color = GENDER_COLORS[gender]

                # 观测值：灰色空心方块，与预测点的圆/三角形状完全不同。
                # 只靠"填充与否"区分两种点，在正文尺寸下会被读错端点，
                # 而这张图恰恰要求读者把同一个领域内的两端连起来看。
                obs_row = select_one(observed, domain=domain, term=gender)
                if obs_row is not None and pd.notna(obs_row["estimate"]):
                    ax.plot([x], [obs_row["estimate"]], marker=OBSERVED_MARKER,
                            markersize=6, markerfacecolor="none",
                            markeredgecolor=OBSERVED_COLOR, color=OBSERVED_COLOR,
                            linestyle="none", zorder=2)
                    ax.annotate("obs {}".format(_n_label(obs_row)),
                                (x, obs_row["estimate"]),
                                textcoords="offset points", xytext=(6, -10),
                                fontsize=5.5, color=OBSERVED_COLOR)
                    if pd.notna(obs_row.get("n_obs")):
                        n_observed.add(int(obs_row["n_obs"]))

                row = select_one(cells, domain=domain,
                                 term=TERM_PRED[(gender, domain)])
                if row is None:
                    flags.append("{}/{} predicted: row absent".format(domain, gender))
                    continue
                if pd.notna(row.get("n_obs")):
                    n_predicted.add(int(row["n_obs"]))
                draw_estimate(ax, row, x, color, GENDER_MARKERS[gender],
                              orientation="v", flags=flags,
                              tag="{}/{} predicted".format(domain, gender),
                              annotate_n=True, unit_scale=True)

            # 域内的性别差直接读结果表的 gap 行，不在图里自己相减：
            # 恒等式虽然成立，但读表里的那一行才能顺带把区间一起写出来
            gap_row = select_one(inter, outcome=outcome, domain=domain,
                                 term=TERM_GAP[domain], model=model)
            male_cell = select_one(cells, domain=domain,
                                   term=TERM_PRED[("male", domain)])
            female_cell = select_one(cells, domain=domain,
                                     term=TERM_PRED[("female", domain)])
            if (gap_row is not None and male_cell is not None
                    and female_cell is not None
                    and pd.notna(male_cell["estimate"])
                    and pd.notna(female_cell["estimate"])):
                y_male = male_cell["estimate"]
                y_female = female_cell["estimate"]
                ax.plot([i - 0.13, i + 0.13], [y_female, y_male],
                        color="#999999", linewidth=0.8, linestyle=":", zorder=1)
                text = "gap {:+.3f}".format(gap_row["estimate"]) \
                    if pd.notna(gap_row["estimate"]) else "gap: no estimate"
                ax.annotate(text, (i, (y_male + y_female) / 2),
                            textcoords="offset points", xytext=(12, 0), fontsize=7,
                            color="#333333")

        # 头条：跨域差中差（domain="both"），三层并列写出
        lines = ["Model-adjusted (GEE, cluster-robust by user);"
                 " cells predicted at {}:".format(model)]
        for layer in MODEL_LAYERS:
            lines.append("  " + did_annotation(inter, outcome, layer))
        for domain in DOMAIN_ORDER:
            row = select_one(inter, outcome=outcome, domain=domain,
                             term=TERM_GAP[domain], model=model)
            if row is None:
                continue
            if pd.isna(row["estimate"]):
                lines.append("  {} gap ({}): 无估计（{}）".format(
                    DOMAIN_LABELS[domain], model, row["note"] or "no note"))
            else:
                lines.append("  {} gap ({}) = {:+.3f} [{:+.3f}, {:+.3f}]".format(
                    DOMAIN_LABELS[domain], model, row["estimate"],
                    row["ci_low"], row["ci_high"]))
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="#f5f5f5",
                          edgecolor="#cccccc", linewidth=0.5))

        ax.set_xticks(range(len(facets)))
        ax.set_xticklabels([DOMAIN_LABELS[d] for d in facets])
        ax.set_xlim(-0.6, len(facets) - 0.4)
        # 两种点来自两个不同的总体，分母与样本量都必须分别写出来：
        # 预测点来自交互模型的样本（两域配对完整的用户，长表两行/人），
        # 观测点来自表 2 的主分母口径（按性别各自的全体）。只写一个分母
        # 会让读者把 n=226,333 的预测点当成 n=52,111/174,222 的观测点。
        ax.set_ylabel(
            "{}\n● / ▲ model-predicted at {} (interaction sample, complete "
            "domain pairs; {}); 95% CI\n□ observed (Table 2, {}; {})".format(
                OUTCOME_LABELS[outcome], model, _n_range_label(n_predicted),
                DENOMINATOR_LABELS[denominator], _n_range_label(n_observed)),
            fontsize=6.5)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        # 图例与缺陷说明都放到轴下方：面板内部已经被四个格子、连线和差中差
        # 标注框占满，硬塞进去只会盖住数据。图例用固定代理句柄，不从画出来的
        # artist 里取——某一格恰好是 ci_unavailable 时，取到的会是一个空心键。
        ax.legend(handles=_gender_legend_handles(" (predicted)",
                                                 with_observed=True),
                  fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.09),
                  ncol=3, frameon=False)
        _annotate_missing(ax, flags, prefix="flagged cells", y=-0.34, va="top")

    fig.suptitle(
        "Figure 3. Model-predicted gender × domain cells with the "
        "difference-in-differences, {}".format(year), fontsize=11)
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
    assert_manifest(data_dir, year)
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
                row = select_one(rows, domain=domain,
                                 term="{}_{}".format(ptype, gender))
                if row is None:
                    missing.append("{}/{}/{}: row absent".format(
                        domain, gender, ptype))
                    values.append(np.nan)
                    continue
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

def predicted_combo_cells(frame, model=HEADLINE_LAYER):
    """取出多项 logit 里各性别在四类组合上的模型预测概率（八行）

    这些行由 models_combination 与各类别的性别 AME 在同一次拟合、同一批
    反事实预测上算出，落行前断言过 pred_male[c] - pred_female[c] == AME[c]
    （容差 1e-10），所以图 5 左右两半画的是同一次拟合的两种写法。
    它们的 domain 一律是 "both"：来源组合是跨领域的分类。
    """
    rows = select(frame, outcome="source_combo", domain=DOMAIN_BOTH, model=model)
    terms = ["pred_{}_{}".format(g, c)
             for g in GENDER_ORDER for c in COMBO_CATEGORIES]
    return rows[rows["term"].astype("object").isin(terms)]


def fig5_combinations(year=config.YEAR, data_dir=None, fig_dir=FIG_DIR,
                      model=HEADLINE_LAYER):
    """四类来源组合：观测分布 + 多项 logit 预测概率 + 各层性别边际效应（§12.5）

    三个面板，从"数据是什么样"一路走到"控制掉活动量之后还剩多少"：

    1. **观测分布**：各性别在四类组合上的占比（Wilson 区间），来自导出层的
       combo_distribution.parquet（表 C 已有的 source_combo 列做一次分组占比）。
    2. **模型预测概率**（§12.5 明确要求的那一半）：多项 logit 在
       `pred_{male,female}_{类别}` 上的预测概率，默认 M1 层，`model=` 可换层。
       这些行与右面板的 AME 来自同一次拟合，落行前被断言过
       pred_male[c] − pred_female[c] == AME[c]，因此中间面板两点之差与右面板
       同一层的点严格一致——读者可以自己在图上量。
    3. **性别 AME 的森林图**：四类各自的男−女差在 M0/M1/M2 上怎么移动。
       保留它是因为"控制活动量之后差异缩水多少"是 §6.5 要读者看的东西，
       而这在两条预测概率上看不出来（预测概率只画一层）。

    三个面板都写在 domain="both" 的行上——来源组合是跨领域的分类，
    这张图里没有按领域分面这回事。观测占比与预测概率是两个不同的量，
    面板标题与横轴标签分别写明，不让读者去猜。
    """
    data_dir = data_dir or figure_data_dir()
    assert_manifest(data_dir, year)
    observed = _read(data_dir, COMBO_FILE, RESULT_COLUMNS)
    multinomial = _read(data_dir, "combination_multinomial.parquet", RESULT_COLUMNS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- 面板 1：观测分布 ---
    ax = axes[0]
    flags = []
    for i, category in enumerate(COMBO_CATEGORIES):
        for gender in GENDER_ORDER:
            cell = select_one(observed, term="{}_{}".format(category, gender))
            y = i + (0.14 if gender == "male" else -0.14)
            if cell is None:
                flags.append("{}/{}: row absent".format(category, gender))
                continue
            draw_estimate(ax, cell, y, GENDER_COLORS[gender],
                          GENDER_MARKERS[gender], orientation="h", flags=flags,
                          tag="{}/{} observed".format(category, gender),
                          annotate_n=True, unit_scale=True)
    ax.set_yticks(range(len(COMBO_CATEGORIES)))
    ax.set_yticklabels([COMBO_LABELS[c] for c in COMBO_CATEGORIES])
    ax.set_ylim(-0.6, len(COMBO_CATEGORIES) - 0.4)
    ax.set_xlabel("Observed share of users (denominator: all users of that gender\n"
                  "with a known source combination; 95% Wilson CI)", fontsize=7)
    ax.set_title("Observed distribution", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.legend(handles=_gender_legend_handles(), fontsize=6.5, loc="lower right")
    _annotate_missing(ax, flags, prefix="flagged rows")

    # --- 面板 2：多项 logit 的预测概率 ---
    ax = axes[1]
    flags = []
    cells = predicted_combo_cells(multinomial, model)
    n_pred = set()
    for i, category in enumerate(COMBO_CATEGORIES):
        for gender in GENDER_ORDER:
            row = select_one(cells, term="pred_{}_{}".format(gender, category))
            y = i + (0.14 if gender == "male" else -0.14)
            if row is None:
                flags.append("{}/{} predicted: row absent".format(category, gender))
                continue
            if pd.notna(row.get("n_obs")):
                n_pred.add(int(row["n_obs"]))
            draw_estimate(ax, row, y, GENDER_COLORS[gender],
                          GENDER_MARKERS[gender], orientation="h", flags=flags,
                          tag="{}/{} predicted".format(category, gender),
                          unit_scale=True)
    ax.set_yticks(range(len(COMBO_CATEGORIES)))
    ax.set_yticklabels([COMBO_LABELS[c] for c in COMBO_CATEGORIES])
    ax.set_ylim(-0.6, len(COMBO_CATEGORIES) - 0.4)
    ax.set_xlabel("Multinomial logit predicted probability at {} ({};\n"
                  "denominator: users with a known source combination; 95% CI)".format(
                      model, _n_range_label(n_pred)), fontsize=7)
    ax.set_title("Model-predicted probabilities ({})".format(model), fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.legend(handles=_gender_legend_handles(" (predicted)"), fontsize=6.5,
              loc="lower right")
    _annotate_missing(ax, flags, prefix="flagged cells")

    # --- 面板 3：各层的性别 AME ---
    ax = axes[2]
    flags = []
    combo_rows = select(multinomial, outcome="source_combo", domain=DOMAIN_BOTH)
    for i, category in enumerate(COMBO_CATEGORIES):
        for k, layer in enumerate(MODEL_LAYERS):
            row = select_one(combo_rows, model=layer,
                             term="gender_male_ame_{}".format(category))
            offset = 0.22 * (1 - k)
            if row is None:
                flags.append("{}/{}: row absent".format(category, layer))
                continue
            color = "#777777" if pd.isna(row["estimate"]) else (
                MALE_COLOR if row["estimate"] >= 0 else FEMALE_COLOR)
            draw_estimate(ax, row, i + offset, color, LAYER_MARKERS[layer],
                          orientation="h", flags=flags,
                          tag="{}/{} AME".format(category, layer))
            if pd.notna(row["estimate"]):
                ax.annotate(layer, (row["estimate"], i + offset),
                            textcoords="offset points", xytext=(7, -2), fontsize=6,
                            color="#555555")
    ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(COMBO_CATEGORIES)))
    ax.set_yticklabels([COMBO_LABELS[c] for c in COMBO_CATEGORIES])
    ax.set_ylim(-0.6, len(COMBO_CATEGORIES) - 0.4)
    ax.set_xlabel("Gender AME (male − female), probability scale\n"
                  "(denominator: users with a known source combination; 95% CI)",
                  fontsize=7)
    ax.set_title("Gender difference across model layers", fontsize=9)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    _annotate_missing(ax, flags, prefix="flagged rows")

    handles = [Line2D([], [], color="#555555", marker=LAYER_MARKERS[layer],
                      linestyle="none", label=LAYER_LABELS[layer])
               for layer in MODEL_LAYERS]
    handles += [Line2D([], [], color="#555555", marker="o", linestyle="none",
                       markerfacecolor="none", markeredgewidth=1.6,
                       label="No CI available"),
                Line2D([], [], color="#b22222", marker="x", linestyle="none",
                       label="No estimate")]
    fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=5,
               frameon=False)
    fig.suptitle("Figure 5. Source-combination categories by gender, {}".format(year),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
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

    **每个月的成色按 estimate_status 判定，与图 1/2/3/5 同一条规则。**
    这张图画的是折线 + 区间带，不是逐点的误差棒，所以带子用
    `where=` 只覆盖区间齐全的月份；区间缺失（ci_unavailable）与整点缺失
    （missing）的月份改走 draw_estimate，分别画成空心点 + "CI n/a" 与
    红色 ✕ 缺口，并列进图上的说明。在一条连续曲线上，一个"有点估计、
    没有区间"的月份如果不标出来，看起来与别的点毫无区别——那正是
    §12 反复要避免的"缺失的不确定性被读成确定"。
    """
    data_dir = data_dir or figure_data_dir()
    assert_manifest(data_dir, year)
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

                # 每个月单独判成色，与图 1/2/3/5 用的是同一条规则
                # （estimate_status）。这张图原来自己画 fill_between、
                # 自己只检查估计值是否有限，于是"有点估计、区间是 NaN"的
                # 月份会画成一个光秃秃的点：既没有带子，也没有 "CI n/a" 的
                # 标记——在一条连续曲线上，那个点看起来和别的点毫无区别，
                # 读者没有任何办法知道它的不确定性是缺失的。
                statuses = [estimate_status(row) for _, row in series.iterrows()]
                banded = np.array([s in ("ok", "clipped") for s in statuses])
                # where= 让缺界的月份在带子上留一个缺口，而不是让 NaN 被
                # 相邻月份的界脑补过去
                ax.fill_between(months, low, high, where=banded, color=color,
                                alpha=0.15, linewidth=0)
                for (_, row), month, status in zip(series.iterrows(), months,
                                                   statuses):
                    if status in ("ok", "clipped"):
                        continue
                    # ci_unavailable -> 空心点 + "CI n/a"；missing -> 红色 ✕ 缺口
                    draw_estimate(ax, row, month, color,
                                  GENDER_MARKERS[gender], orientation="v",
                                  flags=missing,
                                  tag="{} month={:02d}".format(gender, month))

                unusual = flag_unusual_months(values)
                if unusual.any():
                    ax.scatter(months[unusual], values[unusual], s=90,
                               facecolors="none", edgecolors="#b22222",
                               linewidths=1.2, zorder=5)
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
    assert_manifest(data_dir, year)
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
