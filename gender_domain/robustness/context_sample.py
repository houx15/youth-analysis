"""
§13.4 语境效度抽样：词表命中是不是真的在讨论该话题，代码本身回答不了。

一条帖子命中词表 "疫情"，可能是在认真讨论疫情，也可能只是"这波疫情耽误了
我追星"里的一句带过——命中与否是机器判定的，命中是不是"内容参与"要靠人读。
本模块只做两件事，都是为了让这件本该由人判断的事真正被人判断到：

1. `draw`：从全年真实命中帖里，按 domain × gender × post_type × density
   分层抽一份给人工编码的样本，带着足够的语境（清洗后正文、命中词及其
   在正文里的位置）和空白的编码列；
2. `summarise`：编码回收后（哪怕只编了一部分），报告每个分层格的假阳性率
   （Wilson 区间），以及本模块存在的真正理由——**假阳性率会不会随性别
   系统性不同**。词表对某一种性别的帖子更容易误判，会直接把偏差灌进论文
   的头条性别对比，而这件事不会在任何自动化指标里露出破绽，只有人工编码
   配上这里的性别差异检验才看得见。

--------------------------------------------------------------------------
抽样池：复用 incidence 的帖子×词矩阵，不重新扫一遍正文或重新解析 term_counts
--------------------------------------------------------------------------
`incidence.build_post_term_incidence(year, domain)` 已经把表 A 的
`{domain}_term_counts` 解析进了一个 帖子×词 稀疏矩阵（解析用的正是
`build_post_table.decode_term_counts`，见该模块）。本模块直接复用这个矩阵：

- **命中帖**就是矩阵里 `matrix_row != incidence.NO_MATRIX_ROW` 的那些行；
- **密度**（`{domain}_density` 的口径）用 `matrix.dot(词长向量) / n_chars`
  现算，与 `build_post_table.process_frame` 写进表 A 的那一列在全词表下
  逐帖相等（Task 2 的核心测试已经保证这一点）；
- **命中词及次数**直接从矩阵那一行的非零元素读出来（矩阵本身就是
  decode_term_counts 解出来的结果），不用再读一遍表 A 的 term_counts
  字符串列——那一列已经在建矩阵时被读过一次并原地丢弃，为的就是不让
  它在内存里占两份；重新读第二遍纯属浪费。

这样做的直接好处：全年表 A 只被扫一遍（建矩阵那一遍），不会因为本模块
另起一次 `glob` + `read_parquet` 而重复一遍同等规模的 IO。

性别不在这份帖子×词矩阵里（矩阵只知道帖子和词，不知道用户属性）——性别的
权威来源只有一个，是表 C（`user_domain_{year}.parquet`），本模块按 user_id
关联过去，不从表 A 的原始 gender 列另开一条口径。

原始正文只在最后一步、只对**抽中的那几百条**帖子读一次，复用
`vocabulary.load_post_texts`（同一模块已有的、按月份+weibo_id 定向读取
cleaned_weibo_cov 的实现），不再写第二个读正文的函数。

--------------------------------------------------------------------------
分层与再现性
--------------------------------------------------------------------------
分层格是 (domain, gender, post_type, density_tercile) 的每一个非空组合，
格子由数据本身决定（有实际命中帖的组合才存在），不是预先枚举的笛卡尔积——
一个真实数据里从未出现过的组合不该凭空造出一行"0 条"的格子。

密度三分位在**每个 domain 内部、跨性别跨帖子类型合并计算**（分位点只算
一次，不按性别分别切）：与 §13.2 "阈值只能在合并样本上算"是同一条纪律，
理由也相同——按性别分别切三分位会让"高密度"在两个性别组里对应不同的
密度值，抽出来的高密度格子在两组之间就不可比了。

每个格子的抽样种子由 `(seed, domain, gender, post_type, tercile)` 用
SHA-256 派生（不用 Python 内置 `hash()`——那个在跨进程时不保证一致），
同一个 seed 无论在哪台机器上跑、无论组内数据的读取顺序如何（本模块在
分组前先按 weibo_id 排序，见 `_load_hit_posts`），抽出来的样本都逐字节
相同——这是"第二个编码员能拿到完全相同样本"这条要求的唯一保证方式。

`seed` 写进结果表的每一行（哪怕整个抽样过程没有随机性以外的分支），
与本套件其余模块"seed 永远显式记录，不因为看起来是确定性的就省略"
同一条纪律。

--------------------------------------------------------------------------
编码列与 summarise 的口径
--------------------------------------------------------------------------
`draw` 产出三个空白编码列，人工在导出的表里逐行填：
- `is_substantive`：这条命中是不是在实实在在地讨论该话题（人工的主判断）；
- `is_false_positive`：这次词表命中是不是误判（summarise 就读这一列算
  假阳性率——两列分开放是因为让编码员在两个问题上分别给出判断，比只给
  一个笼统的"对不对"更不容易在模糊案例上想当然）；
- `notes`：自由文本，记边界案例、争议判断的理由。

三列一律留空（`pandas` 的 `pd.NA`，用可空 string dtype），不预填任何猜测——
预填等于让编码员对着模型的建议点头，那不是独立编码。

`summarise` 必须能处理**只编码了一部分**的文件：覆盖率（`coverage`）本身
就是输出的一部分，而不是"文件不完整就报错"。假阳性率与 Wilson 区间
（`stats_utils.proportion_ci`）只在"已编码"的行上算，未编码的行只计入
`n_drawn`、不计入 `n_coded`。

最后一组行是性别差异检验（`proportion_diff_ci`，Newcombe 方法，与
`stats_utils` 其余比例差异检验同一个函数）：整体一行，每个 domain 各一行。
这是本模块存在的真正理由——报告说"论文的性别对比对词表误判稳健"之前，
必须先把"误判本身有没有性别差异"这件事量出来，而不是假设它是均匀的。
"""

import hashlib
import os

import fire
import numpy as np
import pandas as pd

from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import id_rules as ir
from gender_domain import stats_utils as su
from gender_domain import text_rules as tr
from gender_domain.robustness import incidence as inc
from gender_domain.robustness import vocabulary as voc

DOMAINS = but.DOMAINS

# 抽样结果表的列。is_substantive / is_false_positive / notes 三列留给人工，
# 一律为空。seed 写在每一行，供第二个编码员核对"这确实是同一次抽样"。
DRAW_SCHEMA = (
    "domain",
    "weibo_id",
    "user_id",
    "month",
    "gender",
    "post_type",
    "n_chars",
    "density",
    "density_tercile",
    "matched_terms",
    "text",
    "text_highlighted",
    "cell_n_available",
    "cell_n_drawn",
    "seed",
    "is_substantive",
    "is_false_positive",
    "notes",
)

# 人工编码后 summarise 产出的汇总表列。level 区分三类行：
#   "overall"     全体一行（不分层）
#   "cell"        每个非空分层格一行
#   "gender_diff" 性别差异检验，整体一行 + 每个 domain 各一行
# 三类行共用同一套列，不适用的字段留 NaN——与 harness/stats_utils 一贯的
# "缺字段是 NaN，不是省略这一列"是同一条纪律。
SUMMARY_SCHEMA = (
    "level",
    "domain",
    "gender",
    "post_type",
    "density_tercile",
    "n_drawn",
    "n_coded",
    "coverage",
    "n_false_positive",
    "false_positive_rate",
    "ci_low",
    "ci_high",
    "rate_male",
    "rate_female",
    "diff",
    "diff_ci_low",
    "diff_ci_high",
    "note",
)

DENSITY_TERCILES = ("low", "mid", "high")

# 高亮标记：全角括号，视觉上比半角括号更容易和正文里可能出现的英文/数字
# 括号区分开，供人工编码时一眼定位命中词的位置。
_HIGHLIGHT_OPEN = "【"
_HIGHLIGHT_CLOSE = "】"


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def robustness_dir():
    """稳健性层唯一允许写入的目录"""
    return os.path.join(config.OUTPUT_DIR, "robustness")


def results_path():
    return os.path.join(robustness_dir(), "context_sample.parquet")


def manifest_dir(year):
    return os.path.join(robustness_dir(), "context_sample_{}".format(year))


# ---------------------------------------------------------------------------
# 确定性子种子
# ---------------------------------------------------------------------------

def _stable_int(text):
    """字符串 -> 确定性整数，不依赖进程内 hash() 随机化

    Python 内置 hash() 对字符串默认逐进程随机化（PYTHONHASHSEED），同一个
    分层键在两次运行里可能派生出不同的子种子——"第二个编码员能拿到同一份
    样本"这条要求就无从谈起。SHA-256 摘要与进程无关，是稳定的。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _cell_seed(seed, domain, gender, post_type, tercile):
    """一个分层格的抽样种子，由主 seed 与格子标签共同派生

    用 SeedSequence 而不是拼接后直接当种子：保证同一个主 seed 下，不同
    格子之间的抽样互不相关（不会出现"两个格子碰巧抽出同一个排列"这种
    看起来像 bug 的巧合）。
    """
    cell_key = "{}|{}|{}|{}".format(domain, gender, post_type, tercile)
    return int(
        np.random.SeedSequence([int(seed), _stable_int(cell_key)]).generate_state(1)[0]
    )


# ---------------------------------------------------------------------------
# 抽样池：命中帖 + 密度 + 性别
# ---------------------------------------------------------------------------

def _load_genders(year):
    """表 C 的 user_id -> gender，性别的唯一权威来源"""
    path = os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year))
    if not os.path.exists(path):
        raise FileNotFoundError("未找到表 C: {}".format(path))
    genders = pd.read_parquet(path, engine="pyarrow", columns=["user_id", "gender"])
    genders["user_id"] = ir.normalize_id_series(genders["user_id"])
    return genders.drop_duplicates(subset=["user_id"]).reset_index(drop=True)


def _load_hit_posts(year, domain, genders):
    """一个领域的命中帖工作帧：来自 incidence 的矩阵，密度现算，性别关联表 C

    Returns:
        (incidence, hit_posts)；hit_posts 为空时 incidence 仍然是真实建好的
        矩阵（调用方不需要用到它），只是没有任何命中帖可抽。
    """
    incidence = inc.build_post_term_incidence(year, domain)
    posts = incidence.posts
    hit_mask = posts["matrix_row"].to_numpy() != inc.NO_MATRIX_ROW
    hit_posts = posts.loc[hit_mask].reset_index(drop=True).copy()
    if len(hit_posts) == 0:
        return incidence, hit_posts

    lengths = inc.term_length_vector(incidence)
    row_chars_hit = incidence.matrix.dot(lengths).astype(np.float64)
    matrix_rows = hit_posts["matrix_row"].to_numpy()
    n_chars = hit_posts["n_chars"].to_numpy(dtype=np.float64)
    chars_hit = row_chars_hit[matrix_rows]
    density = np.divide(
        chars_hit, n_chars, out=np.zeros_like(n_chars), where=n_chars > 0
    )
    hit_posts["density"] = density
    hit_posts["weibo_id"] = hit_posts["weibo_id"].astype(str)
    hit_posts["user_id"] = hit_posts["user_id"].astype(str)
    hit_posts["post_type"] = hit_posts["post_type"].astype(str)

    merged = hit_posts.merge(genders, on="user_id", how="left")
    n_missing = int(merged["gender"].isna().sum())
    if n_missing:
        print(
            "警告: {} 领域有 {} 条命中帖在表 C 里找不到对应用户的性别，"
            "已从抽样池中剔除（不计入任何分层格）".format(domain, n_missing)
        )
    merged = merged[merged["gender"].notna()].reset_index(drop=True)
    return incidence, merged


def _assign_terciles(hit_posts):
    """按该领域全部命中帖的密度分布切三分位（合并全部性别/帖子类型计算分位点）"""
    hit_posts = hit_posts.copy()
    values = hit_posts["density"].to_numpy(dtype=float)
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    hit_posts["density_tercile"] = np.where(
        values <= q1, "low", np.where(values <= q2, "mid", "high")
    )
    return hit_posts


def _inverse_term_index(incidence):
    return {col: term for term, col in incidence.term_index.items()}


def _matched_terms_label(incidence, inv_index, matrix_row):
    """矩阵某一行的命中词与次数，拼成人读的 "词×次数; 词×次数" """
    row = incidence.matrix.getrow(int(matrix_row))
    counts = {inv_index[col]: int(cnt) for col, cnt in zip(row.indices, row.data)}
    return "; ".join("{}×{}".format(term, counts[term]) for term in sorted(counts))


def _highlight(text, vocab):
    """在清洗后正文里，把命中词框起来，供编码员一眼定位

    复用 text_rules.VocabMatcher（与产线判定命中同一套最左最长、区间不
    重叠的匹配逻辑），不自己另写一遍查找命中位置的代码。
    """
    if not text:
        return text
    matcher = tr.VocabMatcher(vocab)
    matches = matcher.find(text)
    if not matches:
        return text
    parts = []
    pos = 0
    for _term, start, end in matches:
        parts.append(text[pos:start])
        parts.append(_HIGHLIGHT_OPEN + text[start:end] + _HIGHLIGHT_CLOSE)
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def _draw_cell(rng_seed, group, n_per_cell):
    """一个分层格内的抽样：够 n_per_cell 就抽满，不够就整格全取"""
    n_available = len(group)
    take = min(int(n_per_cell), n_available)
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(n_available)
    picked = group.iloc[order[:take]].copy()
    picked["cell_n_available"] = n_available
    picked["cell_n_drawn"] = take
    return picked


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------

def draw(year=config.YEAR, n_per_cell=25, seed=0, domains=DOMAINS):
    """§13.4 分层抽样：domain × gender × post_type × density_tercile

    每个非空分层格最多抽 n_per_cell 条命中帖（不够就整格取完，格子本身
    由数据决定，见模块文档）。返回的每一行带着清洗后正文、命中词与位置
    高亮、这一格的可抽样总数与实抽数，以及三个留给人工的空白编码列。

    Args:
        year: 数据年份
        n_per_cell: 每个分层格的目标抽样数
        seed: 主随机种子，写进结果表每一行；同一个 seed 在同一份数据上
            逐字节可复现
        domains: 参与抽样的领域，默认两个都抽

    Returns:
        DataFrame，列为 DRAW_SCHEMA
    """
    genders = _load_genders(year)
    all_frames = []
    cell_log = []

    for domain in domains:
        incidence, hit_posts = _load_hit_posts(year, domain, genders)
        if len(hit_posts) == 0:
            print("警告: {} 领域没有任何命中帖，抽样池为空".format(domain))
            continue

        hit_posts = _assign_terciles(hit_posts)
        # 排序保证分组顺序、组内行序都与"文件读取顺序/分片顺序"无关，
        # 抽样的可复现性因此只依赖 seed，不依赖底层数据的物理排布
        hit_posts = hit_posts.sort_values("weibo_id").reset_index(drop=True)
        vocab = voc.load_vocabulary(domain, year)
        inv_index = _inverse_term_index(incidence)

        picked_cells = []
        grouped = hit_posts.groupby(
            ["gender", "post_type", "density_tercile"], sort=True
        )
        for (gender, post_type, tercile), group in grouped:
            group = group.reset_index(drop=True)
            rng_seed = _cell_seed(seed, domain, gender, post_type, tercile)
            picked = _draw_cell(rng_seed, group, n_per_cell)
            picked_cells.append(picked)
            cell_log.append(
                (domain, gender, post_type, tercile, len(group), len(picked))
            )

        sample = pd.concat(picked_cells, ignore_index=True)

        text_frame = sample[["weibo_id", "month"]].drop_duplicates()
        texts = voc.load_post_texts(year, text_frame)
        matched_terms = [
            _matched_terms_label(incidence, inv_index, row)
            for row in sample["matrix_row"]
        ]
        raw_text = [texts.get(wid, "") for wid in sample["weibo_id"]]
        highlighted = [_highlight(t, vocab) for t in raw_text]
        n = len(sample)

        out = pd.DataFrame({
            "domain": [domain] * n,
            "weibo_id": sample["weibo_id"].to_numpy(),
            "user_id": sample["user_id"].to_numpy(),
            "month": sample["month"].to_numpy(),
            "gender": sample["gender"].to_numpy(),
            "post_type": sample["post_type"].to_numpy(),
            "n_chars": sample["n_chars"].to_numpy(),
            "density": sample["density"].to_numpy(),
            "density_tercile": sample["density_tercile"].to_numpy(),
            "matched_terms": matched_terms,
            "text": raw_text,
            "text_highlighted": highlighted,
            "cell_n_available": sample["cell_n_available"].to_numpy(),
            "cell_n_drawn": sample["cell_n_drawn"].to_numpy(),
            "seed": [seed] * n,
        })
        for col in ("is_substantive", "is_false_positive", "notes"):
            out[col] = pd.array([pd.NA] * n, dtype="string")
        all_frames.append(out)

    if not all_frames:
        result = pd.DataFrame(columns=list(DRAW_SCHEMA))
    else:
        result = pd.concat(all_frames, ignore_index=True)
        result = result.sort_values(
            ["domain", "gender", "post_type", "density_tercile", "weibo_id"]
        ).reset_index(drop=True)

    print(
        "§13.4 抽样: 共抽取 {} 条帖子，覆盖 {} 个非空分层格（seed={}）".format(
            len(result), len(cell_log), seed
        )
    )
    for domain, gender, post_type, tercile, n_available, n_drawn in cell_log:
        print(
            "  {} / gender={} / post_type={} / density={}: "
            "抽 {}/{}".format(domain, gender, post_type, tercile, n_drawn, n_available)
        )
    return result[list(DRAW_SCHEMA)]


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "1.0", "true", "yes", "y", "是", "t"}
_FALSY = {"0", "0.0", "false", "no", "n", "否", "f"}


def _parse_code(value):
    """把编码列的任意常见写法解析成 True / False / None（尚未编码）

    None、NaN、空白字符串一律是"还没编码"，不是 False——一个真正编码成
    "不是假阳性"的行和一个还没人看过的行，语义完全不同，混在一起算覆盖率
    和假阳性率都会错。
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "" or text in ("nan", "none", "<na>"):
        return None
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ValueError(
        "无法识别的编码取值 {!r}，请使用 1/0、true/false、yes/no 或 是/否".format(value)
    )


def _rate_and_ci(n_success, n_total):
    if n_total == 0:
        return np.nan, np.nan, np.nan
    rate = n_success / n_total
    low, high = su.proportion_ci(n_success, n_total)
    return rate, low, high


def _empty_summary_row(**overrides):
    row = {col: np.nan for col in SUMMARY_SCHEMA}
    row.update(overrides)
    return row


def _gender_diff_row(domain_value, group):
    """给定一份（可能跨 domain 的）已编码数据，做一次 m vs f 的假阳性率差异检验"""
    male = group[group["gender"] == "m"]
    female = group[group["gender"] == "f"]
    n_male_coded = int(male["_coded"].notna().sum())
    n_female_coded = int(female["_coded"].notna().sum())
    n_drawn = len(male) + len(female)
    if n_male_coded == 0 or n_female_coded == 0:
        return _empty_summary_row(
            level="gender_diff",
            domain=domain_value,
            n_drawn=n_drawn,
            n_coded=n_male_coded + n_female_coded,
            note="性别 m/f 至少有一侧尚无编码，无法计算差异",
        )
    fp_male = int((male["_coded"] == True).sum())  # noqa: E712
    fp_female = int((female["_coded"] == True).sum())  # noqa: E712
    rate_male = fp_male / n_male_coded
    rate_female = fp_female / n_female_coded
    diff, low, high = su.proportion_diff_ci(
        fp_female, n_female_coded, fp_male, n_male_coded
    )
    return _empty_summary_row(
        level="gender_diff",
        domain=domain_value,
        n_drawn=n_drawn,
        n_coded=n_male_coded + n_female_coded,
        rate_male=rate_male,
        rate_female=rate_female,
        diff=diff,
        diff_ci_low=low,
        diff_ci_high=high,
        note="diff = female - male",
    )


_REQUIRED_COLUMNS = ("domain", "gender", "post_type", "density_tercile", "is_false_positive")


def summarise(coded_df):
    """§13.4 编码结果汇总：即便只编了一部分，也照常报告覆盖率

    Args:
        coded_df: draw() 的输出（或其子集/导出后编码回来的版本），至少要有
            domain / gender / post_type / density_tercile / is_false_positive
            这几列

    Returns:
        DataFrame，列为 SUMMARY_SCHEMA，三类行：
        - level="overall"：全体一行，不分层
        - level="cell"：每个在 coded_df 里出现过的分层格一行
        - level="gender_diff"：性别假阳性率差异检验，整体一行 + 每个
          domain 各一行——这是本模块存在的真正理由，见模块文档
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in coded_df.columns]
    if missing:
        raise ValueError(
            "summarise 收到的表缺少必需列 {}，无法判断这是不是 draw() 的输出"
            "（或其编码后的版本）".format(missing)
        )

    work = coded_df.copy()
    work["_coded"] = work["is_false_positive"].map(_parse_code)
    rows = []

    total_n = len(work)
    total_coded = int(work["_coded"].notna().sum())
    total_fp = int((work["_coded"] == True).sum())  # noqa: E712
    rate, low, high = _rate_and_ci(total_fp, total_coded)
    rows.append(_empty_summary_row(
        level="overall",
        domain="both",
        n_drawn=total_n,
        n_coded=total_coded,
        coverage=(total_coded / total_n if total_n else np.nan),
        n_false_positive=total_fp,
        false_positive_rate=rate,
        ci_low=low,
        ci_high=high,
    ))

    group_cols = ["domain", "gender", "post_type", "density_tercile"]
    for key, group in work.groupby(group_cols, sort=True, dropna=False):
        domain, gender, post_type, tercile = key
        n_drawn = len(group)
        n_coded = int(group["_coded"].notna().sum())
        n_fp = int((group["_coded"] == True).sum())  # noqa: E712
        rate, low, high = _rate_and_ci(n_fp, n_coded)
        rows.append(_empty_summary_row(
            level="cell",
            domain=domain,
            gender=gender,
            post_type=post_type,
            density_tercile=tercile,
            n_drawn=n_drawn,
            n_coded=n_coded,
            coverage=(n_coded / n_drawn if n_drawn else np.nan),
            n_false_positive=n_fp,
            false_positive_rate=rate,
            ci_low=low,
            ci_high=high,
        ))

    rows.append(_gender_diff_row("both", work))
    for domain_value in sorted(work["domain"].dropna().unique()):
        rows.append(_gender_diff_row(domain_value, work[work["domain"] == domain_value]))

    return pd.DataFrame(rows, columns=list(SUMMARY_SCHEMA))


# ---------------------------------------------------------------------------
# 全流程
# ---------------------------------------------------------------------------

def build(year=config.YEAR, n_per_cell=25, seed=0):
    """抽样 -> 落盘 -> 写 manifest，供 `python -m ...context_sample build` 调用"""
    sample = draw(year=year, n_per_cell=n_per_cell, seed=seed)
    out_path = results_path()
    os.makedirs(robustness_dir(), exist_ok=True)
    sample.to_parquet(out_path, engine="pyarrow", index=False)
    print("已写出抽样结果: {}（{} 行）".format(out_path, len(sample)))

    if len(sample):
        cell_summary = (
            sample.groupby(
                ["domain", "gender", "post_type", "density_tercile"], sort=True
            )
            .agg(n_available=("cell_n_available", "first"),
                 n_drawn=("cell_n_drawn", "first"))
            .reset_index()
        )
        by_domain = sample.groupby("domain").size().to_dict()
    else:
        cell_summary = pd.DataFrame(
            columns=["domain", "gender", "post_type", "density_tercile",
                     "n_available", "n_drawn"]
        )
        by_domain = {}

    manifest = config.build_manifest(
        step="robustness_context_sample_{}".format(year),
        inputs=[
            "post_domain_measures_{}".format(year),
            os.path.relpath(
                os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year)),
                config.OUTPUT_DIR,
            ),
        ],
        params={"year": year, "n_per_cell": n_per_cell, "seed": seed},
        counts={
            "n_rows": int(len(sample)),
            "n_cells": int(len(cell_summary)),
            "by_domain": {str(k): int(v) for k, v in by_domain.items()},
            "cells": cell_summary.to_dict(orient="records"),
        },
        fingerprints={
            domain: config.fingerprint_terms(voc.load_vocabulary(domain, year))
            for domain in DOMAINS
        },
    )
    config.write_manifest(manifest, manifest_dir(year))
    return {"results_path": out_path}


if __name__ == "__main__":
    fire.Fire({"draw": draw, "build": build, "summarise": summarise})
