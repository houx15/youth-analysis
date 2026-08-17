"""
§13.5 来源账号稳健性：扩散口径的领域参与，有多依赖那一份账号名单里的
某几个账号。

本模块产出五类变体，全部通过 harness.estimate_all 估计同样的六个量：

1. **留一账号**（`run_leave_one_account_out`）：逐个剔除转发量最高的
   top_n 个账号；
2. **留一类别**（`run_leave_one_category_out`）：公共事务的 8 个机构类别、
   明星的 11 个认证关键词各剔一次；
3. **剔除头部账号**（`run_exclude_top_accounts`）：一次剔掉转发量前 1/5/10 个；
4. **只保留中央级官方媒体**（`run_core_national_media_only`）；
5. **账号 bootstrap**（`run_account_bootstrap`）：**按类别分层、对账号**
   有放回重抽。

--------------------------------------------------------------------------
账号侧没有词表那个重聚合难题——这一点必须写在这里
--------------------------------------------------------------------------
词表侧的重聚合会**恒向低估**：表 A 存的是"最左最长、区间不重叠"消解之后
的逐词计数，剔掉一个长词以后，被它掩盖过的短词在存量里根本不存在，于是
按存量重聚合会漏判命中（见 incidence 模块文档与 vocabulary 模块文档）。

**账号侧没有任何与之对应的问题。** 表 B 是逐（转发事件 × 命中账号）一行
的事件表：一条转发命中哪个来源账号是各账号独立判定的，账号之间不会互相
消解——同一条转发同时命中两个领域的账号时表 B 会各出一行，而不是"最长的
那个账号吃掉另一个"。因此"取账号子集"就是一次干净的列选择，重聚合与
"拿着削减后的名单重新扫一遍表 B"逐用户精确相等，不存在需要回原始层校准
的偏差，也就没有 vocabulary 里那一整套风险集与随机抽样检验。

写下这一段是为了让后来的读者不要把词表那边的告诫照搬过来：那边需要
校准是因为匹配会互相掩盖，这边不需要是因为事件是逐账号记录的，**不是
因为这边偷懒没做**。

--------------------------------------------------------------------------
§13.5 真正问的问题：不是"估计会不会抖"，而是"这是不是一个账号撑起来的"
--------------------------------------------------------------------------
因此本模块把"被剔掉了多少量"与估计值放在同等重要的位置：

- **逐变体记录被剔账号占该领域转发总量的份额**（`excluded_volume_share`）。
  "剔掉头号账号结论还在"这句话，在读者知道那个账号占 2% 还是 83% 之前
  没有任何意义。
- **单独一张集中度表**（`account_concentration.parquet`）：各领域转发量
  前 1/5/10 个账号占比，**逐性别**。如果明星领域的扩散绝大部分来自一个
  艺人的粉丝群，那首先是一条关于"明星参与这个测量到底测了什么"的发现，
  其次才是一次稳健性检验。集中度同时按"该性别自己的排名"和"两性合并的
  排名"各算一份：前者回答"这个性别有多集中"，后者回答"男女是不是被同一
  批账号带动的"，只报一个答不出另一个。
- **top_n 截断必须自报覆盖面**（`considered_volume_share`）：上千个账号
  跑不完全部留一，只能取转发量最高的一批，但一个不声明覆盖面的 top-N
  就是一次自我夸大的稳健性检验。

--------------------------------------------------------------------------
最容易产出错误数字的一处：被剔账号的唯一来源用户必须变成非参与者
--------------------------------------------------------------------------
一个用户在某领域的全部来源转发都来自被剔掉的那个账号时，他在这个变体下
**必须**变成"没有进入过该领域"。这件事不能假定聚合会自动处理：本模块
每个变体都重新走 `incidence.source_by_user`（计数 > 0 才算进入），再左
连接回表 C 并把没有事件的用户填成 0/False，然后把
`n_users_lost_entry`（连同男女各多少）写进诊断表——它既是这条要求的落地
方式，也是读者判断"这个变体到底动了多少人"的依据。

--------------------------------------------------------------------------
类别信息从哪来，以及本机没有名单文件时怎么办
--------------------------------------------------------------------------
权威来源是 `configs/news_user_ids.json` 与 `configs/entertain_user_ids.json`
（服务器上由 get_news_ids.py / get_entertain_ids.py 从 merged_profiles
生成）。它们不在版本库里，本机通常没有：此时本模块**打印一条说清楚
的中文提示，退回用表 B 自带的 source_category 列还原类别**，而不是抛一个
看不懂的 FileNotFoundError。表 B 的这一列本来就是建表时用同一份名单打
上去的，两条路径的类别口径一致，差别只是退回路径看不见"全年一次都没被
转发过的账号"——那些账号对任何一个量都是惰性的。

**两处口径必须分清楚**：

- **留一类别按"成员资格"剔除**：一个账号同时属于 central_news 与
  org_news（或认证理由里同时有"演员"和"歌手"）时，剔掉其中任一类别都会
  把它剔掉。名单本来就不是划分，硬把它当划分只会产生一个"剔了类别 A
  却还留着 A 的账号"的假结果。诊断表用
  `n_accounts_excluded_multi_category` 把这件事显式记下来。
- **bootstrap 的分层必须是一个真正的划分**，否则"分层抽样"没有定义。
  分层键因此取表 B 里那个拼接后的类别串（`"central_news|org_news"` 自成
  一层），每个账号恰好属于一层。

使用方法:
    python -m gender_domain.robustness.accounts build --year 2020 \
        --top_n 50 --n_replicates 200
"""

import os
import tempfile
from collections import namedtuple

import fire
import numpy as np
import pandas as pd

from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import id_rules as ir
from gender_domain import stats_utils as su
from gender_domain.robustness import harness
from gender_domain.robustness import incidence as inc
# replicate_seeds / _annotate_note 一律复用 vocabulary 的实现，不另写一份：
# 两个 family 的 replicate 种子派生规则一旦分叉，跨 family 的比较就不再
# 是同一件事；note 的拼接规则（追加而不覆盖拟合失败的留痕）同理。
from gender_domain.robustness import vocabulary as voc

DOMAINS = but.DOMAINS

VARIANT_FAMILY = "accounts"
# bootstrap 单独一个 family：它是几百个 replicate 的分布，与确定性变体
# 混在一个 family 里会让"变体之间的离散度"这种下游统计把两种东西加在一起。
BOOTSTRAP_FAMILY = "accounts_bootstrap"

DEFAULT_TOP_N = 50
DEFAULT_EXCLUDE_K = (1, 5, 10)
DEFAULT_N_REPLICATES = 200
DEFAULT_CONCENTRATION_K = (1, 5, 10)

# 类别信息的两个来源，写进 manifest，让读者知道这一次的类别是哪来的
CATEGORY_SOURCE_TABLE_B = "table_b_source_category"

# 表 C 里六个量真正用得到的列（与 vocabulary 保持同一份清单）。账号变体
# 只改 {domain}_source_count / {domain}_source_entered 两组列，其余原样
# 透传给 harness。
# 注意 {domain}_source_months / {domain}_source_share 不在这里：用户×账号
# 矩阵没有月份轴，重算不出前者；而六个量一个都不用它们（M0/M1 的协变量
# 是性别、发帖量、转发量、活跃天数、活跃月数）。需要它们的变体属于 §13.1
# 分母口径那一层，不是这里。
USER_TABLE_COLUMNS = list(voc.USER_TABLE_COLUMNS)

# 诊断表 schema。逐 (variant_family, variant_label, domain) 一行，分四组：
# 1) 身份；2) 这次剔了哪些账号；3) **被剔掉的转发量占该领域的份额**——
# §13.5 的头号数字；4) 这个变体让多少用户不再算参与者（男女分开）。
# n_accounts_considered / considered_volume_share 只有做了 top-N 截断的
# 家族才有值，其余家族是 NaN，而不是填一个看起来像"全都看过了"的数。
DIAGNOSTIC_SCHEMA = (
    "variant_family",
    "variant_label",
    "replicate",
    "seed",
    "domain",
    "n_accounts_total",
    "n_accounts_retained",
    "retained_fraction",
    "n_accounts_excluded",
    "n_accounts_excluded_multi_category",
    "excluded_accounts",
    "n_events_total",
    "n_events_retained",
    "excluded_volume_share",
    "n_accounts_considered",
    "considered_volume_share",
    "n_users_entered_baseline",
    "n_users_entered_variant",
    "n_users_lost_entry",
    "lost_entry_share",
    "n_users_lost_entry_male",
    "n_users_lost_entry_female",
    "note",
)

# 集中度表 schema。逐 (领域, 性别, k) 一行。
CONCENTRATION_SCHEMA = (
    "domain",
    "gender",
    "k",
    "n_accounts",
    "n_events",
    "n_users_entered",
    "top_k_share",
    "top_k_share_pooled_ranking",
    "top_accounts",
)

# excluded_accounts 列最多写几个账号 ID，超出的折成 "+N"。留一类别可能
# 一次剔掉上百个账号，整串写进 parquet 只会让这一列没法看。
MAX_LISTED_ACCOUNTS = 20

AccountContext = namedtuple(
    "AccountContext",
    [
        "year",
        "user_table",
        "incidence",
        "categories",
        "category_source",
        "strata",
        "volume",
        "baseline_frame",
    ],
)


# ---------------------------------------------------------------------------
# configs/cn_news_sources_lists.py 的八个机构类别
# ---------------------------------------------------------------------------

def _news_category_lists():
    """{类别名: 名单}，类别名与 get_news_ids.py 写进 JSON 的键一致（常量名小写）

    从 configs/cn_news_sources_lists.py 里反射取全部大写的 list 常量，而不是
    在本模块里再抄一遍八个名字：抄一遍就多一个会与上游悄悄分叉的地方。
    取不到（本机没有 configs 包）时返回空 dict 并打印提示，由调用方决定
    要不要退化——只有 run_core_national_media_only 真的需要它。
    """
    try:
        from configs import cn_news_sources_lists as cnsl
    except Exception as exc:  # noqa: BLE001 —— 缺配置不该炸掉整个模块导入
        print("警告: 无法导入 configs.cn_news_sources_lists（{}: {}），"
              "中央级媒体变体将无法定义".format(type(exc).__name__, exc))
        return {}
    return {
        name.lower(): getattr(cnsl, name)
        for name in dir(cnsl)
        if name.isupper() and isinstance(getattr(cnsl, name), list)
    }


NEWS_CATEGORY_LISTS = _news_category_lists()
NEWS_CATEGORY_NAMES = tuple(sorted(NEWS_CATEGORY_LISTS))

# "中央级官方媒体"取最窄的一种读法：中央级理论网站 + 中央级新闻媒体。
# org_news（部委与行业组织的账号）、gov_release_weibo（党政机构发布号）
# 同样是中央层级，但它们不是"新闻媒体"，把它们算进来会让这个变体和基线
# 差不多，也就检验不到什么。这是一个判断，不是一条事实，所以：类别名写进
# variant_label 与 note，并且是 run_core_national_media_only 的参数，
# 作者想换一种读法只要传一次参。
CORE_NATIONAL_CATEGORIES = ("central_theory_website", "central_news")


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def robustness_dir():
    """稳健性层唯一允许写入的目录"""
    return os.path.join(config.OUTPUT_DIR, "robustness")


def results_path():
    return os.path.join(robustness_dir(), "accounts.parquet")


def diagnostics_path():
    return os.path.join(robustness_dir(), "account_diagnostics.parquet")


def concentration_path():
    return os.path.join(robustness_dir(), "account_concentration.parquet")


def manifest_dir(year):
    return os.path.join(robustness_dir(), "accounts_{}".format(year))


# ---------------------------------------------------------------------------
# 类别与分层
# ---------------------------------------------------------------------------

def _categories_from_table_b(accounts, domain):
    """从表 B 的 source_category 列还原 {类别: {账号}}

    表 B 对跨类别账号存的是 "central_news|org_news" 这样的拼接串（见
    build_retweet_table.build_account_lookup），这里按 "|" 拆回成员资格。
    """
    out = {}
    subset = accounts[accounts["source_domain"] == domain]
    for account_id, category in zip(subset["r_user_id"], subset["source_category"]):
        for part in str(category).split("|"):
            part = part.strip()
            if part:
                out.setdefault(part, set()).add(str(account_id))
    return out


def load_account_categories(domain, accounts):
    """({类别: {账号}}, 这份类别是哪来的)

    优先用 configs 下的正式名单；本机没有那两个 JSON（它们在服务器上由
    merged_profiles 生成，不在版本库里）时打印一条说清楚的提示，退回表 B
    自带的 source_category 列。**绝不在这里抛 FileNotFoundError**：账号
    类别这件事表 B 里本来就有，为了一个缺失的配置文件让整个 §13.5 跑不动
    是没有道理的。

    两条路径都只保留"表 B 里真的出现过"的账号：名单里全年一次都没被转发过
    的账号对任何一个量都没有影响，留着它们只会让 retained_fraction 这类
    比例失去意义。
    """
    observed = set(
        accounts.loc[accounts["source_domain"] == domain, "r_user_id"].astype(str)
    )
    path = (
        config.PUBLIC_ACCOUNTS_FILE if domain == "public"
        else config.CELEBRITY_ACCOUNTS_FILE
    )
    try:
        raw = config.load_source_accounts(domain)
    except FileNotFoundError:
        print(
            "提示: 未找到来源账号名单 {}（服务器上由 get_news_ids.py / "
            "get_entertain_ids.py 从 merged_profiles 生成，不在版本库里）。"
            "已改用表 B 的 source_category 列还原类别——类别口径完全一致，"
            "只是看不到全年一次都没被转发过的账号。".format(path)
        )
        return _categories_from_table_b(accounts, domain), CATEGORY_SOURCE_TABLE_B

    categories = {}
    for category, ids in raw.items():
        kept = {str(uid) for uid in ids} & observed
        if kept:
            categories[category] = kept
    return categories, path


def account_strata(accounts, domain):
    """{分层键: [账号]}，分层键取表 B 里那个拼接后的类别串

    这是 bootstrap 唯一可用的分层：名单本身不是划分（一个账号可以同时属于
    多个类别），而分层抽样要求每个个体恰好属于一层。拼接串
    （"central_news|org_news"）自成一层，因此这个划分既是真的划分，又保留
    了"这个账号跨了哪几类"的信息。
    """
    subset = accounts[accounts["source_domain"] == domain]
    strata = {}
    for account_id, category in zip(subset["r_user_id"], subset["source_category"]):
        strata.setdefault(str(category), []).append(str(account_id))
    return {key: sorted(set(members)) for key, members in sorted(strata.items())}


# ---------------------------------------------------------------------------
# 上下文：矩阵、表 C、类别、逐账号转发量，建一次给所有变体用
# ---------------------------------------------------------------------------

def account_volume(incidence, user_table):
    """逐 (账号, 领域) 的转发事件数，并按性别拆开

    性别取自表 C。**在表 B 里出现、却不在表 C 里的用户**（有转发但全年
    没有留下帖子，因此没有进表 C）在 all 口径里照常计入、在 m/f 口径里
    都不计入——那是"性别未知"，不是"男也不是女"。
    """
    matrix = incidence.matrix
    gender_by_user = dict(zip(user_table["user_id"], user_table["gender"]))
    genders = np.array(
        [gender_by_user.get(uid, "") for uid in incidence.users], dtype=object
    )

    frame = incidence.accounts.copy()
    frame["r_user_id"] = frame["r_user_id"].astype(str)
    if matrix.shape[1] == 0:
        for column in ("n_events", "n_events_m", "n_events_f"):
            frame[column] = np.zeros(len(frame), dtype=np.int64)
        return frame

    frame["n_events"] = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    for gender in ("m", "f"):
        rows = np.flatnonzero(genders == gender)
        if rows.size:
            totals = np.asarray(matrix[rows].sum(axis=0)).ravel().astype(np.int64)
        else:
            totals = np.zeros(matrix.shape[1], dtype=np.int64)
        frame["n_events_{}".format(gender)] = totals
    return frame


def build_context(year=config.YEAR):
    """加载表 C、建用户×账号矩阵、解析类别与分层、算好逐账号转发量"""
    user_path = os.path.join(config.OUTPUT_DIR, "user_domain_{}.parquet".format(year))
    if not os.path.exists(user_path):
        raise FileNotFoundError(
            "未找到表 C: {}（由 python -m gender_domain.build_user_tables build "
            "生成）".format(user_path)
        )
    user_table = pd.read_parquet(user_path, columns=USER_TABLE_COLUMNS)
    user_table["user_id"] = ir.normalize_id_series(user_table["user_id"])
    print("表 C {:,} 个用户".format(len(user_table)))

    incidence = inc.build_user_account_incidence(year)
    volume = account_volume(incidence, user_table)

    categories = {}
    category_source = {}
    strata = {}
    for domain in DOMAINS:
        categories[domain], category_source[domain] = load_account_categories(
            domain, incidence.accounts
        )
        strata[domain] = account_strata(incidence.accounts, domain)
        n_events = int(volume.loc[volume["source_domain"] == domain, "n_events"].sum())
        print(
            "{} 领域: 账号 {} 个、类别 {} 个、分层 {} 层，转发事件 {:,} 条".format(
                domain,
                int((volume["source_domain"] == domain).sum()),
                len(categories[domain]),
                len(strata[domain]),
                n_events,
            )
        )

    context = AccountContext(
        year=year, user_table=user_table, incidence=incidence, categories=categories,
        category_source=category_source, strata=strata, volume=volume,
        baseline_frame=None,
    )
    baseline_frame = user_frame_for_subset(context, None)
    _check_baseline_matches_table_c(user_table, baseline_frame)
    return context._replace(baseline_frame=baseline_frame)


def _check_baseline_matches_table_c(user_table, baseline_frame):
    """全账号重建必须与表 C 逐用户相等，不等就大声说出来

    这是账号侧的"全集重建"检验（词表侧的同名检验在 incidence 里）：如果
    保留全部账号重新聚合都对不上表 C，那么每一个账号变体都在同一个方向上
    错，后面所有数字都不必看了。
    """
    for domain in DOMAINS:
        column = "{}_source_entered".format(domain)
        expected = user_table.set_index("user_id")[column].astype(bool)
        got = baseline_frame.set_index("user_id")[column].astype(bool)
        mismatch = int((expected.reindex(got.index) != got).sum())
        if mismatch:
            print(
                "警告: {} 的全账号重建与表 C 有 {} 个用户对不上——每一个账号"
                "变体都会在同一个方向上错，请先查 incidence 的账号侧重建".format(
                    domain, mismatch
                )
            )


# ---------------------------------------------------------------------------
# 用户帧：账号子集 / 账号重数 -> 逐用户来源计数与进入指示
# ---------------------------------------------------------------------------

def _merge_source(user_table, source):
    """把重算出来的来源计数贴回表 C，表 B 里没有的用户填 0/False

    填 0/False 与 build_user_tables.combine_user_table 完全一致，而且这
    正是"某用户的来源转发全来自被剔账号 -> 他在这个变体下不算参与者"
    这条要求的落地处：他在 source 里的计数会是 0，进入指示随之为 False；
    连一条事件都不剩、在 source 里整行消失时，左连接后同样是 0/False。
    """
    frame = user_table.copy()
    merged = frame[["user_id"]].merge(source, on="user_id", how="left")
    for domain in DOMAINS:
        count_col = "{}_source_count".format(domain)
        entered_col = "{}_source_entered".format(domain)
        frame[count_col] = (
            merged[count_col].fillna(0).to_numpy(dtype=np.int64)
        )
        frame[entered_col] = merged[entered_col].eq(True).to_numpy(dtype=bool)
    return frame


def user_frame_for_subset(context, account_subset):
    """保留 account_subset 这些账号时的用户帧（None 表示全部保留）"""
    source = inc.source_by_user(context.incidence, account_subset)
    return _merge_source(context.user_table, source)


def user_frame_for_weights(context, weights_by_domain):
    """按逐账号重数（bootstrap 抽到几次）重算的用户帧

    重数怎么进计数：按重数分档，每一档调用一次 incidence.source_by_user
    取该领域的计数列，再乘以档位求和——这是对既有接口的组合，而不是把
    "来源计数怎么算"再实现一遍。有放回抽 n 个账号时重数只有寥寥几档
    （1、2、3……），因此代价是每个 replicate 每个领域几次稀疏矩阵乘法。

    **重数只影响 {domain}_source_count，不影响进入指示**（抽到两次的账号
    不会让一个用户"更进入"）。六个预先设定量里，entry 与 DiD-entry 只看
    进入指示，topical 两个量与账号无关，所以账号 bootstrap 对这六个量的
    扰动**完全来自"哪些账号被抽到了"**，与重数无关。这不是实现上的取舍，
    是估计量本身的性质，写在这里免得读者以为 bootstrap 少做了一步。
    """
    frame = context.user_table.copy()
    n_users = len(frame)
    for domain in DOMAINS:
        weights = weights_by_domain.get(domain) or {}
        levels = {}
        for account_id, weight in weights.items():
            weight = int(weight)
            if weight > 0:
                levels.setdefault(weight, set()).add(str(account_id))

        counts = np.zeros(n_users, dtype=np.int64)
        for weight, members in sorted(levels.items()):
            source = inc.source_by_user(context.incidence, members)
            merged = frame[["user_id"]].merge(
                source[["user_id", "{}_source_count".format(domain)]],
                on="user_id", how="left",
            )
            counts += weight * merged[
                "{}_source_count".format(domain)
            ].fillna(0).to_numpy(dtype=np.int64)

        frame["{}_source_count".format(domain)] = counts
        frame["{}_source_entered".format(domain)] = counts > 0
    return frame


def subset_weights(context, domain, retained_ids):
    """{账号: 0/1}，覆盖该领域全部账号——诊断行需要"谁被剔了"的完整账本"""
    accounts = context.volume
    domain_ids = accounts.loc[
        accounts["source_domain"] == domain, "r_user_id"
    ].astype(str)
    return {
        account_id: (1 if retained_ids is None or account_id in retained_ids else 0)
        for account_id in domain_ids
    }


# ---------------------------------------------------------------------------
# 转发量、头部账号与集中度
# ---------------------------------------------------------------------------

def domain_volume(context, domain, genders=None):
    """{账号: 事件数}，genders 为 None 时是两性合计"""
    column = "n_events" if genders is None else None
    frame = context.volume[context.volume["source_domain"] == domain]
    if column is not None:
        values = frame["n_events"].to_numpy(dtype=np.int64)
    else:
        values = np.zeros(len(frame), dtype=np.int64)
        for gender in genders:
            values = values + frame["n_events_{}".format(gender)].to_numpy(
                dtype=np.int64
            )
    return {
        str(account_id): int(value)
        for account_id, value in zip(frame["r_user_id"], values)
    }


def top_accounts(context, domain, n, genders=None):
    """转发量最高的 n 个账号，同量时按账号 ID 升序——排序必须完全确定"""
    volume = domain_volume(context, domain, genders)
    ordered = sorted(volume, key=lambda a: (-volume[a], a))
    return ordered[: max(int(n), 0)]


def volume_share(context, domain, account_ids, genders=None):
    """这些账号占该领域（该性别）转发总量的份额；总量为 0 时是 NaN"""
    volume = domain_volume(context, domain, genders)
    total = sum(volume.values())
    if total == 0:
        return float("nan")
    return sum(volume.get(a, 0) for a in account_ids) / total


def concentration_by_gender(context, k=DEFAULT_CONCENTRATION_K):
    """各领域转发量前 k 个账号占比，逐性别

    两个份额并列，因为它们回答的不是同一个问题：

    - `top_k_share`：按**该性别自己的**转发量排名，前 k 个账号占该性别
      转发量的比例——"这个性别的扩散有多集中"；
    - `top_k_share_pooled_ranking`：按**两性合并**的排名取同一批账号，
      再算它们占该性别转发量的比例——"男女是不是被同一批账号带动的"。
      两个数差得远，说明两性各自集中在不同的账号上。

    份额的计算复用 stats_utils.top_share（传 q=k/账号数，它取
    ceil(q*n) 恰好等于 k），不另写一份排序求和。
    """
    rows = []
    for domain in DOMAINS:
        pooled_ranking = top_accounts(context, domain, max(k) if k else 0)
        for gender, genders in (("all", None), ("m", {"m"}), ("f", {"f"})):
            volume = domain_volume(context, domain, genders)
            values = np.array(list(volume.values()), dtype=float)
            n_accounts = len(values)
            entered = _entered_users(context.baseline_frame, domain, genders)
            for k_value in k:
                k_value = int(k_value)
                ordered = sorted(volume, key=lambda a: (-volume[a], a))[:k_value]
                share = (
                    su.top_share(values, k_value / n_accounts)
                    if n_accounts else float("nan")
                )
                rows.append({
                    "domain": domain,
                    "gender": gender,
                    "k": k_value,
                    "n_accounts": n_accounts,
                    "n_events": int(values.sum()),
                    "n_users_entered": int(entered.sum()),
                    "top_k_share": float(share),
                    "top_k_share_pooled_ranking": float(
                        volume_share(context, domain, pooled_ranking[:k_value], genders)
                    ),
                    "top_accounts": ",".join(ordered),
                })
    frame = pd.DataFrame(rows, columns=list(CONCENTRATION_SCHEMA))
    for _, row in frame[(frame["gender"] == "all") & (frame["k"] == 1)].iterrows():
        print(
            "集中度: {} 领域转发量最高的账号占 {:.1%}（{}）".format(
                row["domain"], row["top_k_share"], row["top_accounts"]
            )
        )
    return frame


def _entered_users(frame, domain, genders=None):
    """某领域、某性别的进入指示布尔数组（frame 为 None 时全 False）"""
    if frame is None:
        return np.zeros(0, dtype=bool)
    entered = frame["{}_source_entered".format(domain)].to_numpy(dtype=bool)
    if genders is None:
        return entered
    mask = frame["gender"].isin(list(genders)).to_numpy(dtype=bool)
    return entered & mask


# ---------------------------------------------------------------------------
# 诊断行
# ---------------------------------------------------------------------------

def _format_accounts(account_ids):
    """被剔账号列表压成一列字符串，超过 MAX_LISTED_ACCOUNTS 个折成 "+N" """
    ordered = sorted(account_ids)
    if len(ordered) <= MAX_LISTED_ACCOUNTS:
        return ",".join(ordered)
    return ",".join(ordered[:MAX_LISTED_ACCOUNTS]) + ",+{}".format(
        len(ordered) - MAX_LISTED_ACCOUNTS
    )


def _n_multi_category(context, domain, account_ids):
    """这些账号里有多少个同时属于该领域的多个类别"""
    categories = context.categories.get(domain, {})
    counts = {}
    for category, members in categories.items():
        for account_id in members:
            counts[account_id] = counts.get(account_id, 0) + 1
    return int(sum(1 for a in account_ids if counts.get(a, 0) > 1))


def diagnostics_row(context, domain, weights, variant_label, variant_frame,
                    replicate=0, seed=None, variant_family=VARIANT_FAMILY,
                    considered=None, note=None):
    """一个 (变体, 领域) 的诊断行

    Args:
        weights: {账号: 重数}，必须覆盖该领域的全部账号（0 表示被剔）
        variant_frame: 这个变体下的用户帧，用来数"多少人不再算参与者"
        considered: (看过的账号数, 这些账号占该领域的量)，只有做了 top-N
            截断的家族、且**正是这个领域**被截断时才传；其余情形留 NaN，
            而不是填一个看起来像"全都看过了"的数，更不能把另一个领域的
            覆盖面写到这个领域的行上
    """
    volume = domain_volume(context, domain)
    total_events = sum(volume.values())
    retained = {a for a, w in weights.items() if int(w) > 0}
    excluded = {a for a, w in weights.items() if int(w) <= 0}
    n_events_retained = int(sum(int(weights.get(a, 0)) * v for a, v in volume.items()))
    excluded_events = sum(volume.get(a, 0) for a in excluded)

    baseline_entered = _entered_users(context.baseline_frame, domain)
    variant_entered = _entered_users(variant_frame, domain)
    lost = baseline_entered & (~variant_entered)
    gender = context.user_table["gender"].to_numpy(dtype=object)

    return {
        "variant_family": variant_family,
        "variant_label": variant_label,
        "replicate": int(replicate),
        "seed": float(seed) if seed is not None else np.nan,
        "domain": domain,
        "n_accounts_total": len(weights),
        "n_accounts_retained": len(retained),
        "retained_fraction": len(retained) / len(weights) if weights else np.nan,
        "n_accounts_excluded": len(excluded),
        "n_accounts_excluded_multi_category": _n_multi_category(
            context, domain, excluded
        ),
        "excluded_accounts": _format_accounts(excluded),
        "n_events_total": int(total_events),
        "n_events_retained": n_events_retained,
        "excluded_volume_share": (
            excluded_events / total_events if total_events else np.nan
        ),
        "n_accounts_considered": (
            int(considered[0]) if considered is not None else np.nan
        ),
        "considered_volume_share": (
            float(considered[1]) if considered is not None else np.nan
        ),
        "n_users_entered_baseline": int(baseline_entered.sum()),
        "n_users_entered_variant": int(variant_entered.sum()),
        "n_users_lost_entry": int(lost.sum()),
        "lost_entry_share": (
            float(lost.sum() / baseline_entered.sum())
            if baseline_entered.sum() else np.nan
        ),
        "n_users_lost_entry_male": int((lost & (gender == "m")).sum()),
        "n_users_lost_entry_female": int((lost & (gender == "f")).sum()),
        "note": note,
    }


def _append_frame(rows, path, schema, label):
    """按给定 schema 增量落盘，与 harness.append_rows 同一套原子改名做法

    单独实现是因为这两张附表的 schema 都不是 ROBUSTNESS_SCHEMA：
    harness.append_rows 读旧文件时显式点名共享 schema 的列，用它写诊断表
    会把诊断列全丢掉。与 harness.append_rows 一样**不去重**：同一个 year
    跑两次 build() 会让表翻倍（重跑前先删旧文件）。
    """
    if path is None:
        return None
    frame = pd.DataFrame(list(rows), columns=list(schema))
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path, engine="pyarrow", columns=list(schema))
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.reset_index(drop=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".append_acc_tmp_", suffix=".parquet", dir=out_dir
    )
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print("已增量写入{}: {}（新增 {} 行，累计 {} 行）".format(
        label, path, len(frame), len(combined)))
    return path


def append_diagnostics(rows, path):
    return _append_frame(rows, path, DIAGNOSTIC_SCHEMA, "诊断")


def append_concentration(rows, path):
    return _append_frame(rows, path, CONCENTRATION_SCHEMA, "集中度")


# ---------------------------------------------------------------------------
# 跑一个变体：估计 + 诊断 + 落盘
# ---------------------------------------------------------------------------

def _volume_note(diagnostics, prefix=None):
    """把"这次剔了多少量"压成一句写进结果行的 note

    诊断表里有全套数字，但只拿到 accounts.parquet 一张表的读者也必须能看见
    这个变体剔掉了该领域多大一块量——"剔掉头号账号结论还在"这句话正是靠
    这个数字才有意义。
    """
    parts = [] if not prefix else [prefix]
    for row in diagnostics:
        parts.append("{}_excluded_volume_share={:.4f}".format(
            row["domain"], row["excluded_volume_share"]))
        parts.append("{}_lost_entry={}".format(row["domain"], row["n_users_lost_entry"]))
    return ";".join(parts)


def _run_variant(context, weights_by_domain, variant_label, out_path, diag_path,
                 replicate=0, seed=None, variant_family=VARIANT_FAMILY,
                 considered=None, note=None, user_df=None):
    """一个账号变体：换一份账号权重，估一次六个量，写两行诊断

    considered 是 {领域: (看过的账号数, 覆盖份额)}，只写在被截断的那个
    领域的行上——留一账号是逐领域做的，把公共事务的覆盖面写到明星那一行
    会让读者以为明星侧也只看了前几个账号。
    """
    if user_df is None:
        user_df = user_frame_for_weights(context, weights_by_domain)
    diagnostics = [
        diagnostics_row(
            context, domain, weights_by_domain[domain], variant_label, user_df,
            replicate=replicate, seed=seed, variant_family=variant_family,
            considered=(considered or {}).get(domain), note=note,
        )
        for domain in DOMAINS
    ]
    rows = harness.estimate_all(
        user_df, variant_family=variant_family, variant_label=variant_label,
        replicate=replicate, seed=seed,
    )
    # note 的拼接规则复用 vocabulary 的实现：绝不覆盖拟合失败的留痕
    rows = voc._annotate_note(rows, _volume_note(diagnostics, prefix=note))
    if out_path is not None:
        harness.append_rows(rows, out_path)
    append_diagnostics(diagnostics, diag_path)
    return rows


def _run_exclusion(context, excluded_ids, variant_label, out_path, diag_path,
                   considered=None, note=None):
    """确定性变体的公共动作：给一组被剔账号，其余全留"""
    all_ids = set(context.volume["r_user_id"].astype(str))
    retained = all_ids - set(excluded_ids)
    weights = {
        domain: subset_weights(context, domain, retained) for domain in DOMAINS
    }
    # 0/1 的情形一次列选择就够，不必按重数分档
    user_df = user_frame_for_subset(context, retained)
    return _run_variant(
        context, weights, variant_label, out_path, diag_path,
        considered=considered, note=note, user_df=user_df,
    )


# ---------------------------------------------------------------------------
# 变体一：留一账号（限于转发量最高的 top_n 个）
# ---------------------------------------------------------------------------

def run_leave_one_account_out(year=config.YEAR, top_n=DEFAULT_TOP_N, context=None,
                              out_path=None, diag_path=None):
    """逐个剔除转发量最高的 top_n 个账号

    完整的留一要跑上千个账号（每个都是一次六个量的拟合），跑不完；只能
    限于量最大的一批。**因此每一行都必须带着 considered_volume_share**：
    读者要知道"看过的这 top_n 个账号"占该领域转发量多少，才能判断这次
    留一检验覆盖了多大一块。一个不声明覆盖面的 top-N 就是一次自我夸大的
    稳健性检验。
    """
    context = context or build_context(year)
    frames = []
    for domain in DOMAINS:
        candidates = top_accounts(context, domain, top_n)
        coverage = volume_share(context, domain, candidates)
        print(
            "{} 留一账号: 逐个剔除转发量最高的 {} 个账号，它们合计占该领域"
            "转发量的 {:.1%}".format(domain, len(candidates), coverage)
        )
        for rank, account_id in enumerate(candidates, start=1):
            frames.append(_run_exclusion(
                context, {account_id},
                "loo_{}_rank{:02d}_{}".format(domain, rank, account_id),
                out_path, diag_path,
                considered={domain: (len(candidates), coverage)},
                note="top_n={};leave_one_out_domain={}".format(top_n, domain),
            ))
    return (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=list(harness.ROBUSTNESS_SCHEMA))
    )


# ---------------------------------------------------------------------------
# 变体二：留一类别
# ---------------------------------------------------------------------------

def run_leave_one_category_out(year=config.YEAR, context=None, out_path=None,
                               diag_path=None):
    """每次剔掉一整个类别：公共事务 8 个机构类别、明星 11 个认证关键词

    按**成员资格**剔除：跨类别的账号（同时属于 central_news 与 org_news、
    认证理由里同时有"演员"和"歌手"）在其所属的每个类别被剔时都会被剔掉。
    名单本来就不是划分，硬把它当划分只会得到一个"剔了类别 A、却还留着 A
    的账号"的假结果。诊断表用 n_accounts_excluded_multi_category 把这件事
    记下来。
    """
    context = context or build_context(year)
    frames = []
    for domain in DOMAINS:
        categories = context.categories.get(domain, {})
        if not categories:
            print("警告: {} 领域没有任何类别信息，留一类别跳过".format(domain))
            continue
        for category in sorted(categories):
            members = set(categories[category])
            frames.append(_run_exclusion(
                context, members,
                "{}_without_{}".format(domain, category),
                out_path, diag_path,
                note="left_out_category:{}:{}".format(domain, category),
            ))
    return (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=list(harness.ROBUSTNESS_SCHEMA))
    )


# ---------------------------------------------------------------------------
# 变体三：剔除头部账号——§13.5 的"是不是一个粉丝群撑起来的"
# ---------------------------------------------------------------------------

def run_exclude_top_accounts(year=config.YEAR, k=DEFAULT_EXCLUDE_K, context=None,
                             out_path=None, diag_path=None):
    """一次剔掉转发量最高的 k 个账号（默认 k = 1、5、10）

    这个变体与集中度表配套使用才有意义：集中度告诉读者头部账号占多大，
    这里告诉读者把它们拿掉之后六个量长什么样。两者缺一，"结论不是一个
    账号撑起来的"这句话就没有证据。
    """
    context = context or build_context(year)
    frames = []
    for domain in DOMAINS:
        for k_value in k:
            k_value = int(k_value)
            excluded = top_accounts(context, domain, k_value)
            share = volume_share(context, domain, excluded)
            print("{} 剔除前 {} 个账号，占该领域转发量 {:.1%}".format(
                domain, len(excluded), share))
            frames.append(_run_exclusion(
                context, set(excluded),
                "{}_exclude_top{}".format(domain, k_value),
                out_path, diag_path,
                note="exclude_top_k={};domain={}".format(k_value, domain),
            ))
    return (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=list(harness.ROBUSTNESS_SCHEMA))
    )


# ---------------------------------------------------------------------------
# 变体四：只保留中央级官方媒体
# ---------------------------------------------------------------------------

def run_core_national_media_only(year=config.YEAR,
                                 categories=CORE_NATIONAL_CATEGORIES,
                                 context=None, out_path=None, diag_path=None):
    """公共事务只保留中央级类别的账号，明星名单原样不动

    "中央级"取的是 configs/cn_news_sources_lists.py 里的 central_theory_website
    与 central_news 两类（见 CORE_NATIONAL_CATEGORIES 上方对这个判断的说明），
    类别名写进 variant_label 与 note，作者想换一种读法只要换参数。
    明星领域不受影响，但它的诊断行照样写：这样读者能直接看到
    excluded_volume_share=0，而不是在表里找不到明星那一行、只好猜。
    """
    context = context or build_context(year)
    public_categories = context.categories.get("public", {})
    unknown = [c for c in categories if c not in public_categories]
    if unknown:
        print("警告: 公共事务名单里没有类别 {}，它们不贡献任何账号".format(unknown))
    keep_public = set()
    for category in categories:
        keep_public |= set(public_categories.get(category, set()))

    public_ids = set(
        context.volume.loc[
            context.volume["source_domain"] == "public", "r_user_id"
        ].astype(str)
    )
    excluded = public_ids - keep_public
    print("只保留中央级官方媒体: 公共事务保留 {} / {} 个账号".format(
        len(public_ids) - len(excluded), len(public_ids)))
    return _run_exclusion(
        context, excluded,
        "core_national_media_only",
        out_path, diag_path,
        note="core_categories:{}".format("+".join(categories)),
    )


# ---------------------------------------------------------------------------
# 变体五：账号 bootstrap（按类别分层，对账号有放回）
# ---------------------------------------------------------------------------

def bootstrap_draw(strata, seed):
    """按分层对**账号**有放回重抽，返回 {账号: 抽到几次}

    每一层各抽自己那么多个账号（有放回），因此每一层的账号数期望不变，
    小类别不会被整层抽没——这正是"分层"这两个字的意思。

    **重抽的单位是账号，不是事件、也不是用户。** 抽事件会把"某个账号被
    整个抽掉"这件事变成"这个账号少了几条转发"，于是永远测不出"结论是不是
    一个账号撑起来的"；抽用户测的是另一回事（那是 §13.6）。
    gender_domain/tests/test_accounts_robustness.py 里的哨兵测试就钉在
    这一条上。
    """
    rng = np.random.default_rng(seed)
    draw = {}
    for key in sorted(strata):
        members = list(strata[key])
        if not members:
            continue
        picked = rng.integers(0, len(members), size=len(members))
        for index in picked:
            account_id = members[int(index)]
            draw[account_id] = draw.get(account_id, 0) + 1
    return draw


def run_account_bootstrap(year=config.YEAR, n_replicates=DEFAULT_N_REPLICATES,
                          seed=0, context=None, out_path=None, diag_path=None):
    """按类别分层、对账号有放回重抽的 bootstrap，逐 replicate 落盘

    每个 replicate 记自己的 seed；子 seed 由主 seed 经 SeedSequence 派生
    （复用 vocabulary.replicate_seeds，两个 family 用同一套派生规则）。
    """
    context = context or build_context(year)
    seeds = voc.replicate_seeds(seed, n_replicates)
    print("账号 bootstrap: {} 次，按类别分层对账号有放回重抽".format(n_replicates))

    collected = []
    for replicate in range(n_replicates):
        replicate_seed = seeds[replicate]
        draw = {
            domain: bootstrap_draw(context.strata[domain], seed=replicate_seed)
            for domain in DOMAINS
        }
        # 诊断行要的是"该领域每个账号被抽到几次"的完整账本：没抽到的账号
        # 必须显式记 0，否则 excluded_volume_share 会漏掉它们
        weights = {
            domain: {
                account_id: draw[domain].get(account_id, 0)
                for account_id in subset_weights(context, domain, None)
            }
            for domain in DOMAINS
        }
        collected.append(_run_variant(
            context, weights, "bootstrap_rep{}".format(replicate),
            out_path, diag_path, replicate=replicate, seed=replicate_seed,
            variant_family=BOOTSTRAP_FAMILY,
        ))
    return (
        pd.concat(collected, ignore_index=True) if collected
        else pd.DataFrame(columns=list(harness.ROBUSTNESS_SCHEMA))
    )


# ---------------------------------------------------------------------------
# 全流程
# ---------------------------------------------------------------------------

def build(year=config.YEAR, top_n=DEFAULT_TOP_N, k=DEFAULT_EXCLUDE_K,
          n_replicates=DEFAULT_N_REPLICATES, seed=0,
          concentration_k=DEFAULT_CONCENTRATION_K):
    """§13.5 全部账号变体，逐变体增量落盘，最后写集中度表与 manifest"""
    context = build_context(year)
    out_path = results_path()
    diag_path = diagnostics_path()
    conc_path = concentration_path()
    os.makedirs(robustness_dir(), exist_ok=True)

    concentration = concentration_by_gender(context, k=concentration_k)
    append_concentration(concentration.to_dict("records"), conc_path)

    run_leave_one_account_out(year, top_n=top_n, context=context,
                              out_path=out_path, diag_path=diag_path)
    run_leave_one_category_out(year, context=context, out_path=out_path,
                               diag_path=diag_path)
    run_exclude_top_accounts(year, k=k, context=context, out_path=out_path,
                             diag_path=diag_path)
    run_core_national_media_only(year, context=context, out_path=out_path,
                                 diag_path=diag_path)
    run_account_bootstrap(year, n_replicates=n_replicates, seed=seed,
                          context=context, out_path=out_path, diag_path=diag_path)

    results = pd.read_parquet(out_path, engine="pyarrow",
                              columns=list(harness.ROBUSTNESS_SCHEMA))
    diagnostics = pd.read_parquet(diag_path, engine="pyarrow",
                                  columns=list(DIAGNOSTIC_SCHEMA))
    top1 = concentration[(concentration["k"] == 1)]
    manifest = config.build_manifest(
        step="robustness_accounts_{}".format(year),
        inputs=[
            "user_domain_{}.parquet".format(year),
            "retweet_domain_events_{}".format(year),
        ],
        params={
            "year": year,
            "top_n": top_n,
            "exclude_k": list(k),
            "concentration_k": list(concentration_k),
            "n_replicates": n_replicates,
            "seed": seed,
            "seeds": voc.replicate_seeds(seed, n_replicates),
            "core_national_categories": list(CORE_NATIONAL_CATEGORIES),
            # 类别是从正式名单来的还是从表 B 还原的，必须写下来：两者的
            # 类别口径一致，但覆盖的账号集合不同
            "category_source": dict(context.category_source),
            "bootstrap_unit": "account_with_replacement_stratified_by_source_category",
            "bootstrap_population": "accounts_observed_in_table_b",
            "diagnostics_file": os.path.basename(diag_path),
            "concentration_file": os.path.basename(conc_path),
        },
        counts={
            "result_rows": int(len(results)),
            "diagnostic_rows": int(len(diagnostics)),
            "variant_labels": int(results["variant_label"].nunique()),
            # §13.5 的头号数字：头部账号占多大、留一检验覆盖了多大一块
            "concentration": {
                "celebrity_top1_share": {
                    row["gender"]: float(row["top_k_share"])
                    for _, row in top1[top1["domain"] == "celebrity"].iterrows()
                },
                "public_top1_share": {
                    row["gender"]: float(row["top_k_share"])
                    for _, row in top1[top1["domain"] == "public"].iterrows()
                },
            },
            "leave_one_out_coverage": {
                domain: float(volume_share(
                    context, domain, top_accounts(context, domain, top_n)
                ))
                for domain in DOMAINS
            },
            "max_excluded_volume_share": float(
                diagnostics["excluded_volume_share"].max()
            ),
            "max_users_lost_entry": int(diagnostics["n_users_lost_entry"].max()),
        },
        fingerprints={
            domain: config.fingerprint_terms(sorted(
                context.volume.loc[
                    context.volume["source_domain"] == domain, "r_user_id"
                ].astype(str)
            ))
            for domain in DOMAINS
        },
    )
    config.write_manifest(manifest, manifest_dir(year))
    return {
        "results_path": out_path,
        "diagnostics_path": diag_path,
        "concentration_path": conc_path,
    }


if __name__ == "__main__":
    fire.Fire({
        "build": build,
        "leave_one_account_out": run_leave_one_account_out,
        "leave_one_category_out": run_leave_one_category_out,
        "exclude_top_accounts": run_exclude_top_accounts,
        "core_national_media_only": run_core_national_media_only,
        "account_bootstrap": run_account_bootstrap,
        "concentration": concentration_by_gender,
    })
