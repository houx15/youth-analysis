"""
稀疏关联结构：帖子 × 词、用户 × 来源账号，各建一次，被所有 replicate 复用。

§13.3 要求对 200-500 份"随机保留 80% 词"的重采样词表分别重估全部内容
结果，§13.5 要求对上千个来源账号做留一/剔除/bootstrap。如果每个
replicate 都回去重扫全年正文（单进程约 32 分钟一遍）或重新聚合三千多万
行帖子，这两节根本跑不完。本模块把这件事变成"建一次矩阵、每个 replicate
一次列选择加一次 groupby"：

- 表 A 的 `{domain}_term_counts` 列（"词:次数" 拼接字符串，唯一权威解析
  实现是 build_post_table.decode_term_counts）展开成一个 **帖子 × 词** 的
  稀疏计数矩阵；
- 表 B 展开成一个 **用户 × 来源账号** 的稀疏计数矩阵。

--------------------------------------------------------------------------
重建口径：必须与主流水线逐用户完全相等
--------------------------------------------------------------------------
`topical_by_user` 复刻的是 build_user_tables.aggregate_posts 的口径，一处
都不许自己另立定义：

1. **分子**：表达帖里 `{domain}_hit` 为真的帖子数。在给定词表子集下，
   "命中"= 该帖保留词的出现次数之和 > 0；全词表时它与表 A 的
   `{domain}_hit` 完全等价（term_counts 非空 <=> 有匹配 <=> hit 为真）。
2. **分母**：`n_expressive_posts`，来自该用户的**全部帖子**，不是"进了
   矩阵的命中帖"。矩阵只保留至少命中一个词的帖子（没命中任何词的帖子在
   任何词表子集下也不可能命中），但 posts 帧仍然是全部帖子——分母必须从
   posts 帧算，用矩阵的行数当分母会把每个用户的份额都抬高。这是本模块
   最容易出错的一处，`topical_by_user` 的 posts 参数因此是必填的：调用方
   每次都得明说"哪一份帖子集合定义了分母"。
3. **表达帖口径**：直接用表 A 写好的 is_expressive 列（由
   text_rules.is_expressive_series 唯一定义：类型属于原创/带评论转发，
   且清洗后字符数 > 0），本模块不自己按 post_type 再推一遍。
4. **零分母是 NaN 不是 0**：除法直接复用 build_user_tables._safe_divide，
   不重写。注意"空词表下、有表达帖的用户"份额是 0.0 而不是 NaN——0 个词
   命中 10 条表达帖中的 0 条，这是一个有定义的 0；NaN 的唯一含义是分母
   为 0。

`source_by_user` 复刻 build_user_tables.aggregate_events：`{domain}_source_count`
是该用户在该领域的事件行数，`{domain}_source_entered` 是计数 > 0。表 B 里
没出现过的用户不在返回结果里，调用方按 combine_user_table 的做法左连接后
填 0/False 即可。

--------------------------------------------------------------------------
已知局限：嵌套遮蔽会让重新聚合**低估**（不会高估）
--------------------------------------------------------------------------
词表匹配是"最左最长、命中区间不重叠"（text_rules.VocabMatcher），表 A 存
下来的逐词计数是**消解重叠之后**的结果。因此按词表子集重新聚合，只有在
"被剔除的词都没有遮蔽住任何被保留的更短的词"时才与重扫原文精确相等：

    正文 "疫情防控" 在全词表下只记 {疫情防控: 1}，没有 {疫情: 1}。
    若某个 replicate 剔除了 "疫情防控"、保留了 "疫情"，重扫原文会命中
    "疫情"，而按存量重新聚合会判定这条帖子不命中 —— 偏差方向恒为低估。

暴露面（在真实词表上实测，见 nested_terms 的测试）：公共事务词表 816 词
中有 112 个（13.7%）是另一个词的子串；明星词表 535 词中只有 5 个（0.9%）。
也就是说明星领域基本精确，公共事务领域有一定但有界的暴露。`nested_terms`
把这批词显式暴露出来，供 §13.3 的 vocabulary.py 逐 replicate 报告"有多少
被保留的词被这次剔除的词遮蔽过"——那个数字就是该 replicate 重聚合误差的
上界，数字大的 replicate 应当被更谨慎地对待，必要时用少量精确重扫复核。

使用方法（本模块不提供 CLI，由 vocabulary.py / accounts.py 等调用）：
    from gender_domain.robustness import incidence as inc
    post_inc = inc.build_post_term_incidence(2020, "public")   # 每个作业建一次
    for subset in subsets:                                     # 每个 replicate 只做列选择
        user_df = inc.topical_by_user(post_inc, subset, post_inc.posts)
"""

import array
import glob
import os
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy import sparse

from gender_domain import build_post_table as bpt
from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import id_rules as ir

DOMAINS = but.DOMAINS

# posts 帧里 "这条帖子没有进矩阵"（一个词都没命中）的行号哨兵值
NO_MATRIX_ROW = -1

# 表 A 分片里本模块需要的列（不含逐域的 term_counts 列，那个按域另加）。
# parquet 必须显式指定 columns：表 A 还有 gender/province/density 等一大批
# 列，全年读进来只是白占内存。这里刻意多带 month 与 post_type——§13.7 的
# 时间限制与 §13.8 的帖子类型变体要复用同一个矩阵、只换 posts 帧，没有
# 这两列就得为了过滤再读一遍全年表 A。
POST_FRAME_COLUMNS = [
    "weibo_id",
    "user_id",
    "month",
    "post_type",
    "n_chars",
    "is_expressive",
]

# 表 B 分片里本模块需要的列
EVENT_FRAME_COLUMNS = ["user_id", "r_user_id", "source_domain", "source_category"]


# namedtuple 而不是普通 tuple：矩阵、索引、附表三者必须一起传递，任何一个
# 单独存在都没有意义（一个 帖子×词 矩阵不知道行是哪条帖子就是一堆数字）。
# 同时它仍然支持按位置解包，与任务简报里 `(matrix, term_index, posts)` 的
# 形状一致。
PostTermIncidence = namedtuple("PostTermIncidence", ["matrix", "term_index", "posts"])

# 账号侧比简报里的三元组多一个 users：用户 × 账号 矩阵的行轴就是用户，
# 没有这条轴，矩阵算出来的计数无法回贴到任何用户身上。简报的三元组里
# 没有位置放它，这里显式多带一项，而不是把用户轴藏进别的字段。
UserAccountIncidence = namedtuple(
    "UserAccountIncidence", ["matrix", "account_index", "accounts", "users"]
)


# ---------------------------------------------------------------------------
# 词表嵌套（遮蔽暴露面）
# ---------------------------------------------------------------------------

def normalize_vocabulary(terms):
    """按 VocabMatcher 的同一口径清洗词表：去首尾空白、去空词、去重后排序

    与 text_rules.VocabMatcher.__init__ 里的清洗保持一致（只是排序键不同，
    那边按长度降序是为了正则最长匹配，这里按字典序是为了确定性），避免
    "词表里有个带空格的重复词"在两处被算成不同的词数。
    """
    return sorted({t.strip() for t in terms if t and t.strip()})


def nested_terms(vocab):
    """返回词表中"是另一个词的子串"的那些词（被遮蔽风险词）

    这些词就是存量重聚合的全部风险来源：只有当某个 replicate 剔除了包含
    它们的长词、又保留了它们本身时，重新聚合才会与重扫原文不一致（且方向
    恒为低估，见模块文档）。§13.3 逐 replicate 报告的"被遮蔽的保留词数"
    就是从这里派生的。

    实现是朴素的两两包含判断（O(n^2) 次子串检查）：真实词表 816 / 535 词，
    量级完全够用，不值得为它引入后缀自动机之类的复杂度。
    """
    cleaned = normalize_vocabulary(vocab)
    return {a for a in cleaned for b in cleaned if a != b and a in b}


# ---------------------------------------------------------------------------
# 分片读取与内存打印
# ---------------------------------------------------------------------------

def _shard_files(name, year):
    """某一步的全年分片文件列表（与 build_user_tables._read_shards 同一命名约定）

    这里不直接用 _read_shards：它一次性 concat 全年，而本模块必须逐分片
    处理——term_counts 是全年三千多万行的字符串列，全部读进内存再展开
    会白白多占一份峰值内存，逐分片展开完就把字符串扔掉要省得多。
    """
    pattern = os.path.join(config.OUTPUT_DIR, f"{name}_{year}", "month=*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到分片: {pattern}")
    return files


def _matrix_mb(matrix):
    """稀疏矩阵自身占用的内存（MB）"""
    total = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return total / (1024 * 1024)


def _frame_mb(frame):
    """DataFrame 占用的内存（MB，含 object 列的实际字符串开销）"""
    return float(frame.memory_usage(deep=True).sum()) / (1024 * 1024)


def _print_matrix_stats(title, matrix, frame_name, frame):
    """打印形状、非零元素数与内存占用

    作业申请资源时唯一需要知道的三个数字。矩阵按 CSR 存，非零元素数直接
    决定内存；附表（posts / accounts）是另一份不容忽视的开销，一并打印，
    免得看着"矩阵才几百 MB"就按那个量级申请内存。
    """
    print(
        f"{title}: 形状 {matrix.shape[0]:,} × {matrix.shape[1]:,}，"
        f"非零元素 {matrix.nnz:,}，矩阵内存 {_matrix_mb(matrix):.1f} MB，"
        f"{frame_name} {len(frame):,} 行占 {_frame_mb(frame):.1f} MB"
    )


# ---------------------------------------------------------------------------
# 帖子 × 词
# ---------------------------------------------------------------------------

def _decode_shard(encoded, term_index, rows, cols, vals, row_offset):
    """把一个分片里非空的 term_counts 展开进 COO 三元组，返回新的行偏移

    term_index 按"首次出现"分配列号，最后在 build_post_term_incidence 里
    统一重排成字典序——边扫边排会让列号依赖分片顺序，重排一次是 O(词数)
    的置换，代价可以忽略。
    """
    row = row_offset
    for enc in encoded:
        for term, count in bpt.decode_term_counts(enc).items():
            col = term_index.get(term)
            if col is None:
                col = len(term_index)
                term_index[term] = col
            rows.append(row)
            cols.append(col)
            vals.append(count)
        row += 1
    return row


def build_post_term_incidence(year=config.YEAR, domain="public"):
    """表 A -> (帖子 × 词 稀疏计数矩阵, 词 -> 列号, 逐帖附表)

    Args:
        year: 数据年份
        domain: "public" 或 "celebrity"，决定读哪一列 term_counts

    Returns:
        PostTermIncidence(matrix, term_index, posts)
        - matrix: csr_matrix，**只含至少命中一个词的帖子**（一个词都没命中
          的帖子在任何词表子集下也不可能命中，留在矩阵里是纯浪费）；
          元素是该帖该词的出现次数（表 A 存量，最左最长消解重叠之后的计数）。
        - term_index: {词: 列号}，只含全年真实出现过的词；词表里存在但全年
          一次没出现的词不占列（子集里带着它们也不会报错）。
        - posts: **全部帖子**每帖一行，列为 POST_FRAME_COLUMNS 加 matrix_row
          （命中帖是它在矩阵里的行号，未命中帖是 NO_MATRIX_ROW = -1）。
          分母必须从这张表算，不是从矩阵行数算，见模块文档第 2 条。
    """
    if domain not in DOMAINS:
        raise ValueError(f"未知的 domain: {domain}，只支持 {DOMAINS}")

    files = _shard_files("post_domain_measures", year)
    columns = POST_FRAME_COLUMNS + [f"{domain}_term_counts"]
    print(f"读取 {len(files)} 个表 A 分片，构建 {domain} 领域的帖子×词矩阵")

    term_index = {}
    # array.array 而不是 Python list：非零元素在真实数据上是千万量级，
    # list 里每个 int 都是一个独立的 Python 对象（28 字节起步加 8 字节
    # 指针），array 是紧凑的 C int 缓冲，差着一个数量级。取出来时用
    # np.intc（= C int）而不是写死 np.int32，两者在主流平台上一致，但
    # 用 np.intc 是"按 array 的实际元素类型解释"，不依赖这个巧合。
    rows = array.array("i")
    cols = array.array("i")
    vals = array.array("i")
    post_frames = []
    n_matrix_rows = 0

    for path in files:
        frame = pd.read_parquet(path, columns=columns)
        frame["user_id"] = ir.normalize_id_series(frame["user_id"])
        # is_expressive 的口径只认表 A 那一列；旧分片缺列时由
        # build_user_tables 用同一个 text_rules 定义现场补（并打印警告），
        # 本模块绝不自己按 post_type 再推一遍
        frame = but._ensure_is_expressive(frame)

        encoded = frame[f"{domain}_term_counts"].fillna("").to_numpy()
        has_hit = np.array([bool(e) for e in encoded], dtype=bool)
        n_hit = int(has_hit.sum())

        matrix_row = np.full(len(frame), NO_MATRIX_ROW, dtype=np.int32)
        matrix_row[has_hit] = np.arange(
            n_matrix_rows, n_matrix_rows + n_hit, dtype=np.int32
        )
        n_matrix_rows = _decode_shard(
            encoded[has_hit], term_index, rows, cols, vals, n_matrix_rows
        )

        part = frame[POST_FRAME_COLUMNS].copy()
        part["matrix_row"] = matrix_row
        post_frames.append(part)
        # 展开完立刻丢掉这一份 term_counts 字符串列，不让它们跨分片堆积
        del frame, encoded

    posts = pd.concat(post_frames, ignore_index=True)
    # 逐列压到最省的表示：user_id / post_type 是三千多万行的重复字符串，
    # categorical 只存一份码表加一列整数码；month 只有 1-12，int8 就够；
    # n_chars / matrix_row 用 int32 足够（单帖字符数与命中帖行号都远小于
    # 2^31）。真实规模下这几处合起来能省掉数百 MB，而 posts 帧要在整个
    # array task 的生命周期里一直驻留。
    posts["user_id"] = posts["user_id"].astype("category")
    posts["n_chars"] = posts["n_chars"].astype(np.int32)
    posts["is_expressive"] = posts["is_expressive"].astype(bool)
    posts["post_type"] = posts["post_type"].astype("category")
    posts["month"] = posts["month"].astype(np.int8)

    n_terms = len(term_index)
    matrix = sparse.coo_matrix(
        (
            np.frombuffer(vals, dtype=np.intc),
            (np.frombuffer(rows, dtype=np.intc), np.frombuffer(cols, dtype=np.intc)),
        ),
        shape=(n_matrix_rows, n_terms),
        dtype=np.int32,
    ).tocsr()
    matrix, term_index = _sort_term_columns(matrix, term_index)

    _print_matrix_stats(f"{domain} 帖子×词矩阵", matrix, "posts 帧", posts)
    print(
        f"  命中帖 {n_matrix_rows:,} / 全部帖 {len(posts):,}"
        f"（{n_matrix_rows / max(len(posts), 1):.1%}），出现过的词 {n_terms:,} 个"
    )
    # posts 帧比矩阵本身大一个数量级（weibo_id 这一列的字符串占了大头），
    # 逐列打印是为了让"内存到底花在哪"一目了然：真要压内存，该动的是
    # posts 帧的列，不是矩阵。
    print("  posts 帧逐列内存（MB）: " + "，".join(
        f"{col} {size / (1024 * 1024):.1f}"
        for col, size in posts.memory_usage(deep=True).items() if col != "Index"
    ))
    return PostTermIncidence(matrix=matrix, term_index=term_index, posts=posts)


def _sort_term_columns(matrix, term_index):
    """把列号从"首次出现顺序"重排成词的字典序，返回 (新矩阵, 新索引)

    列号只在本模块内部使用，本可以不排；排一次是为了让同一份表 A 无论
    分片读取顺序如何，矩阵都逐字节相同——这类确定性在事后复核"这批数字是
    哪次跑出来的"时是必要的。
    """
    if not term_index:
        return matrix.tocsr(), term_index
    terms = list(term_index.keys())
    order = sorted(range(len(terms)), key=lambda i: terms[i])
    # old_to_new[旧列号] = 新列号
    old_to_new = np.empty(len(terms), dtype=np.int32)
    for new_col, old_col in enumerate(order):
        old_to_new[old_col] = new_col
    permuted = matrix.tocsc()[:, np.array(order, dtype=np.int64)].tocsr()
    return permuted, {terms[old]: int(old_to_new[old]) for old in range(len(terms))}


def term_subset_vector(incidence, term_subset):
    """把词表子集转成矩阵列上的 0/1 指示向量（子集里没出现过的词自动忽略）"""
    keep = np.zeros(incidence.matrix.shape[1], dtype=np.int32)
    for term in term_subset:
        col = incidence.term_index.get(term)
        if col is not None:
            keep[col] = 1
    return keep


def topical_by_user(incidence, term_subset, posts):
    """给定词表子集，重算每个用户的命中帖数与 topical_share

    与 build_user_tables.aggregate_posts 的口径逐条对齐，见模块文档。

    Args:
        incidence: build_post_term_incidence 的返回值
        term_subset: 保留的词（可迭代；不在矩阵里的词自动忽略）
        posts: **定义分母的那份帖子集合**，通常就是 incidence.posts。
            这个参数是必填的，因为它同时决定分子与分母的取值范围：
            §13.7 的时间限制、§13.8 的帖子类型变体正是通过传入过滤后的
            posts 帧来复用同一个矩阵的（过滤只能发生在这里，不能发生在
            矩阵上——矩阵里没有未命中帖，用它过滤会把分母算错）。

    Returns:
        DataFrame，列为 user_id / topical_posts / n_expressive_posts /
        topical_share，每个在 posts 里出现过的用户一行。
    """
    keep = term_subset_vector(incidence, term_subset)
    # 一次 CSR 矩阵-向量乘法就得到每条命中帖在该子集下的命中次数之和，
    # 是 O(非零元素数) 的操作；> 0 即为该子集下的 {domain}_hit
    if incidence.matrix.shape[1] == 0:
        row_hit_counts = np.zeros(incidence.matrix.shape[0], dtype=np.int32)
    else:
        row_hit_counts = incidence.matrix.dot(keep)

    matrix_row = posts["matrix_row"].to_numpy()
    in_matrix = matrix_row >= 0
    post_hit = np.zeros(len(posts), dtype=bool)
    post_hit[in_matrix] = row_hit_counts[matrix_row[in_matrix]] > 0

    expressive = posts["is_expressive"].to_numpy(dtype=bool)
    work = pd.DataFrame({
        # 用 .array 保留 categorical：三千多万行的用户 ID 若在这里被摊成
        # object 数组，光这一步就要多占几百 MB
        "user_id": pd.Series(posts["user_id"].array),
        # 分子：既是表达帖、又在该子集下命中的帖子
        "topical_posts": (post_hit & expressive).astype(np.int32),
        # 分母：全部表达帖（未命中帖也在其中，这正是必须传 posts 帧的原因）
        "n_expressive_posts": expressive.astype(np.int32),
    })
    # observed=True：只输出这份 posts 帧里真实出现过的用户。posts 被过滤过
    # 时（§13.7/§13.8），categorical 的码表里仍留着已被过滤掉的用户，
    # observed=False 会给他们凭空造出一行"0 帖 0 命中"，那是编出来的观测。
    grouped = work.groupby("user_id", observed=True, sort=True).sum()

    out = pd.DataFrame({
        "user_id": grouped.index.astype(str),
        "topical_posts": grouped["topical_posts"].to_numpy(dtype=np.int64),
        "n_expressive_posts": grouped["n_expressive_posts"].to_numpy(dtype=np.int64),
    })
    # 除法直接复用主流水线的实现：零分母必须是 NaN 而不是 0，这条口径
    # 只能有一个来源
    out["topical_share"] = but._safe_divide(
        out["topical_posts"], out["n_expressive_posts"]
    ).to_numpy(dtype=np.float64)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 用户 × 来源账号
# ---------------------------------------------------------------------------

def build_user_account_incidence(year=config.YEAR):
    """表 B -> (用户 × 来源账号 稀疏计数矩阵, (账号, 领域) -> 列号, 账号附表, 用户轴)

    列的键是 **(r_user_id, source_domain)** 而不是单独的账号 ID：同一个
    账号可能同时出现在两个领域的名单里（表 B 对这种账号的每条转发会各出
    一行），用单一账号 ID 当列会把两个领域的事件混成一列，`source_by_user`
    再也分不开。按账号剔除时两列一起消失，语义正是"这个账号整个不算"。

    Returns:
        UserAccountIncidence(matrix, account_index, accounts, users)
        - matrix: csr_matrix，元素是该用户转发该（账号, 领域）的事件行数
        - account_index: {(r_user_id, source_domain): 列号}
        - accounts: 每列一行，列为 r_user_id / source_domain / source_category
        - users: pd.Index，矩阵行轴对应的 user_id（只含表 B 里出现过的用户）
    """
    files = _shard_files("retweet_domain_events", year)
    print(f"读取 {len(files)} 个表 B 分片，构建用户×来源账号矩阵")
    frames = []
    for path in files:
        frame = pd.read_parquet(path, columns=EVENT_FRAME_COLUMNS)
        frame["user_id"] = ir.normalize_id_series(frame["user_id"])
        frame["r_user_id"] = ir.normalize_id_series(frame["r_user_id"])
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)

    # 先按 (用户, 账号, 领域) 汇总事件行数，再进矩阵：真实数据里同一个用户
    # 反复转同一个账号很常见，先聚合能把进 COO 的三元组数量压下来。
    # source_category 一并带上（同一 (账号, 领域) 的类别在表 B 里是唯一的，
    # 取 first 即可）。
    grouped = (
        events.groupby(["user_id", "r_user_id", "source_domain"], observed=True)
        .agg(n_events=("source_category", "size"),
             source_category=("source_category", "first"))
        .reset_index()
    )

    users = pd.Index(sorted(grouped["user_id"].unique()), name="user_id")
    user_pos = {uid: i for i, uid in enumerate(users)}

    accounts = (
        grouped[["r_user_id", "source_domain", "source_category"]]
        .drop_duplicates(subset=["r_user_id", "source_domain"])
        .sort_values(["source_domain", "r_user_id"])
        .reset_index(drop=True)
    )
    account_index = {
        (row.r_user_id, row.source_domain): i
        for i, row in enumerate(accounts.itertuples(index=False))
    }

    row_idx = grouped["user_id"].map(user_pos).to_numpy(dtype=np.int64)
    col_idx = np.array(
        [account_index[(a, d)]
         for a, d in zip(grouped["r_user_id"], grouped["source_domain"])],
        dtype=np.int64,
    )
    matrix = sparse.coo_matrix(
        (grouped["n_events"].to_numpy(dtype=np.int32), (row_idx, col_idx)),
        shape=(len(users), len(accounts)),
        dtype=np.int32,
    ).tocsr()

    _print_matrix_stats("用户×来源账号矩阵", matrix, "accounts 帧", accounts)
    print(
        f"  表 B 事件 {len(events):,} 行，涉及用户 {len(users):,} 个、"
        f"（账号, 领域）列 {len(accounts):,} 个"
    )
    return UserAccountIncidence(
        matrix=matrix, account_index=account_index, accounts=accounts, users=users
    )


def account_subset_vector(incidence, account_subset, domain):
    """某个领域下、保留账号所在列的 0/1 指示向量

    account_subset 为 None 表示"全部账号"（基线）。给的是账号 ID 集合，
    不是列键：剔除一个账号意味着它在两个领域的列一起消失（见
    build_user_account_incidence 的说明）。
    """
    keep = np.zeros(incidence.matrix.shape[1], dtype=np.int32)
    kept_ids = None if account_subset is None else set(account_subset)
    for (account_id, account_domain), col in incidence.account_index.items():
        if account_domain != domain:
            continue
        if kept_ids is None or account_id in kept_ids:
            keep[col] = 1
    return keep


def source_by_user(incidence, account_subset=None):
    """给定来源账号子集，重算每个用户各领域的来源转发数与进入指示

    与 build_user_tables.aggregate_events 的口径一致：`{domain}_source_count`
    是事件行数，`{domain}_source_entered` 是计数 > 0。

    Args:
        incidence: build_user_account_incidence 的返回值
        account_subset: 保留的账号 ID 集合；None 表示全部保留（基线）

    Returns:
        DataFrame，列为 user_id 以及各领域的 {domain}_source_count /
        {domain}_source_entered。只含表 B 里出现过的用户——表 B 里没有
        转发记录的用户由调用方按 combine_user_table 的做法左连接后填
        0/False，本模块不去猜表 C 的用户全集。
    """
    out = pd.DataFrame({"user_id": incidence.users.astype(str)})
    for domain in DOMAINS:
        keep = account_subset_vector(incidence, account_subset, domain)
        if incidence.matrix.shape[1] == 0:
            counts = np.zeros(incidence.matrix.shape[0], dtype=np.int64)
        else:
            counts = incidence.matrix.dot(keep).astype(np.int64)
        out[f"{domain}_source_count"] = counts
        out[f"{domain}_source_entered"] = counts > 0
    return out.reset_index(drop=True)
