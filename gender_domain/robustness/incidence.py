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
已知局限：被剔除词掩盖保留词时，重新聚合**低估**（不会高估）
--------------------------------------------------------------------------
词表匹配是"最左最长、命中区间不重叠"（text_rules.VocabMatcher），表 A 存
下来的逐词计数是**消解重叠之后**的结果。因此按词表子集重新聚合，只有在
"被剔除的词都没有掩盖住任何被保留的词"时才与重扫原文精确相等。掩盖有
**两种**方式，下面第一种（子串嵌套）只是较小的那一种，第二种（边界重叠）
在真实词表上多一个数量级——完整说明见 at_risk_pairs 上方的注释：

    正文 "疫情防控" 在全词表下只记 {疫情防控: 1}，没有 {疫情: 1}。
    若某个 replicate 剔除了 "疫情防控"、保留了 "疫情"，重扫原文会命中
    "疫情"，而按存量重新聚合会判定这条帖子不命中 —— 偏差方向恒为低估。

    第二种（边界重叠，子串口径完全看不见）：
    正文 "疫情防控措施"、词表 {疫情防, 防控措施} 在全词表下只记
    {疫情防: 1}。剔除 "疫情防"、保留 "防控措施"，重扫会命中、重聚合
    判不命中，而 "防控措施" 既不是 "疫情防" 的子串也不含它。

**发生率**（在真实词表上实测，见 nested_terms / at_risk_pairs 的测试）：

    子串嵌套词        公共事务 816 词中 112 个（13.7%）；明星 535 词中 5 个
    边界重叠有序词对  公共事务 1502 对；明星 93 对

**但发生率本身严重低估了这件事的严重性，真正要看的是效应量**（复核时
实测，公共事务真实词表、6000 帖语料、10 次保留 80% 的重采样，**只计子串
嵌套那一类**）：

    每个 replicate 被遮蔽的保留词        19-39 个（约占保留词的 4.2%；
                                          明星领域同口径只有 0.79 个）
    重聚合相对重扫丢失的命中帖比例       2.3% - 5.4%（10 次全部为丢失）
    逐用户 topical_share 平均偏差        -0.013 至 -0.032
    最差用户                             -0.27
    偏差方向                             10 次全部是低估，一次高估也没有

那份语料每帖只放一个词，帖子越稠密进入风险集的帖子越少；但这些数字与
§13.3 本身想要检验的效应是同一个数量级——也就是说，**一个未经测量的重
聚合偏差完全可能被误读成一条稳健性结论**。因此：

- `nested_terms(vocab)` / `shadowed_terms(vocab, subset)` /
  `shadowing_exposure(...)` 给出**只看子串**的那一部分；
- `at_risk_pairs(vocab)` / `at_risk_terms(vocab, subset)` /
  `reaggregation_exposure(...)` 给出**子串 + 边界重叠**的完整风险集，
  下游估计重聚合误差应当用这一组；
- 但即使是完整风险集，也只是"按我们理解的匹配失效方式推出来的"，仍然
  可能漏掉没想到的失效方式。唯一不依赖这层理解的检验，是在"重聚合判为
  不命中的表达帖"上随机抽样重扫（vocabulary.random_nonhit_probe）。

§13.3 必须逐 replicate 记录这些数字，对一个子样本做精确重扫校准，并且
用上面那个随机抽样兜住风险集之外的部分。

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
    恒为低估，见模块文档）。这是**词表层面的发生率**，不是效应量——效应量
    见模块文档里实测的那张表，以及 shadowing_exposure。

    实现是朴素的两两包含判断（O(n^2) 次子串检查）：真实词表 816 / 535 词，
    量级完全够用，不值得为它引入后缀自动机之类的复杂度。
    """
    cleaned = normalize_vocabulary(vocab)
    return {a for a in cleaned for b in cleaned if a != b and a in b}


def shadowed_terms(vocab, term_subset):
    """某一次重采样里，被"剔除的词"遮蔽住的保留词集合

    定义：保留词 t，存在一个被剔除的词 d（d != t）满足 t 是 d 的子串。
    这批词就是这次重聚合会低估的那些词——原文里出现 d 的地方，存量计数
    记的是 d，剔除 d 之后重聚合看不到里面的 t，而真正重扫会看到。

    纯词表运算，不需要矩阵，Task 3 可以在抽词的当场就算出来。要连"影响
    了多少条帖子"一起知道，用 shadowing_exposure。
    """
    retained = set(normalize_vocabulary(term_subset))
    dropped = set(normalize_vocabulary(vocab)) - retained
    return {t for t in retained for d in dropped if t != d and t in d}


def shadowing_dropped_terms(vocab, term_subset):
    """反向的一半：这次剔除的词里，哪些真的遮蔽住了某个保留词

    shadowed_terms 回答"哪些保留词会被低估"，这里回答"低估是由哪些被剔除
    的词造成的"。后者才是能落到帖子上的那一侧——一条帖子受不受影响，取决于
    它的存量计数里有没有这些词。
    """
    retained = set(normalize_vocabulary(term_subset))
    dropped = set(normalize_vocabulary(vocab)) - retained
    return {d for d in dropped for t in retained if t != d and t in d}


# ---------------------------------------------------------------------------
# 更完整的风险定义：子串嵌套只是重聚合失效的一小半
# ---------------------------------------------------------------------------
#
# shadowed_terms / nested_terms 只看"一个词是另一个词的子串"。但最左最长、
# 命中区间不重叠的匹配还有第二种失效方式，而且它常见得多——**边界重叠**：
#
#     正文 "疫情防控措施"，词表 {疫情防, 防控措施}
#     全词表下最左最长在位置 0 取 "疫情防"，剩下的 "控措施" 不成词，
#     存量计数只记 {疫情防: 1}。
#     某个 replicate 剔除 "疫情防"、保留 "防控措施"：重扫原文会命中
#     "防控措施"，按存量重新聚合却判定这条帖子不命中。
#
# 这里 "防控措施" 既不是 "疫情防" 的子串、也不含它，nested_terms 与
# shadowed_terms 一个都点不出来。实测两份真实词表：
#
#     子串嵌套词          公共事务 112 个 / 明星 5 个
#     边界重叠有序词对    公共事务 1502 对 / 明星 93 对
#
# 也就是说只看子串，会漏掉这一类错误里绝大部分的来源。因此**下游估计
# 重聚合误差时应当用 at_risk_terms / reaggregation_exposure，而不是只看
# 子串的 shadowed_terms / shadowing_exposure**；后两个保留下来，是因为
# "被子串嵌套遮蔽"本身仍是一个有意义的、更严格的子类，报告里把两者并列
# 才能看出"完整风险集比子串子集大多少"。
#
# 即便如此，这仍然是一个**基于机制推理**的风险集，不是"实际漏判了多少"
# 的测量：它假定我们把匹配器的失效方式想全了。真正不依赖这个假定的检验，
# 是在"重聚合判为不命中的表达帖"上随机抽样重扫（vocabulary.py 的
# random_nonhit_probe），那一条才能兜住这里还没想到的失效方式。


def boundary_overlap(first, second):
    """first 的某个非空真后缀等于 second 的前缀

    为真时，正文里 first 的一次命中可以恰好吃掉 second 开头的那几个字，
    使 second 在全词表下根本没有机会被记进存量计数。
    """
    max_k = min(len(first), len(second)) - 1
    for k in range(1, max_k + 1):
        if first[-k:] == second[:k]:
            return True
    return False


def at_risk_pairs(vocab):
    """{可能掩盖者: {被掩盖的词, ...}}，纯词表运算，一份词表只需算一次

    (a, b) 进入结果的条件：a != b，且下列任意一条成立
      1) b 是 a 的子串（嵌套遮蔽）；
      2) a 的后缀接上 b 的前缀（a 的命中吃掉 b 的开头）；
      3) b 的后缀接上 a 的前缀（a 的命中吃掉 b 的结尾）。

    真实词表 816 / 535 词做两两判断是 O(n^2 · 词长)，约一两秒，建一次
    传给所有 replicate 复用即可（at_risk_terms / at_risk_dropped_terms /
    reaggregation_exposure 都接受 pairs 参数）。
    """
    cleaned = normalize_vocabulary(vocab)
    pairs = {}
    for a in cleaned:
        related = {
            b for b in cleaned
            if b != a and (b in a or boundary_overlap(a, b) or boundary_overlap(b, a))
        }
        if related:
            pairs[a] = related
    return pairs


def at_risk_terms(vocab, term_subset, pairs=None):
    """这次重采样里，重聚合可能漏掉的保留词（子串嵌套 + 边界重叠）

    是 shadowed_terms 的**超集**：后者只认子串嵌套。
    """
    retained = set(normalize_vocabulary(term_subset))
    dropped = set(normalize_vocabulary(vocab)) - retained
    pairs = at_risk_pairs(vocab) if pairs is None else pairs
    return {t for d in dropped for t in pairs.get(d, ()) if t in retained}


def at_risk_dropped_terms(vocab, term_subset, pairs=None):
    """反向的一半：这次剔除的词里，哪些可能掩盖住某个保留词

    这一侧才是能落到帖子上的——一条帖子受不受影响，取决于它的存量计数里
    有没有这些词。
    """
    retained = set(normalize_vocabulary(term_subset))
    dropped = set(normalize_vocabulary(vocab)) - retained
    pairs = at_risk_pairs(vocab) if pairs is None else pairs
    return {d for d in dropped if retained & pairs.get(d, set())}


def _exposure(incidence, term_subset, vocab, risk_terms, risk_dropped):
    """暴露面的公共计算：风险词集合 -> 受影响帖子数

    shadowing_exposure（只看子串）与 reaggregation_exposure（子串 + 边界
    重叠）只差"风险词是怎么定义的"，落到帖子上的算法完全一样，因此只写
    一遍，免得两个口径在别处慢慢分叉。
    """
    retained = normalize_vocabulary(term_subset)
    shadow_vec = np.zeros(incidence.matrix.shape[1], dtype=np.int32)
    for term in risk_dropped:
        col = incidence.term_index.get(term)
        if col is not None:
            shadow_vec[col] = 1

    keep = term_subset_vector(incidence, term_subset, warn_unrecognized=False)
    if incidence.matrix.shape[1] == 0:
        row_shadow = np.zeros(incidence.matrix.shape[0], dtype=bool)
        row_hit = np.zeros(incidence.matrix.shape[0], dtype=bool)
    else:
        row_shadow = incidence.matrix.dot(shadow_vec) > 0
        row_hit = incidence.matrix.dot(keep) > 0

    posts = incidence.posts
    post_shadow = _rows_to_posts(posts, row_shadow)
    post_hit = _rows_to_posts(posts, row_hit)
    expressive = posts["is_expressive"].to_numpy(dtype=bool)
    possibly_lost = post_shadow & (~post_hit) & expressive

    return {
        "shadowed_terms": risk_terms,
        "n_shadowed_terms": len(risk_terms),
        "shadowed_share_of_retained": (
            len(risk_terms) / len(retained) if retained else 0.0
        ),
        "n_shadowing_dropped_terms": len(risk_dropped),
        "n_posts_with_shadowing_term": int(post_shadow.sum()),
        "n_expressive_posts_possibly_lost": int(possibly_lost.sum()),
    }


def reaggregation_exposure(incidence, term_subset, vocab, pairs=None):
    """完整的重聚合暴露面：子串嵌套 **加上** 边界重叠

    返回的键与 shadowing_exposure 完全相同（方便两者并列比较），但风险词
    集合是完整的那一个。**估计重聚合误差请用本函数**，shadowing_exposure
    只覆盖子串那一小半。

    仍然要记住：这是"按我们理解的失效方式推出来的风险集"，不是实测的
    漏判量。它给出的 n_expressive_posts_possibly_lost 是**在这个风险集
    之内**的上界，对风险集之外的失效方式一无所知。
    """
    risk_terms = at_risk_terms(vocab, term_subset, pairs=pairs)
    risk_dropped = at_risk_dropped_terms(vocab, term_subset, pairs=pairs)
    return _exposure(incidence, term_subset, vocab, risk_terms, risk_dropped)


def shadowing_exposure(incidence, term_subset, vocab):
    """**只看子串嵌套**的那一部分暴露面（完整口径见 reaggregation_exposure）

    Args:
        incidence: build_post_term_incidence 的返回值
        term_subset: 这次保留的词
        vocab: 完整词表（用来确定"被剔除的词"是哪些）

    Returns:
        dict：
        - shadowed_terms: 被剔除词以子串方式遮蔽住的保留词集合
        - n_shadowed_terms / shadowed_share_of_retained: 上一项的规模
        - n_posts_with_shadowing_term: 存量计数里含有"遮蔽性剔除词"的帖子数
        - n_expressive_posts_possibly_lost: 其中"在本子集下判定为不命中、
          且是表达帖"的帖子数。

    **重要：这不是重聚合误差的上界，只是其中一部分。** 边界重叠（见上面
    at_risk_pairs 的说明）同样会让重聚合漏判，而且在真实词表上比子串嵌套
    多一个数量级（公共事务 1502 对 vs 112 个）。要估计重聚合误差，用
    reaggregation_exposure；本函数保留下来，是为了让报告能把"子串子集"与
    "完整风险集"并列，看出后者大多少。
    """
    return _exposure(
        incidence, term_subset, vocab,
        shadowed_terms(vocab, term_subset),
        shadowing_dropped_terms(vocab, term_subset),
    )


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


def _compress_post_columns(frame):
    """把一份表 A 分片就地压到最省的表示，供 concat **之前**逐分片调用

    每一列的取值范围都是确定的，所以压缩不丢信息：month 只有 1-12，int8
    就够；n_chars 与 matrix_row 都远小于 2^31，int32 足够。三个字符串列
    统一走 string[pyarrow]——weibo_id 逐帖唯一，category 只会更差（码表和
    值一样长），而 user_id / post_type 虽然重复度高，却要等所有分片到齐
    才好定码表，逐分片转 category 会让 concat 去做类别并集。这一列不能删：
    §13.4 的抽样与精确重扫复核都要靠 weibo_id 回溯到原帖。

    就地改而不是返回新帧：调用方紧接着就要把它挂进 post_frames，多一份
    拷贝就多一份峰值，而峰值正是这个函数存在的理由。
    """
    if "weibo_id" in frame.columns:
        frame["weibo_id"] = frame["weibo_id"].astype("string[pyarrow]")
    frame["user_id"] = frame["user_id"].astype("string[pyarrow]")
    frame["post_type"] = frame["post_type"].astype("string[pyarrow]")
    frame["n_chars"] = frame["n_chars"].astype(np.int32)
    frame["is_expressive"] = frame["is_expressive"].astype(bool)
    frame["month"] = frame["month"].astype(np.int8)
    return frame


def build_post_term_incidence(year=config.YEAR, domain="public",
                              keep_weibo_id=True):
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
    # weibo_id 逐帖唯一，是 posts 帧里最大的一列——真实规模下约 3.4 GB，占
    # 整帧的一半以上。而**只有 §13.4 语境抽样**用得到它（要靠它回溯原帖去
    # 重扫正文）；词表族与测量族从头到尾没碰过这一列，却一直替它付内存。
    # 所以让调用方声明自己要不要，不要的就连读都不读。
    frame_columns = [c for c in POST_FRAME_COLUMNS
                     if keep_weibo_id or c != "weibo_id"]
    columns = frame_columns + [f"{domain}_term_counts"]
    print(f"读取 {len(files)} 个表 A 分片，构建 {domain} 领域的帖子×词矩阵"
          f"（weibo_id: {'保留' if keep_weibo_id else '不载入'}）")

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
        # is_expressive 的口径只认表 A 那一列，本模块绝不自己按 post_type
        # 再推一遍。注意：没有这一列的**旧版分片在这里根本走不到**——上面
        # 那句 read_parquet 显式点名了 is_expressive，缺列会直接抛
        # ArrowInvalid。这正是想要的行为（旧分片的表达帖口径不可信，宁可
        # 硬失败），_ensure_is_expressive 在这里起的作用只剩下"这一列有
        # 缺失值就拒绝继续"那一条，它的旧分片兜底分支在本模块是死路。
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

        part = frame[frame_columns].copy()
        part["matrix_row"] = matrix_row
        # **压缩必须在这里做，不能留到 concat 之后。**
        # 这几个 astype 原来全写在 pd.concat 后面，于是循环里堆的是原始
        # dtype：user_id / weibo_id / post_type 三列都是 object，每个值都是
        # 一个独立的 Python 字符串对象（60 字节起步）。2020 年表 A 是 1.47
        # 亿帖，光这三列就堆到约 26 GB，concat 时再翻一倍到约 59 GB——而作业
        # 的内存上限是 cpus-per-task × 4000M（8 核 32 GB、4 核 16 GB）。
        # 压缩写在 concat 之后，压出来确实只有约 6 GB，但那个峰值早就付掉了：
        # 用到本函数的三个族（词表 / 测量 / 语境抽样）全部在读分片的中途被
        # OOM 杀掉，谁都没活到 concat 那一行。
        _compress_post_columns(part)
        post_frames.append(part)
        # 展开完立刻丢掉这一份 term_counts 字符串列，不让它们跨分片堆积
        del frame, encoded

    posts = pd.concat(post_frames, ignore_index=True)
    # concat 之后立刻放掉分片列表：它与 posts 是两份独立的缓冲，不放掉等于
    # 让整个 array task 的余生一直背着一份多余的拷贝。
    del post_frames
    # 只剩这两列要等所有分片到齐才好定码表：user_id / post_type 重复度极高，
    # category 只存一份码表加一列整数码。循环里它们先转 string[pyarrow] 而不是
    # 直接转 category，是因为各分片的类别集合不同，concat 要做一次类别并集，
    # 那一步比推迟到这里更贵。
    posts["user_id"] = posts["user_id"].astype("category")
    posts["post_type"] = posts["post_type"].astype("category")

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


def unrecognized_terms(incidence, term_subset):
    """子集里在矩阵词轴上找不到的词（按 normalize_vocabulary 归一化之后）

    这些词分两类，而本模块**分不出来**，只能把它们报给调用方：
    1. 合法的"全年一次都没出现过的词"——真实词表里确实存在这样的词；
    2. 拼写错误、没归一化、或者根本传错了词表的词。

    第二类是本模块存在的意义所反对的那种失败：它不会报错，只会让每个
    replicate 都少算一批命中，方向完全一致——正是最难被发现、也最容易被
    误读成"稳健性结论"的偏差。所以 §13.3 应当把这个列表与"已知从未出现
    的词"集合对拍，多出来一个就是配置出了问题。
    """
    return sorted(
        term for term in normalize_vocabulary(term_subset)
        if term not in incidence.term_index
    )


def term_subset_vector(incidence, term_subset, warn_unrecognized=True):
    """把词表子集转成矩阵列上的 0/1 指示向量

    子集先过 normalize_vocabulary（与 VocabMatcher 建词表时同一套清洗：
    去首尾空白、去空词、去重）——否则一个带空格的 " 疫情" 会被静默当成
    "从未出现过的词"丢掉，而它本该命中。认不出来的词数会打印出来
    （warn_unrecognized=False 时不打印，供内部反复调用时避免刷屏），
    具体是哪些词用 unrecognized_terms 取。
    """
    keep = np.zeros(incidence.matrix.shape[1], dtype=np.int32)
    n_unknown = 0
    for term in normalize_vocabulary(term_subset):
        col = incidence.term_index.get(term)
        if col is None:
            n_unknown += 1
            continue
        keep[col] = 1
    if warn_unrecognized and n_unknown:
        print(
            f"提示: 词表子集里有 {n_unknown} 个词不在矩阵词轴上（全年未出现）。"
            "若这个数字超出'已知从未出现的词'的规模，说明传进来的词没有归一化"
            "或根本传错了词表——那会让每个 replicate 朝同一个方向少算命中，"
            "请用 unrecognized_terms 取出具体词核对。"
        )
    return keep


def term_length_vector(incidence):
    """矩阵词轴上每个词的字符长度，用于把逐词计数换算成命中字符数"""
    lengths = np.zeros(incidence.matrix.shape[1], dtype=np.int32)
    for term, col in incidence.term_index.items():
        lengths[col] = len(term)
    return lengths


def _rows_to_posts(posts, row_values):
    """把"逐矩阵行"的布尔量摊回"逐帖"（未进矩阵的帖子恒为 False）"""
    matrix_row = posts["matrix_row"].to_numpy()
    in_matrix = matrix_row >= 0
    out = np.zeros(len(posts), dtype=bool)
    out[in_matrix] = row_values[matrix_row[in_matrix]]
    return out


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

    post_hit = _rows_to_posts(posts, row_hit_counts > 0)
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


def char_measures_by_user(incidence, term_subset, posts):
    """给定词表子集，重算字符口径的三个指标（§13.1 分母稳健性要用）

    与 build_user_tables.aggregate_posts 的字符口径逐条对齐，全部只在表达帖
    上聚合：
        chars        = 表达帖字符数之和
        chars_hit    = 表达帖命中字符数之和
        n_hits       = 表达帖命中次数之和
        char_density = chars_hit / chars
        hits_per_1k  = n_hits / chars * 1000

    逐帖的两个量都是矩阵的一次向量乘法：命中次数是 `matrix.dot(keep)`；
    命中字符数是 `matrix.dot(keep * 词长)`——因为词表匹配保证命中区间
    不重叠，一条帖子的命中字符数就等于 Σ(该词出现次数 × 该词长度)。

    嵌套遮蔽的低估同样作用在这两个量上（而且比命中帖数更直接：剔除一个长词
    等于把它那几个字符整个抹掉），见模块文档。
    """
    keep = term_subset_vector(incidence, term_subset)
    lengths = term_length_vector(incidence)
    if incidence.matrix.shape[1] == 0:
        row_hits = np.zeros(incidence.matrix.shape[0], dtype=np.int64)
        row_chars_hit = np.zeros(incidence.matrix.shape[0], dtype=np.int64)
    else:
        row_hits = incidence.matrix.dot(keep).astype(np.int64)
        row_chars_hit = incidence.matrix.dot(keep * lengths).astype(np.int64)

    matrix_row = posts["matrix_row"].to_numpy()
    in_matrix = matrix_row >= 0
    post_hits = np.zeros(len(posts), dtype=np.int64)
    post_chars_hit = np.zeros(len(posts), dtype=np.int64)
    post_hits[in_matrix] = row_hits[matrix_row[in_matrix]]
    post_chars_hit[in_matrix] = row_chars_hit[matrix_row[in_matrix]]

    expressive = posts["is_expressive"].to_numpy(dtype=bool)
    work = pd.DataFrame({
        "user_id": pd.Series(posts["user_id"].array),
        "chars": np.where(expressive, posts["n_chars"].to_numpy(np.int64), 0),
        "chars_hit": np.where(expressive, post_chars_hit, 0),
        "n_hits": np.where(expressive, post_hits, 0),
    })
    grouped = work.groupby("user_id", observed=True, sort=True).sum()

    out = pd.DataFrame({
        "user_id": grouped.index.astype(str),
        "chars": grouped["chars"].to_numpy(dtype=np.int64),
        "chars_hit": grouped["chars_hit"].to_numpy(dtype=np.int64),
        "n_hits": grouped["n_hits"].to_numpy(dtype=np.int64),
    })
    out["char_density"] = but._safe_divide(out["chars_hit"], out["chars"]).to_numpy(
        dtype=np.float64
    )
    out["hits_per_1k"] = but._safe_divide(out["n_hits"], out["chars"]).to_numpy(
        dtype=np.float64
    ) * 1000
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
