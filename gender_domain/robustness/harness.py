"""
§13 稳健性套件的共享估计器：一个入口，六个预先设定好的量，每次调用都
产出可预期的行数。

设计上要解决的问题只有一个：稳健性检验的每一个 variant（换词表、换
样本、换来源账号、换分母……）都会产出一份改动过的用户级宽表，如果每个
variant 自己重新写一遍"怎么拟合 entry / topical / DiD"，两个 variant
之间任何一处细微分歧都无法与"这是不是一个真实的稳健性发现"区分开——
读者没有办法判断某个 variant 下 entry_public 的估计变了，到底是因为
样本真的变了，还是因为这个 variant 的作者在调用 fit_entry_models 时
不小心漏传了一个协变量。因此**只有这一个入口**知道"六个量分别是什么、
分别调用哪个函数、取哪一行"，所有 variant 一律通过它，不允许绕过。

设计要点（对应任务简报里的每一条要求）：

1. **六个量在模块常量 QUANTITIES 里写死**，与方案文档 §11.3 / §11.4
   的六个预先设定量一一对应，不接受调用方传入别的量、也不做成可配置项。
   六个量互相之间已经靠 (outcome, domain, term) 三元组唯一区分（与
   models_core / models_interaction 里"这个三元组在整张表里唯一"的
   约定完全一致），因此结果表不需要再加一个额外的 "quantity" 列——
   QUANTITY_META 把 QUANTITIES 的六个 key 映射到这三元组，供本模块
   自己筛选，也供下游（synthesis）按同一个映射反查。

2. **estimate_all 永远返回 len(QUANTITIES) * len(layers) 行**，哪怕
   某一层、某个量的拟合失败。models_core / models_interaction 内部已经
   把"模型不收敛 / 完全分离"这类失败翻译成 NaN 行（它们自己的模块文档
   第 7 条），但还有一类失败它们**没有**兜住：调用方传入的用户表缺了
   一整列（例如某个 variant 把某个来源账号类别的用户全部剔除后，恰好
   连带丢了某个结果变量列），这会在 prepare_model_frame / to_long_format
   完成之前就抛出 KeyError。本模块把六个量各自的拟合调用包在
   `_safe_call` 里，补上这最后一层失败留痕，保证"某个量拿不到列"和
   "某个量的模型不收敛"在结果表里长得一样——都是一行 NaN + note，
   不是整行消失。

3. **本模块自己不拟合任何模型。** entry / topical 两组量直接调用
   models_core.fit_entry_models / fit_share_models；DiD 两个量直接
   调用 models_interaction.to_long_format + fit_interaction，取其中
   term==TERM_DID 的那一行。所有统计口径完全继承自这两个模块，本模块
   只做"调用、筛选、贴上变体身份标签"三件事。

4. **seed 在每一行上都写**，哪怕这个 variant 是确定性的（此时写 None，
   不是省略这一列）——否则 seed 列在跨 variant 拼接后会在"有的整数、
   有的 NaN、有的整行没这一列"之间反复横跳，dtype 退化成 object，
   下游按 seed 分组、或者写 parquet 时都会被这种不一致坑到。

使用方法（本模块不提供 CLI，只被各 variant 模块的 build() 调用）：
    from gender_domain.robustness import harness
    rows = harness.estimate_all(user_df, "vocabulary", "keep0.8_rep3",
                                replicate=3, seed=123)
    harness.append_rows(rows, "analysis_data/robustness/vocabulary.parquet")
"""

import os
import tempfile

import numpy as np
import pandas as pd

from gender_domain import models_core as mc
from gender_domain import models_interaction as mi
from gender_domain import stats_utils as su

# 六个预先设定量，来自方案文档 §11.3——顺序与文档表格一致，不是可配置项。
QUANTITIES = (
    "entry_public",
    "entry_celebrity",
    "topical_public",
    "topical_celebrity",
    "did_entry",
    "did_topical",
)

# QUANTITIES 的每个 key -> 在共享结果 schema 里唯一定位这一行的三元组
# (outcome, domain, term) + 报告尺度 scale。这就是"六个量分别是什么"的
# 唯一权威定义——新增/修改任何一个量都只应该改这里，不应该散落在
# estimate_all 内部的分支逻辑里。
QUANTITY_META = {
    "entry_public": {
        "outcome": "source_entered", "domain": "public",
        "term": mc.TERM_AME, "scale": "probability",
    },
    "entry_celebrity": {
        "outcome": "source_entered", "domain": "celebrity",
        "term": mc.TERM_AME, "scale": "probability",
    },
    "topical_public": {
        "outcome": "topical_share", "domain": "public",
        "term": mc.TERM_AME, "scale": "proportion",
    },
    "topical_celebrity": {
        "outcome": "topical_share", "domain": "celebrity",
        "term": mc.TERM_AME, "scale": "proportion",
    },
    "did_entry": {
        "outcome": "source_entered", "domain": mi.DOMAIN_BOTH,
        "term": mi.TERM_DID, "scale": "probability",
    },
    "did_topical": {
        "outcome": "topical_share", "domain": mi.DOMAIN_BOTH,
        "term": mi.TERM_DID, "scale": "proportion",
    },
}

# 六个量按"来源"分三组：entry/topical 各自对应一个 domain，DiD 各自
# 对应一个 models_interaction.OUTCOME_KINDS 的 key。分组只是为了避免
# 同一个 domain / outcome_kind 在 layers 循环里被重复拟合——每组只调用
# 一次底层函数，取回全部层的结果表（entry/topical）或长表（DiD），
# 再按 layers 参数逐层抽取。
_ENTRY_QUANTITIES = {"entry_public": "public", "entry_celebrity": "celebrity"}
_TOPICAL_QUANTITIES = {"topical_public": "public", "topical_celebrity": "celebrity"}
_DID_QUANTITIES = {"did_entry": "source_entry", "did_topical": "topical_share"}

# 扩展字段：稳健性层在共享结果 schema 之外追加的"变体身份"列。
EXTRA_SCHEMA = ("variant_family", "variant_label", "replicate", "seed")
ROBUSTNESS_SCHEMA = tuple(su.RESULT_SCHEMA) + EXTRA_SCHEMA


# ---------------------------------------------------------------------------
# 失败留痕：包一层 try/except，把"调用本身就炸了"翻译成可追溯的 note
# ---------------------------------------------------------------------------

def _safe_call(fn, label):
    """执行一次调用，成功返回 (结果, None)，失败返回 (None, note)

    这里兜住的是 models_core / models_interaction 自己没有兜住的那一类
    失败：结果变量列整个不存在（prepare_model_frame / to_long_format
    读取列名之前就会抛 KeyError），而不是"模型不收敛"那一类——后者早已
    经在 mc._safe_fit / mi._safe_fit 内部被翻译成 NaN 行，走不到这里。
    """
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 —— 失败必须留痕，不能让整批量消失
        note = "{}: {}".format(type(exc).__name__, exc)
        print(f"警告: harness 在 {label} 处捕获异常（{note}），"
              "受影响的量本层写入一行 NaN 结果并注明原因")
        return None, note[:200]


def _nan_quantity_row(quantity, model, n_input, note):
    """某个量在某一层完全拿不到拟合结果时的占位行

    n_obs/n_dropped 没有真实的样本流失记账可用（连 prepare_model_frame
    都没跑到），因此如实记成 "0 个观测进了拟合、全部 n_input 个用户都
    没有" ——这本身就是一条真实信息（这个量在这个 variant 下完全无法
    估计），不是猜出来的数字。
    """
    meta = QUANTITY_META[quantity]
    return su.tidy_result(
        outcome=meta["outcome"], domain=meta["domain"], model=model, term=meta["term"],
        estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, scale=meta["scale"],
        n_obs=0, n_dropped=n_input, drop_reason=None,
        note=note or "harness_call_failed",
    )


def _extract_row(frame, model, quantity, n_input, call_note):
    """从底层拟合函数返回的完整结果表里，取出某个量在某一层的那一行

    调用本身成功（frame 不是 None）时，(model, term, domain) 三元组按
    models_core / models_interaction 自己的约定必然唯一存在；真的取不到
    （不该发生，出现了说明底层模块改了约定）时同样退化成 NaN 行，而不是
    抛异常让整个 estimate_all 炸掉——一个量的记账错误不该连累其它五个。
    """
    if frame is None:
        return _nan_quantity_row(quantity, model, n_input, call_note)
    meta = QUANTITY_META[quantity]
    mask = (
        (frame["model"] == model) & (frame["term"] == meta["term"]) &
        (frame["domain"] == meta["domain"])
    )
    sub = frame[mask]
    if len(sub) != 1:
        note = "quantity_row_missing_for_layer:{}".format(model)
        print(f"警告: {quantity}/{model} 在底层结果表里没有找到恰好一行（找到 {len(sub)} 行），"
              "已写入一行 NaN 结果并注明原因")
        return _nan_quantity_row(quantity, model, n_input, note)
    return sub.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# 扩展 schema：延续 tidy_result "未知字段直接报错" 的纪律，而不是绕过它
# ---------------------------------------------------------------------------

def attach_variant_identity(row, variant_family, variant_label, replicate, seed,
                            **extra):
    """给共享 schema 的一行结果补上四个变体身份字段

    先用 su.tidy_result 校验并规整共享 schema 的部分（沿用它"收到未知
    字段直接报错"的行为），扩展字段单独走 EXTRA_SCHEMA 的白名单校验——
    两段校验都不允许静默塞进一个陌生字段，这是"延续 tidy_result 的拒绝
    行为，而不是绕过它"这条要求的落地方式：没有把扩展字段偷偷塞进
    tidy_result 的 kwargs 里蒙混过关（那样会撞上它自己的未知字段校验），
    而是在它之外再放一道同样不妥协的校验。

    Args:
        row: dict 或 pandas.Series，键需要恰好落在 su.RESULT_SCHEMA 内
        extra: 预留给调用方误传多余字段时触发校验，正常调用不应该用到
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()
    tidy = su.tidy_result(**{col: row[col] for col in su.RESULT_SCHEMA})
    identity = {
        "variant_family": variant_family,
        "variant_label": variant_label,
        "replicate": replicate,
        "seed": seed,
    }
    identity.update(extra)
    unknown = set(identity) - set(EXTRA_SCHEMA)
    if unknown:
        raise ValueError(
            f"attach_variant_identity 收到未知的扩展字段 {sorted(unknown)}，"
            f"超出扩展 schema {EXTRA_SCHEMA} 的范围"
        )
    tidy.update(identity)
    return tidy


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def estimate_all(user_df, variant_family, variant_label, replicate=0, seed=None,
                 layers=("M0", "M1")):
    """六个预先设定量 × 给定层数，返回共享 schema 加四个变体身份列的结果表

    Args:
        user_df: 某个 variant 产出的用户级宽表（表 C 的列结构）
        variant_family: 这一行属于哪一类稳健性检验（例如 "vocabulary"）
        variant_label: 这一类下具体是哪一个变体（例如 "keep0.8_rep3"）
        replicate: 第几次重复（确定性 variant 恒为 0）
        seed: 这个 variant 用的随机种子；确定性 variant 必须显式传 None，
            不能省略——见模块文档第 4 条。**这个 seed 只写进结果表的
            seed 列，记录的是"这次 variant 抽样用的种子"（例如词表重
            抽样、账号 bootstrap），不会被传给
            models_interaction.fit_interaction 内部的 DiD cluster
            bootstrap——那条随机数流由该模块自己的 _SEED 常量控制，
            与 variant 的抽样过程是两个独立的随机性来源，因此这里刻意
            不把两者混在一起、也不会覆盖对方。如果某个 variant 需要让
            DiD 的 bootstrap 区间也随这个 seed 变化，必须在调用方显式
            处理，本函数当前不做这件事。
        layers: 要估计的模型层，默认 ("M0", "M1")；只有 §13.9 画像相关
            的 variant 才应该传入 M2（方案文档 §11.4：M2 会收窄样本，
            混进其它比较会混淆结论）

    Returns:
        DataFrame，恰好 len(QUANTITIES) * len(layers) 行，列集合等于
        ROBUSTNESS_SCHEMA。任何一个量在任何一层拿不到估计，对应的行
        仍然存在，只是 estimate 等数值列为 NaN、note 带着失败原因。
    """
    n_input = len(user_df)

    entry_frames = {
        domain: _safe_call(lambda d=domain: mc.fit_entry_models(user_df, d), f"entry/{domain}")
        for domain in set(_ENTRY_QUANTITIES.values())
    }
    topical_frames = {
        domain: _safe_call(
            lambda d=domain: mc.fit_share_models(user_df, d, "topical_share"), f"topical/{domain}"
        )
        for domain in set(_TOPICAL_QUANTITIES.values())
    }
    did_long = {
        outcome_kind: _safe_call(
            lambda ok=outcome_kind: mi.to_long_format(user_df, ok), f"did_long/{outcome_kind}"
        )
        for outcome_kind in set(_DID_QUANTITIES.values())
    }
    # DiD 按 (outcome_kind, layer) 缓存：fit_interaction 一次只拟合一层，
    # 与 entry/topical 那种"一次调用拿到全部层"的形状不同，所以只能按需
    # 拟合，但同一个 (outcome_kind, layer) 在六个量里只会被用到一次，
    # 不存在重复拟合的问题。
    #
    # 注意：这里没有把 estimate_all 收到的 seed 传给 fit_interaction——
    # 该函数内部的 DiD cluster bootstrap 用的是 models_interaction 自己
    # 的 _SEED 常量，与本函数的 seed 参数是两条独立的随机数流（见
    # estimate_all 的 Args 说明）。
    did_fits = {}

    def _did_fit(outcome_kind, model):
        key = (outcome_kind, model)
        if key not in did_fits:
            long_df, note = did_long[outcome_kind]
            if long_df is None:
                did_fits[key] = (None, note)
            else:
                did_fits[key] = _safe_call(
                    lambda: mi.fit_interaction(long_df, model, outcome_kind=outcome_kind),
                    f"did_fit/{outcome_kind}/{model}",
                )
        return did_fits[key]

    rows = []
    for model in layers:
        for quantity in QUANTITIES:
            if quantity in _ENTRY_QUANTITIES:
                frame, note = entry_frames[_ENTRY_QUANTITIES[quantity]]
            elif quantity in _TOPICAL_QUANTITIES:
                frame, note = topical_frames[_TOPICAL_QUANTITIES[quantity]]
            else:
                frame, note = _did_fit(_DID_QUANTITIES[quantity], model)
            row = _extract_row(frame, model, quantity, n_input, note)
            rows.append(attach_variant_identity(
                row, variant_family, variant_label, replicate, seed
            ))

    n_expected = len(QUANTITIES) * len(layers)
    if len(rows) != n_expected:
        raise ValueError(
            f"estimate_all 产出 {len(rows)} 行，应恰好是 len(QUANTITIES)={len(QUANTITIES)}"
            f" × len(layers)={len(layers)} = {n_expected} 行——这条不变量是下游 synthesis"
            "判断某个 variant 是否真的跑过的唯一依据，绝不能被打破。"
        )
    return pd.DataFrame(rows, columns=list(ROBUSTNESS_SCHEMA))


def baseline(user_df):
    """未经任何改动的参照组：variant_family="baseline"，是每个 variant 的比较基准"""
    return estimate_all(user_df, variant_family="baseline", variant_label="baseline",
                        replicate=0, seed=None)


def append_rows(df, path):
    """把 df 追加写入 path，增量落盘，避免作业被杀掉时丢失已完成的 replicate

    做法是"读旧的（如果存在）、拼新的、写到同目录下的临时文件、再
    os.replace 原子改名覆盖目标"，而不是维护一个真正的追加写 parquet
    （pyarrow 的追加写要跨进程管理 row group/schema，这里的调用量级——
    一个 variant family 几百个 replicate、每个几十行——远用不上那套
    复杂度）。

    **`to_parquet` 直接写目标路径本身不是原子操作**：它打开目标文件时
    就会先截断，再流式写入内容，如果作业在这中间被墙钟 SIGKILL，目标
    文件会被截断成一个连 parquet 魔数都读不出来的半成品——不是丢了这一
    批新行，而是连之前所有已经写盘的 replicate 一起报废，恰好与"增量
    落盘是为了保住已完成的 replicate"这个目的相反。因此这里改为写到
    同一目录下的临时文件（`tempfile.mkstemp` 保证文件名不冲突），成功
    写完后再用 `os.replace` 把临时文件改名覆盖到目标路径——`os.replace`
    在同一文件系统内是原子的，中途被杀掉时磁盘上永远只会是"改名前的旧
    文件完整无损"或"改名后的新文件完整无损"两种状态之一，不会有第三种
    半成品状态。临时文件写失败或改名前抛异常时，显式清理掉这个临时文件，
    不留垃圾。

    只要每完成一个 replicate 就调用一次本函数，被墙钟杀掉的作业在磁盘上
    留下的就是"已完成的那些 replicate"——这与 models_interaction.
    _write_partial 是同一个增量落盘思路，只是这里的调用方是"每个
    replicate 一次"而不是"每层模型一次"，而且额外补上了后者不需要面对
    的"目标文件已经存在、必须读旧拼新"这一步的原子性。
    """
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path, engine="pyarrow", columns=list(ROBUSTNESS_SCHEMA))
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.reset_index(drop=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".append_rows_tmp_", suffix=".parquet", dir=out_dir
    )
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, engine="pyarrow", index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"已增量写入: {path}（新增 {len(df)} 行，累计 {len(combined)} 行）")
    return path
