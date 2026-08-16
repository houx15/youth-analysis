"""
gender_domain.models_interaction 的单元测试（§6.6 Gender × Domain 交互）。

测试设计原则（本模块是论文的核心论断，测试比其它模块更严，理由写在这里）：

1. **真值写死在生成过程里，断言的是"能不能把已知量估回来"。**
   模拟数据里公共事务域的性别差被设为 +0.10、娱乐域被设为 -0.10，
   于是差中差（DiD）的真值恒为 0.20。测试断言估计值接近 0.20 且区间
   不含 0；另一份"两域性别差相同"的数据则断言区间必须含 0。
   只断言"跑通了"对这个数字是不够的——论文的整个论证就是这一个数。

2. **配对必须被守住，且要有一个"配对一旦被打散就会挂"的判别性测试。**
   每个用户贡献两行，bootstrap 只能整簇（用户）连同两行一起重抽样。
   test_cluster_bootstrap_keeps_a_users_two_rows_together 构造了一个
   统计量：只要配对完整，它在任何一次整簇重抽样上都精确等于 0.20
   （用户基线在四个格子间恰好抵消），方差为 0、区间退化成一个点；
   一旦按行重抽样（配对被打散），用户基线不再抵消，区间立刻有宽度。
   这与 test_stats_utils 里那条 cluster 配对 canary 是同一个套路。

3. **单域缺失的处理规则必须被测试固定下来。**
   一个用户在一个域上结果变量是 NaN、另一个域有效时，"整个用户剔除"
   与"保留有效那一行"给出的是不同的数字。本模块选择整个用户剔除
   （理由见 models_interaction 模块文档），测试直接锁死这个选择与它的
   样本量记账，避免以后被一次无心的重构悄悄改掉。

4. **中心数字不允许没有区间。** GEE 拿不到 cov_params() 时必须退回到
   "每次重抽样重新拟合"的 cluster bootstrap，而不是接受一个 NaN 区间。
   test_bootstrap_path_used_when_cov_params_unavailable 用一个刻意让
   cov_params() 抛 NotImplementedError 的代理对象把这条路径逼出来。
"""

import json

import numpy as np
import pandas as pd
import pytest

from gender_domain import models_interaction as mi
from gender_domain import stats_utils as su


# ---------------------------------------------------------------------------
# 模拟数据构造
# ---------------------------------------------------------------------------

def _simulate_wide(n_male=2500, n_female=2500, gap_public=0.10, gap_celebrity=-0.10,
                   base=0.45, seed=0, user_sd=0.05, profile_complete_frac=1.0):
    """构造一份表 C 形状的模拟数据，两域的性别差分别写死为给定值

    每个用户有一个共享的潜在偏移 u_i（两个域同一个值），它制造了
    "同一个人两行相关"这件事——GEE 的 exchangeable 工作相关结构与
    cluster bootstrap 存在的全部理由就是它。u_i 的均值为 0，因此
    四个 gender × domain 格子的总体均值不受它影响，DiD 的真值仍然
    恰好是 gap_public - gap_celebrity。
    """
    rng = np.random.default_rng(seed)
    n = n_male + n_female
    gender = np.array(["m"] * n_male + ["f"] * n_female)
    male = (gender == "m").astype(float)
    u = rng.normal(0.0, user_sd, n)

    p_public = np.clip(base + male * gap_public + u, 0.01, 0.99)
    p_celebrity = np.clip(base + male * gap_celebrity + u, 0.01, 0.99)

    df = pd.DataFrame({
        "user_id": [str(i) for i in range(n)],
        "gender": gender,
        "n_posts": rng.integers(1, 200, n),
        "n_retweets": rng.integers(1, 100, n),
        "n_expressive_posts": rng.integers(1, 80, n),
        "n_active_days": rng.integers(1, 300, n),
        "n_active_months": rng.integers(1, 13, n),
        "n_active_months_panel": rng.integers(1, 13, n),
        "region": rng.choice(["north", "south", "west"], n),
        "verified_flag": rng.integers(0, 2, n).astype(float),
        "log_fans": rng.normal(5, 1, n),
        "log_friends": rng.normal(4, 1, n),
        "profile_complete": rng.random(n) < profile_complete_frac,
    })
    for domain, p in (("public", p_public), ("celebrity", p_celebrity)):
        df[f"{domain}_source_entered"] = rng.random(n) < p
        # 占比型结果变量：均值等于同一个 p，因此 DiD 的真值与进入模型一致
        share = np.clip(p + rng.normal(0.0, 0.08, n), 0.001, 0.999)
        df[f"{domain}_topical_share"] = share
        df[f"{domain}_source_share"] = share
        df[f"{domain}_source_count"] = rng.integers(0, 10, n)
        df[f"{domain}_source_months"] = rng.integers(0, 3, n)
        df[f"{domain}_source_month_share"] = rng.random(n) * 0.5
    return df


def _observed_did(wide, outcome_kind):
    """直接从原始数据算出的四格均值差中差，即模型应当逼近的样本内真值"""
    cols = {
        "source_entry": ("public_source_entered", "celebrity_source_entered"),
        "topical_share": ("public_topical_share", "celebrity_topical_share"),
    }[outcome_kind]
    male = wide["gender"] == "m"
    female = wide["gender"] == "f"
    pub, cel = cols
    gap_pub = wide.loc[male, pub].astype(float).mean() - wide.loc[female, pub].astype(float).mean()
    gap_cel = wide.loc[male, cel].astype(float).mean() - wide.loc[female, cel].astype(float).mean()
    return float(gap_pub - gap_cel)


def _row(frame, model, term):
    sub = frame[(frame["model"] == model) & (frame["term"] == term)]
    assert len(sub) == 1, f"期望恰好一行 {model}/{term}，实际 {len(sub)} 行"
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# to_long_format
# ---------------------------------------------------------------------------

def test_long_format_has_exactly_two_rows_per_user():
    wide = _simulate_wide(n_male=30, n_female=30, seed=1)
    long_df = mi.to_long_format(wide, "source_entry")
    assert len(long_df) == 2 * len(wide)
    counts = long_df.groupby(mi.USER_COLUMN).size()
    assert (counts == 2).all()
    assert sorted(long_df[mi.DOMAIN_COLUMN].unique()) == ["celebrity", "public"]


def test_long_format_round_trips_a_users_values():
    wide = _simulate_wide(n_male=20, n_female=20, seed=2)
    long_df = mi.to_long_format(wide, "topical_share")
    target = wide.iloc[7]
    rows = long_df[long_df[mi.USER_COLUMN] == target["user_id"]].set_index(mi.DOMAIN_COLUMN)
    assert rows.loc["public", mi.OUTCOME_COLUMN] == pytest.approx(target["public_topical_share"])
    assert rows.loc["celebrity", mi.OUTCOME_COLUMN] == pytest.approx(
        target["celebrity_topical_share"]
    )
    # 用户级协变量在两行上完全一致（这正是"控制变量不随域变化"的含义）
    assert rows.loc["public", "log_fans"] == pytest.approx(target["log_fans"])
    assert rows.loc["celebrity", "log_fans"] == pytest.approx(target["log_fans"])
    assert rows.loc["public", mi.DOMAIN_INDICATOR] == 1.0
    assert rows.loc["celebrity", mi.DOMAIN_INDICATOR] == 0.0


def test_long_format_keeps_the_row_even_when_the_outcome_is_nan():
    # 长表本身不做任何剔除：n_obs / n_dropped 的分母就是 len(long_df)//2，
    # 一旦 to_long_format 悄悄丢行，样本流失记账的基准就没了
    wide = _simulate_wide(n_male=10, n_female=10, seed=3)
    wide.loc[0, "public_topical_share"] = np.nan
    long_df = mi.to_long_format(wide, "topical_share")
    assert len(long_df) == 2 * len(wide)
    assert long_df[mi.OUTCOME_COLUMN].isna().sum() == 1


def test_long_format_pairs_are_contiguous():
    # 配对连续是 cluster bootstrap 重标簇号的前提，属于本模块的硬不变量
    wide = _simulate_wide(n_male=15, n_female=15, seed=4)
    long_df = mi.to_long_format(wide, "source_entry")
    users = long_df[mi.USER_COLUMN].to_numpy()
    assert (users[0::2] == users[1::2]).all()


def test_long_format_marks_unusable_gender_as_nan_not_female():
    wide = _simulate_wide(n_male=5, n_female=5, seed=5)
    wide.loc[0, "gender"] = "unknown"
    long_df = mi.to_long_format(wide, "source_entry")
    rows = long_df[long_df[mi.USER_COLUMN] == wide.loc[0, "user_id"]]
    assert rows[mi.FOCAL_COLUMN].isna().all()


def test_long_format_rejects_unknown_outcome_kind():
    wide = _simulate_wide(n_male=5, n_female=5, seed=6)
    with pytest.raises(ValueError):
        mi.to_long_format(wide, "not_an_outcome")


# ---------------------------------------------------------------------------
# 差中差的点估计与区间
# ---------------------------------------------------------------------------

def test_did_recovers_a_known_interaction_on_entry():
    wide = _simulate_wide(n_male=2500, n_female=2500, gap_public=0.10,
                          gap_celebrity=-0.10, seed=11)
    long_df = mi.to_long_format(wide, "source_entry")
    frame = mi.fit_interaction(long_df, "M0")
    row = _row(frame, "M0", mi.TERM_DID)
    assert row["scale"] == "probability"
    assert row["estimate"] == pytest.approx(0.20, abs=0.04)
    assert row["estimate"] == pytest.approx(_observed_did(wide, "source_entry"), abs=0.01)
    assert row["ci_low"] > 0.0
    assert np.isfinite(row["se"])


def test_did_recovers_a_known_interaction_on_share():
    wide = _simulate_wide(n_male=1500, n_female=1500, gap_public=0.10,
                          gap_celebrity=-0.10, seed=12)
    long_df = mi.to_long_format(wide, "topical_share")
    frame = mi.fit_interaction(long_df, "M0")
    row = _row(frame, "M0", mi.TERM_DID)
    assert row["scale"] == "proportion"
    assert row["estimate"] == pytest.approx(0.20, abs=0.04)
    assert row["ci_low"] > 0.0


def test_did_interval_contains_zero_without_an_interaction():
    # 两域的性别差都是 +0.10：性别差本身显著，但 DiD 的真值是 0。
    # 这正是本模块存在的理由——两个各自显著的性别效应并不构成交互。
    wide = _simulate_wide(n_male=1500, n_female=1500, gap_public=0.10,
                          gap_celebrity=0.10, seed=13)
    long_df = mi.to_long_format(wide, "source_entry")
    frame = mi.fit_interaction(long_df, "M0")
    row = _row(frame, "M0", mi.TERM_DID)
    assert row["estimate"] == pytest.approx(0.0, abs=0.05)
    assert row["ci_low"] < 0.0 < row["ci_high"]
    # 但两域各自的性别差都应当稳稳地是正的，否则这份数据没有说服力
    gap_pub = _row(frame, "M0", mi.TERM_GAP_PUBLIC)
    gap_cel = _row(frame, "M0", mi.TERM_GAP_CELEBRITY)
    assert gap_pub["ci_low"] > 0.0
    assert gap_cel["ci_low"] > 0.0


def test_did_equals_the_difference_of_the_two_domain_gaps():
    wide = _simulate_wide(n_male=600, n_female=600, seed=14)
    long_df = mi.to_long_format(wide, "source_entry")
    frame = mi.fit_interaction(long_df, "M0")
    did = _row(frame, "M0", mi.TERM_DID)["estimate"]
    gap_pub = _row(frame, "M0", mi.TERM_GAP_PUBLIC)["estimate"]
    gap_cel = _row(frame, "M0", mi.TERM_GAP_CELEBRITY)["estimate"]
    assert did == pytest.approx(gap_pub - gap_cel, abs=1e-8)


def test_headline_is_not_the_interaction_coefficient():
    # logit 尺度的交互系数与概率尺度的 DiD 是两个数量级不同的数；
    # 任何"直接把系数当 DiD 报出去"的实现都会在这里挂
    wide = _simulate_wide(n_male=1500, n_female=1500, seed=15)
    long_df = mi.to_long_format(wide, "source_entry")
    frame = mi.fit_interaction(long_df, "M0")
    did = _row(frame, "M0", mi.TERM_DID)
    coef = _row(frame, "M0", mi.TERM_COEF)
    assert coef["scale"] == "log_odds"
    assert abs(coef["estimate"]) > 3 * abs(did["estimate"])
    assert -1.0 <= did["estimate"] <= 1.0


def _simulate_zero_coefficient(n_per_cell=6000, seed=41):
    """logit 交互系数恰好为 0、但概率尺度 DiD 明显不为 0 的数据

    两域的性别 log-odds 差相同（都是 log(1.5)），交互系数因此真值为 0；
    但两域的基线概率相差很远（0.50 vs 0.10），同一个 log-odds 差在概率
    尺度上就变成 0.100 与 0.042857，DiD 真值 = 0.057143。

    这是比 "|coef| > 3*|did|" 强得多的判别性设计：
    - "直接报交互系数"的实现给出 ~0；
    - 经典错法"系数 / 4"同样给出 ~0；
    只有真正做四格反事实预测的实现才能给出一个区间不含 0 的正数。
    """
    rng = np.random.default_rng(seed)
    cells = {
        ("m", "public"): 0.60, ("f", "public"): 0.50,
        ("m", "celebrity"): 1.0 / 7.0, ("f", "celebrity"): 0.10,
    }
    n = 2 * n_per_cell
    gender = np.array(["m"] * n_per_cell + ["f"] * n_per_cell)
    df = pd.DataFrame({
        "user_id": [str(i) for i in range(n)],
        "gender": gender,
        "n_posts": rng.integers(1, 200, n),
        "n_retweets": rng.integers(1, 100, n),
        "n_expressive_posts": rng.integers(1, 80, n),
        "n_active_days": rng.integers(1, 300, n),
        "n_active_months": rng.integers(1, 13, n),
        "n_active_months_panel": rng.integers(1, 13, n),
        "region": rng.choice(["north", "south"], n),
        "verified_flag": rng.integers(0, 2, n).astype(float),
        "log_fans": rng.normal(5, 1, n),
        "log_friends": rng.normal(4, 1, n),
        "profile_complete": np.ones(n, dtype=bool),
    })
    for domain in ("public", "celebrity"):
        p = np.where(gender == "m", cells[("m", domain)], cells[("f", domain)])
        df[f"{domain}_source_entered"] = rng.random(n) < p
        df[f"{domain}_topical_share"] = np.clip(p, 0.001, 0.999)
    return df


def test_headline_survives_a_zero_interaction_coefficient():
    """交互系数真值为 0、概率尺度 DiD 真值 0.0571 的判别性设计

    "报系数"和"报系数/4"两种错法在这里都会给出一个区间含 0 的估计，
    而正确实现给出的 DiD 区间必须把 0 排除在外。
    """
    wide = _simulate_zero_coefficient()
    long_df = mi.to_long_format(wide, "source_entry")
    frame = mi.fit_interaction(long_df, "M0")
    did = _row(frame, "M0", mi.TERM_DID)
    coef = _row(frame, "M0", mi.TERM_COEF)

    truth = (0.60 - 0.50) - (1.0 / 7.0 - 0.10)
    assert did["estimate"] == pytest.approx(truth, abs=0.03)
    assert did["ci_low"] > 0.0, "概率尺度的 DiD 区间必须排除 0"
    # 交互系数本身的真值是 0，它的区间应当含 0——"报系数"的实现会因此
    # 得出"没有交互"的结论，与头条正相反
    assert coef["ci_low"] < 0.0 < coef["ci_high"]
    # "系数 / 4" 这种经典错法给出的数也落在 0 附近，同样过不了上面那条
    assert abs(coef["estimate"] / 4.0) < did["estimate"] / 2.0


# ---------------------------------------------------------------------------
# 单域缺失：整个用户剔除
# ---------------------------------------------------------------------------

def test_user_with_one_domain_nan_is_dropped_entirely():
    wide = _simulate_wide(n_male=200, n_female=200, seed=16)
    wide.loc[0, "public_topical_share"] = np.nan     # 只缺一个域
    long_df = mi.to_long_format(wide, "topical_share")
    frame = mi.fit_interaction(long_df, "M0")
    row = _row(frame, "M0", mi.TERM_DID)
    assert row["n_obs"] == len(wide) - 1
    assert row["n_dropped"] == 1
    assert mi.REASON_INCOMPLETE_PAIR in row["drop_reason"]


def test_user_missing_both_domains_is_counted_under_its_own_reason():
    wide = _simulate_wide(n_male=200, n_female=200, seed=17)
    wide.loc[0, ["public_topical_share", "celebrity_topical_share"]] = np.nan
    long_df = mi.to_long_format(wide, "topical_share")
    frame = mi.fit_interaction(long_df, "M0")
    row = _row(frame, "M0", mi.TERM_DID)
    assert row["n_obs"] == len(wide) - 1
    # 两域都缺不是"配对不完整"，是这个用户根本没有有定义的结果变量，
    # 两种流失必须分开记，否则读者无法判断配对规则本身丢了多少人
    assert mi.REASON_INCOMPLETE_PAIR not in (row["drop_reason"] or "")
    assert "no_expressive_posts" in row["drop_reason"]


def test_drop_reason_carries_a_per_reason_user_count():
    """逐条剔除必须带上用户数，且各条相加恰好等于 n_dropped

    只写原因名（"missing_gender+no_expressive_posts+incomplete_domain_pair"）
    加一个合计 n_dropped=3，读者无法判断配对规则本身丢了多少人——而那正是
    本模块第 4 条裁定要求可核对的量。
    """
    wide = _simulate_wide(n_male=200, n_female=200, seed=28)
    wide.loc[0, "gender"] = "unknown"
    wide.loc[[1, 2], ["public_topical_share", "celebrity_topical_share"]] = np.nan
    wide.loc[[3, 4, 5], "public_topical_share"] = np.nan
    long_df = mi.to_long_format(wide, "topical_share")
    row = _row(mi.fit_interaction(long_df, "M0"), "M0", mi.TERM_DID)

    parts = dict(
        (piece.split("=")[0], int(piece.split("=")[1]))
        for piece in row["drop_reason"].split("+")
    )
    assert parts["missing_gender"] == 1
    assert parts["no_expressive_posts"] == 2
    assert parts[mi.REASON_INCOMPLETE_PAIR] == 3
    assert sum(parts.values()) == row["n_dropped"] == 6
    assert row["n_obs"] == len(wide) - 6


def test_format_drop_reason_is_none_when_nothing_was_dropped():
    assert mi.format_drop_reason([]) is None
    assert mi.format_drop_reason([("missing_gender", 2)]) == "missing_gender=2"


def test_sample_accounting_always_adds_up():
    wide = _simulate_wide(n_male=200, n_female=200, seed=18, profile_complete_frac=0.6)
    wide.loc[0, "gender"] = "unknown"
    wide.loc[1, "public_topical_share"] = np.nan
    long_df = mi.to_long_format(wide, "topical_share")
    for layer in ("M0", "M1", "M2"):
        frame = mi.fit_interaction(long_df, layer)
        for _, row in frame.iterrows():
            assert row["n_obs"] + row["n_dropped"] == len(wide)
        assert "missing_gender" in _row(frame, layer, mi.TERM_DID)["drop_reason"]
    m2 = mi.fit_interaction(long_df, "M2")
    assert "incomplete_profile" in _row(m2, "M2", mi.TERM_DID)["drop_reason"]


# ---------------------------------------------------------------------------
# cluster bootstrap：配对 canary
# ---------------------------------------------------------------------------

def _paired_fixture(n_users=40):
    """每个用户两行、共享一个极端的用户基线 b_i

    b_i 取 0 或 100，男女两组里都有这两种基线。只要重抽样是整簇进行的，
    四个 gender × domain 格子里"被抽中男性的平均 b"与"被抽中女性的平均 b"
    在做差中差时会精确抵消，DiD 恒等于 0.20；一旦按行重抽样，公共事务域
    与娱乐域里的用户构成不再相同，b 不再抵消，DiD 立刻抖动起来。
    """
    rows = []
    for i in range(n_users):
        male = float(i % 2)
        b = 0.0 if (i % 4) < 2 else 100.0
        for domain, is_public in (("celebrity", 0.0), ("public", 1.0)):
            eff = male * (0.10 if is_public else -0.10)
            rows.append({
                mi.USER_COLUMN: str(i),
                mi.CLUSTER_COLUMN: str(i),
                mi.DOMAIN_COLUMN: domain,
                mi.DOMAIN_INDICATOR: is_public,
                mi.FOCAL_COLUMN: male,
                mi.OUTCOME_COLUMN: b + eff,
            })
    return pd.DataFrame(rows)


def _cell_means_did(frame):
    """四格样本均值的差中差，不拟合任何模型（让 canary 跑得快且可手算）"""
    cell = frame.groupby([mi.FOCAL_COLUMN, mi.DOMAIN_INDICATOR])[mi.OUTCOME_COLUMN].mean()
    return float((cell[(1.0, 1.0)] - cell[(0.0, 1.0)]) - (cell[(1.0, 0.0)] - cell[(0.0, 0.0)]))


def test_cluster_bootstrap_keeps_a_users_two_rows_together():
    sample = _paired_fixture(n_users=40)
    est, low, high, n_failed = mi._cluster_bootstrap_did(
        sample, _cell_means_did, n_boot=200, seed=3
    )
    assert n_failed == 0
    assert est == pytest.approx(0.20)
    # 配对完整 -> 每次重抽样的 DiD 都精确等于 0.20 -> 区间退化成一个点
    assert low == pytest.approx(0.20, abs=1e-9)
    assert high == pytest.approx(0.20, abs=1e-9)

    # 对照：按行重抽样（配对被打散）时，同一个统计量方差不再为 0。
    # 这条对照就是"打散配对会改变答案"的证据。
    _, low_row, high_row = su.bootstrap_ci(
        sample, _cell_means_did, n_boot=200, seed=3, cluster=None
    )
    assert (high_row - low_row) > 1.0


def test_cluster_bootstrap_relabels_resampled_users_and_preserves_pairs():
    sample = _paired_fixture(n_users=20)
    seen = []

    def _spy(frame):
        seen.append(frame.copy())
        return _cell_means_did(frame)

    mi._cluster_bootstrap_did(sample, _spy, n_boot=25, seed=7)
    assert len(seen) >= 25
    duplicated_somewhere = False
    for frame in seen:
        users = frame[mi.USER_COLUMN].to_numpy()
        domains = frame[mi.DOMAIN_COLUMN].to_numpy()
        # 每两行来自同一个用户，且恰好覆盖两个域
        assert (users[0::2] == users[1::2]).all()
        assert (domains[0::2] != domains[1::2]).all()
        # 被抽中两次的用户必须拿到不同的簇号，否则 GEE 会把 4 行当成一个簇
        clusters = frame[mi.CLUSTER_COLUMN].to_numpy()
        assert len(np.unique(clusters)) == len(frame) // 2
        assert (clusters[0::2] == clusters[1::2]).all()
        if len(np.unique(users)) < len(frame) // 2:
            duplicated_somewhere = True
    assert duplicated_somewhere, "有放回抽样里应当出现被抽中多次的用户"


def test_cluster_bootstrap_rejects_a_frame_whose_pairs_are_broken():
    sample = _paired_fixture(n_users=10)
    broken = sample.drop(index=[0]).reset_index(drop=True)   # 某个用户只剩一行
    with pytest.raises(ValueError):
        mi._cluster_bootstrap_did(broken, _cell_means_did, n_boot=5, seed=1)


# ---------------------------------------------------------------------------
# cov_params 不可用时必须走重拟合的 cluster bootstrap
# ---------------------------------------------------------------------------

class _NoCovParams:
    """把一个已拟合结果包起来，只让 cov_params() 声明"不可用" """

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def cov_params(self):
        raise NotImplementedError("测试用：本次拟合没有协方差矩阵")


def test_bootstrap_path_used_when_cov_params_unavailable():
    wide = _simulate_wide(n_male=120, n_female=120, seed=21)
    long_df = mi.to_long_format(wide, "source_entry")
    sample, *_ = mi._select_sample(long_df, "M0")
    formula, _, _ = mi._build_formula(sample, "M0")
    fit = mi._fit_gee(sample, formula, "source_entry")
    result = mi.difference_in_differences(
        _NoCovParams(fit), sample,
        fit_fn=lambda frame: mi._fit_gee(frame, formula, "source_entry"),
        n_boot=25, seed=5,
    )
    assert result["method"] == mi.METHOD_BOOTSTRAP
    assert np.isfinite(result["se"]) or np.isfinite(result["ci_low"])
    assert np.isfinite(result["ci_low"]) and np.isfinite(result["ci_high"])
    assert result["ci_low"] < result["did"] < result["ci_high"]


def test_delta_method_is_the_default_when_cov_params_exists():
    wide = _simulate_wide(n_male=120, n_female=120, seed=22)
    long_df = mi.to_long_format(wide, "source_entry")
    sample, *_ = mi._select_sample(long_df, "M0")
    formula, _, _ = mi._build_formula(sample, "M0")
    fit = mi._fit_gee(sample, formula, "source_entry")
    result = mi.difference_in_differences(fit, sample)
    assert result["method"] == mi.METHOD_DELTA
    assert np.isfinite(result["se"])


def test_delta_and_bootstrap_intervals_agree():
    # 两条区间路径必须给出实质一致的答案：delta method 是默认路径（22.5 万
    # 用户上唯一现实的选择），重拟合 bootstrap 是它拿不到协方差时的退路。
    # 如果两者不一致，说明其中一条写错了，而论文只会印出其中一条。
    wide = _simulate_wide(n_male=200, n_female=200, seed=31)
    long_df = mi.to_long_format(wide, "source_entry")
    sample, *_ = mi._select_sample(long_df, "M0")
    formula, _, _ = mi._build_formula(sample, "M0")
    fit = mi._fit_gee(sample, formula, "source_entry")

    delta = mi.difference_in_differences(fit, sample)
    boot = mi.difference_in_differences(
        fit, sample, fit_fn=lambda frame: mi._fit_gee(frame, formula, "source_entry"),
        method=mi.METHOD_BOOTSTRAP, n_boot=60, seed=5,
    )
    assert boot["n_boot_failed"] == 0
    assert boot["did"] == pytest.approx(delta["did"], abs=1e-8)
    width_delta = delta["ci_high"] - delta["ci_low"]
    width_boot = boot["ci_high"] - boot["ci_low"]
    assert 0.6 < width_boot / width_delta < 1.6
    assert boot["ci_low"] == pytest.approx(delta["ci_low"], abs=0.4 * width_delta)


def test_auto_refuses_to_enter_an_unaffordable_bootstrap(monkeypatch):
    """auto 不允许自己走进一条跑不完的重拟合 bootstrap

    delta method 失效在全样本上会让 500 次重拟合变成约 10 小时的作业，
    必然撞上批处理墙钟；把上限调到 10 个用户来模拟"样本超限"这一情形。
    """
    wide = _simulate_wide(n_male=80, n_female=80, seed=29)
    long_df = mi.to_long_format(wide, "source_entry")
    sample, *_ = mi._select_sample(long_df, "M0")
    formula, _, _ = mi._build_formula(sample, "M0")
    fit = mi._fit_gee(sample, formula, "source_entry")

    monkeypatch.setattr(mi, "AUTO_BOOTSTRAP_MAX_USERS", 10)
    with pytest.raises(ValueError, match="AUTO|上限|auto"):
        mi.difference_in_differences(
            _NoCovParams(fit), sample,
            fit_fn=lambda frame: mi._fit_gee(frame, formula, "source_entry"),
            method="auto", n_boot=5,
        )
    # 但操作者显式选择 bootstrap 时不受这道闸门限制——上限管的是"自动
    # 走进去"，不是"不许用"
    result = mi.difference_in_differences(
        _NoCovParams(fit), sample,
        fit_fn=lambda frame: mi._fit_gee(frame, formula, "source_entry"),
        method=mi.METHOD_BOOTSTRAP, n_boot=15, seed=2,
    )
    assert result["method"] == mi.METHOD_BOOTSTRAP
    assert np.isfinite(result["ci_low"])


def test_difference_in_differences_refuses_nan_interval_without_a_refit_hook():
    wide = _simulate_wide(n_male=80, n_female=80, seed=23)
    long_df = mi.to_long_format(wide, "source_entry")
    sample, *_ = mi._select_sample(long_df, "M0")
    formula, _, _ = mi._build_formula(sample, "M0")
    fit = mi._fit_gee(sample, formula, "source_entry")
    with pytest.raises(ValueError):
        mi.difference_in_differences(_NoCovParams(fit), sample, fit_fn=None)


# ---------------------------------------------------------------------------
# 结果表结构与失败留痕
# ---------------------------------------------------------------------------

def test_every_layer_emits_the_shared_schema():
    wide = _simulate_wide(n_male=250, n_female=250, seed=24)
    long_df = mi.to_long_format(wide, "source_entry")
    for layer in ("M0", "M1", "M2"):
        frame = mi.fit_interaction(long_df, layer)
        assert list(frame.columns) == list(su.RESULT_SCHEMA)
        terms = set(frame["term"])
        assert {mi.TERM_DID, mi.TERM_GAP_PUBLIC, mi.TERM_GAP_CELEBRITY,
                mi.TERM_COEF} <= terms
        did = _row(frame, layer, mi.TERM_DID)
        assert did["domain"] == mi.DOMAIN_BOTH
        assert did["outcome"] == "source_entered"
        assert mi.NOTE_GEE in did["note"]


def test_failed_layer_leaves_nan_rows_instead_of_disappearing():
    wide = _simulate_wide(n_male=0, n_female=200, seed=25)   # 没有性别变异
    long_df = mi.to_long_format(wide, "source_entry")
    frame = mi.fit_interaction(long_df, "M0")
    assert len(frame) == 4
    assert frame["estimate"].isna().all()
    assert frame["note"].str.contains("no_gender_variation").all()


def test_interaction_table_covers_both_outcomes_and_three_layers():
    wide = _simulate_wide(n_male=250, n_female=250, seed=26)
    frame = mi.interaction_table(wide)
    assert list(frame.columns) == list(su.RESULT_SCHEMA)
    did = frame[frame["term"] == mi.TERM_DID]
    assert sorted(did["outcome"].unique()) == ["source_entered", "topical_share"]
    assert sorted(did["model"].unique()) == ["M0", "M1", "M2"]
    assert len(did) == 6


# ---------------------------------------------------------------------------
# 占比越界必须报错，不能悄悄改变头条数字
# ---------------------------------------------------------------------------

def test_out_of_range_share_raises_instead_of_shifting_the_headline():
    """占比 > 1 只可能是上游算错分子分母，必须炸掉而不是照常出数

    二项族拟似然以 [0,1] 为前提。注入几个 1.7 之后 GEE 照样收敛、note 照样
    干净、drop_reason 照样是 None，只有头条数字被悄悄推走——这种"不报错的
    错误"必须在进模型之前被拦下。
    """
    wide = _simulate_wide(n_male=100, n_female=100, seed=30)
    wide.loc[wide.index[:5], "public_topical_share"] = 1.7
    with pytest.raises(ValueError, match="public_topical_share"):
        mi.to_long_format(wide, "topical_share")


def test_negative_share_also_raises():
    wide = _simulate_wide(n_male=50, n_female=50, seed=31)
    wide.loc[0, "celebrity_topical_share"] = -0.2
    with pytest.raises(ValueError, match="celebrity_topical_share"):
        mi.to_long_format(wide, "topical_share")


def test_nan_share_is_still_allowed():
    # NaN 是合法的"该用户此项无定义"，不能被范围校验误伤
    wide = _simulate_wide(n_male=20, n_female=20, seed=32)
    wide.loc[0, "public_topical_share"] = np.nan
    long_df = mi.to_long_format(wide, "topical_share")
    assert long_df[mi.OUTCOME_COLUMN].isna().sum() == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_build_writes_incrementally_so_a_killed_job_keeps_finished_layers(
        tmp_path, monkeypatch):
    """每完成一层就落盘：中途失败时已经算完的层必须还在磁盘上

    让第 3 层拟合抛异常，然后断言 parquet 已经存在、只含前两层，
    且 manifest 的 completed_layers 如实说明这份文件被截断到哪里。
    """
    wide = _simulate_wide(n_male=100, n_female=100, seed=33)
    monkeypatch.setattr(mi.config, "OUTPUT_DIR", str(tmp_path))
    wide.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                    engine="pyarrow", index=False)

    real_fit = mi.fit_interaction
    calls = {"n": 0}

    def _explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("模拟第三层拟合失败")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(mi, "fit_interaction", _explode)
    with pytest.raises(RuntimeError):
        mi.build(year=2020)

    path = tmp_path / "results" / "interaction_gender_domain.parquet"
    assert path.exists(), "中途失败时已经算完的层必须已经落盘"
    frame = pd.read_parquet(path, columns=list(su.RESULT_SCHEMA))
    assert sorted(frame["model"].unique()) == ["M0", "M1"]

    with open(tmp_path / "results" / "manifest_interaction" / "manifest.json",
              encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["counts"]["n_completed_layers"] == 2
    assert manifest["counts"]["completed_layers"] == [
        "source_entry/M0", "source_entry/M1"
    ]


def test_build_writes_the_interaction_table(tmp_path, monkeypatch):
    wide = _simulate_wide(n_male=150, n_female=150, seed=27)
    monkeypatch.setattr(mi.config, "OUTPUT_DIR", str(tmp_path))
    wide.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                    engine="pyarrow", index=False)

    path = mi.build(year=2020)
    assert path.endswith("interaction_gender_domain.parquet")
    frame = pd.read_parquet(path, columns=list(su.RESULT_SCHEMA))
    assert list(frame.columns) == list(su.RESULT_SCHEMA)
    assert (tmp_path / "results" / "manifest_interaction" / "manifest.json").exists()
