"""
gender_domain.stats_utils 的单元测试。

每个断言的期望数字都在测试内的注释里写出推导过程，方便复核，而不是
只验证"函数跑起来不报错"——这些函数产出的数字是要写进论文的。
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from gender_domain import stats_utils as su


# ---------------------------------------------------------------------------
# proportion_ci（Wilson score interval）
# ---------------------------------------------------------------------------

def test_proportion_ci_wilson_matches_textbook_5_of_20():
    # Wilson 区间闭式解，z = norm.ppf(0.975) = 1.959963984540054：
    #   p_hat = 5/20 = 0.25
    #   denom = 1 + z^2/20 = 1.192059...
    #   center = (0.25 + z^2/40) / denom
    #   margin = z * sqrt(0.25*0.75/20 + z^2/1600) / denom
    # 手算（及用 statsmodels.stats.proportion.proportion_confint(5, 20,
    # method="wilson") 交叉核对，两者一致）得到：
    #   low  = 0.111861701...
    #   high = 0.468700877...
    low, high = su.proportion_ci(5, 20)
    assert low == pytest.approx(0.1119, abs=1e-4)
    assert high == pytest.approx(0.4687, abs=1e-4)


def test_proportion_ci_zero_successes_lower_bound_zero_upper_positive():
    low, high = su.proportion_ci(0, 20)
    assert low == 0.0
    assert high > 0.0
    # 手算：p_hat=0 时 margin == center，因此 low = center - margin = 0
    # 恰好精确成立，不只是"clip 到接近 0"
    assert high == pytest.approx(0.16113, abs=1e-4)


def test_proportion_ci_all_successes_upper_bound_one():
    low, high = su.proportion_ci(20, 20)
    assert high == 1.0
    assert low < 1.0
    assert low == pytest.approx(0.83887, abs=1e-4)


# ---------------------------------------------------------------------------
# proportion_diff_ci（Newcombe 方法 10）
# ---------------------------------------------------------------------------

def test_proportion_diff_ci_contains_observed_difference():
    # 30/100 = 0.30, 45/100 = 0.45, 观测差值 = 0.30 - 0.45 = -0.15
    diff, low, high = su.proportion_diff_ci(30, 100, 45, 100)
    assert diff == pytest.approx(-0.15)
    assert low <= diff <= high


def test_proportion_diff_ci_sign_flips_under_argument_swap():
    diff_a, low_a, high_a = su.proportion_diff_ci(30, 100, 45, 100)
    diff_b, low_b, high_b = su.proportion_diff_ci(45, 100, 30, 100)
    # Newcombe 方法的代数性质：交换两组后 diff 恰好变号，
    # 区间围绕新的差值对称重新定位：low_b == -high_a, high_b == -low_a，
    # 宽度不变
    assert diff_b == pytest.approx(-diff_a)
    assert low_b == pytest.approx(-high_a)
    assert high_b == pytest.approx(-low_a)
    assert (high_b - low_b) == pytest.approx(high_a - low_a)


# ---------------------------------------------------------------------------
# risk_ratio_ci
# ---------------------------------------------------------------------------

def test_risk_ratio_ci_returns_two_for_20_100_vs_10_100():
    # (20/100) / (10/100) = 0.2 / 0.1 = 2.0；用交叉相乘 (20*100)/(100*10)
    # = 2000/1000 = 2.0 计算，避免两次浮点除法各自舍入
    rr, low, high = su.risk_ratio_ci(20, 100, 10, 100)
    assert rr == 2.0
    assert low < 2.0 < high


def test_risk_ratio_ci_nan_bounds_when_numerator_zero_does_not_raise():
    rr, low, high = su.risk_ratio_ci(0, 100, 10, 100)
    assert rr == 0.0
    assert np.isnan(low)
    assert np.isnan(high)

    # 两个分子都为 0：rr 本身也无法定义
    rr2, low2, high2 = su.risk_ratio_ci(0, 100, 0, 100)
    assert np.isnan(rr2)
    assert np.isnan(low2)
    assert np.isnan(high2)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_reproducible_with_fixed_seed():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=5.0, scale=2.0, size=200)
    result_a = su.bootstrap_ci(values, np.mean, n_boot=500, seed=7)
    result_b = su.bootstrap_ci(values, np.mean, n_boot=500, seed=7)
    assert result_a == result_b


def test_bootstrap_ci_cluster_wider_than_ignoring_clusters():
    # 构造 10 个"簇内完全相同"的簇：5 个簇的值全是 0，5 个簇的值全是 10，
    # 每簇 20 行。真实的组间方差由 10 个簇决定，而不是 200 行；
    # 忽略聚类、按行重抽样会把 200 行当作独立信息，系统性低估标准误，
    # 因此区间应该明显更窄。
    cluster_value = np.repeat([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0], 20)
    cluster_labels = np.repeat(np.arange(10), 20)

    _, low_row, high_row = su.bootstrap_ci(
        cluster_value, np.mean, n_boot=2000, seed=1, cluster=None
    )
    _, low_clu, high_clu = su.bootstrap_ci(
        cluster_value, np.mean, n_boot=2000, seed=1, cluster=cluster_labels
    )
    width_row = high_row - low_row
    width_clu = high_clu - low_clu
    assert width_clu > width_row


def test_bootstrap_ci_cluster_keeps_a_cluster_rows_paired_together():
    # 判别性更强的构造：20 个簇，每簇恰好 2 行——一个"小"值（< 1000）、
    # 一个"大"值（>= 1000），二者配对且簇与簇之间的配对关系互不相同。
    # 只要簇是整簇（连同其全部行）被抽中，那么无论抽中哪些簇、抽中几次，
    # 重抽样得到的 40 行里"大值行数 / 总行数"这个统计量永远精确等于
    # 0.5（每个被抽中的簇都恰好贡献 1 个大值 + 1 个小值），因此这个
    # 统计量在簇 bootstrap 下方差应为 0，百分位区间退化为一个点。
    # 如果实现变成"簇内独立重抽样两行"（错误实现之一），同一簇的两行
    # 有 1/4 的概率被抽成 (大,大) 或 (小,小)，破坏这个恰好 0.5 的配对，
    # 该统计量就会出现非零方差——这正是本测试要抓住的错误。
    n_clusters = 20
    small = np.arange(n_clusters, dtype=float)
    big = 1000.0 + np.arange(n_clusters, dtype=float)
    values = np.empty(2 * n_clusters)
    values[0::2] = small
    values[1::2] = big
    cluster_labels = np.repeat(np.arange(n_clusters), 2)

    def frac_big(v):
        return float(np.mean(np.asarray(v) >= 1000.0))

    est, low, high = su.bootstrap_ci(
        values, frac_big, n_boot=500, seed=3, cluster=cluster_labels
    )
    assert est == pytest.approx(0.5)
    assert low == pytest.approx(0.5, abs=1e-9)
    assert high == pytest.approx(0.5, abs=1e-9)

    # 对照：忽略聚类按行重抽样，同一个统计量在 40 行的池子里独立抽样，
    # 不再保证每次重抽样都恰好 20 个大值，方差不再是 0
    _, low_row, high_row = su.bootstrap_ci(
        values, frac_big, n_boot=500, seed=3, cluster=None
    )
    assert (high_row - low_row) > 0.0


# ---------------------------------------------------------------------------
# top_share
# ---------------------------------------------------------------------------

def test_top_share_one_user_holds_all_mass():
    values = np.array([0.0] * 99 + [1000.0])
    # n=100, q=0.01 -> k = ceil(0.01*100) = 1，恰好是那个占全部质量的用户
    assert su.top_share(values, 0.01) == pytest.approx(1.0)
    # q 更大时，头部仍然囊括那个用户，份额依旧是 1.0
    assert su.top_share(values, 0.5) == pytest.approx(1.0)


def test_top_share_uniform_mass_is_approximately_q():
    # 1000 个相同的值，均匀分布质量：k = ceil(0.05*1000) = 50，
    # 份额 = 50/1000 = 0.05，与 q 精确相等（此处能整除，非"约等于"）
    values = np.ones(1000)
    assert su.top_share(values, 0.05) == pytest.approx(0.05)


def test_top_share_excludes_nan_from_denominator_and_numerator():
    values = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
    # 有效值为 [1,2,3,4]，n=4；q=0.25 -> k=ceil(1)=1 -> 顶部值 4/(1+2+3+4)=0.4
    assert su.top_share(values, 0.25) == pytest.approx(4 / 10)


# ---------------------------------------------------------------------------
# tidy_result
# ---------------------------------------------------------------------------

def test_tidy_result_minimal_call_keys_match_schema():
    row = su.tidy_result(
        outcome="source_entered", domain="public", model="M0",
        term="gender", estimate=0.05, scale="probability", n_obs=1000,
    )
    assert set(row.keys()) == set(su.RESULT_SCHEMA)
    assert row["estimate"] == 0.05
    # 未传入的可选字段一律为 None，而不是缺失 key
    for optional_field in ("se", "ci_low", "ci_high", "n_dropped", "drop_reason", "note"):
        assert row[optional_field] is None


def test_tidy_result_full_call_keys_match_schema():
    kwargs = {col: f"val_{col}" for col in su.RESULT_SCHEMA}
    kwargs["estimate"] = 0.12
    kwargs["se"] = 0.01
    kwargs["ci_low"] = 0.10
    kwargs["ci_high"] = 0.14
    kwargs["n_obs"] = 500
    kwargs["n_dropped"] = 5
    row = su.tidy_result(**kwargs)
    assert set(row.keys()) == set(su.RESULT_SCHEMA)
    for col in su.RESULT_SCHEMA:
        assert row[col] == kwargs[col]


def test_tidy_result_rejects_unknown_field():
    with pytest.raises(ValueError):
        su.tidy_result(outcome="x", made_up_field=1)


# ---------------------------------------------------------------------------
# average_marginal_effect
# ---------------------------------------------------------------------------

def _saturated_two_group_logit():
    """两组饱和逻辑回归：MLE 精确复现各组经验比例，没有渐近误差

    构造 n0=1000 行 x=0（其中 s0=300 行 y=1），n1=1000 行 x=1（其中
    s1=500 行 y=1）。模型 y ~ x 只有截距和一个二值哑变量，恰好两个
    自由参数对应两组，是饱和模型：MLE 精确解出
        p0_hat = s0/n0 = 0.30, p1_hat = s1/n1 = 0.50
    （对数似然对截距、斜率的一阶条件就是"预测概率等于组内经验比例"，
    两组两参数，方程组恰好有解且唯一），不依赖大样本近似。
    因此反事实预测：把全部行的 x 设为 1，预测值对每行都是 p1_hat；
    把全部行的 x 设为 0，预测值对每行都是 p0_hat；AME 精确等于
    p1_hat - p0_hat = 0.50 - 0.30 = 0.20，不是仿真估计出来的近似值。
    """
    n0, s0 = 1000, 300
    n1, s1 = 1000, 500
    y = np.array([1] * s0 + [0] * (n0 - s0) + [1] * s1 + [0] * (n1 - s1))
    x = np.array([0] * n0 + [1] * n1)
    df = pd.DataFrame({"y": y, "x": x})
    res = smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()
    return res, df


def test_average_marginal_effect_recovers_analytic_ame_saturated_logit():
    res, df = _saturated_two_group_logit()
    ame, se, low, high = su.average_marginal_effect(res, "x", df)
    assert ame == pytest.approx(0.20, abs=1e-3)

    # delta method 的解析对照：两组独立二项比例之差的方差
    #   Var(AME) = p1(1-p1)/n1 + p0(1-p0)/n0
    #            = 0.5*0.5/1000 + 0.3*0.7/1000 = 0.00025 + 0.00021 = 0.00046
    #   SE = sqrt(0.00046) = 0.0214476...
    expected_se = np.sqrt(0.5 * 0.5 / 1000 + 0.3 * 0.7 / 1000)
    assert se == pytest.approx(expected_se, abs=1e-3)
    assert low < ame < high


class _NoCovWrapper:
    """包一层已拟合结果，模拟 cov_params() 不可用的模型，只用于测试
    average_marginal_effect 的 bootstrap 退化路径"""

    def __init__(self, fitted_result):
        self.params = fitted_result.params
        self.model = fitted_result.model

    def cov_params(self):
        raise RuntimeError("模拟协方差矩阵不可用")


def test_average_marginal_effect_falls_back_to_bootstrap_when_cov_unavailable():
    res, df = _saturated_two_group_logit()
    wrapped = _NoCovWrapper(res)
    ame, se, low, high = su.average_marginal_effect(
        wrapped, "x", df, n_boot=300, seed=5
    )
    assert ame == pytest.approx(0.20, abs=1e-3)
    # 这个 fixture 里模型只有 Intercept + x 两个协变量，反事实预测把 x
    # 强制设为 1/0 后，每一行的预测值只取决于 x（被覆盖为常量），与该行
    # 原本是哪个用户无关；因此对 data 的行做 bootstrap 重抽样时，无论
    # 抽中哪些行，AME 的计算结果都恒等于同一个值——区间理应退化为一个
    # 点（low == ame == high），这正说明退化路径是在"就着已拟合参数、
    # 重新走一遍反事实预测"，而不是在瞎抽样出别的数字。
    assert low <= ame <= high
    assert low == pytest.approx(ame, abs=1e-6)
    assert high == pytest.approx(ame, abs=1e-6)
    assert np.isfinite(se)
