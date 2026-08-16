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


def test_risk_ratio_ci_denominator_zero_numerator_positive_returns_nan_not_inf():
    # s2=0 但 s1>0 时，旧实现返回 rr=inf——"分母组从未发生"已经让比率
    # 本身失去意义，不应该给一个看似有信息量的 inf（下游排序/汇总会被
    # 误导）；修复后统一记为 NaN，与"两组都为 0"的情形一致。
    rr, low, high = su.risk_ratio_ci(5, 100, 0, 100)
    assert np.isnan(rr)
    assert not np.isinf(rr)
    assert np.isnan(low)
    assert np.isnan(high)


# ---------------------------------------------------------------------------
# check_proportion_range
# ---------------------------------------------------------------------------

def test_check_proportion_range_accepts_valid_values_and_counts_them():
    n_valid = su.check_proportion_range([0.0, 0.5, 1.0, np.nan], "some_share")
    # NaN 是合法的"该用户此项无定义"，不参与校验也不计入有效观测
    assert n_valid == 3


def test_check_proportion_range_rejects_values_above_one():
    # 占比 > 1 只可能是上游分子/分母口径算错。二项族拟似然照样能收敛，
    # 只会把估计悄悄推走，所以这里必须抛异常而不是 clip
    with pytest.raises(ValueError, match="public_topical_share"):
        su.check_proportion_range([0.2, 0.4, 1.7], "public_topical_share")


def test_check_proportion_range_rejects_negative_values():
    with pytest.raises(ValueError, match="share_col"):
        su.check_proportion_range([-0.01, 0.5], "share_col")


def test_check_proportion_range_message_names_the_offenders():
    with pytest.raises(ValueError) as exc:
        su.check_proportion_range([0.1, 1.4, 2.5], "bad_col")
    message = str(exc.value)
    assert "2 个观测" in message      # 越界个数
    assert "1.4" in message and "2.5" in message


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


def test_top_share_zero_q_returns_zero_not_the_single_user_floor():
    # 旧实现里 k = max(ceil(q*n), 1)，q=0.0 时也会被地板到 k=1，
    # 于是"前 0 个用户的份额"被悄悄算成了"贡献最大的 1 个用户的份额"，
    # 结果是个看似合理、实则错误的非零数。修复后 q<=0 直接返回 0.0。
    values = np.array([1.0, 2.0, 1000.0])
    assert su.top_share(values, 0.0) == 0.0


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


class _MissingCovWrapper:
    """包一层已拟合结果，模拟"模型压根没有 cov_params 方法"的情况
    （合法的"协方差不可用"信号之一）"""

    def __init__(self, fitted_result):
        self.params = fitted_result.params
        self.model = fitted_result.model


class _NotImplementedCovWrapper:
    """cov_params() 显式抛 NotImplementedError——statsmodels 里部分拟合
    方式表示"这次拟合没有协方差矩阵"的合法信号，也应被当作不可用"""

    def __init__(self, fitted_result):
        self.params = fitted_result.params
        self.model = fitted_result.model

    def cov_params(self):
        raise NotImplementedError("模拟这次拟合没有协方差矩阵")


class _BuggyCovWrapper:
    """cov_params() 抛出与"协方差不可用"无关的任意异常（模拟调用方代码
    本身的 bug），用来确认这类异常必须原样向上抛出，不能被当成
    "协方差不可用"悄悄吞掉——这正是 C1 修复要收窄的那个 except 范围"""

    def __init__(self, fitted_result):
        self.params = fitted_result.params
        self.model = fitted_result.model

    def cov_params(self):
        raise RuntimeError("模拟调用方代码里的真实 bug，不是协方差不可用")


class _FixedCovWrapper:
    """cov_params() 返回调用方指定的（可能是坏的）矩阵，用来测试
    NaN / 形状不匹配两种"协方差矩阵本身有问题"的场景"""

    def __init__(self, fitted_result, cov_matrix):
        self.params = fitted_result.params
        self.model = fitted_result.model
        self._cov_matrix = cov_matrix

    def cov_params(self):
        return self._cov_matrix


def test_average_marginal_effect_returns_nan_interval_when_cov_params_missing(capsys):
    # C1/I3 修复后的契约：协方差矩阵确实不可用时，不再退化成任何区间
    # （旧版本的 bootstrap 退化路径固定住已拟合参数、只重抽样预测数据，
    # 完全不反映参数不确定性，实测比真实 delta method 区间窄了一个
    # 数量级，比"没有区间"更危险）。现在必须返回 NaN，并打印警告说明
    # 原因，而不是让调用方误以为拿到了一个可信的区间。
    res, df = _saturated_two_group_logit()
    ame, se, low, high = su.average_marginal_effect(_MissingCovWrapper(res), "x", df)
    assert ame == pytest.approx(0.20, abs=1e-3)
    assert np.isnan(se)
    assert np.isnan(low)
    assert np.isnan(high)
    captured = capsys.readouterr()
    assert "警告" in captured.out


def test_average_marginal_effect_returns_nan_interval_when_cov_params_not_implemented(capsys):
    res, df = _saturated_two_group_logit()
    ame, se, low, high = su.average_marginal_effect(
        _NotImplementedCovWrapper(res), "x", df
    )
    assert ame == pytest.approx(0.20, abs=1e-3)
    assert np.isnan(se) and np.isnan(low) and np.isnan(high)
    assert "警告" in capsys.readouterr().out


def test_average_marginal_effect_propagates_unrelated_cov_params_error():
    # 与上面两个测试相反：cov_params() 抛出的不是"协方差不可用"的信号，
    # 而是任意异常（模拟真实 bug），必须原样向上抛出，绝不能被吞掉后
    # 悄悄退化成一个看似正常的结果——这是 C1 那个过宽 except 的直接回归测试。
    res, df = _saturated_two_group_logit()
    with pytest.raises(RuntimeError):
        su.average_marginal_effect(_BuggyCovWrapper(res), "x", df)


def test_average_marginal_effect_raises_on_nan_covariance_matrix():
    # cov_params() 返回一个形状正确但全是 NaN 的矩阵：grad @ cov @ grad
    # 不会自动报错（NaN 参与运算只会静默传播成 NaN），旧版本这里被外层
    # except Exception 吞掉后退化成一个 se=0.000457 的窄区间（真实 delta
    # SE 应为 0.006370，区间因此窄了约 14 倍）。修复后必须显式检测无效
    # 方差并 raise，而不是被更外层的 except 拦下来变成一个看似正常的数。
    res, df = _saturated_two_group_logit()
    bad_cov = np.full((2, 2), np.nan)
    with pytest.raises(ValueError):
        su.average_marginal_effect(_FixedCovWrapper(res, bad_cov), "x", df)


def test_average_marginal_effect_raises_on_wrong_shaped_covariance_matrix():
    # cov_params() 返回的矩阵维度与参数个数不匹配（这里模型有 2 个参数，
    # 却返回一个 3x3 矩阵）：numpy 的 grad @ cov @ grad 会在维度不匹配时
    # 自然抛 ValueError；这个测试确认该错误不会被吞掉，而是原样传出。
    res, df = _saturated_two_group_logit()
    wrong_shape_cov = np.eye(3)
    with pytest.raises(ValueError):
        su.average_marginal_effect(_FixedCovWrapper(res, wrong_shape_cov), "x", df)


def test_average_marginal_effect_distinguishes_ame_from_mem_with_skewed_control():
    # I2 修复：饱和两组 fixture 里 focal 变量是唯一协变量，AME 和"在协变
    # 量均值处求边际效应"（MEM）代数上恒等，测试测不出"错误地在均值处
    # 求值"这个陷阱。这里加入一个右偏的连续协变量 z（指数分布，均值
    # 1.5、但有长尾到 17+），并且不经过拟合——直接手写已知真参数，
    # 这样 AME 和 MEM 都能独立于 average_marginal_effect 之外、用同一个
    # sigmoid 定义单独算出来，互相校验：
    #   a=-3.0, b_x=1.2, b_z=2.0
    #   AME_true = mean_i[ sigmoid(a+b_x+b_z*z_i) - sigmoid(a+b_z*z_i) ]
    #   MEM_true = sigmoid(a+b_x+b_z*mean(z)) - sigmoid(a+b_z*mean(z))
    # 这两个数在非线性模型 + 有方差的协变量下一般不相等，算出来相差
    # 约 0.10（远超浮点误差），能明确分辨"反事实平均"和"均值处求值"
    # 两种实现。
    rng = np.random.default_rng(123)
    n = 20000
    x = rng.integers(0, 2, size=n).astype(float)
    z = rng.exponential(scale=1.5, size=n)
    a, b_x, b_z = -3.0, 1.2, 2.0

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    eta1 = a + b_x * 1.0 + b_z * z
    eta0 = a + b_x * 0.0 + b_z * z
    ame_true = float(np.mean(sigmoid(eta1) - sigmoid(eta0)))
    mean_z = z.mean()
    mem_true = float(
        sigmoid(a + b_x * 1.0 + b_z * mean_z) - sigmoid(a + b_x * 0.0 + b_z * mean_z)
    )
    # 确认这个 fixture 确实具有判别力：两个候选答案要离得足够远
    assert abs(ame_true - mem_true) > 0.05

    # 构造一个未拟合的 statsmodels GLM，直接用已知真参数预测（不经过
    # .fit()，没有估计噪声）；exog 用 statsmodels 默认生成的列名
    # const/x1/x2，因此 data 的列名也必须是这三个，focal 变量是 x1
    exog = np.column_stack([np.ones(n), x, z])
    y_placeholder = np.zeros(n)  # 未拟合，endog 内容不影响 predict
    model = sm.GLM(y_placeholder, exog, family=sm.families.Binomial())
    df = pd.DataFrame({"const": 1.0, "x1": x, "x2": z})

    class _KnownParamsWrapper:
        def __init__(self):
            self.params = np.array([a, b_x, b_z])
            self.model = model

    ame, _, _, _ = su.average_marginal_effect(_KnownParamsWrapper(), "x1", df)
    assert ame == pytest.approx(ame_true, abs=1e-9)
    assert ame != pytest.approx(mem_true, abs=1e-3)
