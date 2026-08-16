"""
gender_domain.models_temporal 的单元测试（§9 月度稳定性、§10 转发延迟）。

测试设计原则（写在这里，避免后来者以为某条断言只是"顺手加的"）：

1. **§9 存在的理由是"年度差异是不是被某一两个月推出来的"。**
   因此 leave_one_month_out 必须每个域恰好 13 行（12 次留一月 + 1 行全年
   参照），少一行读者就得自己去别的表里找参照值，比较不再自足；缺月也
   必须留一行 NaN 痕迹，而不是悄悄少一行。

2. **月份固定效应必须真的把月份吃进去，并且按用户聚类。**
   一条"恒定性别差"的模拟数据上必须把这个差估回来（否则固定效应写错了
   也看不出来）；同一份数据上聚类标准误必须明显大于把用户—月当独立观测
   的稳健标准误（否则 cov_type 写没写都一样，那条约定形同虚设）。

3. **清理规则每一条都要单独记数，且相加恒等于被删的行数。**
   fixture 里刻意各放一条：缺失延迟、负延迟、5 万小时延迟、去年旧帖、
   重复的 用户×原帖 对。每条都必须被对应的规则删掉、被对应的计数记下，
   互相不能吞掉对方（例如"去年旧帖"必须在"超长延迟"之前判定，否则
   跨年转发会被记成异常值，而它其实是一个完全正常的行为）。

4. **1 小时内完成的比例必须算在清理后的集合上。**
   fixture 让原始集合里多出一条 0.5 小时、但来源是去年旧帖的记录：算在
   原始集合上会得到 2/6，算在清理后的集合上是 1/5。两个数字刻意不相等，
   这条断言才有意义。

5. **用户级聚合先于男女比较。** fixture 里放一个刷了 100 条、每条都很慢的
   男性用户：记录级中位数会被他一个人拉到 100 小时，用户级中位数的男女差
   仍然是 0。这正是 §10.3 要求"每名用户的中位延迟 + 用户聚类 bootstrap"
   的原因，因此用一条测试钉死。

6. **均值和 t 检验不是头条。** 输出里不允许出现任何 t 检验行；均值行必须
   带着"次要"的 note 标签，扫描输出即可验证。
"""

import json

import numpy as np
import pandas as pd
import pytest

from gender_domain import models_temporal as mt
from gender_domain import stats_utils as su

_EPOCH = pd.Timestamp("1970-01-01")


# ---------------------------------------------------------------------------
# 模拟数据构造
# ---------------------------------------------------------------------------

def _ts(beijing_text):
    """北京时间字符串 -> UTC epoch 秒（原始数据里的时间戳按 UTC epoch 存）"""
    return float((pd.Timestamp(beijing_text) - pd.Timedelta(hours=8) - _EPOCH)
                 .total_seconds())


def _events(records, unit="s"):
    """按 build_retweet_table 的输出列构造一份事件表 fixture

    records 里每条给 (user_id, gender, domain, source_post, source_time,
    delay_hours)；delay_hours 为 None 表示时间戳缺失（换算失败）。
    """
    rows = []
    for i, rec in enumerate(records):
        source_ts = _ts(rec["source_time"])
        delay_hours = rec.get("delay_hours")
        if delay_hours is None:
            retweet_ts = np.nan
            retweet_time = pd.Timestamp(rec["source_time"])
        else:
            retweet_ts = source_ts + delay_hours * 3600.0
            retweet_time = pd.Timestamp(rec["source_time"]) + pd.Timedelta(
                hours=delay_hours
            )
        scale = 1.0 if unit == "s" else 1000.0
        rows.append({
            "user_id": rec["user_id"],
            "weibo_id": f"w{i}",
            "r_weibo_id": rec["source_post"],
            "r_user_id": rec.get("r_user_id", "src1"),
            "date": retweet_time.strftime("%Y-%m-%d"),
            "month": int(retweet_time.month),
            "gender": rec["gender"],
            "source_domain": rec["domain"],
            "source_category": "news",
            "retweet_ts": retweet_ts * scale,
            "source_ts": source_ts * scale,
        })
    frame = pd.DataFrame(rows)
    frame["delay_seconds"] = frame["retweet_ts"] - frame["source_ts"]
    frame["delay_valid"] = frame["delay_seconds"].notna() & (frame["delay_seconds"] > 0)
    return frame


def _clean_frame(records):
    """直接构造一份"已清理"的延迟表（delay_quantiles 的输入形状）"""
    return pd.DataFrame({
        "user_id": [r[0] for r in records],
        "gender": [r[1] for r in records],
        "source_domain": [r[2] for r in records],
        "delay_hours": [float(r[3]) for r in records],
    })


def _panel(n_users=200, months=tuple(range(1, 13)), gap=0.15, base=0.30,
           month_shift=0.0, seed=0, persistent=False):
    """构造一份表 D 形状的用户—月份面板

    每个用户在每个月一行；进入概率 = base + month_shift*(月份-6.5)/12
    （女性）或再加 gap（男性），即**概率尺度上的性别差是恒定的**，
    月份固定效应模型必须把 gap 估回来。
    persistent=True 时把用户层面的持续倾向注入进去（同一用户各月高度相关），
    用来检验聚类标准误确实比"当独立观测"的标准误大。
    """
    rng = np.random.default_rng(seed)
    genders = np.array(["m", "f"] * (n_users // 2))
    if len(genders) < n_users:
        genders = np.append(genders, "f")
    user_effect = rng.random(n_users) if persistent else np.zeros(n_users)
    rows = []
    for i in range(n_users):
        male = genders[i] == "m"
        for month in months:
            p = base + month_shift * (month - 6.5) / 12.0 + (gap if male else 0.0)
            if persistent:
                # 用户效应把同一用户的各月绑在一起：要么几乎每月都参与，
                # 要么几乎每月都不参与
                entered = user_effect[i] < p
            else:
                entered = rng.random() < p
            rows.append({
                "user_id": str(i),
                "month": int(month),
                "gender": genders[i],
                "n_posts": int(rng.integers(1, 30)),
                "n_retweets": int(rng.integers(0, 20)),
                "n_expressive_posts": int(rng.integers(1, 15)),
                "n_active_days": int(rng.integers(1, 28)),
                "public_topical_posts": 0,
                "public_topical_share": float(rng.random() * 0.5),
                "public_char_density": float(rng.random() * 0.2),
                "public_source_count": int(entered),
                "public_source_entered": bool(entered),
                "celebrity_topical_posts": 0,
                "celebrity_topical_share": float(rng.random() * 0.5),
                "celebrity_char_density": float(rng.random() * 0.2),
                "celebrity_source_count": int(rng.random() < 0.3),
                "celebrity_source_entered": bool(rng.random() < 0.3),
            })
    return pd.DataFrame(rows)


def _tiny_panel():
    """手算用的极小面板：2 个月 × 4 个用户（2 男 2 女）"""
    rows = []
    entered = {
        # (user, month) -> public_source_entered
        ("0", 1): True, ("0", 2): True,      # 男
        ("1", 1): False, ("1", 2): True,     # 男
        ("2", 1): True, ("2", 2): False,     # 女
        ("3", 1): False, ("3", 2): False,    # 女
    }
    shares = {
        ("0", 1): 0.4, ("0", 2): 0.2,
        ("1", 1): 0.6, ("1", 2): 0.8,
        ("2", 1): 0.1, ("2", 2): 0.3,
        ("3", 1): np.nan, ("3", 2): 0.5,     # 该用户 1 月没有表达帖
    }
    for user, gender in (("0", "m"), ("1", "m"), ("2", "f"), ("3", "f")):
        for month in (1, 2):
            rows.append({
                "user_id": user,
                "month": month,
                "gender": gender,
                "n_posts": 5,
                "n_retweets": 2,
                "n_expressive_posts": 0 if np.isnan(shares[(user, month)]) else 4,
                "n_active_days": 3,
                "public_topical_posts": 1,
                "public_topical_share": shares[(user, month)],
                "public_char_density": 0.1,
                "public_source_count": int(entered[(user, month)]),
                "public_source_entered": entered[(user, month)],
                "celebrity_topical_posts": 0,
                "celebrity_topical_share": 0.2,
                "celebrity_char_density": 0.1,
                "celebrity_source_count": 0,
                "celebrity_source_entered": False,
            })
    return pd.DataFrame(rows)


def _assert_schema(frame):
    assert list(frame.columns) == list(su.RESULT_SCHEMA)


def _assert_unique_keys(frame):
    key = ["outcome", "domain", "model", "term"]
    dup = frame[frame.duplicated(subset=key, keep=False)]
    assert dup.empty, (
        f"{len(dup)} 行重复键，例如\n{dup.sort_values(key).head(8).to_string(index=False)}"
    )


# ---------------------------------------------------------------------------
# §9.2 分月参与率
# ---------------------------------------------------------------------------

def test_monthly_rates_matches_hand_computed_rate():
    out = mt.monthly_rates(_tiny_panel())
    _assert_schema(out)
    row = out[(out["outcome"] == mt.OUTCOME_ENTRY_RATE)
              & (out["domain"] == "public")
              & (out["model"] == mt.month_label(1))
              & (out["term"] == mt.TERM_MALE)]
    assert len(row) == 1
    # 1 月男性 2 人，其中 1 人进入 -> 0.5
    assert row["estimate"].iloc[0] == pytest.approx(0.5)
    assert row["n_obs"].iloc[0] == 2
    assert 0.0 <= row["ci_low"].iloc[0] <= row["ci_high"].iloc[0] <= 1.0


def test_monthly_rates_gap_is_male_minus_female():
    out = mt.monthly_rates(_tiny_panel())
    row = out[(out["outcome"] == mt.OUTCOME_ENTRY_RATE)
              & (out["domain"] == "public")
              & (out["model"] == mt.month_label(2))
              & (out["term"] == mt.TERM_GENDER)]
    # 2 月：男性 2/2，女性 0/2 -> 差 1.0
    assert row["estimate"].iloc[0] == pytest.approx(1.0)


def test_monthly_share_mean_excludes_undefined_shares_and_counts_them():
    out = mt.monthly_rates(_tiny_panel())
    row = out[(out["outcome"] == mt.OUTCOME_SHARE_MEAN)
              & (out["domain"] == "public")
              & (out["model"] == mt.month_label(1))
              & (out["term"] == mt.TERM_FEMALE)]
    # 1 月女性两人里有一人没有表达帖（share 为 NaN），只用 0.1 这一个观测
    assert row["estimate"].iloc[0] == pytest.approx(0.1)
    assert row["n_obs"].iloc[0] == 1
    assert mt.REASON_NO_EXPRESSIVE in row["drop_reason"].iloc[0]


def test_monthly_rates_accounting_identity_holds_per_month():
    panel = _tiny_panel()
    out = mt.monthly_rates(panel)
    for month in (1, 2):
        n_input = int((panel["month"] == month).sum())
        rows = out[out["model"] == mt.month_label(month)]
        assert len(rows) > 0
        assert ((rows["n_obs"] + rows["n_dropped"]) == n_input).all()


def test_monthly_rates_covers_every_month_present_in_the_panel():
    out = mt.monthly_rates(_panel(n_users=20, months=(1, 2, 3), seed=1))
    labels = set(out["model"])
    assert labels == {mt.month_label(m) for m in (1, 2, 3)}


# ---------------------------------------------------------------------------
# §9.2 月份固定效应
# ---------------------------------------------------------------------------

def test_month_fixed_effects_recovers_a_constant_gender_gap():
    panel = _panel(n_users=1200, gap=0.15, base=0.25, month_shift=0.20, seed=7)
    out = mt.fit_month_fixed_effects(panel, "public")
    _assert_schema(out)
    row = out[(out["outcome"] == mt.OUTCOME_ENTRY)
              & (out["model"] == mt.fe_label(mt.LAYER_M0))
              & (out["term"] == mt.TERM_GENDER)]
    assert len(row) == 1
    assert row["estimate"].iloc[0] == pytest.approx(0.15, abs=0.03)
    assert np.isfinite(row["se"].iloc[0])
    assert row["ci_low"].iloc[0] < row["estimate"].iloc[0] < row["ci_high"].iloc[0]


def test_month_fixed_effects_actually_includes_the_month_dummies():
    panel = _panel(n_users=300, months=(1, 2, 3), seed=3)
    out = mt.fit_month_fixed_effects(panel, "public")
    notes = " ".join(str(n) for n in out["note"].dropna())
    assert mt.NOTE_MONTH_FE in notes


def test_month_fixed_effects_uses_user_clustered_standard_errors():
    """同一用户各月高度相关时，聚类标准误必须明显大于当作独立观测的标准误

    这条测试守的是 §11.1：用户—月份模型按用户聚类。如果 cov_type 写漏了，
    点估计一模一样、只有区间偷偷变窄，没有这条断言就看不出来。
    """
    import statsmodels.formula.api as smf

    panel = _panel(n_users=400, seed=11, persistent=True)
    out = mt.fit_month_fixed_effects(panel, "public")
    clustered = out[(out["outcome"] == mt.OUTCOME_ENTRY)
                    & (out["model"] == mt.fe_label(mt.LAYER_M0))
                    & (out["term"] == mt.TERM_GENDER)]["se"].iloc[0]

    frame = panel.copy()
    frame["male"] = (frame["gender"] == "m").astype(int)
    frame["y"] = frame["public_source_entered"].astype(float)
    naive = smf.logit("y ~ male + C(month)", data=frame).fit(
        cov_type="HC1", disp=False
    )
    ame, se_naive, _, _ = su.average_marginal_effect(naive, "male", frame)
    assert clustered > se_naive * 1.5


def test_missing_month_is_counted_as_a_drop_not_an_exception():
    """月份无法解析的用户—月是一次真实的样本流失，不是让整表崩掉的理由"""
    panel = _panel(n_users=100, months=(1, 2, 3), seed=21)
    panel.loc[panel.index[:5], "month"] = np.nan
    out = mt.fit_month_fixed_effects(panel, "public")
    entry = out[(out["outcome"] == mt.OUTCOME_ENTRY)
                & (out["model"] == mt.fe_label(mt.LAYER_M0))
                & (out["term"] == mt.TERM_GENDER)].iloc[0]
    assert entry["n_obs"] == len(panel) - 5
    assert f"{mt.REASON_MISSING_MONTH}=5" in entry["drop_reason"]
    assert entry["n_obs"] + entry["n_dropped"] == len(panel)
    # 留一月重估同样要能跑，且记账仍然自洽
    lomo = mt.leave_one_month_out(panel, "public")
    assert ((lomo["n_obs"] + lomo["n_dropped"]) == len(panel)).all()


def test_month_fixed_effects_survives_a_covariate_with_missing_values():
    """协变量含 NaN 时，聚类标准误的 groups 必须跟着 patsy 的取舍走

    此前 groups 是在拟合之前从 sample 直接取的整列
    （`sample[USER_COLUMN].to_numpy()`）。patsy 会静默丢掉任何一个协变量
    为 NaN 的行，于是设计矩阵比 groups 短，statsmodels 抛
    "The weights and list don't have the same length."，而 mc._safe_fit
    把异常吞成一行"模型失败"——整个"月份固定效应 + 留一月重估"的块会读
    成"这个假设检验过了但没估出来"，而不是"这里有个 bug"。

    这条路径今天走不到，只是因为 build_user_tables 恰好把面板计数列填满
    了；结果表的正确性不该依赖另一个模块的内部约定（这正是
    mc._reconcile_nobs 明确拒绝依赖的同一件事）。这里在 M1 的协变量
    n_posts 上打 40 个 NaN，断言模型照常估出来、n_obs 记的是真正进入拟合
    的行数、且记账仍然自洽。
    """
    panel = _panel(n_users=200, months=(1, 2, 3), gap=0.15, seed=44)
    panel.loc[panel.index[:40], "n_posts"] = np.nan
    out = mt.fit_month_fixed_effects(panel, "public")

    m1 = out[(out["outcome"] == mt.OUTCOME_ENTRY)
             & (out["model"] == mt.fe_label(mt.LAYER_M1))
             & (out["term"] == mt.TERM_GENDER)].iloc[0]
    # 关键断言：这一层必须真的估出来，而不是留下一行 NaN 说"模型失败"
    assert np.isfinite(m1["estimate"]), m1["note"]
    assert np.isfinite(m1["se"]), m1["note"]
    # n_obs 是真正进入拟合的行数（patsy 丢掉的 40 行不算），且记账自洽
    assert m1["n_obs"] == len(panel) - 40
    assert m1["n_obs"] + m1["n_dropped"] == len(panel)
    assert f"{mt.REASON_MISSING_COVARIATE}=40" in m1["drop_reason"]
    # M0 不含 n_posts，不受影响，仍然用满整张面板
    m0 = out[(out["outcome"] == mt.OUTCOME_ENTRY)
             & (out["model"] == mt.fe_label(mt.LAYER_M0))
             & (out["term"] == mt.TERM_GENDER)].iloc[0]
    assert m0["n_obs"] == len(panel)


def test_cluster_groups_align_with_the_rows_patsy_kept():
    """_cluster_groups 返回的标签必须与设计矩阵逐行对齐

    直接钉住那个函数本身：上一条测试证明"整块不再假失败"，这一条证明
    对齐方式是对的（长度等于 result.nobs，取值来自被保留下来的那些行）。
    """
    import statsmodels.formula.api as smf

    panel = _panel(n_users=120, months=(1, 2), seed=45)
    frame = mt.prepare_panel_frame(panel)
    frame.loc[frame.index[:17], "n_posts"] = np.nan
    model = smf.logit("public_source_entered ~ male + n_posts + C(month)",
                      data=frame)
    groups = mt._cluster_groups(model, frame)
    assert len(groups) == model.exog.shape[0] == len(frame) - 17
    kept = frame.loc[model.data.row_labels, mt.USER_COLUMN].to_numpy()
    assert list(groups) == list(kept)
    # 拟合本身也必须能跑通（长度对不上时 statsmodels 会直接抛 ValueError）
    fitted = model.fit(cov_type="cluster", cov_kwds={"groups": groups},
                       disp=False, maxiter=100)
    assert np.isfinite(fitted.bse).all()


def test_month_fixed_effects_reports_share_on_the_proportion_scale_only():
    """占比模型不给发生比：分数 logit 的指数化系数不是可解释的发生比

    与 models_core 的占比模型同一口径。进入模型（0/1）才同时给出 AME 与
    发生比。
    """
    panel = _panel(n_users=200, months=(1, 2, 3), seed=9)
    out = mt.fit_month_fixed_effects(panel, "public")
    share = out[out["outcome"] == mt.OUTCOME_SHARE]
    assert len(share) == len(mt.PANEL_LAYERS)
    assert set(share["term"]) == {mt.TERM_GENDER}
    assert set(share["scale"]) == {"proportion"}
    entry = out[out["outcome"] == mt.OUTCOME_ENTRY]
    assert mt.TERM_ODDS_RATIO in set(entry["term"])


def test_month_fixed_effects_leaves_a_nan_row_when_the_outcome_never_varies():
    panel = _panel(n_users=40, months=(1, 2), seed=5)
    panel["public_source_entered"] = False
    panel["public_source_count"] = 0
    out = mt.fit_month_fixed_effects(panel, "public")
    entry = out[out["outcome"] == mt.OUTCOME_ENTRY]
    assert len(entry) > 0
    assert entry["estimate"].isna().all()
    assert entry["note"].notna().all()


# ---------------------------------------------------------------------------
# §9.2 leave-one-month-out
# ---------------------------------------------------------------------------

def test_leave_one_month_out_returns_twelve_refits_plus_a_full_year_reference():
    panel = _panel(n_users=200, seed=2)
    out = mt.leave_one_month_out(panel, "public")
    _assert_schema(out)
    assert len(out) == 13, "12 次留一月 + 1 行全年参照"
    assert (out["model"] == mt.lomo_label(mt.LAYER_M0, None)).sum() == 1
    for month in range(1, 13):
        assert (out["model"] == mt.lomo_label(mt.LAYER_M0, month)).sum() == 1
    _assert_unique_keys(out)


def test_leave_one_month_out_row_really_drops_that_month():
    panel = _panel(n_users=100, seed=4)
    out = mt.leave_one_month_out(panel, "public")
    n_month_5 = int((panel["month"] == 5).sum())
    row = out[out["model"] == mt.lomo_label(mt.LAYER_M0, 5)]
    assert row["n_obs"].iloc[0] == len(panel) - n_month_5
    assert f"{mt.REASON_EXCLUDED_MONTH}=" in row["drop_reason"].iloc[0]
    full = out[out["model"] == mt.lomo_label(mt.LAYER_M0, None)]
    assert full["n_obs"].iloc[0] == len(panel)


def test_leave_one_month_out_keeps_a_row_for_a_month_absent_from_the_panel():
    panel = _panel(n_users=120, months=(1, 2, 3), seed=6)
    out = mt.leave_one_month_out(panel, "public")
    assert len(out) == 13
    missing = out[out["model"] == mt.lomo_label(mt.LAYER_M0, 12)]
    assert pd.isna(missing["estimate"].iloc[0])
    assert mt.NOTE_MONTH_ABSENT in str(missing["note"].iloc[0])


def test_leave_one_month_out_estimates_stay_close_when_no_month_drives_it():
    panel = _panel(n_users=600, gap=0.15, base=0.30, seed=8)
    out = mt.leave_one_month_out(panel, "public")
    estimates = out["estimate"].astype(float).dropna()
    assert len(estimates) == 13
    assert estimates.max() - estimates.min() < 0.05


# ---------------------------------------------------------------------------
# §10.2 延迟清理
# ---------------------------------------------------------------------------

def _cleaning_fixture():
    """一条正常记录 + 每种待清理情形各一条"""
    return _events([
        {"user_id": "u1", "gender": "m", "domain": "public",
         "source_post": "p1", "source_time": "2020-06-01 10:00", "delay_hours": 2.0},
        {"user_id": "u2", "gender": "f", "domain": "public",
         "source_post": "p2", "source_time": "2020-06-01 10:00", "delay_hours": -1.0},
        {"user_id": "u3", "gender": "f", "domain": "public",
         "source_post": "p3", "source_time": "2020-06-01 10:00", "delay_hours": 50000.0},
        {"user_id": "u1", "gender": "m", "domain": "public",
         "source_post": "p1", "source_time": "2020-06-01 10:00", "delay_hours": 30.0},
        # 跨年那条的延迟刻意取在 720 小时**之上**：400 小时低于超长延迟
        # 阈值，无论两条规则谁先判，它都只会落在 prior_year 上，于是
        # test_clean_delays_separates_prior_year_sources_from_long_delays
        # 断言的"规则顺序"根本没有被检验（交换两条规则，计数一模一样）。
        # 1000 小时同时命中两条规则，顺序才真的能被区分出来。
        {"user_id": "u4", "gender": "m", "domain": "celebrity",
         "source_post": "p4", "source_time": "2019-12-20 10:00", "delay_hours": 1000.0},
        {"user_id": "u5", "gender": "f", "domain": "celebrity",
         "source_post": "p5", "source_time": "2020-06-01 10:00", "delay_hours": None},
    ])


def test_clean_delays_removes_exactly_the_intended_rows():
    clean, report = mt.clean_delays(_cleaning_fixture(), year=2020)
    assert report["n_input"] == 6
    assert report["removed"][mt.RULE_MISSING] == 1
    assert report["removed"][mt.RULE_NON_POSITIVE] == 1
    assert report["removed"][mt.RULE_IMPLAUSIBLE] == 1
    assert report["removed"][mt.RULE_PRIOR_YEAR] == 1
    assert report["removed"][mt.RULE_DUPLICATE] == 1
    assert report["n_clean"] == 1
    assert list(clean["user_id"]) == ["u1"]


def test_clean_delays_counts_sum_to_the_number_of_removed_rows():
    _, report = mt.clean_delays(_cleaning_fixture(), year=2020)
    assert sum(report["removed"].values()) == report["n_removed"]
    assert report["n_clean"] + report["n_removed"] == report["n_input"]


def test_clean_delays_keeps_the_earliest_of_a_duplicated_pair():
    clean, report = mt.clean_delays(_cleaning_fixture(), year=2020)
    assert clean["delay_hours"].iloc[0] == pytest.approx(2.0)
    assert report["removed"][mt.RULE_DUPLICATE] == 1


def test_clean_delays_separates_prior_year_sources_from_long_delays():
    """跨年转发必须记在 prior_year 上，不能被"超长延迟"吞掉

    fixture 里那条 2019 年旧帖的延迟是 1000 小时，**同时**命中"跨年"与
    "超长延迟（>720 小时）"两条规则——这是本测试成立的前提：只有当一条
    记录同时符合两条规则时，"谁先判"才有可分辨的后果。如果两条规则的
    顺序被调换，这条记录会被记成 implausible，于是 prior_year=0、
    implausible=2，下面两条断言都会挂。

    为什么顺序必须是"先跨年、后超长"：一条 2019 年旧帖被隔了很久才转发，
    是一个完全正常的跨年转发行为（只是不属于当年扩散），不是数据异常值。
    把它记进 implausible 会同时污染两个数字——低估跨年转发的规模，
    高估异常值的规模。
    """
    _, report = mt.clean_delays(_cleaning_fixture(), year=2020)
    assert report["removed"][mt.RULE_PRIOR_YEAR] == 1
    assert report["removed"][mt.RULE_IMPLAUSIBLE] == 1
    # 前提自检：那条记录确实同时越过了超长延迟阈值，否则这个测试什么也没测
    assert 1000.0 > mt.MAX_DELAY_HOURS


def test_clean_delays_detects_millisecond_timestamps():
    frame = _events([
        {"user_id": "u1", "gender": "m", "domain": "public",
         "source_post": "p1", "source_time": "2020-06-01 10:00", "delay_hours": 2.0},
    ], unit="ms")
    clean, report = mt.clean_delays(frame, year=2020)
    assert report["timestamp_unit"] == "ms"
    assert clean["delay_hours"].iloc[0] == pytest.approx(2.0)


def test_clean_delays_unbounded_keeps_the_fifty_thousand_hour_record():
    """不设上限的口径下，超长延迟规则不触发，那条记录必须活下来

    §10.3 在 720 小时与不设上限两个口径上各跑一遍，读者据此自己判断
    阈值这条裁定有没有推动结论——前提是"不设上限"真的没有在删东西。
    """
    clean, report = mt.clean_delays(_cleaning_fixture(), year=2020,
                                    max_delay_hours=None)
    assert report["removed"][mt.RULE_IMPLAUSIBLE] == 0
    assert report["max_delay_hours"] is None
    assert report["n_clean"] == 2
    assert 50000.0 in set(clean["delay_hours"])


def test_clean_delays_reports_removals_by_gender_and_domain():
    """每条规则的删除量必须按 领域 × 性别 拆开

    清理如果在某一性别上删得更狠，§10 想做的男女比较就已经被清理本身
    污染了，而一个全局计数完全看不出这件事。fixture 里跨年那条是
    男性 × 明星域，超长那条是女性 × 公共事务域。
    """
    _, report = mt.clean_delays(_cleaning_fixture(), year=2020)
    by_cell = report["removed_by_cell"]
    assert by_cell[mt.RULE_PRIOR_YEAR] == {"celebrity": {"m": 1}}
    assert by_cell[mt.RULE_IMPLAUSIBLE] == {"public": {"f": 1}}
    assert by_cell[mt.RULE_NON_POSITIVE] == {"public": {"f": 1}}
    assert by_cell[mt.RULE_DUPLICATE] == {"public": {"m": 1}}
    # 各格的输入与幸存行数也在，流失率可以直接算
    assert report["n_input_by_cell"]["public"]["m"] == 2
    assert report["n_clean_by_cell"] == {"public": {"m": 1}}
    # 逐格计数相加必须等于该规则的总数
    for rule, cells in by_cell.items():
        total = sum(n for by_gender in cells.values() for n in by_gender.values())
        assert total == report["removed"][rule]


def test_clean_delays_report_is_json_serialisable_for_the_manifest():
    _, report = mt.clean_delays(_cleaning_fixture(), year=2020)
    json.dumps(report, ensure_ascii=False)


def test_merge_delay_reports_sums_every_rule():
    _, r1 = mt.clean_delays(_cleaning_fixture(), year=2020)
    _, r2 = mt.clean_delays(_cleaning_fixture(), year=2020)
    merged = mt.merge_delay_reports([r1, r2])
    assert merged["n_input"] == 12
    assert merged["removed"][mt.RULE_NON_POSITIVE] == 2
    assert merged["n_clean"] + merged["n_removed"] == merged["n_input"]
    # 逐格计数也要逐层相加，合并之后仍然能按 领域 × 性别 算流失率
    assert merged["removed_by_cell"][mt.RULE_PRIOR_YEAR] == {"celebrity": {"m": 2}}
    assert merged["n_input_by_cell"]["public"]["m"] == 4


def test_merge_delay_reports_refuses_to_mix_thresholds():
    """不同阈值的分片不能合并：合出来的分布不属于任何一个口径"""
    _, bounded = mt.clean_delays(_cleaning_fixture(), year=2020)
    _, unbounded = mt.clean_delays(_cleaning_fixture(), year=2020,
                                   max_delay_hours=None)
    with pytest.raises(ValueError, match="阈值不一致"):
        mt.merge_delay_reports([bounded, unbounded])


def test_dedup_user_source_pairs_works_across_shards():
    frame = mt.clean_delays(_cleaning_fixture(), year=2020)[0]
    doubled = pd.concat([frame, frame], ignore_index=True)
    deduped, removed = mt.dedup_user_source_pairs(doubled)
    assert removed == len(frame)
    assert len(deduped) == len(frame)


# ---------------------------------------------------------------------------
# §10.3 延迟分位数
# ---------------------------------------------------------------------------

def _quantile_fixture():
    """公共事务域男性的延迟恰好是 1,2,3,4,5 小时（手算分位数用）"""
    records = [(f"m{i}", "m", "public", h) for i, h in enumerate([1, 2, 3, 4, 5])]
    records += [(f"f{i}", "f", "public", h) for i, h in enumerate([2, 4, 6, 8, 10])]
    return _clean_frame(records)


def test_delay_quantiles_reproduce_hand_computed_values():
    out = mt.delay_quantiles(_quantile_fixture(), n_boot=20, seed=1)
    _assert_schema(out)
    male = out[(out["domain"] == "public")
               & (out["model"] == mt.record_label("male"))].set_index("term")
    # 1,2,3,4,5 上的线性插值分位数
    assert male.loc[mt.quantile_term(0.50), "estimate"] == pytest.approx(3.0)
    assert male.loc[mt.quantile_term(0.25), "estimate"] == pytest.approx(2.0)
    assert male.loc[mt.quantile_term(0.75), "estimate"] == pytest.approx(4.0)
    assert male.loc[mt.quantile_term(0.90), "estimate"] == pytest.approx(4.6)
    assert male.loc[mt.quantile_term(0.95), "estimate"] == pytest.approx(4.8)
    assert male.loc[mt.quantile_term(0.99), "estimate"] == pytest.approx(4.96)


def test_delay_quantiles_within_hour_shares_are_hand_checkable():
    out = mt.delay_quantiles(_quantile_fixture(), n_boot=20, seed=1)
    male = out[(out["domain"] == "public")
               & (out["model"] == mt.record_label("male"))].set_index("term")
    assert male.loc[mt.within_term(1), "estimate"] == pytest.approx(1 / 5)
    assert male.loc[mt.within_term(6), "estimate"] == pytest.approx(1.0)
    assert male.loc[mt.within_term(24), "estimate"] == pytest.approx(1.0)
    assert male.loc[mt.within_term(1), "scale"] == "probability"


def test_within_one_hour_share_is_computed_on_the_cleaned_set():
    """0.5 小时那条来自去年旧帖：算在原始集合上是 2/6，清理后是 1/5"""
    raw = _events([
        {"user_id": "u1", "gender": "m", "domain": "public",
         "source_post": "p1", "source_time": "2020-03-01 10:00", "delay_hours": 0.5},
        {"user_id": "u2", "gender": "m", "domain": "public",
         "source_post": "p2", "source_time": "2020-03-01 10:00", "delay_hours": 2.0},
        {"user_id": "u3", "gender": "m", "domain": "public",
         "source_post": "p3", "source_time": "2020-03-01 10:00", "delay_hours": 3.0},
        {"user_id": "u4", "gender": "m", "domain": "public",
         "source_post": "p4", "source_time": "2020-03-01 10:00", "delay_hours": 4.0},
        {"user_id": "u5", "gender": "m", "domain": "public",
         "source_post": "p5", "source_time": "2020-03-01 10:00", "delay_hours": 5.0},
        # 去年旧帖的快速转发：清理后不该留在分母里
        {"user_id": "u6", "gender": "m", "domain": "public",
         "source_post": "p6", "source_time": "2019-12-31 23:50", "delay_hours": 0.5},
    ])
    raw_share = float((raw["delay_seconds"] / 3600.0 <= 1).mean())
    clean, _ = mt.clean_delays(raw, year=2020)
    out = mt.delay_quantiles(clean, n_boot=20, seed=1)
    share = out[(out["domain"] == "public")
                & (out["model"] == mt.record_label("male"))
                & (out["term"] == mt.within_term(1))]["estimate"].iloc[0]
    assert share == pytest.approx(1 / 5)
    assert share != pytest.approx(raw_share)


def test_user_level_median_difference_is_not_dominated_by_one_heavy_user():
    """一个刷了 100 条慢转发的男性用户不能左右男女比较

    记录级中位数会被他一个人拉到 100 小时；用户级中位数的男女差仍然是 0。
    """
    records = [("heavy", "m", "public", 100.0)] * 100
    records += [(f"m{i}", "m", "public", 1.0) for i in range(9)]
    records += [(f"f{i}", "f", "public", 1.0) for i in range(10)]
    out = mt.delay_quantiles(_clean_frame(records), n_boot=50, seed=2)
    public = out[out["domain"] == "public"]
    record_median = public[(public["model"] == mt.record_label("male"))
                           & (public["term"] == mt.quantile_term(0.50))]["estimate"].iloc[0]
    user_gap = public[(public["model"] == mt.gap_label())
                      & (public["term"] == mt.TERM_GENDER)]["estimate"].iloc[0]
    assert record_median == pytest.approx(100.0)
    assert user_gap == pytest.approx(0.0)


def test_user_level_gap_carries_a_user_clustered_bootstrap_interval():
    records = [(f"m{i}", "m", "public", 1.0 + i * 0.5) for i in range(30)]
    records += [(f"f{i}", "f", "public", 3.0 + i * 0.5) for i in range(30)]
    out = mt.delay_quantiles(_clean_frame(records), n_boot=100, seed=3)
    gap = out[(out["domain"] == "public")
              & (out["model"] == mt.gap_label())
              & (out["term"] == mt.TERM_GENDER)]
    assert len(gap) == 1
    assert np.isfinite(gap["ci_low"].iloc[0])
    assert np.isfinite(gap["ci_high"].iloc[0])
    assert gap["ci_low"].iloc[0] <= gap["estimate"].iloc[0] <= gap["ci_high"].iloc[0]
    assert mt.NOTE_CLUSTER_BOOTSTRAP in str(gap["note"].iloc[0])


def test_record_level_bootstrap_matches_stats_utils_cluster_bootstrap():
    """记录级分位数的重抽样语义必须与 stats_utils.bootstrap_ci 完全一致

    本模块为了一次重抽样同时算出 6 个分位数 + 3 个比例，没有对每个统计量
    各调一次 bootstrap_ci，而是自己走了一遍同样的循环。这条测试钉死"同样"
    二字：同一个种子下，单个分位数的点估计与区间必须与 bootstrap_ci 逐位相同。
    """
    rng = np.random.default_rng(0)
    values = rng.lognormal(1.0, 1.0, 300)
    clusters = np.repeat([f"u{i}" for i in range(60)], 5)
    est, low, high = su.bootstrap_ci(
        values, lambda v: float(np.percentile(v, 50)), n_boot=50, seed=9,
        cluster=clusters,
    )
    table = mt.cluster_bootstrap_delay_stats(values, clusters, n_boot=50, seed=9)
    row = table[mt.quantile_term(0.50)]
    assert row["estimate"] == pytest.approx(est)
    assert row["ci_low"] == pytest.approx(low)
    assert row["ci_high"] == pytest.approx(high)


def test_mean_row_is_labelled_secondary_and_no_t_test_appears():
    out = mt.delay_quantiles(_quantile_fixture(), n_boot=20, seed=1)
    mean_rows = out[out["term"] == mt.TERM_MEAN]
    assert len(mean_rows) > 0
    assert mean_rows["note"].astype(str).str.contains(mt.NOTE_SECONDARY_MEAN).all()
    text = " ".join(
        out[col].astype(str).str.cat(sep=" ") for col in ("outcome", "term", "model", "note")
    ).lower()
    for banned in ("ttest", "t_test", "t-test", "pvalue", "p_value"):
        assert banned not in text


def test_degenerate_bootstrap_leaves_a_note_instead_of_a_silent_missing_interval():
    """某次重抽样只抽到一个性别时，区间是 NaN——这必须在 note 里留痕

    2 男 2 女的格子上，簇重抽样很容易抽出"四个都是男性"的样本，
    _median_gap 于是返回 NaN，百分位区间跟着变成 NaN，而点估计仍然有限。
    "有估计、没区间、没说明"正是模块文档第 9 条禁止的那种行。
    """
    records = [("m1", "m", "public", 1.0), ("m2", "m", "public", 2.0),
               ("f1", "f", "public", 3.0), ("f2", "f", "public", 4.0)]
    out = mt.delay_quantiles(_clean_frame(records), n_boot=50, seed=0)
    gap = out[(out["domain"] == "public")
              & (out["model"] == mt.gap_label())
              & (out["term"] == mt.TERM_GENDER)].iloc[0]
    # 这份 fixture + seed=0 会稳定命中退化情形（实测），断言因此是无条件的：
    # 写成"如果恰好命中才检查"的形式，将来退化不再发生时这条测试会静默失效
    assert np.isfinite(gap["estimate"])
    assert pd.isna(gap["ci_low"]) and pd.isna(gap["ci_high"])
    assert mt.NOTE_BOOTSTRAP_DEGENERATE in str(gap["note"])


def test_absent_domain_rows_say_so_instead_of_being_a_silent_block_of_nan():
    """清理后某个域一条记录都没有时，NaN 行必须写明"没有数据" """
    out = mt.delay_quantiles(_quantile_fixture(), n_boot=20, seed=1)
    celebrity = out[out["domain"] == "celebrity"]
    assert len(celebrity) > 0
    assert celebrity["estimate"].isna().all()
    assert celebrity["note"].astype(str).str.contains(mt.NOTE_DOMAIN_ABSENT).all()


def test_delay_quantiles_labels_carry_the_threshold_variant():
    """阈值进 model 列，两个口径的行才能并排放进同一张结果表"""
    frame = _quantile_fixture()
    bounded = mt.delay_quantiles(frame, n_boot=20, seed=1)
    unbounded = mt.delay_quantiles(frame, n_boot=20, seed=1, max_delay_hours=None)
    assert mt.record_label("male") in set(bounded["model"])
    assert mt.record_label("male", None) in set(unbounded["model"])
    both = pd.concat([bounded, unbounded], ignore_index=True)
    _assert_unique_keys(both)
    notes = " ".join(str(n) for n in unbounded["note"].dropna())
    assert "delay_threshold=unbounded" in notes


def test_delay_quantiles_keys_are_unique_and_accounting_holds():
    out = mt.delay_quantiles(_quantile_fixture(), n_boot=20, seed=1)
    _assert_unique_keys(out)
    n_public = 10
    rows = out[out["domain"] == "public"]
    assert ((rows["n_obs"] + rows["n_dropped"]) == n_public).all()


# ---------------------------------------------------------------------------
# build：落盘、manifest、结果键唯一
# ---------------------------------------------------------------------------

def _write_inputs(tmp_path, panel, events, year=2020):
    panel.to_parquet(tmp_path / f"user_month_domain_{year}.parquet",
                     engine="pyarrow", index=False)
    event_dir = tmp_path / f"retweet_domain_events_{year}"
    event_dir.mkdir(parents=True, exist_ok=True)
    for month, part in events.groupby("month"):
        part.to_parquet(event_dir / f"month={int(month):02d}.parquet",
                        engine="pyarrow", index=False)


def _build_events(n_users=40, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_users):
        gender = "m" if i % 2 == 0 else "f"
        for k in range(3):
            month = int(rng.integers(1, 13))
            records.append({
                "user_id": f"u{i}",
                "gender": gender,
                "domain": "public" if k % 2 == 0 else "celebrity",
                "source_post": f"p{i}_{k}",
                "source_time": f"2020-{month:02d}-05 09:00",
                "delay_hours": float(rng.lognormal(1.0, 1.0)),
            })
    return _events(records)


def test_build_writes_three_tables_a_manifest_and_unique_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(mt.config, "OUTPUT_DIR", str(tmp_path))
    _write_inputs(tmp_path, _panel(n_users=60, seed=12), _build_events(seed=1))

    paths = mt.build(year=2020, n_boot=20)
    assert len(paths) == 3
    for path in paths:
        out = pd.read_parquet(path, columns=list(su.RESULT_SCHEMA))
        _assert_schema(out)
        assert len(out) > 0
        _assert_unique_keys(out)

    manifest_path = tmp_path / "results" / "manifest_temporal" / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    cleaning = manifest["counts"]["delay_cleaning"]
    # 两个阈值口径各有一份自己的清理记账
    assert set(cleaning) == {mt.threshold_label(t) for t in mt.DELAY_THRESHOLDS}
    for label, report in cleaning.items():
        for rule in mt.CLEAN_RULES:
            assert rule in report["removed"]
            assert rule in report["removed_by_cell"]
    # manifest 记的是本次实际用到的口径，且各口径记的是自己的阈值
    assert manifest["params"]["delay_thresholds"] == [720.0, None]
    assert cleaning[mt.threshold_label(720.0)]["max_delay_hours"] == 720.0
    assert cleaning[mt.threshold_label(None)]["max_delay_hours"] is None


def test_build_emits_both_threshold_variants_side_by_side(tmp_path, monkeypatch):
    """两个阈值口径并列写进 delay_quantiles.parquet，阈值敏感性直接可读"""
    monkeypatch.setattr(mt.config, "OUTPUT_DIR", str(tmp_path))
    _write_inputs(tmp_path, _panel(n_users=40, months=(1, 2), seed=14),
                  _build_events(seed=3))

    mt.build(year=2020, n_boot=20)
    delay = pd.read_parquet(tmp_path / "results" / mt.DELAY_FILE,
                            columns=list(su.RESULT_SCHEMA))
    models = set(delay["model"])
    assert mt.record_label("male") in models
    assert mt.record_label("male", None) in models
    assert mt.gap_label() in models and mt.gap_label(None) in models
    _assert_unique_keys(delay)


def test_build_threshold_override_reaches_the_cleaning_and_the_manifest(
        tmp_path, monkeypatch):
    """--delay_thresholds 必须真的一路传到 clean_delays，并被 manifest 如实记录"""
    monkeypatch.setattr(mt.config, "OUTPUT_DIR", str(tmp_path))
    _write_inputs(tmp_path, _panel(n_users=30, months=(1,), seed=15),
                  _build_events(n_users=20, seed=4))

    mt.build(year=2020, n_boot=20, delay_thresholds=[1.0])
    with open(tmp_path / "results" / "manifest_temporal" / "manifest.json",
              encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["params"]["delay_thresholds"] == [1.0]
    cleaning = manifest["counts"]["delay_cleaning"]
    assert set(cleaning) == {mt.threshold_label(1.0)}
    report = cleaning[mt.threshold_label(1.0)]
    assert report["max_delay_hours"] == 1.0
    # 1 小时的阈值必然砍掉大量记录，证明参数确实生效了
    assert report["removed"][mt.RULE_IMPLAUSIBLE] > 0
    delay = pd.read_parquet(tmp_path / "results" / mt.DELAY_FILE,
                            columns=list(su.RESULT_SCHEMA))
    assert set(delay["model"]) == {
        label for gender in ("male", "female")
        for label in (mt.record_label(gender, 1.0), mt.user_label(gender, 1.0))
    } | {mt.gap_label(1.0)}


def test_normalise_thresholds_accepts_scalars_and_unbounded_spellings():
    assert mt.normalise_thresholds(None) == tuple(mt.DELAY_THRESHOLDS)
    assert mt.normalise_thresholds(720) == (720.0,)
    assert mt.normalise_thresholds("none") == (None,)
    assert mt.normalise_thresholds([168, "unbounded"]) == (168.0, None)
    with pytest.raises(ValueError):
        mt.normalise_thresholds([])


def test_build_writes_incrementally_so_a_killed_job_keeps_finished_stages(
        tmp_path, monkeypatch):
    monkeypatch.setattr(mt.config, "OUTPUT_DIR", str(tmp_path))
    _write_inputs(tmp_path, _panel(n_users=40, seed=13), _build_events(seed=2))

    def _explode(*args, **kwargs):
        raise RuntimeError("模拟延迟阶段失败")

    monkeypatch.setattr(mt, "delay_quantiles", _explode)
    with pytest.raises(RuntimeError):
        mt.build(year=2020, n_boot=20)

    monthly = tmp_path / "results" / mt.MONTHLY_FILE
    assert monthly.exists(), "§9 阶段已经算完，必须已经落盘"
    assert len(pd.read_parquet(monthly, columns=list(su.RESULT_SCHEMA))) > 0
