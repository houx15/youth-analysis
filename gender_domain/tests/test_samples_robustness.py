"""
gender_domain.robustness.samples 的单元测试（§13.2 极端值、§13.6 性别样本、
§13.9 用户类型）。

这份测试的重心按重要性排列：

1. **阈值必须是两性合并（pooled）算出来的，男女用同一个数。**
   §13.2 明确写着"不把男女分别按各自 95% 分位点截尾作为唯一版本"，
   因为不同性别用不同阈值会改变所比较的总体，现有娱乐转发均值还会因此
   方向反转。夹具因此刻意把男性造得比女性活跃，于是"合并分位点"与
   "男性自己的分位点""女性自己的分位点"三个数互不相等——只要实现偷偷
   改成逐性别算阈值，测试立刻失败。

2. **删除"总体最高 1%"同样是合并的 1%，不是各删各的 1%。**
   男性更活跃时，合并的前 1% 里男性更多，两性被删掉的人数**本来就该
   不对称**。测试钉的是这份不对称，而不是把它当成 bug 修掉。

3. **等量抽样必须抽到恰好与男性相同的人数，并把每次抽到的女性活动量
   分布记下来。** 只报估计值、不报抽样样本长什么样，读者就只能假定
   子样本有代表性。

4. **"平衡样本"必须自证平衡。** 平衡前后的标准化均值差都要写出来，
   而且**三个活动量协变量**的 SMD 必须真的变小；一个不自证的"balanced"
   标签比不做平衡更糟。测试刻意**不**要求 verified_flag 也变小：认证账号
   在真实数据与夹具里都极少，匹配后它的 SMD 反而会变大（夹具上
   0.130 -> 0.354），而这正是平衡表要暴露、而不是掩盖的东西——标签宣称的
   是活动量平衡，认证状态的失衡必须留在表上让读者看见。

5. **做不到的变体只留一行注明原因的 NaN 行，绝不留一组估计值。**
   §13.2 的 log(1+x) 对照在本流水线里做不出来（协变量变换写死在
   models_core 里）。早先的做法是复制 untreated 的结果挂上另一个标签，
   note 里写"按构造相同"——但下游 synthesis 的方向一致率是按**变体标签**
   数的，一个坐在正中心的复制标签会同时抬高一致率、压低离散度。测试因此
   钉死这一行是 NaN、且不带任何模型层。

6. **性别字段冲突的用户必须真的被冲突变体剔除、被基线保留**；
   demographic_gender 不存在时必须留一行注明原因，**绝不编一个"一致"
   出来**。
"""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import config
from gender_domain import models_core as mc
from gender_domain.robustness import harness
from gender_domain.robustness import samples as smp


YEAR = 2020
N_MALE = 40
N_FEMALE = 80

# 冲突用户：gender 与 demographic_gender 不一致，必须被冲突变体剔除
CONFLICT_USERS = ("m000", "f000")
# 机构认证账号（verified_type = 3，落在 INSTITUTIONAL_VERIFIED_TYPES 里）
INSTITUTIONAL_USERS = ("m001", "f001")
# 认证个人（verified_type = 0）
VERIFIED_INDIVIDUAL_USERS = ("m002", "f002")


def _user_rows():
    """造一份表 C 形状的用户表

    四件事是**故意**的：
      1) 女性人数（80）是男性（40）的两倍——等量抽样才有意义；
      2) 男性平均活跃度明显高于女性，长尾也全是男性——合并阈值与逐性别
         阈值因此不相等，合并删顶端时被删的男性必然多于女性；
      3) 两性的活动量**大幅重叠**（男性 60 起步、女性最高 246），因为真实
         数据里两性的活跃度分布是重叠的，不是分开的两团：完全分离的夹具
         会让倾向得分模型退化成一个几乎完美的分类器，那测的就不是本模块
         在真实数据上的行为了；
      4) 发帖量与转发量之间加了一点错相位的抖动（i % 3、i % 7），避免两列
         在组内成为严格的线性函数——那会让设计矩阵奇异，同样是夹具造出来
         的、真实数据里没有的问题。
    """
    rows = []
    for index in range(N_MALE):
        rows.append({
            "user_id": "m{:03d}".format(index),
            "gender": "m",
            # 男性：发帖量从 60 起步、步长 8，长尾（最高 372）全在这一侧
            "n_posts": 60 + 8 * index + (index % 3),
            "n_retweets": 20 + 3 * index + (index % 7),
            "n_active_days": 5 + (index % 20),
            "n_active_months": 1 + (index % 12),
        })
    for index in range(N_FEMALE):
        rows.append({
            "user_id": "f{:03d}".format(index),
            "gender": "f",
            # 女性：5 起步、步长 3，最高 246——与男性 60-246 那一段大量重叠
            "n_posts": 5 + 3 * index + (index % 5),
            "n_retweets": 4 + index + (index % 3),
            "n_active_days": 3 + (index % 15),
            "n_active_months": 1 + (index % 10),
        })
    return rows


def _user_table():
    frame = pd.DataFrame(_user_rows())
    frame["n_expressive_posts"] = (frame["n_posts"] * 0.6).astype(int) + 1
    frame["n_active_months_panel"] = frame["n_active_months"]
    frame["region"] = ["East" if i % 3 else "West" for i in range(len(frame))]
    frame["province"] = ["11" if i % 3 else "51" for i in range(len(frame))]

    male = (frame["gender"] == "m").to_numpy()
    for domain, base in (("public", 0), ("celebrity", 1)):
        entered = ((np.arange(len(frame)) + base) % 3 != 0) | male
        frame["{}_source_entered".format(domain)] = entered
        frame["{}_source_count".format(domain)] = np.where(
            entered, 1 + (np.arange(len(frame)) % 5), 0
        )
        topical = (np.arange(len(frame)) + base) % 7
        frame["{}_topical_posts".format(domain)] = topical
        frame["{}_topical_share".format(domain)] = (
            topical / frame["n_expressive_posts"].to_numpy()
        )
    return frame


def _profile_table(user_table, with_demographic_gender=True):
    """在表 C 上补 profile_join 产出的画像列（含 demographic_gender）"""
    frame = user_table.copy()
    verified_type = []
    for user_id in frame["user_id"]:
        if user_id in INSTITUTIONAL_USERS:
            verified_type.append(3.0)          # 机构认证
        elif user_id in VERIFIED_INDIVIDUAL_USERS:
            verified_type.append(0.0)          # 认证个人
        else:
            verified_type.append(-1.0)         # 未认证
    frame["verified_type"] = verified_type
    frame["user_type"] = "normal"
    frame["fans_number"] = 100.0
    frame["friends_count"] = 50.0
    frame["log_fans"] = np.log1p(frame["fans_number"])
    frame["log_friends"] = np.log1p(frame["friends_count"])
    frame["verified_flag"] = [v != -1.0 for v in verified_type]
    frame["profile_complete"] = True
    if with_demographic_gender:
        demographic = []
        for user_id, gender in zip(frame["user_id"], frame["gender"]):
            if user_id in CONFLICT_USERS:
                demographic.append("f" if gender == "m" else "m")
            elif user_id.endswith("5"):
                demographic.append(None)       # 字段缺失，不是冲突
            else:
                demographic.append(gender)
        frame["demographic_gender"] = demographic
    return frame


@pytest.fixture()
def synthetic_project(tmp_path, monkeypatch):
    """写出表 C 与带画像的表 C，并把 config.OUTPUT_DIR 指过去"""
    out_dir = str(tmp_path / "analysis_data")
    os.makedirs(out_dir, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)

    user_table = _user_table()
    user_table.to_parquet(
        os.path.join(out_dir, "user_domain_{}.parquet".format(YEAR)),
        engine="pyarrow", index=False,
    )
    _profile_table(user_table).to_parquet(
        os.path.join(out_dir, "user_domain_{}_with_profile.parquet".format(YEAR)),
        engine="pyarrow", index=False,
    )
    return out_dir


@pytest.fixture()
def context(synthetic_project):
    return smp.build_context(YEAR)


def _read_diagnostics(path):
    return pd.read_parquet(path, engine="pyarrow", columns=list(smp.DIAGNOSTIC_SCHEMA))


def _read_results(path):
    return pd.read_parquet(path, engine="pyarrow",
                           columns=list(harness.ROBUSTNESS_SCHEMA))


# ---------------------------------------------------------------------------
# 上下文：读到的是哪张表、画像列在不在
# ---------------------------------------------------------------------------

def test_context_prefers_the_profile_table_because_13_9_needs_it(context):
    assert context.has_profile is True
    assert context.has_demographic_gender is True
    assert "verified_flag" in context.user_table.columns
    assert len(context.user_table) == N_MALE + N_FEMALE


def test_context_falls_back_to_plain_table_c_and_says_so(synthetic_project, capsys):
    os.remove(os.path.join(
        synthetic_project, "user_domain_{}_with_profile.parquet".format(YEAR)))
    ctx = smp.build_context(YEAR)
    printed = capsys.readouterr().out
    assert ctx.has_profile is False
    assert ctx.has_demographic_gender is False
    assert "user_domain_{}.parquet".format(YEAR) in printed


# ---------------------------------------------------------------------------
# §13.2 最要紧的一条：阈值一律 pooled，男女必须是同一个数
# ---------------------------------------------------------------------------

def test_cut_points_are_pooled_and_numerically_identical_for_both_genders(context):
    frame = context.user_table
    cuts = smp.applied_cut_points(frame, "n_posts", 0.95)
    assert cuts["m"] == cuts["f"] == cuts["pooled"]

    pooled = float(np.quantile(frame["n_posts"].to_numpy(dtype=float), 0.95))
    assert cuts["pooled"] == pytest.approx(pooled)

    # 夹具刻意让男性更活跃：逐性别阈值与合并阈值三个数互不相等，
    # 一旦实现改成逐性别算，上面两条断言必然失败
    male_only = float(np.quantile(
        frame.loc[frame["gender"] == "m", "n_posts"].to_numpy(dtype=float), 0.95))
    female_only = float(np.quantile(
        frame.loc[frame["gender"] == "f", "n_posts"].to_numpy(dtype=float), 0.95))
    assert male_only != pytest.approx(pooled)
    assert female_only != pytest.approx(pooled)


def test_winsorisation_clips_both_genders_at_the_same_pooled_value(context):
    frame, thresholds = smp.winsorise_pooled(context.user_table, 0.95)
    cut = thresholds["n_posts"]
    assert frame["n_posts"].max() <= cut + 1e-9
    male_max = frame.loc[frame["gender"] == "m", "n_posts"].max()
    female_max = frame.loc[frame["gender"] == "f", "n_posts"].max()
    # 男性顶端被压到合并阈值；女性顶端本来就在阈值之下，因此**不该**被压
    assert male_max == pytest.approx(cut)
    assert female_max < cut
    # 截尾只改数值，不改样本
    assert len(frame) == len(context.user_table)


def test_winsorisation_covers_every_activity_column(context):
    frame, thresholds = smp.winsorise_pooled(context.user_table, 0.99)
    assert set(thresholds) == set(smp.ACTIVITY_COLUMNS)
    for column, cut in thresholds.items():
        assert frame[column].max() <= cut + 1e-9


# ---------------------------------------------------------------------------
# §13.2 删顶端：合并的 1%，删掉的男女人数本来就不对称
# ---------------------------------------------------------------------------

def test_trimming_the_top_one_percent_removes_exactly_the_expected_users(context):
    frame = context.user_table
    activity = smp.total_activity(frame).to_numpy(dtype=float)
    cut = float(np.quantile(activity, 0.99))
    expected = set(frame.loc[activity > cut, "user_id"])
    assert expected  # 夹具必须真的有人被删，否则这条测试什么也没测到

    trimmed, threshold, dropped = smp.trim_top_users(frame, 0.01)
    assert threshold == pytest.approx(cut)
    assert set(frame.loc[dropped.to_numpy(dtype=bool), "user_id"]) == expected
    assert len(trimmed) == len(frame) - len(expected)
    assert not set(trimmed["user_id"]) & expected


def test_pooled_trimming_removes_more_men_than_women_when_men_are_more_active(context):
    """这份不对称是结论本身，不是要被"修正"掉的 bug"""
    frame = context.user_table
    trimmed, _, dropped = smp.trim_top_users(frame, 0.05)
    dropped_gender = frame.loc[dropped.to_numpy(dtype=bool), "gender"]
    n_male = int((dropped_gender == "m").sum())
    n_female = int((dropped_gender == "f").sum())
    assert n_male > n_female

    # 对照：如果按逐性别的 95% 分位点各删各的，两性会被删掉几乎相同的**比例**，
    # 女性也会被削掉一批——这正是 §13.2 拒绝的那种做法。合并阈值下女性一个
    # 都没被删，两种做法比较的根本不是同一个总体。
    per_gender = {}
    for gender, group in frame.groupby("gender"):
        gender_activity = smp.total_activity(group).to_numpy(dtype=float)
        gender_cut = float(np.quantile(gender_activity, 0.95))
        per_gender[gender] = int((gender_activity > gender_cut).sum())
    assert per_gender["f"] > 0
    assert n_female == 0
    assert per_gender["m"] < n_male


def test_extreme_value_variants_cover_every_treatment_and_record_the_cut_points(
    context, tmp_path
):
    out_path = str(tmp_path / "samples.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    smp.run_extreme_value_variants(YEAR, context=context, out_path=out_path,
                                   diag_path=diag_path)

    results = _read_results(out_path)
    labels = set(results["variant_label"])
    assert "untreated" in labels
    assert smp.LOG1P_LABEL in labels
    assert "winsorised_pooled_p95" in labels
    assert "winsorised_pooled_p99" in labels
    assert "trim_pooled_top1pct" in labels
    assert "trim_pooled_top5pct" in labels
    # 每个真的估出来的变体恰好六个量 × 两层
    for label in labels - {smp.LOG1P_LABEL}:
        assert len(results[results["variant_label"] == label]) == (
            len(harness.QUANTITIES) * 2
        )
    assert set(results["variant_family"]) == {smp.VARIANT_FAMILY_EXTREME}

    diag = _read_diagnostics(diag_path)
    winsor = diag[diag["variant_label"] == "winsorised_pooled_p95"]
    assert len(winsor) == len(smp.ACTIVITY_COLUMNS)
    # 男女阈值必须逐行完全相等——这是本任务最重要的一条性质
    assert (winsor["threshold_male"] == winsor["threshold_female"]).all()
    assert (winsor["threshold_male"] == winsor["threshold_pooled"]).all()
    assert (winsor["threshold_quantile"] == 0.95).all()


def test_extreme_value_diagnostics_record_who_was_dropped_by_gender(context, tmp_path):
    diag_path = str(tmp_path / "diag.parquet")
    smp.run_extreme_value_variants(YEAR, context=context, out_path=None,
                                   diag_path=diag_path)
    diag = _read_diagnostics(diag_path)
    row = diag[diag["variant_label"] == "trim_pooled_top5pct"].iloc[0]
    assert row["threshold_variable"] == "total_activity"
    assert row["threshold_male"] == row["threshold_female"]
    assert int(row["n_dropped_male"]) > int(row["n_dropped_female"])
    assert int(row["n_users_after"]) == (
        int(row["n_users_before"]) - int(row["n_dropped_male"])
        - int(row["n_dropped_female"])
    )


def test_the_log1p_comparison_is_a_note_only_row_not_a_duplicated_estimate(
    context, tmp_path
):
    """做不到的对照绝不能留下一组估计值

    §13.2 的 log(1+x) 对照在本流水线里做不出来（协变量变换写死在
    models_core 里）。如果把 untreated 的结果复制一份挂上另一个标签，
    下游 synthesis 的方向一致率（按变体标签数）会凭空多出一票坐在正中心的
    赞成，离散度也被压低——note 里写"按构造相同"救不了这件事，因为算术读
    不到 note。这一行必须是 NaN。
    """
    out_path = str(tmp_path / "samples.parquet")
    smp.run_extreme_value_variants(YEAR, context=context, out_path=out_path,
                                   diag_path=str(tmp_path / "diag.parquet"))
    results = _read_results(out_path)
    log_rows = results[results["variant_label"] == smp.LOG1P_LABEL]
    assert len(log_rows) == 1
    assert log_rows["estimate"].isna().all()
    assert log_rows["model"].isna().all()          # 不属于任何模型层
    assert "not_estimable" in log_rows["note"].iloc[0]
    assert "MODEL_LAYERS" in log_rows["note"].iloc[0]

    # 而 untreated 的估计值一个不少，两者不是同一批行
    untreated = results[results["variant_label"] == "untreated"]
    assert len(untreated) == len(harness.QUANTITIES) * 2


def test_winsorised_note_carries_every_activity_column_cut(context, tmp_path):
    """截尾一次截四列，四个截点都必须出现在结果行的 note 里

    只写 n_posts 的截点会让另外三列的干预只存在于诊断表里，拿到
    samples.parquet 一张表的读者看不出这个变体到底截了什么。
    """
    out_path = str(tmp_path / "samples.parquet")
    smp.run_extreme_value_variants(YEAR, context=context, out_path=out_path,
                                   diag_path=str(tmp_path / "diag.parquet"))
    results = _read_results(out_path)
    note = results[results["variant_label"] == "winsorised_pooled_p95"]["note"].iloc[0]
    for column in smp.ACTIVITY_COLUMNS:
        assert "{}_cut=".format(column) in note


def test_trimming_notes_reach_the_result_rows(context, tmp_path):
    """只拿到 samples.parquet 一张表的读者也必须看得见这次删了多少人"""
    out_path = str(tmp_path / "samples.parquet")
    smp.run_extreme_value_variants(YEAR, context=context, out_path=out_path,
                                   diag_path=str(tmp_path / "diag.parquet"))
    results = _read_results(out_path)
    note = results[results["variant_label"] == "trim_pooled_top1pct"]["note"].iloc[0]
    assert "pooled" in note
    assert "n_dropped" in note


# ---------------------------------------------------------------------------
# §13.6 等量抽样
# ---------------------------------------------------------------------------

def test_equal_size_draws_exactly_the_male_count_of_women(context):
    drawn = smp.equal_size_draw(context.user_table, seed=7)
    assert int((drawn["gender"] == "m").sum()) == N_MALE
    assert int((drawn["gender"] == "f").sum()) == N_MALE
    # 男性一个不少地全留着
    assert set(drawn.loc[drawn["gender"] == "m", "user_id"]) == set(
        context.user_table.loc[context.user_table["gender"] == "m", "user_id"]
    )


def test_equal_size_draw_is_reproducible_and_differs_across_seeds(context):
    first = smp.equal_size_draw(context.user_table, seed=7)
    again = smp.equal_size_draw(context.user_table, seed=7)
    other = smp.equal_size_draw(context.user_table, seed=8)
    assert list(first["user_id"]) == list(again["user_id"])
    assert list(first["user_id"]) != list(other["user_id"])


def test_equal_size_records_the_activity_distribution_of_every_draw(context, tmp_path):
    out_path = str(tmp_path / "samples.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    smp.run_equal_size_gender(YEAR, n_replicates=3, seed=0, context=context,
                              out_path=out_path, diag_path=diag_path)

    diag = _read_diagnostics(diag_path)
    draws = diag[diag["variant_family"] == smp.VARIANT_FAMILY_EQUAL_SIZE]
    draws = draws[draws["variant_label"] != "full_sample_reference"]
    assert len(draws) == 3
    for column in ("mean_posts_female", "median_posts_female", "p90_posts_female",
                   "mean_retweets_female", "mean_active_days_female"):
        assert draws[column].notna().all()
    # 抽样分布必须真的在动，否则记它没有意义
    assert draws["mean_posts_female"].nunique() > 1
    # 全样本参照行必须在，读者才能判断子样本有没有代表性
    reference = diag[diag["variant_label"] == "full_sample_reference"]
    assert len(reference) == 1
    assert int(reference.iloc[0]["n_female_before"]) == N_FEMALE

    results = _read_results(out_path)
    assert set(results["replicate"]) == {0, 1, 2}
    assert results["seed"].notna().all()


# ---------------------------------------------------------------------------
# §13.6 活动量平衡：方法写在标签里，平衡前后都要报
# ---------------------------------------------------------------------------

def test_balanced_variant_states_its_method_in_the_label(context, tmp_path):
    out_path = str(tmp_path / "samples.parquet")
    smp.run_activity_balanced(YEAR, context=context, out_path=out_path,
                              diag_path=str(tmp_path / "diag.parquet"),
                              bal_path=str(tmp_path / "balance.parquet"))
    results = _read_results(out_path)
    label = results["variant_label"].iloc[0]
    assert "match" in label
    assert "posts" in label and "retweets" in label and "days" in label


def test_balance_is_reported_before_and_after_and_actually_improves(context, tmp_path):
    balance_path = str(tmp_path / "balance.parquet")
    smp.run_activity_balanced(YEAR, context=context, out_path=None,
                              diag_path=str(tmp_path / "diag.parquet"),
                              bal_path=balance_path)
    balance = pd.read_parquet(balance_path, engine="pyarrow",
                              columns=list(smp.BALANCE_SCHEMA))
    assert set(balance["stage"]) == {"before", "after"}
    for covariate in ("log_posts", "log_retweets", "n_active_days"):
        before = balance[(balance["covariate"] == covariate)
                         & (balance["stage"] == "before")].iloc[0]
        after = balance[(balance["covariate"] == covariate)
                        & (balance["stage"] == "after")].iloc[0]
        assert abs(after["std_mean_diff"]) < abs(before["std_mean_diff"])


def test_balanced_sample_has_the_same_number_of_men_and_women(context, tmp_path):
    diag_path = str(tmp_path / "diag.parquet")
    smp.run_activity_balanced(YEAR, context=context, out_path=None,
                              diag_path=diag_path,
                              bal_path=str(tmp_path / "balance.parquet"))
    diag = _read_diagnostics(diag_path)
    row = diag[diag["variant_family"] == smp.VARIANT_FAMILY_BALANCE].iloc[0]
    assert int(row["n_male_after"]) == int(row["n_female_after"])
    assert int(row["n_male_after"]) > 0


# ---------------------------------------------------------------------------
# §13.9 性别字段与用户类型
# ---------------------------------------------------------------------------

def test_a_conflicting_gender_user_is_excluded_by_the_conflict_variant(context):
    frame = context.user_table
    agree = smp.gender_fields_agree_mask(frame)
    conflict = smp.gender_conflict_mask(frame)
    for user_id in CONFLICT_USERS:
        index = frame.index[frame["user_id"] == user_id][0]
        assert bool(conflict.loc[index]) is True
        assert bool(agree.loc[index]) is False
    # 字段缺失不算冲突，也不算一致——不能把"不知道"算成"一致"
    missing = frame.index[frame["user_id"] == "m005"][0]
    assert bool(conflict.loc[missing]) is False
    assert bool(agree.loc[missing]) is False


def test_the_baseline_keeps_the_conflicting_user_that_the_variant_drops(
    context, tmp_path
):
    out_path = str(tmp_path / "samples.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    smp.run_gender_field_variants(YEAR, context=context, out_path=out_path,
                                  diag_path=diag_path)
    diag = _read_diagnostics(diag_path)
    row = diag[diag["variant_label"] == "exclude_gender_conflict"].iloc[0]
    # 基线里那两个人还在，变体里被剔掉了
    assert int(row["n_users_before"]) == len(context.user_table)
    assert int(row["n_users_after"]) == len(context.user_table) - len(CONFLICT_USERS)
    assert int(row["n_dropped_male"]) == 1
    assert int(row["n_dropped_female"]) == 1


def test_gender_field_variants_emit_a_noted_row_when_the_field_is_missing(
    synthetic_project, tmp_path
):
    """demographic_gender 拿不到时必须留一行说明，绝不编一个"一致"出来"""
    path = os.path.join(synthetic_project,
                        "user_domain_{}_with_profile.parquet".format(YEAR))
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.drop(columns=["demographic_gender"]).to_parquet(
        path, engine="pyarrow", index=False)

    ctx = smp.build_context(YEAR)
    assert ctx.has_demographic_gender is False
    out_path = str(tmp_path / "samples.parquet")
    smp.run_gender_field_variants(YEAR, context=ctx, out_path=out_path,
                                  diag_path=str(tmp_path / "diag.parquet"))
    results = _read_results(out_path)
    unavailable = results[results["variant_label"].str.contains("unavailable")]
    assert len(unavailable) >= 1
    assert unavailable["estimate"].isna().all()
    assert unavailable["note"].str.contains("demographic_gender").all()
    # 其它不依赖该字段的变体照常跑
    assert "exclude_institutional_accounts" in set(results["variant_label"])


def test_institutional_accounts_are_excluded_by_the_stated_verified_type_rule(context):
    frame = context.user_table
    mask = smp.institutional_mask(frame)
    flagged = set(frame.loc[mask.to_numpy(dtype=bool), "user_id"])
    assert flagged == set(INSTITUTIONAL_USERS)
    # 认证个人不是机构
    assert not flagged & set(VERIFIED_INDIVIDUAL_USERS)


def test_verified_individuals_and_ordinary_users_are_two_disjoint_variants(context):
    frame = context.user_table
    verified = smp.verified_individual_mask(frame).to_numpy(dtype=bool)
    ordinary = smp.ordinary_user_mask(frame).to_numpy(dtype=bool)
    assert set(frame.loc[verified, "user_id"]) == set(VERIFIED_INDIVIDUAL_USERS)
    assert not (verified & ordinary).any()
    assert not set(frame.loc[ordinary, "user_id"]) & set(INSTITUTIONAL_USERS)


def test_the_automation_rule_is_a_pooled_percentile_and_is_recorded(context):
    frame = context.user_table
    mask, thresholds = smp.automation_mask(frame, quantile=0.95)
    activity = smp.total_activity(frame).to_numpy(dtype=float)
    pooled = float(np.quantile(activity, 0.95))
    assert thresholds["total_activity"] == pytest.approx(pooled)
    # 阈值来自合并样本，男女共用；标记出来的用户必须真的在阈值之上
    flagged = frame.loc[mask.to_numpy(dtype=bool)]
    assert len(flagged) > 0
    assert set(thresholds) == {"total_activity", "posts_per_active_day"}
    assert smp.AUTOMATION_RULE  # 规则必须以字符串形式写死在模块里


def test_user_type_family_also_estimates_m2_because_it_is_about_profile_data(
    context, tmp_path
):
    out_path = str(tmp_path / "samples.parquet")
    smp.run_gender_field_variants(YEAR, context=context, out_path=out_path,
                                  diag_path=str(tmp_path / "diag.parquet"))
    results = _read_results(out_path)
    estimated = results[~results["variant_label"].str.contains("unavailable")]
    assert set(estimated["model"]) == {"M0", "M1", "M2"}
    for label in set(estimated["variant_label"]):
        assert len(estimated[estimated["variant_label"] == label]) == (
            len(harness.QUANTITIES) * 3
        )


# ---------------------------------------------------------------------------
# build()：只写 robustness 目录、manifest 记下全部规则与种子
# ---------------------------------------------------------------------------

def test_build_writes_everything_under_the_robustness_directory(synthetic_project):
    paths = smp.build(YEAR, n_replicates=2, seed=0)
    robustness = os.path.join(synthetic_project, "robustness")
    for key in ("results_path", "diagnostics_path", "balance_path"):
        assert os.path.exists(paths[key])
        assert os.path.commonpath([robustness, os.path.abspath(paths[key])]) == \
            os.path.abspath(robustness)

    manifest_path = os.path.join(robustness, "samples_{}".format(YEAR), "manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    params = manifest["params"]
    assert params["n_replicates"] == 2
    assert len(params["seeds"]) == 2
    assert params["threshold_scope"] == "pooled_across_genders"
    assert params["automation_rule"] == smp.AUTOMATION_RULE
    assert params["balance_method"]
    assert "demographic_gender_available" in params


def test_manifest_records_how_much_of_demographic_gender_is_usable(synthetic_project):
    """光有一个布尔值不够：认不出来的取值占多少必须能直接读到

    夹具里每 10 个用户有 1 个的 demographic_gender 是空（user_id 以 5 结尾），
    可映射比例因此是 0.9——读者不该靠减法去推这个数。
    """
    smp.build(YEAR, n_replicates=2, seed=0)
    manifest_path = os.path.join(synthetic_project, "robustness",
                                 "samples_{}".format(YEAR), "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        params = json.load(f)["params"]
    n_total = N_MALE + N_FEMALE
    n_unmappable = sum(1 for i in range(N_MALE) if str(i).endswith("5"))
    n_unmappable += sum(1 for i in range(N_FEMALE) if str(i).endswith("5"))
    assert params["demographic_gender_unmappable_users"] == n_unmappable
    assert params["demographic_gender_mappable_users"] == n_total - n_unmappable
    assert params["demographic_gender_mappable_share"] == pytest.approx(
        (n_total - n_unmappable) / n_total)


def test_manifest_dropped_share_is_scoped_to_the_extreme_value_family(
    synthetic_project
):
    """删顶端的不对称只能在 §13.2 这一族里看

    §13.9 的 verified_individuals_only 按定义会剔掉 97% 以上的用户；把最大值
    取遍全部 family，这个数字就会来自那里，而名字说的是截尾。
    """
    paths = smp.build(YEAR, n_replicates=2, seed=0)
    manifest_path = os.path.join(synthetic_project, "robustness",
                                 "samples_{}".format(YEAR), "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        counts = json.load(f)["counts"]
    diag = _read_diagnostics(paths["diagnostics_path"])
    extreme = diag[diag["variant_family"] == smp.VARIANT_FAMILY_EXTREME]
    assert counts["extreme_values_max_dropped_share_male"] == pytest.approx(
        float(extreme["dropped_share_male"].max()))
    # 全族最大值来自 §13.9 的子样本变体，与截尾无关——两者必须不同，
    # 否则这条测试无法证明作用域真的收窄了
    assert float(diag["dropped_share_male"].max()) > float(
        extreme["dropped_share_male"].max())


def test_cli_subcommands_default_to_the_module_output_paths(synthetic_project):
    """从命令行单跑一个子命令不能算完一整轮却一个字节都不写"""
    runner = smp._cli(smp.run_extreme_value_variants)
    runner(YEAR)
    assert os.path.exists(smp.results_path())
    assert os.path.exists(smp.diagnostics_path())


def test_build_covers_all_three_subsections(synthetic_project):
    paths = smp.build(YEAR, n_replicates=2, seed=0)
    results = _read_results(paths["results_path"])
    assert set(results["variant_family"]) == {
        smp.VARIANT_FAMILY_EXTREME,
        smp.VARIANT_FAMILY_EQUAL_SIZE,
        smp.VARIANT_FAMILY_BALANCE,
        smp.VARIANT_FAMILY_USER_TYPE,
    }


# ---------------------------------------------------------------------------
# 复用而不是重写：活动量派生列一律走 models_core
# ---------------------------------------------------------------------------

def test_balance_covariates_use_models_core_log_activity(context):
    frame = mc.add_log_activity(context.user_table)
    stats = smp.balance_statistics(frame, "log_posts")
    male = frame.loc[frame["gender"] == "m", "log_posts"]
    female = frame.loc[frame["gender"] == "f", "log_posts"]
    assert stats["mean_male"] == pytest.approx(float(male.mean()))
    assert stats["mean_female"] == pytest.approx(float(female.mean()))
    pooled_sd = math.sqrt((float(male.var(ddof=1)) + float(female.var(ddof=1))) / 2)
    assert stats["std_mean_diff"] == pytest.approx(
        (float(male.mean()) - float(female.mean())) / pooled_sd
    )
