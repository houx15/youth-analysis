"""
gender_domain.profile_join 的单元测试（表 C 画像控制变量拼接）。

这个模块产出的两样东西直接支撑论文的缺失数据小节（研究协议 §11.4）：
`profile_complete` 决定谁进得了 M2，loss report 是"因收窄而损失了多少
样本、按性别是否有差异"这句话的唯一出处。因此测试盯的是四件事：

1. **左连接绝不丢用户。** M0/M1 依赖完整样本，画像表里找不到的用户必须
   原样留下、标成 profile_complete=False，而不是从表里消失。

2. **profile_complete 是"三项全齐"，不是"某一项没炸"。** 三项 M2 控制
   变量（verified_flag / log_fans / log_friends）任意一项缺失都必须让
   这一行进不了 M2——此前只有 friends_count 为负这一条路径被覆盖，
   于是"只检查了其中一项"这种实现完全测不出来。这里逐项各造一个用户。

3. **缺失（NaN）与异常（负数）是两种不同的输入，但结论相同。**
   两者都必须变成 NaN 而不是 0：填 0 会把"未知"伪装成"该用户没有粉丝"，
   而 log1p(0)=0 是一个会被当成真实观测参与回归的合法值。

4. **loss report 必须自洽。** users_total 是分性别之和；每一格
   m2_ready <= profile_matched <= users_total。这条恒等式是论文里那句
   "M2 损失了 N 个用户"能不能被读者核对的前提，此前一条断言都没有。
"""

import json

import numpy as np
import pandas as pd
import pytest

from gender_domain import config
from gender_domain import profile_join as pj


def _users():
    return pd.DataFrame({
        "user_id": ["1", "2", "3"],
        "gender": ["m", "f", "f"],
        "n_posts": [10, 20, 30],
    })


def _profiles():
    return pd.DataFrame({
        "user_id": [1, 2],                      # 注意：整数，且缺少用户 3
        "verified_type": ["0", "1"],
        "user_type": ["normal", "verified"],
        "fans_number": [100, 0],
        "friends_count": [50, -1],              # -1 为异常值
    })


# ---------------------------------------------------------------------------
# 左连接：以表 C 为准，一个用户都不能丢
# ---------------------------------------------------------------------------

def test_attach_keeps_every_user_even_without_profile():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    assert len(out) == 3
    assert set(out["user_id"]) == {"1", "2", "3"}


def test_missing_profile_marked_not_dropped():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    row = out.set_index("user_id").loc["3"]
    assert not row["profile_complete"]
    assert pd.isna(row["fans_number"])


def test_attach_keeps_original_table_c_columns():
    """拼接不能顺手改掉表 C 自己的列——下游 M0/M1 还在用它们"""
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    assert list(out.sort_values("user_id")["n_posts"]) == [10, 20, 30]
    assert list(out.sort_values("user_id")["gender"]) == ["m", "f", "f"]


def test_duplicate_profile_rows_do_not_inflate_the_user_table():
    """画像表里同一 user_id 出现两次时，表 C 的行数必须纹丝不动

    行数一旦膨胀，后面每一个 n_obs 都会多算，而这种膨胀在结果表上看起来
    只是"样本量比预期大一点"，没有任何报错。
    """
    profiles = pd.concat([_profiles(), _profiles()], ignore_index=True)
    out, report = pj.attach_profile_controls(_users(), profiles)
    assert len(out) == 3
    assert report["users_total"] == 3


def test_integer_profile_ids_join_to_string_user_ids():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    assert out.set_index("user_id").loc["1", "user_type"] == "normal"


def test_missing_profile_column_is_tolerated_not_fatal():
    """画像表少给一列时补全缺失列，而不是 KeyError 掉整个拼接

    这条流水线本来就不保证画像表每个字段都在；本函数的职责就是容忍
    这种不完整，并让受影响的用户在 profile_complete 上如实变成 False。
    """
    profiles = _profiles().drop(columns=["fans_number"])
    out, report = pj.attach_profile_controls(_users(), profiles)
    assert len(out) == 3
    assert out["fans_number"].isna().all()
    assert not out["profile_complete"].any()
    assert report["m2_ready"] == 0


# ---------------------------------------------------------------------------
# 数值清洗：NaN 与负数都不能变成 0
# ---------------------------------------------------------------------------

def test_log_transforms_use_log1p():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    row = out.set_index("user_id").loc["1"]
    assert row["log_fans"] == pytest.approx(np.log1p(100))
    zero = out.set_index("user_id").loc["2"]
    assert zero["log_fans"] == pytest.approx(0.0)      # log1p(0) == 0，合法值


def test_negative_counts_become_nan_not_zero():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    row = out.set_index("user_id").loc["2"]
    assert pd.isna(row["friends_count"])
    assert pd.isna(row["log_friends"])
    assert not row["profile_complete"]


def test_nan_and_negative_inputs_are_distinguished_at_the_input_but_not_the_output():
    """空值与负数是两种不同的输入，清洗后都必须是 NaN——但都不能是 0

    这两条路径此前只有"负数"被测到。它们的区别值得单独钉一次：负数是
    上游给了一个不可信的数值，空值是上游根本没给；把任何一种填成 0，
    log1p(0)=0 会作为一个真实观测进 M2 回归，而不是被 profile_complete
    排除掉——一个"粉丝数为 0 的用户"和一个"粉丝数未知的用户"在回归里
    不是同一个人。
    """
    users = pd.DataFrame({
        "user_id": ["a", "b", "c"],
        "gender": ["m", "m", "f"],
        "n_posts": [1, 2, 3],
    })
    profiles = pd.DataFrame({
        "user_id": ["a", "b", "c"],
        "verified_type": ["0", "0", "0"],
        "user_type": ["normal"] * 3,
        "fans_number": [np.nan, -5, 0],          # 空值 / 负数 / 合法的 0
        "friends_count": [10, 10, 10],
    })
    out, _ = pj.attach_profile_controls(users, profiles)
    indexed = out.set_index("user_id")
    assert pd.isna(indexed.loc["a", "fans_number"])          # 空值 -> NaN
    assert pd.isna(indexed.loc["b", "fans_number"])          # 负数 -> NaN
    assert indexed.loc["c", "fans_number"] == 0.0            # 0 是合法真实取值
    assert pd.isna(indexed.loc["a", "log_fans"])
    assert pd.isna(indexed.loc["b", "log_fans"])
    assert indexed.loc["c", "log_fans"] == pytest.approx(0.0)
    # 只有 0 那一位进得了 M2
    assert list(indexed["profile_complete"]) == [False, False, True]


def test_verified_flag_keeps_unknown_apart_from_not_verified():
    """<NA>（未知）与 False（确认未认证）不是同一件事，绝不能合并

    合并之后 M2 会把"没拿到认证信息的用户"当成"确认未认证的用户"放进
    回归，而这两群人的构成完全不同。
    """
    users = pd.DataFrame({
        "user_id": ["a", "b", "c", "d"],
        "gender": ["m", "m", "f", "f"],
        "n_posts": [1, 2, 3, 4],
    })
    profiles = pd.DataFrame({
        "user_id": ["a", "b", "c"],              # d 完全没有画像行
        "verified_type": ["-1", "0", None],      # 未认证 / 已认证 / 画像行里为空
        "user_type": ["normal"] * 3,
        "fans_number": [1, 2, 3],
        "friends_count": [1, 2, 3],
    })
    out, _ = pj.attach_profile_controls(users, profiles)
    flags = out.set_index("user_id")["verified_flag"]
    assert flags.loc["a"] == False                              # noqa: E712
    assert flags.loc["b"] == True                               # noqa: E712
    assert pd.isna(flags.loc["c"])
    assert pd.isna(flags.loc["d"])
    # 未知的两位进不了 M2，"确认未认证"的那位进得了
    complete = out.set_index("user_id")["profile_complete"]
    assert complete.loc["a"] and complete.loc["b"]
    assert not complete.loc["c"] and not complete.loc["d"]


# ---------------------------------------------------------------------------
# profile_complete：三项 M2 控制变量必须全齐
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broken_column,bad_value", [
    ("verified_type", None),      # -> verified_flag 为 <NA>
    ("fans_number", -1),          # -> log_fans 为 NaN
    ("friends_count", np.nan),    # -> log_friends 为 NaN
])
def test_profile_complete_requires_all_three_m2_controls(broken_column, bad_value):
    """三项控制变量各坏一项，每一次都必须让这一行进不了 M2

    此前只有 friends_count 为负这一条路径被覆盖，于是"只检查了三项中的
    某一项"这种实现照样能通过全部测试，而它会把两批本该被排除的用户放进
    M2——M2 的样本量、以及论文里那句"M2 损失了 N 个用户"都会跟着错。
    """
    users = _users().iloc[[0]].copy()            # 单个男性用户，简化断言
    profiles = pd.DataFrame({
        "user_id": ["1"],
        "verified_type": ["0"],
        "user_type": ["normal"],
        "fans_number": [100],
        "friends_count": [50],
    })
    healthy, _ = pj.attach_profile_controls(users, profiles)
    assert healthy["profile_complete"].iloc[0], "健康输入本身必须是 M2 可用的"

    profiles.loc[0, broken_column] = bad_value
    broken, report = pj.attach_profile_controls(users, profiles)
    assert not broken["profile_complete"].iloc[0], broken_column
    # 画像行确实匹配上了，只是其中一项不可用——两个字段必须能看出这个差别
    assert report["profile_matched"] == 1
    assert report["m2_ready"] == 0


def test_profile_complete_requires_an_actual_matched_profile_row():
    """三项全是 NaN 的未匹配用户不能因为"没有一项显式为 False"而蒙混过关"""
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    assert not out.set_index("user_id").loc["3", "profile_complete"]


# ---------------------------------------------------------------------------
# loss report：记账恒等式
# ---------------------------------------------------------------------------

def test_loss_report_is_per_gender_and_complete():
    _, report = pj.attach_profile_controls(_users(), _profiles())
    assert report["users_total"] == 3
    assert report["by_gender"]["f"]["users_total"] == 2
    assert report["by_gender"]["f"]["profile_matched"] == 1
    assert report["by_gender"]["m"]["profile_matched"] == 1


def test_loss_report_distinguishes_matched_from_m2_ready():
    """匹配到画像 ≠ 满足 M2 全部控制变量：用户 "2" 匹配到了画像行，但
    friends_count 是 -1（异常值，清洗后为 NaN），M2 不能用这一行。
    profile_matched 和 m2_ready 必须能看出这个差别，否则读报告的人会把
    "匹配到 1 个女性用户" 误当成 "1 个女性用户可用于 M2 校正后的模型"。
    """
    _, report = pj.attach_profile_controls(_users(), _profiles())
    assert report["by_gender"]["f"]["profile_matched"] == 1
    assert report["by_gender"]["f"]["m2_ready"] == 0


def test_loss_report_accounting_identity_holds():
    """报告必须自己对得上账，否则 §11.4 那句"损失了 N 个用户"无法被核对

    三条恒等式：
      1) 分性别的 users_total 相加 == 总 users_total；
      2) 分性别的 profile_matched / m2_ready 相加 == 各自的总数；
      3) 每一格 m2_ready <= profile_matched <= users_total
         （严格判定不可能比宽松判定更宽松）。
    """
    users = pd.DataFrame({
        "user_id": [str(i) for i in range(10)],
        "gender": ["m"] * 4 + ["f"] * 5 + [None],   # 含一个性别缺失的用户
        "n_posts": list(range(10)),
    })
    profiles = pd.DataFrame({
        "user_id": [str(i) for i in range(7)],      # 7、8、9 没有画像
        "verified_type": ["0"] * 7,
        "user_type": ["normal"] * 7,
        "fans_number": [10, 20, -1, 40, 50, 60, np.nan],   # 2 与 6 不可用
        "friends_count": [1, 2, 3, 4, 5, 6, 7],
    })
    out, report = pj.attach_profile_controls(users, profiles)

    assert report["users_total"] == len(out) == 10
    by_gender = report["by_gender"]
    assert sum(g["users_total"] for g in by_gender.values()) == report["users_total"]
    for field in ("profile_matched", "m2_ready"):
        assert sum(g[field] for g in by_gender.values()) == report[field]
    for label, g in by_gender.items():
        assert g["m2_ready"] <= g["profile_matched"] <= g["users_total"], label

    # 与 DataFrame 上的那一列必须完全一致：报告不是另算一遍，是同一个定义
    assert report["m2_ready"] == int(out["profile_complete"].sum())
    assert report["m2_ready"] == 5          # 0,1,3,4,5 齐全；2/6 数值坏；7-9 未匹配
    assert report["profile_matched"] == 7


def test_loss_report_counts_gender_missing_users_in_their_own_bucket():
    """性别缺失的用户必须单独成一格，不能被静默并进 m/f 里的任何一边"""
    users = pd.DataFrame({
        "user_id": ["1", "2"],
        "gender": ["m", None],
        "n_posts": [1, 2],
    })
    _, report = pj.attach_profile_controls(users, _profiles())
    assert sum(g["users_total"] for g in report["by_gender"].values()) == 2
    assert report["by_gender"]["m"]["users_total"] == 1


def test_loss_report_is_json_serialisable_for_the_manifest():
    """整份报告会原样写进 manifest 的 counts，必须能 json 序列化"""
    _, report = pj.attach_profile_controls(_users(), _profiles())
    json.dumps(report, ensure_ascii=False, default=str)


def test_controls_present_report_covers_user_type_too():
    """user_type 不进 M2 方程，但它的缺失情况仍然值得单独看一眼"""
    _, report = pj.attach_profile_controls(_users(), _profiles())
    present = report["controls_present"]
    assert set(present) == set(pj._CONTROLS_FOR_PRESENCE_REPORT)
    assert present["user_type"] == 2


# ---------------------------------------------------------------------------
# build()：落盘、manifest 与"不覆盖原表 C"
# ---------------------------------------------------------------------------

@pytest.fixture()
def build_dirs(tmp_path, monkeypatch):
    """搭一个假的 analysis_data/ 与 merged_profiles/，并把 cwd 指过去"""
    root = tmp_path / "analysis_data"
    root.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", str(root))
    _users().to_parquet(root / "user_domain_2020.parquet",
                        engine="pyarrow", index=False)
    profiles_dir = tmp_path / "merged_profiles"
    profiles_dir.mkdir()
    _profiles().to_parquet(profiles_dir / "merged_user_profiles.parquet",
                           engine="pyarrow", index=False)
    monkeypatch.chdir(tmp_path)
    return root


def test_build_writes_a_new_file_without_touching_table_c(build_dirs):
    """刻意不覆盖 user_domain_{year}.parquet：还在读旧结构的下游不该被打破"""
    before = pd.read_parquet(build_dirs / "user_domain_2020.parquet",
                             columns=["user_id", "gender", "n_posts"])
    out_path = pj.build(year=2020)
    assert out_path.endswith("user_domain_2020_with_profile.parquet")

    after = pd.read_parquet(build_dirs / "user_domain_2020.parquet",
                            columns=["user_id", "gender", "n_posts"])
    pd.testing.assert_frame_equal(before, after)

    joined = pd.read_parquet(
        out_path, columns=["user_id", "gender", "profile_complete",
                           "log_fans", "log_friends", "verified_flag"])
    assert len(joined) == 3                      # 一个用户都没丢
    assert joined["profile_complete"].sum() == 1  # 只有用户 1 三项全齐


def test_build_writes_the_loss_report_into_the_manifest(build_dirs):
    """§11.4 要求展示 M2 之前必须能说清损失了多少人、按性别是否有差异

    这些数字的唯一出处就是 manifest 的 counts；不测它，报告写没写进去
    完全看不出来。
    """
    pj.build(year=2020)
    manifest_path = build_dirs / "profile_join_2020" / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    counts = manifest["counts"]
    assert counts["users_total"] == 3
    assert counts["profile_matched"] == 2
    assert counts["m2_ready"] == 1
    assert set(counts["by_gender"]) == {"m", "f"}
    assert counts["by_gender"]["f"]["m2_ready"] == 0
    # 溯源字段：没有它们就无法判断这批数字是哪一版代码、哪一次运行跑出来的
    assert manifest["params"]["profile_columns"] == pj.PROFILE_COLUMNS
    assert manifest["run_id"]
    assert "git_sha" in manifest
