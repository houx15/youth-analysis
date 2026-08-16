import json
import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import build_user_tables as but


def _posts():
    """与表 A 同构的帖子夹具（含表 A 写出的 is_expressive 列）

    w2/w8 是纯转发，n_chars 给 4 —— 真实数据里"转发微博"这个占位文本
    就是 4 个字符。修复前这里写的是 0，恰好是唯一能让表 C 与表 D 的分母
    差异消失的取值，两个关键缺陷因此在测试里完全不可见。
    用户 4 全部活动都在 1 月，用来直接比对表 C 与表 D 的同名列。
    """
    return pd.DataFrame(
        {
            "weibo_id": ["w1", "w2", "w3", "w4", "w5", "w7", "w8"],
            "user_id": ["1", "1", "1", "2", "2", "4", "4"],
            "date": [
                "2020-01-05",
                "2020-01-06",
                "2020-02-01",
                "2020-01-05",
                "2020-01-05",
                "2020-01-09",
                "2020-01-09",
            ],
            "month": [1, 1, 2, 1, 1, 1, 1],
            "gender": ["m", "m", "m", "f", "f", "f", "f"],
            "post_type": [
                "original",
                "retweet_plain",
                "original",
                "original",
                "retweet_comment",
                "original",
                "retweet_plain",
            ],
            "n_chars": [10, 4, 20, 10, 10, 10, 4],
            "is_expressive": [True, False, True, True, True, True, False],
            "public_hit": [True, False, False, False, False, True, False],
            "public_n_hits": [2, 0, 0, 0, 0, 1, 0],
            "public_chars_hit": [4, 0, 0, 0, 0, 2, 0],
            "public_density": [0.4, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            "celebrity_hit": [False, False, True, True, True, False, False],
            "celebrity_n_hits": [0, 0, 1, 1, 1, 0, 0],
            "celebrity_chars_hit": [0, 0, 3, 3, 3, 0, 0],
            "celebrity_density": [0.0, 0.0, 0.15, 0.3, 0.3, 0.0, 0.0],
        }
    )


def _events():
    return pd.DataFrame(
        {
            "user_id": ["1", "1", "2"],
            "weibo_id": ["w2", "w9", "w5"],
            "r_weibo_id": ["r2", "r9", "r5"],
            "date": ["2020-01-06", "2020-03-01", "2020-01-05"],
            "month": [1, 3, 1],
            "gender": ["m", "m", "f"],
            "source_domain": ["public", "public", "celebrity"],
            "delay_seconds": [100.0, 300.0, 50.0],
            "delay_valid": [True, True, True],
        }
    )


def test_aggregate_posts_counts_activity():
    out = but.aggregate_posts(_posts()).set_index("user_id")
    assert out.loc["1", "n_posts"] == 3
    assert out.loc["1", "n_active_days"] == 3
    assert out.loc["1", "n_active_months"] == 2
    assert out.loc["2", "n_posts"] == 2
    assert out.loc["2", "n_active_days"] == 1


def test_missing_province_column_fills_province_and_region_as_missing():
    """旧夹具/旧分片没有 province 列时，province/region 都填缺失，不报错"""
    out = but.aggregate_posts(_posts()).set_index("user_id")
    assert "province" in out.columns
    assert "region" in out.columns
    assert out["province"].isna().all()
    assert out["region"].isna().all()


def test_region_is_derived_from_province_code_and_uses_first_observed_value():
    """region 按国家统计局四区域方案从 province 现场推导；province/region
    都取该用户在表 A 里首次出现的取值，不是众数——用户 1 先出现在
    province "11"（东部）、后面一条改成 "44"（同属东部但不同省），首值
    "11" 才是应该被采用的取值。
    """
    posts = _posts().copy()
    posts["province"] = ["11", "11", "44", "21", "21", "44", "44"]
    out = but.aggregate_posts(posts).set_index("user_id")
    # 用户 1（w1/w2/w3）首条是 "11"，即便后面出现 "44" 也不改变首值
    assert out.loc["1", "province"] == "11"
    assert out.loc["1", "region"] == "East"
    # 用户 2（w4/w5）全部是 "21"（东北）
    assert out.loc["2", "province"] == "21"
    assert out.loc["2", "region"] == "Northeast"
    # 用户 4（w7/w8）全部是 "44"（东部）
    assert out.loc["4", "province"] == "44"
    assert out.loc["4", "region"] == "East"


def test_topical_share_uses_expressive_posts_as_denominator():
    out = but.aggregate_posts(_posts()).set_index("user_id")
    # 用户1有3帖，其中w2是纯转发，表达帖为2
    assert out.loc["1", "public_topical_posts"] == 1
    assert out.loc["1", "public_topical_share"] == pytest.approx(0.5)
    # 全部帖子为分母的并行口径
    assert out.loc["1", "public_topical_share_allposts"] == pytest.approx(1 / 3)


def test_char_density_is_character_weighted_not_post_mean():
    out = but.aggregate_posts(_posts()).set_index("user_id")
    # 用户1表达帖字符 10+20=30，命中字符 4
    assert out.loc["1", "public_char_density"] == pytest.approx(4 / 30)


def test_hits_per_1k_characters():
    out = but.aggregate_posts(_posts()).set_index("user_id")
    assert out.loc["1", "public_hits_per_1k"] == pytest.approx(2 / 30 * 1000)


def test_content_months_counts_distinct_months_with_a_hit():
    out = but.aggregate_posts(_posts()).set_index("user_id")
    assert out.loc["1", "public_content_months"] == 1
    assert out.loc["1", "celebrity_content_months"] == 1


def test_aggregate_events_counts_and_delay_median():
    out = but.aggregate_events(_events()).set_index("user_id")
    assert out.loc["1", "public_source_count"] == 2
    assert out.loc["1", "public_source_months"] == 2
    assert out.loc["1", "public_delay_median"] == pytest.approx(200.0)
    assert out.loc["2", "celebrity_source_count"] == 1


def test_combine_assigns_source_combination():
    combined = but.combine_user_table(
        but.aggregate_posts(_posts()),
        but.aggregate_events(_events()),
        but.aggregate_user_month(_posts(), _events()),
    ).set_index("user_id")
    assert combined.loc["1", "source_combo"] == "public_only"
    assert combined.loc["2", "source_combo"] == "celebrity_only"


def test_source_share_divides_by_total_retweets():
    combined = but.combine_user_table(
        but.aggregate_posts(_posts()),
        but.aggregate_events(_events()),
        but.aggregate_user_month(_posts(), _events()),
    ).set_index("user_id")
    # 用户1的表A里有1条转发帖(w2)，但事件表显示2次公共来源转发(w2,w9)
    # 分母以表A观察到的转发帖数为准，比例上限截到1
    assert combined.loc["1", "public_source_share"] <= 1.0


def test_active_month_denominator_includes_retweet_only_months():
    combined = but.combine_user_table(
        but.aggregate_posts(_posts()),
        but.aggregate_events(_events()),
        but.aggregate_user_month(_posts(), _events()),
    ).set_index("user_id")
    # 用户1在1月、2月有帖，3月只有一次公共来源转发 -> 活跃月份为3
    assert combined.loc["1", "n_active_months"] == 2
    assert combined.loc["1", "n_active_months_panel"] == 3
    assert combined.loc["1", "public_source_month_share"] == pytest.approx(2 / 3)


def test_user_month_panel_has_one_row_per_active_month():
    panel = but.aggregate_user_month(_posts(), _events())
    assert len(panel[panel["user_id"] == "1"]) == 3  # 1月、2月有帖，3月只有转发事件
    row = panel[(panel["user_id"] == "1") & (panel["month"] == 1)].iloc[0]
    assert row["n_posts"] == 2
    assert row["public_source_count"] == 1


def test_zero_denominator_yields_nan_not_zero():
    """没有表达帖的用户：占比/字符密度/千字命中率必须是 NaN，不能是 0

    用户3全部是纯转发（无表达帖），public_topical_posts 这类计数应为 0
    （确实发生了 0 次命中，是有定义的事实），但任何以表达帖数/命中字符数
    为分母的比例（topical_share、char_density、hits_per_1k）在分母为 0
    时必须是 NaN，代表"无法计算"而不是"计算出来是 0"。
    """
    posts = _posts().copy()
    extra = pd.DataFrame(
        {
            "weibo_id": ["w6"],
            "user_id": ["3"],
            "date": ["2020-01-07"],
            "month": [1],
            "gender": ["m"],
            "post_type": ["retweet_plain"],
            "n_chars": [0],
            "is_expressive": [False],
            "public_hit": [False],
            "public_n_hits": [0],
            "public_chars_hit": [0],
            "public_density": [0.0],
            "celebrity_hit": [False],
            "celebrity_n_hits": [0],
            "celebrity_chars_hit": [0],
            "celebrity_density": [0.0],
        }
    )
    out = but.aggregate_posts(pd.concat([posts, extra], ignore_index=True)).set_index("user_id")

    assert out.loc["3", "n_expressive_posts"] == 0
    assert out.loc["3", "public_topical_posts"] == 0
    assert np.isnan(out.loc["3", "public_topical_share"])
    assert np.isnan(out.loc["3", "public_char_density"])
    assert np.isnan(out.loc["3", "public_hits_per_1k"])


def test_user_id_normalization_matches_id_rules_not_bare_astype_str():
    """user_id 必须走 id_rules.normalize_id_series，而不是裸 astype(str)

    帖子表和事件表里如果 user_id 列因缺失值被上转型为 float64，裸
    astype(str) 会把整数 ID 变成 "123.0" 这种伪 ID，导致两表 join 失效。
    这里模拟该场景：user_id 列含 NaN，使整列变成 float64。
    """
    posts = pd.DataFrame(
        {
            "weibo_id": ["w1", "w2"],
            "user_id": [123, np.nan],
            "date": ["2020-01-05", "2020-01-05"],
            "month": [1, 1],
            "gender": ["m", "m"],
            "post_type": ["original", "original"],
            "n_chars": [10, 10],
            "public_hit": [True, False],
            "public_n_hits": [1, 0],
            "public_chars_hit": [2, 0],
            "public_density": [0.2, 0.0],
            "celebrity_hit": [False, False],
            "celebrity_n_hits": [0, 0],
            "celebrity_chars_hit": [0, 0],
            "celebrity_density": [0.0, 0.0],
        }
    )
    out = but.aggregate_posts(posts)
    assert "123.0" not in set(out["user_id"])
    assert "123" in set(out["user_id"])


def test_event_only_user_never_borrows_a_neighbours_gender():
    """事件表专属月份的用户不能从相邻用户那里"借"性别

    回归用例：user_id "9" 只有帖子（gender "m"），user_id "77" 全年
    没有任何帖子、只在事件表里出现一次（2月一次公共来源转发）。
    aggregate_user_month 的 gender 只从帖子表取值（monthly 的
    gender=("gender","first")），因此用户77在整张面板里没有任何
    "自己的" gender 事实可用。

    这正是修复前的 bug 复现场景：
    panel.groupby(level="user_id")["gender"].ffill().bfill() 里，
    SeriesGroupBy.ffill() 返回的是普通 Series（分组信息已丢失），链式
    .bfill() 因此在整张面板上无分组回填——用本文件同款最小夹具手工
    验证过，旧代码会把用户77的 gender 错误地填成用户9的 "m"
    （见任务报告的复现记录）。正确行为是：既不能凭空获得一个
    "属于自己"但实际上不存在的性别，也绝不能借用户9的 "m"，
    应该保持缺失（NaN）。
    """
    posts = pd.DataFrame(
        {
            "weibo_id": ["w1"],
            "user_id": ["9"],
            "date": ["2020-01-05"],
            "month": [1],
            "gender": ["m"],
            "post_type": ["original"],
            "n_chars": [10],
            "public_hit": [False],
            "public_n_hits": [0],
            "public_chars_hit": [0],
            "public_density": [0.0],
            "celebrity_hit": [False],
            "celebrity_n_hits": [0],
            "celebrity_chars_hit": [0],
            "celebrity_density": [0.0],
        }
    )
    events = pd.DataFrame(
        {
            "user_id": ["77"],
            "weibo_id": ["w99"],
            "r_weibo_id": ["r99"],
            "date": ["2020-02-01"],
            "month": [2],
            "gender": ["f"],
            "source_domain": ["public"],
            "delay_seconds": [100.0],
            "delay_valid": [True],
        }
    )
    panel = but.aggregate_user_month(posts, events)
    row77 = panel[(panel["user_id"] == "77") & (panel["month"] == 2)].iloc[0]
    row9 = panel[(panel["user_id"] == "9") & (panel["month"] == 1)].iloc[0]

    assert row9["gender"] == "m"
    # 关键断言：绝不能等于邻居用户9的 "m"（旧代码的实际错误行为）
    assert row77["gender"] != "m"
    # 用户77在帖子表里从未出现过，面板里没有任何"属于自己"的 gender
    # 事实可用，正确行为是保持缺失，而不是凭空冒出一个值
    assert pd.isna(row77["gender"])


def test_gender_missing_in_every_row_of_a_group_stays_missing():
    """一个用户所有帖子行的 gender 都缺失时，不能被组内 ffill/bfill 造出一个值

    正常流水线里表 A/表 B 都会在更早的阶段丢弃 gender 为空的行
    （见 build_post_table.py / build_retweet_table.py 的 _drop_gender_null），
    但 aggregate_user_month 本身不应该依赖这个前提——如果某个用户组内
    gender 全部缺失，组内 transform(ffill().bfill()) 找不到任何非空值
    可供借用，必须原样保留 NaN，不能被强行赋成某个值。
    """
    posts = pd.DataFrame(
        {
            "weibo_id": ["w1", "w2"],
            "user_id": ["5", "5"],
            "date": ["2020-01-05", "2020-01-06"],
            "month": [1, 1],
            "gender": [None, None],
            "post_type": ["original", "original"],
            "n_chars": [10, 10],
            "public_hit": [False, False],
            "public_n_hits": [0, 0],
            "public_chars_hit": [0, 0],
            "public_density": [0.0, 0.0],
            "celebrity_hit": [False, False],
            "celebrity_n_hits": [0, 0],
            "celebrity_chars_hit": [0, 0],
            "celebrity_density": [0.0, 0.0],
        }
    )
    events = pd.DataFrame(
        {
            "user_id": pd.Series([], dtype="object"),
            "weibo_id": pd.Series([], dtype="object"),
            "r_weibo_id": pd.Series([], dtype="object"),
            "date": pd.Series([], dtype="object"),
            "month": pd.Series([], dtype="int64"),
            "gender": pd.Series([], dtype="object"),
            "source_domain": pd.Series([], dtype="object"),
            "delay_seconds": pd.Series([], dtype="float64"),
            "delay_valid": pd.Series([], dtype="bool"),
        }
    )
    panel = but.aggregate_user_month(posts, events)
    row = panel[(panel["user_id"] == "5") & (panel["month"] == 1)].iloc[0]
    assert pd.isna(row["gender"])


def test_topical_posts_stays_integer_dtype_when_a_user_has_zero_expressive_posts():
    """{d}_topical_posts 在混入零表达帖用户后，dtype 仍必须是整数

    reindex 引入的 NaN 会先把整列上转型为 float64；fillna(0) 只改值不改
    dtype。若聚合函数忘记在 fillna 之后 astype(int)，用给定夹具（每个
    用户都至少有一条表达帖）测不出来——只有像这里一样混入一个零表达帖
    用户，才会触发 reindex 产生的 NaN，从而暴露这个 dtype 缺陷。
    """
    posts = pd.DataFrame(
        {
            "weibo_id": ["w1", "w2"],
            "user_id": ["1", "2"],
            "date": ["2020-01-05", "2020-01-05"],
            "month": [1, 1],
            "gender": ["m", "f"],
            # 用户1有一条表达帖；用户2只有纯转发，零表达帖
            "post_type": ["original", "retweet_plain"],
            "n_chars": [10, 0],
            "public_hit": [True, False],
            "public_n_hits": [1, 0],
            "public_chars_hit": [2, 0],
            "public_density": [0.2, 0.0],
            "celebrity_hit": [False, False],
            "celebrity_n_hits": [0, 0],
            "celebrity_chars_hit": [0, 0],
            "celebrity_density": [0.0, 0.0],
        }
    )
    out = but.aggregate_posts(posts).set_index("user_id")
    assert pd.api.types.is_integer_dtype(out["public_topical_posts"])
    assert pd.api.types.is_integer_dtype(out["celebrity_topical_posts"])
    assert out.loc["1", "public_topical_posts"] == 1
    assert out.loc["2", "public_topical_posts"] == 0


def test_diagnose_event_only_users_counts_without_dropping_anyone():
    """左连接假设的诊断：只计数，不改变任何表的内容

    combine_user_table 用左连接把 event_agg 拼到 post_agg 上，隐含假设
    "出现在表 B 的用户一定也出现在表 A"。这个假设从未在真实数据上验证
    过，所以只做计数写入 manifest，不阻断流程——这里直接测这个计数
    函数本身：构造一个只出现在 event_agg、不出现在 post_agg 的用户，
    确认它被正确计数且被列在示例里。
    """
    post_agg = pd.DataFrame({"user_id": ["1", "2"]})
    event_agg = pd.DataFrame({"user_id": ["2", "3"]})
    diag = but.diagnose_event_only_users(post_agg, event_agg)
    assert diag["event_only_user_count"] == 1
    assert diag["example_user_ids"] == ["3"]


# ---------------------------------------------------------------------------
# Fix round 2：表 C 与表 D 必须共用同一个表达帖分母（终审 CRITICAL 1/2）
# ---------------------------------------------------------------------------


def _single_month_user_row(panel, user_id, month=1):
    return panel[(panel["user_id"] == user_id) & (panel["month"] == month)].iloc[0]


def test_table_d_char_density_equals_table_c_for_single_month_user():
    """同名列必须同口径：表 D 的 char_density 分母也只能是表达帖

    用户 4 的全部活动都在 1 月，所以表 C（全年）与表 D（该月）算的是
    同一批帖子，两个数字必须完全相等。手算：表达帖只有原创 w7（10 字，
    命中 2 字），纯转发 w8 的 4 个字（"转发微博"）不进分母 -> 2/10 = 0.2。
    修复前表 D 把 w8 的 4 个字也算进分母，得到 2/14 ≈ 0.1429。
    """
    posts = _posts()
    table_c = but.aggregate_posts(posts).set_index("user_id")
    table_d = but.aggregate_user_month(posts, _events())
    row = _single_month_user_row(table_d, "4")

    assert table_c.loc["4", "public_char_density"] == pytest.approx(0.2)
    assert row["public_char_density"] == pytest.approx(0.2)
    assert row["public_char_density"] == pytest.approx(table_c.loc["4", "public_char_density"])
    # 修复前的错误取值必须不再出现
    assert row["public_char_density"] != pytest.approx(2 / 14)


def test_table_d_topical_share_equals_table_c_for_single_month_user():
    """手算：用户 4 表达帖 1 条（w7），其中命中公共事务 1 条 -> 1/1 = 1.0"""
    posts = _posts()
    table_c = but.aggregate_posts(posts).set_index("user_id")
    table_d = but.aggregate_user_month(posts, _events())
    row = _single_month_user_row(table_d, "4")

    assert table_c.loc["4", "public_topical_share"] == pytest.approx(1.0)
    assert row["public_topical_share"] == pytest.approx(1.0)
    assert row["n_expressive_posts"] == table_c.loc["4", "n_expressive_posts"] == 1
    assert row["public_topical_posts"] == table_c.loc["4", "public_topical_posts"] == 1


def test_table_d_expressive_denominator_matches_table_c_for_every_single_month_user():
    """所有"全部活动都在同一个月"的用户，两张表的同名列都必须相等"""
    posts = _posts()
    table_c = but.aggregate_posts(posts).set_index("user_id")
    table_d = but.aggregate_user_month(posts, _events())
    single_month_users = [
        uid for uid in table_c.index if table_c.loc[uid, "n_active_months"] == 1
    ]
    assert single_month_users  # 夹具里必须真的有这种用户
    for uid in single_month_users:
        month = int(posts[posts["user_id"] == uid]["month"].iloc[0])
        row = _single_month_user_row(table_d, uid, month)
        for domain in but.DOMAINS:
            for col in ("char_density", "topical_share"):
                c_value = table_c.loc[uid, f"{domain}_{col}"]
                d_value = row[f"{domain}_{col}"]
                if pd.isna(c_value):
                    assert pd.isna(d_value)
                else:
                    assert d_value == pytest.approx(c_value)


def _posts_with_hitting_plain_retweet():
    """纯转发帖本身命中了词表词的场景

    真实数据里完全可能发生：转发时平台带出的占位文本或被截断的正文里
    恰好含有词表里的词。修复前表 D 的分子统计全部帖子、分母只统计表达帖，
    这种场景会直接把 topical_share 推到 2.0。
    """
    return pd.DataFrame(
        {
            "weibo_id": ["p1", "p2"],
            "user_id": ["7", "7"],
            "date": ["2020-04-01", "2020-04-02"],
            "month": [4, 4],
            "gender": ["f", "f"],
            "post_type": ["original", "retweet_plain"],
            "n_chars": [10, 4],
            "is_expressive": [True, False],
            "public_hit": [True, True],
            "public_n_hits": [1, 1],
            "public_chars_hit": [2, 2],
            "public_density": [0.2, 0.5],
            "celebrity_hit": [False, False],
            "celebrity_n_hits": [0, 0],
            "celebrity_chars_hit": [0, 0],
            "celebrity_density": [0.0, 0.0],
        }
    )


def _empty_events():
    return pd.DataFrame(
        {
            "user_id": pd.Series([], dtype="object"),
            "weibo_id": pd.Series([], dtype="object"),
            "month": pd.Series([], dtype="int64"),
            "source_domain": pd.Series([], dtype="object"),
            "delay_seconds": pd.Series([], dtype="float64"),
            "delay_valid": pd.Series([], dtype="bool"),
        }
    )


def test_topical_share_never_exceeds_one_when_a_plain_retweet_hits_vocabulary():
    """占比是"命中的表达帖 / 表达帖"，两张表都不可能大于 1

    修复前表 D 在这个夹具上算出 2.0（分子 2 条命中帖 / 分母 1 条表达帖）。
    """
    posts = _posts_with_hitting_plain_retweet()
    table_c = but.aggregate_posts(posts).set_index("user_id")
    table_d = but.aggregate_user_month(posts, _empty_events())

    assert table_c.loc["7", "public_topical_share"] == pytest.approx(1.0)
    row = _single_month_user_row(table_d, "7", month=4)
    assert row["public_topical_share"] == pytest.approx(1.0)

    for domain in but.DOMAINS:
        c_shares = table_c[f"{domain}_topical_share"].dropna()
        d_shares = table_d[f"{domain}_topical_share"].dropna()
        assert (c_shares <= 1.0).all()
        assert (d_shares <= 1.0).all()


def test_topical_share_never_exceeds_one_in_the_main_fixture():
    """主夹具上也把上界断言钉死，防止今后任何改动让它重新越界"""
    posts = _posts()
    table_c = but.aggregate_posts(posts)
    table_d = but.aggregate_user_month(posts, _events())
    for domain in but.DOMAINS:
        assert (table_c[f"{domain}_topical_share"].dropna() <= 1.0).all()
        assert (table_d[f"{domain}_topical_share"].dropna() <= 1.0).all()


def _posts_with_image_only_post():
    """一条纯图片帖（原创但清洗后零字符）+ 一条正常表达帖"""
    return pd.DataFrame(
        {
            "weibo_id": ["i1", "i2"],
            "user_id": ["8", "8"],
            "date": ["2020-06-01", "2020-06-02"],
            "month": [6, 6],
            "gender": ["m", "m"],
            "post_type": ["original", "original"],
            "n_chars": [0, 10],
            "is_expressive": [False, True],
            "public_hit": [False, True],
            "public_n_hits": [0, 1],
            "public_chars_hit": [0, 2],
            "public_density": [0.0, 0.2],
            "celebrity_hit": [False, False],
            "celebrity_n_hits": [0, 0],
            "celebrity_chars_hit": [0, 0],
            "celebrity_density": [0.0, 0.0],
        }
    )


def test_image_only_post_is_excluded_from_expressive_denominator_in_both_tables():
    """Ruling A：零字符帖（纯图片/纯链接）不进任何内容指标的分母

    该用户有 2 条帖子，但只有 1 条表达帖，表 C 与表 D 的分母都必须是 1。
    行本身不被删除，n_posts 仍然是 2。
    """
    posts = _posts_with_image_only_post()
    table_c = but.aggregate_posts(posts).set_index("user_id")
    table_d = but.aggregate_user_month(posts, _empty_events())
    row = _single_month_user_row(table_d, "8", month=6)

    assert table_c.loc["8", "n_posts"] == 2
    assert table_c.loc["8", "n_expressive_posts"] == 1
    assert row["n_posts"] == 2
    assert row["n_expressive_posts"] == 1
    # 分母是 1 不是 2
    assert table_c.loc["8", "public_topical_share"] == pytest.approx(1.0)
    assert row["public_topical_share"] == pytest.approx(1.0)
    # 字符分母也只数表达帖
    assert table_c.loc["8", "public_char_density"] == pytest.approx(0.2)
    assert row["public_char_density"] == pytest.approx(0.2)


def test_allposts_variant_keeps_its_literal_all_posts_denominator():
    """_allposts 稳健性口径刻意不采纳零字符帖排除规则

    它存在的意义是与旧流水线可比，跟着主口径改就失去了比较价值。
    用户 8 有 2 条帖子、1 条命中 -> 1/2 = 0.5（而主口径是 1/1 = 1.0）。
    """
    table_c = but.aggregate_posts(_posts_with_image_only_post()).set_index("user_id")
    assert table_c.loc["8", "public_topical_share_allposts"] == pytest.approx(0.5)
    assert table_c.loc["8", "public_topical_share"] == pytest.approx(1.0)


def test_aggregations_consume_table_a_is_expressive_column_instead_of_re_deriving():
    """表 A 写出的 is_expressive 是唯一权威口径，聚合端不得自己重推

    这里把 is_expressive 人为改成与 post_type 不一致的取值：如果聚合函数
    仍然按 post_type 自行推导，就会忽略这一列，两张表就又会各推各的。
    """
    posts = _posts_with_image_only_post()
    posts["is_expressive"] = [True, False]  # 与 post_type/n_chars 刻意不一致
    table_c = but.aggregate_posts(posts).set_index("user_id")
    table_d = but.aggregate_user_month(posts, _empty_events())
    assert table_c.loc["8", "n_expressive_posts"] == 1
    assert _single_month_user_row(table_d, "8", month=6)["n_expressive_posts"] == 1
    # 被标为表达帖的是零字符的 i1，命中帖 i2 反而不算 -> 命中数为 0
    assert table_c.loc["8", "public_topical_posts"] == 0


def test_missing_is_expressive_column_falls_back_to_the_same_single_definition():
    """旧版分片没有 is_expressive 列时，按同一个定义现场推导（并打印警告）"""
    posts = _posts_with_image_only_post().drop(columns=["is_expressive"])
    table_c = but.aggregate_posts(posts).set_index("user_id")
    assert table_c.loc["8", "n_expressive_posts"] == 1


def test_source_combo_is_vectorised_and_covers_every_combination():
    entered_public = pd.Series([True, True, False, False, None])
    entered_celebrity = pd.Series([True, False, True, False, None])
    combo = but._source_combo_series(entered_public, entered_celebrity)
    assert list(combo) == ["both", "public_only", "celebrity_only", "neither", "neither"]


# ---------------------------------------------------------------------------
# Fix round 2：分片读取只取需要的列 + 表 C/D manifest 的输入与指纹
# ---------------------------------------------------------------------------


def _write_post_shards(output_dir, with_manifest_months=(1,)):
    shard_dir = output_dir / "post_domain_measures_2020"
    shard_dir.mkdir(parents=True)
    frame = _posts().copy()
    # 表 A 真实存在、但表 C/表 D 一次都用不到的列
    frame["public_terms"] = "疫情|防控"
    frame["celebrity_terms"] = ""
    frame["chain_stripped"] = False
    frame["province"] = "11"
    for month in sorted(frame["month"].unique()):
        part = frame[frame["month"] == month]
        part.to_parquet(
            shard_dir / f"month={month:02d}.parquet", engine="pyarrow", index=False
        )
        if month in with_manifest_months:
            manifest_dir = shard_dir / f"manifest_month_{month:02d}"
            manifest_dir.mkdir()
            with open(manifest_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": f"post_domain_measures_2020_{month:02d}",
                        "fingerprints": {
                            "public_vocab": "aaa11111",
                            "celebrity_vocab": "bbb22222",
                        },
                    },
                    f,
                )
    return shard_dir


def _write_event_shards(output_dir):
    shard_dir = output_dir / "retweet_domain_events_2020"
    shard_dir.mkdir(parents=True)
    frame = _events().copy()
    for month in sorted(frame["month"].unique()):
        part = frame[frame["month"] == month]
        part.to_parquet(
            shard_dir / f"month={month:02d}.parquet", engine="pyarrow", index=False
        )
        manifest_dir = shard_dir / f"manifest_month_{month:02d}"
        manifest_dir.mkdir()
        with open(manifest_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": f"retweet_domain_events_2020_{month:02d}",
                    "fingerprints": {
                        "public_accounts": "ccc33333",
                        "celebrity_accounts": "ddd44444",
                    },
                },
                f,
            )
    return shard_dir


def test_read_shards_requests_only_the_columns_the_aggregations_use(tmp_path, monkeypatch):
    """parquet 读取必须显式传 columns

    表 A 带着 {domain}_terms 两列竖线拼接的命中词字符串，表 C/表 D 一次
    都用不到，全年读进内存纯属浪费（也违反项目的全局约定）。
    """
    monkeypatch.setattr(but.config, "OUTPUT_DIR", str(tmp_path))
    _write_post_shards(tmp_path)
    frame, files = but._read_shards("post_domain_measures", 2020, but.POST_SHARD_COLUMNS)
    assert list(frame.columns) == but.POST_SHARD_COLUMNS
    assert "public_terms" not in frame.columns
    assert "celebrity_terms" not in frame.columns
    assert len(files) == 2


def test_collect_shard_fingerprints_records_missing_manifests_explicitly(tmp_path):
    """分月 manifest 缺失必须显式记录，不能静默省略指纹

    "没有指纹"和"指纹一致"在文件里长得一模一样，静默省略等于伪装成正常。
    """
    shard_dir = _write_post_shards(tmp_path, with_manifest_months=(1,))
    files = sorted(str(p) for p in shard_dir.glob("month=*.parquet"))
    fingerprints = but.collect_shard_fingerprints(str(shard_dir), files)
    assert fingerprints["public_vocab"] == ["aaa11111"]
    assert fingerprints["celebrity_vocab"] == ["bbb22222"]
    assert fingerprints["missing_manifests"] == [
        "post_domain_measures_2020/manifest_month_02"
    ]


def test_collect_shard_fingerprints_surfaces_inconsistent_months(tmp_path):
    """不同月份用了不同词表时，两个取值都必须留在 manifest 里"""
    shard_dir = _write_post_shards(tmp_path, with_manifest_months=(1, 2))
    other = shard_dir / "manifest_month_02" / "manifest.json"
    with open(other, "w", encoding="utf-8") as f:
        json.dump({"fingerprints": {"public_vocab": "zzz99999"}}, f)
    files = sorted(str(p) for p in shard_dir.glob("month=*.parquet"))
    fingerprints = but.collect_shard_fingerprints(str(shard_dir), files)
    assert fingerprints["public_vocab"] == ["aaa11111", "zzz99999"]
    assert fingerprints["missing_manifests"] == []


def test_build_writes_tables_and_a_self_describing_manifest(tmp_path, monkeypatch):
    """表 C/表 D 端到端：落盘 + manifest 记录实际分片与继承的指纹

    这两张表是论文数字的直接来源，却曾是全流程里唯一不记指纹、inputs
    只写目录名的一步（目录名说不清这次到底读到了哪几个月）。
    """
    monkeypatch.setattr(but.config, "OUTPUT_DIR", str(tmp_path))
    _write_post_shards(tmp_path, with_manifest_months=(1,))
    _write_event_shards(tmp_path)

    user_path, panel_path = but.build(2020)
    assert os.path.exists(user_path)
    assert os.path.exists(panel_path)

    with open(tmp_path / "user_tables_2020" / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    # inputs 是实际 glob 到的分片文件，而不是两个目录名
    assert "post_domain_measures_2020/month=01.parquet" in manifest["inputs"]
    assert "post_domain_measures_2020/month=02.parquet" in manifest["inputs"]
    assert "retweet_domain_events_2020/month=01.parquet" in manifest["inputs"]
    # 指纹从上游分片 manifest 继承
    assert manifest["fingerprints"]["post_shards"]["public_vocab"] == ["aaa11111"]
    assert manifest["fingerprints"]["post_shards"]["celebrity_vocab"] == ["bbb22222"]
    assert manifest["fingerprints"]["retweet_shards"]["public_accounts"] == ["ccc33333"]
    # 缺失的分月 manifest 被显式记录
    assert manifest["fingerprints"]["post_shards"]["missing_manifests"] == [
        "post_domain_measures_2020/manifest_month_02"
    ]
    assert manifest["params"]["expressive_requires_nonzero_chars"] is True
    assert "git_dirty" in manifest

    # 落盘的表 C/表 D 在同名列上必须给出同一个数字（用户 4 只活跃在 1 月）
    users = pd.read_parquet(
        user_path, columns=["user_id", "public_char_density", "public_topical_share"]
    ).set_index("user_id")
    panel = pd.read_parquet(
        panel_path,
        columns=["user_id", "month", "public_char_density", "public_topical_share"],
    )
    row = panel[(panel["user_id"] == "4") & (panel["month"] == 1)].iloc[0]
    assert row["public_char_density"] == pytest.approx(users.loc["4", "public_char_density"])
    assert row["public_topical_share"] == pytest.approx(users.loc["4", "public_topical_share"])


def test_missing_values_in_is_expressive_column_fail_loudly(monkeypatch):
    """is_expressive 列有缺失时必须报错，不能被 astype(bool) 静默当成 True

    astype(bool) 会把 NaN 变成 True，等于凭空把一批帖子塞进内容指标的分母。
    """
    posts = _posts_with_image_only_post()
    posts["is_expressive"] = [True, None]
    with pytest.raises(ValueError, match="is_expressive"):
        but.aggregate_posts(posts)
