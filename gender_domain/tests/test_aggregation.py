import numpy as np
import pandas as pd
import pytest

from gender_domain import build_user_tables as but


def _posts():
    return pd.DataFrame(
        {
            "weibo_id": ["w1", "w2", "w3", "w4", "w5"],
            "user_id": ["1", "1", "1", "2", "2"],
            "date": ["2020-01-05", "2020-01-06", "2020-02-01", "2020-01-05", "2020-01-05"],
            "month": [1, 1, 2, 1, 1],
            "gender": ["m", "m", "m", "f", "f"],
            "post_type": ["original", "retweet_plain", "original", "original", "retweet_comment"],
            "n_chars": [10, 0, 20, 10, 10],
            "public_hit": [True, False, False, False, False],
            "public_n_hits": [2, 0, 0, 0, 0],
            "public_chars_hit": [4, 0, 0, 0, 0],
            "public_density": [0.4, 0.0, 0.0, 0.0, 0.0],
            "celebrity_hit": [False, False, True, True, True],
            "celebrity_n_hits": [0, 0, 1, 1, 1],
            "celebrity_chars_hit": [0, 0, 3, 3, 3],
            "celebrity_density": [0.0, 0.0, 0.15, 0.3, 0.3],
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
