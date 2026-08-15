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
