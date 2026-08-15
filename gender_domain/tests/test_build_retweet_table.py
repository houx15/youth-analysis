import pandas as pd

from gender_domain import build_retweet_table as brt


PUBLIC = {"central_news": {"100", "101"}, "gov_release_weibo": {"101"}}
CELEBRITY = {"演员": {"200"}}


def _frame():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "weibo_id": ["w1", "w2", "w3", "w4", "w5"],
            "r_weibo_id": ["r1", "r2", "r3", "r4", "r5"],
            "r_user_id": ["100", "200", "999", "101", "100"],
            "is_retweet": ["1", "1", "1", "1", "0"],
            "gender": ["m", "f", "m", "f", "m"],
            "time_stamp": [1000, 2000, 3000, 4000, 5000],
            "r_time_stamp": [900, 1500, 2000, 5000, 100],
            "date": ["2020-05-02"] * 5,
        }
    )


def test_build_account_lookup_joins_multiple_categories():
    lookup = brt.build_account_lookup(PUBLIC)
    assert lookup["100"] == "central_news"
    assert lookup["101"] == "central_news|gov_release_weibo"


def test_process_frame_keeps_only_retweets_of_known_sources():
    out = brt.process_frame(_frame(), brt.build_account_lookup(PUBLIC), brt.build_account_lookup(CELEBRITY))
    # w3 转发未知账号，w5 不是转发
    assert set(out["weibo_id"]) == {"w1", "w2", "w4"}


def test_process_frame_labels_domain_and_category():
    out = brt.process_frame(_frame(), brt.build_account_lookup(PUBLIC), brt.build_account_lookup(CELEBRITY)).set_index("weibo_id")
    assert out.loc["w1", "source_domain"] == "public"
    assert out.loc["w1", "source_category"] == "central_news"
    assert out.loc["w2", "source_domain"] == "celebrity"
    assert out.loc["w2", "source_category"] == "演员"


def test_process_frame_computes_delay_and_flags_non_positive():
    out = brt.process_frame(_frame(), brt.build_account_lookup(PUBLIC), brt.build_account_lookup(CELEBRITY)).set_index("weibo_id")
    assert out.loc["w1", "delay_seconds"] == 100
    assert out.loc["w1", "delay_valid"]
    # w4: 转发时间早于原帖时间，必须保留但标记无效
    assert out.loc["w4", "delay_seconds"] == -1000
    assert not out.loc["w4", "delay_valid"]


def test_process_frame_emits_one_row_per_domain_for_overlapping_account():
    overlapping = brt.build_account_lookup({"演员": {"100"}})
    out = brt.process_frame(_frame(), brt.build_account_lookup(PUBLIC), overlapping)
    w1_rows = out[out["weibo_id"] == "w1"]
    assert set(w1_rows["source_domain"]) == {"public", "celebrity"}


def test_process_frame_excludes_source_accounts_own_posts():
    df = _frame()
    df.loc[0, "user_id"] = 100  # 来源账号自己转发自己
    out = brt.process_frame(df, brt.build_account_lookup(PUBLIC), brt.build_account_lookup(CELEBRITY))
    assert "w1" not in set(out["weibo_id"])
