import pandas as pd

from gender_domain import build_post_table as bpt
from gender_domain import text_rules as tr


def _frame():
    return pd.DataFrame(
        {
            "weibo_id": ["w1", "w2", "w3", "w4"],
            "user_id": [1, 1, 2, 2],
            "weibo_content": ["疫情防控通报", "转发微博", "喜欢周杰伦的歌", None],
            "is_retweet": ["0", "1", "0", "1"],
            "gender": ["m", "m", "f", "f"],
            "province": ["11", "11", "44", "44"],
            "date": ["2020-03-01"] * 4,
        }
    )


def _matchers():
    return tr.VocabMatcher(["疫情", "防控"]), tr.VocabMatcher(["周杰伦"])


def test_process_frame_assigns_post_types():
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb)
    assert list(out["post_type"]) == [
        "original",
        "retweet_plain",
        "original",
        "retweet_plain",
    ]


def test_process_frame_measures_both_domains_independently():
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb).set_index("weibo_id")
    assert out.loc["w1", "public_hit"]
    assert not out.loc["w1", "celebrity_hit"]
    assert out.loc["w3", "celebrity_hit"]
    assert not out.loc["w3", "public_hit"]


def test_process_frame_stores_terms_as_pipe_joined_string():
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb).set_index("weibo_id")
    assert out.loc["w1", "public_terms"] == "防控|疫情"
    assert out.loc["w2", "public_terms"] == ""


def test_process_frame_adds_month_from_date():
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb)
    assert set(out["month"]) == {3}


def test_process_frame_keeps_null_content_rows_with_zero_measures():
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb).set_index("weibo_id")
    assert out.loc["w4", "n_chars"] == 0
    assert out.loc["w4", "public_density"] == 0.0


def test_process_frame_flags_chain_stripped_posts():
    public, celeb = _matchers()
    df = _frame()
    df.loc[0, "weibo_content"] = "疫情防控通报//@张三:原帖内容"
    out = bpt.process_frame(df, public, celeb).set_index("weibo_id")
    assert out.loc["w1", "chain_stripped"]
    assert not out.loc["w3", "chain_stripped"]
    # 剥离后只剩本人文本，字符数与命中都不含转发链内容
    assert out.loc["w1", "n_chars"] == len("疫情防控通报")


def test_process_frame_deduplicates_weibo_id():
    public, celeb = _matchers()
    doubled = pd.concat([_frame(), _frame()], ignore_index=True)
    out = bpt.process_frame(doubled, public, celeb)
    assert len(out) == 4
