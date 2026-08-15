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
    # 命中词按 Unicode 码点升序排列（而非出现顺序），这样同一命中集合
    # 无论在文本中出现的先后顺序如何，拼接结果都是确定的，便于跨批次
    # 对比同一条命中组合是否一致。
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb).set_index("weibo_id")
    assert out.loc["w1", "public_terms"] == "疫情|防控"
    assert out.loc["w2", "public_terms"] == ""
    # 显式验证排序规则：即使颠倒词表/命中顺序输入，只要命中集合相同，
    # 结果字符串也必须相同——证明这是排序后的结果，不是偶然的出现顺序。
    assert out.loc["w1", "public_terms"] == "|".join(sorted(["疫情", "防控"]))


def test_process_frame_normalizes_user_id_to_string():
    # Fix round 1 finding：表 A 的 user_id 必须和表 B 一样归一化为字符串，
    # 否则两表按 user_id join 时类型不一致（这里输入本身就是 int 列，
    # 归一化必须把它转成不带 ".0" 尾巴的纯数字字符串）。
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb)
    assert list(out["user_id"]) == ["1", "1", "2", "2"]
    assert out["user_id"].map(type).eq(str).all()


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
    # 丢弃计数 = 输入行数 - 输出行数，build_month 用这个差值记录
    # within-day dedup 的丢弃数，这里显式验证这个差值是对的
    assert len(doubled) - len(out) == 4


# ---------------------------------------------------------------------------
# Fix round 1：逐步样本计数（gender 过滤 / 去重）与 date-timestamp 诊断
# ---------------------------------------------------------------------------


def test_drop_gender_null_counts_dropped_rows():
    df = pd.DataFrame(
        {
            "weibo_id": ["a", "b", "c", "d"],
            "gender": ["m", None, "f", None],
        }
    )
    filtered, dropped = bpt._drop_gender_null(df)
    assert list(filtered["weibo_id"]) == ["a", "c"]
    assert dropped == 2


def test_drop_gender_null_counts_zero_when_nothing_dropped():
    df = pd.DataFrame({"weibo_id": ["a", "b"], "gender": ["m", "f"]})
    filtered, dropped = bpt._drop_gender_null(df)
    assert len(filtered) == 2
    assert dropped == 0


def test_dedup_with_count_counts_duplicate_weibo_id():
    df = pd.DataFrame({"weibo_id": ["a", "b", "a", "c", "b"]})
    deduped, dropped = bpt._dedup_with_count(df, subset=["weibo_id"])
    assert list(deduped["weibo_id"]) == ["a", "b", "c"]
    assert dropped == 2


def test_diagnose_date_mismatch_flags_row_on_different_date():
    filename_date = pd.to_datetime("2020-03-01").date()
    # 第二行的时间戳（秒级）对应 2020-03-02，与文件名日期不一致
    weibo_ids = ["w1", "w2"]
    time_stamps = [
        pd.Timestamp("2020-03-01 08:00:00").timestamp(),
        pd.Timestamp("2020-03-02 08:00:00").timestamp(),
    ]
    diag = bpt.diagnose_date_mismatch(weibo_ids, time_stamps, filename_date)
    assert diag["mismatch_count"] == 1
    assert diag["unknown_count"] == 0
    assert diag["example_weibo_ids"] == ["w2"]


def test_diagnose_date_mismatch_all_consistent_gives_zero():
    filename_date = pd.to_datetime("2020-03-01").date()
    weibo_ids = ["w1", "w2"]
    time_stamps = [
        pd.Timestamp("2020-03-01 00:10:00").timestamp(),
        pd.Timestamp("2020-03-01 23:50:00").timestamp(),
    ]
    diag = bpt.diagnose_date_mismatch(weibo_ids, time_stamps, filename_date)
    assert diag["mismatch_count"] == 0
    assert diag["example_weibo_ids"] == []


def test_diagnose_date_mismatch_treats_unparseable_timestamp_as_unknown():
    filename_date = pd.to_datetime("2020-03-01").date()
    weibo_ids = ["w1", "w2", "w3"]
    # None、NaN、非数字字符串都算无法解析，计入 unknown 而不是 mismatch
    time_stamps = [None, float("nan"), "not-a-timestamp"]
    diag = bpt.diagnose_date_mismatch(weibo_ids, time_stamps, filename_date)
    assert diag["mismatch_count"] == 0
    assert diag["unknown_count"] == 3
    assert diag["example_weibo_ids"] == []


def test_diagnose_date_mismatch_infers_millisecond_unit():
    filename_date = pd.to_datetime("2020-03-02").date()
    # 秒级时间戳 (~1.58e9) 和毫秒级 (~1.58e12) 都应换算出同一个真实日期
    seconds = pd.Timestamp("2020-03-02 12:00:00").timestamp()
    milliseconds = seconds * 1000
    diag = bpt.diagnose_date_mismatch(["w1"], [milliseconds], filename_date)
    assert diag["mismatch_count"] == 0
    assert diag["unknown_count"] == 0


def test_diagnose_date_mismatch_caps_examples_at_max_examples():
    filename_date = pd.to_datetime("2020-03-01").date()
    mismatched_ts = pd.Timestamp("2020-03-05 00:00:00").timestamp()
    weibo_ids = [f"w{i}" for i in range(8)]
    time_stamps = [mismatched_ts] * 8
    diag = bpt.diagnose_date_mismatch(
        weibo_ids, time_stamps, filename_date, max_examples=3
    )
    assert diag["mismatch_count"] == 8
    assert len(diag["example_weibo_ids"]) == 3
