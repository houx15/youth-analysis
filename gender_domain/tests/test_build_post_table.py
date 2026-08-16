import json
import os

import pandas as pd
import pytest

from gender_domain import build_post_table as bpt
from gender_domain import config
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


def test_process_frame_stores_term_counts_as_pipe_joined_string():
    # 命中词按 Unicode 码点升序排列（而非出现顺序），这样同一命中集合
    # 无论在文本中出现的先后顺序如何，拼接结果都是确定的，便于跨批次
    # 对比同一条命中组合是否一致。
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb).set_index("weibo_id")
    assert out.loc["w1", "public_term_counts"] == "疫情:1|防控:1"
    assert out.loc["w2", "public_term_counts"] == ""


def test_process_frame_term_counts_records_repeat_occurrences():
    # 同一词命中两次必须记为 2，而不是像旧版 {domain}_terms 那样去重成 1，
    # 否则重新聚合时无法从存量表还原出 n_hits/chars_hit。
    public, celeb = _matchers()
    df = _frame()
    df.loc[0, "weibo_content"] = "疫情疫情防控"
    out = bpt.process_frame(df, public, celeb).set_index("weibo_id")
    assert out.loc["w1", "public_term_counts"] == "疫情:2|防控:1"


def test_process_frame_term_counts_are_consistent_with_n_hits_and_chars_hit():
    """{domain}_n_hits/{domain}_chars_hit 必须能由 term_counts 精确还原

    这是重新聚合（§13.3 词表重采样）成立的前提：n_hits 等于所有计数之和，
    chars_hit 等于每个词的 len(term) * count 之和。
    """
    public, celeb = _matchers()
    df = _frame()
    df.loc[0, "weibo_content"] = "疫情疫情防控"
    out = bpt.process_frame(df, public, celeb)
    for _, row in out.iterrows():
        for prefix in ("public", "celebrity"):
            counts = bpt.decode_term_counts(row[f"{prefix}_term_counts"])
            assert sum(counts.values()) == row[f"{prefix}_n_hits"]
            assert sum(len(t) * c for t, c in counts.items()) == row[f"{prefix}_chars_hit"]


def test_encode_term_counts_sorts_by_term_for_determinism():
    # 无论 dict 的插入/遍历顺序如何，只要命中集合相同，编码结果必须相同。
    assert bpt.encode_term_counts({"防控": 1, "疫情": 2}) == "疫情:2|防控:1"
    assert bpt.encode_term_counts({"疫情": 2, "防控": 1}) == "疫情:2|防控:1"


def test_encode_term_counts_empty_dict_is_empty_string():
    assert bpt.encode_term_counts({}) == ""


def test_encode_term_counts_rejects_terms_containing_delimiters():
    # 一旦词表混入分隔符字符，朴素拼接就会产生无法正确切分的字符串，
    # 必须在编码时就报错，而不是让 decode_term_counts 悄悄解析错。
    for bad_term in ("疫|情", "疫:情", "疫：情"):
        with pytest.raises(ValueError):
            bpt.encode_term_counts({bad_term: 1})


def test_decode_term_counts_round_trips_with_encode():
    original = {"疫情": 2, "防控": 1}
    encoded = bpt.encode_term_counts(original)
    assert bpt.decode_term_counts(encoded) == original


def test_decode_term_counts_empty_string_is_empty_dict():
    assert bpt.decode_term_counts("") == {}


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
    filtered, dropped_null, dropped_blank = bpt._drop_gender_null(df)
    assert list(filtered["weibo_id"]) == ["a", "c"]
    assert dropped_null == 2
    assert dropped_blank == 0


def test_drop_gender_null_counts_zero_when_nothing_dropped():
    df = pd.DataFrame({"weibo_id": ["a", "b"], "gender": ["m", "f"]})
    filtered, dropped_null, dropped_blank = bpt._drop_gender_null(df)
    assert len(filtered) == 2
    assert dropped_null == 0
    assert dropped_blank == 0


def test_drop_gender_null_also_excludes_empty_and_whitespace_gender():
    """空字符串/纯空白的 gender 必须和空值一样被排除，且分开计数

    notna() 挡不住 "" 和 "  "：它们会一路活到表 C，在所有按 gender 分组的
    统计里变成一个静默的第三性别组。这里同时验证两点：确实被排除了，
    以及两类原因（真空值 vs 空白串）在 manifest 里能分别看到量级。
    """
    df = pd.DataFrame(
        {
            "weibo_id": ["a", "b", "c", "d", "e"],
            "gender": ["m", "", "   ", None, "f"],
        }
    )
    filtered, dropped_null, dropped_blank = bpt._drop_gender_null(df)
    assert list(filtered["weibo_id"]) == ["a", "e"]
    assert set(filtered["gender"]) == {"m", "f"}
    assert dropped_null == 1
    assert dropped_blank == 2


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


# ---------------------------------------------------------------------------
# Fix round 2：表达帖口径（零字符帖排除）与 build_month 端到端
# ---------------------------------------------------------------------------


def test_process_frame_writes_is_expressive_column():
    """表 A 必须自带 is_expressive 列——它是表 C/表 D 共用的唯一权威口径"""
    public, celeb = _matchers()
    out = bpt.process_frame(_frame(), public, celeb).set_index("weibo_id")
    assert "is_expressive" in out.columns
    assert out.loc["w1", "is_expressive"]  # 原创且有正文
    assert not out.loc["w2", "is_expressive"]  # 纯转发
    assert not out.loc["w4", "is_expressive"]  # 转发且正文缺失


def test_zero_char_original_is_not_expressive():
    """清洗后零字符的原创帖（纯图片/纯链接）不算表达帖

    研究负责人裁定：这类帖子不承载任何可测量的文字表达，必须排除出所有
    内容/表达指标的分子分母；但行本身保留在表 A 里，不做删除。
    """
    public, celeb = _matchers()
    df = _frame()
    df.loc[2, "weibo_content"] = "http://t.cn/abc"  # w3 原创但清洗后只剩空串
    out = bpt.process_frame(df, public, celeb).set_index("weibo_id")
    assert out.loc["w3", "post_type"] == "original"
    assert out.loc["w3", "n_chars"] == 0
    assert not out.loc["w3", "is_expressive"]
    # 行必须仍然在表里
    assert "w3" in out.index


def _write_day(data_dir, year, date_str, rows):
    """把一天的原始帖子写成服务器同构的日文件"""
    day_dir = os.path.join(data_dir, str(year))
    os.makedirs(day_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        os.path.join(day_dir, f"{date_str}.parquet"), engine="pyarrow", index=False
    )


def _raw_post(weibo_id, user_id, content, is_retweet="0", gender="m"):
    return {
        "weibo_id": weibo_id,
        "user_id": user_id,
        "weibo_content": content,
        "is_retweet": is_retweet,
        "gender": gender,
        "province": "11",
        "time_stamp": 1583020800,
    }


def test_build_month_end_to_end_writes_parquet_and_manifest(tmp_path, monkeypatch):
    """表 A 端到端：真实落盘 + manifest 内容逐项核对

    逐步样本流必须自洽：rows_in == 各项丢弃之和 + rows_written。
    这条端到端用例存在的意义是：纯函数测试测不到 build_month 里的记账、
    去重、manifest 组装（第 7 号发现正是因为没有这类用例才活到了终审）。
    """
    data_dir = tmp_path / "cleaned_weibo_cov"
    output_dir = tmp_path / "analysis_data"
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(config, "load_public_vocabulary", lambda year: ["疫情", "防控"])
    monkeypatch.setattr(config, "load_celebrity_vocabulary", lambda year: ["周杰伦"])

    _write_day(
        str(data_dir),
        2020,
        "2020-03-01",
        [
            _raw_post("w1", 1, "疫情防控通报"),
            _raw_post("w2", 1, "转发微博", is_retweet="1"),
            # 纯图片帖：原创但清洗后零字符，行保留、不算表达帖
            _raw_post("w3", 2, None, gender="f"),
            # gender 为空白串 / 空值，两类必须分开计数并双双排除
            _raw_post("w4", 3, "喜欢周杰伦的歌", gender="  "),
            _raw_post("w5", 4, "测试", gender=None),
            # 同一天内重复的 weibo_id
            _raw_post("w1", 1, "疫情防控通报"),
        ],
    )
    _write_day(
        str(data_dir),
        2020,
        "2020-03-02",
        [
            # 跨日文件重复的 weibo_id
            _raw_post("w1", 1, "疫情防控通报"),
            _raw_post("w6", 2, "疫情", gender="f"),
        ],
    )

    out_path = bpt.build_month(2020, 3)
    assert out_path is not None
    assert os.path.exists(out_path)

    written = pd.read_parquet(out_path, columns=bpt.OUTPUT_COLUMNS)
    assert set(written["weibo_id"]) == {"w1", "w2", "w3", "w6"}
    # 空白 gender 绝不能变成第三个性别组
    assert set(written["gender"]) == {"m", "f"}

    manifest_path = os.path.join(
        str(output_dir), "post_domain_measures_2020", "manifest_month_03", "manifest.json"
    )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    counts = manifest["counts"]
    assert counts["rows_in"] == 8
    assert counts["rows_dropped_gender_null"] == 1
    assert counts["rows_dropped_gender_blank"] == 1
    assert counts["rows_dropped_within_day_dedup"] == 1
    assert counts["rows_dropped_cross_file_dedup"] == 1
    assert counts["rows_written"] == 4
    # 逐步样本流自洽
    assert counts["rows_in"] == (
        counts["rows_dropped_gender_null"]
        + counts["rows_dropped_gender_blank"]
        + counts["rows_dropped_within_day_dedup"]
        + counts["rows_dropped_cross_file_dedup"]
        + counts["rows_written"]
    )
    # 零字符帖的量级必须可测：w3 是唯一一条，且它本来会被算作表达帖
    assert counts["zero_char_posts"] == {"total": 1, "would_be_expressive": 1}
    # 表达帖：w1、w6（w2 是纯转发，w3 零字符）
    assert counts["n_expressive_posts"] == 2
    assert manifest["fingerprints"]["public_vocab"] == config.fingerprint_terms(["疫情", "防控"])
    assert manifest["fingerprints"]["celebrity_vocab"] == config.fingerprint_terms(["周杰伦"])
    assert "git_dirty" in manifest
