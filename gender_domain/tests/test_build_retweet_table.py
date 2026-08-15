import json
import os

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


def test_process_frame_handles_float_user_id_column_with_null():
    # Fix round 1 finding：当天文件里如果有缺失 user_id，pandas 会把整数列
    # 上转型为 float64；归一化必须把 100.0 变成 "100" 而不是 "100.0"，
    # 否则来源账号自身转发排除（isin(known_sources)）会对整批行静默失配。
    df = _frame()
    df["user_id"] = [100.0, 2.0, 3.0, 4.0, None]
    out = brt.process_frame(df, brt.build_account_lookup(PUBLIC), brt.build_account_lookup(CELEBRITY))
    # w1 的 user_id 归一化后应等于来源账号 "100"，因而被排除
    assert "w1" not in set(out["weibo_id"])
    assert set(out["weibo_id"]) == {"w2", "w4"}


def test_process_frame_normalizes_user_id_and_r_user_id_to_string():
    out = brt.process_frame(
        _frame(), brt.build_account_lookup(PUBLIC), brt.build_account_lookup(CELEBRITY)
    )
    assert out["user_id"].map(type).eq(str).all()
    assert out["r_user_id"].map(type).eq(str).all()


# ---------------------------------------------------------------------------
# Fix round 2：gender 空白串、空结果帧 dtype、跨文件去重与 build_month 端到端
# ---------------------------------------------------------------------------


def test_drop_gender_null_excludes_blank_and_counts_separately():
    """与表 A 完全一致：空白串 gender 也必须排除，且与空值分开计数"""
    df = pd.DataFrame({"weibo_id": ["a", "b", "c", "d"], "gender": ["m", "", None, "\t"]})
    filtered, dropped_null, dropped_blank = brt._drop_gender_null(df)
    assert list(filtered["weibo_id"]) == ["a"]
    assert dropped_null == 1
    assert dropped_blank == 2


def test_empty_events_keeps_delay_valid_boolean_after_concat():
    """空结果帧必须带正确 dtype，否则 concat 会把 delay_valid 降级成 object

    降级之后 ~result["delay_valid"] 走的是整数按位取反（~True == -2），
    manifest 里的 non_positive_delay 会算出一个负数。
    """
    real = brt.process_frame(
        _frame(), brt.build_account_lookup(PUBLIC), brt.build_account_lookup(CELEBRITY)
    )
    combined = pd.concat([brt.empty_events(), real, brt.empty_events()], ignore_index=True)
    assert combined["delay_valid"].dtype == bool
    assert int((~combined["delay_valid"]).sum()) == 1


def test_process_frame_returns_typed_empty_frame_when_no_events():
    out = brt.process_frame(
        _frame().assign(is_retweet="0"),
        brt.build_account_lookup(PUBLIC),
        brt.build_account_lookup(CELEBRITY),
    )
    assert len(out) == 0
    assert list(out.columns) == brt.OUTPUT_COLUMNS
    assert out["delay_valid"].dtype == bool


def _write_day(data_dir, year, date_str, rows):
    day_dir = os.path.join(data_dir, str(year))
    os.makedirs(day_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        os.path.join(day_dir, f"{date_str}.parquet"), engine="pyarrow", index=False
    )


def _raw_retweet(weibo_id, user_id, r_user_id, ts, r_ts, is_retweet="1", gender="m"):
    return {
        "user_id": user_id,
        "weibo_id": weibo_id,
        "r_weibo_id": f"r_{weibo_id}",
        "r_user_id": r_user_id,
        "is_retweet": is_retweet,
        "gender": gender,
        "time_stamp": ts,
        "r_time_stamp": r_ts,
    }


def _fake_accounts(domain):
    # "300" 同时出现在两份名单里：一条转发它的记录应当真的产出两行
    if domain == "public":
        return {"central_news": {"100", "300"}}
    return {"演员": {"200", "300"}}


def _build_month_fixture(tmp_path, monkeypatch):
    data_dir = tmp_path / "cleaned_weibo_cov"
    output_dir = tmp_path / "analysis_data"
    monkeypatch.setattr(brt.config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(brt.config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(brt.config, "load_source_accounts", _fake_accounts)

    _write_day(
        str(data_dir),
        2020,
        "2020-05-01",
        [
            _raw_retweet("w1", 1, "100", 1000, 900),
            _raw_retweet("w2", 2, "300", 2000, 1500),  # 双领域，应出两行
            _raw_retweet("w5", 5, "999", 3000, 2000),  # 未命中任何来源账号
        ],
    )
    _write_day(
        str(data_dir),
        2020,
        "2020-05-02",
        [
            _raw_retweet("w1", 1, "100", 1000, 900),  # 跨日文件重复
            _raw_retweet("w2", 2, "300", 2000, 1500),  # 跨日文件重复（双领域）
            _raw_retweet("w3", 3, "100", 4000, 5000),  # 非正延迟
            _raw_retweet("w3", 3, "100", 4000, 5000),  # 同一天内重复
            _raw_retweet("w5", 5, "999", 3000, 2000),
        ],
    )
    # 这一天完全没有转发 -> 当天产出空事件帧，用来验证空帧不会污染 dtype
    _write_day(
        str(data_dir),
        2020,
        "2020-05-03",
        [_raw_retweet("w4", 4, "100", 6000, 5000, is_retweet="0")],
    )
    return output_dir


def _read_manifest(output_dir):
    path = os.path.join(
        str(output_dir), "retweet_domain_events_2020", "manifest_month_05", "manifest.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_build_month_dedups_duplicate_events_and_records_the_drop(tmp_path, monkeypatch):
    """Ruling B：同一 weibo_id 出现在两个日文件里，每个领域只应留一行

    表 A 会跨日文件按 weibo_id 去重，表 B 修复前不去重，于是一条物理转发
    可能产出两行事件，直接抬高 {d}_source_count。去重键是
    (weibo_id, source_domain)，所以真正的双领域事件仍然保留两行。
    """
    output_dir = _build_month_fixture(tmp_path, monkeypatch)
    out_path = brt.build_month(2020, 5)
    events = pd.read_parquet(out_path, columns=brt.OUTPUT_COLUMNS)

    # w1 只转了一个公共来源账号 -> 恰好一行
    assert len(events[events["weibo_id"] == "w1"]) == 1
    # w2 转的账号同时属于两个领域 -> 每个领域各一行，去重不能把它折叠掉
    w2 = events[events["weibo_id"] == "w2"]
    assert len(w2) == 2
    assert set(w2["source_domain"]) == {"public", "celebrity"}
    # w3 在同一天内重复两次 -> 也只留一行
    assert len(events[events["weibo_id"] == "w3"]) == 1
    assert len(events) == 4

    counts = _read_manifest(output_dir)["counts"]
    assert counts["events_before_dedup"] == 8
    assert counts["rows_dropped_cross_file_dedup"] == 4
    assert counts["rows_written"] == 4


def test_build_month_non_positive_delay_is_correct_when_some_days_have_no_events(
    tmp_path, monkeypatch
):
    """第 7 号发现的回归用例：有"当天无事件"的日文件时计数仍必须正确

    空事件帧一旦是全 object 列，concat 会把 delay_valid 降级成 object，
    ~ 变成整数按位取反，这个计数会变成负数（终审实测报出 -3）。
    """
    output_dir = _build_month_fixture(tmp_path, monkeypatch)
    brt.build_month(2020, 5)
    counts = _read_manifest(output_dir)["counts"]
    # 只有 w3 是非正延迟（转发时间早于原帖）
    assert counts["non_positive_delay"] == 1


def test_build_month_unmatched_source_count_is_consistent_with_duplicates(
    tmp_path, monkeypatch
):
    """rows_dropped_unmatched_source 不能用 nunique 算

    nunique 会把同一天内重复的 weibo_id 折叠成 1，于是重复行被错误地记成
    "未命中来源"——恰好在 Ruling B 的重复存在时虚高。这里 w3 在同一天内
    重复了一次：真正未命中来源的只有 w5 的两行。
    """
    output_dir = _build_month_fixture(tmp_path, monkeypatch)
    brt.build_month(2020, 5)
    counts = _read_manifest(output_dir)["counts"]
    assert counts["eligible_retweet_rows"] == 8
    assert counts["matched_retweet_rows"] == 6
    assert counts["rows_dropped_unmatched_source"] == 2  # w5 的两行
    # 逐步样本流自洽
    assert counts["rows_in"] == 9
    assert (
        counts["rows_in"]
        - counts["rows_dropped_gender_null"]
        - counts["rows_dropped_gender_blank"]
        - counts["rows_dropped_not_retweet"]
        - counts["rows_dropped_source_self_retweet"]
        == counts["eligible_retweet_rows"]
    )
    assert (
        counts["eligible_retweet_rows"] - counts["rows_dropped_unmatched_source"]
        == counts["matched_retweet_rows"]
    )
    assert (
        counts["matched_retweet_rows"] + counts["rows_emitted_twice_overlap"]
        == counts["events_before_dedup"]
    )
    assert (
        counts["events_before_dedup"] - counts["rows_dropped_cross_file_dedup"]
        == counts["rows_written"]
    )


def test_build_month_records_account_fingerprints(tmp_path, monkeypatch):
    output_dir = _build_month_fixture(tmp_path, monkeypatch)
    brt.build_month(2020, 5)
    manifest = _read_manifest(output_dir)
    assert manifest["fingerprints"]["public_accounts"] == brt.config.fingerprint_terms(
        ["100", "300"]
    )
    assert manifest["fingerprints"]["celebrity_accounts"] == brt.config.fingerprint_terms(
        ["200", "300"]
    )
    assert "git_dirty" in manifest
