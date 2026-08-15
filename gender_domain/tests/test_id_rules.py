import pandas as pd

from gender_domain import id_rules as ir


def test_normalize_int_column_yields_plain_digit_strings():
    out = ir.normalize_id_series(pd.Series([1, 2, 300], dtype="int64"))
    assert list(out) == ["1", "2", "300"]


def test_normalize_float_column_with_null_does_not_produce_dot_zero():
    # 列中存在缺失值时 pandas 会把整数列上转型为 float64，
    # 这里验证真实 ID 归一化后是 "123" 而不是 "123.0"
    out = ir.normalize_id_series(pd.Series([123.0, None, 456.0]))
    assert list(out) == ["123", "", "456"]


def test_normalize_string_column_passes_through_unchanged():
    out = ir.normalize_id_series(pd.Series(["001", "123", "0099"]))
    assert list(out) == ["001", "123", "0099"]


def test_normalize_mixed_object_column_with_none():
    out = ir.normalize_id_series(pd.Series([1, None, "3", float("nan")], dtype=object))
    assert list(out) == ["1", "", "3", ""]


def test_normalize_id_value_scalar_matches_series_behavior():
    assert ir.normalize_id_value(123) == "123"
    assert ir.normalize_id_value(123.0) == "123"
    assert ir.normalize_id_value("123") == "123"
    assert ir.normalize_id_value(None) == ""
    assert ir.normalize_id_value(float("nan")) == ""


def test_round_trip_float_column_matches_string_id_set_via_isin():
    # 这就是 finding 描述的确切失败场景：user_id 因该日文件里存在缺失值
    # 被上转型为 float64，astype(str) 会产出 "123.0" 从而在 isin 里
    # 匹配不上字符串来源账号 ID 集合。归一化后必须能正确命中。
    df = pd.DataFrame({"user_id": [123.0, 456.0, None]})
    normalized = ir.normalize_id_series(df["user_id"])
    known_sources = {"123", "999"}
    assert list(normalized.isin(known_sources)) == [True, False, False]
