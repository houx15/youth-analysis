import pandas as pd
import pytest

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


def test_normalize_real_weibo_scale_id_still_exact():
    # 真实 2020 微博 uid 大约 10 位数字，远低于 float64 精确整数上限
    # (2**53 约 9.0e15)，必须原样精确归一化，不受精度边界检查影响
    out = ir.normalize_id_value(6091236422.0)
    assert out == "6091236422"


def test_normalize_id_value_raises_above_float64_safe_integer_range():
    # Fix round 2 residual：超过 2**53 的浮点 ID 在到达这里之前，
    # pandas 把整数列上转型为 float64 那一步就已经丢失精度（例如
    # 19 位 id 6234567890123456789 会变成 6234567890123456512），
    # 归一化不应该悄悄返回一个错误的字符串，而必须报错
    too_big = float(2**53) + 1024.0
    with pytest.raises(ValueError):
        ir.normalize_id_value(too_big)


def test_round_trip_float_column_matches_string_id_set_via_isin():
    # 这就是 finding 描述的确切失败场景：user_id 因该日文件里存在缺失值
    # 被上转型为 float64，astype(str) 会产出 "123.0" 从而在 isin 里
    # 匹配不上字符串来源账号 ID 集合。归一化后必须能正确命中。
    df = pd.DataFrame({"user_id": [123.0, 456.0, None]})
    normalized = ir.normalize_id_series(df["user_id"])
    known_sources = {"123", "999"}
    assert list(normalized.isin(known_sources)) == [True, False, False]
