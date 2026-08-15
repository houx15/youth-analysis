"""
用户 ID 归一化：把整数/浮点/字符串等混杂类型的 ID 统一转成规范字符串。

纯函数模块，不做文件 IO，也不依赖 config，方便脱离服务器环境单测（与
text_rules.py 同一思路）。

背景：pandas 读取一列 ID 时，只要该列含有缺失值，原本的整数列会被
向上转型为 float64；如果这时直接 astype(str)，"123" 会变成 "123.0"，
导致后续所有基于字符串的 isin/等值匹配、以及跨表 join 全部静默失效
（不会报错，只是匹配不上）。本模块统一在"归一化为字符串"这一步就去掉
浮点尾巴，两张表（帖子表、转发事件表）都必须用它处理 user_id，才能保证
两表 user_id 的字符串表示完全一致，可以直接 join。

缺失值（None/NaN）统一映射为空字符串 ""，而不是字符串 "nan"：空字符串
不可能等于任何真实 ID，不会被 isin/等值比较误判命中；下游如需判断"该行
ID 缺失"应显式检查 == MISSING_ID。
"""

import numbers

import pandas as pd

MISSING_ID = ""


def normalize_id_value(value):
    """把单个 ID 值归一化为字符串；缺失值返回 MISSING_ID（空字符串）"""
    if pd.isna(value):
        return MISSING_ID
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        # 浮点 ID：多数情况是整数列因缺失值被 pandas 上转型为 float64，
        # 数值上是整数就去掉 ".0" 尾巴，避免变成 "123.0" 这种伪 ID
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def normalize_id_series(series):
    """对整列 ID 做归一化，逐元素复用 normalize_id_value，返回字符串 Series"""
    return series.map(normalize_id_value)
