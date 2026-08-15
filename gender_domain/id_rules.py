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

精度边界：整数列一旦因缺失值被 pandas 上转型为 float64，转型本身就已经
发生在本模块之前——float64 只能精确表示到 2**53（约 9.0e15），超过这个
量级的整数在上转型那一步就已经丢失精度（例如 6234567890123456789 会变成
6234567890123456512），本模块拿到手的浮点数已经是错的，去掉 ".0" 尾巴
也救不回来。所以这里不去猜测、不静默返回一个可能错误的字符串，而是对
超过 2**53 的浮点值直接抛 ValueError。2020 年微博 uid 在个位数十亿量级
（约 10 位数字），远低于这个边界，属于安全区；这个检查只是给"万一超过"
的情况一个明确报错，而不是静默产出错误 ID。
"""

import numbers

import pandas as pd

MISSING_ID = ""
# float64 只能精确表示到 2**53；超过这个量级的整数在 pandas 把列上转型为
# float64 时就已经丢失精度，此模块无法恢复，只能拒绝而不是猜测。
MAX_SAFE_FLOAT_ID = 2 ** 53


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
        if abs(value) > MAX_SAFE_FLOAT_ID:
            # 超出 float64 精确整数范围：上转型那一步就已经丢失精度，
            # 这里拿到的浮点数本身可能已经不是原始 ID，宁可报错也不猜
            raise ValueError(
                f"ID 值 {value!r} 超过 float64 精确整数范围（2**53），"
                "该值以浮点形式到达此函数时精度可能已经丢失，无法安全归一化"
            )
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def normalize_id_series(series):
    """对整列 ID 做归一化，逐元素复用 normalize_id_value，返回字符串 Series"""
    return series.map(normalize_id_value)
