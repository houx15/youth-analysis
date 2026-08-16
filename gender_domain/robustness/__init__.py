"""
§13 稳健性套件（robustness suite）。

每一个 variant（vocabulary.py / accounts.py / samples.py / measures.py /
context_sample.py……）都只负责产出一份改动过的用户级宽表（改样本、改
分母、改度量口径），再把它交给 harness.estimate_all 去拟合——本包存在
的唯一理由，就是保证"改动样本"和"改动在估什么"这两件事被结构性地分开，
不给任何一个 variant 留下"顺手换一个估计量"的空间。
"""
