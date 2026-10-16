"""联结词项行为与语义坐标；只接受同版本词表的真实词项汇总。

用法：python -m cultural_participation.behavior_semantic_link --behavior DIR --date YYYYMMDD
输入为解压后的四文件汇总目录。输出日期前缀CSV、PDF及方法说明。
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from cultural_participation.semantic_report import AXES, DOMAINS, MODELS, M_LABELS
from cultural_participation.term_semantics import correlation

METRICS = ["retweet_entry", "expression_entry", "retweet_share", "expression_share", "log_delay", "comment_on_retweet_share"]


def validate_join(semantics, gaps, coverage, semantic_manifest, behavior_manifest):
    """先验证词表哈希和键，禁止按领域扩充词项系数。"""
    expected = semantic_manifest["input"]["signature"]["vocabulary_sha256"]
    if behavior_manifest.get("status") != "complete" or behavior_manifest["source_signature"]["vocabulary_sha256"] != expected:
        raise ValueError("行为与语义的词表版本不一致或行为未完成")
    if semantics.term.duplicated().any() or gaps.duplicated(["term", "metric", "model"]).any():
        raise ValueError("词项或行为键重复")
    if coverage.duplicated(["term", "gender"]).any():
        raise ValueError("行为覆盖键重复")
    universe = set(semantics.term)
    if set(gaps.term) != universe or set(coverage.term) != universe:
        raise ValueError("词项集合不完整或含词表外词")
    if set(gaps.metric) != set(METRICS) or set(gaps.model) != {"unadjusted_OLS", "activity_adjusted_OLS"}:
        raise ValueError("缺少预期行为指标或模型")
    if len(gaps) != len(universe) * len(METRICS) * 2:
        raise ValueError("行为指标与模型组合不完整")
    for _, group in gaps.groupby(["metric", "model"]):
        if set(group.term) != universe:
            raise ValueError("某行为指标缺少词项审计行")
    if set(coverage.gender) != {"m", "f"} or len(coverage) != 2 * len(universe):
        raise ValueError("缺少分性别覆盖数据")
    counts = coverage.groupby("term")[["retweets", "expressions", "retweet_users", "expression_users"]].sum()
    # status在两类表中有不同含义，必须重命名，避免覆盖地位轴。
    behavior = gaps.rename(columns={"model": "behavior_model", "status": "estimate_status"})
    joined = behavior.merge(semantics, on="term", validate="many_to_one").merge(counts, on="term", validate="many_to_one")
    joined["estimable"] = joined.estimate_status.eq("ok") & np.isfinite(joined.estimate_f_minus_m)
    return joined


def association_table(joined):
    rows = []
    for (metric, model), base in joined.groupby(["metric", "behavior_model"]):
        count = "expression_users" if metric.startswith("expression") else "retweet_users"
        variants = {
            "all_covered_estimable": base.covered & base.estimable,
            "at_least_20_participants": base.covered & base.estimable & base[count].ge(20),
            "at_least_100_participants": base.covered & base.estimable & base[count].ge(100),
            "drop_known_person": base.covered & base.estimable & ~base.known_person,
            "drop_gender_marked": base.covered & base.estimable & ~base.gender_marked,
        }
        for variant, mask in variants.items():
            sample = base[mask]
            for scope in ["all", *DOMAINS]:
                sub = sample if scope == "all" else sample.loc[sample.domains.str.split("|").map(lambda d: scope in d).astype(bool)]
                for axis in AXES:
                    valid = sub[np.isfinite(sub[axis])]
                    rows.append(dict(metric=metric, behavior_model=model, variant=variant, scope=scope, axis=axis,
                                     n_terms=len(valid), pearson=correlation(valid.estimate_f_minus_m, valid[axis]),
                                     spearman=correlation(valid.estimate_f_minus_m, valid[axis], True)))
    return pd.DataFrame(rows)


def run(behavior, semantics, output, date):
    source, semantic_root, target = Path(behavior), Path(semantics), Path(output)
    manifest = json.loads((source / "manifest.json").read_text())
    gaps = pd.read_csv(source / "term_gender_gaps.csv")
    coverage = pd.read_csv(source / "term_coverage.csv")
    target.mkdir(parents=True, exist_ok=True)
    provenance = {"behavior": manifest, "inputs_sha256": {}}
    with PdfPages(target / f"{date}_behavior_semantic_figures.pdf") as pdf:
        for model, label in zip(MODELS, M_LABELS):
            term_file = semantic_root / f"{model}_term_scores.csv"
            semantic_manifest = json.loads((semantic_root / f"{model}_manifest.json").read_text())
            joined = validate_join(pd.read_csv(term_file), gaps, coverage, semantic_manifest, manifest)
            joined.to_csv(target / f"{date}_{model}_behavior_semantic_terms.csv", index=False)
            association_table(joined).to_csv(target / f"{date}_{model}_behavior_semantic_associations.csv", index=False)
            for metric in ["retweet_share", "expression_share", "log_delay", "comment_on_retweet_share"]:
                fig, axes = plt.subplots(2, 3, figsize=(12, 7))
                for row, behavior_model in enumerate(["unadjusted_OLS", "activity_adjusted_OLS"]):
                    sub = joined[(joined.metric == metric) & (joined.behavior_model == behavior_model) & joined.covered & joined.estimable]
                    for col, axis in enumerate(["gender", "prestige", "status"]):
                        ax = axes[row, col]
                        ax.scatter(sub[axis], sub.estimate_f_minus_m, s=7, alpha=.2, color="#555555", rasterized=True)
                        ax.axhline(0, linewidth=.7, color="#999999")
                        ax.set_xlabel(axis); ax.set_ylabel("Female minus male")
                        ax.set_title(f"{behavior_model}\nn={len(sub)}; rank r={correlation(sub[axis], sub.estimate_f_minus_m, True):.3f}", fontsize=8)
                        ax.grid(alpha=.2, linewidth=.5)
                fig.suptitle(f"{label} | {metric}")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
            provenance[model] = semantic_manifest
            provenance["inputs_sha256"][str(term_file)] = hashlib.sha256(term_file.read_bytes()).hexdigest()
    for name in ["term_gender_gaps.csv", "term_coverage.csv", "manifest.json"]:
        provenance["inputs_sha256"][str(source / name)] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    (target / f"{date}_behavior_semantic_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    (target / f"{date}_behavior_semantic_methods.md").write_text("""# 词项行为与语义联结说明

每词只与自身行为系数联结；正系数为女性减男性。保留未调整和活动量调整结果，条件性指标遵循原行为分母。全部词项保存在审计表，图只显示有向量且行为可估计的词。

相关为词项等权描述，不是因果效应或独立样本显著性检验。同一批用户参与多个词，系数存在相关误差；本模块不报告跨词相关的常规独立样本p值。各模型覆盖不同，跨模型数值差异需同时参考覆盖与共同样本诊断。

参与用户数20和100仅为透明的稀疏度敏感性阈值，不按显著性筛词；时间／附言指标的参与人数不等于有效时间／附言样本量，应同时检查n_users。分别报告总体与领域内关系，多领域词可进入多个领域，但总体只计一次。删除人物只针对已有标签，不能解释为排除所有人物。尊崇与地位仍为候选语义量表。
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--semantics", default="cultural_participation/reports/20261016_semantic_tables")
    parser.add_argument("--output", default="cultural_participation/reports")
    parser.add_argument("--date", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
