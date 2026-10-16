"""从正式词表生成互斥全量主版本与各领域高频250词稳健性版本。

频率为包含该词的独立热搜标题数；并列按词项Unicode顺序排序。
用法：python -m cultural_participation.derive_vocabularies
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from cultural_participation.vocabulary import Vocabulary


def derive(source, output_root, top_n=250):
    if top_n < 1:
        raise ValueError("top_n必须为正整数")
    original = Vocabulary(source).data
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    excluded = [r for r in original["terms"] if len(set(r["domains"])) > 1]
    retained = [r for r in original["terms"] if len(set(r["domains"])) == 1]
    selected = []
    for domain in original["domains"]:
        rows = [r for r in retained if r["domains"] == [domain]]
        if len(rows) < top_n:
            raise ValueError(f"{domain}只有{len(rows)}词，无法选择{top_n}词")
        if any(not isinstance(r.get("distinct_title_count"), int) or r["distinct_title_count"] < 0 for r in rows):
            raise ValueError("频率缺失或无效")
        selected.extend(sorted(rows, key=lambda r: (-r["distinct_title_count"], r["term"]))[:top_n])
    results = {}
    for name, rows, role in [("exclusive_full_v1", retained, "primary"), (f"exclusive_top{top_n}_v1", selected, "vocabulary-size robustness")]:
        target = Path(output_root) / name
        if target.exists():
            raise FileExistsError(f"新版本目录已存在，保留原文件：{target}")
    for name, rows, role in [("exclusive_full_v1", retained, "primary"), (f"exclusive_top{top_n}_v1", selected, "vocabulary-size robustness")]:
        target = Path(output_root) / name
        target.mkdir(parents=True)
        provenance = {
            "source": str(source), "source_version": original["version"], "source_sha256": source_hash,
            "analysis_role": role, "cross_domain_policy": "Remove every multi-domain term; no reassignment.",
            "frequency_definition": "Number of distinct hot-search titles containing a term; not Weibo occurrence frequency.",
            "selection_order": "descending distinct_title_count, then ascending Unicode term",
            "top_n_per_domain": top_n if role != "primary" else None,
        }
        payload = {**original, "version": name, "selection_policy": "Retain only originally single-domain terms." + (f" Then select top {top_n} per domain by distinct_title_count." if role != "primary" else " Retain all such terms."), "derivation": provenance, "terms": rows}
        (target / "vocabulary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        with (target / "vocabulary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({**r, "domains": "|".join(r["domains"])} for r in rows)
        summary = {**provenance, "terms": len(rows), "domain_counts": dict(Counter(r["domains"][0] for r in rows)),
                   "excluded_multi_domain_terms": len(excluded),
                   "excluded_multi_domain": [{"term": r["term"], "domains": r["domains"]} for r in excluded]}
        (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        Vocabulary(target / "vocabulary.json")
        results[name] = summary["domain_counts"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="cultural_participation/vocabularies/strict_v1/vocabulary.json")
    parser.add_argument("--output-root", default="cultural_participation/vocabularies")
    parser.add_argument("--top-n", type=int, default=250)
    print(json.dumps(derive(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
