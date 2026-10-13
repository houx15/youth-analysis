"""合并独立可用的裁决词与双模型一致词，导出可追溯的严格词表。

使用方法：python -m cultural_participation.finalize_vocabulary --help
只读取缩减后的分类表，不读取原始微博或大模型完整响应。
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


DOMAINS = {
    "politics": "政治与治理",
    "business": "经济与商业",
    "technology": "科学与技术",
    "sports": "体育（含电竞与大众运动）",
    "arts": "文艺与娱乐",
}


def labels(value):
    """领域组合不受原始标签顺序影响。"""
    return set(value.split("|"))


def finalize(paired_path, reviewed_path):
    """裁决优先；未裁决词必须领域完全一致且双方都允许独立使用。"""
    reviews = {}
    with Path(reviewed_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["term"] in reviews:
                raise ValueError(f"重复裁决词：{row['term']}")
            reviews[row["term"]] = row
    selected = []
    seen = set()
    counts = Counter()
    with Path(paired_path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            term = row["term"]
            if term in seen:
                raise ValueError(f"重复配对词：{term}")
            seen.add(term)
            review = reviews.get(term)
            ds, qw = labels(row["ds_domains"]), labels(row["qw_domains"])
            if review is not None:
                domains = set(review["final_domains"])
                keep = review["final_decision"] == "standalone"
                source = "adjudicated_standalone"
                reason = review["adjudication_reason"]
            else:
                domains = ds
                keep = (
                    ds == qw and bool(ds) and ds <= DOMAINS.keys()
                    and row["ds_decision"] == row["qw_decision"] == "standalone"
                )
                source = "dual_model_standalone_agreement"
                reason = "双方目标领域组合完全一致，且均判为standalone；本轮合并保留，未新增逐词裁决。"
            if not keep:
                counts["excluded_from_paired"] += 1
                continue
            if not domains or not domains <= DOMAINS.keys():
                raise ValueError(f"保留词领域无效：{term}")
            counts[source] += 1
            selected.append({
                "term": term,
                "domains": sorted(domains),
                "decision": "standalone",
                "approved": True,
                "distinct_title_count": int(row["freq"]),
                "selection_source": source,
                "term_type": review.get("term_type", "not_separately_typed") if review else "not_separately_typed",
                "participation_basis": review.get("participation_basis", "term_semantics") if review else "not_separately_typed",
                "scope_id": review["scope_id"] if review else "",
                "selection_reason": reason,
            })
    if set(reviews) - seen:
        raise ValueError("裁决表含配对表中不存在的词")
    selected.sort(key=lambda r: (-r["distinct_title_count"], r["term"]))
    return selected, dict(counts)


def export(paired_path, reviewed_path, output_dir, version):
    """保存词表、逐词来源和描述统计；approved表示采用规则获准纳入，不表示人工验证。"""
    rows, counts = finalize(paired_path, reviewed_path)
    if not rows:
        raise ValueError("最终词表为空")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": version,
        "unique_terms": len(rows),
        "selection_counts": counts,
        "domain_counts": {d: sum(d in r["domains"] for r in rows) for d in DOMAINS},
        "multi_domain_terms": sum(len(r["domains"]) > 1 for r in rows),
        "domain_assignments": sum(len(r["domains"]) for r in rows),
        "explicitly_tagged_person_terms": sum(r["term_type"] == "person" for r in rows),
        "frequency_bins": dict(Counter(
            "1" if r["distinct_title_count"] == 1 else
            "2-4" if r["distinct_title_count"] < 5 else
            "5-9" if r["distinct_title_count"] < 10 else
            "10-49" if r["distinct_title_count"] < 50 else "50+"
            for r in rows
        )),
        "top_terms_by_domain": {d: [
            {"term": r["term"], "distinct_title_count": r["distinct_title_count"]}
            for r in rows if d in r["domains"]
        ][:10] for d in DOMAINS},
        "input_sha256": {},
        "frequency_definition": "Containing distinct hot-search titles; not Weibo token counts, users, reposts, or ranking appearances.",
        "person_tag_caveat": "Only adjudicated explicit person tags are counted; consensus terms have not been exhaustively typed.",
    }
    for name, path in [("paired_terms", paired_path), ("reviewed_terms", reviewed_path)]:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        summary["input_sha256"][name] = digest.hexdigest()
    payload = {
        "version": version,
        "domains": DOMAINS,
        "selection_policy": "Adjudicated standalone overrides original models; otherwise exact target-domain agreement and both standalone. No context_required or exclude terms.",
        "approval_basis": "User-authorized inclusion policy; model agreement and model adjudication are not independent human validation.",
        "terms": rows,
    }
    for name, data in [("vocabulary.json", payload), ("summary.json", summary)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "vocabulary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "domains": "|".join(row["domains"])})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.paired, args.reviewed, args.output_dir, args.version), ensure_ascii=False, indent=2))
