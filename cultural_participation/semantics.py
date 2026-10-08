"""可配置语义轴与文化对象测量；不预设英文词轴直接适用于中文。"""

import csv
import hashlib
import json
from pathlib import Path


def score(vectors, axes, objects, output):
    """流式扫描 word2vec 文本向量，仅保留所需锚词和对象词，不训练模型。"""
    import numpy as np

    axes_data = json.loads(Path(axes).read_text())
    objects_data = json.loads(Path(objects).read_text())
    if not axes_data or not objects_data:
        raise ValueError("语义轴和对象集合不能为空")
    needed = set()
    for name, axis in axes_data.items():
        if not axis.get("positive") or not axis.get("negative"):
            raise ValueError(f"轴 {name} 需要正负两组锚词")
        if set(axis["positive"]) & set(axis["negative"]):
            raise ValueError("正负锚词不能重合")
        needed.update(axis["positive"] + axis["negative"])
    for obj in objects_data:
        if not obj.get("object_id") or not obj.get("domain") or not obj.get("terms"):
            raise ValueError("对象需要 object_id、domain、terms")
        needed.update(obj["terms"])
    if len({o["object_id"] for o in objects_data}) != len(objects_data):
        raise ValueError("object_id 重复")
    selected = {}
    digest = hashlib.sha256()
    dimension = None
    with Path(vectors).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            parts = raw.decode("utf-8").split()
            if line_number == 1 and len(parts) == 2 and all(p.isdigit() for p in parts):
                dimension = int(parts[1])
                continue
            if not parts or parts[0] not in needed:
                continue
            vector = np.array([float(x) for x in parts[1:]])
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                raise ValueError(f"向量格式错误，行 {line_number}")
            if parts[0] in selected:
                raise ValueError("向量文件包含重复目标词")
            selected[parts[0]] = vector / np.linalg.norm(vector)
    axis_vectors = {}
    for name, axis in axes_data.items():
        absent = set(axis["positive"] + axis["negative"]) - selected.keys()
        if absent:
            raise ValueError(f"轴 {name} 缺少锚词：{sorted(absent)}；请修订轴文件，不自动删锚词")
        v = np.mean([selected[t] for t in axis["positive"]], axis=0) - np.mean([selected[t] for t in axis["negative"]], axis=0)
        if np.linalg.norm(v) < 1e-12:
            raise ValueError(f"轴 {name} 退化")
        axis_vectors[name] = v / np.linalg.norm(v)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    with (target / "scores.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object_id", "domain", "axis", "score", "total_terms", "matched_terms", "coverage", "missing_terms"])
        for obj in objects_data:
            terms = sorted(set(obj["terms"]))
            found = [t for t in terms if t in selected]
            for name, axis in axis_vectors.items():
                value = float(np.mean([selected[t] @ axis for t in found])) if found else None
                writer.writerow([obj["object_id"], obj["domain"], name, value, len(terms), len(found), len(found) / len(terms), json.dumps(sorted(set(terms) - selected.keys()), ensure_ascii=False)])
    metadata = {"vectors": str(Path(vectors).resolve()), "vectors_sha256": digest.hexdigest(), "axes": axes_data, "objects": objects_data,
                "method": "normalized words; normalized difference of anchor centroids; object mean cosine; no OOV imputation",
                "interpretation": "corpus-specific associations, not automatic social consensus or causal devaluation"}
    (target / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(target), "matched_vocabulary": len(selected)}


def survey_check(scores, ratings, output):
    """外部调查逐对象平均评价与 embedding 对照；需同名轴、同一对象的真实评分。"""
    import numpy as np

    semantic = {}
    with Path(scores).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["object_id"], row["axis"])
            if key in semantic:
                raise ValueError("语义得分存在重复 object_id/axis")
            if row["score"]:
                semantic[key] = float(row["score"])
    totals = {}
    with Path(ratings).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row["rating"].strip():
                continue
            value = float(row["rating"])
            if not np.isfinite(value):
                raise ValueError("调查评分必须为有限数值")
            key = (row["object_id"], row["axis"])
            total, n = totals.get(key, (0, 0))
            totals[key] = (total + value, n + 1)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    paired = {}
    with (target / "object_ratings.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object_id", "axis", "embedding_score", "survey_mean", "n_ratings"])
        for key in sorted(totals):
            total, n = totals[key]
            writer.writerow([*key, semantic.get(key), total / n, n])
            if key in semantic:
                paired.setdefault(key[1], []).append((semantic[key], total / n))
    report = {}
    for axis, pairs in paired.items():
        values = np.array(pairs)
        r = float(np.corrcoef(values.T)[0, 1]) if len(pairs) >= 3 and np.all(values.std(axis=0) > 0) else None
        report[axis] = {"n_objects": len(pairs), "pearson_r": r}
    (target / "summary.json").write_text(json.dumps({"axes": report, "note": "unweighted object means, descriptive correlation, no representative-population or historical-validity claim"}, indent=2), encoding="utf-8")
    return {"output": str(target)}
