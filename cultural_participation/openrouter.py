"""独立调用 OpenRouter，逐批保存原始响应并校验；不自动重试付费请求。"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import quote
from uuid import uuid4

from tqdm import tqdm


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DECISIONS = ["standalone", "context_required", "exclude"]


def response_format(domains):
    """领域取值动态读取同一任务中的分类体系。"""
    fields = {
        "term": {"type": "string"},
        "domains": {"type": "array", "items": {"type": "string", "enum": list(domains)}},
        "decision": {"type": "string", "enum": DECISIONS},
        "reason": {"type": "string"},
        "suggested_domain": {"type": "string"},
    }
    item = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    schema = {"type": "object", "properties": {"results": {"type": "array", "items": item}},
              "required": ["results"], "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": "vocabulary_labels", "strict": True, "schema": schema}}


def validate_response(response, candidates, domains):
    """严格检查遗漏、重复、额外词、非法标签及 none 互斥。"""
    if response.get("error"):
        raise ValueError("服务返回错误对象")
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("模型未正常完成，可能被截断或拒绝")
        payload = json.loads(choice["message"]["content"])
        if set(payload) != {"results"} or not isinstance(payload["results"], list):
            raise ValueError("响应必须含 results 数组")
        rows = payload["results"]
        for row in rows:
            if set(row) != {"term", "domains", "decision", "reason", "suggested_domain"}:
                raise ValueError("结果字段不符合约定")
            if not all(isinstance(row[k], str) for k in ("term", "decision", "reason", "suggested_domain")):
                raise ValueError("文本字段类型错误")
            labels = row["domains"]
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                raise ValueError("领域必须为字符串列表")
            if len(labels) != len(set(labels)) or not set(labels).issubset(domains):
                raise ValueError("领域重复或不在分类体系内")
            if "none" in labels and len(labels) != 1:
                raise ValueError("都不是不得与其他领域并选")
            if row["decision"] not in DECISIONS:
                raise ValueError("适用性标签非法")
        expected = [c["term"] for c in candidates]
        if len(expected) != len(set(expected)):
            raise ValueError("任务含重复候选词")
        if Counter(r["term"] for r in rows) != Counter(expected):
            raise ValueError("模型遗漏、重复或新增了候选词")
        return rows
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("响应结构异常") from exc


def request_completion(body, api_key):
    request = Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            return json.load(response)
    except HTTPError as exc:
        raise ValueError(f"OpenRouter HTTP {exc.code}；本次未自动重试") from None
    except (URLError, TimeoutError) as exc:
        raise ValueError("OpenRouter 网络失败；计费状态可能未知，本次未自动重试") from None


def classify(jobs, output, model, max_batches, max_tokens=6000):
    """单次单模型；按完整模型 ID／UTC 日期／唯一运行保存，不自动重试。"""
    if not model.strip() or model in {".", ".."}:
        raise ValueError("需要有效模型 ID")
    if max_batches < 1 or max_tokens < 1:
        raise ValueError("max_batches 和 max_tokens 必须为正数")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("请在运行环境设置 OPENROUTER_API_KEY")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%H%M%S") + "_" + uuid4().hex[:8]
    target = Path(output) / quote(model, safe="") / now.strftime("%Y-%m-%d") / run_id
    counts = Counter()
    with Path(jobs).open(encoding="utf-8") as source:
        total = min(max_batches, sum(1 for _ in source))
    with Path(jobs).open(encoding="utf-8") as source:
        target.mkdir(parents=True, exist_ok=False)
        manifest = {"model": model, "started_at_utc": now.isoformat(), "jobs": str(Path(jobs).resolve()),
                    "max_batches": max_batches, "max_tokens": max_tokens, "endpoint": ENDPOINT,
                    "reasoning": {"enabled": False}}
        (target / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tqdm.write(f"结果目录：{target}")
        # 单词任务每批就是一个词；保留“批”单位以兼容旧的多词任务文件。
        with (target / "responses.jsonl").open("x", encoding="utf-8") as stream, tqdm(
            total=total, desc=model, unit="批", dynamic_ncols=True
        ) as progress:
            for batch_number, line in enumerate(source, 1):
                if batch_number > max_batches:
                    break
                job = json.loads(line)
                payload = json.loads(job["messages"][1]["content"])
                domains = payload["taxonomy"]["domains"]
                if "none" not in domains:
                    raise ValueError("分类体系缺少都不是（none）选项")
                task_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                body = {"model": model, "messages": job["messages"], "max_tokens": max_tokens,
                        "reasoning": {"enabled": False},
                        "response_format": response_format(domains), "provider": {"require_parameters": True}}
                record = {"task_sha256": task_hash, "batch": batch_number, "requested_model": model,
                          "request": body, "job": job, "status": "error"}
                try:
                    response = request_completion(body, api_key)
                    record["response"] = response
                    record["results"] = validate_response(response, payload["candidates"], domains)
                    record["status"] = "ok"
                except ValueError as exc:
                    record["error"] = str(exc)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                counts[record["status"]] += 1
                progress.set_postfix(ok=counts["ok"], error=counts["error"], refresh=False)
                progress.update(1)
                if record["status"] == "error":
                    tqdm.write(f"第 {batch_number} 批失败：{record['error']}")
    return {"output": str(target), "requests": dict(counts)}


def compare(responses, output):
    """读取两次单模型结果，按同任务同词比较；不对分歧自动裁决。"""
    if len(responses) != 2:
        raise ValueError("需要两个 responses.jsonl 路径")
    observations = []
    models = []
    batch_counts = []
    for path in responses:
        rows = {}
        tasks = set()
        model = None
        counts = Counter()
        # 此处只读取词表分类结果，不读取微博原始语料。
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if model is not None and model != record["requested_model"]:
                    raise ValueError("每份文件必须仅包含一个模型")
                model = record["requested_model"]
                task = record["task_sha256"]
                if task in tasks:
                    raise ValueError("同任务重复，需先确定使用哪次结果")
                tasks.add(task)
                counts[record["status"]] += 1
                if record["status"] == "ok":
                    for row in record["results"]:
                        rows[(task, row["term"])] = row
        if model is None:
            raise ValueError("结果文件为空")
        models.append(model)
        observations.append(rows)
        batch_counts.append(dict(counts))
    left, right = observations
    common = sorted(left.keys() & right.keys())
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    domain_agree = decision_agree = none_conflicts = uncertain_conflicts = 0
    with (target / "disagreements.jsonl").open("x", encoding="utf-8") as stream:
        for key in common:
            a, b = left[key], right[key]
            same_domain = set(a["domains"]) == set(b["domains"])
            same_decision = a["decision"] == b["decision"]
            domain_agree += same_domain
            decision_agree += same_decision
            none_conflicts += ("none" in a["domains"]) != ("none" in b["domains"])
            uncertain_conflicts += (not a["domains"]) != (not b["domains"])
            if not (same_domain and same_decision):
                stream.write(json.dumps({"task_sha256": key[0], "term": key[1], "left": a, "right": b}, ensure_ascii=False) + "\n")
    report = {"models": models, "paired_terms": len(common),
              "valid_terms_by_run": [len(rows) for rows in observations],
              "batch_status_by_run": batch_counts,
              "response_files": [str(Path(p).resolve()) for p in responses],
              "unpaired_valid_terms": len(left.keys() ^ right.keys()),
              "domain_exact_agreement": domain_agree / len(common) if common else None,
              "decision_agreement": decision_agree / len(common) if common else None,
              "none_conflicts": none_conflicts, "uncertain_conflicts": uncertain_conflicts,
              "note": "一致率只基于同任务双方有效的词条；模型间一致不等于分类正确，也不是独立人工验证。"}
    (target / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
