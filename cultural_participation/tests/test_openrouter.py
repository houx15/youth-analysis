"""不联网验证模型响应校验、单模型存储隔离和跨运行一致性。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cultural_participation.openrouter import classify, compare, validate_response


def response(labels):
    rows = [{"term": term, "domains": domains, "decision": "standalone", "reason": "示例理由", "suggested_domain": ""} for term, domains in labels]
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"results": rows})}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10}}


class OpenRouterTests(unittest.TestCase):
    def test_none_mutual_exclusion_missing_and_truncation(self):
        candidates = [{"term": "足球"}]
        self.assertEqual(validate_response(response([("足球", ["none"])]), candidates, {"sports", "none"})[0]["domains"], ["none"])
        self.assertEqual(validate_response(response([("足球", [])]), candidates, {"sports", "none"})[0]["domains"], [])
        for bad in [response([("足球", ["none", "sports"])]), response([]), response([("足球", ["sports"]), ("足球", ["sports"])])]:
            with self.assertRaises(ValueError):
                validate_response(bad, candidates, {"sports", "none"})
        bad = response([("足球", ["sports"])])
        bad["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            validate_response(bad, candidates, {"sports", "none"})

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-not-a-real-key"})
    @patch("cultural_participation.openrouter.request_completion")
    def test_single_model_storage_and_comparison(self, call):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"taxonomy": {"domains": {"sports": "体育", "none": "都不是"}}, "candidates": [{"term": "足球"}, {"term": "未知"}]}
            job = {"messages": [{"role": "system", "content": "测试"}, {"role": "user", "content": json.dumps(payload)}]}
            jobs = root / "jobs.jsonl"
            jobs.write_text(json.dumps(job) + "\n" + json.dumps(job) + "\n")
            call.side_effect = [response([("足球", ["sports"]), ("未知", ["none"])]), response([("足球", ["sports"]), ("未知", [])])]
            a = classify(jobs, root / "out", "test/model-a", max_batches=1)
            b = classify(jobs, root / "out", "test/model-b", max_batches=1)
            self.assertEqual(call.call_count, 2)
            for request in call.call_args_list:
                self.assertEqual(request.args[0]["reasoning"], {"enabled": False})
            manifest = json.loads((Path(a["output"]) / "run.json").read_text())
            self.assertEqual(manifest["reasoning"], {"enabled": False})
            self.assertEqual(Path(a["output"]).parts[-3], "test%2Fmodel-a")
            self.assertNotEqual(a["output"], b["output"])
            files = [Path(run["output"]) / "responses.jsonl" for run in [a, b]]
            result = compare(files, root / "comparison")
            self.assertEqual(result["paired_terms"], 2)
            self.assertEqual(result["domain_exact_agreement"], .5)
            self.assertEqual(result["decision_agreement"], 1)
            self.assertEqual(result["none_conflicts"], 1)
            self.assertEqual(result["uncertain_conflicts"], 1)
            self.assertNotIn("test-not-a-real-key", files[0].read_text())

    def test_disjoint_tasks_not_reported_as_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for i in range(2):
                path = root / f"{i}.jsonl"
                row = {"task_sha256": str(i), "requested_model": str(i), "status": "ok", "results": [{"term": "同词", "domains": ["none"], "decision": "exclude"}]}
                path.write_text(json.dumps(row) + "\n")
                files.append(path)
            result = compare(files, root / "comparison")
            self.assertEqual(result["paired_terms"], 0)
            self.assertIsNone(result["domain_exact_agreement"])
            self.assertEqual(result["unpaired_valid_terms"], 2)
