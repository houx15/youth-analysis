"""用人工构造的小样本验证过滤、去重、可追溯性和分类任务边界。"""

from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cultural_participation.pipeline import collect, export, extract, prepare


def raw_line(titles, board_type="1", time="123", extra=None):
    items = [{"card_type": 4, "desc": title} for title in titles]
    items.extend(extra or [])
    board = {"cards": [{"card_type": 11, "card_group": items}]}
    record = {"type": board_type, "crawler_time_stamp": time, "bangdan": json.dumps(board)}
    return "prefix\t" + json.dumps(record, ensure_ascii=False) + "\n"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "topics.sqlite"
        self.raw = self.root / "weibo_bangdan.2020-01-01"
        self.raw.write_text(
            raw_line(["科技 新品", "科技 新品"], extra=[
                {"card_type": 4, "desc": "广告", "actionlog": {"ext": "ads_word=1"}},
                {"card_type": 4, "desc": ""},
            ])
            + raw_line(["科技 科技 研发"], time="456")
            + raw_line(["应被排除"], board_type="2")
            + "broken\n" + "x\t[]\n", encoding="utf-8"
        )

    def test_counts_provenance_and_no_overwrite(self):
        result = collect(self.root, "weibo_bangdan.*", self.db)
        self.assertEqual(result["unique_titles"], 2)
        self.assertEqual(result["occurrences"], 3)
        self.assertEqual(result["advertisements"], 1)
        self.assertEqual(result["excluded_board_rows"], 1)
        self.assertEqual(result["invalid_rows"], 2)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT DISTINCT crawler_time FROM occurrences ORDER BY crawler_time").fetchall(), [("123",), ("456",)])
            self.assertEqual(conn.execute("SELECT DISTINCT source FROM occurrences").fetchone()[0], str(self.raw.resolve()))
        with self.assertRaises(FileExistsError):
            collect(self.root, "weibo_bangdan.*", self.db)

    def test_global_sample_limit(self):
        (self.root / "weibo_bangdan.2020-01-02").write_text(raw_line(["后一天"]))
        result = collect(self.root, "weibo_bangdan.*", self.db, max_lines=1)
        self.assertEqual(result["lines"], 1)
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["unique_titles"], 1)

    def test_term_frequency_uses_unique_titles_and_export(self):
        collect(self.root, "weibo_bangdan.*", self.db)
        tokenizer = lambda text: [(word, "n") for word in text.split()]
        extract(self.db, tokenizer=tokenizer)
        extract(self.db, tokenizer=tokenizer)  # 重跑不累加
        taxonomy = Path(__file__).resolve().parents[1] / "taxonomy.json"
        output = self.root / "jobs.jsonl"
        candidates = self.root / "candidates.csv"
        export(self.db, candidates, min_titles=2)
        report = json.loads(candidates.with_suffix(".csv.summary.json").read_text())
        self.assertEqual(report["all_candidate_terms"], 3)
        self.assertEqual(report["exported_terms"], 1)
        self.assertEqual(report["retained_terms_by_min_titles"]["2"], 1)
        with candidates.open() as stream:
            self.assertEqual([r["term"] for r in csv.DictReader(stream)], ["科技"])
        result = prepare(candidates, output, taxonomy, batch_size=1)
        self.assertEqual(result["exported_terms"], 1)
        job = json.loads(output.read_text())
        payload = json.loads(job["messages"][1]["content"])
        self.assertEqual(payload["candidates"][0]["term"], "科技")
        self.assertEqual(payload["candidates"][0]["distinct_title_count"], 2)
        self.assertEqual(len(payload["candidates"][0]["examples"]), 2)
        self.assertIn("都不是", payload["taxonomy"]["domains"]["none"])
        self.assertIn('domains=["none"]', job["messages"][0]["content"])
        self.assertIn("无法判断", job["messages"][0]["content"])
        with self.assertRaises(FileExistsError):
            prepare(candidates, output, taxonomy)

    def test_missing_extract_fails_without_output(self):
        collect(self.root, "weibo_bangdan.*", self.db)
        output = self.root / "candidates.csv"
        with self.assertRaises(ValueError):
            export(self.db, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
