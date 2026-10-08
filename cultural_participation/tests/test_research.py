"""行为口径、真实 parquet 分批入口、子抽样和语义测量的小型集成测试。"""

import csv
import json
from pathlib import Path
import tempfile
import sqlite3
import unittest

from cultural_participation.analysis import database, materialize, ols, subsample, summarize
from cultural_participation.behavior import build, build_records
from cultural_participation.semantics import score, survey_check
from cultural_participation.vocabulary import Vocabulary


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vocab = self.root / "vocab.json"
        terms = [{"term": t, "domains": [d], "approved": True, "decision": "standalone"} for t, d in [("足球", "sports"), ("女足", "sports"), ("中国女足", "sports"), ("科技", "technology")]]
        terms.append({"term": "苹果", "domains": ["technology"], "approved": True, "decision": "context_required", "context_any": ["手机"]})
        self.vocab.write_text(json.dumps({"version": "test", "domains": {"sports": "体育", "technology": "科技"}, "terms": terms}))
        self.records = [
            self.row("1", "f", "f", "哈哈//@别人:足球", "科技", True),
            self.row("2", "f", "f", "足球", "", False),
            self.row("3", "m", "m", "转发微博", "中国女足", True),
            self.row("4", "m", "m", "", "", True),
            self.row("5", "empty", "m", "", "", False),
        ]
        self.records.append(dict(self.records[0]))

    def row(self, pid, uid, gender, own, source, rt):
        return {"weibo_id": pid, "user_id": uid, "gender": gender, "weibo_content": own, "r_weibo_content": source,
                "is_retweet": "1" if rt else "0", "time_stamp": 1578000000, "r_time_stamp": 1577999990000 if rt else None}

    def test_context_and_nested_matches(self):
        vocab = Vocabulary(self.vocab)
        self.assertEqual(vocab.match("吃苹果"), [])
        self.assertEqual(vocab.match("苹果手机"), ["苹果"])
        self.assertEqual(set(vocab.match("中国女足")), {"中国女足", "女足"})

    def test_behavior_denominators_and_exact_subsets(self):
        run = self.root / "run"
        manifest = build_records(self.records, self.vocab, run, 2020)
        self.assertEqual(manifest["counts"]["unique_posts"], 5)
        self.assertEqual(manifest["counts"]["duplicate_posts"], 1)
        with database(run / "behavior.sqlite") as conn:
            terms = list(Vocabulary(self.vocab).rules)
            materialize(conn, terms)
            f = conn.execute("SELECT retweet_count,expression_count,retweet_share,expression_share,log_delay FROM user_domain WHERE user_id='f' AND domain='technology'").fetchone()
            self.assertEqual(f[:4], (1, 0, 1, 0))
            self.assertAlmostEqual(f[4], __import__('math').log1p(10))
            self.assertEqual(conn.execute("SELECT retweet_share FROM user_domain WHERE user_id='m' AND domain='sports'").fetchone()[0], .5)
            self.assertEqual(conn.execute("SELECT expression_share,retweet_share FROM user_domain WHERE user_id='empty' AND domain='sports'").fetchone(), (None, None))
            materialize(conn, ["女足"])
            self.assertEqual(conn.execute("SELECT retweet_count FROM user_domain WHERE user_id='m' AND domain='sports'").fetchone()[0], 1)
        summarize(run / "behavior.sqlite", self.root / "summary")
        subsample(run / "behavior.sqlite", self.root / "sample", fractions=[1], repeats=2)
        with (self.root / "sample/sensitivity.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        baseline = [{k: v for k, v in r.items() if k != "repeat"} for r in rows if r["repeat"] == "-1"]
        draw = [{k: v for k, v in r.items() if k != "repeat"} for r in rows if r["repeat"] == "0"]
        self.assertEqual(baseline, draw)

    def test_gender_conflict_and_negative_delay(self):
        records = [self.row("a", "same", "m", "足球", "", False), self.row("b", "same", "f", "科技", "", False), self.row("c", "u", "m", "", "足球", True)]
        records[2]["r_time_stamp"] = records[2]["time_stamp"] + 1
        run = self.root / "conflict"
        build_records(records, self.vocab, run, 2020)
        with database(run / "behavior.sqlite") as conn:
            materialize(conn, list(Vocabulary(self.vocab).rules))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM totals").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT log_delay,delay_n FROM user_domain WHERE domain='sports'").fetchone(), (None, 0))

    def test_actual_parquet_batches(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("需要 pyarrow 验证真实 parquet 接口")
        source = self.root / "source"
        source.mkdir()
        pq.write_table(pa.Table.from_pylist(self.records), source / "fixture.parquet")
        result = build(source, self.vocab, self.root / "parquet_run", batch_size=1)
        self.assertEqual(result["counts"]["unique_posts"], 5)

    def test_semantic_axes_and_missing_anchors(self):
        vectors = self.root / "vectors.vec"
        vectors.write_text("5 2\n尊重 1 0\n贬低 -1 0\n足球 1 1\n科技 1 -1\n无关 0 1\n")
        axes = self.root / "axes.json"
        axes.write_text(json.dumps({"prestige": {"positive": ["尊重"], "negative": ["贬低"]}}))
        objects = self.root / "objects.json"
        objects.write_text(json.dumps([{"object_id": "sports", "domain": "sports", "terms": ["足球", "未登录词"]}]))
        score(vectors, axes, objects, self.root / "scores")
        with (self.root / "scores/scores.csv").open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row["coverage"]), .5)
        self.assertAlmostEqual(float(row["score"]), 2 ** -.5)
        ratings = self.root / "ratings.csv"
        ratings.write_text("object_id,axis,rating\nsports,prestige,3\nsports,prestige,5\n")
        survey_check(self.root / "scores/scores.csv", ratings, self.root / "validation")
        axes.write_text(json.dumps({"prestige": {"positive": ["缺失"], "negative": ["贬低"]}}))
        with self.assertRaises(ValueError):
            score(vectors, axes, objects, self.root / "bad_scores")

    def test_streaming_ols_matches_direct_hc1(self):
        import numpy as np

        rng = np.random.default_rng(18)
        n = 80
        gender = np.array([i % 2 for i in range(n)])
        activities = rng.integers(1, 100, size=(n, 3))
        x = np.column_stack([np.ones(n), gender, np.log1p(activities)])
        y = x @ np.array([1, .3, .1, -.05, .2]) + rng.normal(0, .1, n)
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE fixture(y REAL,gender TEXT,posts REAL,retweets REAL,days REAL)")
        conn.executemany("INSERT INTO fixture VALUES (?,?,?,?,?)", [(float(y[i]), "f" if gender[i] else "m", *map(float, activities[i])) for i in range(n)])
        result = ols(conn, "SELECT * FROM fixture", (), True)
        inv = np.linalg.inv(x.T @ x)
        beta = inv @ x.T @ y
        residuals = y - x @ beta
        covariance = inv @ ((x * residuals[:, None]).T @ (x * residuals[:, None])) @ inv * n / (n - 5)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["estimate_f_minus_m"], beta[1], places=10)
        self.assertAlmostEqual(result["se"], covariance[1, 1] ** .5, places=10)
