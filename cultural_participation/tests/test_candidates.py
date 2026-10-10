"""真实失败案例回归：缩写、词形补充、跨标题计数以及缓存重提词。"""

from contextlib import closing
import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from cultural_participation.candidates import candidates
from cultural_participation.pipeline import SCHEMA, export, extract
from cultural_participation.reextract import refresh


class CandidateTests(unittest.TestCase):
    def test_ascii_and_url(self):
        text = "NBA7人阳性 CBA GDP增长 5G iPhone12 2020 123 http://example.com/secret"
        terms = {t for t, _, _ in candidates(text, lambda text: [], 2, ['n', 'v'], [])}
        self.assertTrue({'NBA', 'NBA7', 'CBA', 'GDP', '5G', 'iPhone12', 'iPhone'}.issubset(terms))
        self.assertTrue({'123', '2020', 'http', 'example.com', 'secret'}.isdisjoint(terms))

    def test_phrase_requires_evidence(self):
        extracted = list(candidates("优衣库《三十而已》", lambda text: [('优衣', 'n')], 2, ['n', 'v'], ['优衣库', '华鼎奖']))
        terms = {t for t, _, _ in extracted}
        self.assertEqual(terms, {'优衣', '优衣库', '三十而已'})

    def make_db(self, path, titles):
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript(SCHEMA)
            conn.executemany('INSERT INTO topics(title) VALUES (?)', [(t,) for t in titles])
            conn.execute('INSERT INTO metadata VALUES (?,?)', ('stage', json.dumps('collected')))

    def test_unique_titles_and_export_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / 'topics.sqlite'
            self.make_db(db, ['NBA NBA 优衣库', 'NBA7人阳性'])
            extract(db, tokenizer=lambda text: [('NBA', 'n'), ('优衣', 'n')] if '优衣库' in text else [])
            out = root / 'candidates.csv'
            export(db, out)
            with out.open() as stream:
                rows = {r['term']: r for r in csv.DictReader(stream)}
            self.assertEqual(rows['NBA']['distinct_title_count'], '2')
            self.assertIn('jieba', rows['NBA']['extraction_sources'])
            self.assertIn('latin_or_alphanumeric', rows['NBA']['extraction_sources'])
            self.assertIn('part_of_observed_phrase', rows['优衣']['review_flags'])
            self.assertEqual(json.loads(rows['优衣']['containing_phrases_json']), ['优衣库'])
            self.assertIn('优衣库', rows)

    @unittest.skipUnless(importlib.util.find_spec('jieba'), '需要真实jieba环境')
    def test_real_jieba_and_preserved_source_db(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.sqlite'
            self.make_db(source, ['NBA7人新冠阳性', '5G快速公交智能调度', 'CBA在青岛比赛', 'GDP正增长', '优衣库悄悄涨价', '华鼎奖提名名单'])
            before = source.read_bytes()
            refresh(source, root / 'new')
            self.assertEqual(source.read_bytes(), before)
            with (root / 'new/candidates.csv').open() as stream:
                terms = {r['term'] for r in csv.DictReader(stream)}
            self.assertTrue({'NBA', '5G', 'CBA', 'GDP', '优衣库', '华鼎奖'}.issubset(terms))
            with closing(sqlite3.connect(root / 'new/topics.sqlite')) as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM topics').fetchone()[0], 6)
                self.assertIsNotNone(conn.execute("SELECT value FROM metadata WHERE key='reextraction'").fetchone())
