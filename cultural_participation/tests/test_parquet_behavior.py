"""分月Parquet与原行为口径的合成数据回归测试。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cultural_participation.analysis import database, materialize, summarize as old_summary, METRICS
from cultural_participation.behavior import build_records
from cultural_participation.parquet_behavior import FastMatcher, aggregate, measure_frame, month, reduce_bucket, buckets
from cultural_participation.parquet_summary import summarize
from cultural_participation.tests import test_research


class ParquetTests(unittest.TestCase):
    setUp = test_research.ResearchTests.setUp
    row = test_research.ResearchTests.row

    def test_matcher(self):
        matcher = FastMatcher(self.vocab)
        for text in ['中国女足足球', '苹果', '苹果手机', '', '足球足球科技']:
            self.assertEqual(json.loads(matcher.match(text)[1]), matcher.vocab.match(text))
        matcher = FastMatcher(Path(__file__).parents[1] / 'vocabularies/strict_v1/vocabulary.json')
        for text in ['RAISE WNBA NBA AI', '中国女足', '讨论AI技术']:
            self.assertEqual(json.loads(matcher.match(text)[1]), matcher.vocab.match(text))
        self.assertEqual(buckets(pd.Series(['a','b']), 64).tolist(), [buckets(pd.Series([v]),64)[0] for v in ['a','b']])

    def test_frame_equivalence(self):
        records = self.records + [self.row('c','conflict','m','科技','',False), self.row('c','conflict','f','科技','',False)]
        records += [dict(self.records[0], weibo_id='invalid', time_stamp=None), dict(self.records[0], weibo_id='old',time_stamp=1)]
        fast = FastMatcher(self.vocab)
        frame, _ = measure_frame(pd.DataFrame(records), fast, 2020)
        users, stats = aggregate(frame, fast.domains)
        self.assertNotIn('conflict', set(users.user_id))
        build_records(records, self.vocab, self.root/'old', 2020)
        with database(self.root/'old/behavior.sqlite') as conn:
            materialize(conn, list(fast.vocab.rules))
            for uid,d,*values in conn.execute('SELECT user_id,domain,'+','.join(METRICS)+' FROM user_domain'):
                actual = users.set_index('user_id').loc[uid]
                np.testing.assert_allclose([actual[f'{d}_{m}'] for m in METRICS], [np.nan if v is None else v for v in values], equal_nan=True)
        empty, _ = measure_frame(pd.DataFrame(records[:1]).assign(time_stamp=1), fast,2020)
        self.assertTrue(empty.empty)

    def make_months(self, records):
        source = self.root/'input'
        source.mkdir()
        # 一条重复记录落在另一月份文件，必须全年去重。
        pd.DataFrame(records).to_parquet(source/'2020-01-01.parquet',index=False)
        pd.DataFrame(records[:1]).to_parquet(source/'2020-02-01.parquet',index=False)
        for m in range(1,13):
            month(source,self.vocab,self.root/'months',2020,m,bucket_count=3,batch_size=37)
        month(source,self.vocab,self.root/'months',2020,1,bucket_count=3,batch_size=37)

    def test_pipeline_and_statistics_equivalence(self):
        records=[]
        for u in range(60):
            for j in range(2+u%7):
                r=self.row(f'{u}-{j}',str(u),'m' if u%2 else 'f', '科技' if (u+j)%3 else '足球', '中国女足' if (u+j)%4 else '科技', j%3!=0)
                r['time_stamp'] += (j%4)*86400
                records.append(r)
        records += [self.row('conflict','x','m','科技','',False), self.row('conflict','x','f','科技','',False)]
        self.make_months(records)
        for b in range(3):
            reduce_bucket(self.root/'months',self.root/'users',b)
        reduce_bucket(self.root/'months',self.root/'users',0)
        summarize(self.root/'users',self.root/'summary')
        build_records(records + records[:1],self.vocab,self.root/'old',2020)
        old_summary(self.root/'old/behavior.sqlite',self.root/'old_summary')
        a=pd.read_csv(self.root/'summary/user_domain.csv').sort_values(['user_id','domain'])
        b=pd.read_csv(self.root/'old_summary/user_domain.csv').sort_values(['user_id','domain'])
        np.testing.assert_allclose(a[METRICS],b[METRICS],equal_nan=True)
        for name in ['gender_gaps','cross_domain_contrasts']:
            a=pd.read_csv(self.root/f'summary/{name}.csv')
            b=pd.read_csv(self.root/f'old_summary/{name}.csv')
            keys=['domain','metric','model'] + (['contrast_domain'] if name=='cross_domain_contrasts' else [])
            a=a.sort_values(keys); b=b.sort_values(keys)
            np.testing.assert_allclose(a[['estimate_f_minus_m','se']],b[['estimate_f_minus_m','se']],equal_nan=True,atol=1e-8)
        meta=json.loads((self.root/'summary/manifest.json').read_text())
        self.assertEqual(meta['unique_posts'],len(records)-1)

    def test_cross_month_conflict_and_versions(self):
        records = [self.row('same','conflict','m','科技','',False), self.row('good','ok','f','足球','科技',True)]
        self.make_months(records)
        # 重新创建一个输出目录，二月份的重复版本给出不同性别。
        pd.DataFrame([dict(records[0], gender='f')]).to_parquet(self.root/'input/2020-02-01.parquet',index=False)
        with self.assertRaisesRegex(ValueError,'不同输入'):
            month(self.root/'input',self.vocab,self.root/'months',2020,2,bucket_count=3)
        for m in range(1,13):
            month(self.root/'input',self.vocab,self.root/'fresh',2020,m,bucket_count=3)
        for b in range(3):
            reduce_bucket(self.root/'fresh',self.root/'users',b)
        summarize(self.root/'users',self.root/'summary')
        users=pd.read_csv(self.root/'summary/user_domain.csv')
        self.assertEqual(set(users.user_id), {'ok'})

    def test_owner_conflict(self):
        self.make_months([self.row('same','u','m','科技','',False),self.row('same','v','f','科技','',False)])
        b=int(buckets(pd.Series(['same']),3)[0])
        with self.assertRaisesRegex(ValueError,'多个用户'):
            reduce_bucket(self.root/'months',self.root/'users',b)
