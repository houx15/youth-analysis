"""词项重聚合与原始SQLite参考实现逐词、逐指标、逐模型对照。"""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from cultural_participation.tests import test_research, test_parquet_behavior
from cultural_participation.parquet_behavior import reduce_bucket
from cultural_participation.term_behavior import build, summarize
from cultural_participation.analysis import METRICS, database, materialize, ols
from cultural_participation.behavior import build_records


class TermBehaviorTests(unittest.TestCase):
    setUp=test_research.ResearchTests.setUp
    row=test_research.ResearchTests.row
    make_months=test_parquet_behavior.ParquetTests.make_months

    def test_term_results_match_reference(self):
        records=[]
        for u in range(30):
            for j in range(2+u%7):
                r=self.row(f'{u}-{j}',str(u),'f' if u%2 else 'm', '足球科技' if (u+j)%3 else '苹果手机', '中国女足' if j%2 else '科技', j%3!=0)
                r['time_stamp']+=(j%4)*86400
                if j==1 and u%5==0:r['r_time_stamp']=None
                records.append(r)
        records += [self.row('conflict','x','m','科技','',False),self.row('conflict','x','f','科技','',False)]
        self.make_months(records)
        for b in range(3):
            reduce_bucket(self.root/'months',self.root/'users',b)
            build(self.root,self.root/'terms',b,self.vocab)
        summarize(self.root/'terms',self.root/'term_summary',term_batch=2)
        actual=pd.read_csv(self.root/'term_summary/term_gender_gaps.csv').set_index(['term','metric','model'])
        build_records(records+records[:1],self.vocab,self.root/'old',2020)
        with database(self.root/'old/behavior.sqlite') as conn:
            v=json.loads(self.vocab.read_text())
            for term in v['terms']:
                materialize(conn,[term['term']])
                for metric in METRICS:
                    query=f'SELECT {metric},gender,total_posts,total_retweets,active_days FROM user_domain WHERE domain=?'
                    for adjusted in [False,True]:
                        expected=ols(conn,query,(term['domains'][0],),adjusted)
                        row=actual.loc[(term['term'],metric,'activity_adjusted_OLS' if adjusted else 'unadjusted_OLS')]
                        self.assertEqual(row.status,expected['status'])
                        self.assertEqual(row.n_users,expected['n_users'])
                        for key in ['estimate_f_minus_m','se']:
                            np.testing.assert_allclose(row[key],np.nan if expected[key] is None else expected[key],atol=1e-8,equal_nan=True)
        coverage=pd.read_csv(self.root/'term_summary/term_coverage.csv')
        self.assertEqual(set(coverage.n_users),{15})
        self.assertEqual(len(coverage),len(v['terms'])*2)
        # 同域嵌套词保留各自命中，单词内部不会重复计数。
        self.assertTrue((coverage.retweets>=coverage.retweet_users).all())
