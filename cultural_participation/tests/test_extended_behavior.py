"""用户月度零值、去重、活动量匹配和画像分层的合成数据验证。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cultural_participation.extended_behavior import monthly_panel, match_population, profiles, region, reduce, prepare, summarize, values
from cultural_participation.parquet_behavior import FastMatcher, measure_frame, aggregate, month
from cultural_participation.tests import test_research, test_parquet_behavior


class ExtendedTests(unittest.TestCase):
    setUp=test_research.ResearchTests.setUp
    row=test_research.ResearchTests.row
    make_months=test_parquet_behavior.ParquetTests.make_months

    def test_month_zero_duplicates_and_frequency(self):
        records=[self.row('a','u','f','足球足球','科技',True), self.row('b','u','f','足球','',False)]
        frame,_=measure_frame(pd.DataFrame(records),FastMatcher(self.vocab),2020)
        users,_=aggregate(frame,list(FastMatcher(self.vocab).domains))
        panel=monthly_panel(pd.concat([frame,frame.iloc[:1]]),users,list(FastMatcher(self.vocab).domains))
        self.assertEqual(len(panel),12)
        self.assertEqual(panel.total_posts.sum(),2)
        self.assertEqual(panel.sports_expressions.sum(),2)
        self.assertEqual((panel.total_posts==0).sum(),11)
        self.assertTrue(np.isnan(values(panel,'sports','retweet_share')[1:]).all())
        self.assertAlmostEqual(panel.technology_seconds_sum.sum(),np.expm1(frame.technology_delay_sum).sum())

    def test_matching_balance_and_determinism(self):
        users=pd.DataFrame([dict(user_id=f'{s}{i}',gender=s,total_posts=10+i,total_retweets=i,active_days=3+i)
            for s,n in [('m',8),('f',12)] for i in range(n)])
        out,balance=match_population(users)
        other,_=match_population(users.sample(frac=1,random_state=9))
        pd.testing.assert_frame_equal(out,other)
        pair=out[out.matched].groupby('pair_id').gender.agg(set)
        self.assertTrue(all(s=={'m','f'} for s in pair))
        self.assertEqual(out[out.matched].gender.value_counts().to_dict(),{'f':8,'m':8})
        self.assertLess(balance[balance['sample']=='matched'].smd.abs().max(),.1)

    def test_profile_mapping_and_conflicts(self):
        path=self.root/'profiles.csv';mapping=self.root/'ids.json'
        pd.DataFrame([['a','2005-01-01','北京'],['b','2006-02-01','广东'],['b','2007-01-01','四川']],columns=['user_id','birthday','province']).to_csv(path,index=False)
        mapping.write_text(json.dumps({'a':'u','b':'v'}))
        result=profiles(path,mapping,{'u','v'},2020).set_index('user_id')
        self.assertEqual(result.loc['u','age'],15)
        self.assertTrue(np.isnan(result.loc['v','age']))
        self.assertEqual(result.loc['v','region'],'Unknown')
        self.assertEqual(region('11.0'),'East')
        self.assertEqual(region('广西壮族自治区'),'West')
        self.assertEqual(region('海外'),'Unknown')

    def test_complete_synthetic_pipeline(self):
        records=[]
        for i in range(8):
            for s in ['m','f']:
                for j in range(i+1):records.append(self.row(f'{s}{i}-{j}',f'{s}{i}',s,'足球' if s=='f' else '科技','足球科技',True))
        self.make_months(records)
        for b in range(3):reduce(self.root,b)
        path=self.root/'profiles.csv'
        pd.DataFrame([dict(user_id=f'{s}{i}',birthday='2005-01-01',province='北京') for s in ['m','f'] for i in range(8)]).to_csv(path,index=False)
        other=self.root/'alternative';v=json.loads(self.vocab.read_text())
        v['terms']=[t for t in v['terms'] if t['term']!='足球']
        alternative=self.root/'alternative.json';alternative.write_text(json.dumps(v))
        for number in range(1,13):month(self.root/'input',alternative,other/'months',2020,number,bucket_count=3,batch_size=37)
        for b in range(3):reduce(other,b)
        prepare(self.root,other,self.root/'sample',path,'')
        prepare(self.root,other,self.root/'sample',path,'')
        summarize(self.root,self.root/'sample',self.root/'extended')
        summarize(self.root,self.root/'sample',self.root/'extended')
        summarize(other,self.root/'sample',other/'extended')
        different=pd.read_csv(other/'extended/descriptives.csv')
        self.assertEqual(different[(different.metric=='expressions')&(different.domain=='sports')]['mean'].max(),0)
        d=pd.read_csv(self.root/'extended/descriptives.csv')
        jan=d[(d.period=='month_01')&(d['sample']=='all')&(d.stratum=='all')&(d.domain=='sports')&(d.metric=='expressions')].set_index('gender')
        self.assertEqual(jan.loc['f','mean'],4.5)
        self.assertEqual(jan.loc['m','mean'],0)
        annual=d[(d.period=='annual_user')&(d['sample']=='matched')&(d.stratum=='all')&(d.domain=='sports')&(d.metric=='expressions')].set_index('gender')
        self.assertEqual(annual.loc['f','mean'],4.5/12)
        dist=pd.read_csv(self.root/'extended/distributions.csv')
        sums=dist.groupby(['period','sample','stratum','level','gender','domain','metric'])['n'].sum()
        expected=d.set_index(['period','sample','stratum','level','gender','domain','metric']).n_users
        pd.testing.assert_series_equal(sums.sort_index(),expected.sort_index(),check_names=False)


if __name__=='__main__':unittest.main()
