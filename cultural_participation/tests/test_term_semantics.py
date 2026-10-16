"""词轴投影、压缩模型提取、缺失和质心的边界测试。"""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cultural_participation.term_semantics import AXES, axis_vector, extract, measure


class TermSemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.vocab=self.root/'vocab.json'
        self.vocab.write_text(json.dumps({'version':'fixture','domains':{'sports':'体育','arts':'文艺'},'terms':[
            {'term':'篮球','domains':['sports']},{'term':'电影','domains':['arts']},{'term':'人工智能','domains':['arts']},
            {'term':'AI','domains':['arts']},{'term':'不存在','domains':['sports']}]}))
        self.axes=json.loads(AXES.read_text());words={w for a in self.axes['axes'].values() for p in a['pairs'] for w in p}|{'篮球','电影','人工智能','AI'}
        rng=np.random.default_rng(42);self.vectors={w:rng.normal(size=8) for w in sorted(words)}

    def write_vectors(self,omit=(),declared_offset=0):
        values={w:v for w,v in self.vectors.items() if w not in omit}
        path=self.root/'model.gz'
        with gzip.open(path,'wt',encoding='utf-8') as f:
            f.write(f'{len(values)+declared_offset} 8\n')
            for word,vector in values.items():f.write(word+' '+' '.join(map(str,vector))+'\n')
        return path

    def test_projection_and_centroid(self):
        path=self.write_vectors();extract(path,self.root/'subset',self.vocab,AXES)
        measure(self.root/'subset',self.root/'scores','fixture',self.vocab,AXES)
        scores=pd.read_csv(self.root/'scores/term_scores.csv').set_index('term')
        self.assertFalse(scores.loc['不存在','covered']);self.assertTrue(np.isnan(scores.loc['不存在','gender']))
        norm={w:v/np.linalg.norm(v) for w,v in self.vectors.items()};axis=axis_vector(norm,self.axes['axes']['gender']['pairs'])
        for word in ['篮球','电影','人工智能','AI']:
            expected=sum(float(x)*float(y) for x,y in zip(norm[word],axis))
            self.assertAlmostEqual(scores.loc[word,'gender'],expected)
        centroid=pd.read_csv(self.root/'scores/centroids.csv')
        row=centroid[(centroid.variant=='all_covered')&(centroid.domain=='arts')&(centroid.axis=='gender')].iloc[0]
        self.assertAlmostEqual(row.mean_projection,scores.loc[['电影','人工智能','AI'],'gender'].mean())
        extract(path,self.root/'subset',self.vocab,AXES) # 精确版本可重用

    def test_missing_anchor_is_not_silently_removed(self):
        path=self.write_vectors(omit=['女性']);extract(path,self.root/'subset',self.vocab,AXES)
        measure(self.root/'subset',self.root/'scores','fixture',self.vocab,AXES)
        scores=pd.read_csv(self.root/'scores/term_scores.csv');self.assertTrue(scores.gender.isna().all())
        coverage=pd.read_csv(self.root/'scores/axis_coverage.csv').set_index('axis')
        self.assertEqual(coverage.loc['gender','status'],'missing_anchor_pairs')

    def test_truncated_model_rejected(self):
        path=self.write_vectors(declared_offset=1)
        with self.assertRaisesRegex(ValueError,'行数不完整'):extract(path,self.root/'subset',self.vocab,AXES)
        self.assertFalse((self.root/'subset').exists())
