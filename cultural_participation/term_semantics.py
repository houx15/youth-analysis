"""全部词项的静态向量测量：压缩包流式提取、词轴投影、覆盖和稳健性统计。

只保留正式词表与锚词，不把完整词向量模型载入内存。
"""
import argparse
import bz2
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

VOCABULARY = Path('cultural_participation/vocabularies/strict_v1/vocabulary.json')
AXES = Path('cultural_participation/semantic_specs/axes_zh_v1.json')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def extract(source, output, vocabulary=VOCABULARY, axes=AXES):
    source,output=Path(source),Path(output)
    v=json.loads(Path(vocabulary).read_text());spec=json.loads(Path(axes).read_text())
    needed={r['term'] for r in v['terms']}
    needed.update(w for axis in spec['axes'].values() for pair in axis['pairs'] for w in pair)
    signature={'source_size':source.stat().st_size,'vocabulary_sha256':digest(vocabulary),'axes_sha256':digest(axes),'source_sha256':digest(source)}
    if output.exists():
        old=json.loads((output/'manifest.json').read_text())
        if old['signature'] != signature: raise ValueError('已存在不同向量或词表版本的提取结果')
        print(f'提取已完成：{output}',flush=True);return
    output.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='.extract-',dir=output.parent))
    opener=gzip.open if source.suffix=='.gz' else bz2.open if source.suffix=='.bz2' else open
    wanted={t.encode('utf-8') for t in needed};found={};dimension=None;declared=None;scanned=0
    with opener(source,'rb') as stream:
        for i,raw in enumerate(stream):
            if i==0:
                header=raw.split()
                if len(header)==2 and all(x.isdigit() for x in header):
                    declared,dimension=map(int,header);continue
            scanned+=1
            if scanned%200000==0: print(f'{source.name}: 扫描{scanned:,}词，保留{len(found):,}词',flush=True)
            # 绝大部分行无需解码或解析数值。
            word,sep,values=raw.rstrip().partition(b' ')
            if word not in wanted: continue
            term=word.decode('utf-8')
            if term in found: raise ValueError(f'模型含重复目标词：{term}')
            vector=np.fromstring(values.decode('ascii'),sep=' ',dtype=np.float64)
            if dimension is None: dimension=len(vector)
            if len(vector)!=dimension or not np.isfinite(vector).all() or np.linalg.norm(vector)==0:
                raise ValueError(f'目标词向量无效：{term}')
            found[term]=vector/np.linalg.norm(vector)
    if declared is not None and scanned != declared: raise ValueError(f'模型行数不完整：{scanned} != {declared}')
    words=sorted(found)
    np.savez_compressed(work/'vectors.npz',terms=np.array(words),vectors=np.stack([found[t] for t in words]))
    manifest={'status':'complete','signature':signature,'source':str(source),'dimension':dimension,'declared_rows':declared,
              'scanned_rows':scanned,'needed_terms':len(needed),'found_terms':len(found),'missing_terms':sorted(needed-found.keys()),
              'representation':'exact case-sensitive vocabulary entries; no subword synthesis or imputation'}
    (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    work.rename(output);print(f'完成：{output}，找到{len(found)}/{len(needed)}词',flush=True)


def axis_vector(vectors,pairs):
    differences=np.stack([vectors[a]-vectors[b] for a,b in pairs])
    vector=differences.mean(axis=0);norm=np.linalg.norm(vector)
    if norm<1e-12: raise ValueError('词轴退化')
    return vector/norm


def correlation(x,y,rank=False):
    keep=np.isfinite(x)&np.isfinite(y);x=np.asarray(x)[keep];y=np.asarray(y)[keep]
    if len(x)<3: return np.nan
    if rank: x=pd.Series(x).rank().to_numpy();y=pd.Series(y).rank().to_numpy()
    if np.std(x)==0 or np.std(y)==0: return np.nan
    return float(np.corrcoef(x,y)[0,1])


def annotate(v):
    # 稳健性集合明确命名为已标记/规则识别；不声称完整实体识别。
    brands=set('华为 瑞幸 美团 爱奇艺 亚马逊 谷歌 滴滴 格力 阅文 TikTok Facebook OPPO iPad iPhoneSE QQ 支付宝 新浪 微博 腾讯 百度 阿里巴巴 京东 小米 苹果 特斯拉 SpaceX 微软 耐克 阿迪达斯 星巴克 海底捞 麦当劳 肯德基 必胜客 网易 快手 抖音 哔哩哔哩 B站 淘宝 天猫 拼多多 苏宁 优酷 索尼 三星 联想 vivo VIVO'.split())
    aliases=[['人工智能','AI'],['世界卫生组织','世卫'],['中国乒乓球队','国乒']]
    alias_map={term:group[0] for group in aliases for term in group}
    rows=[]
    for r in v['terms']:
        term=r['term'];person=r.get('term_type')=='person'
        organization=not person and bool(re.search(r'(委员会|外交部|教育部|国防部|商务部|公安部|卫健委|法院|检察院|海关|使馆|教育局|研究院|科学院|大学|公司|银行)$',term))
        rows.append({'term':term,'domains':'|'.join(r['domains']),'domain_count':len(r['domains']),
                     'known_person':person,'organization_suffix_rule':organization,'known_brand_or_platform':term in brands,
                     'gender_marked':bool(re.search('[男女]',term)), 'alias_group':alias_map.get(term,term),
                     'term_type_source':r.get('term_type','not_separately_typed'),
                     'hotsearch_title_count':r.get('distinct_title_count',0)})
    return pd.DataFrame(rows)


def measure(extracted, output, model, vocabulary=VOCABULARY, axes=AXES):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(axes).read_text());v=json.loads(Path(vocabulary).read_text())
    extracted_meta=json.loads((Path(extracted)/'manifest.json').read_text())
    if extracted_meta['signature']['vocabulary_sha256']!=digest(vocabulary) or extracted_meta['signature']['axes_sha256']!=digest(axes):
        raise ValueError('提取缓存的词表/锚词与测量配置不同')
    with np.load(Path(extracted)/'vectors.npz',allow_pickle=False) as data:
        vectors=dict(zip(data['terms'].tolist(),data['vectors']))
    table=annotate(v);table['model']=model;table['covered']=table.term.isin(vectors)
    table['is_anchor']=table.term.isin({w for axis in spec['axes'].values() for pair in axis['pairs'] for w in pair})
    words=table.loc[table.covered,'term'];matrix=np.stack([vectors[w] for w in words]);positions=table.index[table.covered]
    axisrows=[];pairrows=[];loo=[];axisvectors={}
    for name,axis in spec['axes'].items():
        pairs=axis['pairs'];available=[p for p in pairs if all(w in vectors for w in p)]
        missing=[p for p in pairs if p not in available]
        complete=not missing
        status='complete' if complete else 'missing_anchor_pairs'
        axisrows.append({'axis':name,'status':status,'n_pairs':len(pairs),'available_pairs':len(available),'missing_pairs':json.dumps(missing,ensure_ascii=False)})
        table[name]=np.nan
        if not complete: continue  # 缺锚词不自动改变主测量轴。
        direction=axis_vector(vectors,pairs);axisvectors[name]=direction
        scores=np.einsum("ij,j->i",matrix,direction)
        if not np.isfinite(scores).all() or np.max(np.abs(scores))>1+1e-9: raise ValueError("投影数值无效")
        table.loc[positions,name]=scores
        for i,pair in enumerate(pairs):
            delta=vectors[pair[0]]-vectors[pair[1]];delta=delta/np.linalg.norm(delta)
            pairrows.append({'axis':name,'positive':pair[0],'negative':pair[1],'alignment_with_axis':float(delta@direction)})
            if len(pairs)<3: continue
            other=axis_vector(vectors,pairs[:i]+pairs[i+1:]);alternative=np.einsum("ij,j->i",matrix,other)
            for term,score in zip(words,alternative):
                loo.append({'term':term,'axis':name,'removed_pair':'/'.join(pair),'score':score})
    table.to_csv(output/'term_scores.csv',index=False)
    pd.DataFrame(axisrows).to_csv(output/'axis_coverage.csv',index=False)
    pd.DataFrame(pairrows).to_csv(output/'anchor_alignment.csv',index=False)
    pd.DataFrame(loo).to_csv(output/'leave_pair_out_scores.csv',index=False)
    cosines=[{'axis_a':a,'axis_b':b,'cosine':float(axisvectors[a]@axisvectors[b])} for a,b in itertools.combinations(axisvectors,2)]
    pd.DataFrame(cosines).to_csv(output/'axis_cosines.csv',index=False)
    coverage=[]
    for domain in ['all',*v['domains']]:
        subset=table if domain=='all' else table[table.domains.str.split('|').map(lambda x:domain in x)]
        coverage.append({'domain':domain,'n_terms':len(subset),'n_covered':int(subset.covered.sum()),'coverage':subset.covered.mean(),
                         'n_known_person':int(subset.known_person.sum()),'known_person_coverage':subset.loc[subset.known_person,'covered'].mean()})
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
    (output/'axes.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    (output/'manifest.json').write_text(json.dumps({'status':'complete','model':model,'input':extracted_meta,'axes_sha256':digest(axes),
        'measurement_status':'exploratory Chinese axes; not survey-validated','known_person_tags':'original adjudication tags; not exhaustive',
        'alias_groups':[['人工智能','AI'],['世界卫生组织','世卫'],['中国乒乓球队','国乒']],
        'alias_scope':'limited explicit high-confidence alias sensitivity, not exhaustive entity resolution'},ensure_ascii=False,indent=2))
    analyze(table,output,axisvectors,matrix,positions)
    print(f'语义测量完成：{output}',flush=True)


def subsets(table):
    yield 'all_covered',table[table.covered]
    yield 'exclude_known_person',table[table.covered&~table.known_person]
    yield 'exclude_organization_rule',table[table.covered&~table.organization_suffix_rule]
    yield 'exclude_known_brand',table[table.covered&~table.known_brand_or_platform]
    yield 'exclude_gender_marked',table[table.covered&~table.gender_marked]
    yield 'exclude_axis_anchors',table[table.covered&~table.is_anchor]
    yield 'exclude_all_flagged_entities',table[table.covered&~(table.known_person|table.organization_suffix_rule|table.known_brand_or_platform)]
    numeric=table.select_dtypes(include='number').columns.tolist()
    collapsed=table[table.covered].groupby('alias_group',sort=False)[numeric].mean().reset_index()
    yield 'merge_reviewed_aliases',collapsed


def analyze(table,output,axisvectors,matrix,positions):
    evaluations=['prestige','status','cultivation','potency','morality','affluence']
    results=[]
    for variant,sub in subsets(table):
        for axis in evaluations:
            a=sub[['gender',axis]].dropna()
            results.append({'variant':variant,'scope':'all','axis':axis,'n':len(a),'pearson':correlation(a.gender,a[axis]),'spearman':correlation(a.gender,a[axis],True)})
            if 'domains' not in sub: continue
            for domain in ['politics','business','technology','sports','arts']:
                b=sub[sub.domains.str.split('|').map(lambda x:domain in x)][['gender',axis]].dropna()
                results.append({'variant':variant,'scope':domain,'axis':axis,'n':len(b),'pearson':correlation(b.gender,b[axis]),'spearman':correlation(b.gender,b[axis],True)})
            # 多标签领域控制；同一词不因所属多个领域而复制成多行。
            b=sub.dropna(subset=['gender',axis])
            design=np.column_stack([np.ones(len(b))]+[b.domains.str.split('|').map(lambda x:d in x).to_numpy(float) for d in ['politics','business','technology','sports','arts']])
            x=b.gender.to_numpy();y=b[axis].to_numpy()
            if len(b)>10:
                xr=x-np.einsum("ij,j->i",design,np.linalg.lstsq(design,x,rcond=None)[0]);yr=y-np.einsum("ij,j->i",design,np.linalg.lstsq(design,y,rcond=None)[0])
                results.append({'variant':variant,'scope':'domain_adjusted_residual','axis':axis,'n':len(b),'pearson':correlation(xr,yr),'spearman':np.nan})
    pd.DataFrame(results).to_csv(output/'associations.csv',index=False)
    centroids=[];index_lookup={index:i for i,index in enumerate(positions)}
    for variant,sub in subsets(table):
        if 'domains' not in sub: continue
        for domain in ['politics','business','technology','sports','arts']:
            selected=sub[sub.domains.str.split('|').map(lambda x:domain in x)]
            if selected.empty:continue
            center=matrix[[index_lookup[i] for i in selected.index]].mean(axis=0);norm=float(np.linalg.norm(center))
            for name,direction in axisvectors.items():
                centroids.append({'variant':variant,'domain':domain,'axis':name,'n_terms':len(selected),'mean_projection':float(center@direction),
                                  'normalized_centroid_projection':float(center@direction/norm),'centroid_norm':norm})
    pd.DataFrame(centroids).to_csv(output/'centroids.csv',index=False)
    extremes=[]
    for axis in axisvectors:
        valid=table[table.covered].dropna(subset=[axis]).sort_values([axis,'term'])
        for side,sub in [('low',valid.head(15)),('high',valid.tail(15).iloc[::-1])]:
            for _,row in sub.iterrows(): extremes.append({'axis':axis,'side':side,'term':row.term,'domains':row.domains,'score':row[axis]})
    pd.DataFrame(extremes).to_csv(output/'extremes.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('extract');e.add_argument('--source',required=True);e.add_argument('--output',required=True)
    m=sub.add_parser('measure');m.add_argument('--extracted',required=True);m.add_argument('--output',required=True);m.add_argument('--model',required=True)
    a=vars(p.parse_args());command=a.pop('command');{'extract':extract,'measure':measure}[command](**a)


if __name__=='__main__':main()
