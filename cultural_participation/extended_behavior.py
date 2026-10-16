"""互斥词表新分析：用户月度面板、活动量匹配、年龄与地区分层。

仅在SLURM调用；原始文本由parquet_behavior.month重新处理。
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cultural_participation.parquet_behavior import BASE, SUFFIXES, load_parts, manifests, reduce_bucket
from cultural_participation.term_semantics import digest
from cultural_participation.behavior import normalize_id

ACTIVITY = ['total_posts', 'total_retweets', 'active_days']
REGIONS = {'East': ['北京','天津','河北','上海','江苏','浙江','福建','山东','广东','海南'],
           'Central': ['山西','安徽','江西','河南','湖北','湖南'],
           'West': ['内蒙古','广西','重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆'],
           'Northeast': ['辽宁','吉林','黑龙江']}
CODES = dict(zip(['11','12','13','31','32','33','35','37','44','46','14','34','36','41','42','43','15','45','50','51','52','53','54','61','62','63','64','65','21','22','23'],
                REGIONS['East']+REGIONS['Central']+REGIONS['West']+REGIONS['Northeast']))
METRICS = ['expressions', 'retweets', 'expression_share', 'retweet_share', 'delay_seconds', 'log_delay', 'comment_share']


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def monthly_panel(frame, users, domains):
    """全年去重后按实际发帖月份汇总，为年度有效用户补齐12月零频次。"""
    frame = frame.drop_duplicates('weibo_id', keep='first')
    frame = frame[frame.user_id.isin(users.user_id)].copy()
    frame['month'] = pd.to_datetime(frame.day).dt.month
    for d in domains:
        frame[f'{d}_seconds_sum'] = np.expm1(frame[f'{d}_delay_sum'])
    columns = ['rt','expr'] + [f'{d}_{s}' for d in domains for s in [*SUFFIXES,'seconds_sum']]
    frame[columns] = frame[columns].astype(float)
    grouped = frame.groupby(['user_id','month'], sort=False)
    result = grouped[columns].sum()
    result['total_posts'] = grouped.size()
    result['active_days'] = grouped.day.nunique()
    index = pd.MultiIndex.from_product([users.user_id, range(1,13)], names=['user_id','month'])
    result = result.reindex(index, fill_value=0).reset_index()
    result = result.rename(columns={'rt':'total_retweets','expr':'total_expression'})
    totals = result.groupby('user_id')[['total_posts','total_retweets','total_expression','active_days']].sum()
    np.testing.assert_allclose(totals.loc[users.user_id], users.set_index('user_id').loc[:,totals.columns])
    return result


def reduce(input_dir, bucket):
    root = Path(input_dir)
    reduce_bucket(root/'months', root/'users', bucket)
    meta = manifests(root/'months'); domains = meta[0]['signature']['domains']
    target = root/'monthly_users'/f'bucket={bucket:03d}'
    signature = {'source_manifests':meta,'implementation_sha256':digest(__file__),'bucket':bucket}
    if target.exists():
        if json.loads((target/'manifest.json').read_text()) == signature:
            print(f'月度分片{bucket}已完成', flush=True); return
        raise ValueError('月度输出版本不一致，请使用新的RUN_LABEL')
    target.parent.mkdir(parents=True,exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.monthly-',dir=target.parent))
    user_file = root/'users'/f'bucket={bucket:03d}'/'users.parquet'
    if user_file.exists():
        users = pd.read_parquet(user_file, columns=['user_id','total_posts','total_retweets','total_expression','active_days'])
        cols = BASE + ['lag_status','matched_retweets','matched_expression'] + [f'{d}_{s}' for d in domains for s in SUFFIXES]
        posts = load_parts(root/'months',bucket,'posts',cols)
        panel = monthly_panel(posts, users, domains)
        panel.to_parquet(work/'monthly.parquet',index=False)
    dump(work/'manifest.json',signature); work.rename(target)
    print(f'月度分片{bucket}完成',flush=True)


def load_population(root):
    root = Path(root); meta = manifests(root/'months'); count = meta[0]['signature']['bucket_count']
    parts=[]
    for b in range(count):
        folder=root/'users'/f'bucket={b:03d}'
        m=json.loads((folder/'manifest.json').read_text())
        if m['status']!='complete' or m['source_manifests']!=meta:
            raise ValueError('年度用户分片版本不一致')
        file=folder/'users.parquet'
        if file.exists():
            parts.append(pd.read_parquet(file,columns=['user_id','gender',*ACTIVITY]))
    users=pd.concat(parts,ignore_index=True).sort_values('user_id').reset_index(drop=True)
    if users.user_id.duplicated().any():raise ValueError('用户跨分片重复')
    return users,meta


def region(value):
    if pd.isna(value):return 'Unknown'
    value=str(value).strip().removesuffix('.0')
    value=CODES.get(value,value)
    for name,provinces in REGIONS.items():
        if any(value==p or value.startswith(p+'省') or value.startswith(p+'市') or value.startswith(p+'自治区') or value.startswith(p+'壮族') or value.startswith(p+'回族') or value.startswith(p+'维吾尔') for p in provinces):return name
    return 'Unknown'


def profiles(path, mapping_path, ids, year):
    """按原始用户ID拼接；重复且冲突字段设缺失，不选取任意一行。"""
    mapping = None
    if mapping_path:
        raw=json.loads(Path(mapping_path).read_text())
        mapping={normalize_id(k):normalize_id(v) for k,v in raw.items()}
    cols=['user_id','birthday','province']; pieces=[]
    for chunk in pd.read_csv(path,usecols=cols,dtype=str,chunksize=100000):
        chunk['user_id']=chunk.user_id.map(normalize_id)
        if mapping is not None:chunk['user_id']=chunk.user_id.map(mapping)
        chunk=chunk[chunk.user_id.isin(ids)].copy()
        if not chunk.empty:pieces.append(chunk)
    if not pieces:return pd.DataFrame(columns=['user_id','age','region'])
    frame=pd.concat(pieces,ignore_index=True)
    frame['age']=year-pd.to_datetime(frame.birthday,format='%Y-%m-%d',errors='coerce').dt.year
    frame.loc[~frame.age.between(0,120),'age']=np.nan
    frame['region']=frame.province.map(region)
    rows=[]
    for uid,g in frame.groupby('user_id',sort=False):
        ages=g.age.dropna().unique(); locations=g.loc[g.region!='Unknown','region'].unique()
        rows.append({'user_id':uid,'age':ages[0] if len(ages)==1 else np.nan,'region':locations[0] if len(locations)==1 else 'Unknown'})
    return pd.DataFrame(rows)


def match_population(users, width=.25, seed=20261016):
    """对三项log1p活动量粗化精确匹配；每格男女各取较小人数，1:1无放回。"""
    if width<=0:raise ValueError('匹配分箱宽度必须为正')
    out=users.copy().sort_values('user_id').reset_index(drop=True)
    x=np.log1p(out[ACTIVITY].to_numpy(float))
    if not np.isfinite(x).all():raise ValueError('活动量无效')
    cells=np.floor(x/width).astype(int)
    out['match_cell']=['_'.join(map(str,row)) for row in cells]
    out['matched']=False;out['pair_id']=''
    rng=np.random.default_rng(seed);pairs=0
    for cell,g in out.groupby('match_cell',sort=True):
        men=g.index[g.gender=='m'].to_numpy();women=g.index[g.gender=='f'].to_numpy()
        n=min(len(men),len(women))
        if not n:continue
        men=rng.permutation(men)[:n];women=rng.permutation(women)[:n]
        out.loc[np.r_[men,women],'matched']=True
        names=[f'p{i}' for i in range(pairs,pairs+n)];pairs+=n
        out.loc[men,'pair_id']=names;out.loc[women,'pair_id']=names
    balance=[]
    for col in ACTIVITY:
        for scale in ['raw','log1p']:
            values=out[col].astype(float) if scale=='raw' else np.log1p(out[col].astype(float))
            f0=values[out.gender=='f'];m0=values[out.gender=='m']
            denom=np.sqrt((f0.var()+m0.var())/2)
            for sample,keep in [('all',np.ones(len(out),dtype=bool)),('matched',out.matched)]:
                f=values[keep&(out.gender=='f')];m=values[keep&(out.gender=='m')]
                diff=f.mean()-m.mean()
                balance.append(dict(variable=col,scale=scale,sample=sample,n_f=len(f),n_m=len(m),mean_f=f.mean(),mean_m=m.mean(),smd=diff/denom if denom>0 else (0 if diff==0 else np.nan)))
    return out,pd.DataFrame(balance)


def prepare(full_run, top_run, output, profile_csv, id_mapping, year=2020, match_width=.25):
    target=Path(output)
    users,meta=load_population(full_run);other,other_meta=load_population(top_run)
    if target.exists():
        previous=json.loads((target/'manifest.json').read_text())
        checks={'status':'complete','full_source':meta,'top_source':other_meta,'year':year,'match_width':match_width,
                'profile_csv_sha256':digest(profile_csv),'id_mapping_sha256':digest(id_mapping) if id_mapping else None,'implementation_sha256':digest(__file__)}
        if all(previous.get(k)==v for k,v in checks.items()) and (target/'sample.parquet').is_file():
            print('匹配样本已完成，复用同一版本',flush=True);return
        raise ValueError('已有匹配样本版本不一致，请使用新输出目录')
    pd.testing.assert_frame_equal(users,other)
    if meta[0]['signature']['year']!=year or other_meta[0]['signature']['year']!=year:raise ValueError('年份不同')
    demo=profiles(profile_csv,id_mapping,set(users.user_id),year)
    if demo.empty:
        raise ValueError('画像与行为用户没有ID交集，请核对PROFILE_CSV和ID_MAPPING；不会输出虚假的分层结果')
    users=users.merge(demo,on='user_id',how='left',validate='one_to_one')
    users['region']=users.region.fillna('Unknown')
    # 2020年龄按出生年计算，预先固定分组；同时保留单岁年龄便于复核。
    users['age_group']=pd.cut(users.age,[-1,13,15,17,20,30,120],labels=['0-13','14-15','16-17','18-20','21-30','31+']).astype('string').fillna('Unknown')
    users,balance=match_population(users,match_width)
    if not users.matched.any():
        raise ValueError('没有活动量共同支持的男女匹配对，请检查活动量分布与MATCH_WIDTH')
    target.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='.sample-',dir=target.parent))
    users.to_parquet(work/'sample.parquet',index=False);balance.to_csv(work/'matching_balance.csv',index=False)
    users.groupby(['gender','matched','age_group','region'],observed=True).size().rename('n_users').reset_index().to_csv(work/'sample_coverage.csv',index=False)
    users.groupby(['gender','age'],dropna=False).size().rename('n_users').reset_index().to_csv(work/'age_distribution.csv',index=False)
    audit=users.groupby('gender').agg(n_users=('user_id','size'),n_matched=('matched','sum'),n_age_known=('age','count'),n_region_known=('region',lambda x: int((x!='Unknown').sum())))
    audit.to_csv(work/'profile_coverage.csv')
    print(audit.to_string(),flush=True)
    # 全局活动量匹配后，各年龄/地区内部未必保持平衡，另列分层诊断。
    strat_balance=[]
    for field in ['age_group','region']:
        for level,g in users.groupby(field,observed=True):
            for col in ACTIVITY:
                x=np.log1p(g[col].astype(float));sd=np.sqrt((x[g.gender=='f'].var()+x[g.gender=='m'].var())/2)
                for label,keep in [('all',np.ones(len(g),dtype=bool)),('matched',g.matched)]:
                    f=x[keep & g.gender.eq('f')];m=x[keep & g.gender.eq('m')]
                    strat_balance.append(dict(stratum=field,level=str(level),variable=col,sample=label,n_f=len(f),n_m=len(m),smd=(f.mean()-m.mean())/sd if sd>0 else np.nan))
    pd.DataFrame(strat_balance).to_csv(work/'stratified_balance.csv',index=False)
    dump(work/'manifest.json',{'status':'complete','full_run':str(full_run),'top_run':str(top_run),'full_source':meta,'top_source':other_meta,
         'profile_csv_sha256':digest(profile_csv),'id_mapping_sha256':digest(id_mapping) if id_mapping else None,
         'year':year,'matching':'1:1 without replacement within floor(log1p(activity)/width) cells; seed=20261016',
         'match_width':match_width,'matched_pairs':int(users.matched.sum()/2),'implementation_sha256':digest(__file__)})
    work.rename(target);print(f'匹配完成：{users.matched.sum()//2:,}对；请查看matching_balance.csv',flush=True)


# 固定直方图区间，保留零值与长尾；不把分箱近似分位数当作精确分位数。
COUNT_EDGES=np.array([0,.0000001,.25,.5,1,2,3,5,10,20,50,100,200,500,1000,5000,np.inf])
DELAY_EDGES=np.array([0,60,300,900,3600,21600,86400,604800,2592000,31536000,np.inf])
SHARE_EDGES=np.array([0,.0000001,.001,.005,.01,.025,.05,.1,.2,.4,.6,.8,1.00000001,np.inf])


def values(frame, domain, metric, annual=False):
    if metric in ['expressions','retweets']:
        return frame[f'{domain}_{metric}'].to_numpy(float)/(12 if annual else 1)
    if metric=='expression_share':num=f'{domain}_expressions';den='total_expression'
    elif metric=='retweet_share':num=f'{domain}_retweets';den='total_retweets'
    elif metric=='comment_share':num=f'{domain}_commented';den=f'{domain}_retweets'
    elif metric=='delay_seconds':num=f'{domain}_seconds_sum';den=f'{domain}_delay_n'
    else:num=f'{domain}_delay_sum';den=f'{domain}_delay_n'
    a=frame[num].to_numpy(float);b=frame[den].to_numpy(float)
    return np.divide(a,b,out=np.full(len(a),np.nan),where=b>0)


def summarize(input_dir, sample_dir, output):
    root=Path(input_dir);target=Path(output)
    sample_root=Path(sample_dir);sample_meta=json.loads((sample_root/'manifest.json').read_text())
    meta=manifests(root/'months');sig=meta[0]['signature'];domains=sig['domains']
    if meta not in [sample_meta['full_source'],sample_meta['top_source']]:raise ValueError('匹配样本与本轮原始数据不一致')
    if target.exists():
        previous=json.loads((target/'manifest.json').read_text())
        if previous.get('status')=='complete' and previous.get('source')==meta and previous.get('sample_manifest')==sample_meta and previous.get('sample_sha256')==digest(sample_root/'sample.parquet') and previous.get('implementation_sha256')==digest(__file__):
            print('扩展汇总已完成，跳过',flush=True);return
        raise ValueError('已有扩展汇总版本不一致')
    sample=pd.read_parquet(sample_root/'sample.parquet',columns=['user_id','gender','age_group','age','region','matched'])
    records={}; hist={}
    def collect(frame,period,annual=False):
        for sample_name in ['all','matched']:
            chosen=frame if sample_name=='all' else frame[frame.matched]
            groups=[('all','all',chosen)]
            for field in ['age_group','region']:
                groups.extend((field,str(label),g) for label,g in chosen.groupby(field,observed=True))
            for strat,level,g in groups:
                for sex,part in g.groupby('gender'):
                    for d in domains:
                        for metric in METRICS:
                            y=values(part,d,metric,annual);y=y[np.isfinite(y)]
                            key=(period,sample_name,strat,level,sex,d,metric)
                            stat=records.setdefault(key,np.zeros(4));stat+= [len(y),y.sum(),np.dot(y,y),np.count_nonzero(y==0)]
                            edges=COUNT_EDGES if metric in ['expressions','retweets'] else DELAY_EDGES if metric=='delay_seconds' else np.log1p(DELAY_EDGES) if metric=='log_delay' else SHARE_EDGES
                            counts=np.histogram(y,bins=edges)[0]
                            hist.setdefault(key,np.zeros(len(edges)-1,dtype=np.int64))[:]+=counts
    for b in range(sig['bucket_count']):
        folder=root/'monthly_users'/f'bucket={b:03d}'
        m=json.loads((folder/'manifest.json').read_text())
        if m!={'source_manifests':meta,'implementation_sha256':digest(__file__),'bucket':b}:raise ValueError('月度用户分片版本不一致')
        path=folder/'monthly.parquet'
        if not path.exists():continue
        cols=['user_id','month','total_posts','total_retweets','total_expression','active_days']+[f'{d}_{s}' for d in domains for s in [*SUFFIXES,'seconds_sum']]
        frame=pd.read_parquet(path,columns=cols)
        frame=frame.merge(sample,on='user_id',how='left',validate='many_to_one')
        if frame.gender.isna().any():raise ValueError('月度用户缺少样本信息')
        for month,g in frame.groupby('month'):collect(g,f'month_{month:02d}')
        numeric=[c for c in cols if c not in ['user_id','month']]
        annual=frame.groupby('user_id')[numeric].sum().reset_index().merge(sample,on='user_id',validate='one_to_one')
        collect(annual,'annual_user',True)
        print(f'分布汇总：{b+1}/{sig["bucket_count"]}分片',flush=True)
    columns=['period','sample','stratum','level','gender','domain','metric']; rows=[]; distributions=[]
    for key,(n,total,squares,zero) in records.items():
        mean=total/n if n else np.nan;sd=np.sqrt(max(0,(squares-total*total/n)/(n-1))) if n>1 else np.nan
        rows.append(dict(zip(columns,key),n_users=int(n),mean=mean,sd=sd,se=sd/np.sqrt(n) if n else np.nan,zero_fraction=zero/n if n else np.nan))
        metric=key[-1];edges=COUNT_EDGES if metric in ['expressions','retweets'] else DELAY_EDGES if metric=='delay_seconds' else np.log1p(DELAY_EDGES) if metric=='log_delay' else SHARE_EDGES
        cum=0
        for lower,upper,num in zip(edges[:-1],edges[1:],hist[key]):
            cum+=int(num)
            distributions.append(dict(zip(columns,key),lower=lower,upper=upper,n=int(num),proportion=num/n if n else np.nan,cumulative=cum/n if n else np.nan))
    result=pd.DataFrame(rows)
    index=[c for c in columns if c!='gender']
    f=result[result.gender=='f'].set_index(index);m=result[result.gender=='m'].set_index(index)
    # 描述性均值差；不将匹配后的男女误当成独立样本计算标准误。
    gaps=(f['mean']-m['mean']).rename('female_minus_male').reset_index()
    target.parent.mkdir(parents=True,exist_ok=True);work=Path(tempfile.mkdtemp(prefix='.extended-summary-',dir=target.parent))
    result.to_csv(work/'descriptives.csv',index=False)
    pd.DataFrame(distributions).to_csv(work/'distributions.csv',index=False)
    gaps.to_csv(work/'gender_gaps.csv',index=False)
    dump(work/'manifest.json',{'status':'complete','source':meta,'sample_manifest':sample_meta,'sample_sha256':digest(sample_root/'sample.parquet'),
         'zero_months':'All 12 calendar months for each eligible annual user; count zeros retained. Conditional shares and delays missing when denominator is zero.',
         'annual_user':'counts divided by 12; shares from annual counts; delays averaged over valid reposts within user',
         'distribution':'fixed bins [lower, upper); no exact quantiles inferred','implementation_sha256':digest(__file__)})
    work.rename(target)


def annual(input_dir, output):
    from cultural_participation.parquet_summary import summarize as annual_summary
    target=Path(output);source=manifests(Path(input_dir)/'months')
    if target.exists():
        record=json.loads((target/'manifest.json').read_text())
        if record.get('status')=='complete' and record.get('source_manifests')==source:
            print('年度汇总已完成，跳过',flush=True);return
        raise ValueError('年度汇总版本不一致')
    annual_summary(Path(input_dir)/'users',target)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    r=s.add_parser('reduce');r.add_argument('--input-dir',required=True);r.add_argument('--bucket',type=int,required=True)
    r=s.add_parser('prepare');r.add_argument('--full-run',required=True);r.add_argument('--top-run',required=True);r.add_argument('--output',required=True);r.add_argument('--profile-csv',required=True);r.add_argument('--id-mapping',default='');r.add_argument('--year',type=int,default=2020);r.add_argument('--match-width',type=float,default=.25)
    r=s.add_parser('summarize');r.add_argument('--input-dir',required=True);r.add_argument('--sample-dir',required=True);r.add_argument('--output',required=True)
    r=s.add_parser('annual');r.add_argument('--input-dir',required=True);r.add_argument('--output',required=True)
    a=vars(p.parse_args());command=a.pop('command');globals()[command](**a)


if __name__=='__main__':main()
