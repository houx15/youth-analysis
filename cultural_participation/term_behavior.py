"""从月度逐帖命中缓存重聚合词项行为；不重新扫描原始文本，不使用数据库。

服务器通过SLURM先运行64个build分片，再运行summarize。所有词项均保留。
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cultural_participation.analysis import METRICS
from cultural_participation.parquet_behavior import manifests
from cultural_participation.term_semantics import digest

USER_COLUMNS=['user_id','gender','total_posts','total_retweets','total_expression','active_days']
COUNT_NAMES=['retweets','expressions','delay_n','commented']


def accumulate(posts, users, terms, domains, batch_size=50000):
    """一个用户分片的稀疏命中写入有限大小计数数组，所有未命中用户仍在分母中。"""
    uids={str(u):i for i,u in enumerate(users.user_id)};term_ids={t:i for i,t in enumerate(terms)}
    n,p=len(users),len(terms)
    # 防止配置错误把全年用户一次载入；当前约3500用户/分片。
    estimated=n*p*(4*len(COUNT_NAMES)+8)
    if estimated>12*1024**3: raise ValueError(f'单分片计数矩阵需{estimated/1024**3:.1f}GiB；请增加原始用户分片数后再运行')
    arrays={name:np.zeros((n,p),dtype=np.int32) for name in COUNT_NAMES}
    arrays['delay_sum']=np.zeros((n,p),dtype=np.float64)
    seen=set();processed=np.zeros(n,dtype=np.int64);stats=Counter()
    columns=['weibo_id','user_id','retweet_terms','expression_terms','lag_status']+[f'{d}_delay_sum' for d in domains]
    for path in posts:
        print(f'读取词项命中：{path}',flush=True)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,columns=columns):
            frame=batch.to_pandas();indices={k:[] for k in arrays};delays=[]
            for row in frame.itertuples(index=False,name=None):
                pid,uid,raw_rt,raw_ex,lag_status,*domain_delays=row
                stats['input_rows']+=1
                if uid not in uids: continue
                if pid in seen: stats['duplicate_posts']+=1;continue
                seen.add(pid);u=uids[uid];processed[u]+=1
                rt=set(json.loads(raw_rt));ex=set(json.loads(raw_ex))
                unknown=(rt|ex)-term_ids.keys()
                if unknown:raise ValueError(f'命中词不在当前词表：{sorted(unknown)[:5]}')
                r=[u*p+term_ids[t] for t in rt];e=[u*p+term_ids[t] for t in ex]
                indices['retweets'].extend(r);indices['expressions'].extend(e)
                indices['commented'].extend(u*p+term_ids[t] for t in rt&ex)
                if lag_status=='positive' and rt:
                    # 同一帖各命中领域保存相同log延迟；取一份，绝不相加。
                    positive=[x for x in domain_delays if x>0]
                    if not positive or not np.allclose(positive,positive[0],rtol=1e-12,atol=1e-12):
                        raise ValueError('领域延迟不能还原唯一逐帖时间差')
                    indices['delay_n'].extend(r);indices['delay_sum'].extend(r);delays.extend([positive[0]]*len(r))
            for name,idx in indices.items():
                if idx:np.add.at(arrays[name].ravel(),np.asarray(idx,dtype=np.int64),delays if name=='delay_sum' else 1)
        print(f'分片累计输入{stats["input_rows"]:,}行，有效唯一帖子{len(seen):,}',flush=True)
    np.testing.assert_array_equal(processed,users.total_posts.to_numpy(dtype=np.int64))
    stats['eligible_unique_posts']=len(seen);stats['eligible_users']=n
    return arrays,dict(stats)


def build(input_dir, output, bucket, vocabulary):
    source=Path(input_dir);meta=manifests(source/'months');signature=meta[0]['signature']
    if not 0<=bucket<signature['bucket_count']:raise ValueError('分片编号不正确')
    if digest(vocabulary)!=signature['vocabulary_sha256']:raise ValueError('词表与月度命中版本不同')
    v=json.loads(Path(vocabulary).read_text());terms=[r['term'] for r in v['terms']]
    userdir=source/'users'/f'bucket={bucket:03d}';user_meta=json.loads((userdir/'manifest.json').read_text())
    if user_meta['status']!='complete' or user_meta['source_manifests']!=meta:raise ValueError('年度用户分片与月份结果不一致')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);target=root/f'bucket={bucket:03d}'
    record={'format':'term-behavior-v1','bucket':bucket,'vocabulary_sha256':digest(vocabulary),'source_manifests':meta,
            'implementation_sha256':digest(__file__)}
    if target.exists():
        previous=json.loads((target/'manifest.json').read_text())
        if previous['signature']==record and previous['status']=='complete':print(f'已完成，跳过：{target}');return
        raise ValueError('已有结果版本不一致，请更换词项输出目录')
    users=pd.read_parquet(userdir/'users.parquet',columns=USER_COLUMNS) if (userdir/'users.parquet').exists() else pd.DataFrame(columns=USER_COLUMNS)
    users['user_id']=users.user_id.astype(str)
    posts=[source/'months'/f'month={m:02d}'/f'posts-{bucket:03d}.parquet' for m in range(1,13)]
    arrays,stats=accumulate([p for p in posts if p.exists()],users,terms,signature['domains'])
    work=Path(tempfile.mkdtemp(prefix=f'.term-{bucket:03d}-',dir=root))
    users.to_parquet(work/'users.parquet',index=False)
    np.savez_compressed(work/'counts.npz',**arrays)
    (work/'terms.json').write_text(json.dumps(terms,ensure_ascii=False))
    (work/'manifest.json').write_text(json.dumps({'status':'complete','signature':record,'stats':stats},ensure_ascii=False,indent=2))
    work.rename(target);print(f'词项行为分片完成：{target}',flush=True)


def outcomes(arrays,users,metric,start,end):
    if metric=='retweet_entry':return (arrays['retweets'][:,start:end]>0).astype(float)
    if metric=='expression_entry':return (arrays['expressions'][:,start:end]>0).astype(float)
    if metric=='retweet_share':num=arrays['retweets'][:,start:end];den=users.total_retweets.to_numpy(float)[:,None]
    elif metric=='expression_share':num=arrays['expressions'][:,start:end];den=users.total_expression.to_numpy(float)[:,None]
    elif metric=='log_delay':num=arrays['delay_sum'][:,start:end];den=arrays['delay_n'][:,start:end]
    else:num=arrays['commented'][:,start:end];den=arrays['retweets'][:,start:end]
    result=np.full(num.shape,np.nan,dtype=float)
    np.divide(num,den,out=result,where=den>0)
    return result


def summarize(input_dir,output,term_batch=128):
    root=Path(input_dir);first=json.loads((root/'bucket=000/manifest.json').read_text());source=first['signature']['source_manifests']
    count=source[0]['signature']['bucket_count'];terms=json.loads((root/'bucket=000/terms.json').read_text());p=len(terms);q=len(METRICS)
    for b in range(count):
        path=root/f'bucket={b:03d}';m=json.loads((path/'manifest.json').read_text())
        expected={**first['signature'],'bucket':b}
        if m['status']!='complete' or m['signature']!=expected or json.loads((path/'terms.json').read_text())!=terms:raise ValueError('词项分片版本或词序不一致')
    target=Path(output)
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True);work=Path(tempfile.mkdtemp(prefix='.term-summary-',dir=target.parent))
    # 仅保留每词充分统计量，年度用户数据逐分片读取。
    xtx=np.zeros((q,p,5,5));xty=np.zeros((q,p,5));n=np.zeros((q,p),dtype=np.int64)
    desc=np.zeros((q,p,2,3));beta=np.full((q,2,p,5),np.nan);inv=np.full((q,2,p,5,5),np.nan);meat=np.zeros_like(inv)
    totals=np.zeros((p,2,4),dtype=np.int64);user_counts=np.zeros(2,dtype=np.int64)
    for phase in [1,2]:
        for b in range(count):
            path=root/f'bucket={b:03d}';users=pd.read_parquet(path/'users.parquet',columns=USER_COLUMNS)
            print(f'词项汇总第{phase}/2遍：用户分片{b+1}/{count}，{len(users):,}用户',flush=True)
            if users.empty:continue
            female=users.gender.eq('f').to_numpy();x=np.column_stack([np.ones(len(users)),female.astype(float),np.log1p(users[['total_posts','total_retweets','active_days']].to_numpy(float))])
            xx=np.einsum('ni,nj->nij',x,x).reshape(len(users),25)
            with np.load(path/'counts.npz',allow_pickle=False) as z:
                arrays={name:z[name] for name in [*COUNT_NAMES,'delay_sum']}
            if phase==1:
                for sex in [0,1]:
                    mask=female==bool(sex);user_counts[sex]+=mask.sum()
                    totals[:,sex,0]+=arrays['retweets'][mask].sum(axis=0);totals[:,sex,1]+=arrays['expressions'][mask].sum(axis=0)
                    totals[:,sex,2]+=(arrays['retweets'][mask]>0).sum(axis=0);totals[:,sex,3]+=(arrays['expressions'][mask]>0).sum(axis=0)
            for mi,metric in enumerate(METRICS):
                for start in range(0,p,term_batch):
                    end=min(start+term_batch,p);y=outcomes(arrays,users,metric,start,end);valid=np.isfinite(y);y0=np.where(valid,y,0)
                    if phase==1:
                        n[mi,start:end]+=valid.sum(axis=0)
                        xtx[mi,start:end]+=np.einsum('nt,nq->tq',valid.astype(float),xx,optimize=True).reshape(end-start,5,5)
                        xty[mi,start:end]+=np.einsum('nt,ni->ti',y0,x,optimize=True)
                        for sex in [0,1]:
                            mask=female==bool(sex)
                            desc[mi,start:end,sex,0]+=valid[mask].sum(axis=0)
                            desc[mi,start:end,sex,1]+=y0[mask].sum(axis=0)
                            desc[mi,start:end,sex,2]+=(y0[mask]**2).sum(axis=0)
                    else:
                        for model,size in [(0,2),(1,5)]:
                            coefficients=beta[mi,model,start:end,:size]
                            residual=y0-np.einsum('ni,ti->nt',x[:,:size],coefficients,optimize=True)
                            sq=np.where(valid&np.isfinite(residual),residual**2,0)
                            meat[mi,model,start:end]+=np.einsum('nt,nq->tq',sq,xx,optimize=True).reshape(end-start,5,5)
            del arrays
        if phase==1:
            for mi in range(q):
                for ti in range(p):
                    for model,size in [(0,2),(1,5)]:
                        matrix=xtx[mi,ti,:size,:size]
                        if n[mi,ti]>size and np.linalg.matrix_rank(matrix)==size:
                            inverse=np.linalg.inv(matrix);inv[mi,model,ti,:size,:size]=inverse
                            beta[mi,model,ti,:size]=np.einsum('ij,j->i',inverse,xty[mi,ti,:size])
    descriptions=[];gaps=[]
    for mi,metric in enumerate(METRICS):
        for ti,term in enumerate(terms):
            for sex in [0,1]:
                number,total,squares=desc[mi,ti,sex];mean=total/number if number else np.nan
                se=np.sqrt(max(0,(squares-total**2/number)/(number-1))/number) if number>1 else np.nan
                descriptions.append({'term':term,'gender':['m','f'][sex],'metric':metric,'n_users':int(number),'mean':mean,'se':se,'ci_low':mean-1.96*se,'ci_high':mean+1.96*se})
            for model,size in [(0,2),(1,5)]:
                effect=beta[mi,model,ti,1];se=np.nan;status='insufficient_or_singular'
                if np.isfinite(effect):
                    inverse=inv[mi,model,ti,:size,:size];cov=np.einsum('ij,jk,kl->il',inverse,meat[mi,model,ti,:size,:size],inverse)*n[mi,ti]/(n[mi,ti]-size)
                    se=float(np.sqrt(max(0,cov[1,1])));status='ok'
                gaps.append({'term':term,'metric':metric,'model':['unadjusted_OLS','activity_adjusted_OLS'][model],'n_users':int(n[mi,ti]),'estimate_f_minus_m':effect,'se':se,'ci_low':effect-1.96*se,'ci_high':effect+1.96*se,'status':status})
    pd.DataFrame(descriptions).to_csv(work/'term_descriptives.csv',index=False)
    pd.DataFrame(gaps).to_csv(work/'term_gender_gaps.csv',index=False)
    coverage=[]
    for ti,term in enumerate(terms):
        for sex in [0,1]:
            coverage.append(dict(term=term,gender=['m','f'][sex],n_users=int(user_counts[sex]),retweets=int(totals[ti,sex,0]),expressions=int(totals[ti,sex,1]),retweet_users=int(totals[ti,sex,2]),expression_users=int(totals[ti,sex,3])))
    pd.DataFrame(coverage).to_csv(work/'term_coverage.csv',index=False)
    (work/'manifest.json').write_text(json.dumps({'status':'complete','source_signature':first['signature'],'terms':len(terms),'n_users':int(user_counts.sum()),
        'method':'same user-weighted definitions as five-domain pipeline; OLS/linear probability with HC1; female minus male; activity covariates log1p posts, retweets, active days',
        'implementation_sha256':digest(__file__)},ensure_ascii=False,indent=2))
    work.rename(target);print(f'词项汇总完成：{target}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('build');b.add_argument('--input-dir',required=True);b.add_argument('--output',required=True);b.add_argument('--bucket',type=int,required=True);b.add_argument('--vocabulary',required=True)
    a=s.add_parser('summarize');a.add_argument('--input-dir',required=True);a.add_argument('--output',required=True);a.add_argument('--term-batch',type=int,default=128)
    args=vars(p.parse_args());c=args.pop('command');{'build':build,'summarize':summarize}[c](**args)


if __name__=='__main__':main()
