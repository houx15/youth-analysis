"""从用户级Parquet分片批量计算描述统计与HC1性别/跨领域对比。"""

from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cultural_participation.analysis import METRICS


BASE = ['user_id', 'gender', 'total_posts', 'total_retweets', 'total_expression', 'available_retweets', 'active_days', 'matched_retweets', 'matched_expression']


def batches(root, count, columns):
    for bucket in range(count):
        path = Path(root) / f'bucket={bucket:03d}' / 'users.parquet'
        if path.exists():
            for batch in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=columns):
                yield batch.to_pandas()


def outcomes(frame, domains):
    for a, b in [(d, '') for d in domains] + list(itertools.combinations(sorted(domains), 2)):
        for metric in METRICS:
            y = frame[f'{a}_{metric}'].to_numpy(float)
            if b:
                y = y - frame[f'{b}_{metric}'].to_numpy(float)
            yield (a, b, metric), y


def summarize(input_dir, output):
    root = Path(input_dir)
    first = json.loads((root / 'bucket=000/manifest.json').read_text())
    source = first['source_manifests']
    signature = source[0]['signature']
    count, domains = signature['bucket_count'], signature['domains']
    exclusions = Counter()
    unique_posts = 0
    for bucket in range(count):
        m = json.loads((root / f'bucket={bucket:03d}' / 'manifest.json').read_text())
        if m['status'] != 'complete' or m['bucket'] != bucket or m['source_manifests'] != source:
            raise ValueError('用户分片不完整或来自不同的月份结果')
        unique_posts += m['stats']['unique_posts']
        for row in m['stats']['user_exclusions']:
            exclusions[(row['gender'], row['conflict'])] += row['users']
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.summary-', dir=target.parent))
    cols = BASE + [f'{d}_{m}' for d in domains for m in METRICS]
    cols += [f'{d}_{s}' for d in domains for s in ['retweets', 'expressions', 'delay_n']]
    models = {}
    desc = defaultdict(lambda: np.zeros(3))
    coverage = {g: Counter() for g in ['m', 'f']}
    wrote_users = False
    for pass_number in [1, 2]:
        print(f'汇总第{pass_number}/2遍：'+('用户统计及回归充分统计量' if pass_number == 1 else 'HC1标准误'), flush=True)
        for frame in batches(root, count, cols):
            gender = frame.gender.to_numpy()
            x = np.column_stack([np.ones(len(frame)), (gender == 'f').astype(float), np.log1p(frame[['total_posts', 'total_retweets', 'active_days']].to_numpy(float))])
            if pass_number == 1:
                for g in ['m', 'f']:
                    part = frame.loc[frame.gender.eq(g)]
                    coverage[g]['n_users'] += len(part)
                    for name in BASE[2:]:
                        coverage[g][name] += int(part[name].sum())
                for d in domains:
                    for metric in METRICS:
                        for g in ['m', 'f']:
                            y = frame.loc[frame.gender.eq(g), f'{d}_{metric}'].dropna().to_numpy(float)
                            desc[(d, g, metric)] += [len(y), y.sum(), np.dot(y, y)]
                    part = frame[BASE].copy()
                    part['domain'] = d
                    for metric in METRICS:
                        part[metric] = frame[f'{d}_{metric}'].to_numpy()
                    for src, dest in [('retweets', 'retweet_count'), ('expressions', 'expression_count'), ('delay_n', 'delay_n')]:
                        part[dest] = frame[f'{d}_{src}'].to_numpy()
                    part.to_csv(work / 'user_domain.csv', mode='a', header=not wrote_users, index=False)
                    wrote_users = True
            for key, y in outcomes(frame, domains):
                valid = np.isfinite(y)
                yy = y[valid]
                for adjusted, size in [(False, 2), (True, 5)]:
                    k = (*key, adjusted)
                    xx = x[valid, :size]
                    if pass_number == 1:
                        if k not in models:
                            models[k] = {'n': 0, 'xtx': np.zeros((size, size)), 'xty': np.zeros(size), 'meat': np.zeros((size, size))}
                        model = models[k]
                        model['n'] += len(yy)
                        model['xtx'] += xx.T @ xx
                        model['xty'] += xx.T @ yy
                    else:
                        model = models[k]
                        if 'beta' in model:
                            residual = yy - xx @ model['beta']
                            weighted = xx * residual[:, None]
                            model['meat'] += weighted.T @ weighted
        if pass_number == 1:
            for model in models.values():
                size = len(model['xty'])
                if model['n'] > size and np.linalg.matrix_rank(model['xtx']) == size:
                    model['inv'] = np.linalg.inv(model['xtx'])
                    model['beta'] = model['inv'] @ model['xty']
    description = []
    for (d, g, metric), (n, total, squares) in desc.items():
        mean = total / n if n else None
        se = np.sqrt(max(0, (squares - total * total / n) / (n - 1)) / n) if n > 1 else None
        description.append(dict(domain=d, gender=g, metric=metric, n_users=int(n), mean=mean, se=se,
                                ci_low=mean-1.96*se if se is not None else None, ci_high=mean+1.96*se if se is not None else None))
    contrasts = []
    for (a, b, metric, adjusted), model in models.items():
        row = dict(domain=a, contrast_domain=b, metric=metric, model='activity_adjusted_OLS' if adjusted else 'unadjusted_OLS',
                   n_users=model['n'], estimate_f_minus_m=None, se=None, ci_low=None, ci_high=None, status='insufficient_or_singular')
        if 'beta' in model:
            n, size = model['n'], len(model['beta'])
            cov = model['inv'] @ model['meat'] @ model['inv'] * n / (n-size)
            effect, se = float(model['beta'][1]), np.sqrt(max(0, cov[1, 1]))
            row.update(estimate_f_minus_m=effect, se=se, ci_low=effect-1.96*se, ci_high=effect+1.96*se, status='ok')
        contrasts.append(row)
    for name, rows in [('descriptives', description), ('gender_gaps', [r for r in contrasts if not r['contrast_domain']]),
                       ('cross_domain_contrasts', [r for r in contrasts if r['contrast_domain']]),
                       ('coverage', [dict(gender=g, **c) for g, c in coverage.items()]),
                       ('user_exclusions', [dict(gender=g, conflict=c, users=n) for (g, c), n in exclusions.items()])]:
        pd.DataFrame(rows).to_csv(work / f'{name}.csv', index=False)
    if not wrote_users:
        raise ValueError('没有可分析的有效性别用户；请检查分片诊断，不发布空的成功结果')
    report = {'status': 'complete', 'source_manifests': source, 'unique_posts': unique_posts,
              'scope': 'Observed valid-year users with consistent m/f; institutions not separately excluded.',
              'models': 'User equally weighted OLS/linear probability, HC1; activity adjusted for log1p posts, retweets and active days; female minus male.',
              'delay': 'Mean log1p seconds among positive-lag retweets; not exposure response speed.',
              'architecture': 'Monthly Parquet; batch in-memory processing; user partitions; no database.'}
    (work / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    work.rename(target)
    return str(target)
