"""五领域行为分析：分月批量测量、Parquet分片、内存聚合，无数据库。

python -m cultural_participation.parquet_behavior --help
月份是输入文件的调度单位；保留北京时间发帖年份过滤及原有行为口径。
"""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cultural_participation.behavior import COLUMNS, normalize_id
from cultural_participation.vocabulary import Vocabulary, ascii_token, clean_text, expressive


FORMAT = 'cultural-parquet-v2'
HASH_KEY = 'culturalhash2020'
BASE = ['weibo_id', 'user_id', 'gender', 'day', 'rt', 'expr', 'available']
SUFFIXES = ['retweets', 'expressions', 'delay_n', 'delay_sum', 'commented']


class FastMatcher:
    """C实现的自动机；与原词表接口保持大小写、ASCII边界及重叠命中语义。"""

    def __init__(self, path):
        import ahocorasick

        self.vocab = Vocabulary(path)
        self.domains = list(self.vocab.domains)
        self.bits = {d: 1 << i for i, d in enumerate(self.domains)}
        self.automaton = ahocorasick.Automaton()
        for term in self.vocab.rules:
            self.automaton.add_word(term, term)
        self.automaton.make_automaton()

    @lru_cache(maxsize=20000)
    def match(self, text):
        found = set()
        for end, term in self.automaton.iter(text):
            start = end - len(term) + 1
            if (ascii_token(term[0]) and start and ascii_token(text[start - 1])) or (
                ascii_token(term[-1]) and end + 1 < len(text) and ascii_token(text[end + 1])):
                continue
            rule = self.vocab.rules[term]
            if rule['decision'] == 'standalone' or any(t in text for t in rule['context_any']):
                found.add(term)
        mask = 0
        for term in found:
            for domain in self.vocab.rules[term]['domains']:
                mask |= self.bits[domain]
        # JSON避免词中分隔符歧义；保留逐词命中，供以后重采样。
        return mask, json.dumps(sorted(found), ensure_ascii=False, separators=(',', ':'))


def buckets(series, count):
    return (pd.util.hash_pandas_object(series.astype(str), index=False, hash_key=HASH_KEY).to_numpy() % count).astype(int)


def seconds(series):
    values = pd.to_numeric(series, errors='coerce').astype(float)
    values = values.where(np.isfinite(values) & (values > 0))
    return values.where(values <= 1e11, values / 1000)


def measure_frame(frame, matcher, year):
    """批内清理/匹配；不提前删除重复行，保留所有性别冲突证据。"""
    counts = Counter(input_rows=len(frame))
    frame = frame.copy()
    for name in ['weibo_id', 'user_id']:
        frame[name] = frame[name].map(normalize_id)
    keep = (frame.weibo_id != '') & (frame.user_id != '')
    counts['missing_id'] = int((~keep).sum())
    frame = frame.loc[keep].copy()
    flag = frame.is_retweet.astype(str).str.strip().str.lower()
    keep = flag.isin(['0', '1', '0.0', '1.0', 'true', 'false'])
    counts['invalid_retweet_flag'] = int((~keep).sum())
    frame = frame.loc[keep].copy()
    frame['rt'] = flag.loc[keep].isin(['1', '1.0', 'true']).astype('int8')
    ts = seconds(frame.time_stamp)
    dates = pd.to_datetime(ts, unit='s', utc=True, errors='coerce').dt.tz_convert('Asia/Shanghai')
    keep = dates.dt.year == year
    counts['missing_or_outside_year_timestamp'] = int((~keep).sum())
    frame = frame.loc[keep].copy()
    frame['day'] = dates.loc[keep].dt.strftime('%Y-%m-%d')
    frame['gender'] = frame.gender.astype(str).str.strip().str.lower().map({'m': 'm', 'f': 'f', '男': 'm', '女': 'f'}).fillna('')
    own = frame.weibo_content.map(clean_text)
    source = frame.r_weibo_content.map(clean_text).where(frame.rt.eq(1), '')
    frame['expr'] = own.map(expressive).astype('int8')
    frame['available'] = source.map(expressive).astype('int8')
    lag = ts.loc[keep] - seconds(frame.r_time_stamp)
    valid_lag = frame.rt.eq(1) & lag.gt(0) & np.isfinite(lag)
    frame['lag_status'] = np.select([frame.rt.eq(0), lag.isna(), valid_lag], ['not_retweet', 'missing', 'positive'], default='nonpositive')
    own_matches = [matcher.match(t) if enabled else (0, '[]') for t, enabled in zip(own, frame.expr)]
    source_matches = [matcher.match(t) if enabled else (0, '[]') for t, enabled in zip(source, frame.available)]
    own_mask = np.array([r[0] for r in own_matches], dtype='int64')
    source_mask = np.array([r[0] for r in source_matches], dtype='int64')
    frame['expression_terms'] = [r[1] for r in own_matches]
    frame['retweet_terms'] = [r[1] for r in source_matches]
    frame['matched_retweets'] = (source_mask != 0).astype('int8')
    frame['matched_expression'] = (own_mask != 0).astype('int8')
    for domain, bit in matcher.bits.items():
        rt = ((source_mask & bit) != 0)
        expr = ((own_mask & bit) != 0)
        frame[f'{domain}_retweets'] = rt.astype('int8')
        frame[f'{domain}_expressions'] = expr.astype('int8')
        frame[f'{domain}_delay_n'] = (rt & valid_lag).astype('int8')
        frame[f'{domain}_delay_sum'] = np.log1p(lag.where(rt & valid_lag, 0))
        frame[f'{domain}_commented'] = (rt & expr).astype('int8')
    columns = BASE + ['lag_status', 'matched_retweets', 'matched_expression', 'expression_terms', 'retweet_terms']
    columns += [f'{d}_{s}' for d in matcher.domains for s in SUFFIXES]
    counts['valid_rows_before_dedup'] = len(frame)
    return frame[columns].reset_index(drop=True), counts


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def month(input_dir, vocabulary, output, year, month, bucket_count=64, batch_size=50000):
    """一个月一个任务；每批在内存处理，压缩顺序写出Parquet，不积累原始全文。"""
    if not 1 <= month <= 12 or bucket_count < 1 or batch_size < 1:
        raise ValueError('月份、分片数或批大小无效')
    files = sorted(Path(input_dir).glob(f'{year}-{month:02d}-*.parquet'))
    matcher = FastMatcher(vocabulary)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'month={month:02d}'
    signature = {
        'format': FORMAT, 'year': year, 'month': month, 'bucket_count': bucket_count,
        'vocabulary_sha256': matcher.vocab.fingerprint, 'domains': matcher.domains,
        'pandas_version': pd.__version__, 'hash_key': HASH_KEY, 'git_sha': revision(),
        'inputs': [{'path': str(p.resolve()), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files],
    }
    if target.exists():
        previous = json.loads((target / 'manifest.json').read_text())
        if previous['signature'] != signature or previous['status'] != 'complete':
            raise ValueError(f'已存在不同输入/版本的分片：{target}，请换输出目录')
        print(f'已完成，跳过：{target}', flush=True)
        return str(target)
    # 每个任务独占临时目录，成功后原子发布；失败只重跑该月。
    work = Path(tempfile.mkdtemp(prefix=f'.month-{month:02d}-', dir=root))
    writers = {}
    counts = Counter()
    try:
        for file in files:
            print(f'处理 {file.name}', flush=True)
            reader = pq.ParquetFile(file)
            if set(COLUMNS) - set(reader.schema_arrow.names):
                raise ValueError(f'{file} 缺少字段：{set(COLUMNS) - set(reader.schema_arrow.names)}')
            for batch in reader.iter_batches(batch_size=batch_size, columns=COLUMNS):
                measured, stats = measure_frame(batch.to_pandas(), matcher, year)
                counts.update(stats)
                if measured.empty:
                    continue
                # 用户分片用于全年用户内去重和汇总；所有者轻量分片检查跨用户重复ID。
                for kind, data, key in [('posts', measured, 'user_id'), ('owners', measured[['weibo_id', 'user_id']], 'weibo_id')]:
                    groups = buckets(data[key], bucket_count)
                    for bucket in np.unique(groups):
                        chunk = data.loc[groups == bucket]
                        table = pa.Table.from_pandas(chunk, preserve_index=False)
                        name = f'{kind}-{bucket:03d}.parquet'
                        if name not in writers:
                            writers[name] = pq.ParquetWriter(work / name, table.schema, compression='snappy')
                        writers[name].write_table(table)
            print(f'完成 {file.name}；累计输入{counts["input_rows"]:,}行', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    (work / 'vocabulary.json').write_text(json.dumps(matcher.vocab.data, ensure_ascii=False), encoding='utf-8')
    (work / 'manifest.json').write_text(json.dumps({'status': 'complete', 'signature': signature, 'counts': dict(counts)}, ensure_ascii=False, indent=2))
    work.rename(target)
    return str(target)


def manifests(root):
    values = []
    for month_id in range(1, 13):
        path = Path(root) / f'month={month_id:02d}' / 'manifest.json'
        value = json.loads(path.read_text())
        if value['status'] != 'complete' or value['signature']['month'] != month_id:
            raise ValueError(f'月份未完成：{path}')
        values.append(value)
    shared = [{k: v for k, v in m['signature'].items() if k not in ['month', 'inputs']} for m in values]
    if any(s != shared[0] for s in shared):
        raise ValueError('月份分片的词表、年份、分片参数、代码或pandas版本不一致')
    if not any(m['signature']['inputs'] for m in values):
        raise ValueError('全年没有输入文件，请检查路径及YYYY-MM-DD.parquet命名')
    return values


def load_parts(root, bucket, kind, columns):
    parts = []
    for month_id in range(1, 13):
        path = Path(root) / f'month={month_id:02d}' / f'{kind}-{bucket:03d}.parquet'
        if path.exists():
            for batch in pq.ParquetFile(path).iter_batches(batch_size=50000, columns=columns):
                parts.append(batch.to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def aggregate(frame, domains):
    """全年同一用户落在同一分片；先去重再算年度分母，不平均月比例。"""
    if frame.empty:
        return pd.DataFrame(), {'rows_before_dedup': 0, 'unique_posts': 0, 'user_exclusions': []}
    gender_flags = frame.gender.map({'m': 1, 'f': 2, '': 0})
    flags = pd.DataFrame({'user_id': frame.user_id, 'm': gender_flags.eq(1), 'f': gender_flags.eq(2)}).groupby('user_id', sort=False).max()
    frame = frame.drop_duplicates('weibo_id', keep='first').copy()
    stats = {'unique_posts': len(frame), 'rows_before_dedup': len(gender_flags), 'lag_status': frame.lag_status.value_counts().to_dict()}
    summed = ['rt', 'expr', 'available', 'matched_retweets', 'matched_expression'] + [f'{d}_{s}' for d in domains for s in SUFFIXES]
    # int8命中标记必须先扩展，否则部分聚合/序列化可能溢出。
    frame[summed] = frame[summed].astype('float64')
    result = frame.groupby('user_id', sort=False)[summed].sum()
    result['total_posts'] = frame.groupby('user_id', sort=False).size()
    result['active_days'] = frame.groupby('user_id', sort=False).day.nunique()
    result = result.rename(columns={'rt': 'total_retweets', 'expr': 'total_expression', 'available': 'available_retweets'})
    result['gender'] = np.select([flags.loc[result.index, 'm'] & ~flags.loc[result.index, 'f'], flags.loc[result.index, 'f'] & ~flags.loc[result.index, 'm']], ['m', 'f'], default='')
    result['conflict'] = (flags.loc[result.index, 'm'] & flags.loc[result.index, 'f']).to_numpy()
    excluded = result.groupby(['gender', 'conflict']).size().reset_index(name='users')
    stats['user_exclusions'] = excluded.to_dict('records')
    result = result.loc[result.gender.isin(['m', 'f']) & ~result.conflict].copy()
    for d in domains:
        result[f'{d}_retweet_entry'] = result[f'{d}_retweets'].gt(0).astype(float)
        result[f'{d}_expression_entry'] = result[f'{d}_expressions'].gt(0).astype(float)
        result[f'{d}_retweet_share'] = result[f'{d}_retweets'] / result.total_retweets.replace(0, np.nan)
        result[f'{d}_expression_share'] = result[f'{d}_expressions'] / result.total_expression.replace(0, np.nan)
        result[f'{d}_log_delay'] = result[f'{d}_delay_sum'] / result[f'{d}_delay_n'].replace(0, np.nan)
        result[f'{d}_comment_on_retweet_share'] = result[f'{d}_commented'] / result[f'{d}_retweets'].replace(0, np.nan)
    return result.reset_index(), stats


def reduce_bucket(input_dir, output, bucket):
    meta = manifests(input_dir)
    signature = meta[0]['signature']
    if not 0 <= bucket < signature['bucket_count']:
        raise ValueError('用户分片编号超出范围')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'bucket={bucket:03d}'
    if target.exists():
        previous = json.loads((target / 'manifest.json').read_text())
        if previous.get('status') == 'complete' and previous.get('bucket') == bucket and previous.get('source_manifests') == meta:
            print(f'已完成，跳过：{target}', flush=True)
            return str(target)
        raise ValueError(f'结果与输入不一致，请换输出目录：{target}')
    # 先按帖子ID分片检查跨用户冲突，避免用户分片掩盖原始数据错误。
    owners = load_parts(input_dir, bucket, 'owners', ['weibo_id', 'user_id']).drop_duplicates()
    if owners.weibo_id.duplicated().any():
        raise ValueError('同一帖子ID对应多个用户，请检查原始数据')
    del owners
    domains = signature['domains']
    columns = BASE + ['lag_status', 'matched_retweets', 'matched_expression'] + [f'{d}_{s}' for d in domains for s in SUFFIXES]
    frame = load_parts(input_dir, bucket, 'posts', columns)
    users, stats = aggregate(frame, domains)
    del frame
    work = Path(tempfile.mkdtemp(prefix=f'.bucket-{bucket:03d}-', dir=root))
    if not users.empty:
        users.to_parquet(work / 'users.parquet', index=False)
    (work / 'manifest.json').write_text(json.dumps({'status': 'complete', 'bucket': bucket, 'source_manifests': meta, 'stats': stats}, ensure_ascii=False, indent=2))
    work.rename(target)
    print(f'用户分片{bucket}完成：{len(users):,}名有效用户', flush=True)
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('month')
    p.add_argument('--input-dir', required=True)
    p.add_argument('--vocabulary', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--year', type=int, default=2020)
    p.add_argument('--month', type=int, required=True)
    p.add_argument('--bucket-count', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=50000)
    p = sub.add_parser('reduce')
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bucket', type=int, required=True)
    p = sub.add_parser('summarize')
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'summarize':
        from cultural_participation.parquet_summary import summarize
        print(summarize(**args))
    else:
        print({'month': month, 'reduce': reduce_bucket}[command](**args))


if __name__ == '__main__':
    main()
