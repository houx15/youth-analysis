"""抽查两个2020年已清理微博文件：只读元数据和有界首批样本，不导出正文或用户ID。

运行：python -m cultural_participation.inspect_cleaned_source
输出目录内含报告和CSV清单；同时生成同名tar.gz供下载。
日期来自路径，仅用于文件覆盖清点，不等同于发帖日期覆盖。
"""
import argparse
import csv
import hashlib
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

KEYS = ['weibo_id', 'user_id', 'weibo_content', 'is_retweet', 'time_stamp',
        'r_weibo_id', 'r_user_id', 'r_weibo_content', 'r_time_stamp',
        'gender', 'birthday', 'province', 'user_type', 'verified_type']
SAFE_VALUES = {'is_retweet', 'gender', 'user_type', 'verified_type'}


def inspect(root, output, sample_rows=256, samples_per_year=4):
    import pyarrow.parquet as pq

    root, output = Path(root), Path(output)
    if not root.is_dir():
        raise FileNotFoundError(root)
    output.mkdir(parents=True, exist_ok=False)
    # 仅列文件名选取两个文件，不读取其余文件元数据或内容。
    paths = sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                   and '2020' in str(p.relative_to(root)) and '.parquet' in p.name)
    if not paths:
        raise FileNotFoundError('未找到路径含2020的Parquet文件（含.parquet.temp_*分片）')
    normal = [p for p in paths if '2020-06-30' not in str(p)]
    special = [p for p in paths if '2020-06-30' in str(p)]
    selected = {normal[0] if normal else paths[0]}
    if special:
        selected.add(special[0])
    if len(selected) < 2:
        selected.update(next(([p] for p in paths if p not in selected), []))
    print('仅抽查：' + ', '.join(str(p.relative_to(root)) for p in sorted(selected)), flush=True)
    schemas, totals = {}, defaultdict(Counter)
    candidates = defaultdict(list)
    dates = defaultdict(set)
    issues = []
    unassigned = 0
    with (output / 'files.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'year', 'date', 'bytes', 'rows', 'row_groups', 'schema', 'status'])
        writer.writeheader()
        count = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs.sort()
            for name in sorted(files):
                path = Path(directory) / name
                if path not in selected:
                    continue
                rel = str(path.relative_to(root))
                match = re.search(r'(?<!\d)(202[0-5])-([01]\d)-([0-3]\d)(?!\d)', rel)
                year_match = re.search(r'(?<!\d)(202[0-5])(?!\d)', rel)
                if not year_match:
                    unassigned += 1
                    continue
                year = year_match[1]
                date = match[0] if match else ''
                row = dict(path=rel, year=year, date=date, bytes='', rows='', row_groups='', schema='', status='')
                totals[year]['files'] += 1
                try:
                    row['bytes'] = path.stat().st_size
                    totals[year]['bytes'] += row['bytes']
                    # 检查magic而非扩展名：兼容 .parquet.temp_1 等旧输出。
                    with path.open('rb') as src:
                        magic = src.read(4)
                    if magic != b'PAR1':
                        row['status'] = 'not_parquet'
                        totals[year]['not_parquet'] += 1
                    else:
                        with pq.ParquetFile(path) as pf:
                            fields = [{'name': f.name, 'type': str(f.type)} for f in pf.schema_arrow]
                            sid = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]
                            schemas[sid] = fields
                            row.update(rows=pf.metadata.num_rows, row_groups=pf.metadata.num_row_groups, schema=sid, status='ok')
                        totals[year]['parquet_files'] += 1
                        totals[year]['metadata_rows_before_dedup'] += row['rows']
                        candidates[(year, sid)].append(rel)
                        if date:
                            try:
                                datetime.strptime(date, '%Y-%m-%d')
                                dates[year].add(date)
                            except ValueError:
                                issues.append({'path': rel, 'error': 'invalid filename date'})
                except Exception as exc:
                    row['status'] = 'error'
                    totals[year]['errors'] += 1
                    issues.append({'path': rel, 'error_type': type(exc).__name__})
                writer.writerow(row)
                count += 1
                if count % 100 == 0:
                    print(f'已检查{count}个文件（只读元数据）', flush=True)
    samples = []
    for (year, sid), paths in sorted(candidates.items()):
        paths.sort()
        # 每年每种schema均匀抽文件；额外覆盖已知特殊日期。
        n = min(samples_per_year, len(paths))
        picked = {paths[round(i * (len(paths)-1) / max(1, n-1))] for i in range(n)}
        special = [p for p in paths if '2020-06-30' in p]
        picked.update(special[:1])
        for rel in sorted(picked):
            item = {'path': rel, 'year': year, 'schema': sid}
            try:
                with pq.ParquetFile(root / rel) as pf:
                    names = pf.schema_arrow.names
                    cols = [c for c in KEYS if c in names]
                    item['missing_key_columns'] = [c for c in KEYS if c not in names]
                    batch = next(pf.iter_batches(batch_size=sample_rows, columns=cols), None) if cols else None
                    item['sample_rows'] = batch.num_rows if batch is not None else 0
                    stats = {}
                    if batch is not None:
                        for col in cols:
                            values = batch.column(batch.schema.get_field_index(col)).to_pylist()
                            strings = [str(v) for v in values if v is not None]
                            s = {'null_count': sum(v is None for v in values),
                                 'blank_count': sum(not v.strip() for v in strings),
                                 'max_string_length': max(map(len, strings), default=0)}
                            if col in SAFE_VALUES:
                                # 仅输出短编码；不输出自由文本。
                                s['short_code_counts'] = dict(Counter(v for v in strings if re.fullmatch(r'[A-Za-z0-9_-]{1,12}|男|女', v)).most_common(20))
                            if col.endswith('time_stamp'):
                                s['numeric_digit_lengths'] = dict(Counter(str(len(v.lstrip('-'))) for v in strings if re.fullmatch(r'-?\d+', v)))
                            stats[col] = s
                    item['columns'] = stats
            except Exception as exc:
                item['error_type'] = type(exc).__name__
            samples.append(item)
    report = {'root': str(root), 'years': dict(sorted(totals.items())),
              'unassigned_year_files': unassigned, 'schemas': schemas,
              'filename_dates': {y: sorted(d) for y, d in sorted(dates.items())},
              'samples': samples, 'issues': issues,
              'limitations': ['仅抽查至多两个2020年文件，不能确认全年或跨年格式一致；表中数量仅为抽查文件。', '元数据行数包含重复记录，不是唯一微博数。',
                              '文件名日期不是正文时间戳覆盖；抽样是所选文件首批，非随机总体样本。',
                              '缺失画像字段不意味着不可分析，需要另接原始画像。',
                              '没有读取完整正文、计算全量用户数或执行去重。']}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 两个2020年文件的结构抽查', '', '|年份|文件数|Parquet数|元数据行数（未去重）|GB|错误|', '|---|---:|---:|---:|---:|---:|']
    for year, t in sorted(totals.items()):
        lines.append(f"|{year}|{t['files']}|{t['parquet_files']}|{t['metadata_rows_before_dedup']:,}|{t['bytes']/1e9:.2f}|{t['errors']}|")
    lines += ['', *report['limitations'], '', '详细列名、类型、字段缺失及抽样检查见 report.json；逐文件清单见 files.csv。']
    (output / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    archive = Path(str(output) + '.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz') as tf:
        for name in ['report.json', 'files.csv', 'README.md']:
            tf.add(output / name, arcname=f'{output.name}/{name}')
    print(f'检查完成：{archive}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/gpfs/share/home/2401111059/cleaned_weibo_data')
    parser.add_argument('--output', default='gender_norms/newspaper_data/cleaned_source_inspection_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--sample-rows', type=int, default=256)
    parser.add_argument('--samples-per-year', type=int, default=4, help='每年每种schema抽查文件数')
    args = parser.parse_args()
    if not 1 <= args.sample_rows <= 1024 or not 1 <= args.samples_per_year <= 12:
        parser.error('sample-rows须为1–1024，samples-per-year须为1–12')
    inspect(args.root, args.output, args.sample_rows, args.samples_per_year)


if __name__ == '__main__':
    main()
