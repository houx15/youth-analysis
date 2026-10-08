"""分批读取原始微博，建立可追溯的内容命中缓存；所有大规模运行须经 SLURM。"""

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import subprocess
from zoneinfo import ZoneInfo

from cultural_participation.vocabulary import Vocabulary, clean_text, expressive


COLUMNS = ["weibo_id", "user_id", "gender", "is_retweet", "weibo_content", "r_weibo_content", "time_stamp", "r_time_stamp"]
SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE users(user_id TEXT PRIMARY KEY, gender TEXT, conflict INTEGER DEFAULT 0);
CREATE TABLE posts(post_id TEXT PRIMARY KEY,user_id TEXT,rt INTEGER,expr INTEGER,source_available INTEGER,
 day TEXT,lag REAL,lag_status TEXT);
CREATE TABLE hits(post_id TEXT,kind TEXT,term TEXT,PRIMARY KEY(post_id,kind,term));
CREATE TABLE terms(term TEXT,domain TEXT,kind TEXT,PRIMARY KEY(term,domain));
CREATE INDEX posts_user ON posts(user_id);
CREATE INDEX hits_term ON hits(term,kind,post_id);
"""


def normalize_id(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) > 2 ** 53 or not value.is_integer():
            raise ValueError("浮点 ID 精度不安全")
        return str(int(value))
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "null"}:
        return ""
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def timestamp(value):
    try:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            return None
        return number / 1000 if number > 1e11 else number
    except (TypeError, ValueError):
        return None


def parquet_rows(files, batch_size):
    import pyarrow.parquet as pq

    for file in files:
        reader = pq.ParquetFile(file)
        missing = set(COLUMNS) - set(reader.schema_arrow.names)
        if missing:
            raise ValueError(f"{file} 缺少必要字段：{sorted(missing)}")
        for batch in reader.iter_batches(batch_size=batch_size, columns=COLUMNS):
            yield from batch.to_pylist()


def build_records(records, vocabulary, output, year, provenance=None):
    """按全局 post_id 去重，冲突性别用户整用户排除由汇总阶段实施。"""
    vocab = Vocabulary(vocabulary)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    (target / "vocabulary.json").write_text(json.dumps(vocab.data, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter()
    with closing(sqlite3.connect(target / "behavior.sqlite")) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO terms VALUES (?,?,?)", [(term, d, row.get("kind", "unspecified")) for term, row in vocab.rules.items() for d in row["domains"]])
        conn.commit()
        for row in records:
            counts["input_rows"] += 1
            uid, pid = normalize_id(row.get("user_id")), normalize_id(row.get("weibo_id"))
            if not uid or not pid:
                counts["missing_id"] += 1
                continue
            flag = str(row.get("is_retweet")).strip().lower()
            if flag not in {"0", "1", "0.0", "1.0", "true", "false"}:
                counts["invalid_retweet_flag"] += 1
                continue
            ts = timestamp(row.get("time_stamp"))
            try:
                day = datetime.fromtimestamp(ts, ZoneInfo("Asia/Shanghai")) if ts else None
            except (ValueError, OverflowError, OSError):
                day = None
            if day is None or day.year != year:
                counts["missing_or_outside_year_timestamp"] += 1
                continue
            gender = {"m": "m", "f": "f", "男": "m", "女": "f"}.get(str(row.get("gender")).strip().lower(), "")
            conn.execute("""INSERT INTO users(user_id,gender) VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET
            conflict=MAX(conflict,CASE WHEN gender!='' AND excluded.gender!='' AND gender!=excluded.gender THEN 1 ELSE 0 END),
            gender=CASE WHEN gender='' THEN excluded.gender ELSE gender END""", (uid, gender))
            rt = int(flag in {"1", "1.0", "true"})
            own = clean_text(row.get("weibo_content"))
            expr = int(expressive(own))
            source = clean_text(row.get("r_weibo_content")) if rt else ""
            available = int(rt and expressive(source))
            src_ts = timestamp(row.get("r_time_stamp"))
            lag = ts - src_ts if rt and src_ts is not None else None
            status = "not_retweet" if not rt else ("missing" if lag is None else ("positive" if lag > 0 else "nonpositive"))
            inserted = conn.execute("INSERT OR IGNORE INTO posts VALUES (?,?,?,?,?,?,?,?)", (pid, uid, rt, expr, available, day.date().isoformat(), lag, status)).rowcount
            if not inserted:
                previous = conn.execute("SELECT user_id FROM posts WHERE post_id=?", (pid,)).fetchone()[0]
                if previous != uid:
                    raise ValueError("同一帖子 ID 对应多个用户，请先检查原始数据")
                counts["duplicate_posts"] += 1
                continue
            counts["unique_posts"] += 1
            counts[f"lag_{status}"] += 1
            counts["retweets_missing_source"] += int(rt and not available)
            for kind, text, enabled in [("expression", own, expr), ("retweet", source, available)]:
                if enabled:
                    conn.executemany("INSERT INTO hits VALUES (?,?,?)", [(pid, kind, term) for term in vocab.match(text)])
            if counts["unique_posts"] % 10000 == 0:
                conn.commit()
        manifest = {"status": "complete", "year": year, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "vocabulary_sha256": vocab.fingerprint, "counts": dict(counts), "provenance": provenance or {},
                    "matching": "all overlapping literal terms; any-hit binary domain participation", "timezone": "Asia/Shanghai"}
        conn.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(manifest, ensure_ascii=False),))
        conn.commit()
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build(input_dir, vocabulary, output, year=2020, batch_size=5000, pattern="*.parquet"):
    if batch_size < 1:
        raise ValueError("batch_size 必须为正数")
    files = sorted(Path(input_dir).glob(pattern))
    if not files:
        raise ValueError("没有匹配的 parquet 文件")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "cultural_participation"], check=False).returncode != 0 if revision != "unknown" else None
    provenance = {"git_sha": revision, "module_tracked_dirty": dirty, "files": [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files], "batch_size": batch_size, "python_reading": "pyarrow.iter_batches"}
    return build_records(parquet_rows(files, batch_size), vocabulary, output, year, provenance)
