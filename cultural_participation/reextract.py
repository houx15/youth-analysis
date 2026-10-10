"""从旧标题缓存建立独立新版词表，保留旧库；不复制大体积抓取明细。"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from cultural_participation.pipeline import SCHEMA, export, extract


def refresh(source_db, output):
    source_path = Path(source_db).resolve()
    target = Path(output)
    with closing(sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)) as source:
        metadata = dict(source.execute("SELECT key,value FROM metadata"))
        if json.loads(metadata.get("stage", '""')) != "collected":
            raise ValueError("源库未完成标题收集")
        target.mkdir(parents=True, exist_ok=False)
        db = target / "topics.sqlite"
        with closing(sqlite3.connect(db)) as dest:
            dest.executescript(SCHEMA)
            with dest:
                cursor = source.execute("SELECT id,title FROM topics ORDER BY id")
                while True:
                    batch = cursor.fetchmany(1000)
                    if not batch:
                        break
                    dest.executemany("INSERT INTO topics VALUES (?,?)", batch)
                dest.executemany("INSERT INTO metadata VALUES (?,?)", [(k, v) for k, v in metadata.items() if k not in {"extraction", "reextraction"}])
                lineage = {"source_db": str(source_path), "source_bytes": source_path.stat().st_size,
                           "source_mtime_ns": source_path.stat().st_mtime_ns,
                           "copied": "unique titles only; occurrences and their provenance remain in source_db"}
                dest.execute("INSERT INTO metadata VALUES ('reextraction',?)", (json.dumps(lineage),))
    print("标题缓存已另存，开始新版提词", flush=True)
    result = extract(db)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return export(db, target / "candidates.csv")
