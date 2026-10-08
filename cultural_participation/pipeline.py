"""
热搜辅助构建领域词表的准备流程。

逐行读取旧 bangdan 格式，以磁盘数据库保存标题、来源及词条关联。
这里只生成候选词和离线模型分类任务，不生成正式词表或测量参与。
使用方法：python -m cultural_participation --help
"""

import argparse
from collections import Counter
from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3

from cultural_participation.classification import prepare


SCHEMA = """
CREATE TABLE topics (id INTEGER PRIMARY KEY, title TEXT NOT NULL UNIQUE);
CREATE TABLE occurrences (
    topic_id INTEGER NOT NULL, source TEXT NOT NULL, line INTEGER NOT NULL,
    position INTEGER NOT NULL, crawler_time TEXT, board_type TEXT,
    PRIMARY KEY (source, line, position)
);
CREATE TABLE terms (
    term TEXT NOT NULL, topic_id INTEGER NOT NULL, pos TEXT NOT NULL,
    PRIMARY KEY (term, topic_id)
);
CREATE INDEX term_topics ON terms(topic_id);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def decode_line(line, stats, board_type):
    """兼容旧文件的第二列 JSON 及嵌套 bangdan JSON，记录被排除的行。"""
    try:
        parts = line.rstrip("\n").split("\t")
        record = json.loads(parts[1])
        if not isinstance(record, dict):
            raise ValueError("记录不是对象")
        actual_type = record.get("type")
        if board_type != "all" and str(actual_type) != board_type:
            stats["excluded_board_rows"] += 1
            return None
        board = record.get("bangdan")
        if isinstance(board, str):
            board = json.loads(board)
        if not isinstance(board, dict) or not isinstance(board.get("cards"), list):
            raise ValueError("榜单缺少 cards")
        return record, board
    except (IndexError, ValueError, TypeError):
        stats["invalid_rows"] += 1
        return None


def collect(input_dir, pattern, db, board_type="1", max_lines=None):
    """读取已解压文件；不覆盖已有数据库。max_lines 是全局试跑上限。"""
    if max_lines is not None and max_lines < 1:
        raise ValueError("max_lines 必须为正数")
    root = Path(input_dir)
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    if not files:
        raise ValueError("没有匹配的输入文件，请检查 input-dir 和 pattern")
    target = Path(db)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch(exist_ok=False)
    stats = Counter()
    with closing(sqlite3.connect(target)) as conn:
        conn.executescript(SCHEMA)
        with conn:
            for path in files:
                if max_lines is not None and stats["lines"] >= max_lines:
                    break
                stats["files"] += 1
                with path.open(encoding="utf-8", errors="replace") as stream:
                    for line_no, line in enumerate(stream, 1):
                        if max_lines is not None and stats["lines"] >= max_lines:
                            break
                        stats["lines"] += 1
                        parsed = decode_line(line, stats, board_type)
                        if parsed is None:
                            continue
                        record, board = parsed
                        position = 0
                        for card in board["cards"]:
                            if not isinstance(card, dict) or str(card.get("card_type")) != "11":
                                continue
                            group = card.get("card_group")
                            if not isinstance(group, list):
                                stats["invalid_groups"] += 1
                                continue
                            for item in group:
                                position += 1
                                if not isinstance(item, dict) or str(item.get("card_type")) != "4":
                                    continue
                                action = item.get("actionlog") or {}
                                if "ads_word" in str(action):
                                    stats["advertisements"] += 1
                                    continue
                                title = item.get("desc")
                                if not isinstance(title, str) or not title.strip():
                                    stats["empty_titles"] += 1
                                    continue
                                title = title.strip()
                                conn.execute("INSERT OR IGNORE INTO topics(title) VALUES (?)", (title,))
                                topic_id = conn.execute("SELECT id FROM topics WHERE title=?", (title,)).fetchone()[0]
                                conn.execute(
                                    "INSERT INTO occurrences VALUES (?,?,?,?,?,?)",
                                    (topic_id, str(path.resolve()), line_no, position,
                                     str(record.get("crawler_time_stamp", "")), str(record.get("type", ""))),
                                )
                                stats["occurrences"] += 1
            stats["unique_titles"] = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            metadata = {
                "stage": "collected", "input_dir": str(root.resolve()),
                "pattern": pattern, "board_type": board_type, "max_lines": max_lines,
                "stats": dict(stats),
            }
            conn.executemany("INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v, ensure_ascii=False)) for k, v in metadata.items()])
    return dict(stats)


def connect_existing(db):
    """避免把拼错的输入路径静默创建为新库。"""
    return sqlite3.connect(Path(db).resolve().as_uri() + "?mode=rw", uri=True)


def extract(db, min_length=2, pos_prefixes=("n", "v"), tokenizer=None):
    """保留全部候选，不截断 top-N；每个完整标题对每个词只贡献一次。"""
    if min_length < 1 or not pos_prefixes:
        raise ValueError("需要正的 min_length 和非空词性前缀")
    version = "injected"
    if tokenizer is None:
        try:
            import jieba
            import jieba.posseg as pseg
        except ImportError as exc:
            raise ValueError("提词需要 jieba；请在分析环境安装 jieba==0.42.1") from exc

        tokenizer = pseg.cut
        version = jieba.__version__
    with closing(connect_existing(db)) as conn, conn:
        conn.execute("DELETE FROM terms")
        for topic_id, title in conn.execute("SELECT id, title FROM topics ORDER BY id"):
            for word, pos in tokenizer(title):
                word = word.strip()
                if len(word) >= min_length and pos.startswith(tuple(pos_prefixes)):
                    conn.execute("INSERT OR IGNORE INTO terms VALUES (?,?,?)", (word, topic_id, pos))
        settings = {"min_length": min_length, "pos_prefixes": pos_prefixes, "jieba_version": version}
        conn.execute("INSERT OR REPLACE INTO metadata VALUES ('extraction', ?)", (json.dumps(settings),))
        count = conn.execute("SELECT COUNT(DISTINCT term) FROM terms").fetchone()[0]
    return {"candidate_terms": count, "settings": settings}


def export(db, output, min_titles=1):
    """独立导出候选词 CSV 和规模报告；门槛不修改数据库内完整候选集。"""
    if min_titles < 1:
        raise ValueError("min_titles 必须为正数")
    target = Path(output)
    report_path = target.with_suffix(target.suffix + ".summary.json")
    if target.exists() or report_path.exists():
        raise FileExistsError("词表或规模报告已存在，请使用新输出名")
    target.parent.mkdir(parents=True, exist_ok=True)
    lengths = Counter()
    frequencies = Counter()
    thresholds = {n: 0 for n in (1, 2, 3, 5, 10, 20, 50, 100)}
    exported = 0
    all_terms = 0
    term_chars = 0
    context_chars = 0
    with closing(connect_existing(db)) as conn:
        metadata = {k: json.loads(v) for k, v in conn.execute("SELECT key,value FROM metadata")}
        if "extraction" not in metadata:
            raise ValueError("请先执行 extract")
        with target.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["term", "distinct_title_count", "length", "pos_tags", "examples_json"])
            for term, count in conn.execute(
                "SELECT term,COUNT(*) AS n FROM terms GROUP BY term ORDER BY n DESC,term"
            ):
                all_terms += 1
                frequencies[count] += 1
                for threshold in thresholds:
                    thresholds[threshold] += int(count >= threshold)
                if count < min_titles:
                    continue
                tags = sorted({r[0] for r in conn.execute("SELECT DISTINCT pos FROM terms WHERE term=?", (term,))})
                examples = [r[0] for r in conn.execute(
                    "SELECT title FROM topics JOIN terms ON topics.id=terms.topic_id WHERE term=? ORDER BY title LIMIT 3", (term,)
                )]
                writer.writerow([term, count, len(term), "|".join(tags), json.dumps(examples, ensure_ascii=False)])
                exported += 1
                lengths[len(term)] += 1
                term_chars += len(term)
                context_chars += sum(map(len, examples))
    report = {
        "all_candidate_terms": all_terms, "exported_terms": exported,
        "min_titles": min_titles, "exported_length_distribution": dict(sorted(lengths.items())),
        "all_title_frequency_distribution": dict(sorted(frequencies.items())),
        "retained_terms_by_min_titles": thresholds,
        "exported_term_characters": term_chars, "exported_example_characters": context_chars,
        "csv_bytes": target.stat().st_size, "provenance": metadata,
        "budget_note": "字符数不是 token 数或费用；选定模型、提示词及输出结构后再估算预算。",
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return {"all_candidate_terms": all_terms, "exported_terms": exported,
            "output": str(target), "report": str(report_path)}


def main():
    parser = argparse.ArgumentParser(description="热搜辅助领域词表：收集、提词、准备离线分类任务")
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--input-dir", required=True)
    collect_parser.add_argument("--pattern", default="weibo_bangdan.*")
    collect_parser.add_argument("--db", required=True)
    collect_parser.add_argument("--board-type", default="1", help="默认实时榜；all 保留所有类型")
    collect_parser.add_argument("--max-lines", type=int)
    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("--db", required=True)
    extract_parser.add_argument("--min-length", type=int, default=2)
    extract_parser.add_argument("--pos-prefixes", nargs="+", default=["n", "v"])
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--db", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--min-titles", type=int, default=1)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--candidates", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--taxonomy", default=str(Path(__file__).with_name("taxonomy.json")))
    prepare_parser.add_argument("--batch-size", type=int, default=30)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        print(json.dumps({"collect": collect, "extract": extract, "export": export, "prepare": prepare}[command](**args), ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"操作失败：{exc}\n")
