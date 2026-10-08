"""用户等权的跨领域比较和词表子抽样；SQLite 中汇总，全程不加载微博全表。"""

from contextlib import contextmanager
import csv
import itertools
import json
import math
from pathlib import Path
import random
import sqlite3


METRICS = ["retweet_entry", "expression_entry", "retweet_share", "expression_share", "log_delay", "comment_on_retweet_share"]


@contextmanager
def database(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-32768")
    conn.create_function("log1p", 1, lambda x: math.log1p(x) if x is not None else None)
    try:
        manifest = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if not manifest or json.loads(manifest[0]).get("status") != "complete":
            raise ValueError("命中缓存尚未完整构建")
        conn.executescript("""
        CREATE TEMP TABLE totals AS SELECT p.user_id,u.gender,COUNT(*) total_posts,SUM(rt) total_retweets,
          SUM(expr) total_expression,SUM(source_available) available_retweets,COUNT(DISTINCT day) active_days
          FROM posts p JOIN users u USING(user_id) WHERE u.gender IN ('m','f') AND u.conflict=0 GROUP BY p.user_id;
        CREATE UNIQUE INDEX temp.total_user ON totals(user_id);
        CREATE TEMP TABLE domains AS SELECT DISTINCT domain FROM terms;
        CREATE TEMP TABLE retained(term TEXT PRIMARY KEY);
        """)
        yield conn
    finally:
        conn.close()


def materialize(conn, terms):
    """保留完整的重叠命中缓存，子集选择不会丢失被长词遮蔽的短词。"""
    conn.execute("DELETE FROM retained")
    conn.executemany("INSERT INTO retained VALUES (?)", [(t,) for t in terms])
    conn.executescript("""
    DROP TABLE IF EXISTS temp.matched;
    DROP TABLE IF EXISTS temp.domain_counts;
    DROP TABLE IF EXISTS temp.user_domain;
    CREATE TEMP TABLE matched AS SELECT h.post_id,t.domain,
      MAX(h.kind='retweet') rt_hit,MAX(h.kind='expression') expr_hit
      FROM hits h JOIN retained r USING(term) JOIN terms t USING(term) GROUP BY h.post_id,t.domain;
    CREATE INDEX temp.matched_post ON matched(post_id);
    CREATE TEMP TABLE domain_counts AS SELECT p.user_id,m.domain,SUM(rt_hit) retweets,SUM(expr_hit) expressions,
      SUM(CASE WHEN rt_hit AND lag_status='positive' THEN 1 ELSE 0 END) delay_n,
      AVG(CASE WHEN rt_hit AND lag_status='positive' THEN log1p(lag) END) log_delay,
      SUM(rt_hit*expr_hit) expressed_retweets
      FROM matched m JOIN posts p USING(post_id) GROUP BY p.user_id,m.domain;
    CREATE UNIQUE INDEX temp.counts_user_domain ON domain_counts(user_id,domain);
    CREATE TEMP TABLE user_domain AS SELECT u.*,d.domain,
      COALESCE(c.retweets,0) retweet_count,COALESCE(c.expressions,0) expression_count,
      COALESCE(c.delay_n,0) delay_n,c.log_delay,
      CAST(COALESCE(c.retweets,0)>0 AS REAL) retweet_entry,
      CAST(COALESCE(c.expressions,0)>0 AS REAL) expression_entry,
      1.0*COALESCE(c.retweets,0)/NULLIF(u.total_retweets,0) retweet_share,
      1.0*COALESCE(c.expressions,0)/NULLIF(u.total_expression,0) expression_share,
      1.0*COALESCE(c.expressed_retweets,0)/NULLIF(c.retweets,0) comment_on_retweet_share
      FROM totals u CROSS JOIN domains d LEFT JOIN domain_counts c ON u.user_id=c.user_id AND d.domain=c.domain;
    CREATE UNIQUE INDEX temp.ud_user_domain ON user_domain(user_id,domain);
    """)


def write_query(conn, query, path, parameters=()):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        cursor = conn.execute(query, parameters)
        writer = csv.writer(stream)
        writer.writerow([c[0] for c in cursor.description])
        writer.writerows(cursor)


def descriptives(conn):
    for metric in METRICS:
        for domain, gender, n, mean, total, squares in conn.execute(f"SELECT domain,gender,COUNT({metric}),AVG({metric}),SUM({metric}),SUM({metric}*{metric}) FROM user_domain GROUP BY domain,gender"):
            variance = max(0, (squares - total * total / n) / (n - 1)) if n and n > 1 else None
            se = math.sqrt(variance / n) if variance is not None else None
            yield {"domain": domain, "gender": gender, "metric": metric, "n_users": n, "mean": mean,
                   "se": se, "ci_low": mean - 1.96 * se if se is not None else None,
                   "ci_high": mean + 1.96 * se if se is not None else None}


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        iterator = iter(rows)
        first = next(iterator, None)
        if first is None:
            return
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def ols(conn, query, parameters, adjusted):
    """流式 OLS + HC1；每个用户一行。返回女性减男性差异，非因果效应。"""
    import numpy as np

    size = 5 if adjusted else 2
    xtx = np.zeros((size, size))
    xty = np.zeros(size)
    n = 0

    def rows():
        for outcome, gender, posts, retweets, days in conn.execute(query, parameters):
            if outcome is None:
                continue
            x = [1, float(gender == "f")]
            if adjusted:
                x.extend([math.log1p(posts), math.log1p(retweets), math.log1p(days)])
            yield float(outcome), np.array(x)

    for y, x in rows():
        xtx += np.outer(x, x)
        xty += x * y
        n += 1
    if n <= size or np.linalg.matrix_rank(xtx) < size:
        return {"n_users": n, "estimate_f_minus_m": None, "se": None, "ci_low": None, "ci_high": None, "status": "insufficient_or_singular"}
    inv = np.linalg.inv(xtx)
    beta = inv @ xty
    meat = np.zeros_like(xtx)
    for y, x in rows():
        meat += np.outer(x, x) * (y - x @ beta) ** 2
    covariance = inv @ meat @ inv * n / (n - size)
    se = math.sqrt(max(0, float(covariance[1, 1])))
    effect = float(beta[1])
    return {"n_users": n, "estimate_f_minus_m": effect, "se": se, "ci_low": effect - 1.96 * se, "ci_high": effect + 1.96 * se, "status": "ok"}


def contrasts(conn, paired=False):
    domains = [r[0] for r in conn.execute("SELECT domain FROM domains ORDER BY domain")]
    comparisons = itertools.combinations(domains, 2) if paired else [(d, "") for d in domains]
    for a, b in comparisons:
        for metric in METRICS:
            if paired:
                query = f"SELECT a.{metric}-b.{metric},a.gender,a.total_posts,a.total_retweets,a.active_days FROM user_domain a JOIN user_domain b USING(user_id) WHERE a.domain=? AND b.domain=?"
                params = (a, b)
            else:
                query = f"SELECT {metric},gender,total_posts,total_retweets,active_days FROM user_domain WHERE domain=?"
                params = (a,)
            for adjusted in [False, True]:
                yield {"domain": a, "contrast_domain": b, "metric": metric, "model": "activity_adjusted_OLS" if adjusted else "unadjusted_OLS", **ols(conn, query, params, adjusted)}


def summarize(db, output):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    with database(db) as conn:
        terms = [r[0] for r in conn.execute("SELECT DISTINCT term FROM terms")]
        materialize(conn, terms)
        write_query(conn, "SELECT * FROM user_domain ORDER BY user_id,domain", target / "user_domain.csv")
        write_rows(target / "descriptives.csv", descriptives(conn))
        write_rows(target / "gender_gaps.csv", contrasts(conn))
        write_rows(target / "cross_domain_contrasts.csv", contrasts(conn, paired=True))
        write_query(conn, """SELECT t.gender,COUNT(*) n_users,SUM(total_posts) posts,SUM(total_retweets) retweets,
          SUM(available_retweets) available_retweets,SUM(total_expression) expressive_posts,
          SUM(COALESCE(c.matched_retweets,0)) matched_retweets,SUM(COALESCE(c.matched_expression,0)) matched_expression
          FROM totals t LEFT JOIN (
            SELECT p.user_id,COUNT(DISTINCT CASE WHEN m.rt_hit THEN m.post_id END) matched_retweets,
              COUNT(DISTINCT CASE WHEN m.expr_hit THEN m.post_id END) matched_expression
            FROM matched m JOIN posts p USING(post_id) GROUP BY p.user_id
          ) c USING(user_id) GROUP BY t.gender""", target / "coverage.csv")
        write_query(conn, "SELECT gender,conflict,COUNT(*) users FROM users GROUP BY gender,conflict", target / "user_exclusions.csv")
        source_manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
    report = {"source_manifest": source_manifest, "units": "user equally weighted", "gap": "female minus male",
              "delay": "mean log(1+seconds) among positive-lag observed retweets; paired contrasts require both domains",
              "models": "descriptive OLS/linear probability, HC1; adjusted for log1p posts, retweets, active days; not original logit AME pipeline",
              "scope": "observed valid-year users with consistent m/f labels; not all platform users; institutions not yet separately identified"}
    (target / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(target)}


def subsample(db, output, fractions=(0.5, 0.8), repeats=100, seed=2020):
    if repeats < 1 or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("需要正 repeats 和 (0,1] 范围的 fractions")
    fractions = [float(f) for f in fractions]
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    with database(db) as conn, (target / "draws.jsonl").open("x", encoding="utf-8") as log:
        terms = [r[0] for r in conn.execute("SELECT DISTINCT term FROM terms ORDER BY term")]

        def results():
            for fraction, repeat in [(1.0, -1)] + [(f, r) for f in fractions for r in range(repeats)]:
                kept = terms if repeat == -1 else sorted(rng.sample(terms, max(1, math.floor(len(terms) * fraction))))
                log.write(json.dumps({"fraction": fraction, "repeat": repeat, "terms": kept}, ensure_ascii=False) + "\n")
                materialize(conn, kept)
                for row in descriptives(conn):
                    yield {"fraction": fraction, "repeat": repeat, "kept_terms": len(kept), **row}
        write_rows(target / "sensitivity.csv", results())
    (target / "manifest.json").write_text(json.dumps({"db": str(Path(db).resolve()), "seed": seed, "fractions": fractions, "repeats": repeats,
        "method": "global vocabulary sampling without replacement; not confidence intervals; a domain can lose all terms"}, indent=2), encoding="utf-8")
    return {"output": str(target)}
