"""后续分析独立入口：python -m cultural_participation.research --help。"""

import argparse
import json

from cultural_participation.analysis import subsample, summarize
from cultural_participation.behavior import build
from cultural_participation.semantics import score, survey_check


def main():
    parser = argparse.ArgumentParser(description="内容行为、跨领域比较、词表敏感性及语义评价")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--vocabulary", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--pattern", default="*.parquet")
    p = sub.add_parser("summarize")
    p.add_argument("--db", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("subsample")
    p.add_argument("--db", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fractions", nargs="+", type=float, default=[0.5, 0.8])
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=2020)
    p = sub.add_parser("score")
    for name in ["vectors", "axes", "objects", "output"]:
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("survey-check")
    for name in ["scores", "ratings", "output"]:
        p.add_argument(f"--{name}", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    print(json.dumps({"build": build, "summarize": summarize, "subsample": subsample, "score": score, "survey-check": survey_check}[command](**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
