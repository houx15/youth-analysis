"""从固定配置顺序执行词表准备；Shell 仅负责激活环境和启动 Python。"""

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

from cultural_participation.pipeline import collect, export, extract


def read_config(path):
    values = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        if len(parts) != 1 or "=" not in parts[0]:
            raise ValueError(f"配置第{number}行需要 KEY=VALUE 格式")
        key, value = parts[0].split("=", 1)
        if key in values or key not in {"INPUT_DIR", "RUN_LABEL", "MAX_LINES"}:
            raise ValueError(f"配置字段重复或未知：{key}")
        values[key] = value
    if set(values) != {"INPUT_DIR", "RUN_LABEL", "MAX_LINES"}:
        raise ValueError("配置需要 INPUT_DIR、RUN_LABEL、MAX_LINES")
    if not values["INPUT_DIR"] or not re.fullmatch(r"[a-zA-Z0-9_-]+", values["RUN_LABEL"]):
        raise ValueError("输入目录不能为空；运行名称只允许英文字母、数字、下划线和连字符")
    if not re.fullmatch(r"0|[1-9][0-9]*", values["MAX_LINES"]):
        raise ValueError("MAX_LINES 必须是非负整数，0表示全量")
    return values


def run_job(repo_root=None):
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    config = read_config(root / "cultural_participation/vocabulary_job.conf")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("请通过 sbatch prepare_cultural_vocabulary.sh 提交")
    if importlib.util.find_spec("jieba") is None:
        raise ValueError("当前 Python 环境缺少 jieba，请确认 opinion 环境已激活")
    source = Path(config["INPUT_DIR"])
    if not source.is_absolute():
        source = root / source
    target = root / "gender_norms/newspaper_data/cultural_participation" / f'{config["RUN_LABEL"]}_{job_id}'
    max_lines = int(config["MAX_LINES"]) or None
    print(f"输入目录：{source}\n输出目录：{target}\n最多读取行数：{max_lines or '全量'}", flush=True)
    target.mkdir(parents=True, exist_ok=False)
    db = target / "topics.sqlite"
    print("步骤1/3：收集热搜标题", flush=True)
    print(json.dumps(collect(source, "weibo_bangdan.*", db, max_lines=max_lines), ensure_ascii=False), flush=True)
    print("步骤2/3：提取候选词", flush=True)
    print(json.dumps(extract(db), ensure_ascii=False), flush=True)
    print("步骤3/3：导出词表和规模报告", flush=True)
    print(json.dumps(export(db, target / "candidates.csv"), ensure_ascii=False), flush=True)
