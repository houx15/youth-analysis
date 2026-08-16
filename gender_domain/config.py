"""
路径、词表/账号名单加载与运行溯源信息。

所有正式词表与账号名单只有一个权威来源：configs/。
每次正式运行都要写出 manifest.json，记录代码版本、输入、词表指纹和逐步样本数。
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime

# 服务器上的输入层与新分析表输出目录
DATA_DIR = "cleaned_weibo_cov"
OUTPUT_DIR = "analysis_data"
CONFIG_DIR = "configs"
YEAR = 2020

PUBLIC_VOCAB_TEMPLATE = os.path.join(CONFIG_DIR, "news_vocabulary_{year}.txt")
CELEBRITY_VOCAB_TEMPLATE = os.path.join(CONFIG_DIR, "entertainment_nouns_{year}.txt")
PUBLIC_ACCOUNTS_FILE = os.path.join(CONFIG_DIR, "news_user_ids.json")
CELEBRITY_ACCOUNTS_FILE = os.path.join(CONFIG_DIR, "entertain_user_ids.json")


def fingerprint_terms(terms):
    """对词表内容生成 8 位指纹，与顺序和首尾空白无关"""
    normalized = sorted({t.strip() for t in terms if t and t.strip()})
    joined = "\n".join(normalized).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:8]


def _read_term_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到词表文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_public_vocabulary(year=YEAR):
    """公共事务词表（人工审核定稿，不再筛词）"""
    return _read_term_file(PUBLIC_VOCAB_TEMPLATE.format(year=year))


def load_celebrity_vocabulary(year=YEAR):
    """明星议题词表（人工审核定稿，不再筛词）"""
    return _read_term_file(CELEBRITY_VOCAB_TEMPLATE.format(year=year))


def load_source_accounts(domain):
    """加载来源账号，返回 {类别: {user_id 字符串}}"""
    if domain not in ("public", "celebrity"):
        raise ValueError(f"未知的 domain: {domain}")
    path = PUBLIC_ACCOUNTS_FILE if domain == "public" else CELEBRITY_ACCOUNTS_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到来源账号文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {category: {str(uid) for uid in ids} for category, ids in data.items()}


def _git_revision():
    """返回 (git_sha, git_dirty)，用于 manifest 的可复现性声明

    git_sha 是整份 manifest 唯一的"这份数字是哪一版代码跑出来的"证据，
    所以两件事都不能静默：
    1) 取不到 SHA（不在 git 仓库里、git 不可用）必须打印中文警告，否则
       manifest 里一个 "unknown" 混在正常字段里很容易被当成正常值；
    2) 工作区有未提交改动时，SHA 指向的代码和真正跑出这批数字的代码并不
       相同——不记录这一点的话，脏工作区跑出来的 manifest 与干净工作区
       跑出来的完全无法区分。git_dirty 为 True 表示这次运行不可复现。
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as exc:
        print(f"警告: 无法获取 git 版本号（{exc.__class__.__name__}: {exc}），"
              "manifest 的 git_sha 记为 unknown，本次运行无法追溯到具体代码版本")
        return "unknown", "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
    except Exception as exc:
        print(f"警告: 无法判断工作区是否有未提交改动（{exc.__class__.__name__}: {exc}），"
              "git_dirty 记为 unknown")
        return sha, "unknown"

    dirty = bool(status.strip())
    if dirty:
        print(f"警告: 工作区存在未提交改动，本次运行的代码与 {sha} 并不完全一致，"
              "manifest 已记录 git_dirty=true")
    return sha, dirty


# ---------------------------------------------------------------------------
# 运行标识（run_id）与结果表的"同一批次"校验
# ---------------------------------------------------------------------------

# 一次作业里所有步骤共用的运行标识从这个环境变量读，由 slurm/run_results.slurm
# 在启动时导出一次。六个步骤是六个独立的 python 进程，只有环境变量能把它们
# 串成"同一批"。
RUN_ID_ENV = "GENDER_DOMAIN_RUN_ID"
# 结果目录下记录"每张结果表属于哪一次运行"的小文件。它与各步骤的 manifest
# 子目录互补：manifest 记的是"这一步跑了什么"，这里记的是"盘上这张表是
# 哪一次运行留下的"，导出层据此判断有没有把上周的数字和本周的混在一起。
RUN_STAMP_FILE = "run_stamps.json"

_PROCESS_RUN_ID = None


def run_id():
    """本次运行的标识：优先读环境变量，缺失时按进程生成一个并告警

    为什么必须有：`slurm/run_results.slurm` 里六个步骤是六个独立进程，
    中间任何一步崩掉，后面的步骤要么不再运行（set -e 之后），要么在盘上
    留下"新旧混装"的一组结果表——旧的那几张看起来完全正常，文件也在，
    导出层只检查存在性就会照单全收，最后一张图上会同时出现上周的估计与
    本周的样本量。运行标识就是让这种混装无处可藏的那一个字段。

    环境变量缺失时（本地手工跑单个模块）退化为"每个进程一个新标识"，
    并打印中文告警：这时分步跑出来的结果表天然带着不同的标识，导出层会
    当场报错——这正是想要的行为，要分步跑就必须自己先把
    GENDER_DOMAIN_RUN_ID 导出一次，把"这几步属于同一批"这件事说出来。
    """
    global _PROCESS_RUN_ID
    from_env = os.environ.get(RUN_ID_ENV, "").strip()
    if from_env:
        return from_env
    if _PROCESS_RUN_ID is None:
        _PROCESS_RUN_ID = "local-{}-{}".format(
            datetime.now().strftime("%Y%m%d%H%M%S"), os.getpid()
        )
        print(
            f"警告: 环境变量 {RUN_ID_ENV} 未设置，本进程自行生成运行标识 "
            f"{_PROCESS_RUN_ID}。分步手工运行时，每个步骤都会拿到不同的标识，"
            "导出图数据时会因为结果表来自不同批次而报错。要分步跑请先 "
            f"`export {RUN_ID_ENV}=$(date +%Y%m%d%H%M%S)`，把"
            "'这几步属于同一批'这件事显式说出来。"
        )
    return _PROCESS_RUN_ID


def _stamp_path(results_dir):
    return os.path.join(results_dir, RUN_STAMP_FILE)


def read_run_stamps(results_dir):
    """读取结果目录的运行标识记录；文件不存在时返回空 dict"""
    path = _stamp_path(results_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stamp_result_files(results_dir, filenames, step):
    """把本次运行标识写到这些结果表名下（增量更新，不覆盖别的表的记录）

    刻意写成一份并排的小 JSON，而不是往 parquet 里塞一列：结果表的列集合
    是 stats_utils.RESULT_SCHEMA 那一份共享 schema，多一列就得让所有下游
    跟着改；而"这张表是哪一次运行写的"本来就是文件级别的元信息。
    """
    stamps = read_run_stamps(results_dir)
    git_sha, git_dirty = _git_revision()
    current = run_id()
    for name in filenames:
        stamps[os.path.basename(name)] = {
            "run_id": current,
            "step": step,
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": git_sha,
            "git_dirty": git_dirty,
        }
    os.makedirs(results_dir, exist_ok=True)
    with open(_stamp_path(results_dir), "w", encoding="utf-8") as f:
        json.dump(stamps, f, ensure_ascii=False, indent=2, default=str)
    print(f"已记录运行标识 {current}（{len(filenames)} 张表）: {_stamp_path(results_dir)}")
    return current


def verify_same_run(results_dir, filenames):
    """校验这些结果表全部来自同一次运行，否则报错并点名不一致的文件

    这是 C1 那条"崩掉的一步会静默发表上一次的数字"的最后一道闸门：
    上游 slurm 脚本加了 set -euo pipefail 之后，崩掉的那一步会让整个作业
    停下，但盘上仍然可能留着"前几张表是新的、后几张是上一次的"这种半新
    半旧的组合（作业撞墙钟被杀、或有人只重跑了其中一个模块）。文件存在性
    检查对此完全无能为力——旧文件同样存在。

    Returns:
        本次校验通过时公用的 run_id 字符串。
    """
    stamps = read_run_stamps(results_dir)
    missing = [name for name in filenames if name not in stamps]
    if missing:
        raise ValueError(
            "结果目录 {} 的运行标识记录（{}）里缺少这些表: {}。"
            "它们要么是在加入运行标识之前跑出来的旧文件，要么是手工拷进来的——"
            "无论哪一种，都无法证明它们与其余结果表属于同一次运行。"
            "请重新跑一遍 slurm/run_results.slurm。".format(
                results_dir, RUN_STAMP_FILE, ", ".join(sorted(missing))
            )
        )
    by_run = {}
    for name in filenames:
        by_run.setdefault(stamps[name]["run_id"], []).append(name)
    if len(by_run) > 1:
        detail = "\n".join(
            "  run_id={}: {}".format(rid, ", ".join(sorted(names)))
            for rid, names in sorted(by_run.items())
        )
        raise ValueError(
            "结果目录 {} 里的结果表来自 {} 次不同的运行，混在一起导出会让图上的"
            "估计与相邻的样本量、manifest 属于不同批次的数字：\n{}\n"
            "请重新完整跑一遍 slurm/run_results.slurm（它会用同一个 {} "
            "重写全部结果表），不要只补跑缺的那几张。".format(
                results_dir, len(by_run), detail, RUN_ID_ENV
            )
        )
    return next(iter(by_run))


def build_manifest(step, inputs=None, params=None, counts=None, fingerprints=None):
    """构造运行溯源信息"""
    git_sha, git_dirty = _git_revision()
    return {
        "step": step,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # 同一次 run_results.slurm 作业里的所有步骤共用同一个 run_id，
        # 导出层据此判断结果表有没有新旧混装（见 verify_same_run）
        "run_id": run_id(),
        "git_sha": git_sha,
        # True 表示运行时工作区有未提交改动，git_sha 不足以复现这批数字
        "git_dirty": git_dirty,
        "inputs": inputs or [],
        "params": params or {},
        "counts": counts or {},
        "fingerprints": fingerprints or {},
    }


def write_manifest(manifest, out_dir):
    """写出 manifest.json，同目录已有文件会被覆盖"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"已写出运行记录: {path}")
    return path
