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


def build_manifest(step, inputs=None, params=None, counts=None, fingerprints=None):
    """构造运行溯源信息"""
    git_sha, git_dirty = _git_revision()
    return {
        "step": step,
        "created_at": datetime.now().isoformat(timespec="seconds"),
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
