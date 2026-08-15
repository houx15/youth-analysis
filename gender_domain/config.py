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


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def build_manifest(step, inputs=None, params=None, counts=None, fingerprints=None):
    """构造运行溯源信息"""
    return {
        "step": step,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
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
