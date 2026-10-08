"""正式词表接口与全重叠匹配：支持精确的删词重采样，不把模型输出直接当定稿。"""

import hashlib
import json
from pathlib import Path
import re


CHAIN = re.compile(r"//\s*@.*$", re.S)
URL = re.compile(r"https?://[^\s]+(?:\s*网页链接)?")
PLACEHOLDERS = {"", "转发", "转发微博", "轉發微博", "repost"}


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", URL.sub("", CHAIN.sub("", value))).strip()


def expressive(text):
    return bool(text) and text.strip(" 。.!！").lower() not in PLACEHOLDERS


class Vocabulary:
    """JSON 顶层为 version、domains 和 terms；所有词条必须经过明确审核。"""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        self.domains = self.data["domains"]
        if not self.data.get("version") or not isinstance(self.domains, dict) or not self.domains:
            raise ValueError("正式词表需要 version 和非空 domains")
        if "none" in self.domains:
            raise ValueError("none 只用于分类审核，不属于行为分析目标领域")
        self.rules = {}
        self.trie = {}
        for row in self.data["terms"]:
            term = row["term"]
            if not isinstance(term, str) or not term.strip() or term != term.strip() or term in self.rules:
                raise ValueError("词条必须非空、无边缘空白且不重复")
            labels = row["domains"]
            if not isinstance(labels, list) or not labels or not set(labels).issubset(self.domains):
                raise ValueError(f"词条领域无效：{term}")
            if row.get("approved") is not True:
                raise ValueError(f"词条尚未审核：{term}")
            decision = row.get("decision")
            context = row.get("context_any", [])
            if not isinstance(context, list) or not all(isinstance(x, str) and x.strip() for x in context):
                raise ValueError("context_any 必须是非空字符串的列表")
            if decision not in {"standalone", "context_required"} or (decision == "context_required" and not context):
                raise ValueError(f"需要上下文的词必须配置 context_any：{term}")
            self.rules[term] = {**row, "domains": sorted(set(labels)), "context_any": context}
            node = self.trie
            for char in term:
                node = node.setdefault(char, {})
            node[None] = term
        if not self.rules:
            raise ValueError("词表为空")

    def match(self, text):
        """返回所有命中词，包括嵌套词；只用于二值参与，不作为不重叠字数密度。"""
        found = set()
        for start in range(len(text)):
            node = self.trie
            for index in range(start, len(text)):
                node = node.get(text[index])
                if node is None:
                    break
                if None in node:
                    term = node[None]
                    rule = self.rules[term]
                    if rule["decision"] == "standalone" or any(x in text for x in rule["context_any"]):
                        found.add(term)
        return sorted(found)
