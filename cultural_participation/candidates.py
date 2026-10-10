"""可追溯的补充提词规则：缩写、数字字母组合、书名号短语及已核实完整词形。"""

import re


ASCII = re.compile(r"[A-Za-z0-9]+(?:[._+\-][A-Za-z0-9]+)*")
LETTERS = re.compile(r"[A-Za-z]{2,}")
WORK = re.compile(r"《([^《》\n]{2,40})》")
URL = re.compile(r"https?://\S+")


def candidates(title, tokenizer, min_length, prefixes, protected):
    """同词可有多个提取来源；保留原大小写与原标题，不自动合并同义词。"""
    clean = URL.sub(" ", title)
    for word, pos in tokenizer(clean):
        word = word.strip()
        if len(word) >= min_length and pos.startswith(tuple(prefixes)):
            yield word, pos, "jieba"
    for match in ASCII.finditer(clean):
        word = match.group()
        if min_length <= len(word) <= 64 and any(c.isalpha() for c in word):
            yield word, "eng", "latin_or_alphanumeric"
            # NBA7人中的NBA、iPhone12中的iPhone；完整组合也保留供审核。
            for component in LETTERS.findall(word):
                if component != word and len(component) >= min_length:
                    yield component, "eng", "latin_component"
    for match in WORK.finditer(clean):
        word = match.group(1).strip()
        if len(word) >= min_length:
            yield word, "nz", "book_title_phrase"
    for word in protected:
        if len(word) >= min_length and word in clean:
            yield word, "nz", "protected_phrase"


def review_flags(term, count, tags, longer_phrases):
    flags = []
    if count == 1:
        flags.append("single_title")
    if tags and all(tag.startswith("v") for tag in tags):
        flags.append("verb_only")
    if longer_phrases:
        flags.append("part_of_observed_phrase")
    if re.search(r"^(?:和|被|给).{2,}$|^.{1,}(?:着|给|的)$", term):
        flags.append("possible_segmentation_fragment")
    return flags
