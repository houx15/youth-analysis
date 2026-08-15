"""
文本清理、帖子类型判定与词表匹配（纯函数，无 IO，可在本地测试）。

与旧版 utils.utils.sentence_cleaner 的区别：
1. 转发链的正则改为 //@昵称: 的字面匹配，旧版 [//@].*?[:] 是字符类，
   会把正文中的 @提及 一直删到下一个冒号。昵称部分做了「有界」处理：
   长度上限约 40（真实平台限制 30 左右，留了余量），最多允许 3 个
   以单个空格分隔的片段（容纳"网易 看客"这类带空格的真实昵称）。
   有界是关键——如果去掉长度上限，就等于重新引入旧版 bug，会把
   //@ 之后一段很长的正常中文句子也当成昵称吞掉；
2. 词表匹配改为最左最长且命中区间不重叠，旧版逐词 count 会让嵌套词重复计字符。
"""

import re

# 纯转发的占位文本
PLAIN_RETWEET_PLACEHOLDERS = {
    "",
    "转发微博",
    "转发微博。",
    "轉發微博",
    "轉發微博。",
    "转发",
    "Repost",
    "repost",
}

# 微博客户端渲染链接卡片时，会在短链接后紧跟固定展示文案"网页链接"；
# 这里只清除紧邻在 URL 之后（中间最多空白）的这一份文案，不是全文匹配
# "网页链接"字样——如果它单独出现在正文其他位置，不受影响。
_URL_PATTERN = re.compile(
    r"https?://[a-zA-Z0-9./?&=:_%,~#\-]+(?:\s*网页链接)?", re.S
)
# 转发链：//@昵称: 之后的内容都不是本人表达。
# 微博昵称上限约 30 个字符，可能包含个别空格（如"网易 看客"），这里给了
# 到 40 的宽限（覆盖真实昵称长度统计口径的误差），并把昵称拆成「最多 3 个
# 不含空白的片段、以单个空格分隔」的形式——既能容纳真实昵称里的空格，
# 又保持有界，不会退化成旧版 [//@].*?[:] 那种一直吃到下一个冒号的字符类，
# 吞掉不带 // 前缀、和转发链无关的正常长句。
_NICK_SEGMENT = r"[^\s:：]{1,40}"
_RETWEET_CHAIN_PATTERN = re.compile(
    r"//\s*@" + _NICK_SEGMENT + r"(?: " + _NICK_SEGMENT + r"){0,2}\s*[:：].*$",
    re.S,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(content):
    """清理微博文本，只保留用户本人的可见表达

    步骤：去链接 -> 去转发链及其后的全部内容 -> 折叠空白。
    缺失值和非字符串一律返回空串。
    """
    if content is None:
        return ""
    # pandas 的缺失值判断，避免引入 pandas 依赖
    if isinstance(content, float) and content != content:
        return ""
    text = str(content)
    text = _URL_PATTERN.sub("", text)
    text = _RETWEET_CHAIN_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def classify_post_type(is_retweet, cleaned_text):
    """判定帖子类型：原创 / 转发新增评论 / 纯转发

    Args:
        is_retweet: 原始字段，字符串 "1" 或整数 1 表示转发
        cleaned_text: 已经过 clean_text 处理的文本
    """
    retweet_flag = str(is_retweet).strip() == "1"
    if not retweet_flag:
        return "original"
    if cleaned_text.strip() in PLAIN_RETWEET_PLACEHOLDERS:
        return "retweet_plain"
    return "retweet_comment"


class VocabMatcher:
    """最左最长、命中区间不重叠的词表匹配器

    实现方式：把词表按长度降序拼成一个正则交替式。Python 的 re 在同一
    起始位置按交替顺序取第一个成功的分支，因此长词在前即可保证最长匹配；
    finditer 本身保证命中区间不重叠。
    """

    def __init__(self, terms):
        cleaned = sorted(
            {t.strip() for t in terms if t and t.strip()},
            key=lambda t: (-len(t), t),
        )
        self.terms = cleaned
        if cleaned:
            pattern = "|".join(re.escape(t) for t in cleaned)
            self._regex = re.compile(pattern)
        else:
            self._regex = None

    def find(self, text):
        """返回 [(命中词, 起点, 终点)]，按出现顺序排列"""
        if not text or self._regex is None:
            return []
        return [(m.group(0), m.start(), m.end()) for m in self._regex.finditer(text)]


def measure_text(text, matcher):
    """对一条已清理文本计算命中指标

    Returns:
        n_chars: 有效字符数
        hit: 是否命中
        n_hits: 命中次数（重复出现分别计数）
        n_chars_hit: 命中区间字符数（不重叠）
        terms: 去重后的命中词列表
        density: n_chars_hit / n_chars
    """
    n_chars = len(text) if text else 0
    if n_chars == 0:
        return {
            "n_chars": 0,
            "hit": False,
            "n_hits": 0,
            "n_chars_hit": 0,
            "terms": [],
            "density": 0.0,
        }

    matches = matcher.find(text)
    n_chars_hit = sum(end - start for _, start, end in matches)
    unique_terms = sorted({term for term, _, _ in matches})
    return {
        "n_chars": n_chars,
        "hit": len(matches) > 0,
        "n_hits": len(matches),
        "n_chars_hit": n_chars_hit,
        "terms": unique_terms,
        "density": n_chars_hit / n_chars,
    }
