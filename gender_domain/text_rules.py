"""
文本清理、帖子类型判定与词表匹配（纯函数，无 IO，可在本地测试）。

与旧版 utils.utils.sentence_cleaner 的区别：
1. 转发链的正则改为要求字面的 //@ 前缀，旧版 [//@].*?[:] 是字符类，
   单个裸的 @ 就能触发，会把正文中的 @提及 一直删到下一个冒号。
   经产品侧确认（见 basic_text_extractor.py 对 weibo_content 的抓取
   方式，链式转发格式是微博平台自身的产出，不是本项目解析引入的）：
   本人评论必然在第一个 //@ 之前，//@ 之后一律是更早转发者的内容，
   不存在"//@ 之后还是本人表达"的情况。因此不再对昵称做任何长度或
   空格限制，//@ 出现后直接删到字符串结尾，不要求后面一定有冒号；
   清理顺序也很关键：必须先删转发链、再删链接。原始文本里的 URL
   字符类不含 @，如果先删链接，短链接紧跟 //@（中间没有空格）时，
   URL 匹配会把这个 // 吃掉，导致后面的 //@ 转发链完全检测不到，
   别人的转发内容就会泄漏进本人文本；
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

# 转发链：//@ 出现后，直接删到字符串结尾，不解析昵称、不要求冒号。
# 依据产品侧确认的规则："//@" 之后一律是更早转发者的内容，本人表达
# 只可能出现在第一个 //@ 之前，因此没有"删太多"的风险，可以放心不设
# 上限。要求字面的 // 前缀（而不是旧版 [//@] 字符类）是防止裸 @提及
# 触发误删的关键，必须保留。
# 必须在删链接之前运行（见下方 clean_text 的清理顺序说明）。
_RETWEET_CHAIN_PATTERN = re.compile(r"//\s*@.*$", re.S)
# 微博客户端渲染链接卡片时，会在短链接后紧跟固定展示文案"网页链接"；
# 这里只清除紧邻在 URL 之后（中间最多空白）的这一份文案，不是全文匹配
# "网页链接"字样——如果它单独出现在正文其他位置，不受影响。
# 必须在转发链正则之后运行：URL 字符类不含 @，如果先删链接，短链接
# 紧跟 //@（例如 http://t.cn/xxx//@张三:...）会把这个 // 当成 URL
# 路径的一部分吃掉，导致后面的转发链检测不到，别人的转发内容会泄漏
# 进本人文本。普通 URL（如 http://t.cn/xxx 或 http://user@host/）
# 对转发链正则总是安全的——它要求 // 后紧跟 @（可有空白），而
# "http://" 后面是协议路径字符，"http://user@host/" 里的 // 后面是
# "u"，都不会被转发链正则命中。
_URL_PATTERN = re.compile(
    r"https?://[a-zA-Z0-9./?&=:_%,~#\-]+(?:\s*网页链接)?", re.S
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_or_none(content):
    """把缺失值统一处理成 None，其余一律转成字符串

    clean_text 和 has_retweet_chain 共用这一步，保证两者对"什么算缺失"
    的判断完全一致。
    """
    if content is None:
        return None
    # pandas 的缺失值判断，避免引入 pandas 依赖
    if isinstance(content, float) and content != content:
        return None
    return str(content)


def clean_text(content):
    """清理微博文本，只保留用户本人的可见表达

    步骤：去转发链及其后的全部内容 -> 去链接 -> 折叠空白。
    转发链必须先删——见模块顶部注释里清理顺序的说明。
    缺失值和非字符串一律返回空串。
    """
    text = _normalize_or_none(content)
    if text is None:
        return ""
    text = _RETWEET_CHAIN_PATTERN.sub("", text)
    text = _URL_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def has_retweet_chain(content):
    """判断原始文本里是否含有 clean_text 会清除的转发链

    供下游写出 chain_stripped 审计列，用来统计规则命中率。
    直接在原始文本（未删链接）上用同一个转发链正则判断，和 clean_text
    先删转发链、再删链接的顺序保持一致，保证二者的结论不会互相矛盾：
    has_retweet_chain(x) 为 True 当且仅当 clean_text(x) 确实删掉过一段
    转发链。缺失值处理方式与 clean_text 共用同一个归一化函数。
    """
    text = _normalize_or_none(content)
    if text is None:
        return False
    return _RETWEET_CHAIN_PATTERN.search(text) is not None


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
