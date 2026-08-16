import pytest

from gender_domain import text_rules as tr


# ---- clean_text ----

def test_clean_text_removes_urls():
    assert "网页链接" not in tr.clean_text("看这个 http://t.cn/abc123 网页链接")
    assert tr.clean_text("疫情通报 https://weibo.com/x") == "疫情通报"


def test_clean_text_removes_retweet_chain_but_keeps_own_comment():
    # 用户自己的评论在第一个 //@ 之前
    text = "说得好//@张三:同意//@李四:转发微博"
    assert tr.clean_text(text) == "说得好"


def test_clean_text_keeps_at_mention_inside_normal_sentence():
    # 旧版 sentence_cleaner 的 [//@].*?[:] 会从 @ 一直删到冒号，这里必须保留
    text = "感谢@人民日报 的报道：内容很好"
    cleaned = tr.clean_text(text)
    assert "的报道" in cleaned
    assert "内容很好" in cleaned


def test_clean_text_handles_missing_values():
    assert tr.clean_text(None) == ""
    assert tr.clean_text(float("nan")) == ""
    assert tr.clean_text("") == ""


# ---- 转发链：//@ 之后一律删到字符串结尾（Fix round 2 定案规则） ----
#
# 产品侧确认：微博本人评论必然在第一个 //@ 之前，//@ 之后一律是更早
# 转发者的内容，不存在"//@ 之后还是本人表达"的情况。因此不再对昵称
# 做任何长度或空格限制，也不要求后面一定有冒号。round 1 曾经引入过
# 「有界多段」的昵称正则来处理带空格、超长昵称，但复查发现那只是把
# 触发失败的阈值从 30 移到了 40、从 0 个空格移到了 1 个空格，问题本质
# 没解决；round 2 按产品侧定案直接改成无界匹配，下面的用例相应更新。

def test_clean_text_strips_retweet_chain_with_space_in_nickname():
    # 真实微博昵称可能带空格，例如"网易 新闻"
    text = "自己的评论//@网易 新闻:转发内容"
    assert tr.clean_text(text) == "自己的评论"


def test_clean_text_strips_retweet_chain_with_multiple_spaces_in_nickname():
    # round 1 的有界多段正则只容忍单个空格，round 2 复查发现 2 个及以上
    # 空格仍会导致整条链匹配失败；无界匹配下不应再有这个问题
    text = "自己的评论//@网易   新闻:转发内容"
    assert tr.clean_text(text) == "自己的评论"


def test_clean_text_strips_retweet_chain_with_very_long_nickname():
    # 200 字符的昵称，远超任何真实平台限制，也应该被完整清除
    nickname = "A" * 200
    text = "自己的评论//@" + nickname + ":转发内容"
    assert tr.clean_text(text) == "自己的评论"


def test_clean_text_strips_retweet_chain_without_colon():
    # //@昵称 后面不一定跟冒号（例如客户端渲染差异），只要出现 //@
    # 就应该整体删除，不要求冒号存在
    text = "说得好//@张三 转发"
    assert tr.clean_text(text) == "说得好"


def test_clean_text_strips_multi_segment_retweet_chain():
    # 多层转发链 //@a:x//@b:y 仍要能完整清除
    text = "说得好//@a:x//@b:y"
    assert tr.clean_text(text) == "说得好"


def test_clean_text_keeps_ordinary_prose_colon_after_at_mention_without_slashes():
    # 没有 // 前缀的 @提及，无论后面跟多长的正常语句和冒号，都不是转发链，
    # 必须原样保留——这是和旧版 [//@].*?[:] 行为区分开的核心场景，也是
    # 无界匹配仍然安全的前提：触发条件是字面的 //，不是裸的 @
    text = "感谢@人民日报 的报道，希望大家都能遵守防疫规定，共同努力：内容很好"
    cleaned = tr.clean_text(text)
    assert "共同努力" in cleaned
    assert "内容很好" in cleaned


def test_clean_text_strips_long_prose_after_double_slash_at():
    # round 1 曾用这个场景证明"有界"保护了 //@ 后面的长句不被吞掉；
    # round 2 按产品侧定案反转了这个断言——//@ 之后一律不是本人表达，
    # 所以现在应该被完整清除，包括"：后面的内容"
    long_prose = "这是一段很长很长完全不像昵称的正常中文描述文字" * 3
    text = "自己的评论//@" + long_prose + "：后面的内容"
    assert tr.clean_text(text) == "自己的评论"


# ---- has_retweet_chain ----

def test_has_retweet_chain_true_for_retweet_chain():
    assert tr.has_retweet_chain("说得好//@张三:同意") is True
    assert tr.has_retweet_chain("说得好//@张三 转发") is True


def test_has_retweet_chain_false_for_plain_text():
    assert tr.has_retweet_chain("今天天气很好") is False


def test_has_retweet_chain_false_for_bare_at_mention():
    # 裸的 @提及（没有 // 前缀）不算转发链
    assert tr.has_retweet_chain("感谢@人民日报 的报道：内容很好") is False


def test_has_retweet_chain_handles_missing_values():
    assert tr.has_retweet_chain(None) is False
    assert tr.has_retweet_chain(float("nan")) is False
    assert tr.has_retweet_chain("") is False


# ---- clean_text / has_retweet_chain 一致性（Fix round 3 回归用例） ----
#
# round 2 把转发链正则改成无界匹配后，has_retweet_chain 直接在原始文本
# （未删链接）上跑同一个正则；但 clean_text 当时仍然是先删链接、再删
# 转发链。URL 字符类不含 @，短链接紧跟 //@（中间无空格）时，先删链接
# 会把这个 // 当成 URL 路径的一部分吃掉，导致 clean_text 检测不到后面
# 的转发链、别人的转发内容泄漏进本人文本，但 has_retweet_chain 仍然
# 在未处理的原始文本上找到了 //@，两者结论矛盾。round 3 把 clean_text
# 的顺序改成先删转发链、再删链接，并让 has_retweet_chain 复用同一个
# 归一化 + 正则，从根源上保证两者不会分歧。

def test_clean_text_and_has_retweet_chain_agree_on_url_immediately_before_chain():
    # 复查给出的原始复现串：短链接后面紧跟 //@，中间没有空格
    content = "分享一个链接http://t.cn/RxYz1a//@张三:同意这个观点"
    assert tr.clean_text(content) == "分享一个链接"
    assert tr.has_retweet_chain(content) is True


def test_clean_text_and_has_retweet_chain_agree_on_url_without_chain():
    # 只有链接、没有转发链：链接本身仍要被完整清除，且不应被误判为转发链
    content = "看这个 http://t.cn/abc123 消息"
    cleaned = tr.clean_text(content)
    assert "t.cn" not in cleaned
    assert "abc123" not in cleaned
    assert tr.has_retweet_chain(content) is False


def test_clean_text_and_has_retweet_chain_agree_on_chain_without_url():
    # 只有转发链、没有链接：不受清理顺序调整影响，行为应保持不变
    content = "说得好//@张三:同意"
    assert tr.clean_text(content) == "说得好"
    assert tr.has_retweet_chain(content) is True


def test_clean_text_and_has_retweet_chain_agree_on_bare_at_mention_with_colon():
    # 裸的 @提及（没有 // 前缀）后面跟正常语句和冒号，两者都不应误判
    content = "感谢@人民日报 的报道：内容很好"
    cleaned = tr.clean_text(content)
    assert "内容很好" in cleaned
    assert tr.has_retweet_chain(content) is False


# ---- classify_post_type ----

def test_classify_original_post():
    assert tr.classify_post_type("0", "今天天气很好") == "original"


def test_classify_plain_retweet_placeholder():
    assert tr.classify_post_type("1", "转发微博") == "retweet_plain"
    assert tr.classify_post_type("1", "") == "retweet_plain"
    assert tr.classify_post_type("1", "轉發微博") == "retweet_plain"


def test_classify_retweet_with_comment():
    assert tr.classify_post_type("1", "说得好") == "retweet_comment"


def test_classify_accepts_integer_is_retweet():
    assert tr.classify_post_type(1, "转发微博") == "retweet_plain"
    assert tr.classify_post_type(0, "原创") == "original"


# ---- VocabMatcher ----

def test_matcher_finds_single_term():
    m = tr.VocabMatcher(["疫情"])
    assert m.find("今天疫情通报") == [("疫情", 2, 4)]


def test_matcher_prefers_longest_term_at_same_position():
    m = tr.VocabMatcher(["新冠", "新冠肺炎"])
    assert m.find("新冠肺炎防控") == [("新冠肺炎", 0, 4)]


def test_matcher_returns_non_overlapping_spans():
    m = tr.VocabMatcher(["防控", "肺炎防控"])
    matches = m.find("肺炎防控工作")
    assert matches == [("肺炎防控", 0, 4)]


def test_matcher_counts_repeated_terms_separately():
    m = tr.VocabMatcher(["疫情"])
    assert len(m.find("疫情疫情")) == 2


def test_matcher_escapes_regex_metacharacters_in_terms():
    m = tr.VocabMatcher(["A.B"])
    assert m.find("xA.By") == [("A.B", 1, 4)]
    assert m.find("xAQBy") == []


def test_matcher_with_empty_vocabulary_finds_nothing():
    m = tr.VocabMatcher([])
    assert m.find("任何文本") == []


# ---- measure_text ----

def test_measure_text_reports_density_over_cleaned_length():
    m = tr.VocabMatcher(["疫情"])
    result = tr.measure_text("疫情通报", m)
    assert result["n_chars"] == 4
    assert result["hit"] is True
    assert result["n_hits"] == 1
    assert result["n_chars_hit"] == 2
    assert result["terms"] == ["疫情"]
    assert result["density"] == pytest.approx(0.5)


def test_measure_text_never_double_counts_nested_terms():
    # 旧实现会把"新冠"和"新冠肺炎"各算一次，共 6 字，超过命中区间的 4 字
    m = tr.VocabMatcher(["新冠", "新冠肺炎"])
    result = tr.measure_text("新冠肺炎", m)
    assert result["n_chars_hit"] == 4
    assert result["density"] == pytest.approx(1.0)


def test_measure_text_on_empty_text():
    m = tr.VocabMatcher(["疫情"])
    result = tr.measure_text("", m)
    assert result["n_chars"] == 0
    assert result["hit"] is False
    assert result["density"] == 0.0


def test_measure_text_deduplicates_term_list_but_not_hit_count():
    m = tr.VocabMatcher(["疫情"])
    result = tr.measure_text("疫情疫情", m)
    assert result["n_hits"] == 2
    assert result["terms"] == ["疫情"]


def test_measure_text_term_counts_records_per_term_occurrences():
    # term_counts 是 build_post_table.py 编码 {domain}_term_counts 列的
    # 唯一数据来源，重复出现的词必须记为对应的次数，而不是像 terms 那样去重。
    m = tr.VocabMatcher(["疫情", "防控"])
    result = tr.measure_text("疫情疫情防控", m)
    assert result["term_counts"] == {"疫情": 2, "防控": 1}


def test_measure_text_term_counts_empty_on_empty_text():
    m = tr.VocabMatcher(["疫情"])
    result = tr.measure_text("", m)
    assert result["term_counts"] == {}
