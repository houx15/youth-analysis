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
