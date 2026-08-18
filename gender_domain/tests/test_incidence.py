"""
gender_domain.robustness.incidence 的单元测试。

这份测试的重心只有一条：**全词表重建必须与主流水线（表 C）逐个用户完全
相等**。§13.3 的 200-500 次词表重采样全部建立在"按存量逐词计数重新聚合
等价于重扫全年正文"这个前提上；如果全词表下的重建结果与表 C 有任何偏差，
那么每一次重采样都会朝同一个方向错下去，整套稳健性结论都不成立，而且
下游没有任何一步能把这个偏差再找回来。所以这里刻意不手写期望值：

    夹具先用真实的 build_post_table.process_frame 生成表 A，再用真实的
    build_user_tables.aggregate_posts / aggregate_events / combine_user_table
    从这份表 A、表 B 生成表 C。

也就是说，测试里的表 C 确确实实是"由这份表 A 跑出来的"，而不是人手编出来
的一组数字——手写期望值只能证明"重建结果等于我以为的主流水线"，证明不了
"重建结果等于主流水线"。

夹具刻意覆盖了几处最容易出错的地方：

1. **分母陷阱**：u3 有两条表达帖，只有一条命中。分母必须来自全部帖子，
   如果误用"进了矩阵的命中帖"当分母，u3 的 topical_share 会从 0.5 变成
   1.0（每个用户的份额都会被抬高）。
2. **命中但不是表达帖**：公共事务词表里放了"转发"，于是纯转发帖
   "转发微博" 会命中词表却不是表达帖。它既不能进分子也不能进分母。
3. **零表达帖用户**：u4 全年只有一条纯转发，分母为 0，份额必须是 NaN
   而不是 0（_safe_divide 的口径）。
4. **嵌套遮蔽**："疫情" 是 "疫情防控" 的子串。剔除长词后重新聚合会低估
   短词——这是存量重聚合无法避免的已知局限，测试把它钉死成一个显式的
   已知行为，而不是让它在某次重采样里悄悄出现。
5. **跨分片行号偏移**：帖子分成 1 月、2 月两个分片，矩阵行号必须跨分片
   连续，不能每个分片各自从 0 开始。
"""

import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import build_post_table as bpt
from gender_domain import build_retweet_table as brt
from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import text_rules as tr
from gender_domain.robustness import incidence as inc


YEAR = 2020

# 合成公共事务词表。"疫情" 是 "疫情防控" 的子串（嵌套遮蔽）；"转发" 会让
# 纯转发占位文本 "转发微博" 命中词表却仍然不是表达帖；"封城" 全年一次都
# 没出现过，用来覆盖"词在词表里、但一列都没进矩阵"这条路径。
PUBLIC_VOCAB = ["疫情", "疫情防控", "复工", "转发", "封城"]
CELEBRITY_VOCAB = ["顶流", "粉丝"]

# 合成来源账号名单。a4 同时出现在两个领域，用来覆盖"同一个账号在两个
# 领域各占一列"这件事——按账号剔除时两列必须一起消失。
PUBLIC_ACCOUNTS = {"central_media": {"a1", "a2"}, "local_media": {"a4"}}
CELEBRITY_ACCOUNTS = {"star": {"a3"}, "fanclub": {"a4"}}


def _raw_posts():
    """合成的原始帖子（cleaned_weibo_cov 口径），交给真实的 process_frame 处理"""
    rows = [
        # weibo_id, user_id, content, is_retweet, date
        ("p1", "u1", "今天疫情防控形势", "0", "2020-01-05"),
        ("p2", "u1", "复工了粉丝好多", "0", "2020-01-06"),
        ("p3", "u1", "转发微博", "1", "2020-01-07"),
        ("p6", "u3", "普通日常内容", "0", "2020-01-08"),
        ("p8", "u4", "转发微博", "1", "2020-01-09"),
        ("p4", "u2", "疫情 复工 疫情", "0", "2020-02-01"),
        ("p5", "u2", "顶流粉丝合体", "1", "2020-02-02"),
        ("p7", "u3", "疫情之下的生活", "0", "2020-02-03"),
        ("p9", "u5", "http://t.cn/abcdef", "0", "2020-02-04"),
        ("p10", "u5", "疫情防控 疫情防控", "0", "2020-02-05"),
    ]
    return pd.DataFrame(
        {
            "weibo_id": [r[0] for r in rows],
            "user_id": [r[1] for r in rows],
            "weibo_content": [r[2] for r in rows],
            "is_retweet": [r[3] for r in rows],
            "gender": ["m" if r[1] in ("u1", "u3", "u5") else "f" for r in rows],
            "province": ["11" for _ in rows],
            "time_stamp": [1580000000 for _ in rows],
            "date": [r[4] for r in rows],
        }
    )


def _raw_retweets():
    """合成的原始转发记录，交给真实的 build_retweet_table.process_frame 处理"""
    rows = [
        # user_id, weibo_id, r_weibo_id, r_user_id, date
        ("u1", "rw1", "s1", "a1", "2020-01-05"),
        ("u1", "rw2", "s2", "a1", "2020-01-06"),
        ("u1", "rw3", "s3", "a2", "2020-01-07"),
        ("u2", "rw4", "s4", "a3", "2020-01-08"),
        ("u3", "rw5", "s5", "a4", "2020-01-09"),
        ("u5", "rw6", "s6", "zzz", "2020-01-10"),
    ]
    return pd.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "weibo_id": [r[1] for r in rows],
            "r_weibo_id": [r[2] for r in rows],
            "r_user_id": [r[3] for r in rows],
            "is_retweet": ["1" for _ in rows],
            "gender": ["m" if r[0] in ("u1", "u3", "u5") else "f" for r in rows],
            "time_stamp": [1580000100 for _ in rows],
            "r_time_stamp": [1580000000 for _ in rows],
            "date": [r[4] for r in rows],
        }
    )


def _write_shards(out_dir, public_vocab=None, celebrity_vocab=None):
    """用真实流水线写出表 A、表 B 的分月分片，返回 (表 A 全量, 表 B 全量)"""
    posts = bpt.process_frame(
        _raw_posts(),
        tr.VocabMatcher(PUBLIC_VOCAB if public_vocab is None else public_vocab),
        tr.VocabMatcher(CELEBRITY_VOCAB if celebrity_vocab is None else celebrity_vocab),
    )
    post_dir = os.path.join(out_dir, f"post_domain_measures_{YEAR}")
    os.makedirs(post_dir, exist_ok=True)
    for month, part in posts.groupby("month"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(post_dir, f"month={month:02d}.parquet"),
            engine="pyarrow", index=False,
        )

    events = brt.process_frame(
        _raw_retweets(),
        brt.build_account_lookup(PUBLIC_ACCOUNTS),
        brt.build_account_lookup(CELEBRITY_ACCOUNTS),
    )
    event_dir = os.path.join(out_dir, f"retweet_domain_events_{YEAR}")
    os.makedirs(event_dir, exist_ok=True)
    for month, part in events.groupby("month"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(event_dir, f"month={month:02d}.parquet"),
            engine="pyarrow", index=False,
        )
    return posts, events


@pytest.fixture()
def synthetic_tables(tmp_path, monkeypatch):
    """写出互相自洽的表 A / 表 B / 表 C，并把 config.OUTPUT_DIR 指过去

    表 C 由真实的 aggregate_posts / aggregate_events / combine_user_table
    从刚写出的表 A、表 B 生成——它确实是"这份表 A 跑出来的表 C"。
    """
    out_dir = str(tmp_path / "analysis_data")
    os.makedirs(out_dir, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    posts, events = _write_shards(out_dir)

    post_agg = but.aggregate_posts(posts[but.POST_SHARD_COLUMNS])
    event_agg = but.aggregate_events(events[but.EVENT_SHARD_COLUMNS])
    panel = but.aggregate_user_month(posts[but.POST_SHARD_COLUMNS],
                                     events[but.EVENT_SHARD_COLUMNS])
    table_c = but.combine_user_table(post_agg, event_agg, panel)
    return {"out_dir": out_dir, "posts": posts, "events": events, "table_c": table_c}


# ---------------------------------------------------------------------------
# 最重要的一条：全词表重建 == 表 C
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain,vocab", [
    ("public", PUBLIC_VOCAB),
    ("celebrity", CELEBRITY_VOCAB),
])
def test_full_vocabulary_reconstruction_matches_table_c(synthetic_tables, domain, vocab):
    """全词表重新聚合必须与表 C 的 {domain}_topical_share 逐用户完全相等

    这是整个稳健性方案里最重要的一个断言，理由见模块文档。
    """
    table_c = synthetic_tables["table_c"]
    incidence = inc.build_post_term_incidence(YEAR, domain)
    result = inc.topical_by_user(incidence, vocab, incidence.posts)

    # 用户集合必须与表 C 完全一致（不能只剩有命中的用户）
    assert sorted(result["user_id"]) == sorted(table_c["user_id"])

    merged = table_c[["user_id", f"{domain}_topical_share", f"{domain}_topical_posts",
                      "n_expressive_posts"]].merge(
        result, on="user_id", how="left", suffixes=("_c", "_r")
    )
    pd.testing.assert_series_equal(
        merged["topical_posts"].astype("int64"),
        merged[f"{domain}_topical_posts"].astype("int64"),
        check_names=False,
    )
    # 分母也必须逐用户相等：份额相等而分母不等，说明分子分母同时错了
    pd.testing.assert_series_equal(
        merged["n_expressive_posts_r"].astype("int64"),
        merged["n_expressive_posts_c"].astype("int64"),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["topical_share"].astype("float64"),
        merged[f"{domain}_topical_share"].astype("float64"),
        check_names=False,
    )
    # 夹具本身必须真的包含一个零表达帖用户（NaN），否则上面的相等断言
    # 会在一个没有 NaN 的表上"轻松通过"，白白放过分母口径的错误
    assert merged[f"{domain}_topical_share"].isna().sum() == 1


def test_subset_reconstruction_matches_a_true_rescan_when_no_shadowing_is_involved(
    synthetic_tables,
):
    """剔除一个不参与嵌套的词后，存量重聚合必须与真正重扫原文完全相等

    这一条比全词表重建更强，也更贴近 §13.3 真正在做的事：全词表下矩阵的
    每一行都必然命中（进矩阵的前提就是至少命中一个词），所以全词表重建
    对"矩阵行与帖子错位"完全不敏感——错位了照样全命中。这里把词表减掉
    "复工"（它既不是任何词的子串、也不包含任何词，因此不存在遮蔽问题），
    再用真实的 process_frame + aggregate_posts 重扫一遍原文当作真值，
    逐用户比对。行错位、列错位、计数错位都会在这里当场暴露。
    """
    reduced_public = [t for t in PUBLIC_VOCAB if t != "复工"]
    assert "复工" not in inc.nested_terms(PUBLIC_VOCAB)
    rescanned = bpt.process_frame(
        _raw_posts(), tr.VocabMatcher(reduced_public), tr.VocabMatcher(CELEBRITY_VOCAB)
    )
    truth = but.aggregate_posts(rescanned[but.POST_SHARD_COLUMNS])

    incidence = inc.build_post_term_incidence(YEAR, "public")
    result = inc.topical_by_user(incidence, reduced_public, incidence.posts)

    merged = truth[["user_id", "public_topical_posts", "public_topical_share"]].merge(
        result, on="user_id", how="left"
    )
    pd.testing.assert_series_equal(
        merged["topical_posts"].astype("int64"),
        merged["public_topical_posts"].astype("int64"),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["topical_share"].astype("float64"),
        merged["public_topical_share"].astype("float64"),
        check_names=False,
    )
    # 这个子集确实是一次真实的干预：重扫真值与全词表的表 C 至少有一个
    # 用户不同（否则上面两条相等断言在"什么都没变"的表上也会通过，
    # 毫无信息量）
    table_c = synthetic_tables["table_c"]
    versus_full = table_c[["user_id", "public_topical_posts"]].merge(
        truth[["user_id", "public_topical_posts"]], on="user_id", suffixes=("_full", "_reduced")
    )
    assert (
        versus_full["public_topical_posts_full"]
        != versus_full["public_topical_posts_reduced"]
    ).any()


def test_denominator_comes_from_all_posts_not_only_hitting_posts(synthetic_tables):
    """分母是该用户的全部表达帖，不是"进了矩阵的命中帖"

    u3 有两条表达帖（一条命中、一条没命中），正确份额是 0.5；如果分母误用
    命中帖，会算成 1.0。这是本任务最容易出错的一处，单独钉一条断言。
    """
    incidence = inc.build_post_term_incidence(YEAR, "public")
    result = inc.topical_by_user(incidence, PUBLIC_VOCAB, incidence.posts).set_index("user_id")
    assert result.loc["u3", "n_expressive_posts"] == 2
    assert result.loc["u3", "topical_posts"] == 1
    assert result.loc["u3", "topical_share"] == pytest.approx(0.5)


def test_hitting_but_non_expressive_posts_enter_neither_numerator_nor_denominator(
    synthetic_tables,
):
    """命中词表但不是表达帖的纯转发（"转发微博" 命中 "转发"）两头都不进

    u4 全年只有这一条帖：表达帖数为 0、命中帖数为 0、份额必须是 NaN。
    """
    incidence = inc.build_post_term_incidence(YEAR, "public")
    result = inc.topical_by_user(incidence, PUBLIC_VOCAB, incidence.posts).set_index("user_id")
    assert result.loc["u4", "n_expressive_posts"] == 0
    assert result.loc["u4", "topical_posts"] == 0
    assert np.isnan(result.loc["u4", "topical_share"])


# ---------------------------------------------------------------------------
# 词表子集的行为
# ---------------------------------------------------------------------------

def test_subset_drops_exactly_the_posts_whose_only_term_was_removed(synthetic_tables):
    """剔除一个词，只有"仅靠这个词命中"的帖子会掉出分子

    剔除 "疫情防控"：u1 的 p1、u5 的 p10 只靠这个词命中，掉出分子；
    u2 的 p4（疫情 + 复工）、u3 的 p7（疫情）不受影响。
    """
    incidence = inc.build_post_term_incidence(YEAR, "public")
    subset = [t for t in PUBLIC_VOCAB if t != "疫情防控"]
    result = inc.topical_by_user(incidence, subset, incidence.posts).set_index("user_id")
    assert result.loc["u1", "topical_posts"] == 1   # 只剩 p2（复工）
    assert result.loc["u5", "topical_posts"] == 0
    assert result.loc["u5", "topical_share"] == pytest.approx(0.0)
    assert result.loc["u2", "topical_posts"] == 1
    assert result.loc["u3", "topical_posts"] == 1


def test_post_with_two_terms_still_counts_once_when_one_is_dropped(synthetic_tables):
    """一条帖命中两个词，剔除其中一个后仍然只算一条命中帖（不是两条、也不是零条）"""
    incidence = inc.build_post_term_incidence(YEAR, "public")
    full = inc.topical_by_user(incidence, PUBLIC_VOCAB, incidence.posts).set_index("user_id")
    dropped = inc.topical_by_user(
        incidence, [t for t in PUBLIC_VOCAB if t != "复工"], incidence.posts
    ).set_index("user_id")
    # p4 同时命中 "疫情"(2 次) 与 "复工"(1 次)，剔除 "复工" 后仍靠 "疫情" 命中
    assert full.loc["u2", "topical_posts"] == 1
    assert dropped.loc["u2", "topical_posts"] == 1


def test_empty_subset_yields_zero_topical_posts_and_nan_only_for_zero_denominator(
    synthetic_tables,
):
    """空词表：命中帖数全为 0；份额为 0.0，只有分母为 0 的用户才是 NaN

    "0 个词命中 10 条表达帖中的 0 条"是 0.0，不是 NaN——NaN 的唯一含义
    是"分母为 0、比例无定义"，与 build_user_tables._safe_divide 完全一致。
    """
    incidence = inc.build_post_term_incidence(YEAR, "public")
    result = inc.topical_by_user(incidence, [], incidence.posts).set_index("user_id")
    assert (result["topical_posts"] == 0).all()
    assert result.loc["u1", "topical_share"] == pytest.approx(0.0)
    assert np.isnan(result.loc["u4", "topical_share"])  # 唯一零表达帖用户


def test_term_subset_may_contain_terms_that_never_occurred(synthetic_tables):
    """子集里含有全年一次都没出现过的词（"封城"）不报错，也不改变任何结果"""
    incidence = inc.build_post_term_incidence(YEAR, "public")
    assert "封城" not in incidence.term_index
    with_unused = inc.topical_by_user(incidence, PUBLIC_VOCAB, incidence.posts)
    without_unused = inc.topical_by_user(
        incidence, [t for t in PUBLIC_VOCAB if t != "封城"], incidence.posts
    )
    pd.testing.assert_frame_equal(with_unused, without_unused)


def test_unrecognized_subset_terms_are_reported_instead_of_silently_ignored(
    synthetic_tables, capsys
):
    """认不出来的词必须能被调用方点名核对，不能只是悄悄少算

    "全年没出现过"和"词拼错了/没归一化"在结果上长得一模一样，而后者会让
    每个 replicate 朝同一个方向少算命中——正是本模块要防的那类失败。
    """
    incidence = inc.build_post_term_incidence(YEAR, "public")
    # "封城" 是合法的从未出现词；"疫情防控措施" 是词表里根本没有的词
    assert inc.unrecognized_terms(incidence, ["疫情", "封城", "疫情防控措施"]) == [
        "封城", "疫情防控措施",
    ]
    assert inc.unrecognized_terms(incidence, ["疫情", "复工"]) == []

    inc.term_subset_vector(incidence, ["疫情", "封城"])
    assert "不在矩阵词轴上" in capsys.readouterr().out


def test_term_subset_is_normalized_before_lookup(synthetic_tables):
    """带首尾空白的词必须被归一化后命中，而不是被当成"从未出现过"丢掉"""
    incidence = inc.build_post_term_incidence(YEAR, "public")
    assert inc.unrecognized_terms(incidence, ["  疫情防控  "]) == []
    padded = inc.topical_by_user(incidence, ["  疫情防控  "], incidence.posts)
    clean = inc.topical_by_user(incidence, ["疫情防控"], incidence.posts)
    pd.testing.assert_frame_equal(padded, clean)
    assert padded["topical_posts"].sum() == 2  # p1, p10


@pytest.mark.parametrize("domain,vocab", [
    ("public", PUBLIC_VOCAB),
    ("celebrity", CELEBRITY_VOCAB),
])
def test_char_measures_reproduce_table_c_on_the_full_vocabulary(
    synthetic_tables, domain, vocab
):
    """字符口径（char_density / hits_per_1k）的全词表重建也必须等于表 C"""
    table_c = synthetic_tables["table_c"]
    incidence = inc.build_post_term_incidence(YEAR, domain)
    result = inc.char_measures_by_user(incidence, vocab, incidence.posts)

    merged = table_c[["user_id", f"{domain}_char_density", f"{domain}_hits_per_1k"]].merge(
        result, on="user_id", how="left"
    )
    pd.testing.assert_series_equal(
        merged["char_density"].astype("float64"),
        merged[f"{domain}_char_density"].astype("float64"),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["hits_per_1k"].astype("float64"),
        merged[f"{domain}_hits_per_1k"].astype("float64"),
        check_names=False,
    )


def test_char_measures_under_a_subset_match_a_true_rescan(synthetic_tables):
    """字符口径在词表子集下也与真正重扫原文相等（不涉及嵌套的子集）"""
    reduced_public = [t for t in PUBLIC_VOCAB if t != "复工"]
    rescanned = bpt.process_frame(
        _raw_posts(), tr.VocabMatcher(reduced_public), tr.VocabMatcher(CELEBRITY_VOCAB)
    )
    truth = but.aggregate_posts(rescanned[but.POST_SHARD_COLUMNS])

    incidence = inc.build_post_term_incidence(YEAR, "public")
    result = inc.char_measures_by_user(incidence, reduced_public, incidence.posts)
    merged = truth[["user_id", "public_char_density", "public_hits_per_1k"]].merge(
        result, on="user_id", how="left"
    )
    pd.testing.assert_series_equal(
        merged["char_density"].astype("float64"),
        merged["public_char_density"].astype("float64"),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["hits_per_1k"].astype("float64"),
        merged["public_hits_per_1k"].astype("float64"),
        check_names=False,
    )


def test_posts_frame_uses_compact_dtypes(synthetic_tables):
    """posts 帧要在整个作业生命周期里驻留，逐列 dtype 是内存的主要杠杆"""
    posts = inc.build_post_term_incidence(YEAR, "public").posts
    # pandas 2.x 的 str(dtype) 只打印 "string"，storage 才区分 python/pyarrow
    assert isinstance(posts["weibo_id"].dtype, pd.StringDtype)
    assert posts["weibo_id"].dtype.storage == "pyarrow"
    assert str(posts["user_id"].dtype) == "category"
    assert str(posts["post_type"].dtype) == "category"
    assert posts["month"].dtype == np.int8
    assert posts["n_chars"].dtype == np.int32
    assert posts["matrix_row"].dtype == np.int32
    assert posts["is_expressive"].dtype == bool


def test_matrix_is_restricted_to_hitting_posts_and_rows_are_continuous_across_shards(
    synthetic_tables,
):
    """矩阵只包含至少命中一个词的帖子；行号跨分片连续，且与 posts 帧对齐"""
    posts_table_a = synthetic_tables["posts"]
    incidence = inc.build_post_term_incidence(YEAR, "public")
    n_hit = int(posts_table_a["public_hit"].sum())
    assert incidence.matrix.shape[0] == n_hit
    assert len(incidence.posts) == len(posts_table_a)  # posts 帧仍是全部帖子

    rows = incidence.posts["matrix_row"].to_numpy()
    assigned = np.sort(rows[rows >= 0])
    assert assigned.tolist() == list(range(n_hit))

    # 逐帖核对：矩阵这一行的逐词计数 == 表 A 该帖 term_counts 的解码结果
    by_id = posts_table_a.set_index("weibo_id")["public_term_counts"].to_dict()
    reverse = {col: term for term, col in incidence.term_index.items()}
    dense = incidence.matrix.toarray()
    for weibo_id, row in zip(incidence.posts["weibo_id"], rows):
        expected = bpt.decode_term_counts(by_id[weibo_id])
        if row < 0:
            assert expected == {}
            continue
        got = {reverse[j]: int(v) for j, v in enumerate(dense[row]) if v}
        assert got == expected


def test_topical_by_user_accepts_a_filtered_posts_frame(synthetic_tables):
    """posts 帧可以先被过滤再传进来（§13.7/§13.8 复用同一个矩阵的方式）

    只保留原创帖后，u2 的表达帖只剩 p4（p5 是带评论转发），分母从 2 变 1。
    """
    incidence = inc.build_post_term_incidence(YEAR, "celebrity")
    originals = incidence.posts[incidence.posts["post_type"] == "original"]
    result = inc.topical_by_user(incidence, CELEBRITY_VOCAB, originals).set_index("user_id")
    assert result.loc["u2", "n_expressive_posts"] == 1
    assert result.loc["u2", "topical_posts"] == 0  # p5 被过滤掉了
    assert result.loc["u2", "topical_share"] == pytest.approx(0.0)


def test_a_domain_with_no_hits_at_all_still_builds_and_aggregates(tmp_path, monkeypatch):
    """全年一次都没有命中的领域，矩阵是 0 行 0 列，聚合仍然给出全 0 / NaN

    真实数据里不会出现，但 §13.3 的"留一类别"变体完全可能把某个领域的词
    全部剔光；这时候必须给出一张规规矩矩的全 0 表，而不是在 scipy 的空
    矩阵上抛异常。
    """
    out_dir = str(tmp_path / "analysis_data")
    os.makedirs(out_dir, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    _write_shards(out_dir, celebrity_vocab=["外星人到访"])

    incidence = inc.build_post_term_incidence(YEAR, "celebrity")
    assert incidence.matrix.shape == (0, 0)
    result = inc.topical_by_user(incidence, ["外星人到访"], incidence.posts).set_index("user_id")
    assert (result["topical_posts"] == 0).all()
    assert result.loc["u1", "topical_share"] == pytest.approx(0.0)
    assert np.isnan(result.loc["u4", "topical_share"])


# ---------------------------------------------------------------------------
# 嵌套遮蔽：存量重聚合的已知局限
# ---------------------------------------------------------------------------

def test_nested_terms_flags_terms_that_are_substrings_of_another_term():
    assert inc.nested_terms(PUBLIC_VOCAB) == {"疫情"}
    assert inc.nested_terms(CELEBRITY_VOCAB) == set()
    assert inc.nested_terms(["  疫情  ", "疫情防控", ""]) == {"疫情"}


def test_nested_terms_on_the_real_vocabularies_matches_the_documented_exposure():
    """真实词表上的暴露面：公共事务 112/816，明星 5/535（模块文档里写死的数字）"""
    public = config.load_public_vocabulary(YEAR)
    celebrity = config.load_celebrity_vocabulary(YEAR)
    assert len(inc.nested_terms(public)) == 112
    assert len(inc.nested_terms(celebrity)) == 5


def test_shadowed_terms_reports_the_retained_terms_a_replicate_will_undercount(
    synthetic_tables,
):
    """shadowed_terms 给出"被本次剔除的词遮蔽住的保留词"，无遮蔽时为空集"""
    # 剔除 "疫情防控" 会遮蔽保留下来的 "疫情"
    assert inc.shadowed_terms(PUBLIC_VOCAB, ["疫情", "复工", "转发", "封城"]) == {"疫情"}
    # 剔除 "疫情"（短词）不会遮蔽任何保留词——方向是单向的
    assert inc.shadowed_terms(PUBLIC_VOCAB, ["疫情防控", "复工", "转发", "封城"]) == set()
    # 保留全部词 => 没有被剔除的词 => 没有遮蔽
    assert inc.shadowed_terms(PUBLIC_VOCAB, PUBLIC_VOCAB) == set()
    assert inc.shadowing_dropped_terms(PUBLIC_VOCAB, ["疫情", "复工"]) == {"疫情防控"}


def test_shadowing_exposure_counts_the_posts_a_replicate_may_undercount(synthetic_tables):
    """shadowing_exposure 给出这次重采样受影响的帖子数与可能丢失的表达帖数

    剔除 "疫情防控"、保留 "疫情" 后：p1（u1）与 p10（u5）的存量计数里只有
    "疫情防控"，在本子集下判定为不命中，但重扫原文会命中 "疫情"——这两条
    就是本次 replicate 可能丢失的表达帖，也是低估的上界。
    """
    incidence = inc.build_post_term_incidence(YEAR, "public")
    subset = ["疫情", "复工", "转发", "封城"]
    exposure = inc.shadowing_exposure(incidence, subset, PUBLIC_VOCAB)
    assert exposure["shadowed_terms"] == {"疫情"}
    assert exposure["n_shadowed_terms"] == 1
    assert exposure["n_shadowing_dropped_terms"] == 1
    assert exposure["n_posts_with_shadowing_term"] == 2      # p1, p10
    assert exposure["n_expressive_posts_possibly_lost"] == 2

    # 与 topical_by_user 的实际损失对得上：全词表 vs 本子集，命中表达帖
    # 恰好少了这 2 条
    full = inc.topical_by_user(incidence, PUBLIC_VOCAB, incidence.posts)
    reduced = inc.topical_by_user(incidence, subset, incidence.posts)
    lost = int(full["topical_posts"].sum() - reduced["topical_posts"].sum())
    assert lost == exposure["n_expressive_posts_possibly_lost"]

    # 没有遮蔽的子集：暴露面必须是干净的 0，不能永远报一个非零数
    clean = inc.shadowing_exposure(
        incidence, [t for t in PUBLIC_VOCAB if t != "复工"], PUBLIC_VOCAB
    )
    assert clean["n_shadowed_terms"] == 0
    assert clean["n_posts_with_shadowing_term"] == 0
    assert clean["n_expressive_posts_possibly_lost"] == 0


def test_at_risk_terms_catch_boundary_overlap_that_substring_nesting_never_sees():
    """边界重叠：被剔除词与保留词互不包含，重聚合照样漏判

    正文 "疫情防控措施"、词表 {疫情防, 防控措施, 两会}：最左最长在位置 0
    取 "疫情防"，存量计数只有 {疫情防: 1}。剔除 "疫情防"、保留 "防控措施"
    后，真正重扫会命中，重聚合会判不命中——而 "防控措施" 既不是 "疫情防"
    的子串、也不包含它，所以 shadowed_terms / nested_terms 一个都点不出来。
    这正是只看子串的口径漏掉的那一类。
    """
    vocab = ["疫情防", "防控措施", "两会"]
    subset = ["防控措施", "两会"]

    # 先用生产匹配器确认这个例子是真的（不是构造出来的假象）
    stored = tr.measure_text("疫情防控措施", tr.VocabMatcher(vocab))["term_counts"]
    assert stored == {"疫情防": 1}
    assert tr.measure_text("疫情防控措施", tr.VocabMatcher(subset))["hit"] is True

    # 子串口径：完全看不见
    assert inc.nested_terms(vocab) == set()
    assert inc.shadowed_terms(vocab, subset) == set()
    assert inc.shadowing_dropped_terms(vocab, subset) == set()

    # 完整口径：点得出来
    assert inc.at_risk_terms(vocab, subset) == {"防控措施"}
    assert inc.at_risk_dropped_terms(vocab, subset) == {"疫情防"}
    # 反方向的重叠（被剔除词的前缀接保留词的后缀）同样算
    assert inc.boundary_overlap("疫情防", "防控措施") is True
    assert inc.boundary_overlap("防控措施", "疫情防") is False


def test_at_risk_definition_is_a_superset_of_substring_shadowing(synthetic_tables):
    """完整口径必须包含子串口径的全部结论，并给出不小于它的帖子数"""
    incidence = inc.build_post_term_incidence(YEAR, "public")
    subset = ["疫情", "复工", "转发", "封城"]
    narrow = inc.shadowing_exposure(incidence, subset, PUBLIC_VOCAB)
    full = inc.reaggregation_exposure(incidence, subset, PUBLIC_VOCAB)
    assert narrow["shadowed_terms"] <= full["shadowed_terms"]
    assert (full["n_expressive_posts_possibly_lost"]
            >= narrow["n_expressive_posts_possibly_lost"])


def test_at_risk_pairs_on_the_real_vocabularies_dwarf_the_nested_terms():
    """真实词表上，边界重叠比子串嵌套多一个数量级——这就是只看子串的代价"""
    public = config.load_public_vocabulary(YEAR)
    celebrity = config.load_celebrity_vocabulary(YEAR)

    def _overlap_pairs(vocab):
        terms = inc.normalize_vocabulary(vocab)
        return sum(
            1 for a in terms for b in terms
            if a != b and b not in a and a not in b and inc.boundary_overlap(a, b)
        )

    assert _overlap_pairs(public) == 1502
    assert _overlap_pairs(celebrity) == 93
    assert len(inc.nested_terms(public)) == 112
    assert len(inc.nested_terms(celebrity)) == 5


def test_at_risk_pairs_can_be_precomputed_once_and_reused():
    """pairs 预算一次传进去，结果必须与每次现算完全一致"""
    vocab = PUBLIC_VOCAB + ["情防控", "防控措施"]
    subset = ["疫情", "防控措施", "复工"]
    pairs = inc.at_risk_pairs(vocab)
    assert inc.at_risk_terms(vocab, subset, pairs=pairs) == inc.at_risk_terms(vocab, subset)
    assert (inc.at_risk_dropped_terms(vocab, subset, pairs=pairs)
            == inc.at_risk_dropped_terms(vocab, subset))


def test_dropping_a_shadowing_term_undercounts_the_retained_shorter_term(synthetic_tables):
    """已知局限：剔除长词后，被它遮蔽的短词在存量计数里找不回来

    p1 原文含 "疫情防控"，最左最长匹配只记了 "疫情防控:1"，没有 "疫情"。
    剔除 "疫情防控"、保留 "疫情" 后重新聚合会判定 p1 不命中，而真正重扫
    原文会命中 "疫情"。这条测试把这个方向（低估，不是高估）钉死，并保证
    nested_terms 能提前把这类词点出来。
    """
    incidence = inc.build_post_term_incidence(YEAR, "public")
    subset = ["疫情", "复工", "转发"]
    result = inc.topical_by_user(incidence, subset, incidence.posts).set_index("user_id")
    assert result.loc["u5", "topical_posts"] == 0  # 重扫原文会是 1
    assert "疫情" in inc.nested_terms(PUBLIC_VOCAB)


# ---------------------------------------------------------------------------
# 账号侧
# ---------------------------------------------------------------------------

def test_account_incidence_reproduces_table_c_source_counts_and_entry(synthetic_tables):
    """全账号重建必须与表 C 的 {domain}_source_count / _source_entered 完全相等"""
    table_c = synthetic_tables["table_c"]
    incidence = inc.build_user_account_incidence(YEAR)
    result = inc.source_by_user(incidence, None)

    merged = table_c[[
        "user_id", "public_source_count", "public_source_entered",
        "celebrity_source_count", "celebrity_source_entered",
    ]].merge(result, on="user_id", how="left", suffixes=("_c", "_r"))
    for domain in ("public", "celebrity"):
        # 表 C 用左连接把没有任何来源转发的用户填成 0/False，这里同口径补齐
        counts = merged[f"{domain}_source_count_r"].fillna(0).astype(int)
        entered = merged[f"{domain}_source_entered_r"].eq(True)
        pd.testing.assert_series_equal(
            counts, merged[f"{domain}_source_count_c"].astype(int), check_names=False
        )
        pd.testing.assert_series_equal(
            entered, merged[f"{domain}_source_entered_c"].astype(bool), check_names=False
        )
    # 夹具必须真的包含"完全没有来源转发"的用户，否则填 0 那一步没被检验到
    assert (merged["public_source_count_c"] + merged["celebrity_source_count_c"] == 0).any()


def test_dropping_an_account_flips_entry_for_users_whose_only_source_it_was(
    synthetic_tables,
):
    """剔除账号后，只靠这个账号进入该领域的用户必须变成未进入"""
    incidence = inc.build_user_account_incidence(YEAR)
    kept = [a for a in incidence.accounts["r_user_id"].unique() if a != "a3"]
    result = inc.source_by_user(incidence, kept).set_index("user_id")
    # u2 只转发过 a3
    assert result.loc["u2", "celebrity_source_count"] == 0
    assert bool(result.loc["u2", "celebrity_source_entered"]) is False
    # u1 转发过 a1×2 + a2，剔除 a1 后仍然进入公共事务领域
    kept_no_a1 = [a for a in incidence.accounts["r_user_id"].unique() if a != "a1"]
    result2 = inc.source_by_user(incidence, kept_no_a1).set_index("user_id")
    assert result2.loc["u1", "public_source_count"] == 1
    assert bool(result2.loc["u1", "public_source_entered"]) is True


def test_an_account_in_both_domains_occupies_one_column_per_domain(synthetic_tables):
    """同一个账号出现在两个领域名单里时，两个领域各占一列，按账号剔除时一起消失"""
    incidence = inc.build_user_account_incidence(YEAR)
    a4_cols = incidence.accounts[incidence.accounts["r_user_id"] == "a4"]
    assert set(a4_cols["source_domain"]) == {"public", "celebrity"}

    full = inc.source_by_user(incidence, None).set_index("user_id")
    assert full.loc["u3", "public_source_count"] == 1
    assert full.loc["u3", "celebrity_source_count"] == 1

    kept = [a for a in incidence.accounts["r_user_id"].unique() if a != "a4"]
    dropped = inc.source_by_user(incidence, kept).set_index("user_id")
    assert dropped.loc["u3", "public_source_count"] == 0
    assert dropped.loc["u3", "celebrity_source_count"] == 0


def test_account_index_and_category_are_carried(synthetic_tables):
    """账号索引与类别随矩阵一起返回，供 §13.5 按类别剔除使用"""
    incidence = inc.build_user_account_incidence(YEAR)
    assert incidence.account_index[("a1", "public")] < incidence.matrix.shape[1]
    row = incidence.accounts.set_index(["r_user_id", "source_domain"])
    assert row.loc[("a1", "public"), "source_category"] == "central_media"
    assert row.loc[("a3", "celebrity"), "source_category"] == "star"


# ---------------------------------------------------------------------------
# 内存与规模的可见性
# ---------------------------------------------------------------------------

def test_build_prints_shape_nnz_and_memory(synthetic_tables, capsys):
    """构建时必须打印形状、非零元素数与内存占用，供作业申请资源时判断"""
    inc.build_post_term_incidence(YEAR, "public")
    out = capsys.readouterr().out
    assert "形状" in out and "非零" in out and "MB" in out
    inc.build_user_account_incidence(YEAR)
    out = capsys.readouterr().out
    assert "形状" in out and "非零" in out and "MB" in out


# ---------------------------------------------------------------------------
# 峰值内存：压缩必须发生在 concat 之前
# ---------------------------------------------------------------------------

def test_每个分片在进入concat之前就已经压好(synthetic_tables, monkeypatch):
    """pd.concat 拿到的分片必须已经是压过的 dtype，不能是 object 字符串

    这一条守的是一次真实的 OOM。压缩原来全写在 pd.concat **后面**，于是
    循环里堆的是原始 dtype——三个字符串列都是 object，每个值一个独立的
    Python 字符串对象。2020 年表 A 一亿四千七百万帖，光这三列就堆到约
    26 GB，concat 再翻一倍到约 59 GB，而作业内存上限是
    cpus-per-task × 4000M（8 核 32 GB / 4 核 16 GB）。词表、测量、语境抽样
    三个族全部在读分片的中途被杀，谁都没活到那几行压缩。

    所以断言必须落在**顺序**上：只查最终 posts 的 dtype 是查不出这个 bug 的，
    压缩写在 concat 之后同样能得到一模一样的最终结果。这里把 pd.concat
    换成一个探针，检查它收到的那些分片本身。
    """
    seen = []
    real_concat = pd.concat

    def spy(objs, *args, **kwargs):
        frames = list(objs)
        seen.extend(dict(f.dtypes) for f in frames)
        return real_concat(frames, *args, **kwargs)

    monkeypatch.setattr(inc.pd, "concat", spy)
    inc.build_post_term_incidence(YEAR, "public")

    assert seen, "没有截到 pd.concat 收到的分片"
    for dtypes in seen:
        for column in ("weibo_id", "user_id", "post_type"):
            assert dtypes[column] != object, (
                f"{column} 还是 object dtype 就进了 concat："
                "压缩又被挪回 concat 之后了"
            )
        assert dtypes["n_chars"] == np.int32
        assert dtypes["month"] == np.int8


def test_压缩之后最终的posts列类型没有变(synthetic_tables):
    """把压缩挪到循环里，不能顺手改掉下游看到的 dtype

    user_id / post_type 仍然要在 concat 之后转成 category（码表要等所有
    分片到齐才好定），weibo_id 仍然是 string[pyarrow]——§13.4 要靠它回溯
    原帖。这条是纯粹的回归护栏。
    """
    posts = inc.build_post_term_incidence(YEAR, "public").posts
    assert isinstance(posts["user_id"].dtype, pd.CategoricalDtype)
    assert isinstance(posts["post_type"].dtype, pd.CategoricalDtype)
    assert posts["weibo_id"].dtype == "string[pyarrow]"
    assert posts["n_chars"].dtype == np.int32
    assert posts["month"].dtype == np.int8
    assert posts["is_expressive"].dtype == bool


def test_不要weibo_id时它根本不进内存(synthetic_tables):
    """keep_weibo_id=False 必须让这一列连读都不读，而不是读进来再 drop

    weibo_id 逐帖唯一，是 posts 帧里最大的一列（真实规模约 3.4 GB，占整帧
    一半以上），而只有 §13.4 语境抽样用得到它。读进来再删等于峰值照付，
    所以断言落在"列不存在"上，并且 parquet 的 columns= 也不该点它。
    """
    inc_kept = inc.build_post_term_incidence(YEAR, "public", keep_weibo_id=True)
    assert "weibo_id" in inc_kept.posts.columns

    inc_pruned = inc.build_post_term_incidence(YEAR, "public", keep_weibo_id=False)
    assert "weibo_id" not in inc_pruned.posts.columns

    # 剪掉一列不许动到结果：矩阵与每个用户的重聚合份额必须完全一致
    assert inc_pruned.matrix.shape == inc_kept.matrix.shape
    kept = inc.topical_by_user(inc_kept, PUBLIC_VOCAB, inc_kept.posts)
    pruned = inc.topical_by_user(inc_pruned, PUBLIC_VOCAB, inc_pruned.posts)
    pd.testing.assert_frame_equal(
        kept.sort_values("user_id").reset_index(drop=True),
        pruned.sort_values("user_id").reset_index(drop=True),
    )
