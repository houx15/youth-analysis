"""
gender_domain.robustness.vocabulary 的单元测试（§13.3 词表稳健性）。

这份测试的重心有三条，按重要性排列：

1. **校准路径必须真的被测到。**
   incidence 模块的文档已经实测过：按存量逐词计数重新聚合，在"剔除的长词
   遮蔽了保留的短词"时会**恒向低估**（每个 replicate 丢 2.3%-5.4% 的命中
   帖，逐用户 topical_share 平均偏 -0.013 至 -0.032）。这与 §13.3 本身想
   检验的效应是同一个数量级，所以本模块必须对一部分 replicate 用精确重扫
   原文做校准。夹具因此**刻意含有真实的嵌套遮蔽**（"疫情" ⊂ "疫情防控"、
   "复工" ⊂ "复工复产"、"粉丝" ⊂ "粉丝团"），并且有帖子只写了长词——
   如果重扫与重聚合在测试里从来不打架，那这条校准路径就是没被测过的。
   test_calibration_recovers_exactly_the_posts_reaggregation_misses 里的
   真值不是手写的期望值，而是用真实的 process_frame + aggregate_posts
   在削减后的词表上**重扫一遍原文**跑出来的。

2. **重采样必须可复现、并且真的分层。**
   同一个 seed 两次必须逐词相同；每个长度层、每个频次层的保留比例都必须
   落在 keep 上——"随机保留 80%"如果实际上把短词整批留下、长词整批扔掉，
   得到的就不是一个词表扰动，而是一次"只测短词"的变体。

3. **拿不到类别信息时必须留一行说明，而不是从字符串上猜类别。**
   两份词表文件都是逐行纯词，没有任何类别列。留一类别、"只保留人名"这
   两个变体因此在真实词表上无法定义——测试钉死的是"输出里有一行说明它
   为什么做不了"，而不是"猜出了一批人名"。
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from gender_domain import build_post_table as bpt
from gender_domain import build_retweet_table as brt
from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import text_rules as tr
from gender_domain.robustness import harness
from gender_domain.robustness import incidence as inc
from gender_domain.robustness import vocabulary as voc


YEAR = 2020

# 合成词表。两类掩盖关系都是**故意**放进来的，没有它们重聚合与重扫永远
# 一致，校准路径就等于没测：
#   子串嵌套：疫情 ⊂ 疫情防控、复工 ⊂ 复工复产、粉丝 ⊂ 粉丝团
#   边界重叠：两会代表 的后缀 "代表" 接 代表委员 的前缀（互不包含！）
# "封城" 全年一次都没出现，用来覆盖"词在词表里、但一列都没进矩阵"
# （频次为 0）这条分层路径。
PUBLIC_VOCAB = [
    "疫情", "疫情防控", "复工", "复工复产", "两会代表", "代表委员", "口罩", "封城",
]
CELEBRITY_VOCAB = ["顶流", "粉丝", "粉丝团", "综艺", "打榜"]

PUBLIC_ACCOUNTS = {"central_media": {"a1", "a2"}, "local_media": {"a5"}}
CELEBRITY_ACCOUNTS = {"star": {"a3"}, "fanclub": {"a4"}}

# 帖子正文模板。第 0、1 条只含长词（"疫情防控" / "复工复产"），它们正是
# 剔除长词后重聚合会漏掉、重扫会重新命中的那一批。
POST_TEMPLATES = [
    "今天疫情防控形势不错",
    "复工复产的第一天",
    "疫情之下的日常生活",
    # 全词表下最左最长只记 {两会代表: 1}；剔掉 "两会代表" 后重扫会命中
    # "代表委员"，而两者互不包含——只看子串的口径完全看不见这一条
    "两会代表委员都到齐了",
    "粉丝团又在打榜",
    "顶流的综艺真好看",
    "普通的一天没什么可说",
    "复工了 粉丝好多",
]

N_USERS = 40
POSTS_PER_USER = 6


def _raw_posts():
    """合成的原始帖子（cleaned_weibo_cov 口径），交给真实的 process_frame 处理"""
    rows = []
    for i in range(N_USERS):
        uid = "u{:03d}".format(i)
        gender = "m" if i % 2 == 0 else "f"
        for j in range(POSTS_PER_USER):
            month = 1 if (i + j) % 2 == 0 else 2
            day = 1 + ((i * POSTS_PER_USER + j) % 20)
            # 男性多写公共事务模板、女性多写明星模板，让六个量有一点信号，
            # 模型不至于在一个完全无差异的表上退化
            offset = 0 if gender == "m" else 4
            content = POST_TEMPLATES[(i + j + offset) % len(POST_TEMPLATES)]
            rows.append({
                "weibo_id": "p{:03d}_{:d}".format(i, j),
                "user_id": uid,
                "weibo_content": content,
                "is_retweet": "1" if j == POSTS_PER_USER - 1 else "0",
                "gender": gender,
                "province": "11",
                "time_stamp": 1580000000,
                "date": "2020-{:02d}-{:02d}".format(month, day),
            })
    return pd.DataFrame(rows)


def _raw_retweets():
    """合成的原始转发记录，交给真实的 build_retweet_table.process_frame 处理"""
    accounts = ["a1", "a2", "a3", "a4", "a5", "zzz"]
    rows = []
    for i in range(N_USERS):
        uid = "u{:03d}".format(i)
        gender = "m" if i % 2 == 0 else "f"
        for j in range(3):
            rows.append({
                "user_id": uid,
                "weibo_id": "rw{:03d}_{:d}".format(i, j),
                "r_weibo_id": "s{:03d}_{:d}".format(i, j),
                "r_user_id": accounts[(i + j * 2) % len(accounts)],
                "is_retweet": "1",
                "gender": gender,
                "time_stamp": 1580000100,
                "r_time_stamp": 1580000000,
                "date": "2020-{:02d}-{:02d}".format(1 if i % 2 == 0 else 2, 1 + j),
            })
    return pd.DataFrame(rows)


def _write_raw_day_files(raw_dir, posts):
    """把原始帖子按天写成 cleaned_weibo_cov 的日文件，供精确重扫读取"""
    year_dir = os.path.join(raw_dir, str(YEAR))
    os.makedirs(year_dir, exist_ok=True)
    for date, part in posts.groupby("date"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(year_dir, "{}.parquet".format(date)),
            engine="pyarrow", index=False,
        )


@pytest.fixture()
def synthetic_project(tmp_path, monkeypatch):
    """写出互相自洽的原始日文件、表 A、表 B、表 C，并把 config 的路径指过去

    表 C 由真实的 build_user_tables.build 从刚写出的表 A/表 B 生成，
    原始日文件与表 A 来自同一批帖子——精确重扫读回来的正文，确确实实
    就是当初生成表 A 那一批正文。
    """
    out_dir = str(tmp_path / "analysis_data")
    raw_dir = str(tmp_path / "raw")
    os.makedirs(out_dir, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "DATA_DIR", raw_dir)
    monkeypatch.setattr(config, "load_public_vocabulary", lambda year=YEAR: list(PUBLIC_VOCAB))
    monkeypatch.setattr(
        config, "load_celebrity_vocabulary", lambda year=YEAR: list(CELEBRITY_VOCAB)
    )

    raw = _raw_posts()
    _write_raw_day_files(raw_dir, raw)

    posts = bpt.process_frame(
        raw, tr.VocabMatcher(PUBLIC_VOCAB), tr.VocabMatcher(CELEBRITY_VOCAB)
    )
    post_dir = os.path.join(out_dir, "post_domain_measures_{}".format(YEAR))
    os.makedirs(post_dir, exist_ok=True)
    for month, part in posts.groupby("month"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(post_dir, "month={:02d}.parquet".format(month)),
            engine="pyarrow", index=False,
        )

    events = brt.process_frame(
        _raw_retweets(),
        brt.build_account_lookup(PUBLIC_ACCOUNTS),
        brt.build_account_lookup(CELEBRITY_ACCOUNTS),
    )
    event_dir = os.path.join(out_dir, "retweet_domain_events_{}".format(YEAR))
    os.makedirs(event_dir, exist_ok=True)
    for month, part in events.groupby("month"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(event_dir, "month={:02d}.parquet".format(month)),
            engine="pyarrow", index=False,
        )

    but.build(YEAR)
    return {"out_dir": out_dir, "raw_dir": raw_dir, "raw": raw, "posts": posts}


@pytest.fixture()
def context(synthetic_project):
    return voc.build_context(YEAR)


def _rescan_truth(public_vocab, celebrity_vocab):
    """真值：用真实流水线在给定词表上重扫原文，再逐用户聚合"""
    rescanned = bpt.process_frame(
        _raw_posts(), tr.VocabMatcher(public_vocab), tr.VocabMatcher(celebrity_vocab)
    )
    return but.aggregate_posts(rescanned[but.POST_SHARD_COLUMNS])


# ---------------------------------------------------------------------------
# 分层重采样
# ---------------------------------------------------------------------------

def test_stratified_resample_is_reproducible_under_a_seed():
    terms = ["词{}".format(i) * (1 + i % 4) for i in range(200)]
    first = voc.stratified_resample(terms, keep=0.8, seed=7)
    second = voc.stratified_resample(terms, keep=0.8, seed=7)
    assert first == second
    other = voc.stratified_resample(terms, keep=0.8, seed=8)
    assert other != first


def test_stratified_resample_balances_length_strata():
    """每个长度层的保留比例都必须是 keep，不能整批留下短词、扔掉长词"""
    terms = (
        ["{}{}".format(chr(0x4E00 + i), chr(0x4E50 + i)) for i in range(50)]      # 长度 2
        + ["{}{}{}".format(chr(0x5E00 + i), "中", "文") for i in range(30)]        # 长度 3
        + ["{}{}{}{}{}".format(chr(0x6E00 + i), "甲", "乙", "丙", "丁") for i in range(20)]  # 长度 5
    )
    kept = voc.stratified_resample(terms, keep=0.8, seed=0)
    by_length = {}
    for term in terms:
        by_length.setdefault(len(term), []).append(term)
    kept_set = set(kept)
    for length, group in by_length.items():
        retained = sum(1 for t in group if t in kept_set)
        assert retained == round(0.8 * len(group)), (
            "长度 {} 层保留 {} / {}，不是 80%".format(length, retained, len(group))
        )


def test_stratified_resample_balances_corpus_frequency_strata(context):
    """频次层同样必须均衡，而且频次来自矩阵实测，不是假设出来的"""
    vocab = PUBLIC_VOCAB
    strata = voc.term_strata(context.incidence["public"], vocab, n_frequency_bins=2)
    frequencies = voc.term_document_frequency(context.incidence["public"], vocab)
    # 全年一次没出现的词，频次必须是 0（而不是被静默当成低频真实词）
    assert frequencies["封城"] == 0
    assert frequencies["疫情防控"] > 0
    # 分层键里必须同时含长度与频次两个维度
    assert len(set(strata.values())) > 1
    assert strata["封城"] != strata["疫情防控"] or len("封城") != len("疫情防控")

    kept = set(voc.stratified_resample(vocab, keep=0.5, strata=strata, seed=3))
    by_stratum = {}
    for term in vocab:
        by_stratum.setdefault(strata[term], []).append(term)
    for key, group in by_stratum.items():
        retained = sum(1 for t in group if t in kept)
        assert retained == round(0.5 * len(group)), (
            "分层 {} 保留 {} / {}".format(key, retained, len(group))
        )


def test_replicate_seeds_are_distinct_and_reproducible():
    first = voc.replicate_seeds(0, 20)
    assert len(set(first)) == 20
    assert first == voc.replicate_seeds(0, 20)
    assert first != voc.replicate_seeds(1, 20)


# ---------------------------------------------------------------------------
# 风险计数：有掩盖关系时非零、没有时为零
# ---------------------------------------------------------------------------

def test_at_risk_term_count_is_nonzero_when_a_dropped_term_can_mask_a_retained_one(context):
    subset = [t for t in PUBLIC_VOCAB if t not in ("疫情防控", "复工复产")]
    diag = voc.diagnostics_row(
        context, "public", subset, variant_label="drop_long", replicate=0, seed=1
    )
    assert diag["n_at_risk_terms"] >= 2          # 疫情、复工 都被掩盖
    assert diag["n_posts_with_at_risk_term"] > 0
    assert diag["n_expressive_posts_possibly_lost"] > 0
    assert diag["n_retained_terms"] == len(subset)
    assert diag["retained_fraction"] == pytest.approx(len(subset) / len(PUBLIC_VOCAB))
    assert diag["seed"] == 1


def test_at_risk_count_is_zero_when_the_dropped_term_can_mask_nothing(context):
    """剔除一个与其它词既不嵌套也不重叠的词，风险计数必须是 0"""
    subset = [t for t in PUBLIC_VOCAB if t != "口罩"]
    assert "口罩" not in inc.nested_terms(PUBLIC_VOCAB)
    assert inc.at_risk_terms(PUBLIC_VOCAB, subset) == set()
    diag = voc.diagnostics_row(
        context, "public", subset, variant_label="drop_kouzhao", replicate=0, seed=1
    )
    assert diag["n_at_risk_terms"] == 0
    assert diag["n_shadowed_terms"] == 0
    assert diag["n_posts_with_at_risk_term"] == 0
    assert diag["n_expressive_posts_possibly_lost"] == 0


def test_diagnostics_report_the_full_risk_set_beside_the_substring_only_subset(context):
    """诊断行必须让读者看见"只看子串"比完整口径小多少

    剔掉 "两会代表" 是一次纯边界重叠：它与 "代表委员" 互不包含，只看子串
    的口径报 0，完整口径报 1。两个数字并排写在同一行上，才可能看出后者
    大多少——只写一个数字，读者无从判断风险集是不是被低估了。
    """
    subset = [t for t in PUBLIC_VOCAB if t != "两会代表"]
    diag = voc.diagnostics_row(
        context, "public", subset, variant_label="drop_lianghui", replicate=0, seed=1
    )
    assert diag["n_shadowed_terms"] == 0
    assert diag["n_expressive_posts_possibly_lost_substring_only"] == 0
    assert diag["n_at_risk_terms"] == 1
    assert diag["n_expressive_posts_possibly_lost"] > 0


# ---------------------------------------------------------------------------
# 校准：精确重扫 vs 存量重聚合
# ---------------------------------------------------------------------------

def test_calibration_recovers_a_boundary_overlap_miss_that_substring_nesting_never_sees(
    context,
):
    """纯边界重叠：只看子串的口径一条都不会重扫，完整口径必须找回来

    剔掉 "两会代表"、保留 "代表委员"，两者互不包含：
    shadowing_dropped_terms 返回空集，按旧口径这条帖子根本不会进重扫名单，
    校准会报"零偏差"——而真值明明变了。这条测试把那个盲区钉死。
    """
    subset = [t for t in PUBLIC_VOCAB if t != "两会代表"]
    assert inc.shadowing_dropped_terms(PUBLIC_VOCAB, subset) == set()
    assert inc.at_risk_dropped_terms(PUBLIC_VOCAB, subset) == {"两会代表"}

    reagg, corrected, stats = voc.calibrate_topical(context, "public", subset)
    truth = _rescan_truth(subset, CELEBRITY_VOCAB)
    merged = truth[["user_id", "public_topical_posts"]].merge(
        corrected, on="user_id", how="left"
    ).merge(
        reagg[["user_id", "topical_posts"]], on="user_id", how="left",
        suffixes=("", "_reagg"),
    )
    pd.testing.assert_series_equal(
        merged["topical_posts"].astype("int64"),
        merged["public_topical_posts"].astype("int64"),
        check_names=False,
    )
    assert (merged["topical_posts_reagg"] < merged["topical_posts"]).any()
    assert stats["n_expressive_posts_recovered"] > 0


def test_calibration_recovers_exactly_the_posts_reaggregation_misses(context):
    """校准后的逐用户命中帖数必须等于真正重扫原文的真值，且真的与重聚合不同

    这是本模块最重要的一条断言。真值来自真实的 process_frame +
    aggregate_posts，不是手写的期望值。
    """
    subset = [t for t in PUBLIC_VOCAB if t not in ("疫情防控", "复工复产")]
    reagg, corrected, stats = voc.calibrate_topical(context, "public", subset)
    truth = _rescan_truth(subset, CELEBRITY_VOCAB)

    merged = truth[["user_id", "public_topical_posts", "public_topical_share"]].merge(
        corrected, on="user_id", how="left"
    ).merge(
        reagg[["user_id", "topical_posts"]], on="user_id", how="left",
        suffixes=("", "_reagg"),
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
    # 重聚合必须**确实**漏掉了一批帖子，否则这条测试对校准逻辑毫无约束力
    assert (merged["topical_posts_reagg"] < merged["topical_posts"]).any()
    assert stats["n_expressive_posts_recovered"] > 0
    # 偏差方向恒为低估：逐用户 Δ = 重聚合 - 真值 <= 0
    assert stats["mean_delta_topical_share"] < 0
    assert stats["max_abs_delta_topical_share"] > 0
    # 上界必须真的是上界
    assert stats["n_expressive_posts_recovered"] <= stats["n_expressive_posts_possibly_lost"]


def test_calibration_finds_no_bias_when_nothing_can_be_masked(context):
    """没有掩盖关系时，校准必须报告零偏差（而不是报告一个凭空的修正）"""
    subset = [t for t in PUBLIC_VOCAB if t != "口罩"]
    reagg, corrected, stats = voc.calibrate_topical(context, "public", subset)
    pd.testing.assert_frame_equal(
        reagg.reset_index(drop=True), corrected.reset_index(drop=True)
    )
    assert stats["n_expressive_posts_recovered"] == 0
    assert stats["mean_delta_topical_share"] == 0.0
    # 随机抽样检验同样必须报 0：这一次它确实什么都没漏
    assert stats["n_probe_flipped"] == 0


def test_random_probe_detects_misses_the_risk_set_definition_would_have_hidden(context):
    """独立检验：把风险集人为置空，随机抽样仍然必须看见漏判

    这是"理论想错了也兜得住"那条证据的测试。把 at_risk_dropped_terms 打成
    恒返回空集（等价于"我们对匹配失效方式的理解完全错了"），风险集内的
    重扫会一条都不做、报告零偏差；随机抽样不看词表关系，因此仍然必须抽到
    翻案帖，并且把它们全部记在"风险集之外"。
    """
    subset = [t for t in PUBLIC_VOCAB if t not in ("疫情防控", "复工复产", "两会代表")]
    honest = voc.calibrate_topical(context, "public", subset, n_probe=10 ** 6)[2]
    assert honest["n_probe_flipped"] > 0
    assert honest["n_probe_flipped_outside_at_risk"] == 0    # 理论正确时全在集内

    saved = inc.at_risk_dropped_terms
    inc.at_risk_dropped_terms = lambda *a, **k: set()
    try:
        blind = voc.calibrate_topical(context, "public", subset, n_probe=10 ** 6)[2]
    finally:
        inc.at_risk_dropped_terms = saved

    assert blind["n_expressive_posts_recovered"] == 0          # 风险集内什么都没看见
    assert blind["mean_delta_topical_share"] == 0.0
    assert blind["n_probe_flipped"] > 0                        # 抽样照样看得见
    assert blind["n_probe_flipped_outside_at_risk"] == blind["n_probe_flipped"]
    assert blind["probe_implied_missed_posts"] > 0
    assert blind["probe_flip_rate_ci_low"] > 0


def test_resampling_exercises_the_boundary_overlap_correction_end_to_end(
    context, tmp_path
):
    """整条 run_resampling 里必须真的走到边界重叠修正，而不是只在单测里走过

    只在 calibrate_topical 上单独测过的修正路径，正是上一轮 Critical 的
    来源：单点正确、整条流水线没接上，测试也发现不了。这里断言重采样里
    确实存在这样一个 replicate——它剔掉的词一个都不是子串嵌套关系
    （n_shadowed_terms == 0），却仍然有帖子被重扫翻案。
    """
    diag_path = str(tmp_path / "diag.parquet")
    voc.run_resampling(
        YEAR, n_replicates=3, keep=0.8, seed=0, n_calibration=3, n_probe=50,
        context=context, out_path=str(tmp_path / "vocabulary.parquet"),
        diag_path=diag_path,
    )
    diag = pd.read_parquet(diag_path, columns=list(voc.DIAGNOSTIC_SCHEMA))
    base = diag[(diag["variant_family"] == voc.VARIANT_FAMILY)
                & (diag["domain"] == "public")]
    overlap_only = base[
        (base["n_shadowed_terms"] == 0)
        & (base["n_at_risk_terms"] > 0)
        & (base["n_expressive_posts_recovered"] > 0)
    ]
    assert len(overlap_only) > 0, (
        "没有任何 replicate 走到纯边界重叠的修正路径，夹具没覆盖到这条路："
        + base[["variant_label", "n_shadowed_terms", "n_at_risk_terms",
                "n_expressive_posts_recovered"]].to_string()
    )
    # 这些 replicate 的重扫名单必须确实比只看子串的名单大
    row = overlap_only.iloc[0]
    assert row["n_expressive_posts_possibly_lost"] > 0
    assert row["n_expressive_posts_possibly_lost_substring_only"] == 0


def test_exposure_is_computed_once_per_replicate_and_domain(context, tmp_path, monkeypatch):
    """暴露面每个 (replicate, 领域) 只能算一次

    每算一份是两次稀疏矩阵-向量乘法，真实数据上是两次三千多万行的整表
    扫描；校准过的 replicate 会用到四处（一次校准 + 三行诊断），各自现算
    就是十来遍白扫。
    """
    calls = {"full": 0, "narrow": 0}
    real_full = inc.reaggregation_exposure
    real_narrow = inc.shadowing_exposure

    def _full(*args, **kwargs):
        calls["full"] += 1
        return real_full(*args, **kwargs)

    def _narrow(*args, **kwargs):
        calls["narrow"] += 1
        return real_narrow(*args, **kwargs)

    monkeypatch.setattr(inc, "reaggregation_exposure", _full)
    monkeypatch.setattr(inc, "shadowing_exposure", _narrow)
    voc.run_resampling(
        YEAR, n_replicates=1, keep=0.8, seed=0, n_calibration=1, n_probe=20,
        context=context, out_path=str(tmp_path / "vocabulary.parquet"),
        diag_path=str(tmp_path / "diag.parquet"),
    )
    # 1 个 replicate × 2 个领域，一次校准 + 3 行诊断，全部共用同一份
    assert calls["full"] == len(voc.DOMAINS)
    assert calls["narrow"] == len(voc.DOMAINS)


def test_probe_draw_uses_a_stream_independent_of_the_term_draw():
    """抽帖子的随机流必须与抽词的分开——独立检验不该与被检验的对象同源"""
    seed = voc.replicate_seeds(0, 1)[0]
    for domain in voc.DOMAINS:
        terms_seed = voc._domain_seed(seed, domain, stream=voc._STREAM_TERMS)
        probe_seed = voc._domain_seed(seed, domain, stream=voc._STREAM_PROBE)
        assert terms_seed != probe_seed
    # 领域之间也必须分开，且两次派生完全可复现
    assert (voc._domain_seed(seed, "public", stream=voc._STREAM_PROBE)
            != voc._domain_seed(seed, "celebrity", stream=voc._STREAM_PROBE))
    assert (voc._domain_seed(seed, "public", stream=voc._STREAM_PROBE)
            == voc._domain_seed(seed, "public", stream=voc._STREAM_PROBE))


def test_random_probe_reports_a_confidence_interval_not_just_a_point(context):
    """抽到 0 条翻案不等于漏判率为 0，必须给出区间上限"""
    subset = [t for t in PUBLIC_VOCAB if t != "口罩"]
    stats = voc.calibrate_topical(context, "public", subset, n_probe=20)[2]
    assert stats["n_probe_flipped"] == 0
    assert stats["probe_flip_rate"] == 0.0
    assert stats["probe_flip_rate_ci_low"] == 0.0
    assert 0.0 < stats["probe_flip_rate_ci_high"] < 1.0
    assert stats["n_probe_sampled"] <= stats["n_nonhit_expressive_posts"]


def test_rescan_reads_raw_text_only_for_the_months_it_needs(context, monkeypatch):
    """精确重扫必须只读受影响帖子所在月份的日文件，不是重扫全年"""
    subset = [t for t in PUBLIC_VOCAB if t not in ("疫情防控",)]
    affected = voc.at_risk_affected_posts(context.incidence["public"], subset, PUBLIC_VOCAB)
    assert len(affected) > 0
    assert set(affected.columns) >= {"weibo_id", "month"}

    read_files = []
    real_read = pd.read_parquet

    def _spy(path, *args, **kwargs):
        read_files.append(str(path))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(voc.pd, "read_parquet", _spy)
    hits = voc.rescan_hits(YEAR, affected, subset)
    assert set(hits["weibo_id"]) == set(affected["weibo_id"])
    needed_months = {"{}-{:02d}".format(YEAR, m) for m in affected["month"].unique()}
    for path in read_files:
        assert any(month in os.path.basename(path) for month in needed_months), (
            "重扫读到了不需要的日文件: {}".format(path)
        )


def test_calibration_reads_the_raw_layer_only_once_for_all_replicates(
    context, tmp_path, monkeypatch
):
    """多个 replicate 做校准时，全年日文件只能被扫一遍

    每个被校准的 replicate 各扫一遍原始层，在真实数据上是几个小时的纯
    IO（一年三千多万条正文 × 20 次）。受影响帖子的并集是纯矩阵运算，
    读正文之前就能算出来，所以只该读一次。
    """
    raw_reads = []
    real_read = pd.read_parquet

    def _spy(path, *args, **kwargs):
        if str(path).startswith(str(config.DATA_DIR)):
            raw_reads.append(str(path))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(voc.pd, "read_parquet", _spy)
    voc.run_resampling(
        YEAR, n_replicates=4, keep=0.6, seed=1, n_calibration=4, n_probe=20,
        context=context,
        out_path=str(tmp_path / "vocabulary.parquet"),
        diag_path=str(tmp_path / "diag.parquet"),
    )
    # 先确认这次真的走到了原始层（否则下面那条断言在"一次都没读"上
    # 也会通过，毫无信息量）
    assert len(raw_reads) > 0
    assert len(raw_reads) == len(set(raw_reads)), "同一个原始日文件被读了不止一遍"


# ---------------------------------------------------------------------------
# 词表构成变体
# ---------------------------------------------------------------------------

def test_drop_short_terms_retains_the_expected_counts_on_the_real_vocabularies():
    """真实词表上 min_len=3 剔除 337/816 公共事务词、182/535 明星词"""
    public = config.load_public_vocabulary(2020)
    celebrity = config.load_celebrity_vocabulary(2020)
    assert len(public) == 816 and len(celebrity) == 535
    assert len(public) - len(voc.drop_short_terms(public, 3)) == 337
    assert len(celebrity) - len(voc.drop_short_terms(celebrity, 3)) == 182


def test_drop_nested_removes_exactly_the_nested_terms():
    kept = voc.drop_nested_terms(PUBLIC_VOCAB)
    assert set(PUBLIC_VOCAB) - set(kept) == inc.nested_terms(PUBLIC_VOCAB)
    assert "疫情" not in kept and "疫情防控" in kept


def test_leave_one_category_out_removes_exactly_that_category(context, tmp_path):
    categories = {
        "public": {"epidemic": ["疫情", "疫情防控", "口罩"],
                   "economy": ["复工", "复工复产"],
                   "politics": ["两会代表", "代表委员", "封城"]},
        "celebrity": {"person": ["顶流"], "fandom": ["粉丝", "粉丝团"],
                      "works": ["综艺", "打榜"]},
    }
    out_path = str(tmp_path / "vocabulary.parquet")
    diag_path = str(tmp_path / "vocabulary_diagnostics.parquet")
    rows = voc.run_leave_one_category_out(
        YEAR, categories=categories, context=context,
        out_path=out_path, diag_path=diag_path,
    )
    labels = set(rows["variant_label"])
    assert "public_without_epidemic" in labels
    diag = pd.read_parquet(diag_path, columns=list(voc.DIAGNOSTIC_SCHEMA))
    epidemic = diag[diag["variant_label"] == "public_without_epidemic"].iloc[0]
    assert epidemic["n_retained_terms"] == len(PUBLIC_VOCAB) - 3
    assert epidemic["domain"] == "public"


def test_leave_one_category_out_without_categories_emits_a_single_noted_row(
    context, tmp_path
):
    out_path = str(tmp_path / "vocabulary.parquet")
    rows = voc.run_leave_one_category_out(
        YEAR, categories=None, context=context, out_path=out_path,
    )
    assert len(rows) == len(voc.DOMAINS)
    assert all("categor" in str(note) for note in rows["note"])
    assert set(rows["variant_family"]) == {voc.VARIANT_FAMILY}
    assert set(rows.columns) == set(harness.ROBUSTNESS_SCHEMA)


def test_celebrity_person_only_without_categories_emits_a_single_noted_row(
    context, tmp_path
):
    """词表文件没有类别列时，只能留一行说明，绝不能从字符串上猜哪个是作品名"""
    out_path = str(tmp_path / "vocabulary.parquet")
    rows = voc.run_celebrity_person_only(YEAR, context=context, out_path=out_path)
    assert len(rows) == 1
    note = str(rows.iloc[0]["note"])
    assert "categor" in note
    assert rows.iloc[0]["variant_label"] == "celebrity_person_only_unavailable"
    assert set(rows.columns) == set(harness.ROBUSTNESS_SCHEMA)


def test_drop_short_terms_variant_carries_the_retained_fraction(context, tmp_path):
    out_path = str(tmp_path / "vocabulary.parquet")
    diag_path = str(tmp_path / "vocabulary_diagnostics.parquet")
    rows = voc.run_drop_short_terms(
        YEAR, min_len=3, context=context, out_path=out_path, diag_path=diag_path
    )
    diag = pd.read_parquet(diag_path, columns=list(voc.DIAGNOSTIC_SCHEMA))
    public = diag[(diag["domain"] == "public") & (diag["variant_label"] == "drop_short_terms_min3")]
    assert len(public) == 1
    expected = len(voc.drop_short_terms(PUBLIC_VOCAB, 3)) / len(PUBLIC_VOCAB)
    assert public.iloc[0]["retained_fraction"] == pytest.approx(expected)

    # 结果表自己也要说清这次干预有多大：只拿到 vocabulary.parquet 一张表的
    # 读者，不该从 "drop_short_terms_min3" 这个标签去猜剔了几个词
    for note in rows["note"]:
        assert "min_len=3" in str(note)
        assert "public_retained={}/{}".format(
            len(voc.drop_short_terms(PUBLIC_VOCAB, 3)), len(PUBLIC_VOCAB)
        ) in str(note)


def test_note_annotation_never_overwrites_a_failure_note():
    """追加说明不能盖掉拟合失败的留痕——那是排查问题唯一的线索"""
    frame = pd.DataFrame({"note": [None, "harness_call_failed", np.nan]})
    out = voc._annotate_note(frame, "min_len=3")
    assert list(out["note"]) == [
        "min_len=3", "harness_call_failed;min_len=3", "min_len=3",
    ]


# ---------------------------------------------------------------------------
# 全流程
# ---------------------------------------------------------------------------

def test_build_writes_results_diagnostics_and_manifest_incrementally(synthetic_project):
    out = voc.build(YEAR, n_replicates=3, keep=0.8, seed=0, n_calibration=1)
    results = pd.read_parquet(out["results_path"], columns=list(harness.ROBUSTNESS_SCHEMA))
    diag = pd.read_parquet(out["diagnostics_path"], columns=list(voc.DIAGNOSTIC_SCHEMA))

    # 输出只允许落在 analysis_data/robustness/ 下
    robust_dir = os.path.join(config.OUTPUT_DIR, "robustness")
    assert out["results_path"].startswith(robust_dir)
    assert out["diagnostics_path"].startswith(robust_dir)

    # 每个 replicate 都记了 seed
    resampled = results[results["variant_label"].str.startswith("keep")]
    assert resampled["seed"].notna().all()
    assert resampled["replicate"].nunique() == 3
    # 每个 replicate 恰好 6 个量 × 2 层
    per_replicate = resampled.groupby(["variant_label"]).size()
    assert set(per_replicate.unique()) == {len(harness.QUANTITIES) * 2}

    # 校准的那一次 replicate 有配对的两组行
    calib = results[results["variant_family"] == voc.CALIBRATION_FAMILY]
    labels = set(calib["variant_label"])
    assert any(lab.endswith("_reaggregated") for lab in labels)
    assert any(lab.endswith("_rescanned") for lab in labels)
    assert len(calib) == 2 * len(harness.QUANTITIES) * 2

    # 诊断表逐 (variant, 领域) 一行，保留词数、风险词数、抽样检验都在
    assert set(diag["domain"]) == {"public", "celebrity"}
    assert (diag["n_retained_terms"] > 0).all()
    assert diag["calibrated"].any()
    calibrated = diag[diag["calibrated"]]
    assert calibrated["n_expressive_posts_recovered"].notna().all()
    assert calibrated["n_probe_sampled"].notna().all()
    assert calibrated["probe_flip_rate_ci_high"].notna().all()
    # 未校准的行必须是 NaN 而不是 0——"没测过"和"测出来是 0"要能分开
    assert diag[~diag["calibrated"]]["n_probe_sampled"].isna().all()

    manifest_path = os.path.join(robust_dir, "vocabulary_{}".format(YEAR), "manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["params"]["n_replicates"] == 3
    assert len(manifest["params"]["seeds"]) == 3
    assert "public" in manifest["fingerprints"]
    assert "random_probe" in manifest["counts"]


def test_every_result_row_that_should_have_diagnostics_can_find_them(synthetic_project):
    """结果表 -> 诊断表的连接键必须真的连得上

    连接键是 (variant_family, variant_label)。按设计只有两类行没有诊断：
    做不到的注明行，以及……没有第三类。domain="both" 的差中差行同样连得上
    该标签下的两行诊断（它本来就同时依赖两个领域）。
    """
    out = voc.build(YEAR, n_replicates=2, keep=0.8, seed=0, n_calibration=1)
    results = pd.read_parquet(out["results_path"], columns=list(harness.ROBUSTNESS_SCHEMA))
    diag = pd.read_parquet(out["diagnostics_path"], columns=list(voc.DIAGNOSTIC_SCHEMA))

    keys = set(zip(diag["variant_family"], diag["variant_label"]))
    unmatched = results[[
        (fam, lab) not in keys
        for fam, lab in zip(results["variant_family"], results["variant_label"])
    ]]
    # 未连上的只能是"做不到"的注明行
    assert set(unmatched["variant_label"]) == {
        "public_leave_one_category_out_unavailable",
        "celebrity_leave_one_category_out_unavailable",
        "celebrity_person_only_unavailable",
    }
    # 论文最可能引用的校准行必须连得上，而且不需要剥掉后缀
    calib = results[results["variant_family"] == voc.CALIBRATION_FAMILY]
    assert len(calib) > 0
    for fam, lab in set(zip(calib["variant_family"], calib["variant_label"])):
        assert (fam, lab) in keys
    # 差中差行（domain="both"）也在连接键之内
    both = results[results["domain"] == "both"]
    assert len(both) > 0
    for fam, lab in set(zip(both["variant_family"], both["variant_label"])):
        assert (fam, lab) in keys
    # 每个键下恰好两个领域各一行
    per_key = diag.groupby(["variant_family", "variant_label"])["domain"].nunique()
    assert set(per_key.unique()) == {2}


def test_build_is_resumable_because_rows_are_appended_per_replicate(
    synthetic_project, monkeypatch, tmp_path
):
    """一个 replicate 跑完就落盘：中途炸掉，已完成的 replicate 必须还在盘上"""
    context = voc.build_context(YEAR)
    out_path = str(tmp_path / "vocabulary.parquet")
    calls = {"n": 0}
    real_estimate_all = harness.estimate_all

    def _boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("模拟作业被杀")
        return real_estimate_all(*args, **kwargs)

    monkeypatch.setattr(voc.harness, "estimate_all", _boom)
    with pytest.raises(RuntimeError):
        voc.run_resampling(
            YEAR, n_replicates=5, context=context, out_path=out_path,
            diag_path=str(tmp_path / "diag.parquet"), n_calibration=0,
        )

    assert os.path.exists(out_path)
    done = pd.read_parquet(out_path, columns=list(harness.ROBUSTNESS_SCHEMA))
    assert len(done) > 0
    assert len(done) % (len(harness.QUANTITIES) * 2) == 0
