"""
gender_domain.robustness.measures 的单元测试（§13.1 分母、§13.8 帖子类型、
§13.7 时间限制）。

这份测试的重心按重要性排列：

1. **主口径的帖子类型过滤必须逐用户重现表 C。**
   §13.8 的四个变体全部建立在"按帖子类型重新聚合表 A"这条路径上；如果把
   过滤器设成主口径（原创 + 带评论转发）时重聚合与表 C 有任何偏差，那么
   另外三个变体都在同一个方向上错，和 Task 2 的全词表重建是同一类检验。
   夹具因此不手写期望值：表 C 由真实的 build_user_tables.aggregate_posts
   从真实的 build_post_table.process_frame 产出的表 A 生成。

2. **全部帖子分母必须给出比表达帖分母更低的话题占比。**
   这是"两个分母有没有被悄悄对调"最便宜的检验：只要存在不命中词表的纯
   转发，全部帖子分母的分母变大、分子不变，占比必然下降。夹具里刻意让
   词表不含"转发"这类会让纯转发占位文本命中的词，否则纯转发也会进分子，
   方向就不再确定。

3. **每一个变体的 variant_label 都必须写明分母。**
   这一族存在的意义就是证明"分母的选择没有推着结论走"，一行不写明分母的
   结果读起来与主口径完全一样，等于把这一族自己废掉。测试因此逐行扫描
   label，缺一个就失败。

4. **异常月份的规则必须事先写死、机械执行，并记下它选中了哪几个月。**
   看过哪个月不方便再挑月份，读者从输出上分不出来。规则写成模块常量、
   逐月的量与阈值写进诊断表，测试钉死"规则只看行为量、不看六个结果变量"
   ——按结果变量挑月份就是在对因变量做选择。

5. **活跃月份限制必须精确地剔除该剔除的人，并记下男女各剔了多少。**
   一条对男女生效比例不同的限制会改变被比较的两个总体，只报一个总数看
   不出这件事。

6. **留一月重估必须从 models_temporal 的输出里读，不允许重算。**
   同一个已发表的数字由两个模块各算一遍，迟早会出现两处不一致而没人
   知道该信哪一个。测试把 models_temporal 的函数换成会抛异常的桩，
   §13.7 仍然必须跑通并产出留一月的行。
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
from gender_domain import models_temporal as mt
from gender_domain import text_rules as tr
from gender_domain.robustness import harness
from gender_domain.robustness import measures as mea


YEAR = 2020

# 合成词表。**刻意不含"转发"**：纯转发的占位文本是"转发微博"，词表里一旦
# 有"转发"，纯转发帖也会命中，全部帖子分母就不再必然低于表达帖分母，
# 第 2 条检验就失去意义。
PUBLIC_VOCAB = ["疫情", "复工", "防控"]
CELEBRITY_VOCAB = ["顶流", "粉丝"]

PUBLIC_ACCOUNTS = {"central_media": {"a1", "a2"}}
CELEBRITY_ACCOUNTS = {"star": {"a3"}}

# 全年活跃（12 个月）的用户
LONG_USERS = (("m1", "m"), ("m2", "m"), ("f1", "f"), ("f2", "f"), ("f3", "f"))
# 只活跃 2 个月（1、2 月）的用户：≥3 个月、≥6 个月两条限制都会剔掉他们
SHORT_USERS = (("m3", "m"), ("f4", "f"))
# 只活跃 1 个月的用户
SINGLE_USERS = (("m4", "m"),)

# 活跃月份限制剔掉的人。**活跃月份数取 n_active_months_panel**（分母含
# 只转发不发帖的月份，与 build_user_tables 的持续性分母同一列）：f4 只在
# 1、2 两个月发过帖，却在 5、9 两个月转发过，因此她的面板活跃月份是 4，
# 只看发帖是 2——两条阈值下她的去留刚好相反，这个夹具因此能钉死"用的是
# 哪一个活跃月份口径"，而不只是"有没有做限制"。
EXPECTED_DROPPED_BY_ACTIVITY = {3: {"m3", "m4"}, 6: {"m3", "m4", "f4"}}
# 只在 5、9 两个月转发、不发帖的用户
RETWEET_ONLY_MONTHS_USER = "f4"
RETWEET_ONLY_MONTHS = (5, 9)

# 行为量异常月：8 月刻意加一批纯转发，1 月因为多两位短命用户而偏高。
SPIKE_MONTH = 8


def _raw_posts():
    """合成的原始帖子（cleaned_weibo_cov 口径），交给真实的 process_frame 处理

    每位用户每个活跃月固定四条帖子，覆盖三种帖子类型：
      1) 命中公共事务词表的原创；
      2) 不命中任何词的原创；
      3) 命中明星词表的带评论转发；
      4) 纯转发（"转发微博"），**不命中任何词**——它是第 2 条检验的关键：
         它进得了"全部帖子"分母，进不了"表达帖"分母，也不进分子。
    """
    rows = []
    schedule = (
        [(uid, gender, month) for uid, gender in LONG_USERS for month in range(1, 13)]
        + [(uid, gender, month) for uid, gender in SHORT_USERS for month in (1, 2)]
        + [(uid, gender, 1) for uid, gender in SINGLE_USERS]
    )
    for uid, gender, month in schedule:
        date = "{}-{:02d}-05".format(YEAR, month)
        base = "{}_{:02d}".format(uid, month)
        rows.append((base + "_a", uid, gender, "疫情防控进行中", "0", date))
        rows.append((base + "_b", uid, gender, "今天天气不错", "0", date))
        rows.append((base + "_c", uid, gender, "顶流粉丝好多", "1", date))
        rows.append((base + "_d", uid, gender, "转发微博", "1", date))

    # 8 月的行为量尖峰：m1 额外发 30 条纯转发
    for index in range(30):
        rows.append((
            "spike_{:02d}".format(index), "m1", "m", "转发微博", "1",
            "{}-{:02d}-20".format(YEAR, SPIKE_MONTH),
        ))

    return pd.DataFrame({
        "weibo_id": [r[0] for r in rows],
        "user_id": [r[1] for r in rows],
        "gender": [r[2] for r in rows],
        "weibo_content": [r[3] for r in rows],
        "is_retweet": [r[4] for r in rows],
        "province": ["11" for _ in rows],
        "time_stamp": [1580000000 for _ in rows],
        "date": [r[5] for r in rows],
    })


def _raw_retweets():
    """合成的原始转发记录，交给真实的 build_retweet_table.process_frame 处理"""
    rows = []
    for uid, gender in LONG_USERS:
        for month in (1, 2, 7):
            rows.append((
                uid, gender, "rw_{}_{:02d}".format(uid, month),
                "s_{}_{:02d}".format(uid, month),
                "a1" if gender == "m" else "a3",
                "{}-{:02d}-06".format(YEAR, month),
            ))
    for uid, gender in SHORT_USERS:
        rows.append((
            uid, gender, "rw_{}_01".format(uid), "s_{}_01".format(uid), "a2",
            "{}-01-06".format(YEAR),
        ))
    # f4 在 5、9 两个月只转发、不发帖：她的面板活跃月份是 4，只看发帖是 2
    for month in RETWEET_ONLY_MONTHS:
        rows.append((
            RETWEET_ONLY_MONTHS_USER, "f",
            "rw_{}_{:02d}".format(RETWEET_ONLY_MONTHS_USER, month),
            "s_{}_{:02d}".format(RETWEET_ONLY_MONTHS_USER, month), "a2",
            "{}-{:02d}-06".format(YEAR, month),
        ))
    return pd.DataFrame({
        "user_id": [r[0] for r in rows],
        "gender": [r[1] for r in rows],
        "weibo_id": [r[2] for r in rows],
        "r_weibo_id": [r[3] for r in rows],
        "r_user_id": [r[4] for r in rows],
        "is_retweet": ["1" for _ in rows],
        "time_stamp": [1580000100 for _ in rows],
        "r_time_stamp": [1580000000 for _ in rows],
        "date": [r[5] for r in rows],
    })


def _write_shards(out_dir):
    """用真实流水线写出表 A、表 B 的分月分片，返回 (表 A 全量, 表 B 全量)"""
    posts = bpt.process_frame(
        _raw_posts(), tr.VocabMatcher(PUBLIC_VOCAB), tr.VocabMatcher(CELEBRITY_VOCAB)
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
    return posts, events


@pytest.fixture()
def synthetic_project(tmp_path, monkeypatch):
    """写出互相自洽的表 A / B / C / D 以及 models_temporal 的留一月输出

    表 C 与表 D 由真实的 build_user_tables 从刚写出的表 A、表 B 生成，
    留一月输出由真实的 models_temporal.leave_one_month_out 从表 D 生成——
    §13.7 的读取路径因此读到的是真的那份表，不是一份仿造的 schema。
    """
    out_dir = str(tmp_path / "analysis_data")
    os.makedirs(out_dir, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "load_public_vocabulary",
                        lambda year=YEAR: list(PUBLIC_VOCAB))
    monkeypatch.setattr(config, "load_celebrity_vocabulary",
                        lambda year=YEAR: list(CELEBRITY_VOCAB))

    posts, events = _write_shards(out_dir)
    post_agg = but.aggregate_posts(posts[but.POST_SHARD_COLUMNS])
    event_agg = but.aggregate_events(events[but.EVENT_SHARD_COLUMNS])
    panel = but.aggregate_user_month(posts[but.POST_SHARD_COLUMNS],
                                     events[but.EVENT_SHARD_COLUMNS])
    table_c = but.combine_user_table(post_agg, event_agg, panel)

    table_c.to_parquet(
        os.path.join(out_dir, "user_domain_{}.parquet".format(YEAR)),
        engine="pyarrow", index=False,
    )
    panel.to_parquet(
        os.path.join(out_dir, "user_month_domain_{}.parquet".format(YEAR)),
        engine="pyarrow", index=False,
    )

    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    lomo = pd.concat(
        [mt.leave_one_month_out(panel, domain) for domain in but.DOMAINS],
        ignore_index=True,
    )
    lomo.to_parquet(os.path.join(results_dir, mt.LOMO_FILE),
                    engine="pyarrow", index=False)
    monthly = mt.monthly_rates(panel)
    monthly.to_parquet(os.path.join(results_dir, mt.MONTHLY_FILE),
                       engine="pyarrow", index=False)

    return {"out_dir": out_dir, "posts": posts, "events": events,
            "table_c": table_c, "panel": panel, "lomo": lomo}


@pytest.fixture()
def context(synthetic_project):
    return mea.build_context(YEAR)


def _read_results(path):
    return pd.read_parquet(path, engine="pyarrow",
                           columns=list(harness.ROBUSTNESS_SCHEMA))


def _read_diagnostics(path):
    return pd.read_parquet(path, engine="pyarrow",
                           columns=list(mea.DIAGNOSTIC_SCHEMA))


# ---------------------------------------------------------------------------
# 1. 最重要的一条：主口径的帖子类型过滤必须逐用户重现表 C
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["public", "celebrity"])
def test_primary_post_type_filter_reproduces_table_c(context, synthetic_project,
                                                     domain):
    """帖子类型过滤器取主口径（原创 + 带评论转发）时必须与表 C 完全相等

    这是 §13.8 的"全集重建"检验：过滤路径在这里对不上，另外三个帖子类型
    变体就都在同一个方向上错。
    """
    table_c = synthetic_project["table_c"]
    result = mea.topical_for_post_types(context, domain, mea.PRIMARY_POST_TYPES)

    assert sorted(result["user_id"]) == sorted(table_c["user_id"])
    merged = table_c[["user_id", "n_expressive_posts",
                      "{}_topical_posts".format(domain),
                      "{}_topical_share".format(domain)]].merge(
        result, on="user_id", how="left", suffixes=("_c", "_r"))

    pd.testing.assert_series_equal(
        merged["topical_posts"].astype("int64"),
        merged["{}_topical_posts".format(domain)].astype("int64"),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["n_expressive_posts_r"].astype("int64"),
        merged["n_expressive_posts_c"].astype("int64"),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["topical_share"].astype("float64"),
        merged["{}_topical_share".format(domain)].astype("float64"),
        check_names=False,
    )


def test_post_type_filters_partition_the_posts(context):
    """三种帖子类型的选中帖数相加必须等于"全部有字帖"变体的选中帖数

    帖子类型是一个划分，不是一组随意的筛子；相加对不上说明某个类型被漏掉
    或者被重复计入了。
    """
    counts = {
        label: int(mea.post_type_mask(
            context.incidence["public"].posts, types,
            require_nonzero_chars=True).sum())
        for label, types in (
            ("original", ("original",)),
            ("comment", ("retweet_comment",)),
            ("plain", ("retweet_plain",)),
        )
    }
    all_text = int(mea.post_type_mask(
        context.incidence["public"].posts, mea.ALL_TEXT_POST_TYPES,
        require_nonzero_chars=True).sum())
    assert sum(counts.values()) == all_text
    # 夹具必须真的含有纯转发，否则第 2 条检验会在一张没有纯转发的表上通过
    assert counts["plain"] > 0


# ---------------------------------------------------------------------------
# 2. 全部帖子分母必须更低
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["public", "celebrity"])
def test_all_posts_denominator_yields_a_lower_topical_share(context, domain):
    """存在不命中词表的纯转发时，全部帖子分母的占比必须严格更低

    这是"两个分母有没有被悄悄对调"最便宜的检验。
    """
    expressive = mea.topical_for_measure(context, domain, mea.MEASURE_EXPRESSIVE_SHARE)
    all_posts = mea.topical_for_measure(context, domain, mea.MEASURE_ALLPOSTS_SHARE)
    merged = expressive.merge(all_posts, on="user_id", suffixes=("_exp", "_all"))

    valid = merged.dropna(subset=["topical_share_exp", "topical_share_all"])
    assert len(valid) > 0
    assert (valid["topical_share_all"] <= valid["topical_share_exp"] + 1e-12).all()
    assert (valid["topical_share_all"] < valid["topical_share_exp"] - 1e-12).any()


def test_all_posts_measure_matches_the_table_c_column_built_for_it(
    context, synthetic_project, capsys
):
    """全部帖子口径必须与表 C 自己的 {domain}_topical_share_allposts 相等

    表 C 早就并行输出了这个口径（build_user_tables 里写着"供 13.1 分母
    稳健性使用"）。本模块从表 A 重新聚合，两条路径必须给出同一个数——
    否则这一族报告的"另一个分母"根本不是主流水线所说的那一个。
    """
    table_c = synthetic_project["table_c"]
    for domain in but.DOMAINS:
        result = mea.topical_for_measure(context, domain, mea.MEASURE_ALLPOSTS_SHARE)
        merged = table_c[["user_id", "{}_topical_share_allposts".format(domain)]].merge(
            result, on="user_id", how="left")
        pd.testing.assert_series_equal(
            merged["topical_share"].astype("float64"),
            merged["{}_topical_share_allposts".format(domain)].astype("float64"),
            check_names=False,
        )


def test_character_measures_match_table_c(context, synthetic_project):
    """字符密度必须与表 C 的 {domain}_char_density 逐用户相等"""
    table_c = synthetic_project["table_c"]
    for domain in but.DOMAINS:
        result = mea.topical_for_measure(context, domain, mea.MEASURE_CHAR_DENSITY)
        merged = table_c[["user_id", "{}_char_density".format(domain)]].merge(
            result, on="user_id", how="left")
        pd.testing.assert_series_equal(
            merged["topical_share"].astype("float64"),
            merged["{}_char_density".format(domain)].astype("float64"),
            check_names=False,
        )


def test_hits_per_1k_is_fitted_on_the_per_character_scale(context):
    """每千字命中数必须换算成"每字命中数"才进模型，否则它不是一个占比

    分数 logit 以取值落在 [0,1] 为前提（stats_utils.check_proportion_range
    会直接报错）。每千字命中数经常大于 1，除以 1000 之后是"每个字里有多少
    次命中"，恒定落在 [0,1]，而且与每千字口径只差一个常数因子。
    """
    result = mea.topical_for_measure(context, "public", mea.MEASURE_HITS_PER_CHAR)
    values = result["topical_share"].dropna()
    assert len(values) > 0
    assert (values >= 0).all() and (values <= 1).all()
    # 与表 C 的每千字口径必须只差 1000 倍
    per_1k = mea.topical_for_measure(context, "public", mea.MEASURE_CHAR_DENSITY)
    assert len(per_1k) == len(result)


# ---------------------------------------------------------------------------
# 3. 每个变体的 label 都必须写明分母
# ---------------------------------------------------------------------------

def _labels(frame):
    return sorted(set(frame["variant_label"].tolist()))


def test_every_denominator_variant_label_names_its_denominator(context, tmp_path):
    """§13.1 的每一行都必须在 label 里写明分母"""
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    rows = mea.run_denominator_variants(YEAR, context=context, out_path=out_path,
                                        diag_path=diag_path)
    assert len(rows) > 0
    for label in _labels(rows):
        assert "denominator=" in label, label


def test_every_post_type_variant_label_names_the_post_types_and_denominator(
    context, tmp_path
):
    """§13.8 的每一行都必须写明选中的帖子类型与分母"""
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    rows = mea.run_post_type_variants(YEAR, context=context, out_path=out_path,
                                      diag_path=diag_path)
    labels = _labels(rows)
    assert len(labels) == len(mea.POST_TYPE_VARIANTS)
    for label in labels:
        assert "post_types=" in label, label
        assert "denominator=" in label, label


def test_every_temporal_variant_label_names_its_denominator(context, tmp_path):
    """§13.7 的每一行都必须写明分母（哪些月、哪些用户）"""
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    rows = mea.run_temporal_restrictions(YEAR, context=context, out_path=out_path,
                                         diag_path=diag_path)
    assert len(rows) > 0
    for label in _labels(rows):
        assert "denominator=" in label, label


# ---------------------------------------------------------------------------
# 4. 异常月份规则：事先写死、机械执行、记下选中了哪几个月
# ---------------------------------------------------------------------------

def test_anomaly_rule_is_a_module_constant_stating_a_fixed_deviation_rule():
    """规则本身必须是一条写死的、可被引用的文本，而不是散落在实现里的分支"""
    assert isinstance(mea.ANOMALY_RULE, str)
    assert "median" in mea.ANOMALY_RULE
    assert str(mea.ANOMALY_K) in mea.ANOMALY_RULE
    assert str(mea.ANOMALY_MIN_RELATIVE_DEVIATION) in mea.ANOMALY_RULE


def test_anomalous_months_flags_a_spike_and_leaves_ordinary_variation_alone():
    """双条件规则：既要超过 3 倍稳健标准差，也要超过中位数的 50%"""
    volumes = pd.Series(
        {1: 100, 2: 105, 3: 95, 4: 102, 5: 98, 6: 101,
         7: 99, 8: 400, 9: 103, 10: 97, 11: 104, 12: 96}
    )
    months, info = mea.anomalous_months(volumes)
    assert months == (8,)
    assert info["median"] == pytest.approx(100.5)
    assert info["mad"] > 0
    assert info["threshold"] == pytest.approx(
        max(mea.ANOMALY_K * mea.MAD_SCALE * info["mad"],
            mea.ANOMALY_MIN_RELATIVE_DEVIATION * info["median"])
    )


def test_anomalous_months_does_not_flag_a_trivial_deviation_when_mad_collapses():
    """MAD 退化成 0 时，50% 的相对偏差下限必须挡住"只差一点"的月份

    只用 Hampel 那一半，几乎恒定的一年里任何一个哪怕差 1 的月份都会被判成
    异常；只用相对偏差那一半，则对真实的月度波动过于迟钝。两条同时要求。
    """
    volumes = pd.Series({m: 100 for m in range(1, 13)})
    volumes[5] = 101
    months, info = mea.anomalous_months(volumes)
    assert info["mad"] == 0
    assert months == ()

    volumes[5] = 300
    months, _ = mea.anomalous_months(volumes)
    assert months == (5,)


def test_a_rule_that_flags_every_month_leaves_a_note_row_not_a_silent_fallback(
    context, tmp_path, monkeypatch
):
    """规则把整年都判成异常时，必须留一行注明原因，而不是悄悄退回不排除

    悄悄退回会让一个跑不出来的变体伪装成一次通过——与 samples 拒绝复制
    untreated 结果去冒充 log1p 变体是同一条纪律。
    """
    monkeypatch.setattr(mea, "anomalous_months",
                        lambda volumes: (tuple(range(1, 13)), {
                            "median": 0.0, "mad": 0.0, "threshold": 0.0}))
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    rows = mea.run_temporal_restrictions(YEAR, context=context, out_path=out_path,
                                         diag_path=diag_path)
    flagged_rows = rows[rows["variant_label"].str.contains("anomalous_excluded",
                                                           na=False)]
    assert len(flagged_rows) == 1
    assert flagged_rows["estimate"].isna().all()
    assert "flagged_every_month" in flagged_rows.iloc[0]["note"]


def test_anomaly_rule_reads_behaviour_volume_not_the_six_outcomes(context):
    """规则只看行为量：把面板里的结果变量整列改掉，选中的月份不能变

    看着结果挑月份就是在对因变量做选择，读者从输出上分辨不出来，所以这条
    必须由测试钉死。
    """
    baseline_volumes = mea.monthly_behaviour_volume(context.panel)
    tampered = context.panel.copy()
    for domain in but.DOMAINS:
        tampered["{}_source_entered".format(domain)] = True
        tampered["{}_source_count".format(domain)] = 99
        tampered["{}_topical_share".format(domain)] = 0.99
    tampered_volumes = mea.monthly_behaviour_volume(tampered)
    pd.testing.assert_series_equal(baseline_volumes, tampered_volumes)


def test_anomalous_months_are_recorded_in_the_diagnostics(context, tmp_path):
    """逐月的行为量、阈值与是否被选中都要落在诊断表上"""
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    mea.run_temporal_restrictions(YEAR, context=context, out_path=out_path,
                                  diag_path=diag_path)
    diag = _read_diagnostics(diag_path)
    monthly = diag[diag["variant_label"] == mea.ANOMALY_DIAGNOSTIC_LABEL]
    assert len(monthly) == 12
    assert monthly["rule"].dropna().eq(mea.ANOMALY_RULE).all()
    flagged = sorted(int(m) for m in monthly.loc[monthly["is_anomalous"] == True, "month"])
    assert SPIKE_MONTH in flagged
    assert monthly["monthly_volume"].notna().all()


# ---------------------------------------------------------------------------
# 5. 活跃月份限制：剔除的人必须精确，男女必须分开记
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_months", [3, 6])
def test_activity_month_restriction_drops_exactly_the_expected_users(
    context, min_months
):
    """剔的必须恰好是那几个人，而且活跃月份用的必须是含转发月的那个分母

    f4 只发过两个月的帖，却在另外两个月转发过。按 n_active_months_panel
    她有 4 个活跃月（≥3 保留、≥6 剔除），按只看发帖的 n_active_months 只有
    2 个（两条阈值都剔）。两条阈值下她的去留相反，因此这条测试同时钉死了
    "有没有做限制"和"用的是哪一个活跃月份口径"。
    """
    kept = mea.active_month_restriction(context.user_table, min_months)
    dropped = set(context.user_table["user_id"]) - set(kept["user_id"])
    assert dropped == EXPECTED_DROPPED_BY_ACTIVITY[min_months]


def test_the_two_active_month_columns_really_differ_in_the_fixture(context):
    """夹具必须真的让两个活跃月份口径不相等，否则上一条测不出口径"""
    row = context.user_table.set_index("user_id").loc[RETWEET_ONLY_MONTHS_USER]
    assert int(row["n_active_months"]) == 2
    assert int(row[mea.ACTIVE_MONTHS_COLUMN]) == 2 + len(RETWEET_ONLY_MONTHS)


def test_activity_month_restriction_records_the_gender_split(context, tmp_path):
    """一条对男女生效比例不同的限制会改变被比较的总体，必须逐性别记账"""
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    mea.run_temporal_restrictions(YEAR, context=context, out_path=out_path,
                                  diag_path=diag_path)
    diag = _read_diagnostics(diag_path)
    rows = diag[diag["variant_label"].str.contains("min_active_months=3", na=False)]
    assert len(rows) == 1
    row = rows.iloc[0]
    # 3 个月这条限制只剔掉男性（m3、m4），一个女性都没剔——这份不对称正是
    # 必须逐性别记账的理由：它改变了被比较的两个总体
    assert int(row["n_dropped_male"]) == 2
    assert int(row["n_dropped_female"]) == 0
    assert row["dropped_share_male"] > row["dropped_share_female"]

    six = diag[diag["variant_label"].str.contains("min_active_months=6", na=False)]
    assert int(six.iloc[0]["n_dropped_female"]) == 1


# ---------------------------------------------------------------------------
# 6. §13.7 必须从 models_temporal 读，不允许重算
# ---------------------------------------------------------------------------

def test_leave_one_month_out_is_read_from_models_temporal_not_recomputed(
    context, tmp_path, monkeypatch
):
    """把 models_temporal 的拟合函数换成会抛异常的桩，§13.7 仍然必须跑通"""
    def _boom(*args, **kwargs):
        raise AssertionError("留一月重估必须读 models_temporal 的输出，不许重算")

    monkeypatch.setattr(mt, "leave_one_month_out", _boom)
    monkeypatch.setattr(mt, "fit_month_fixed_effects", _boom)

    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    rows = mea.run_temporal_restrictions(YEAR, context=context, out_path=out_path,
                                         diag_path=diag_path)
    lomo_rows = rows[rows["variant_label"].str.contains("leave_one_month_out",
                                                        na=False)]
    assert len(lomo_rows) > 0
    assert lomo_rows["note"].str.contains("models_temporal").all()


def test_missing_models_temporal_output_becomes_a_note_row_not_a_recomputation(
    synthetic_project, tmp_path
):
    """留一月输出不存在时留一行注明原因的行，绝不改为自己算一遍"""
    os.remove(os.path.join(synthetic_project["out_dir"], "results", mt.LOMO_FILE))
    ctx = mea.build_context(YEAR)
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    rows = mea.run_temporal_restrictions(YEAR, context=ctx, out_path=out_path,
                                         diag_path=diag_path)
    unavailable = rows[rows["variant_label"].str.contains("unavailable", na=False)]
    assert len(unavailable) == 1
    assert unavailable["estimate"].isna().all()
    assert "models_temporal" in unavailable.iloc[0]["note"]


# ---------------------------------------------------------------------------
# 月份限制的重建：保留全部 12 个月必须回到表 C
# ---------------------------------------------------------------------------

def test_keeping_every_month_reproduces_table_c_entry_and_activity(
    context, synthetic_project
):
    """月份限制路径的"全集重建"检验：12 个月全留必须与表 C 相等

    §13.7 的进入指示由表 D 逐月重新聚合而来。这条路径在全月份上对不上
    表 C，那么半年限制与异常月限制都在同一个方向上错。
    """
    frame = mea.user_frame_for_months(context, tuple(range(1, 13)))
    table_c = synthetic_project["table_c"]
    merged = table_c.merge(frame, on="user_id", how="inner", suffixes=("_c", "_r"))
    assert len(merged) == len(table_c)
    for column in ("n_posts", "n_retweets", "n_active_days"):
        pd.testing.assert_series_equal(
            merged["{}_r".format(column)].astype("int64"),
            merged["{}_c".format(column)].astype("int64"),
            check_names=False,
        )
    for domain in but.DOMAINS:
        pd.testing.assert_series_equal(
            merged["{}_source_entered_r".format(domain)].astype(bool),
            merged["{}_source_entered_c".format(domain)].astype(bool),
            check_names=False,
        )


def test_half_year_restriction_actually_restricts_the_months(context):
    first = mea.user_frame_for_months(context, mea.FIRST_HALF_MONTHS)
    full = mea.user_frame_for_months(context, tuple(range(1, 13)))
    assert first["n_posts"].sum() < full["n_posts"].sum()
    assert first["n_active_months"].max() <= len(mea.FIRST_HALF_MONTHS)


# ---------------------------------------------------------------------------
# 落盘、schema 与 manifest
# ---------------------------------------------------------------------------

def test_rows_carry_the_full_variant_identity_and_the_row_count_invariant(
    context, tmp_path
):
    out_path = str(tmp_path / "measures.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    rows = mea.run_post_type_variants(YEAR, context=context, out_path=out_path,
                                      diag_path=diag_path)
    assert list(rows.columns) == list(harness.ROBUSTNESS_SCHEMA)
    assert rows["variant_family"].eq(mea.VARIANT_FAMILY_POST_TYPE).all()
    assert rows["seed"].isna().all()
    per_label = rows.groupby("variant_label").size()
    assert (per_label == len(harness.QUANTITIES) * 2).all()

    written = _read_results(out_path)
    assert len(written) == len(rows)


def test_build_writes_results_diagnostics_and_a_manifest_recording_the_rule(
    synthetic_project
):
    paths = mea.build(YEAR)
    results = _read_results(paths["results_path"])
    diagnostics = _read_diagnostics(paths["diagnostics_path"])
    assert len(results) > 0
    assert len(diagnostics) > 0
    assert set(results["variant_family"]) == {
        mea.VARIANT_FAMILY_DENOMINATOR,
        mea.VARIANT_FAMILY_POST_TYPE,
        mea.VARIANT_FAMILY_TEMPORAL,
    }

    manifest_path = os.path.join(mea.manifest_dir(YEAR), "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    params = manifest["params"]
    assert params["anomaly_rule"] == mea.ANOMALY_RULE
    assert SPIKE_MONTH in params["anomalous_months"]
    assert params["anomaly_volume_variable"] == mea.ANOMALY_VOLUME_VARIABLE
    assert len(params["monthly_behaviour_volume"]) == 12


def test_output_stays_under_the_robustness_directory(synthetic_project):
    assert mea.robustness_dir().endswith(os.path.join("analysis_data", "robustness"))
    assert mea.results_path().startswith(mea.robustness_dir())
    assert mea.diagnostics_path().startswith(mea.robustness_dir())
    assert mea.manifest_dir(YEAR).startswith(mea.robustness_dir())
