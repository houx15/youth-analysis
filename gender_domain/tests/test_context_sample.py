"""
gender_domain.robustness.context_sample 的单元测试（§13.4 语境效度抽样）。

分两块，对应模块的两个接口：

1. `draw`：夹具刻意构造出几个"密度完全相同、数量已知"的分层格（例如
   6 条内容完全一样的原创帖），这样才能不靠猜测地断言"这一格恰好有 6 条
   可抽、n_per_cell=2 时恰好抽 2 条"，而不是走运抽中了预期的数量。
   同一个 seed 必须逐字节可复现——用真实的 build_post_table.process_frame
   + build_user_tables.build 生成表 A/表 C，第二个"编码员"能不能拿到同一
   份样本，测的就是这条链路，不是靠夹具本身可复现（那是 pandas 的事）。

2. `summarise`：不需要 draw() 的输出，直接手写一份"已编码"的表——这正是
   brief 说的"hand-coded fixture"：已知的假阳性数、已知的 m/f 分布，
   算出来的比例、Wilson 区间、性别差异必须与直接调用 stats_utils 的结果
   完全一致（不手写 Wilson 公式的期望值，避免测试自己重新实现一遍公式）。
   另外单独测"只编了一部分"的情况，覆盖率必须被如实报告，不能因为不完整
   就报错。
"""

import numpy as np
import pandas as pd
import pytest

from gender_domain import build_post_table as bpt
from gender_domain import build_retweet_table as brt
from gender_domain import build_user_tables as but
from gender_domain import config
from gender_domain import stats_utils as su
from gender_domain import text_rules as tr
from gender_domain.robustness import context_sample as cs

YEAR = 2020

# "转发" 本身就是纯转发占位文本集合里的一员（text_rules.PLAIN_RETWEET_
# PLACEHOLDERS），放进公共事务词表是为了造出一条"命中但是纯转发"的帖子，
# 覆盖 post_type=retweet_plain 这一分层维度。
PUBLIC_VOCAB = ["复工", "转发"]
CELEBRITY_VOCAB = ["顶流"]

FILLER = "填"


def _content(term, filler_len=0):
    return term + FILLER * filler_len


def _raw_posts():
    """构造密度完全已知的合成正文，覆盖 domain × gender × post_type 的组合

    每一组的"可抽样总数"都是数出来的，不是估计的：
    - dense（男/原创/公共事务）：6 条内容完全相同的 "复工"（密度 1.0），
      用来验证 n_per_cell 生效时确实"抽满就停，不是有多少抽多少"；
    - sparse（女/原创/公共事务）：1 条低密度帖，验证"不够 n_per_cell 时
      整格取完"；
    - midset（男/转发评论/公共事务）：3 条中等密度帖，同一格里再验证一次
      抽样上限，确认不是只有第一个格子生效；
    - plain（女/纯转发/公共事务）：1 条纯转发帖，命中词是"转发"本身，
      覆盖 retweet_plain 这个 post_type；
    - celebrity（男/原创/明星）：1 条明星词命中帖，验证两个领域互不串味。
    """
    rows = []

    # dense: 6 条同密度原创帖，男性
    for i in range(6):
        rows.append({
            "weibo_id": "dense_{:02d}".format(i),
            "user_id": "u_dense_{:02d}".format(i),
            "weibo_content": _content("复工"),
            "is_retweet": "0",
            "gender": "m",
            "province": "11",
            "time_stamp": 1580000000,
            "date": "2020-01-05",
        })

    # sparse: 1 条低密度原创帖，女性
    rows.append({
        "weibo_id": "sparse_00",
        "user_id": "u_sparse_00",
        "weibo_content": _content("复工", filler_len=20),
        "is_retweet": "0",
        "gender": "f",
        "province": "11",
        "time_stamp": 1580000000,
        "date": "2020-01-06",
    })

    # midset: 3 条中等密度转发评论帖，男性。//@ 之后的内容会被 clean_text
    # 剥离，剩下的正文就是 _content(...) 本身
    for i in range(3):
        rows.append({
            "weibo_id": "mid_{:02d}".format(i),
            "user_id": "u_mid_{:02d}".format(i),
            "weibo_content": _content("复工", filler_len=6) + "//@某人:随便说说",
            "is_retweet": "1",
            "gender": "m",
            "province": "11",
            "time_stamp": 1580000000,
            "date": "2020-02-01",
        })

    # plain: 1 条纯转发帖（清洗后正文恰好是占位文本"转发"），女性
    rows.append({
        "weibo_id": "plain_00",
        "user_id": "u_plain_00",
        "weibo_content": "转发//@某人:随便说说",
        "is_retweet": "1",
        "gender": "f",
        "province": "11",
        "time_stamp": 1580000000,
        "date": "2020-02-02",
    })

    # celebrity: 1 条明星域命中帖，男性
    rows.append({
        "weibo_id": "star_00",
        "user_id": "u_star_00",
        "weibo_content": _content("顶流"),
        "is_retweet": "0",
        "gender": "m",
        "province": "11",
        "time_stamp": 1580000000,
        "date": "2020-03-01",
    })

    return pd.DataFrame(rows)


def _raw_retweets(user_ids):
    """给每个用户配一条无关紧要的转发事件，只是为了让 but.build() 的表 B 非空"""
    rows = []
    for i, uid in enumerate(user_ids):
        rows.append({
            "user_id": uid,
            "weibo_id": "rw_{:03d}".format(i),
            "r_weibo_id": "s_{:03d}".format(i),
            "r_user_id": "acct_other",
            "is_retweet": "1",
            "gender": "m",
            "time_stamp": 1580000100,
            "r_time_stamp": 1580000000,
            "date": "2020-01-10",
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_project(tmp_path, monkeypatch):
    """写出互相自洽的原始日文件、表 A、表 B、表 C，并把 config 的路径指过去

    与 test_vocabulary_robustness.py 的同名夹具同一个思路：原始日文件与
    表 A 来自同一批正文，context_sample.draw() 读回来的正文因此确确实实
    就是生成表 A 时用的那一批。
    """
    out_dir = str(tmp_path / "analysis_data")
    raw_dir = str(tmp_path / "raw")
    import os

    os.makedirs(out_dir, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "DATA_DIR", raw_dir)
    monkeypatch.setattr(config, "load_public_vocabulary", lambda year=YEAR: list(PUBLIC_VOCAB))
    monkeypatch.setattr(
        config, "load_celebrity_vocabulary", lambda year=YEAR: list(CELEBRITY_VOCAB)
    )

    raw = _raw_posts()
    year_dir = os.path.join(raw_dir, str(YEAR))
    os.makedirs(year_dir, exist_ok=True)
    for date, part in raw.groupby("date"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(year_dir, "{}.parquet".format(date)),
            engine="pyarrow", index=False,
        )

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
        _raw_retweets(list(raw["user_id"].unique())),
        brt.build_account_lookup({"cat": {"acct_other"}}),
        brt.build_account_lookup({}),
    )
    event_dir = os.path.join(out_dir, "retweet_domain_events_{}".format(YEAR))
    os.makedirs(event_dir, exist_ok=True)
    for month, part in events.groupby("month"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(event_dir, "month={:02d}.parquet".format(month)),
            engine="pyarrow", index=False,
        )

    but.build(YEAR)
    return {"out_dir": out_dir, "raw_dir": raw_dir}


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------

def _cell_frame(result, gender, post_type):
    """从 draw() 结果里取出某个 (gender, post_type) 对应的行（跨 tercile 合并）

    不直接按 density_tercile 过滤：分位点是跨性别/帖子类型合并算出来的，
    测试只关心"这一批本来就密度相同的帖子有没有被分到同一格、有没有被
    正确限流"，不关心那一格最终被贴上 low/mid/high 里的哪个名字。
    """
    return result[(result["gender"] == gender) & (result["post_type"] == post_type)]


def test_draw_caps_a_dense_cell_at_n_per_cell(synthetic_project):
    """dense 格（男/原创/公共事务）有 6 条同密度帖，n_per_cell=2 时只能抽 2 条"""
    result = cs.draw(year=YEAR, n_per_cell=2, seed=0, domains=("public",))
    dense = _cell_frame(result, "m", "original")
    assert len(dense) == 2
    assert (dense["cell_n_available"] == 6).all()
    assert (dense["cell_n_drawn"] == 2).all()
    # 六条帖子密度完全相同，必须被分进同一个分层格（同一个 density_tercile）
    assert dense["density_tercile"].nunique() == 1


def test_draw_takes_everything_in_a_cell_smaller_than_n_per_cell(synthetic_project):
    """sparse 格（女/原创/公共事务）只有 1 条帖，n_per_cell=2 时应该整格取完"""
    result = cs.draw(year=YEAR, n_per_cell=2, seed=0, domains=("public",))
    sparse = _cell_frame(result, "f", "original")
    assert len(sparse) == 1
    assert sparse.iloc[0]["cell_n_available"] == 1
    assert sparse.iloc[0]["cell_n_drawn"] == 1


def test_draw_caps_a_second_cell_independently(synthetic_project):
    """midset 格（男/转发评论）有 3 条帖，n_per_cell=2 时也要限流，不只是第一格生效"""
    result = cs.draw(year=YEAR, n_per_cell=2, seed=0, domains=("public",))
    midset = _cell_frame(result, "m", "retweet_comment")
    assert len(midset) == 2
    assert (midset["cell_n_available"] == 3).all()


def test_draw_covers_plain_retweet_post_type(synthetic_project):
    """纯转发帖（命中词恰好是占位文本本身）必须作为独立的 post_type 出现"""
    result = cs.draw(year=YEAR, n_per_cell=2, seed=0, domains=("public",))
    plain = _cell_frame(result, "f", "retweet_plain")
    assert len(plain) == 1
    assert plain.iloc[0]["weibo_id"] == "plain_00"
    assert plain.iloc[0]["matched_terms"] == "转发×1"


def test_draw_keeps_domains_separate(synthetic_project):
    """明星域的命中帖不能混进公共事务域，反之亦然"""
    result = cs.draw(year=YEAR, n_per_cell=25, seed=0, domains=("public", "celebrity"))
    celebrity = result[result["domain"] == "celebrity"]
    assert len(celebrity) == 1
    assert celebrity.iloc[0]["weibo_id"] == "star_00"
    assert celebrity.iloc[0]["matched_terms"] == "顶流×1"
    assert set(result["domain"]) == {"public", "celebrity"}


def test_draw_includes_cleaned_text_and_highlights_the_matched_term(synthetic_project):
    """抽样结果必须带清洗后正文，且命中词要在正文里被标出来"""
    result = cs.draw(year=YEAR, n_per_cell=25, seed=0, domains=("public",))
    dense = _cell_frame(result, "m", "original").iloc[0]
    assert dense["text"] == "复工"
    assert dense["text_highlighted"] == "【复工】"

    sparse = _cell_frame(result, "f", "original").iloc[0]
    assert sparse["text"] == "复工" + FILLER * 20
    assert sparse["text_highlighted"] == "【复工】" + FILLER * 20


def test_draw_coding_columns_are_present_and_empty(synthetic_project):
    result = cs.draw(year=YEAR, n_per_cell=25, seed=0, domains=("public", "celebrity"))
    for col in ("is_substantive", "is_false_positive", "notes"):
        assert col in result.columns
        assert result[col].isna().all()
    assert list(result.columns) == list(cs.DRAW_SCHEMA)


def test_draw_records_the_seed_on_every_row(synthetic_project):
    result = cs.draw(year=YEAR, n_per_cell=25, seed=777, domains=("public",))
    assert (result["seed"] == 777).all()


def test_draw_is_byte_identical_for_the_same_seed(synthetic_project):
    """同一个 seed 两次抽样必须完全一致——第二个编码员靠的就是这条"""
    first = cs.draw(year=YEAR, n_per_cell=2, seed=13, domains=("public", "celebrity"))
    second = cs.draw(year=YEAR, n_per_cell=2, seed=13, domains=("public", "celebrity"))
    pd.testing.assert_frame_equal(first, second)


def test_cell_seed_differs_across_cells_and_seeds():
    """分层格之间、不同主 seed 之间派生出的子种子必须互不相同

    直接测种子派生函数本身，避免用"不同 seed 抽样结果不同"这种依赖具体
    数据分布的说法去断言——那种断言在候选数很少时可能碰巧相等，是脆弱的。
    """
    s1 = cs._cell_seed(0, "public", "m", "original", "low")
    s2 = cs._cell_seed(0, "public", "f", "original", "low")
    s3 = cs._cell_seed(1, "public", "m", "original", "low")
    assert len({s1, s2, s3}) == 3


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

def _coded_row(domain, gender, post_type, tercile, is_fp):
    return {
        "domain": domain, "gender": gender, "post_type": post_type,
        "density_tercile": tercile, "is_false_positive": is_fp,
    }


def test_summarise_reproduces_a_known_false_positive_rate_and_wilson_interval():
    rows = (
        [_coded_row("public", "m", "original", "mid", v) for v in [True, False, False, False]]
        + [_coded_row("public", "f", "original", "mid", v) for v in [True, True, False, False]]
    )
    coded_df = pd.DataFrame(rows)
    summary = cs.summarise(coded_df)

    overall = summary[summary["level"] == "overall"].iloc[0]
    assert overall["n_drawn"] == 8
    assert overall["n_coded"] == 8
    assert overall["coverage"] == pytest.approx(1.0)
    assert overall["n_false_positive"] == 3
    assert overall["false_positive_rate"] == pytest.approx(3 / 8)
    expected_low, expected_high = su.proportion_ci(3, 8)
    assert overall["ci_low"] == pytest.approx(expected_low)
    assert overall["ci_high"] == pytest.approx(expected_high)

    male_cell = summary[
        (summary["level"] == "cell") & (summary["gender"] == "m")
    ].iloc[0]
    assert male_cell["n_coded"] == 4
    assert male_cell["n_false_positive"] == 1
    assert male_cell["false_positive_rate"] == pytest.approx(0.25)
    low, high = su.proportion_ci(1, 4)
    assert male_cell["ci_low"] == pytest.approx(low)
    assert male_cell["ci_high"] == pytest.approx(high)

    female_cell = summary[
        (summary["level"] == "cell") & (summary["gender"] == "f")
    ].iloc[0]
    assert female_cell["n_false_positive"] == 2
    assert female_cell["false_positive_rate"] == pytest.approx(0.5)

    diff_both = summary[
        (summary["level"] == "gender_diff") & (summary["domain"] == "both")
    ].iloc[0]
    assert diff_both["rate_male"] == pytest.approx(0.25)
    assert diff_both["rate_female"] == pytest.approx(0.5)
    expected_diff, expected_dlow, expected_dhigh = su.proportion_diff_ci(2, 4, 1, 4)
    assert diff_both["diff"] == pytest.approx(expected_diff)
    assert diff_both["diff_ci_low"] == pytest.approx(expected_dlow)
    assert diff_both["diff_ci_high"] == pytest.approx(expected_dhigh)

    # domain 只有一个取值时，"整体"与"该 domain"的性别差异行应当一致
    diff_domain = summary[
        (summary["level"] == "gender_diff") & (summary["domain"] == "public")
    ].iloc[0]
    assert diff_domain["diff"] == pytest.approx(expected_diff)


def test_summarise_reports_coverage_on_a_partially_coded_file():
    rows = (
        [_coded_row("celebrity", "m", "original", "low", False)]
        + [_coded_row("celebrity", "m", "original", "low", None) for _ in range(2)]
        + [_coded_row("celebrity", "f", "original", "low", None) for _ in range(2)]
    )
    coded_df = pd.DataFrame(rows)
    summary = cs.summarise(coded_df)

    overall = summary[summary["level"] == "overall"].iloc[0]
    assert overall["n_drawn"] == 5
    assert overall["n_coded"] == 1
    assert overall["coverage"] == pytest.approx(1 / 5)
    assert overall["false_positive_rate"] == pytest.approx(0.0)

    male_cell = summary[
        (summary["level"] == "cell") & (summary["gender"] == "m")
    ].iloc[0]
    assert male_cell["n_drawn"] == 3
    assert male_cell["n_coded"] == 1
    assert male_cell["coverage"] == pytest.approx(1 / 3)

    female_cell = summary[
        (summary["level"] == "cell") & (summary["gender"] == "f")
    ].iloc[0]
    assert female_cell["n_drawn"] == 2
    assert female_cell["n_coded"] == 0
    assert female_cell["coverage"] == pytest.approx(0.0)
    assert np.isnan(female_cell["false_positive_rate"])

    # 女性一侧完全没有编码，性别差异检验算不出来，必须留一行说明原因
    diff_row = summary[
        (summary["level"] == "gender_diff") & (summary["domain"] == "celebrity")
    ].iloc[0]
    assert np.isnan(diff_row["diff"])
    assert diff_row["note"] is not None and "无法计算" in diff_row["note"]


def test_summarise_rejects_a_frame_missing_required_columns():
    bad = pd.DataFrame({"domain": ["public"], "gender": ["m"]})
    with pytest.raises(ValueError):
        cs.summarise(bad)


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False),
    ("1", True), ("0", False),
    ("true", True), ("False", False),
    ("yes", True), ("no", False),
    ("是", True), ("否", False),
    (None, None), (np.nan, None), ("", None), ("  ", None),
])
def test_parse_code_handles_common_spreadsheet_spellings(value, expected):
    assert cs._parse_code(value) is expected


def test_parse_code_rejects_unrecognized_values():
    with pytest.raises(ValueError):
        cs._parse_code("maybe")
