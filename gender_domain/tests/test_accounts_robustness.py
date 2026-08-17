"""
gender_domain.robustness.accounts 的单元测试（§13.5 来源账号稳健性）。

这份测试的重心按重要性排列：

1. **"某个账号的转发全没了，那个用户就必须不再算参与者"必须被真的测到。**
   §13.5 的所有变体都靠这一条成立：如果剔除账号之后聚合仍然把这些用户
   算作"进入过该领域"，那么每一个留一/剔除变体都会得到"结论完全不变"
   的假象。夹具因此刻意安排了两个**只转过一个来源账号**的用户
   （u046 只转 pa3、u047 只转 ca4），测试钉死他们在对应变体下的
   entry 指示由 True 翻成 False，而其他用户不受影响。

2. **集中度与被剔除的量必须报出来，而不是只报估计值变了多少。**
   "剔掉头号账号结论还在"这句话，在读者知道那个账号占该领域转发量
   2% 还是 83% 之前没有意义。夹具里 ca1 独占明星领域 80/96 的转发量，
   正是 §13.5 担心的"一个粉丝群撑起整个测量"的形状；测试用夹具自己的
   事件规格独立算出期望份额，而不是照抄模块算出来的数。

3. **bootstrap 抽的是账号，不是事件、也不是用户。**
   参照 test_stats_utils 里的簇 bootstrap 哨兵：夹具里 u045 只转
   prov_release_weibo 这一层的两个账号（pa7 3 次、pa8 5 次），该层恰好
   2 个账号。按账号有放回重抽 2 个，u045 的公共来源转发数只可能是
   3a+5b（a+b=2），即 {6, 8, 10}；一旦实现改成重抽事件，这个数会稳定
   在 8 附近并出现 7、9 之类的值，测试立刻失败。
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
from gender_domain import stats_utils as su
from gender_domain import text_rules as tr
from gender_domain.robustness import accounts as acc
from gender_domain.robustness import harness
from gender_domain.robustness import incidence as inc


YEAR = 2020
N_USERS = 48
POSTS_PER_USER = 6

PUBLIC_VOCAB = ["疫情", "复工", "口罩", "两会"]
CELEBRITY_VOCAB = ["顶流", "粉丝", "综艺", "打榜"]

POST_TEMPLATES = [
    "今天疫情形势不错",
    "复工的第一天",
    "出门记得戴口罩",
    "两会开幕了",
    "粉丝又在打榜",
    "顶流的综艺真好看",
]

# 账号规格：{账号: {"domains": {领域: [类别]}, "users": [(用户序号, 事件数)]}}
# 三件事是**故意**放进来的，缺了任何一件都有一类错误测不到：
#   1) 跨类别账号（pa6、ca2）：真实的 news_user_ids.json /
#      entertain_user_ids.json 都不是划分，一个账号可以同时属于
#      central_news 与 org_news，一个艺人的 verified_reason 里可以同时有
#      "演员"和"歌手"。留一类别按成员资格剔除，bootstrap 的分层则必须用
#      表 B 里那个拼接后的类别串（它才是一个真正的划分）。
#   2) **同时在两份名单上的账号（xa9）**：表 B 的 manifest 专门记了
#      overlapping_accounts，说明真实数据里就有。剔除必须按 (账号, 领域)
#      键——按账号 ID 全局剔除会让"留一公共事务账号"顺手削掉明星领域的
#      转发量，variant_label 说的就不是真正跑的那件事了。
#   3) 只转过一个来源账号的用户（u046 -> pa3、u047 -> ca4）：进入指示
#      必须由 True 翻成 False。
ACCOUNT_SPEC = {
    "pa1": {"domains": {"public": ["central_news"]},
            "users": [(i, 2) for i in range(24)]},
    "pa2": {"domains": {"public": ["central_news"]},
            "users": [(i, 1) for i in range(8)]},
    # 只有 u046 转过它——留一账号必须让 u046 变成非参与者
    "pa3": {"domains": {"public": ["local_news_units"]}, "users": [(46, 1)]},
    "pa4": {"domains": {"public": ["local_news_units"]},
            "users": [(i, 1) for i in range(24, 32)]},
    "pa5": {"domains": {"public": ["gov_release_weibo"]},
            "users": [(i, 1) for i in range(32, 36)]},
    "pa6": {"domains": {"public": ["central_news", "org_news"]},
            "users": [(i, 1) for i in range(36, 40)]},
    # prov_release_weibo 这一层恰好两个账号，且只有 u045 转过它们，
    # 3 次与 5 次——账号 bootstrap 哨兵就靠这个构造
    "pa7": {"domains": {"public": ["prov_release_weibo"]}, "users": [(45, 3)]},
    "pa8": {"domains": {"public": ["prov_release_weibo"]}, "users": [(45, 5)]},
    # 明星领域高度集中：ca1 一家占八成以上
    "ca1": {"domains": {"celebrity": ["演员"]},
            "users": [(i, 2) for i in range(40)]},
    "ca2": {"domains": {"celebrity": ["演员", "歌手"]},
            "users": [(i, 1) for i in range(10)]},
    "ca3": {"domains": {"celebrity": ["歌手"]},
            "users": [(i, 1) for i in range(40, 45)]},
    # 只有 u047 转过它
    "ca4": {"domains": {"celebrity": ["导演"]}, "users": [(47, 1)]},
    # 同时在两份名单上：一条转发同时命中两个领域，表 B 会各出一行
    "xa9": {"domains": {"public": ["gov_release_weibo"], "celebrity": ["演员"]},
            "users": [(10, 1), (11, 1)]},
}

OVERLAP_ACCOUNT = "xa9"

SOLO_PUBLIC_USER = "u046"
SOLO_PUBLIC_ACCOUNT = "pa3"
SOLO_CELEBRITY_USER = "u047"
SOLO_CELEBRITY_ACCOUNT = "ca4"


def _user_id(index):
    return "u{:03d}".format(index)


def _gender(index):
    return "m" if index % 2 == 0 else "f"


def _accounts_by_domain(domain):
    return sorted(
        a for a, spec in ACCOUNT_SPEC.items() if domain in spec["domains"]
    )


def _category_map(domain):
    """{类别: {账号}}，与 configs/*.json 的形状一致（可跨类别）"""
    out = {}
    for account, spec in ACCOUNT_SPEC.items():
        for category in spec["domains"].get(domain, []):
            out.setdefault(category, set()).add(account)
    return out


def _expected_volume(domain, genders=None):
    """夹具自己算出来的 {账号: 事件数}，不经过被测模块

    同时在两份名单上的账号，同一条转发在两个领域各算一次——与表 B
    "一条转发命中两个领域就出两行"的口径一致。
    """
    out = {}
    for account, spec in ACCOUNT_SPEC.items():
        if domain not in spec["domains"]:
            continue
        total = 0
        for index, n_events in spec["users"]:
            if genders is None or _gender(index) in genders:
                total += n_events
        out[account] = total
    return out


def _raw_posts():
    rows = []
    for i in range(N_USERS):
        uid = _user_id(i)
        gender = _gender(i)
        for j in range(POSTS_PER_USER):
            month = 1 if (i + j) % 2 == 0 else 2
            day = 1 + ((i * POSTS_PER_USER + j) % 20)
            offset = 0 if gender == "m" else 3
            rows.append({
                "weibo_id": "p{:03d}_{:d}".format(i, j),
                "user_id": uid,
                "weibo_content": POST_TEMPLATES[(i + j + offset) % len(POST_TEMPLATES)],
                "is_retweet": "1" if j == POSTS_PER_USER - 1 else "0",
                "gender": gender,
                "province": "11",
                "time_stamp": 1580000000,
                "date": "2020-{:02d}-{:02d}".format(month, day),
            })
    return pd.DataFrame(rows)


def _raw_retweets():
    """按 ACCOUNT_SPEC 展开成原始转发记录"""
    rows = []
    counter = 0
    for account in sorted(ACCOUNT_SPEC):
        for index, n_events in ACCOUNT_SPEC[account]["users"]:
            for _ in range(n_events):
                counter += 1
                rows.append({
                    "user_id": _user_id(index),
                    "weibo_id": "rw{:05d}".format(counter),
                    "r_weibo_id": "s{:05d}".format(counter),
                    "r_user_id": account,
                    "is_retweet": "1",
                    "gender": _gender(index),
                    "time_stamp": 1580000100,
                    "r_time_stamp": 1580000000,
                    "date": "2020-{:02d}-{:02d}".format(
                        1 if counter % 2 == 0 else 2, 1 + counter % 20
                    ),
                })
    return pd.DataFrame(rows)


def _write_account_files(config_dir):
    os.makedirs(config_dir, exist_ok=True)
    public_path = os.path.join(config_dir, "news_user_ids.json")
    celebrity_path = os.path.join(config_dir, "entertain_user_ids.json")
    for path, domain in ((public_path, "public"), (celebrity_path, "celebrity")):
        payload = {
            category: sorted(ids) for category, ids in _category_map(domain).items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    return public_path, celebrity_path


@pytest.fixture()
def synthetic_project(tmp_path, monkeypatch):
    """写出互相自洽的表 A、表 B、表 C 与两份账号名单，并把 config 路径指过去"""
    out_dir = str(tmp_path / "analysis_data")
    config_dir = str(tmp_path / "configs")
    os.makedirs(out_dir, exist_ok=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    public_path, celebrity_path = _write_account_files(config_dir)
    monkeypatch.setattr(config, "PUBLIC_ACCOUNTS_FILE", public_path)
    monkeypatch.setattr(config, "CELEBRITY_ACCOUNTS_FILE", celebrity_path)
    monkeypatch.setattr(config, "load_public_vocabulary", lambda year=YEAR: list(PUBLIC_VOCAB))
    monkeypatch.setattr(
        config, "load_celebrity_vocabulary", lambda year=YEAR: list(CELEBRITY_VOCAB)
    )

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
        brt.build_account_lookup(_category_map("public")),
        brt.build_account_lookup(_category_map("celebrity")),
    )
    event_dir = os.path.join(out_dir, "retweet_domain_events_{}".format(YEAR))
    os.makedirs(event_dir, exist_ok=True)
    for month, part in events.groupby("month"):
        part.reset_index(drop=True).to_parquet(
            os.path.join(event_dir, "month={:02d}.parquet".format(month)),
            engine="pyarrow", index=False,
        )

    but.build(YEAR)
    return {"out_dir": out_dir, "config_dir": config_dir}


@pytest.fixture()
def context(synthetic_project):
    return acc.build_context(YEAR)


# ---------------------------------------------------------------------------
# 类别来源：有 configs/*.json 用它，没有就退回表 B，绝不是抛一个看不懂的错
# ---------------------------------------------------------------------------

def test_categories_come_from_the_config_account_lists_when_they_exist(context):
    assert context.categories["public"]["central_news"] == {"pa1", "pa2", "pa6"}
    assert context.categories["celebrity"]["演员"] == {"ca1", "ca2", "xa9"}
    assert context.category_source["public"].endswith("news_user_ids.json")


def test_categories_fall_back_to_table_b_with_a_clear_message_when_lists_are_missing(
    synthetic_project, monkeypatch, capsys
):
    """本机没有 configs/*.json（它们在服务器上由 merged_profiles 生成）时
    必须退回表 B 自带的 source_category，并说清楚发生了什么"""
    monkeypatch.setattr(config, "PUBLIC_ACCOUNTS_FILE", "/nonexistent/news_user_ids.json")
    monkeypatch.setattr(
        config, "CELEBRITY_ACCOUNTS_FILE", "/nonexistent/entertain_user_ids.json"
    )
    context = acc.build_context(YEAR)
    printed = capsys.readouterr().out
    assert "news_user_ids.json" in printed
    assert "表 B" in printed
    # 表 B 里记的类别与名单口径一致（跨类别账号在表 B 里是拼接串，
    # 退回路径必须把它拆回成员资格）
    assert context.categories["public"]["central_news"] == {"pa1", "pa2", "pa6"}
    assert context.categories["celebrity"]["歌手"] == {"ca2", "ca3"}
    assert context.categories["public"]["gov_release_weibo"] == {"pa5", "xa9"}
    assert context.category_source["public"] == acc.CATEGORY_SOURCE_TABLE_B


# ---------------------------------------------------------------------------
# 最要紧的一条：被剔除账号的唯一来源用户必须变成非参与者
# ---------------------------------------------------------------------------

def test_dropping_the_only_account_a_user_retweeted_makes_them_a_non_participant(context):
    baseline = acc.user_frame_for_subset(context, None)
    solo = baseline[baseline["user_id"] == SOLO_PUBLIC_USER].iloc[0]
    assert bool(solo["public_source_entered"]) is True
    assert int(solo["public_source_count"]) == 1

    retained = set(context.incidence.accounts["r_user_id"]) - {SOLO_PUBLIC_ACCOUNT}
    variant = acc.user_frame_for_subset(context, retained)
    solo_after = variant[variant["user_id"] == SOLO_PUBLIC_USER].iloc[0]
    assert bool(solo_after["public_source_entered"]) is False
    assert int(solo_after["public_source_count"]) == 0

    # 只有那一个用户翻转，其他人纹丝不动
    flipped = baseline["public_source_entered"].to_numpy(dtype=bool) & (
        ~variant["public_source_entered"].to_numpy(dtype=bool)
    )
    assert baseline.loc[flipped, "user_id"].tolist() == [SOLO_PUBLIC_USER]


def test_dropping_a_celebrity_account_flips_its_solo_user_too(context):
    retained = set(context.incidence.accounts["r_user_id"]) - {SOLO_CELEBRITY_ACCOUNT}
    variant = acc.user_frame_for_subset(context, retained)
    solo = variant[variant["user_id"] == SOLO_CELEBRITY_USER].iloc[0]
    assert bool(solo["celebrity_source_entered"]) is False
    # 另一个领域不受牵连
    assert int(variant["celebrity_source_entered"].sum()) == int(
        acc.user_frame_for_subset(context, None)["celebrity_source_entered"].sum()
    ) - 1


def test_a_user_with_no_retweets_at_all_is_a_non_participant_not_a_missing_value(context):
    frame = acc.user_frame_for_subset(context, set())
    assert frame["public_source_entered"].notna().all()
    assert not frame["public_source_entered"].any()
    assert not frame["celebrity_source_entered"].any()
    assert int(frame["public_source_count"].sum()) == 0
    # 用户全集不变：变体不改样本，只改测量
    assert len(frame) == len(context.user_table)


def test_the_rebuilt_baseline_reproduces_table_c_entry_indicators(context):
    """全账号重建必须与表 C 完全一致，否则每个变体都在同一个方向上错"""
    baseline = acc.user_frame_for_subset(context, None)
    for domain in acc.DOMAINS:
        expected = context.user_table.set_index("user_id")[
            "{}_source_entered".format(domain)
        ]
        got = baseline.set_index("user_id")["{}_source_entered".format(domain)]
        assert got.astype(bool).equals(expected.astype(bool).reindex(got.index))
        expected_count = context.user_table.set_index("user_id")[
            "{}_source_count".format(domain)
        ]
        got_count = baseline.set_index("user_id")["{}_source_count".format(domain)]
        assert got_count.astype(int).equals(expected_count.astype(int).reindex(got_count.index))


# ---------------------------------------------------------------------------
# 留一账号：被剔除的量必须写下来，top_n 截断的覆盖面也必须写下来
# ---------------------------------------------------------------------------

def test_leave_one_account_out_flips_the_entry_of_the_solo_user_and_records_it(
    context, tmp_path
):
    out_path = str(tmp_path / "accounts.parquet")
    diag_path = str(tmp_path / "diag.parquet")
    acc.run_leave_one_account_out(
        YEAR, top_n=len(_accounts_by_domain("public")), context=context,
        out_path=out_path, diag_path=diag_path,
    )
    diag = pd.read_parquet(diag_path, columns=list(acc.DIAGNOSTIC_SCHEMA))
    row = diag[
        (diag["domain"] == "public")
        & (diag["excluded_accounts"] == SOLO_PUBLIC_ACCOUNT)
    ]
    assert len(row) == 1
    assert int(row.iloc[0]["n_users_lost_entry"]) == 1
    # u046 是偶数号，性别为男
    assert int(row.iloc[0]["n_users_lost_entry_male"]) == 1
    assert int(row.iloc[0]["n_users_lost_entry_female"]) == 0


def test_leave_one_account_out_records_the_excluded_volume_share(context, tmp_path):
    diag_path = str(tmp_path / "diag.parquet")
    acc.run_leave_one_account_out(
        YEAR, top_n=2, context=context, out_path=None, diag_path=diag_path
    )
    diag = pd.read_parquet(diag_path, columns=list(acc.DIAGNOSTIC_SCHEMA))
    volume = _expected_volume("public")
    total = sum(volume.values())
    row = diag[(diag["domain"] == "public") & (diag["excluded_accounts"] == "pa1")]
    assert len(row) == 1
    assert row.iloc[0]["excluded_volume_share"] == pytest.approx(volume["pa1"] / total)
    assert int(row.iloc[0]["n_events_total"]) == total


def test_leave_one_account_out_records_the_coverage_of_its_top_n_truncation(
    context, tmp_path
):
    """一个不声明覆盖面的 top-N 就是一次自我夸大的稳健性检验"""
    diag_path = str(tmp_path / "diag.parquet")
    acc.run_leave_one_account_out(
        YEAR, top_n=2, context=context, out_path=None, diag_path=diag_path
    )
    diag = pd.read_parquet(diag_path, columns=list(acc.DIAGNOSTIC_SCHEMA))
    public = diag[
        (diag["domain"] == "public")
        & (diag["variant_label"].str.startswith("loo_public_"))
    ]
    volume = _expected_volume("public")
    top_two = sorted(volume, key=lambda a: (-volume[a], a))[:2]
    expected = sum(volume[a] for a in top_two) / sum(volume.values())
    assert set(public["excluded_accounts"]) == set(top_two)
    assert (public["n_accounts_considered"] == 2).all()
    assert public["considered_volume_share"].apply(
        lambda v: v == pytest.approx(expected)
    ).all()
    # 明星侧的留一变体不会把公共事务的覆盖面写到自己头上，反之亦然
    cross = diag[
        (diag["domain"] == "public")
        & (diag["variant_label"].str.startswith("loo_celebrity_"))
    ]
    assert cross["n_accounts_considered"].isna().all()


def test_leave_one_account_out_notes_the_truncation_on_the_result_rows(context, tmp_path):
    out_path = str(tmp_path / "accounts.parquet")
    rows = acc.run_leave_one_account_out(
        YEAR, top_n=2, context=context, out_path=out_path,
        diag_path=str(tmp_path / "diag.parquet"),
    )
    assert len(rows) == 2 * 2 * len(harness.QUANTITIES) * 2
    for note in rows["note"]:
        assert "top_n=2" in str(note)
        assert "excluded_volume_share=" in str(note)


def test_excluding_a_public_account_leaves_the_celebrity_domain_untouched(
    context, tmp_path
):
    """剔除按 (账号, 领域) 键，不按账号 ID 全局键

    xa9 同时在两份名单上（真实数据里确有这种账号，表 B 的 manifest 专门
    记了 overlapping_accounts）。"留一公共事务账号"若按账号 ID 全局剔除，
    会顺手把 xa9 的**明星**那一列也删掉，于是一个本应只扰动公共事务的
    变体把明星的转发量也削掉了一块——variant_label 说的和真正跑的就不是
    一回事。§13.5 要问的是"少了这一个机构账号，公共事务会不会变"。
    """
    diag_path = str(tmp_path / "diag.parquet")
    acc.run_leave_one_account_out(
        YEAR, top_n=len(_accounts_by_domain("public")), context=context,
        out_path=None, diag_path=diag_path,
    )
    diag = pd.read_parquet(diag_path, columns=list(acc.DIAGNOSTIC_SCHEMA))
    label = diag.loc[
        (diag["domain"] == "public") & (diag["excluded_accounts"] == OVERLAP_ACCOUNT),
        "variant_label",
    ]
    assert len(label) == 1
    celebrity = diag[
        (diag["variant_label"] == label.iloc[0]) & (diag["domain"] == "celebrity")
    ].iloc[0]
    assert celebrity["excluded_volume_share"] == pytest.approx(0.0)
    assert int(celebrity["n_accounts_excluded"]) == 0
    assert int(celebrity["n_users_lost_entry"]) == 0

    # 逐用户核对：xa9 的两个转发者在明星领域的计数一个都不能少
    weights = {
        "public": {a: (0 if a == OVERLAP_ACCOUNT else 1)
                   for a in acc.subset_weights(context, "public", None)},
        "celebrity": acc.subset_weights(context, "celebrity", None),
    }
    variant = acc.user_frame_for_weights(context, weights)
    baseline = context.baseline_frame
    assert variant["celebrity_source_count"].equals(baseline["celebrity_source_count"])
    assert variant["celebrity_source_entered"].equals(
        baseline["celebrity_source_entered"]
    )
    # 而公共事务那一侧确实动了
    assert int(variant["public_source_count"].sum()) < int(
        baseline["public_source_count"].sum()
    )


# ---------------------------------------------------------------------------
# 留一类别
# ---------------------------------------------------------------------------

def test_category_exclusion_removes_exactly_the_accounts_in_that_category(
    context, tmp_path
):
    diag_path = str(tmp_path / "diag.parquet")
    acc.run_leave_one_category_out(
        YEAR, context=context, out_path=None, diag_path=diag_path
    )
    diag = pd.read_parquet(diag_path, columns=list(acc.DIAGNOSTIC_SCHEMA))
    row = diag[
        (diag["variant_label"] == "public_without_central_news")
        & (diag["domain"] == "public")
    ]
    assert len(row) == 1
    excluded = set(str(row.iloc[0]["excluded_accounts"]).split(","))
    # pa6 同时属于 central_news 与 org_news：留一类别按成员资格剔除，
    # 因此它也必须被剔掉
    assert excluded == {"pa1", "pa2", "pa6"}
    assert int(row.iloc[0]["n_accounts_excluded_multi_category"]) == 1
    volume = _expected_volume("public")
    assert row.iloc[0]["excluded_volume_share"] == pytest.approx(
        sum(volume[a] for a in excluded) / sum(volume.values())
    )


def test_leave_one_category_out_covers_both_the_media_categories_and_the_keywords(
    context, tmp_path
):
    rows = acc.run_leave_one_category_out(
        YEAR, context=context, out_path=str(tmp_path / "accounts.parquet"),
        diag_path=str(tmp_path / "diag.parquet"),
    )
    labels = set(rows["variant_label"])
    assert "public_without_gov_release_weibo" in labels
    assert "celebrity_without_演员" in labels
    assert len(labels) == len(_category_map("public")) + len(_category_map("celebrity"))


# ---------------------------------------------------------------------------
# 剔除头部账号 + 集中度：§13.5 真正问的那件事
# ---------------------------------------------------------------------------

def test_exclude_top_accounts_removes_the_k_highest_volume_accounts(context, tmp_path):
    diag_path = str(tmp_path / "diag.parquet")
    acc.run_exclude_top_accounts(
        YEAR, k=(1, 2), context=context, out_path=None, diag_path=diag_path
    )
    diag = pd.read_parquet(diag_path, columns=list(acc.DIAGNOSTIC_SCHEMA))
    row = diag[
        (diag["variant_label"] == "celebrity_exclude_top1") & (diag["domain"] == "celebrity")
    ]
    assert len(row) == 1
    volume = _expected_volume("celebrity")
    assert str(row.iloc[0]["excluded_accounts"]) == "ca1"
    assert row.iloc[0]["excluded_volume_share"] == pytest.approx(
        volume["ca1"] / sum(volume.values())
    )
    # 夹具里 ca1 独占八成以上：这正是"一个粉丝群撑起整个测量"的形状
    assert row.iloc[0]["excluded_volume_share"] > 0.8


def test_concentration_reports_top_k_shares_per_gender(context):
    conc = acc.concentration_by_gender(context, k=(1, 5, 10))
    assert set(conc.columns) == set(acc.CONCENTRATION_SCHEMA)
    assert set(conc["gender"]) == {"all", "m", "f"}
    for gender, genders in (("all", None), ("m", {"m"}), ("f", {"f"})):
        volume = _expected_volume("celebrity", genders)
        total = sum(volume.values())
        top1 = max(volume.values())
        row = conc[
            (conc["domain"] == "celebrity") & (conc["gender"] == gender)
            & (conc["k"] == 1)
        ]
        assert len(row) == 1
        assert row.iloc[0]["top_k_share"] == pytest.approx(top1 / total)
        assert int(row.iloc[0]["n_events"]) == total


def test_top_k_share_covers_exactly_k_accounts_where_the_quantile_round_trip_fails():
    """恰好 k 个，不是 k+1 个

    stats_utils.top_share 的入口是"头部用户占比 q"，内部取 k = ceil(q*n)。
    把"前 k 个账号"换算成 q = k/n 再喂给它，看上去应该还原成 k，在浮点上
    并不成立：n=147、k=5 时 ceil((5/147)*147) = 6，报出去的就成了前 6 个
    账号的份额，而同一行的 top_accounts 只列了 5 个。集中度是读者最可能
    直接引用的数字，这种"悄悄多算一个账号"正是简报警告的自我夸大。
    """
    n, k = 147, 5
    values = np.arange(n, 0, -1, dtype=float)
    exact = values[:k].sum() / values.sum()

    assert acc.top_k_share(values, k) == pytest.approx(exact)
    # 钉死这个 n 上的往返确实会多算一个账号（否则本测试就失去意义了）
    assert int(np.ceil((k / n) * n)) == k + 1
    assert su.top_share(values, k / n) == pytest.approx(values[: k + 1].sum() / values.sum())
    assert acc.top_k_share(values, k) != pytest.approx(su.top_share(values, k / n))


def test_top_k_share_edges():
    assert acc.top_k_share([3.0, 1.0], 0) == 0.0
    assert acc.top_k_share([3.0, 1.0], 5) == pytest.approx(1.0)
    assert np.isnan(acc.top_k_share([0.0, 0.0], 1))
    assert np.isnan(acc.top_k_share([], 1))


def test_concentration_share_matches_the_accounts_the_row_lists(context):
    """每一行的份额必须正好是这一行 top_accounts 里那几个账号的份额"""
    conc = acc.concentration_by_gender(context, k=(1, 2, 3))
    for _, row in conc.iterrows():
        listed = [a for a in str(row["top_accounts"]).split(",") if a]
        volume = _expected_volume(
            row["domain"], None if row["gender"] == "all" else {row["gender"]}
        )
        assert len(listed) == min(int(row["k"]), int(row["n_accounts"]))
        assert row["top_k_share"] == pytest.approx(
            sum(volume[a] for a in listed) / sum(volume.values())
        )


def test_concentration_marks_k_larger_than_the_account_count(context):
    """k 超过该领域账号数时份额恒为 1.0，必须能从 n_accounts 看出这件事"""
    conc = acc.concentration_by_gender(context, k=(10,))
    row = conc[(conc["domain"] == "celebrity") & (conc["gender"] == "all")].iloc[0]
    assert int(row["n_accounts"]) == len(_accounts_by_domain("celebrity"))
    assert int(row["k"]) > int(row["n_accounts"])
    assert row["top_k_share"] == pytest.approx(1.0)


def test_concentration_also_ranks_by_the_pooled_volume_so_genders_are_comparable(context):
    """按各自性别排名与按整体排名是两件事，两个都要有——否则无法回答
    '男女是不是被同一批账号带动的'"""
    conc = acc.concentration_by_gender(context, k=(1,))
    volume_all = _expected_volume("celebrity")
    top_account = sorted(volume_all, key=lambda a: (-volume_all[a], a))[0]
    for gender, genders in (("m", {"m"}), ("f", {"f"})):
        volume = _expected_volume("celebrity", genders)
        row = conc[
            (conc["domain"] == "celebrity") & (conc["gender"] == gender)
            & (conc["k"] == 1)
        ].iloc[0]
        assert row["top_k_share_pooled_ranking"] == pytest.approx(
            volume[top_account] / sum(volume.values())
        )


# ---------------------------------------------------------------------------
# 只保留中央级官方媒体
# ---------------------------------------------------------------------------

def test_core_national_media_only_keeps_central_categories_and_leaves_celebrity_alone(
    context, tmp_path
):
    diag_path = str(tmp_path / "diag.parquet")
    acc.run_core_national_media_only(
        YEAR, context=context, out_path=None, diag_path=diag_path
    )
    diag = pd.read_parquet(diag_path, columns=list(acc.DIAGNOSTIC_SCHEMA))
    public = diag[diag["domain"] == "public"].iloc[0]
    celebrity = diag[diag["domain"] == "celebrity"].iloc[0]
    # central_news 里有 pa1/pa2/pa6，central_theory_website 在夹具里没有账号
    assert int(public["n_accounts_retained"]) == 3
    assert celebrity["excluded_volume_share"] == pytest.approx(0.0)
    assert int(celebrity["n_accounts_retained"]) == len(_accounts_by_domain("celebrity"))


def test_core_national_media_only_uses_the_config_module_categories():
    """中央级类别名必须来自 configs/cn_news_sources_lists.py 的划分，
    不是在本模块里临时编一组名字"""
    for category in acc.CORE_NATIONAL_CATEGORIES:
        assert category in acc.NEWS_CATEGORY_NAMES


# ---------------------------------------------------------------------------
# 账号 bootstrap
# ---------------------------------------------------------------------------

def test_account_bootstrap_draw_is_reproducible_under_a_seed(context):
    a = acc.bootstrap_draw(context.strata["public"], seed=7)
    b = acc.bootstrap_draw(context.strata["public"], seed=7)
    c = acc.bootstrap_draw(context.strata["public"], seed=8)
    assert a == b
    assert a != c


def test_account_bootstrap_uses_an_independent_stream_per_domain(context):
    """两个领域各抽各的：共用一个种子会在两份互不相干的名单之间造出一种
    没有实质含义的固定对应关系（vocabulary 出于同样的理由逐领域派生）"""
    replicate_seed = 12345
    seeds = {domain: acc.domain_seed(replicate_seed, domain) for domain in acc.DOMAINS}
    assert seeds["public"] != seeds["celebrity"]
    assert seeds["public"] != replicate_seed
    # 派生是确定性的
    assert acc.domain_seed(replicate_seed, "public") == seeds["public"]


def test_account_bootstrap_is_stratified_by_category(context):
    """每一层各抽自己那么多个账号：否则小类别会被整层抽没，
    '按类别分层'就名不副实"""
    strata = context.strata["public"]
    for seed in range(10):
        draw = acc.bootstrap_draw(strata, seed=seed)
        for stratum, members in strata.items():
            drawn = sum(n for a, n in draw.items() if a in set(members))
            assert drawn == len(members)


def test_account_bootstrap_resamples_accounts_not_events(context):
    """哨兵：u045 只转 prov_release_weibo 这一层的两个账号（3 次与 5 次），
    该层恰好 2 个账号。按账号有放回抽 2 个，u045 的公共来源转发数只能是
    3a+5b（a+b=2），即 {6, 8, 10}；改成重抽事件会稳定在 8 附近并出现
    7、9 之类的值，本测试立刻失败。
    """
    observed = set()
    for seed in range(60):
        draw = {"public": acc.bootstrap_draw(context.strata["public"], seed=seed),
                "celebrity": acc.bootstrap_draw(context.strata["celebrity"], seed=seed)}
        frame = acc.user_frame_for_weights(context, draw)
        count = int(
            frame.loc[frame["user_id"] == "u045", "public_source_count"].iloc[0]
        )
        observed.add(count)
    assert observed <= {6, 8, 10}, "重抽样单位不是账号：出现了 3a+5b 之外的计数"
    assert len(observed) >= 2, "计数一次都没变过，说明重抽样根本没有作用在账号上"


def test_account_bootstrap_entry_is_decided_by_whether_the_account_was_drawn(context):
    """只转过一个账号的用户，entry 必须恰好等于'那个账号有没有被抽到'"""
    for seed in range(20):
        draw = {
            domain: acc.bootstrap_draw(context.strata[domain], seed=seed)
            for domain in acc.DOMAINS
        }
        frame = acc.user_frame_for_weights(context, draw)
        entered = bool(
            frame.loc[
                frame["user_id"] == SOLO_CELEBRITY_USER, "celebrity_source_entered"
            ].iloc[0]
        )
        assert entered == (draw["celebrity"].get(SOLO_CELEBRITY_ACCOUNT, 0) > 0)


def test_account_bootstrap_counts_a_twice_drawn_account_twice(context):
    """重复抽到的账号必须按重数计入来源转发数，这才是有放回重抽样"""
    draw = {"public": {"pa7": 2}, "celebrity": {}}
    frame = acc.user_frame_for_weights(context, draw)
    row = frame[frame["user_id"] == "u045"].iloc[0]
    assert int(row["public_source_count"]) == 6
    assert bool(row["public_source_entered"]) is True


def test_account_bootstrap_writes_one_seed_per_replicate(context, tmp_path):
    out_path = str(tmp_path / "accounts.parquet")
    rows = acc.run_account_bootstrap(
        YEAR, n_replicates=3, seed=0, context=context, out_path=out_path,
        diag_path=str(tmp_path / "diag.parquet"),
    )
    assert rows["seed"].notna().all()
    assert rows["replicate"].nunique() == 3
    assert len(rows) == 3 * len(harness.QUANTITIES) * 2
    assert set(rows["variant_family"]) == {acc.BOOTSTRAP_FAMILY}


def test_account_bootstrap_is_reproducible_end_to_end(context, tmp_path):
    first = acc.run_account_bootstrap(
        YEAR, n_replicates=2, seed=3, context=context, out_path=None, diag_path=None
    )
    second = acc.run_account_bootstrap(
        YEAR, n_replicates=2, seed=3, context=context, out_path=None, diag_path=None
    )
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# 全流程
# ---------------------------------------------------------------------------

def test_build_writes_results_diagnostics_concentration_and_manifest(synthetic_project):
    out = acc.build(YEAR, top_n=2, k=(1,), n_replicates=2, seed=0)
    results = pd.read_parquet(out["results_path"], columns=list(harness.ROBUSTNESS_SCHEMA))
    diag = pd.read_parquet(out["diagnostics_path"], columns=list(acc.DIAGNOSTIC_SCHEMA))
    conc = pd.read_parquet(out["concentration_path"], columns=list(acc.CONCENTRATION_SCHEMA))

    robust_dir = os.path.join(config.OUTPUT_DIR, "robustness")
    for key in ("results_path", "diagnostics_path", "concentration_path"):
        assert out[key].startswith(robust_dir)

    families = set(results["variant_family"])
    assert acc.VARIANT_FAMILY in families
    assert acc.BOOTSTRAP_FAMILY in families
    # 每个变体恰好 6 个量 × 2 层
    per_label = results.groupby(["variant_family", "variant_label"]).size()
    assert set(per_label.unique()) == {len(harness.QUANTITIES) * 2}
    # 每一行诊断都带着被剔除的量
    assert diag["excluded_volume_share"].notna().all()
    assert set(conc["gender"]) == {"all", "m", "f"}

    manifest_path = os.path.join(robust_dir, "accounts_{}".format(YEAR), "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["params"]["top_n"] == 2
    assert len(manifest["params"]["seeds"]) == 2
    assert manifest["params"]["category_source"]["public"].endswith("news_user_ids.json")
    assert "celebrity_top1_share" in manifest["counts"]["concentration"]
    assert "public" in manifest["fingerprints"]


def test_every_result_row_can_find_its_diagnostics(synthetic_project):
    out = acc.build(YEAR, top_n=1, k=(1,), n_replicates=1, seed=0)
    results = pd.read_parquet(out["results_path"], columns=list(harness.ROBUSTNESS_SCHEMA))
    diag = pd.read_parquet(out["diagnostics_path"], columns=list(acc.DIAGNOSTIC_SCHEMA))
    keys = set(zip(diag["variant_family"], diag["variant_label"]))
    unmatched = {
        (fam, lab)
        for fam, lab in zip(results["variant_family"], results["variant_label"])
        if (fam, lab) not in keys
    }
    assert unmatched == set()
    # 每个键下两个领域各一行
    per_key = diag.groupby(["variant_family", "variant_label"])["domain"].nunique()
    assert set(per_key.unique()) == {2}


def test_build_is_resumable_because_rows_are_appended_per_variant(
    synthetic_project, monkeypatch, tmp_path
):
    context = acc.build_context(YEAR)
    out_path = str(tmp_path / "accounts.parquet")
    calls = {"n": 0}
    real_estimate_all = harness.estimate_all

    def _boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("模拟作业被杀")
        return real_estimate_all(*args, **kwargs)

    monkeypatch.setattr(acc.harness, "estimate_all", _boom)
    with pytest.raises(RuntimeError):
        acc.run_account_bootstrap(
            YEAR, n_replicates=5, context=context, out_path=out_path,
            diag_path=str(tmp_path / "diag.parquet"),
        )
    done = pd.read_parquet(out_path, columns=list(harness.ROBUSTNESS_SCHEMA))
    assert len(done) == 2 * len(harness.QUANTITIES) * 2


def test_missing_input_tables_degrade_with_a_message_about_where_they_come_from(
    tmp_path, monkeypatch
):
    """本机没有分析表时报的错必须说清缺的是哪张表、由哪一步生成"""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as excinfo:
        acc.build_context(YEAR)
    message = str(excinfo.value)
    assert "user_domain_{}.parquet".format(YEAR) in message
    assert "build_user_tables" in message
