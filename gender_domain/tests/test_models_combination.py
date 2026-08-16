"""
gender_domain.models_combination 的单元测试（§7 来源/内容分解、§8 参与组合）。

测试设计原则（写在这里，避免后来者以为某条断言只是"顺手加的"）：

1. **四分类必须是一个划分。** neither / source_only / content_only / both
   四格互斥且穷尽，是本模块所有 §7 数字的前提。fixture 覆盖全部四格，
   断言"每个被分类的用户恰好落在一格"以及"四格占比相加为 1"。

2. **phi 必须与手算的 2×2 对得上。** 相关系数很容易写成另一个公式还
   看起来合理（例如漏掉某个边际），因此这里用一个手算得出 0.408248 的
   2×2 表钉死它。

3. **相关只能用两个量都有定义的用户算，且必须报告丢了多少人。**
   没有表达帖的用户两个域的 topical_share 都是 NaN，把它当成 0 会凭空
   制造相关。fixture 刻意让"把 NaN 当 0"会显著改变结果，断言实际估计
   等于只用有定义用户算出来的值、且 n_dropped 如实计数。

4. **多项 logit 的性别 AME 必须四格相加为 0**（一个用户被推进某一格，
   必然从另一格里被推出来）。这是一条内部一致性检查：如果 AME 是从
   系数里读出来的、或者某一格漏算，这条恒等式立刻破。M0 层还有更强的
   断言——只有性别一个自变量时多项 logit 恰好拟合各性别的观测占比，
   因此 AME 必须精确等于"男性占比 - 女性占比"。

5. **§8.3 的禁令要有测试守着。** 任何输出行都不允许出现"新闻 vs 娱乐
   偏好分数"这类一维指标，测试直接扫描输出的 outcome/term/model/note
   与模块源码里的可疑命名。
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from gender_domain import models_combination as mcomb
from gender_domain import stats_utils as su


# ---------------------------------------------------------------------------
# 模拟数据构造
# ---------------------------------------------------------------------------

def _base_frame(n, seed=0, gender=None):
    """构造一份表 C 形状的骨架（不含结果变量），列名与 build_user_tables 一致"""
    rng = np.random.default_rng(seed)
    if gender is None:
        gender = np.array(["m"] * (n // 2) + ["f"] * (n - n // 2))
    frame = pd.DataFrame({
        "user_id": [str(i) for i in range(n)],
        "gender": gender,
        "n_posts": rng.integers(1, 200, n),
        "n_retweets": rng.integers(1, 100, n),
        "n_expressive_posts": rng.integers(1, 80, n),
        "n_active_days": rng.integers(1, 300, n),
        "n_active_months": rng.integers(1, 13, n),
        "n_active_months_panel": rng.integers(1, 13, n),
        "region": rng.choice(["north", "south", "west"], n),
        "verified_flag": rng.integers(0, 2, n).astype(float),
        "log_fans": rng.normal(5, 1, n),
        "log_friends": rng.normal(4, 1, n),
        "profile_complete": np.ones(n, dtype=bool),
    })
    for domain in mcomb.DOMAINS:
        frame[f"{domain}_source_entered"] = rng.random(n) < 0.4
        frame[f"{domain}_source_count"] = rng.integers(0, 20, n)
        frame[f"{domain}_topical_posts"] = rng.integers(0, 10, n)
        frame[f"{domain}_topical_share"] = rng.random(n)
    frame["source_combo"] = _combo(frame)
    return frame


def _combo(frame):
    """与 build_user_tables._source_combo_series 同一口径的组合标签"""
    public = frame["public_source_entered"].to_numpy()
    celebrity = frame["celebrity_source_entered"].to_numpy()
    return np.select(
        [public & celebrity, public & ~celebrity, ~public & celebrity],
        ["both", "public_only", "celebrity_only"],
        default="neither",
    )


def _four_cell_fixture():
    """四格各一个用户的最小 fixture（domain=public）

    第 5 个用户没有表达帖：两个域的 topical_share 都是 NaN，内容参与
    无定义，按本模块的口径应当被剔除并计数，而不是当成"没有内容参与"。
    """
    frame = _base_frame(5, seed=1, gender=np.array(["m", "f", "m", "f", "m"]))
    frame["public_source_entered"] = [True, True, False, False, True]
    frame["celebrity_source_entered"] = [False, True, True, False, False]
    frame["public_topical_share"] = [0.5, 0.0, 0.5, 0.0, np.nan]
    frame["celebrity_topical_share"] = [0.5, 0.0, 0.5, 0.0, np.nan]
    frame["n_expressive_posts"] = [10, 10, 10, 10, 0]
    frame["source_combo"] = _combo(frame)
    return frame


def _phi_fixture():
    """2×2 计数写死为 a=40, b=10, c=20, d=30 的 fixture（domain=public）

    手算 phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
             = (40*30 - 10*20) / sqrt(50 * 50 * 60 * 40)
             = 1000 / 2449.4897... = 0.4082482905
    """
    counts = [(True, True, 40), (True, False, 10), (False, True, 20), (False, False, 30)]
    rows = []
    for source, content, k in counts:
        for _ in range(k):
            rows.append((source, content))
    n = len(rows)
    frame = _base_frame(n, seed=2, gender=np.array(["f"] * n))
    frame["public_source_entered"] = [r[0] for r in rows]
    frame["public_topical_share"] = [0.5 if r[1] else 0.0 for r in rows]
    frame["source_combo"] = _combo(frame)
    return frame


PHI_EXPECTED = 1000.0 / np.sqrt(50 * 50 * 60 * 40)


def _combo_fixture(n_per_gender=1500, seed=7):
    """按写死的组合分布抽样，M0 层的性别 AME 因此有解析真值

    多项 logit 只放性别一个自变量时是饱和模型，拟合出来的各格概率恰好
    等于观测占比，所以 AME 必须精确等于"男性占比 - 女性占比"。
    """
    rng = np.random.default_rng(seed)
    probs = {
        "m": {"neither": 0.25, "public_only": 0.40, "celebrity_only": 0.15, "both": 0.20},
        "f": {"neither": 0.25, "public_only": 0.10, "celebrity_only": 0.45, "both": 0.20},
    }
    genders, combos = [], []
    for g in ("m", "f"):
        cats = list(probs[g])
        draw = rng.choice(cats, size=n_per_gender, p=[probs[g][c] for c in cats])
        genders.extend([g] * n_per_gender)
        combos.extend(draw.tolist())
    frame = _base_frame(len(genders), seed=seed, gender=np.array(genders))
    frame["source_combo"] = combos
    frame["public_source_entered"] = [c in ("public_only", "both") for c in combos]
    frame["celebrity_source_entered"] = [c in ("celebrity_only", "both") for c in combos]
    return frame


def _row(frame, **filters):
    sub = frame
    for key, value in filters.items():
        sub = sub[sub[key] == value]
    assert len(sub) == 1, f"期望恰好一行 {filters}，实际 {len(sub)} 行"
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# §7.1 四分类
# ---------------------------------------------------------------------------

def test_classification_is_exhaustive_and_mutually_exclusive():
    frame = _four_cell_fixture()
    labels = mcomb.classify_participation(frame, "public")
    classified = labels.dropna()
    assert set(classified) == set(mcomb.PARTICIPATION_TYPES)
    assert len(classified) == 4               # 第 5 个用户内容参与无定义
    assert sorted(classified.tolist()) == sorted(mcomb.PARTICIPATION_TYPES)


def test_participation_shares_sum_to_one_within_each_gender():
    frame = _base_frame(200, seed=3)
    out = mcomb.participation_types(frame, "public")
    for gender in ("male", "female"):
        shares = out[out["term"].str.endswith(f"_{gender}")]["estimate"]
        assert len(shares) == len(mcomb.PARTICIPATION_TYPES)
        assert shares.sum() == pytest.approx(1.0)


def test_threshold_variant_moves_users_out_of_content_participation():
    frame = _base_frame(200, seed=4)
    frame["public_topical_share"] = 0.02      # 人人都有命中，但占比很低
    any_hit = mcomb.classify_participation(frame, "public")
    strict = mcomb.classify_participation(frame, "public", content_threshold=0.10)
    assert set(any_hit.dropna()) <= {"content_only", "both"}
    assert set(strict.dropna()) <= {"neither", "source_only"}


def test_threshold_is_recorded_in_the_output():
    frame = _base_frame(120, seed=5)
    out = mcomb.participation_types(frame, "public", content_threshold=0.10)
    assert (out["model"] == "share_ge_0.1").all()
    assert out["note"].str.contains("content_threshold=0.1").all()

    any_hit = mcomb.participation_types(frame, "public")
    assert (any_hit["model"] == mcomb.VARIANT_ANY_HIT).all()
    assert any_hit["note"].str.contains("content_threshold=any_hit").all()


def test_users_without_expressive_posts_are_dropped_and_counted():
    frame = _four_cell_fixture()
    out = mcomb.participation_types(frame, "public")
    male_rows = out[out["term"].str.endswith("_male")]
    # 三个男性用户里有一个没有表达帖，内容参与无定义
    assert (male_rows["n_obs"] == 2).all()
    assert (male_rows["n_dropped"] == 1).all()
    assert male_rows["drop_reason"].str.contains(mcomb.REASON_NO_EXPRESSIVE).all()


def test_participation_types_uses_the_shared_schema():
    frame = _base_frame(80, seed=6)
    out = mcomb.participation_types(frame, "celebrity")
    assert list(out.columns) == list(su.RESULT_SCHEMA)


# ---------------------------------------------------------------------------
# §7.2 交叉关系
# ---------------------------------------------------------------------------

def test_phi_matches_hand_computed_two_by_two():
    frame = _phi_fixture()
    out = mcomb.cross_relations(frame, "public")
    row = _row(out, term="phi_all", model=mcomb.VARIANT_ANY_HIT)
    assert row["estimate"] == pytest.approx(PHI_EXPECTED, abs=1e-9)
    assert row["scale"] == "correlation"


def test_conditional_shares_match_hand_computed_values():
    frame = _phi_fixture()
    out = mcomb.cross_relations(frame, "public")
    # 来源参与者 50 人，其中 40 人有内容参与
    row = _row(out, term="content_given_source_all", model=mcomb.VARIANT_ANY_HIT)
    assert row["estimate"] == pytest.approx(40 / 50)
    assert row["n_obs"] == 50
    # 内容参与者 60 人，其中 40 人有来源参与
    row = _row(out, term="source_given_content_all", model=mcomb.VARIANT_ANY_HIT)
    assert row["estimate"] == pytest.approx(40 / 60)
    assert row["n_obs"] == 60


def test_activity_conditional_spearman_removes_a_pure_activity_artifact():
    """两个量都只由活跃度驱动时，原始相关很强、控制活跃度后应当大幅衰减"""
    rng = np.random.default_rng(11)
    n = 600
    frame = _base_frame(n, seed=11)
    activity = rng.gamma(2.0, 20.0, n)
    frame["n_posts"] = activity.round().astype(int) + 1
    frame["n_retweets"] = (activity * 0.8).round().astype(int) + 1
    frame["n_active_days"] = (activity * 1.5).round().astype(int) + 1
    frame["n_active_months"] = np.clip((activity / 20).round().astype(int) + 1, 1, 12)
    frame["public_source_count"] = (activity * 0.5 + rng.normal(0, 1, n)).round().clip(0)
    frame["public_topical_share"] = np.clip(
        activity / activity.max() + rng.normal(0, 0.02, n), 0.0, 1.0
    )
    out = mcomb.cross_relations(frame, "public")
    raw = _row(out, term="spearman_source_count_topical_share_all",
               model=mcomb.VARIANT_ANY_HIT)["estimate"]
    partial = _row(out, term="spearman_source_count_topical_share_partial_all",
                   model=mcomb.VARIANT_ANY_HIT)["estimate"]
    assert raw > 0.8
    assert abs(partial) < 0.5 * raw


def test_cross_relations_uses_the_shared_schema():
    frame = _base_frame(100, seed=12)
    out = mcomb.cross_relations(frame, "public")
    assert list(out.columns) == list(su.RESULT_SCHEMA)


# ---------------------------------------------------------------------------
# §7.3 内容参与 ~ 来源参与 × 性别
# ---------------------------------------------------------------------------

def test_content_on_source_recovers_a_positive_association():
    rng = np.random.default_rng(13)
    n = 1200
    frame = _base_frame(n, seed=13)
    source = rng.random(n) < 0.5
    frame["public_source_entered"] = source
    p = np.where(source, 0.8, 0.3)
    hit = rng.random(n) < p
    frame["public_topical_share"] = np.where(hit, 0.4, 0.0)
    frame["source_combo"] = _combo(frame)

    out = mcomb.fit_content_on_source(frame, "public")
    row = _row(out, model="M0", term=mcomb.TERM_SOURCE_AME)
    assert row["estimate"] == pytest.approx(0.5, abs=0.08)
    assert row["ci_low"] > 0
    assert row["scale"] == "probability"


def test_content_on_source_emits_all_layers_and_schema():
    frame = _base_frame(400, seed=14)
    out = mcomb.fit_content_on_source(frame, "public")
    assert list(out.columns) == list(su.RESULT_SCHEMA)
    assert sorted(out["model"].unique()) == ["M0", "M1", "M2"]
    for term in (mcomb.TERM_SOURCE_AME, mcomb.TERM_GENDER_AME,
                 mcomb.TERM_SOURCE_X_GENDER, mcomb.TERM_SOURCE_X_GENDER_COEF):
        assert (out["term"] == term).sum() == 3


# ---------------------------------------------------------------------------
# §8.1 多项 logit
# ---------------------------------------------------------------------------

def test_multinomial_ames_sum_to_zero_in_every_layer():
    frame = _combo_fixture(n_per_gender=400, seed=15)
    out = mcomb.fit_combination_multinomial(frame)
    for layer in ("M0", "M1", "M2"):
        ames = out[(out["model"] == layer)
                   & (out["term"].isin(mcomb.combo_ame_terms()))]["estimate"]
        assert len(ames) == 4
        assert ames.sum() == pytest.approx(0.0, abs=1e-8)


def test_multinomial_m0_recovers_the_observed_category_gaps():
    frame = _combo_fixture(n_per_gender=1500, seed=16)
    out = mcomb.fit_combination_multinomial(frame)
    male = frame[frame["gender"] == "m"]["source_combo"].value_counts(normalize=True)
    female = frame[frame["gender"] == "f"]["source_combo"].value_counts(normalize=True)
    for category in mcomb.COMBO_CATEGORIES:
        expected = float(male.get(category, 0.0) - female.get(category, 0.0))
        row = _row(out, model="M0", term=mcomb.combo_ame_term(category))
        assert row["estimate"] == pytest.approx(expected, abs=1e-4)
        assert np.isfinite(row["se"]) and row["se"] > 0


def test_multinomial_emits_a_sum_check_row_and_shared_schema():
    frame = _combo_fixture(n_per_gender=300, seed=17)
    out = mcomb.fit_combination_multinomial(frame)
    assert list(out.columns) == list(su.RESULT_SCHEMA)
    row = _row(out, model="M0", term=mcomb.TERM_AME_SUM)
    assert row["estimate"] == pytest.approx(0.0, abs=1e-8)
    assert "internal_consistency" in row["note"]


def test_multinomial_without_gender_variation_leaves_a_nan_row_not_an_exception():
    frame = _combo_fixture(n_per_gender=200, seed=18)
    frame["gender"] = "f"
    out = mcomb.fit_combination_multinomial(frame)
    row = _row(out, model="M0", term=mcomb.combo_ame_term("both"))
    assert np.isnan(row["estimate"])
    assert "no_gender_variation" in row["note"]


# ---------------------------------------------------------------------------
# §8.2 两域 topical_share 的相关
# ---------------------------------------------------------------------------

def test_content_correlation_excludes_undefined_users_and_counts_them():
    """把无定义的用户当成 0 会明显改变相关，本函数必须只用两侧都有定义的用户"""
    rng = np.random.default_rng(19)
    n_defined, n_missing = 200, 60
    n = n_defined + n_missing
    frame = _base_frame(n, seed=19, gender=np.array(["f"] * n))
    public = np.concatenate([rng.random(n_defined), np.full(n_missing, np.nan)])
    celebrity = np.concatenate([
        np.clip(1.0 - public[:n_defined] + rng.normal(0, 0.05, n_defined), 0, 1),
        np.full(n_missing, np.nan),
    ])
    frame["public_topical_share"] = public
    frame["celebrity_topical_share"] = celebrity
    frame["n_expressive_posts"] = [10] * n_defined + [0] * n_missing

    out = mcomb.content_correlation(frame)
    row = _row(out, model="raw", term="spearman_female")
    expected = spearmanr(public[:n_defined], celebrity[:n_defined]).statistic
    assert row["estimate"] == pytest.approx(expected, abs=1e-9)
    assert row["n_obs"] == n_defined
    assert row["n_dropped"] == n_missing
    assert mcomb.REASON_NO_EXPRESSIVE in row["drop_reason"]

    # 把缺失当成 0 会得到一个明显不同的数——这正是必须剔除它们的理由
    filled_public = np.nan_to_num(public)
    filled_celebrity = np.nan_to_num(celebrity)
    filled = spearmanr(filled_public, filled_celebrity).statistic
    assert abs(filled - expected) > 0.05


def test_content_correlation_reports_raw_and_activity_conditional():
    frame = _base_frame(300, seed=20)
    out = mcomb.content_correlation(frame)
    assert list(out.columns) == list(su.RESULT_SCHEMA)
    assert sorted(out["model"].unique()) == ["activity_partial", "raw"]
    for model in ("raw", "activity_partial"):
        for gender in ("male", "female", "all"):
            row = _row(out, model=model, term=f"spearman_{gender}")
            assert np.isfinite(row["estimate"])


# ---------------------------------------------------------------------------
# §8.3 禁令
# ---------------------------------------------------------------------------

_FORBIDDEN = ("preference", "leaning", "news_minus", "minus_celebrity",
              "domain_index", "preference_score", "orientation")


def test_no_single_preference_score_appears_in_any_output():
    frame = _combo_fixture(n_per_gender=200, seed=21)
    frames = [
        mcomb.participation_types(frame, "public"),
        mcomb.cross_relations(frame, "public"),
        mcomb.fit_content_on_source(frame, "public"),
        mcomb.fit_combination_multinomial(frame),
        mcomb.content_correlation(frame),
    ]
    for out in frames:
        text = " ".join(
            out[col].dropna().astype(str).str.cat(sep=" ")
            for col in ("outcome", "domain", "model", "term", "scale", "note")
        ).lower()
        for word in _FORBIDDEN:
            assert word not in text, f"输出里出现了被 §8.3 禁止的偏好分数命名: {word}"


def test_module_source_defines_no_preference_index():
    import inspect

    source = inspect.getsource(mcomb).lower()
    # 模块文档里会解释"为什么不做偏好分数"，所以只扫描代码行，跳过注释与文档
    code_lines = [
        line.split("#")[0] for line in source.splitlines()
        if not line.strip().startswith("#")
    ]
    code = " ".join(code_lines)
    for word in ("preference_score", "news_minus", "domain_index", "leaning"):
        assert word not in code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_build_writes_both_tables_and_a_manifest(tmp_path, monkeypatch):
    frame = _combo_fixture(n_per_gender=150, seed=22)
    monkeypatch.setattr(mcomb.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(mcomb, "CONTENT_THRESHOLDS", (0.10,))
    frame.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                     engine="pyarrow", index=False)

    paths = mcomb.build(year=2020)
    assert len(paths) == 2
    for path in paths:
        out = pd.read_parquet(path, columns=list(su.RESULT_SCHEMA))
        assert list(out.columns) == list(su.RESULT_SCHEMA)
        assert len(out) > 0

    manifest_path = tmp_path / "results" / "manifest_combination" / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["counts"]["n_users"] == len(frame)
    assert manifest["params"]["content_thresholds"] == [0.10]
    assert len(manifest["counts"]["completed_stages"]) >= 2


def test_build_writes_incrementally_so_a_killed_job_keeps_finished_stages(
        tmp_path, monkeypatch):
    frame = _combo_fixture(n_per_gender=120, seed=23)
    monkeypatch.setattr(mcomb.config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(mcomb, "CONTENT_THRESHOLDS", (0.10,))
    frame.to_parquet(tmp_path / "user_domain_2020_with_profile.parquet",
                     engine="pyarrow", index=False)

    def _explode(*args, **kwargs):
        raise RuntimeError("模拟多项 logit 阶段失败")

    monkeypatch.setattr(mcomb, "fit_combination_multinomial", _explode)
    with pytest.raises(RuntimeError):
        mcomb.build(year=2020)

    decomposition = tmp_path / "results" / "decomposition_source_content.parquet"
    assert decomposition.exists(), "§7 阶段已经算完，必须已经落盘"
    out = pd.read_parquet(decomposition, columns=list(su.RESULT_SCHEMA))
    assert len(out) > 0
