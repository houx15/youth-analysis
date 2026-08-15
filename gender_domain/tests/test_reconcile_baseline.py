import json

import pandas as pd
import pytest

from gender_domain import config
from gender_domain import reconcile_baseline as rb


def _users():
    """4 个用户：f 组 3 人（1 个来源转发者），m 组 1 人（未转发）

    专门构造成"来源转发进入率"两种分母会给出不同数字的样子：
    entered_rate_all_users 的分母是全部用户，entered_rate_among_retweeters
    的分母只数转发过的用户。
    """
    return pd.DataFrame(
        {
            "gender": ["f", "f", "f", "m"],
            "n_retweets": [3, 0, 0, 0],
            "public_source_entered": [True, False, False, False],
            "public_topical_share": [0.4, 0.0, 0.2, None],
            "celebrity_source_entered": [False, False, False, False],
            "celebrity_topical_share": [0.1, 0.0, 0.0, None],
        }
    )


def _overview():
    return pd.DataFrame(
        {
            "gender": ["f", "m"],
            "total_users": [3, 1],
            "users_with_retweet": [1, 0],
            "users_with_news_retweet": [1, 0],
            "users_with_entertain_retweet": [0, 0],
        }
    )


def _density():
    return pd.DataFrame(
        {
            "type": ["news", "news"],
            "level": ["user", "user"],
            "gender": ["f", "m"],
            "total_count": [3, 1],
            "zero_count": [2, 1],
            "zero_ratio": [0.667, 1.0],
            "non_zero_count": [1, 0],
            "non_zero_mean": [0.05, 0.0],
            "non_zero_median": [0.05, 0.0],
            "non_zero_std": [0.0, 0.0],
        }
    )


def test_summarize_new_table_computes_gender_split():
    summary = rb.summarize_new_table(_users())
    assert summary["users"] == {"f": 3, "m": 1}
    assert summary["public_source_entered"]["f"] == 1
    assert summary["public_source_entered"]["m"] == 0


def test_entered_rate_denominators_differ_for_all_users_vs_retweeters():
    """同一个数字用两种分母算出来必须不同，且都要出现在报告里"""
    summary = rb.summarize_new_table(_users())
    # 分母=全部同性别用户(3)：1/3
    assert summary["public_entered_rate_all_users"]["f"] == pytest.approx(1 / 3)
    # 分母=该性别全部转发者(1)：1/1，与旧基线口径一致
    assert summary["public_entered_rate_among_retweeters"]["f"] == pytest.approx(1.0)
    # 两个分母算出来的比例不应相等，这正是需要在报告里分别标注的原因
    assert summary["public_entered_rate_all_users"]["f"] != summary[
        "public_entered_rate_among_retweeters"
    ]["f"]


def test_summarize_new_table_topical_share_mean_ignores_missing():
    summary = rb.summarize_new_table(_users())
    # m 组唯一一行 public_topical_share 是 None，均值应为 NaN 而不是报错
    assert pd.isna(summary["public_topical_share_mean"]["m"])
    assert summary["public_topical_share_mean"]["f"] == pytest.approx((0.4 + 0.0 + 0.2) / 3)


def test_build_report_includes_baseline_when_present():
    report = rb.build_report(_users(), _overview(), _density(), year=2020)
    assert report["year"] == 2020
    assert "retweet_overview" in report["baseline"]
    assert "density_summary" in report["baseline"]
    assert report["baseline"]["retweet_overview"][0]["gender"] == "f"
    # 三个已知差异来源都必须在 notes 里留痕
    joined_notes = "\n".join(report["notes"])
    assert "分母" in joined_notes
    assert "表达帖" in joined_notes
    assert "嵌套词" in joined_notes


def test_build_report_labels_both_denominators_explicitly():
    """报告里必须显式标注两种转发进入率各自的分母，不能只靠字段名隐含"""
    report = rb.build_report(_users(), _overview(), _density(), year=2020)
    labels = report["new"]["retweet_ratio_denominators"]
    assert "entered_rate_all_users" in labels
    assert "entered_rate_among_retweeters" in labels
    assert "全部" in labels["entered_rate_all_users"]
    assert "转发者" in labels["entered_rate_among_retweeters"]


def test_build_report_missing_baseline_degrades_gracefully_not_crash():
    """基线文件缺失时不能抛异常，必须在 notes 里留下明确记录"""
    report = rb.build_report(
        _users(),
        None,
        None,
        year=2020,
        overview_path="viz_data/retweet_overview_2020.parquet",
        density_path="viz_data/density_summary_2020.parquet",
    )
    assert "retweet_overview" not in report["baseline"]
    assert "density_summary" not in report["baseline"]
    joined_notes = "\n".join(report["notes"])
    assert "retweet_overview_2020.parquet" in joined_notes
    assert "density_summary_2020.parquet" in joined_notes


def test_build_report_partial_baseline_missing_only_one_file():
    """只有一个基线文件缺失时，另一个仍要正常写入报告"""
    report = rb.build_report(
        _users(),
        _overview(),
        None,
        year=2020,
        overview_path="viz_data/retweet_overview_2020.parquet",
        density_path="viz_data/density_summary_2020.parquet",
    )
    assert "retweet_overview" in report["baseline"]
    assert "density_summary" not in report["baseline"]


def test_build_report_empty_user_table_does_not_crash():
    """空用户表(本地没有服务器产出的数据)不应报错，各项统计应为空字典"""
    empty_users = pd.DataFrame(
        {
            "gender": pd.Series([], dtype="object"),
            "n_retweets": pd.Series([], dtype="int64"),
            "public_source_entered": pd.Series([], dtype="bool"),
            "public_topical_share": pd.Series([], dtype="float64"),
            "celebrity_source_entered": pd.Series([], dtype="bool"),
            "celebrity_topical_share": pd.Series([], dtype="float64"),
        }
    )
    report = rb.build_report(empty_users, None, None, year=2020)
    assert report["new"]["users"] == {}
    assert report["new"]["public_source_entered"] == {}
    assert report["new"]["public_entered_rate_all_users"] == {}
    assert report["new"]["public_entered_rate_among_retweeters"] == {}


def test_reconcile_end_to_end_writes_json(tmp_path, monkeypatch):
    """整条 reconcile() 流程：从落盘的 parquet 读取，写出 JSON 报告"""
    output_dir = tmp_path / "analysis_data"
    viz_dir = tmp_path / "viz_data"
    output_dir.mkdir()
    viz_dir.mkdir()

    monkeypatch.setattr(config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(rb, "VIZ_DIR", str(viz_dir))

    _users().to_parquet(output_dir / "user_domain_2020.parquet", engine="pyarrow", index=False)
    _overview().to_parquet(viz_dir / "retweet_overview_2020.parquet", engine="pyarrow", index=False)
    # density baseline 缺失，验证端到端也能优雅降级

    report = rb.reconcile(year=2020)

    out_path = output_dir / "reconciliation_2020.json"
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["year"] == 2020
    assert "retweet_overview" in loaded["baseline"]
    assert "density_summary" not in loaded["baseline"]
    assert report["year"] == 2020


def test_reconcile_missing_user_table_raises_not_silently_empty(tmp_path, monkeypatch):
    """user_domain 表本身缺失应该报错，而不是被当成"空表"悄悄处理

    与基线文件不同：user_domain_{year}.parquet 是本次对照的主表，主表
    缺失意味着上游流水线没跑完，必须显式失败，不能返回一份看起来
    正常、实际全是空字典的报告。
    """
    output_dir = tmp_path / "analysis_data"
    output_dir.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(rb, "VIZ_DIR", str(tmp_path / "viz_data"))

    with pytest.raises(FileNotFoundError):
        rb.reconcile(year=2020)
