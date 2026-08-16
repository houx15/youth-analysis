import numpy as np
import pandas as pd
import pytest

from gender_domain import profile_join as pj


def _users():
    return pd.DataFrame({
        "user_id": ["1", "2", "3"],
        "gender": ["m", "f", "f"],
        "n_posts": [10, 20, 30],
    })


def _profiles():
    return pd.DataFrame({
        "user_id": [1, 2],                      # 注意：整数，且缺少用户 3
        "verified_type": ["0", "1"],
        "user_type": ["normal", "verified"],
        "fans_number": [100, 0],
        "friends_count": [50, -1],              # -1 为异常值
    })


def test_attach_keeps_every_user_even_without_profile():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    assert len(out) == 3
    assert set(out["user_id"]) == {"1", "2", "3"}


def test_missing_profile_marked_not_dropped():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    row = out.set_index("user_id").loc["3"]
    assert not row["profile_complete"]
    assert pd.isna(row["fans_number"])


def test_log_transforms_use_log1p():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    row = out.set_index("user_id").loc["1"]
    assert row["log_fans"] == pytest.approx(np.log1p(100))
    zero = out.set_index("user_id").loc["2"]
    assert zero["log_fans"] == pytest.approx(0.0)      # log1p(0) == 0，合法值


def test_negative_counts_become_nan_not_zero():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    row = out.set_index("user_id").loc["2"]
    assert pd.isna(row["friends_count"])
    assert pd.isna(row["log_friends"])
    assert not row["profile_complete"]


def test_integer_profile_ids_join_to_string_user_ids():
    out, _ = pj.attach_profile_controls(_users(), _profiles())
    assert out.set_index("user_id").loc["1", "user_type"] == "normal"


def test_loss_report_is_per_gender_and_complete():
    _, report = pj.attach_profile_controls(_users(), _profiles())
    assert report["users_total"] == 3
    assert report["by_gender"]["f"]["users_total"] == 2
    assert report["by_gender"]["f"]["profile_complete"] == 1
    assert report["by_gender"]["m"]["profile_complete"] == 1
