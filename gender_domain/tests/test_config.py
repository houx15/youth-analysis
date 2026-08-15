import json
import os

from gender_domain import config


def test_fingerprint_is_order_and_whitespace_insensitive():
    a = config.fingerprint_terms(["北京", "上海", "广州"])
    b = config.fingerprint_terms([" 广州 ", "北京", "上海\n"])
    assert a == b
    assert len(a) == 8


def test_fingerprint_changes_when_a_term_changes():
    a = config.fingerprint_terms(["北京", "上海"])
    b = config.fingerprint_terms(["北京", "深圳"])
    assert a != b


def test_load_public_vocabulary_reads_configs_dir():
    terms = config.load_public_vocabulary(2020)
    assert len(terms) == 816
    assert all(t == t.strip() and t for t in terms)


def test_load_celebrity_vocabulary_reads_configs_dir():
    terms = config.load_celebrity_vocabulary(2020)
    assert len(terms) == 535


def test_write_manifest_roundtrip(tmp_path):
    manifest = config.build_manifest(
        step="test", inputs=["a.parquet"], params={"year": 2020}, counts={"rows": 5}
    )
    path = config.write_manifest(manifest, str(tmp_path))
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["step"] == "test"
    assert loaded["counts"]["rows"] == 5
    assert "git_sha" in loaded
    assert "created_at" in loaded


# ---------------------------------------------------------------------------
# Fix round 2：git 溯源信息不能静默失败，也不能对脏工作区一言不发
# ---------------------------------------------------------------------------


def test_manifest_records_whether_working_tree_was_dirty():
    """git_sha 是整份 manifest 唯一的代码溯源证据，必须同时记录脏/干净

    没有 git_dirty 的话，改了代码没提交就跑出来的一批数字，和干净工作区
    跑出来的 manifest 长得一模一样，"这批数字对应哪版代码"就无从判断。
    """
    manifest = config.build_manifest(step="test")
    assert "git_dirty" in manifest
    assert manifest["git_dirty"] in (True, False, "unknown")


def test_git_revision_warns_in_chinese_and_returns_unknown_when_git_fails(
    monkeypatch, capsys
):
    """取不到 git 版本号时必须打印中文警告，而不是静默写一个 unknown"""

    def _boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(config.subprocess, "check_output", _boom)
    sha, dirty = config._git_revision()
    assert sha == "unknown"
    assert dirty == "unknown"
    captured = capsys.readouterr().out
    assert "警告" in captured
    assert "git_sha" in captured


def test_git_revision_reports_dirty_working_tree(monkeypatch, capsys):
    """工作区有未提交改动时，git_dirty 必须为 True 并给出中文警告"""
    calls = []

    def _fake_check_output(args, **kwargs):
        calls.append(args)
        if "rev-parse" in args:
            return b"abc1234\n"
        return b" M gender_domain/config.py\n"

    monkeypatch.setattr(config.subprocess, "check_output", _fake_check_output)
    sha, dirty = config._git_revision()
    assert sha == "abc1234"
    assert dirty is True
    assert "警告" in capsys.readouterr().out


def test_git_revision_reports_clean_working_tree(monkeypatch):
    def _fake_check_output(args, **kwargs):
        if "rev-parse" in args:
            return b"abc1234\n"
        return b"\n"

    monkeypatch.setattr(config.subprocess, "check_output", _fake_check_output)
    sha, dirty = config._git_revision()
    assert sha == "abc1234"
    assert dirty is False
