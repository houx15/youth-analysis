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
