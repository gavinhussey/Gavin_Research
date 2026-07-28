"""Unit tests for the diagnostic-only, read-only legacy artifact audit.

Uses a throwaway ``tmp_path`` tree that mimics the legacy repository's
shape -- never touches the real ``Arnold_Quant`` checkout in this suite.
"""

import json

from atlas_quant.strategies.filing_momentum_ml.production.legacy_audit import (
    DEFAULT_LEGACY_ARTIFACT_PATHS,
    LegacyArtifactClassification,
    audit_legacy_artifact,
    audit_legacy_repository,
)


def test_missing_artifact_is_unknown_provenance(tmp_path):
    audit = audit_legacy_artifact(tmp_path, "does_not_exist.pkl")
    assert audit.exists is False
    assert audit.classification == LegacyArtifactClassification.UNKNOWN_PROVENANCE
    assert audit.sha256 is None


def test_pickle_is_diagnostic_only_and_never_deserialized(tmp_path):
    path = tmp_path / "cache.pkl"
    path.write_bytes(b"\x80\x04not-really-valid-pickle-but-never-loaded")
    audit = audit_legacy_artifact(tmp_path, "cache.pkl")
    assert audit.file_format == "pickle"
    assert audit.classification == LegacyArtifactClassification.DIAGNOSTIC_ONLY
    assert audit.sha256 is not None
    assert "never deserialized" in audit.reason


def test_valid_json_is_partially_verified(tmp_path):
    path = tmp_path / "universe_cache.json"
    path.write_text(json.dumps({"tickers": ["AAPL", "MSFT"]}))
    audit = audit_legacy_artifact(tmp_path, "universe_cache.json")
    assert audit.file_format == "json"
    assert audit.classification == LegacyArtifactClassification.PARTIALLY_VERIFIED


def test_malformed_json_is_incompatible(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    audit = audit_legacy_artifact(tmp_path, "broken.json")
    assert audit.classification == LegacyArtifactClassification.INCOMPATIBLE


def test_parquet_is_diagnostic_only(tmp_path):
    path = tmp_path / "price_cache.parquet"
    path.write_bytes(b"PAR1fakecontent")
    audit = audit_legacy_artifact(tmp_path, "price_cache.parquet")
    assert audit.file_format == "parquet"
    assert audit.classification == LegacyArtifactClassification.DIAGNOSTIC_ONLY


def test_html_report_is_diagnostic_only(tmp_path):
    path = tmp_path / "report.html"
    path.write_text("<html><body>legacy report</body></html>")
    audit = audit_legacy_artifact(tmp_path, "report.html")
    assert audit.file_format == "html"
    assert audit.classification == LegacyArtifactClassification.DIAGNOSTIC_ONLY
    assert "report_current.html" in audit.reason


def test_directory_is_diagnostic_only_with_file_count(tmp_path):
    directory = tmp_path / "edgar_cache"
    directory.mkdir()
    (directory / "a.json").write_text("{}")
    (directory / "b.json").write_text("{}")
    audit = audit_legacy_artifact(tmp_path, "edgar_cache")
    assert audit.file_format == "directory"
    assert audit.classification == LegacyArtifactClassification.DIAGNOSTIC_ONLY
    assert "2 file(s)" in audit.reason


def test_audit_legacy_repository_covers_every_default_path(tmp_path):
    results = audit_legacy_repository(tmp_path)
    assert [r.relative_path for r in results] == list(DEFAULT_LEGACY_ARTIFACT_PATHS)
    assert all(r.exists is False for r in results)


def test_audit_never_writes_to_legacy_root(tmp_path):
    path = tmp_path / "data_cache.json"
    path.write_text(json.dumps({"a": 1}))
    before = {p.name: p.stat().st_mtime for p in tmp_path.iterdir()}
    audit_legacy_repository(tmp_path)
    after = {p.name: p.stat().st_mtime for p in tmp_path.iterdir()}
    assert before == after
    assert set(tmp_path.iterdir()) == {path}
