"""Unit tests for the generic atlas_quant.reporting foundation."""

import json
from pathlib import Path

import pytest

from atlas_quant.reporting.domain import (
    ArtifactIdentity,
    ComparisonRecord,
    ComparisonStatus,
    ReportMetadata,
    ReproducibilityStatus,
)
from atlas_quant.reporting.html import escape, wrap_page
from atlas_quant.reporting.serialization import (
    ArtifactExistsError,
    write_json_atomic,
    write_text_atomic,
)
from atlas_quant.reporting.validation import check, summarize


class TestReportMetadata:
    def test_construction(self):
        metadata = ReportMetadata(
            atlasquant_display_name="AtlasQuant", strategy_id="filing_momentum_ml",
            strategy_display_name="Filing Momentum ML", strategy_version="0.1.0",
            report_schema_version="1", config_identity="a" * 64, backtest_run_identity="b" * 64,
            performance_analysis_identity="c" * 64, report_identity="d" * 64,
            provenance_notes=("note",), warnings=(), reproducibility_status=ReproducibilityStatus.NOT_RUN,
        )
        assert metadata.strategy_id == "filing_momentum_ml"

    def test_no_secret_fields_exist(self):
        # ReportMetadata's declared fields must never include anything
        # secret-shaped (credentials, tokens, API keys).
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ReportMetadata)}
        forbidden = {"api_key", "token", "secret", "password", "credential"}
        assert not (field_names & forbidden)


class TestArtifactIdentity:
    def test_json_identity_deterministic(self):
        a = ArtifactIdentity.of_json('{"a": 1}')
        b = ArtifactIdentity.of_json('{"a": 1}')
        assert a.content_identity == b.content_identity

    def test_different_content_different_identity(self):
        a = ArtifactIdentity.of_json('{"a": 1}')
        b = ArtifactIdentity.of_json('{"a": 2}')
        assert a.content_identity != b.content_identity


class TestValidation:
    def test_summarize_all_passed(self):
        checks = [check("a", True, "ok"), check("b", True, "ok")]
        summary = summarize(checks)
        assert summary.all_passed
        assert summary.failed_checks == ()

    def test_summarize_with_failure(self):
        checks = [check("a", True, "ok"), check("b", False, "not ok")]
        summary = summarize(checks)
        assert not summary.all_passed
        assert len(summary.failed_checks) == 1


class TestComparisonRecord:
    def test_construction(self):
        record = ComparisonRecord(
            key="sharpe", source_value=0.99, atlasquant_value=1.0, absolute_difference=0.01,
            relative_difference=0.0101, tolerance=0.05, status=ComparisonStatus.WITHIN_TOLERANCE,
            explanation="test", provenance="test",
        )
        assert record.status == ComparisonStatus.WITHIN_TOLERANCE


class TestHtmlEscaping:
    def test_escapes_script_tags(self):
        assert "<script>" not in escape("<script>alert(1)</script>")

    def test_escapes_quotes(self):
        result = escape("it's a \"test\"")
        assert "'" not in result.replace("&#x27;", "")
        assert "&#x27;" in result or "&#39;" in result

    def test_escapes_ampersand(self):
        assert escape("A & B") == "A &amp; B"

    def test_non_string_input_converted_and_escaped(self):
        assert escape(None) == "None"
        assert escape(3.14) == "3.14"


class TestWrapPage:
    def test_title_is_escaped(self):
        html = wrap_page("<script>bad</script>", "<p>body</p>")
        assert "<script>bad</script>" not in html
        assert "&lt;script&gt;" in html

    def test_no_external_cdn_reference(self):
        html = wrap_page("Title", "<p>body</p>")
        assert "http://" not in html
        assert "https://" not in html

    def test_self_contained_structure(self):
        html = wrap_page("Title", "<p>body</p>")
        assert html.startswith("<!doctype html>")
        assert "<style>" in html


class TestAtomicWrites:
    def test_write_text_atomic(self, tmp_path):
        path = write_text_atomic(tmp_path / "out.txt", "hello")
        assert path.read_text() == "hello"

    def test_write_json_atomic_deterministic(self, tmp_path):
        path = write_json_atomic(tmp_path / "out.json", {"b": 1, "a": 2})
        content = path.read_text()
        assert content.index('"a"') < content.index('"b"')  # sorted keys
        assert json.loads(content) == {"a": 2, "b": 1}

    def test_overwrite_disabled_raises_on_existing(self, tmp_path):
        path = tmp_path / "out.txt"
        write_text_atomic(path, "first")
        with pytest.raises(ArtifactExistsError):
            write_text_atomic(path, "second", overwrite=False)
        assert path.read_text() == "first"

    def test_overwrite_enabled_replaces_content(self, tmp_path):
        path = tmp_path / "out.txt"
        write_text_atomic(path, "first")
        write_text_atomic(path, "second", overwrite=True)
        assert path.read_text() == "second"

    def test_no_leftover_temp_files(self, tmp_path):
        write_text_atomic(tmp_path / "out.txt", "content")
        leftovers = list(tmp_path.glob("*.tmp*"))
        assert leftovers == []

    def test_creates_parent_directories(self, tmp_path):
        path = write_text_atomic(tmp_path / "nested" / "dir" / "out.txt", "content")
        assert path.exists()
