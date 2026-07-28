"""Unit tests for the typed, deterministic data-provenance manifest."""

from datetime import date, datetime, timezone

import pytest

from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest


def _manifest(**overrides) -> DataProvenanceManifest:
    defaults = dict(
        dataset_identity_label="2024Q1-research-snapshot",
        provider_name="sec_edgar",
        provider_version="v1",
        retrieval_date=date(2024, 5, 1),
        data_cutoff=datetime(2024, 5, 1, tzinfo=timezone.utc),
        universe_identity="sp500+ndx100-snapshot-2024-05-01",
        universe_construction_method="present_day_snapshot_applied_retroactively",
        survivorship_biased=True,
        filing_source="sec_edgar",
        filing_point_in_time_status="filed_at_verified",
        price_source="polygon",
        price_convention="split_dividend_adjusted",
        sector_source="gics",
        sector_override_identity="none",
        trading_calendar_source="nyse",
        coverage_start=date(2015, 1, 1),
        coverage_end=date(2024, 1, 1),
        row_counts={"filings": 400, "prices": 100_000},
        missing_data_summary={"filings": 0},
        duplicate_summary={"filings": 0},
        corporate_action_treatment="split_dividend_adjusted_close",
        delisting_treatment="excluded_after_delisting",
        data_corrections=(),
        source_file_hashes={"filings.json": "abc123"},
        strategy_config_identity="strategy-config-hash",
        git_commit="deadbeef",
        notes=(),
    )
    defaults.update(overrides)
    return DataProvenanceManifest(**defaults)


def test_valid_manifest_constructs():
    manifest = _manifest()
    assert manifest.dataset_identity_label == "2024Q1-research-snapshot"


def test_empty_label_rejected():
    with pytest.raises(ValueError, match="dataset_identity_label"):
        _manifest(dataset_identity_label="")


def test_coverage_end_before_start_rejected():
    with pytest.raises(ValueError, match="coverage_end"):
        _manifest(coverage_start=date(2024, 1, 1), coverage_end=date(2020, 1, 1))


def test_identity_is_deterministic():
    first = _manifest().identity()
    second = _manifest().identity()
    assert first == second


def test_identity_changes_with_row_counts():
    base = _manifest().identity()
    changed = _manifest(row_counts={"filings": 401, "prices": 100_000}).identity()
    assert base != changed


def test_to_dict_from_dict_round_trips():
    manifest = _manifest()
    round_tripped = DataProvenanceManifest.from_dict(manifest.to_dict())
    assert round_tripped == manifest
    assert round_tripped.identity() == manifest.identity()


def test_identity_insensitive_to_row_counts_dict_ordering():
    a = _manifest(row_counts={"filings": 400, "prices": 100_000}).identity()
    b = _manifest(row_counts={"prices": 100_000, "filings": 400}).identity()
    assert a == b
