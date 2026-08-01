"""Unit tests for the Filing Momentum ML production-research CLI.

Every test calls ``main(argv, stdout=..., stderr=...)`` directly (never a
subprocess) so the suite stays fast and fully offline. Raw-data JSON
fixtures are written to ``tmp_path`` -- the CLI never touches a real
production or legacy path in this suite.
"""

import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from atlas_quant.strategies.filing_momentum_ml.production import orchestration as orchestration_module
from atlas_quant.cli.filing_momentum import main

from tests.fixtures.filing_momentum_ml import FakeEstimator


def _write_raw_data(root: Path, *, include_universe: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    quarter_ends = ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31"]
    filings = []
    revenue = 100.0
    for i, q in enumerate(quarter_ends):
        y, m, d = q.split("-")
        filed = f"{y}-{int(m):02d}-{int(d):02d}"
        filed_dt = date.fromisoformat(filed) + timedelta(days=30)
        filings.append({
            "symbol": "AAA", "asset_class": "equity", "fiscal_period": f"Q{(int(m)-1)//3+1}",
            "fiscal_year": int(y), "quarter_end": q, "filed_at": filed_dt.isoformat() + "T00:00:00",
            "revenue": revenue, "gross_profit": revenue * 0.4, "operating_income": revenue * 0.15,
            "net_income": revenue * 0.1, "diluted_eps": 1.0 + i * 0.05, "stockholders_equity": 500.0 + i * 10,
            "operating_cash_flow": 20.0, "capital_expenditure": 5.0, "accession_number": f"acc-{i}",
            "source": "fixture", "retrieved_at": "2023-06-01T00:00:00",
        })
        revenue += 10.0
    (root / "filings.json").write_text(json.dumps(filings))

    prices = []
    current = date(2019, 1, 1)
    price = 100.0
    symbols = ["AAA", "SPY", "VGT"]
    while current <= date(2023, 6, 30):
        if current.weekday() < 5:
            for symbol in symbols:
                prices.append({
                    "symbol": symbol, "asset_class": "equity", "trading_date": current.isoformat(),
                    "close": price, "price_convention": "split_dividend_adjusted",
                    "source": "fixture", "retrieved_at": "2023-06-01T00:00:00",
                })
            price *= 1.0003
        current += timedelta(days=1)
    (root / "prices.json").write_text(json.dumps(prices))

    universe = []
    if include_universe:
        universe = [{
            "symbol": "AAA", "asset_class": "equity", "as_of": "2023-01-01T00:00:00",
            "source": "fixture", "survivorship_biased": True, "retrieved_at": "2023-06-01T00:00:00",
        }]
    (root / "universe.json").write_text(json.dumps(universe))

    sic_history = [{
        "symbol": "AAA", "asset_class": "equity", "accession_number": "acc-0",
        "filed_at": "2023-01-01T00:00:00", "sic_code": 7372, "gics_sector": "Information Technology",
        "source": "fixture", "retrieved_at": "2023-06-01T00:00:00",
    }]
    (root / "sic_history.json").write_text(json.dumps(sic_history))


def _write_manifest(path: Path) -> None:
    from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest

    manifest = DataProvenanceManifest(
        dataset_identity_label="cli-test", provider_name="fixture", provider_version=None,
        retrieval_date=date(2023, 6, 1), data_cutoff=datetime(2023, 6, 1),
        universe_identity="cli-test-universe", universe_construction_method="fixture",
        survivorship_biased=True, filing_source="fixture", filing_point_in_time_status="fixture",
        price_source="fixture", price_convention="split_dividend_adjusted", sector_source="fixture",
        sector_override_identity="none", trading_calendar_source="fixture",
        coverage_start=date(2019, 1, 1), coverage_end=date(2023, 6, 30),
        row_counts={}, missing_data_summary={}, duplicate_summary={},
        corporate_action_treatment="none", delisting_treatment="none", data_corrections=(),
        source_file_hashes={}, strategy_config_identity="x", git_commit=None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict()))


def _run(argv, **kwargs):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = main(argv, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_validate_data_reports_info_only_and_exits_zero(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    code, out, err = _run([
        "filing-momentum", "validate-data", "--raw-root", str(raw_root),
        "--start-quarter", "2023-03-31", "--end-quarter", "2023-03-31",
    ])
    assert code == 0
    assert "fatal=0" in out


def test_validate_data_empty_universe_is_fatal(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root, include_universe=False)
    code, out, err = _run(["filing-momentum", "validate-data", "--raw-root", str(raw_root)])
    assert code == 2
    assert "fatal=1" in out


def test_build_features_dry_run_writes_nothing(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    code, out, err = _run([
        "filing-momentum", "build-features", "--raw-root", str(raw_root),
        "--start-quarter", "2023-03-31", "--end-quarter", "2023-03-31", "--dry-run",
    ])
    assert code == 0
    assert "built 1 feature observation(s)" in out
    assert not (tmp_path / "cache").exists()


def test_build_features_writes_cache_and_refuses_overwrite(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    cache_root = tmp_path / "cache"
    argv = [
        "filing-momentum", "build-features", "--raw-root", str(raw_root),
        "--start-quarter", "2023-03-31", "--end-quarter", "2023-03-31", "--cache-root", str(cache_root),
    ]
    code, out, err = _run(argv)
    assert code == 0
    assert cache_root.exists()

    code, out, err = _run(argv)
    assert code == 1
    assert "already exists" in err

    code, out, err = _run(argv + ["--overwrite"])
    assert code == 0


def test_build_labels_writes_output_and_refuses_overwrite(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    output_root = tmp_path / "outputs"
    argv = [
        "filing-momentum", "build-labels", "--raw-root", str(raw_root),
        "--start-quarter", "2023-03-31", "--end-quarter", "2023-03-31", "--output-root", str(output_root),
    ]
    code, out, err = _run(argv)
    assert code == 0
    labels_path = output_root / "labels.json"
    assert labels_path.exists()
    data = json.loads(labels_path.read_text())
    assert "2023-03-31" in data

    code, out, err = _run(argv)
    assert code == 1
    assert "already exists" in err

    code, out, err = _run(argv + ["--overwrite"])
    assert code == 0


def test_run_backtest_blocked_missing_dependency(monkeypatch, tmp_path):
    import atlas_quant.strategies.filing_momentum_ml.production.orchestration as orchestration_module
    from atlas_quant.dependency_status import DependencyAvailability, DependencyCategory, DependencyStatus

    monkeypatch.setattr(
        orchestration_module, "missing_required_for_production",
        lambda report: (
            DependencyStatus(
                "scikit-learn", DependencyCategory.PRODUCTION_DATA,
                DependencyAvailability.MISSING_REQUIRED_FOR_PRODUCTION_BACKTEST, None, "1.3.0",
                detail="module 'sklearn' not found",
            ),
        ),
    )
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    code, out, err = _run([
        "filing-momentum", "run-backtest", "--raw-root", str(raw_root), "--manifest", str(manifest_path),
        "--start-quarter", "2023-03-31", "--end-quarter", "2023-03-31", "--dry-run",
    ])
    assert code == 3
    assert "blocked_missing_dependency" in out
    assert "scikit-learn" in out


def _fake_estimator_factory(model_config):
    from atlas_quant.strategies.filing_momentum_ml.estimator import EstimatorBuildInfo

    return FakeEstimator(), EstimatorBuildInfo(estimator_type="FakeEstimator", parameters={}, library="test", library_version=None)


def test_build_report_and_compare_report_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestration_module, "missing_required_for_production", lambda report: ())
    monkeypatch.setattr(orchestration_module, "build_hgbc_estimator", _fake_estimator_factory)

    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    output_root = tmp_path / "outputs"

    code, out, err = _run([
        "filing-momentum", "build-report", "--raw-root", str(raw_root), "--manifest", str(manifest_path),
        "--start-quarter", "2023-03-31", "--end-quarter", "2023-03-31", "--output-root", str(output_root),
        "--checkpoint-root", str(tmp_path / "checkpoints"),
    ])
    assert code == 0
    written = list(output_root.glob("*.json"))
    assert written
    report_path = written[0]

    code, out, err = _run(["filing-momentum", "compare-report", "--report-json", str(report_path)])
    assert code == 0
    assert "no comparison records" in out


def test_run_all_stops_at_first_blocked_step(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root, include_universe=False)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    output_root = tmp_path / "outputs"
    code, out, err = _run([
        "filing-momentum", "run-all", "--raw-root", str(raw_root), "--manifest", str(manifest_path),
        "--start-quarter", "2023-03-31", "--end-quarter", "2023-03-31", "--output-root", str(output_root),
        "--dry-run",
    ])
    assert code == 2
    assert not output_root.exists()


class _FakeLivePriceProvider:
    """Deterministic, injectable live-price provider -- no network."""

    def __init__(self, prices: dict[str, float], *, as_of: date):
        self._prices = prices
        self._as_of = as_of

    def fetch_recent_history(self, symbol: str):
        import pandas as pd

        if symbol not in self._prices:
            return pd.DataFrame({"Close": []})
        return pd.DataFrame({"Close": [self._prices[symbol]]}, index=[pd.Timestamp(self._as_of)])


def test_current_status_runs_end_to_end_and_never_touches_network(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestration_module, "missing_required_for_production", lambda report: ())
    monkeypatch.setattr(orchestration_module, "build_hgbc_estimator", _fake_estimator_factory)
    monkeypatch.setattr(
        orchestration_module, "YFinanceLivePriceProvider",
        lambda: _FakeLivePriceProvider({"AAA": 111.0, "SPY": 222.0}, as_of=date(2023, 6, 1)),
    )

    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    model_cache_root = tmp_path / "models"
    decision_log_root = tmp_path / "decisions"

    code, out, err = _run([
        "filing-momentum", "current-status", "--raw-root", str(raw_root), "--manifest", str(manifest_path),
        "--start-quarter", "2022-03-31", "--as-of", "2023-06-01T00:00:00",
        "--model-cache-root", str(model_cache_root), "--decision-log-root", str(decision_log_root),
    ])
    assert code == 0
    assert "state: completed" in out
    assert "held cohort:" in out
    assert "next scheduled cohort:" in out

    code, out, err = _run([
        "filing-momentum", "current-status", "--raw-root", str(raw_root), "--manifest", str(manifest_path),
        "--start-quarter", "2022-03-31", "--as-of", "2023-06-01T00:00:00", "--json",
        "--model-cache-root", str(model_cache_root), "--decision-log-root", str(decision_log_root),
    ])
    assert code == 0
    payload = json.loads(out)
    assert payload["state"] == "completed"
    assert payload["held_quarter_end"] == "2023-03-31"
    assert payload["next_quarter_end"] == "2023-06-30"


def test_cli_never_imports_network_or_legacy_access():
    """``acquire-data`` is this CLI's one deliberate, disclosed exception
    for real network access (SEC EDGAR/Wikipedia/yfinance, via the
    acquisition package) -- every other subcommand remains network-free.
    This test verifies structurally that this module itself never imports
    subprocess/urllib/requests/yfinance directly (only via the acquisition
    package's own lazy-import boundary), and never hardcodes a reference
    to the legacy repository's path anywhere outside the one docstring
    disclosure sentence."""
    import atlas_quant.cli.filing_momentum as cli_module

    source = Path(cli_module.__file__).read_text()
    for forbidden_import in ("import subprocess", "import requests", "import urllib", "import yfinance"):
        assert forbidden_import not in source
    # The one mention is the module docstring's own disclosure sentence --
    # never a path this module actually opens or constructs.
    assert source.count("Arnold_Quant") == 1


def test_validate_data_json_output_is_well_formed(tmp_path):
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    code, out, err = _run(["filing-momentum", "validate-data", "--raw-root", str(raw_root), "--json"])
    assert code == 0
    payload = json.loads(out)
    assert "counts" in payload and "issues" in payload


def _fake_acquisition_result():
    import atlas_quant.cli.filing_momentum as cli_module
    from atlas_quant.strategies.filing_momentum_ml.acquisition.run_acquisition import AcquisitionResult
    from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawPriceRecord, RawUniverseRecord

    now = datetime(2024, 6, 1)
    return AcquisitionResult(
        filings=(), prices=(RawPriceRecord("AAA", "equity", date(2024, 1, 2), 100.0, "split_dividend_adjusted", "yfinance", now),),
        universe=(RawUniverseRecord("AAA", "equity", now, "wikipedia_sp500_nasdaq100", True, now),),
        symbols_attempted=1, symbols_with_filings=0, symbols_with_prices=1,
        warnings=("AAA: no SEC filings acquired (no CIK match or no quarterly facts found)",),
    )


def test_acquire_data_requires_sec_user_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    code, out, err = _run([
        "filing-momentum", "acquire-data", "--raw-root", str(tmp_path / "raw"), "--dry-run",
    ])
    assert code == 1
    assert "User-Agent" in err


def test_acquire_data_dry_run_writes_nothing(monkeypatch, tmp_path):
    import atlas_quant.cli.filing_momentum as cli_module

    monkeypatch.setattr(cli_module, "run_full_acquisition", lambda *a, **k: _fake_acquisition_result())
    raw_root = tmp_path / "raw"
    code, out, err = _run([
        "filing-momentum", "acquire-data", "--raw-root", str(raw_root),
        "--sec-user-agent", "Test test@example.com", "--dry-run",
    ])
    assert code == 0
    assert "acquired 0 filing row(s), 1 price row(s)" in out
    assert not raw_root.exists()


def test_acquire_data_writes_raw_files_and_manifest(monkeypatch, tmp_path):
    import atlas_quant.cli.filing_momentum as cli_module

    monkeypatch.setattr(cli_module, "run_full_acquisition", lambda *a, **k: _fake_acquisition_result())
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"
    code, out, err = _run([
        "filing-momentum", "acquire-data", "--raw-root", str(raw_root), "--manifest", str(manifest_path),
        "--sec-user-agent", "Test test@example.com",
    ])
    assert code == 0
    assert (raw_root / "prices.json").exists()
    assert (raw_root / "filings.json").exists()
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text())
    assert manifest_data["provider_name"] == "sec_edgar+yfinance+wikipedia"


def _fake_sic_history_result():
    from atlas_quant.strategies.filing_momentum_ml.acquisition.sic_history import SicHistoryResult
    from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawSicHistoryRecord

    now = datetime(2026, 7, 29)
    return SicHistoryResult(
        records=(
            RawSicHistoryRecord(
                symbol="AAA", asset_class="equity", accession_number="acc-0", filed_at=now,
                sic_code=7372, gics_sector="Information Technology",
                source="sec_edgar_sic_header", retrieved_at=now,
            ),
        ),
        pairs_attempted=1, pairs_with_sic=1, warnings=(),
    )


def test_acquire_sic_history_requires_sec_user_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    code, out, err = _run([
        "filing-momentum", "acquire-sic-history", "--raw-root", str(tmp_path / "raw"), "--dry-run",
    ])
    assert code == 1
    assert "User-Agent" in err


def test_acquire_sic_history_requires_existing_filings(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True)
    code, out, err = _run([
        "filing-momentum", "acquire-sic-history", "--raw-root", str(raw_root),
        "--sec-user-agent", "Test test@example.com", "--dry-run",
    ])
    assert code == 1
    assert "filings.json" in err


def _write_filings_only(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "filings.json").write_text(json.dumps([{
        "symbol": "AAA", "asset_class": "equity", "fiscal_period": "Q1", "fiscal_year": 2023,
        "quarter_end": "2023-03-31", "filed_at": "2023-04-30T00:00:00", "revenue": 100.0,
        "gross_profit": None, "operating_income": None, "net_income": None, "diluted_eps": None,
        "stockholders_equity": None, "operating_cash_flow": None, "capital_expenditure": None,
        "accession_number": "acc-0", "source": "fixture", "retrieved_at": "2023-06-01T00:00:00",
    }]))


def test_acquire_sic_history_dry_run_writes_nothing(monkeypatch, tmp_path):
    import atlas_quant.cli.filing_momentum as cli_module

    monkeypatch.setattr(cli_module, "fetch_ticker_to_cik_map", lambda *a, **k: {"AAA": "0000000001"})
    monkeypatch.setattr(cli_module, "fetch_sic_history", lambda *a, **k: _fake_sic_history_result())
    raw_root = tmp_path / "raw"
    _write_filings_only(raw_root)
    code, out, err = _run([
        "filing-momentum", "acquire-sic-history", "--raw-root", str(raw_root),
        "--sec-user-agent", "Test test@example.com", "--dry-run",
    ])
    assert code == 0
    assert "acquired SIC for 1/1" in out
    assert not (raw_root / "sic_history.json").exists()


def test_acquire_sic_history_writes_file(monkeypatch, tmp_path):
    import atlas_quant.cli.filing_momentum as cli_module

    monkeypatch.setattr(cli_module, "fetch_ticker_to_cik_map", lambda *a, **k: {"AAA": "0000000001"})
    monkeypatch.setattr(cli_module, "fetch_sic_history", lambda *a, **k: _fake_sic_history_result())
    raw_root = tmp_path / "raw"
    _write_filings_only(raw_root)
    code, out, err = _run([
        "filing-momentum", "acquire-sic-history", "--raw-root", str(raw_root),
        "--sec-user-agent", "Test test@example.com",
    ])
    assert code == 0
    written = json.loads((raw_root / "sic_history.json").read_text())
    assert len(written) == 1
    assert written[0]["gics_sector"] == "Information Technology"


def test_acquire_sic_history_refuses_overwrite_without_flag(monkeypatch, tmp_path):
    import atlas_quant.cli.filing_momentum as cli_module

    monkeypatch.setattr(cli_module, "fetch_ticker_to_cik_map", lambda *a, **k: {"AAA": "0000000001"})
    monkeypatch.setattr(cli_module, "fetch_sic_history", lambda *a, **k: _fake_sic_history_result())
    raw_root = tmp_path / "raw"
    _write_raw_data(raw_root)
    (raw_root / "sic_history.json").write_text("[]")
    code, out, err = _run([
        "filing-momentum", "acquire-sic-history", "--raw-root", str(raw_root),
        "--sec-user-agent", "Test test@example.com",
    ])
    assert code == 1
    assert "overwrite" in err


def test_acquire_data_refuses_overwrite_without_flag(monkeypatch, tmp_path):
    import atlas_quant.cli.filing_momentum as cli_module

    monkeypatch.setattr(cli_module, "run_full_acquisition", lambda *a, **k: _fake_acquisition_result())
    raw_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"
    argv = [
        "filing-momentum", "acquire-data", "--raw-root", str(raw_root), "--manifest", str(manifest_path),
        "--sec-user-agent", "Test test@example.com",
    ]
    code, out, err = _run(argv)
    assert code == 0

    code, out, err = _run(argv)
    assert code == 1
    assert "already exists" in err

    code, out, err = _run(argv + ["--overwrite"])
    assert code == 0
