"""Unit tests for the top-level acquisition orchestration.

Every network-shaped call is a FakeHttpClient/FakeHistoryProvider --
never real network I/O. Verifies per-symbol failure isolation, the
written JSON round-trips through the CLI's own raw-data loader, and the
manifest's derived fields.
"""

from datetime import date, datetime

import pandas as pd
import pytest

from atlas_quant.strategies.filing_momentum_ml.acquisition.run_acquisition import (
    build_acquisition_manifest,
    run_full_acquisition,
    write_raw_data_files,
)
from atlas_quant.strategies.filing_momentum_ml.acquisition.universe import (
    NASDAQ100_WIKIPEDIA_URL,
    SP500_WIKIPEDIA_URL,
)

from tests.fixtures.acquisition import FakeHistoryProvider, FakeHttpClient

_RETRIEVED_AT = datetime(2024, 6, 1)

_SP500_HTML = """
<table>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
<tr><td>AAA</td><td>Alpha Co</td><td>Industrials</td><td>X</td></tr>
<tr><td>BBB</td><td>Beta Co</td><td>Financials</td><td>Y</td></tr>
</table>
"""

_NASDAQ100_HTML = """
<table>
<tr><th>Ticker</th><th>Company</th><th>ICB Industry[1]</th><th>ICB Subsector[1]</th></tr>
<tr><td>AAA</td><td>Alpha Co</td><td>Technology</td><td>Z</td></tr>
</table>
"""

_TICKER_MAP_JSON = {
    "0": {"cik_str": 1, "ticker": "AAA"},
    # BBB deliberately absent -- exercises the "no CIK match" path.
}

_FACTS_JSON = {
    "cik": 1,
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {"USD": [
                    {"start": "2023-01-01", "end": "2023-03-31", "val": 100.0, "accn": "acc-1",
                     "fy": 2023, "fp": "Q1", "form": "10-Q", "filed": "2023-04-28"},
                ]}
            }
        }
    },
}


def _history_df(rows):
    index = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    return pd.DataFrame({"Close": [r[1] for r in rows]}, index=index)


def _client():
    return FakeHttpClient(
        text_responses={SP500_WIKIPEDIA_URL: _SP500_HTML, NASDAQ100_WIKIPEDIA_URL: _NASDAQ100_HTML},
        json_responses={
            "https://www.sec.gov/files/company_tickers.json": _TICKER_MAP_JSON,
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": _FACTS_JSON,
        },
    )


def _provider():
    return FakeHistoryProvider({
        "AAA": _history_df([("2023-01-03", 100.0), ("2023-01-04", 101.0)]),
        "BBB": _history_df([("2023-01-03", 50.0)]),
    })


def test_run_full_acquisition_happy_path():
    result = run_full_acquisition(
        _client(), _provider(), sec_user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT,
    )
    assert result.symbols_attempted == 2
    assert result.symbols_with_prices == 2
    assert result.symbols_with_filings == 1  # only AAA has a CIK
    assert len(result.filings) == 1
    assert len(result.prices) == 3
    assert len(result.universe) == 2
    assert len(result.sectors) == 2
    assert any("BBB" in w and "no SEC filings" in w for w in result.warnings)


def test_run_full_acquisition_isolates_per_symbol_price_failure():
    provider = FakeHistoryProvider({"AAA": _history_df([("2023-01-03", 100.0)])})  # BBB deliberately missing
    result = run_full_acquisition(
        _client(), provider, sec_user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT,
    )
    assert result.symbols_with_prices == 1
    assert any("BBB" in w and "yfinance fetch failed" in w for w in result.warnings)
    # AAA's own data is unaffected by BBB's failure.
    assert len(result.prices) == 1


def test_symbol_limit_filters_universe_and_sectors_consistently():
    result = run_full_acquisition(
        _client(), _provider(), sec_user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT, symbol_limit=1,
    )
    assert result.symbols_attempted == 1
    assert {r.symbol for r in result.universe} == {"AAA"}
    assert {r.symbol for r in result.sectors} == {"AAA"}


def test_progress_callback_invoked_per_symbol():
    calls = []
    run_full_acquisition(
        _client(), _provider(), sec_user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT,
        progress_callback=lambda symbol, i, total: calls.append((symbol, i, total)),
    )
    assert calls == [("AAA", 1, 2), ("BBB", 2, 2)]


def test_write_raw_data_files_round_trips_through_cli_loader(tmp_path):
    from atlas_quant.cli.filing_momentum import load_normalized_bundle

    result = run_full_acquisition(
        _client(), _provider(), sec_user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT,
    )
    written = write_raw_data_files(result, tmp_path)
    assert set(written) == {"filings.json", "prices.json", "universe.json", "sectors.json"}

    bundle = load_normalized_bundle(tmp_path)
    assert bundle.issues == ()
    assert len(bundle.universe) == 2
    assert len(bundle.prices_by_instrument) == 2


def test_build_acquisition_manifest_derives_coverage_from_prices():
    result = run_full_acquisition(
        _client(), _provider(), sec_user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT,
    )
    manifest = build_acquisition_manifest(
        result, dataset_identity_label="test-run", strategy_config_identity="s1",
        retrieval_date=date(2024, 6, 1), data_cutoff=_RETRIEVED_AT, git_commit="abc123",
    )
    assert manifest.coverage_start == date(2023, 1, 3)
    assert manifest.coverage_end == date(2023, 1, 4)
    assert manifest.row_counts["filings"] == 1
    assert manifest.row_counts["prices"] == 3
    assert manifest.survivorship_biased is True
    assert manifest.git_commit == "abc123"
