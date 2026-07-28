"""Unit tests for the S&P 500 / Nasdaq-100 universe+sector acquisition adapter.

Uses small, hand-crafted HTML tables shaped like the real Wikipedia pages
-- never a real network call.
"""

from datetime import datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.acquisition.universe import (
    build_universe_and_sector_records,
    fetch_nasdaq100_constituents,
    fetch_sp500_constituents,
    parse_nasdaq100_table,
    parse_sp500_table,
)
from atlas_quant.strategies.filing_momentum_ml.acquisition.universe import (
    NASDAQ100_WIKIPEDIA_URL,
    SP500_WIKIPEDIA_URL,
)

from tests.fixtures.acquisition import FakeHttpClient

_NOW = datetime(2024, 6, 1)

_SP500_HTML = """
<table>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
<tr><td>MMM</td><td>3M</td><td>Industrials</td><td>Industrial Conglomerates</td></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Technology Hardware</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td></tr>
</table>
"""

_NASDAQ100_HTML = """
<table>
<tr><th>Ticker</th><th>Company</th><th>ICB Industry[1]</th><th>ICB Subsector[1]</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Technology</td><td>Technology Hardware</td></tr>
<tr><td>ADBE</td><td>Adobe Inc.</td><td>Technology</td><td>Software</td></tr>
</table>
"""


def test_parse_sp500_table_normalizes_dotted_symbols():
    rows = parse_sp500_table(_SP500_HTML)
    by_symbol = dict(rows)
    assert by_symbol["MMM"] == "Industrials"
    assert by_symbol["BRK-B"] == "Financials"  # BRK.B -> BRK-B
    assert len(rows) == 3


def test_parse_nasdaq100_table():
    rows = parse_nasdaq100_table(_NASDAQ100_HTML)
    by_symbol = dict(rows)
    assert by_symbol["AAPL"] == "Technology"
    assert by_symbol["ADBE"] == "Technology"


def test_parse_sp500_table_missing_columns_raises():
    with pytest.raises(ValueError, match="missing expected columns"):
        parse_sp500_table("<table><tr><th>Foo</th><th>Bar</th></tr><tr><td>1</td><td>2</td></tr></table>")


def test_fetch_sp500_constituents_uses_correct_url():
    client = FakeHttpClient(text_responses={SP500_WIKIPEDIA_URL: _SP500_HTML})
    rows = fetch_sp500_constituents(client)
    assert len(rows) == 3
    assert client.requested_urls == [SP500_WIKIPEDIA_URL]


def test_fetch_nasdaq100_constituents_uses_correct_url():
    client = FakeHttpClient(text_responses={NASDAQ100_WIKIPEDIA_URL: _NASDAQ100_HTML})
    rows = fetch_nasdaq100_constituents(client)
    assert len(rows) == 2
    assert client.requested_urls == [NASDAQ100_WIKIPEDIA_URL]


def test_build_universe_and_sector_records_dedupes_and_prefers_sp500_sector():
    client = FakeHttpClient(text_responses={
        SP500_WIKIPEDIA_URL: _SP500_HTML,
        NASDAQ100_WIKIPEDIA_URL: _NASDAQ100_HTML,
    })
    universe, sectors = build_universe_and_sector_records(client, as_of=_NOW, retrieved_at=_NOW)

    symbols = {r.symbol for r in universe}
    assert symbols == {"MMM", "AAPL", "BRK-B", "ADBE"}  # AAPL appears in both, counted once
    assert all(r.survivorship_biased is True for r in universe)

    sector_by_symbol = {r.symbol: r.raw_sector for r in sectors}
    # AAPL is in both lists -- S&P 500's GICS classification ("Information
    # Technology") must win over Nasdaq-100's ICB classification ("Technology").
    assert sector_by_symbol["AAPL"] == "Information Technology"
    assert sector_by_symbol["ADBE"] == "Technology"  # only in the Nasdaq-100 table
    assert sector_by_symbol["BRK-B"] == "Financials"
