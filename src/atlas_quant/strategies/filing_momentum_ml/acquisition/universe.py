"""S&P 500 + Nasdaq 100 universe acquisition.

Uses each index's own Wikipedia page (free, no key) -- the standard,
widely-used community-maintained source for current constituents. This
is, by construction, a **present-day snapshot applied retroactively**:
report §5.4's own documented, survivorship-biased approach, not a defect
this module hides.

Sector data does *not* come from here (or from Wikipedia at all) --
that was this module's original design, deleted because it applied one
present-day GICS/ICB classification retroactively across the whole
backtest, an undisclosed lookahead (sector membership genuinely changes
over time). Sector now comes from each filing's own point-in-time SEC SIC
code via ``acquisition/sec_edgar.py``'s ``fetch_filing_sic`` +
``sic_gics_crosswalk.py`` -- see ``acquisition/sic_history.py``.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from atlas_quant.strategies.filing_momentum_ml.acquisition.http_client import HttpClient
from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawUniverseRecord

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


def _normalize_symbol(raw_symbol: str) -> str:
    """Wikipedia lists dual-class tickers with a dot (``BRK.B``) -- most
    price/EDGAR providers use a hyphen (``BRK-B``). Converting once here
    keeps every downstream record consistent."""
    return raw_symbol.strip().upper().replace(".", "-")


def parse_sp500_table(html_text: str) -> list[str]:
    """Parse the S&P 500 Wikipedia page's constituent table.

    Returns the list of member symbols. Raises :class:`ValueError` if the
    expected column is not found -- Wikipedia's table structure changing
    is a real risk this must surface clearly, never silently return an
    empty/wrong universe.
    """
    tables = pd.read_html(io.StringIO(html_text), flavor="lxml")
    table = tables[0]
    symbol_col = next((c for c in table.columns if str(c).strip() == "Symbol"), None)
    if symbol_col is None:
        raise ValueError(
            f"S&P 500 Wikipedia table is missing expected columns; found {list(table.columns)!r}"
        )
    return [_normalize_symbol(str(v)) for v in table[symbol_col]]


def parse_nasdaq100_table(html_text: str) -> list[str]:
    """Parse the Nasdaq-100 Wikipedia page's constituent table.

    Returns the list of member symbols.
    """
    tables = pd.read_html(io.StringIO(html_text), flavor="lxml")
    table = tables[0]
    ticker_col = next((c for c in table.columns if str(c).strip() == "Ticker"), None)
    if ticker_col is None:
        raise ValueError(
            f"Nasdaq-100 Wikipedia table is missing expected columns; found {list(table.columns)!r}"
        )
    return [_normalize_symbol(str(v)) for v in table[ticker_col]]


def fetch_sp500_constituents(client: HttpClient) -> list[str]:
    html_text = client.get_text(SP500_WIKIPEDIA_URL, headers={"User-Agent": "AtlasQuant Research"})
    return parse_sp500_table(html_text)


def fetch_nasdaq100_constituents(client: HttpClient) -> list[str]:
    html_text = client.get_text(NASDAQ100_WIKIPEDIA_URL, headers={"User-Agent": "AtlasQuant Research"})
    return parse_nasdaq100_table(html_text)


def build_universe_records(client: HttpClient, *, as_of: datetime, retrieved_at: datetime) -> list[RawUniverseRecord]:
    """Build the present-day S&P 500 + Nasdaq 100 universe from real
    Wikipedia data."""
    sp500 = fetch_sp500_constituents(client)
    nasdaq100 = fetch_nasdaq100_constituents(client)
    all_symbols = sorted(set(sp500) | set(nasdaq100))

    return [
        RawUniverseRecord(
            symbol=symbol, asset_class="equity", as_of=as_of, source="wikipedia_sp500_nasdaq100",
            survivorship_biased=True, retrieved_at=retrieved_at,
        )
        for symbol in all_symbols
    ]
