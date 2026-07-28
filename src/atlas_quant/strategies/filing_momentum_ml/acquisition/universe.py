"""S&P 500 + Nasdaq 100 universe and sector acquisition.

Uses each index's own Wikipedia page (free, no key) -- the standard,
widely-used community-maintained source for current constituents. This
is, by construction, a **present-day snapshot applied retroactively**:
report §5.4's own documented, survivorship-biased approach, not a defect
this module hides.

Sector data is a byproduct of the same tables (S&P 500's GICS Sector
column, Nasdaq-100's ICB Industry column) rather than a separate
per-ticker lookup -- when a symbol appears in both indices, the S&P 500
table's GICS classification is preferred (GICS is the taxonomy report
§5.2's ``exclude_sectors`` values are expressed in).
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from atlas_quant.strategies.filing_momentum_ml.acquisition.http_client import HttpClient
from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawSectorRecord, RawUniverseRecord

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


def _normalize_symbol(raw_symbol: str) -> str:
    """Wikipedia lists dual-class tickers with a dot (``BRK.B``) -- most
    price/EDGAR providers use a hyphen (``BRK-B``). Converting once here
    keeps every downstream record consistent."""
    return raw_symbol.strip().upper().replace(".", "-")


def parse_sp500_table(html_text: str) -> list[tuple[str, str]]:
    """Parse the S&P 500 Wikipedia page's constituent table.

    Returns a list of ``(symbol, gics_sector)`` pairs. Raises
    :class:`ValueError` if the expected columns are not found -- Wikipedia's
    table structure changing is a real risk this must surface clearly,
    never silently return an empty/wrong universe.
    """
    tables = pd.read_html(io.StringIO(html_text), flavor="lxml")
    table = tables[0]
    symbol_col = next((c for c in table.columns if str(c).strip() == "Symbol"), None)
    sector_col = next((c for c in table.columns if str(c).strip().startswith("GICS Sector")), None)
    if symbol_col is None or sector_col is None:
        raise ValueError(
            f"S&P 500 Wikipedia table is missing expected columns; found {list(table.columns)!r}"
        )
    return [
        (_normalize_symbol(str(row[symbol_col])), str(row[sector_col]).strip())
        for _, row in table.iterrows()
    ]


def parse_nasdaq100_table(html_text: str) -> list[tuple[str, str]]:
    """Parse the Nasdaq-100 Wikipedia page's constituent table.

    Returns a list of ``(symbol, icb_industry)`` pairs.
    """
    tables = pd.read_html(io.StringIO(html_text), flavor="lxml")
    table = tables[0]
    ticker_col = next((c for c in table.columns if str(c).strip() == "Ticker"), None)
    industry_col = next((c for c in table.columns if str(c).strip().startswith("ICB Industry")), None)
    if ticker_col is None or industry_col is None:
        raise ValueError(
            f"Nasdaq-100 Wikipedia table is missing expected columns; found {list(table.columns)!r}"
        )
    return [
        (_normalize_symbol(str(row[ticker_col])), str(row[industry_col]).strip())
        for _, row in table.iterrows()
    ]


def fetch_sp500_constituents(client: HttpClient) -> list[tuple[str, str]]:
    html_text = client.get_text(SP500_WIKIPEDIA_URL, headers={"User-Agent": "AtlasQuant Research"})
    return parse_sp500_table(html_text)


def fetch_nasdaq100_constituents(client: HttpClient) -> list[tuple[str, str]]:
    html_text = client.get_text(NASDAQ100_WIKIPEDIA_URL, headers={"User-Agent": "AtlasQuant Research"})
    return parse_nasdaq100_table(html_text)


def build_universe_and_sector_records(
    client: HttpClient, *, as_of: datetime, retrieved_at: datetime,
) -> tuple[list[RawUniverseRecord], list[RawSectorRecord]]:
    """Build the present-day S&P 500 + Nasdaq 100 universe and each
    member's raw sector classification, from real Wikipedia data."""
    sp500 = fetch_sp500_constituents(client)
    nasdaq100 = fetch_nasdaq100_constituents(client)

    sector_by_symbol: dict[str, str] = {}
    for symbol, sector in nasdaq100:
        sector_by_symbol.setdefault(symbol, sector)
    for symbol, sector in sp500:  # S&P 500's GICS classification takes priority when both list a symbol
        sector_by_symbol[symbol] = sector

    all_symbols = sorted(sector_by_symbol)

    universe_records = [
        RawUniverseRecord(
            symbol=symbol, asset_class="equity", as_of=as_of, source="wikipedia_sp500_nasdaq100",
            survivorship_biased=True, retrieved_at=retrieved_at,
        )
        for symbol in all_symbols
    ]
    sector_records = [
        RawSectorRecord(
            symbol=symbol, asset_class="equity", raw_sector=sector_by_symbol[symbol], as_of=as_of,
            source="wikipedia_sp500_nasdaq100", retrieved_at=retrieved_at,
        )
        for symbol in all_symbols
    ]
    return universe_records, sector_records
