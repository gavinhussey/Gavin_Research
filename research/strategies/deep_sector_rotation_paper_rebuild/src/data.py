"""Paper universe, canonical ordering, and raw price-data loading.

Source: paper Table 1 (p.3). See ../docs/paper_source_audit.md #1, #9.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical order = Table 1 row order (sector-name order as printed),
# NOT Figure 1's alphabetical-by-symbol order. See audit #1/#9.
PAPER_UNIVERSE: tuple[str, ...] = (
    "XLK",  # Information Technology
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "VOX",  # Communication Services
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLB",  # Materials
    "IYR",  # Real Estate
    "XLE",  # Energy
)

SECTOR_NAMES: dict[str, str] = {
    "XLK": "Information Technology",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "VOX": "Communication Services",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "IYR": "Real Estate",
    "XLE": "Energy",
}

# Forbidden per task brief: the archived project's Vanguard-fund universe.
VANGUARD_UNIVERSE_FORBIDDEN: frozenset[str] = frozenset(
    {"VGT", "VHT", "VCR", "VFH", "VIS", "VDC", "VPU", "VAW", "VNQ", "VDE"}
)

RAW_PRICES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "prices"


def assert_canonical_order(tickers: list[str] | tuple[str, ...]) -> None:
    """Assert a ticker-indexed sequence matches PAPER_UNIVERSE order exactly."""
    if tuple(tickers) != PAPER_UNIVERSE:
        raise AssertionError(
            f"Ticker ordering does not match canonical PAPER_UNIVERSE order.\n"
            f"Expected: {PAPER_UNIVERSE}\nGot:      {tuple(tickers)}"
        )


def assert_no_vanguard_substitution(tickers: list[str] | tuple[str, ...]) -> None:
    overlap = VANGUARD_UNIVERSE_FORBIDDEN.intersection(tickers)
    if overlap:
        raise AssertionError(
            f"Vanguard-adaptation tickers {sorted(overlap)} must not appear in "
            f"the paper rebuild universe -- see CLAUDE.md/task brief hard "
            f"constraints (do not preserve Vanguard universe)."
        )


def load_raw_prices(symbol: str) -> pd.DataFrame:
    """Load immutable raw OHLCV data for one symbol (Yahoo Finance via yfinance)."""
    path = RAW_PRICES_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No raw price data for {symbol!r} at {path}. Data must be fetched "
            f"from Yahoo Finance (real data only -- no synthetic substitution "
            f"permitted) before this function can be used."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_universe_prices() -> dict[str, pd.DataFrame]:
    """Load raw price data for the full 11-ticker paper universe, in canonical order."""
    assert_canonical_order(PAPER_UNIVERSE)
    assert_no_vanguard_substitution(PAPER_UNIVERSE)
    return {symbol: load_raw_prices(symbol) for symbol in PAPER_UNIVERSE}
