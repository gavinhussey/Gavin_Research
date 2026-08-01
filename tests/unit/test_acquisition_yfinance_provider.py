"""Unit tests for the yfinance-backed daily price acquisition adapter.

Uses small, hand-crafted pandas DataFrames shaped like yfinance's own
``Ticker.history()`` output -- never a real network call.
"""

from datetime import datetime

import pandas as pd

from atlas_quant.strategies.filing_momentum_ml.acquisition.yfinance_provider import (
    fetch_prices_for_symbol,
    parse_price_history_to_records,
)

from tests.fixtures.acquisition import FakeHistoryProvider

_RETRIEVED_AT = datetime(2024, 6, 1)


def _history_df(rows: list[tuple[str, float]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    return pd.DataFrame({"Close": [r[1] for r in rows]}, index=index)


def test_parse_price_history_to_records_basic():
    history = _history_df([("2024-01-02", 100.0), ("2024-01-03", 101.5)])
    records = parse_price_history_to_records("AAPL", history, source="yfinance", retrieved_at=_RETRIEVED_AT)
    assert len(records) == 2
    assert records[0].symbol == "AAPL"
    assert records[0].trading_date.isoformat() == "2024-01-02"
    assert records[0].close == 100.0
    assert records[0].price_convention == "split_dividend_adjusted"


def test_parse_price_history_skips_non_positive_and_nan_closes():
    history = _history_df([("2024-01-02", 100.0), ("2024-01-03", 0.0), ("2024-01-04", -5.0)])
    history.loc["2024-01-05"] = float("nan")
    records = parse_price_history_to_records("AAPL", history, source="yfinance", retrieved_at=_RETRIEVED_AT)
    assert len(records) == 1
    assert records[0].trading_date.isoformat() == "2024-01-02"


def test_fetch_prices_for_symbol_end_to_end():
    history = _history_df([("2024-01-02", 100.0)])
    provider = FakeHistoryProvider({"AAPL": history})
    records = fetch_prices_for_symbol(provider, "AAPL", source="yfinance", retrieved_at=_RETRIEVED_AT)
    assert len(records) == 1
    assert provider.requested_symbols == ["AAPL"]
