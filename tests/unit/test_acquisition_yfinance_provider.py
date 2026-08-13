"""Unit tests for the yfinance-backed daily price acquisition adapter.

Uses small, hand-crafted pandas DataFrames shaped like yfinance's own
``Ticker.history()`` output -- never a real network call.
"""

from datetime import datetime

import pandas as pd

from atlas_quant.strategies.filing_momentum_ml.acquisition.yfinance_provider import (
    fetch_prices_for_symbol,
    parse_price_history,
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
    assert records[0].price_convention == "unadjusted"
    assert records[0].raw_close == 100.0


def test_parse_price_history_keeps_adjusted_close_audit_and_actions_separate():
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    history = pd.DataFrame(
        {
            "Open": [99.0, 50.0], "High": [101.0, 51.0], "Low": [98.0, 49.0],
            "Close": [100.0, 50.5], "Adj Close": [95.0, 50.5],
            "Dividends": [0.0, 0.25], "Stock Splits": [0.0, 2.0],
        },
        index=index,
    )
    parsed = parse_price_history("AAPL", history, source="yfinance", retrieved_at=_RETRIEVED_AT)
    assert parsed.prices[0].raw_open == 99.0
    assert parsed.prices[0].adjusted_close == 95.0
    assert parsed.prices[0].price_semantics == "raw_unadjusted_ohlc; adjusted_close_audit_only"
    assert [(a.action_type, a.effective_date.isoformat(), a.value) for a in parsed.corporate_actions] == [
        ("split", "2024-01-03", 2.0),
        ("dividend", "2024-01-03", 0.25),
    ]


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
