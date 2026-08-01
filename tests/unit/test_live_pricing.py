"""Unit tests for atlas_quant.strategies.filing_momentum_ml.production.live_pricing.

No network access -- ``YFinanceLivePriceProvider`` is never exercised
from a test; only the pure parsing/aggregation functions are.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.production.live_pricing import (
    fetch_latest_quotes,
    parse_latest_quote,
)


def _history(rows: list[tuple[str, float]]) -> pd.DataFrame:
    index = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"Close": [r[1] for r in rows]}, index=index)


class TestParseLatestQuote:
    def test_returns_the_latest_close(self):
        history = _history([("2026-07-28", 100.0), ("2026-07-29", 101.0), ("2026-07-30", 102.5)])
        quote = parse_latest_quote("AAA", history, asset_class=AssetClass.EQUITY, retrieved_at=datetime(2026, 7, 30, 9))
        assert quote.price == 102.5
        assert quote.as_of == date(2026, 7, 30)
        assert quote.instrument_id == InstrumentId(symbol="AAA", asset_class=AssetClass.EQUITY)

    def test_skips_nan_and_non_positive_rows(self):
        history = _history([("2026-07-28", 100.0), ("2026-07-29", 0.0), ("2026-07-30", -5.0)])
        history.loc[history.index[1], "Close"] = float("nan")
        quote = parse_latest_quote("AAA", history, asset_class=AssetClass.EQUITY, retrieved_at=datetime(2026, 7, 30))
        # The only genuinely usable row is the first (100.0); NaN and negative are excluded.
        assert quote.price == 100.0
        assert quote.as_of == date(2026, 7, 28)

    def test_empty_history_returns_none(self):
        assert parse_latest_quote("AAA", pd.DataFrame({"Close": []}), asset_class=AssetClass.EQUITY, retrieved_at=datetime.now()) is None

    def test_none_history_returns_none(self):
        assert parse_latest_quote("AAA", None, asset_class=AssetClass.EQUITY, retrieved_at=datetime.now()) is None


class _FakeProvider:
    def __init__(self, data: dict[str, pd.DataFrame | None]):
        self._data = data

    def fetch_recent_history(self, symbol: str):
        if symbol not in self._data:
            raise RuntimeError(f"no data for {symbol}")
        return self._data[symbol]


class TestFetchLatestQuotes:
    def test_fetches_every_resolvable_symbol(self):
        provider = _FakeProvider({
            "AAA": _history([("2026-07-30", 10.0)]),
            "BBB": _history([("2026-07-30", 20.0)]),
        })
        quotes = fetch_latest_quotes(provider, ["AAA", "BBB"], retrieved_at=datetime(2026, 7, 30))
        assert set(quotes) == {"AAA", "BBB"}
        assert quotes["AAA"].price == 10.0

    def test_one_bad_ticker_does_not_block_the_rest(self):
        provider = _FakeProvider({"AAA": _history([("2026-07-30", 10.0)])})  # BBB fetch raises
        quotes = fetch_latest_quotes(provider, ["AAA", "BBB"], retrieved_at=datetime(2026, 7, 30))
        assert set(quotes) == {"AAA"}

    def test_empty_symbol_list_returns_empty(self):
        provider = _FakeProvider({})
        assert fetch_latest_quotes(provider, []) == {}
