"""Daily price acquisition via yfinance.

``yfinance`` is imported lazily, only inside :class:`YFinancePriceProvider`'s
own methods -- this module (and the rest of the acquisition package)
always imports cleanly even when ``yfinance`` is not installed; a caller
that needs the real provider must catch :class:`ImportError`, never
silently substitute a fake.

Future note (not implemented here): the platform's own dependency-status
metadata already tracks ``schwab-py`` as a distinct, optional provider for
a later live/paper-trading stage -- this module exists specifically for
*historical* research acquisition and does not attempt to satisfy that
future live-data need. Swapping in a Schwab-backed provider later means
implementing :class:`PriceHistoryProvider` again, not changing anything
downstream of it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import RawPriceRecord


@runtime_checkable
class PriceHistoryProvider(Protocol):
    """The minimal contract this module needs from any price-history source."""

    def fetch_daily_history(self, symbol: str):
        """Return a pandas DataFrame with a DatetimeIndex and a ``Close`` column."""
        ...


class YFinancePriceProvider:
    """The real price provider, backed by ``yfinance``. Never used to
    acquire data from a test -- see ``tests/fixtures``' ``FakeHistoryProvider``."""

    def fetch_daily_history(self, symbol: str):
        import yfinance as yf  # raises ImportError if not installed

        ticker = yf.Ticker(symbol)
        return ticker.history(period="max", auto_adjust=True)


def parse_price_history_to_records(
    symbol: str, history, *, source: str, retrieved_at: datetime,
) -> list[RawPriceRecord]:
    """Convert one symbol's daily-history DataFrame into raw price records.

    Rows with a non-finite or non-positive close are skipped (reported
    later by :mod:`production.validation`, never silently zeroed);
    ``price_convention`` is always ``"split_dividend_adjusted"`` --
    ``auto_adjust=True``'s own documented behavior.
    """
    records: list[RawPriceRecord] = []
    for timestamp, row in history.iterrows():
        close = row.get("Close")
        if close is None or close != close or close <= 0:  # NaN check via self-inequality
            continue
        records.append(
            RawPriceRecord(
                symbol=symbol, asset_class="equity", trading_date=timestamp.date(), close=float(close),
                price_convention="split_dividend_adjusted", source=source, retrieved_at=retrieved_at,
            )
        )
    return records


def fetch_prices_for_symbol(
    provider: PriceHistoryProvider, symbol: str, *, source: str, retrieved_at: datetime,
) -> list[RawPriceRecord]:
    """Fetch and parse one symbol's real daily price history end-to-end."""
    history = provider.fetch_daily_history(symbol)
    return parse_price_history_to_records(symbol, history, source=source, retrieved_at=retrieved_at)
