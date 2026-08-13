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
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawCorporateActionRecord, RawPriceRecord


@runtime_checkable
class PriceHistoryProvider(Protocol):
    """The minimal contract this module needs from any price-history source."""

    def fetch_daily_history(self, symbol: str):
        """Return a pandas DataFrame with raw OHLC, optional ``Adj Close``, and actions."""
        ...


class YFinancePriceProvider:
    """The real price provider, backed by ``yfinance``. Never used to
    acquire data from a test -- see ``tests/fixtures``' ``FakeHistoryProvider``.

    New acquisition intentionally requests raw bars with explicit actions.
    yfinance supplies effective-dated split/dividend histories, not
    vendor-grade announcement/vintage timestamps; downstream code therefore
    treats them as economic events for return intervals and never lets them
    restate model-feature inputs before their effective dates.
    """

    def fetch_daily_history(self, symbol: str):
        import yfinance as yf  # raises ImportError if not installed

        ticker = yf.Ticker(symbol)
        return ticker.history(period="max", auto_adjust=False, actions=True)


@dataclass(frozen=True, slots=True)
class ParsedPriceHistory:
    prices: tuple[RawPriceRecord, ...]
    corporate_actions: tuple[RawCorporateActionRecord, ...]


def parse_price_history_to_records(
    symbol: str, history, *, source: str, retrieved_at: datetime,
) -> list[RawPriceRecord]:
    """Convert one symbol's daily-history DataFrame into raw price records.

    Rows with a non-finite or non-positive close are skipped (reported
    later by :mod:`production.validation`, never silently zeroed). New
    yfinance acquisition stores the source's raw/unadjusted close as the
    normalized close and preserves ``Adj Close`` only for audit.
    """
    return list(parse_price_history(symbol, history, source=source, retrieved_at=retrieved_at).prices)


def parse_price_history(
    symbol: str, history, *, source: str, retrieved_at: datetime,
) -> ParsedPriceHistory:
    """Convert one yfinance history frame into raw bars plus action events."""
    records: list[RawPriceRecord] = []
    actions: list[RawCorporateActionRecord] = []
    for timestamp, row in history.iterrows():
        close = row.get("Close")
        if close is None or close != close or close <= 0:  # NaN check via self-inequality
            raw_record = None
        else:
            raw_record = RawPriceRecord(
                symbol=symbol, asset_class="equity", trading_date=timestamp.date(), close=float(close),
                price_convention="unadjusted", source=source, retrieved_at=retrieved_at,
                raw_open=_finite_positive_or_none(row.get("Open")),
                raw_high=_finite_positive_or_none(row.get("High")),
                raw_low=_finite_positive_or_none(row.get("Low")),
                raw_close=float(close),
                adjusted_close=_finite_positive_or_none(row.get("Adj Close")),
                price_semantics="raw_unadjusted_ohlc; adjusted_close_audit_only",
            )
            records.append(raw_record)

        split = _finite_positive_or_none(row.get("Stock Splits"))
        if split is not None:
            actions.append(
                RawCorporateActionRecord(
                    symbol=symbol, asset_class="equity", action_type="split",
                    effective_date=timestamp.date(), value=split, source=source, retrieved_at=retrieved_at,
                )
            )
        dividend = _finite_positive_or_none(row.get("Dividends"))
        if dividend is not None:
            actions.append(
                RawCorporateActionRecord(
                    symbol=symbol, asset_class="equity", action_type="dividend",
                    effective_date=timestamp.date(), value=dividend, source=source, retrieved_at=retrieved_at,
                )
            )
    return ParsedPriceHistory(prices=tuple(records), corporate_actions=tuple(actions))


def _finite_positive_or_none(value) -> float | None:
    if value is None or value != value or value <= 0:
        return None
    return float(value)


def fetch_price_history_for_symbol(
    provider: PriceHistoryProvider, symbol: str, *, source: str, retrieved_at: datetime,
) -> ParsedPriceHistory:
    """Fetch and parse one symbol's real daily price/action history end-to-end."""
    history = provider.fetch_daily_history(symbol)
    return parse_price_history(symbol, history, source=source, retrieved_at=retrieved_at)


def fetch_prices_for_symbol(
    provider: PriceHistoryProvider, symbol: str, *, source: str, retrieved_at: datetime,
) -> list[RawPriceRecord]:
    """Fetch and parse one symbol's daily price history end-to-end."""
    return list(fetch_price_history_for_symbol(provider, symbol, source=source, retrieved_at=retrieved_at).prices)
