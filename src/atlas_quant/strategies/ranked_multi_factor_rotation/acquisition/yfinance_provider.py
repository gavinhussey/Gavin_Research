"""Daily OHLC acquisition via yfinance.

``yfinance`` is imported lazily, only inside :class:`YFinanceOHLCProvider`'s
own methods -- this module always imports cleanly even when ``yfinance``
is not installed; a caller that needs the real provider must catch
``ImportError``, never silently substitute a fake. Mirrors
``filing_momentum_ml/acquisition/yfinance_provider.py``'s shape, but
parses the full OHLC range (not close-only) into
:class:`atlas_quant.data.records.DailyOHLCObservation`, since spec §2.4's
true-range/ATR formulas need high/low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from atlas_quant.data.records import DailyOHLCObservation
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance


@runtime_checkable
class OHLCHistoryProvider(Protocol):
    """The minimal contract this module needs from any OHLC-history source."""

    def fetch_daily_history(self, symbol: str):
        """Return a pandas DataFrame with a DatetimeIndex and
        ``Open``/``High``/``Low``/``Close`` columns."""
        ...


class YFinanceOHLCProvider:
    """The real OHLC provider, backed by ``yfinance``. Never used from a
    test -- see ``tests/fixtures``' fake providers."""

    def fetch_daily_history(self, symbol: str):
        import yfinance as yf  # raises ImportError if not installed

        ticker = yf.Ticker(symbol)
        return ticker.history(period="max", auto_adjust=True)


@dataclass(frozen=True, slots=True)
class OHLCParseResult:
    """One symbol's parsed observations plus any rows skipped and why --
    a skip is always disclosed, never silent."""

    observations: tuple[DailyOHLCObservation, ...]
    skipped_rows: tuple[str, ...] = field(default_factory=tuple)


def parse_ohlc_history_to_observations(
    symbol: str, history, *, source: str, retrieved_at: datetime,
) -> OHLCParseResult:
    """Convert one symbol's daily OHLC history DataFrame into
    :class:`DailyOHLCObservation` rows.

    A row with a non-finite, non-positive, or internally-inconsistent
    (``high < low``, ``open``/``close`` outside ``[low, high]``) value is
    skipped and recorded in ``skipped_rows`` -- never silently zeroed or
    coerced, matching ``filing_momentum_ml``'s price-parsing convention.
    """
    observations: list[DailyOHLCObservation] = []
    skipped: list[str] = []
    instrument_id = InstrumentId(symbol=symbol, asset_class=AssetClass.ETF)

    for timestamp, row in history.iterrows():
        raw = {
            "open": row.get("Open"),
            "high": row.get("High"),
            "low": row.get("Low"),
            "close": row.get("Close"),
        }
        trading_date = timestamp.date()
        if any(v is None or v != v for v in raw.values()):  # NaN check via self-inequality
            skipped.append(f"{symbol} {trading_date}: missing/NaN OHLC value")
            continue
        try:
            observations.append(
                DailyOHLCObservation(
                    instrument_id=instrument_id,
                    trading_date=trading_date,
                    open=float(raw["open"]),
                    high=float(raw["high"]),
                    low=float(raw["low"]),
                    close=float(raw["close"]),
                    price_convention="split_dividend_adjusted",
                    provenance=DataProvenance(
                        source=source, as_of=trading_date, retrieved_at=retrieved_at,
                    ),
                )
            )
        except ValueError as exc:
            skipped.append(f"{symbol} {trading_date}: {exc}")

    return OHLCParseResult(observations=tuple(observations), skipped_rows=tuple(skipped))


def fetch_ohlc_for_symbol(
    provider: OHLCHistoryProvider, symbol: str, *, source: str, retrieved_at: datetime,
) -> OHLCParseResult:
    """Fetch and parse one symbol's real daily OHLC history end-to-end."""
    history = provider.fetch_daily_history(symbol)
    return parse_ohlc_history_to_observations(symbol, history, source=source, retrieved_at=retrieved_at)
