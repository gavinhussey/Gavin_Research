"""Live/current-price lookups for the current-status report.

Deliberately separate from
:mod:`atlas_quant.strategies.filing_momentum_ml.acquisition.yfinance_provider`'s
batch historical acquisition path -- that module's own docstring scopes
itself to *historical* research acquisition and explicitly defers
near-real-time pricing to a future stage. This module is that stage's
minimal first step: fetch only the latest available close for a handful
of tickers, on demand, never written into the provenance-tracked raw-data
manifest and never used by any offline acquisition/backtest code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Sequence, runtime_checkable

from atlas_quant.domain.identifiers import AssetClass, InstrumentId


@runtime_checkable
class LivePriceProvider(Protocol):
    """The minimal contract this module needs from any live-price source."""

    def fetch_recent_history(self, symbol: str):
        """Return a pandas DataFrame with a DatetimeIndex and a ``Close`` column, most recent last."""
        ...


class YFinanceLivePriceProvider:
    """The real live-price provider, backed by ``yfinance``. Never used from
    a test -- inject a fake :class:`LivePriceProvider` instead."""

    def fetch_recent_history(self, symbol: str):
        import yfinance as yf  # raises ImportError if not installed

        ticker = yf.Ticker(symbol)
        return ticker.history(period="5d", auto_adjust=True)


@dataclass(frozen=True, slots=True)
class LiveQuote:
    """One symbol's most recently available daily close, as of ``as_of``."""

    instrument_id: InstrumentId
    price: float
    as_of: date
    retrieved_at: datetime


def parse_latest_quote(
    symbol: str, history, *, asset_class: AssetClass, retrieved_at: datetime
) -> LiveQuote | None:
    """The latest finite, positive close in ``history``, or ``None`` if none exists.

    A missing/empty/all-non-positive history is never fabricated into a
    price -- callers must treat ``None`` as "current price unavailable."
    """
    if history is None or history.empty:
        return None
    closes = history["Close"].dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return None
    last_index = closes.index[-1]
    as_of = last_index.date() if hasattr(last_index, "date") else last_index
    return LiveQuote(
        instrument_id=InstrumentId(symbol=symbol, asset_class=asset_class),
        price=float(closes.iloc[-1]), as_of=as_of, retrieved_at=retrieved_at,
    )


def fetch_latest_quotes(
    provider: LivePriceProvider,
    symbols: Sequence[str],
    *,
    asset_class: AssetClass = AssetClass.EQUITY,
    retrieved_at: datetime | None = None,
) -> dict[str, LiveQuote]:
    """Fetch and parse the latest quote for every symbol in ``symbols``.

    A symbol whose quote can't be resolved (fetch failure, empty/all-NaN
    history) is simply absent from the result rather than raising -- one
    bad ticker must never block reporting on the rest of the portfolio.
    """
    now = retrieved_at or datetime.now()
    quotes: dict[str, LiveQuote] = {}
    for symbol in symbols:
        try:
            history = provider.fetch_recent_history(symbol)
        except Exception:  # noqa: BLE001 - one bad ticker must not block the others
            continue
        quote = parse_latest_quote(symbol, history, asset_class=asset_class, retrieved_at=now)
        if quote is not None:
            quotes[symbol] = quote
    return quotes
