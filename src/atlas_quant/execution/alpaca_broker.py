"""Thin adapter around Alpaca's paper-trading REST API.

``alpaca-py`` is imported lazily, only inside the methods that need it,
the same way ``model_store.py`` imports ``joblib`` lazily -- it ships in
the optional ``trading`` extra, not the core dependency set, so this
module must always import cleanly without it installed.

Every method raises on failure; there is no silent fallback anywhere in
this adapter. A caller relying on broker state to size or gate an order
must see a real error, never a guessed value -- that guarantee is what
lets :mod:`atlas_quant.execution.risk_gates` treat broker calls as safe
to build fail-closed checks on top of.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from atlas_quant.config.secrets import SecretsConfig


class BrokerError(Exception):
    """A broker API call failed or returned data this adapter can't trust."""


@dataclass(frozen=True, slots=True)
class AccountState:
    equity: float
    cash: float
    buying_power: float


@dataclass(frozen=True, slots=True)
class MarketClock:
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True, slots=True)
class BrokerOrderResult:
    broker_order_id: str
    symbol: str
    side: str
    filled_qty: float
    filled_price: float
    filled_at: datetime


class AlpacaBroker:
    """Wraps ``alpaca.trading.client.TradingClient`` for one shared paper-trading account."""

    def __init__(self, secrets: SecretsConfig, *, paper: bool = True) -> None:
        if not secrets.alpaca_api_key or not secrets.alpaca_api_secret:
            raise BrokerError("ALPACA_API_KEY / ALPACA_API_SECRET are not set -- cannot construct AlpacaBroker")
        self._secrets = secrets
        self._paper = paper
        self._client = self._build_trading_client()
        self._data_client = None

    def _build_trading_client(self):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise BrokerError(
                "alpaca-py is not installed -- install the 'trading' extra to use AlpacaBroker"
            ) from exc

        return TradingClient(
            api_key=self._secrets.alpaca_api_key, secret_key=self._secrets.alpaca_api_secret, paper=self._paper,
        )

    def _get_data_client(self):
        if self._data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._data_client = StockHistoricalDataClient(
                self._secrets.alpaca_api_key, self._secrets.alpaca_api_secret,
            )
        return self._data_client

    def get_account(self) -> AccountState:
        try:
            account = self._client.get_account()
        except Exception as exc:  # noqa: BLE001 - any broker API failure must surface, never be swallowed
            raise BrokerError(f"failed to fetch account state: {exc}") from exc
        return AccountState(
            equity=float(account.equity), cash=float(account.cash), buying_power=float(account.buying_power),
        )

    def get_positions(self) -> dict[str, float]:
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"failed to fetch positions: {exc}") from exc
        return {p.symbol: float(p.qty) for p in positions}

    def get_clock(self) -> MarketClock:
        try:
            clock = self._client.get_clock()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"failed to fetch market clock: {exc}") from exc
        return MarketClock(is_open=bool(clock.is_open), next_open=clock.next_open, next_close=clock.next_close)

    def get_last_price(self, symbol: str) -> float | None:
        """Return Alpaca's own last ask quote for ``symbol``, or ``None`` if unavailable.

        Sizing an order off the same venue that will fill it (rather than
        the yfinance-backed ``live_pricing`` provider used for reporting)
        keeps the share-count math consistent with actual execution.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        try:
            quotes = self._get_data_client().get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"{symbol}: failed to fetch latest quote: {exc}") from exc
        quote = quotes.get(symbol)
        if quote is None or not quote.ask_price:
            return None
        return float(quote.ask_price)

    def submit_market_order(self, symbol: str, qty: float, side: str) -> BrokerOrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY)
        try:
            order = self._client.submit_order(request)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"{symbol}: order submission failed: {exc}") from exc
        if order.filled_avg_price is None or order.filled_qty is None:
            raise BrokerError(
                f"{symbol}: order {order.id} submitted but not yet filled -- market orders should fill immediately"
            )
        return BrokerOrderResult(
            broker_order_id=str(order.id), symbol=symbol, side=side,
            filled_qty=float(order.filled_qty), filled_price=float(order.filled_avg_price),
            filled_at=order.filled_at or datetime.now(),
        )
