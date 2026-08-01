"""Fail-closed pre-submission checks for automated order execution.

Every check here raises on failure rather than warning and continuing --
a missing price, a closed market, or an outlier-sized order blocks the
order or run it applies to instead of the caller guessing or partially
proceeding. Per-instrument gates (:func:`require_price`,
:func:`check_order_size`) are meant to be caught per order, so one bad
instrument never aborts an entire run; :func:`require_market_open` is a
whole-run gate with no per-instrument granularity to fall back to.

:func:`check_order_size` enforces
:attr:`atlas_quant.config.risk.RiskConfig.max_single_instrument_weight`,
declared in that module as "not yet enforced" pending a consumer -- this
is that consumer's first use of it.
"""

from __future__ import annotations


class RiskGateBlocked(Exception):
    """A risk gate refused to let an order or run proceed."""


def require_price(symbol: str, price: float | None) -> float:
    """Return ``price``, or raise if it's missing or non-positive.

    Never guesses a price and never proceeds on a partial/stale value --
    a caller that can't get a real price for ``symbol`` gets a blocked
    order for that symbol, not a sized one built on bad data.
    """
    if price is None:
        raise RiskGateBlocked(f"{symbol}: no price available -- refusing to size an order without one")
    if price <= 0:
        raise RiskGateBlocked(f"{symbol}: non-positive price {price!r} -- refusing to size an order from it")
    return price


def require_market_open(is_open: bool) -> None:
    """Raise if the market is not open -- no orders are submitted while closed."""
    if not is_open:
        raise RiskGateBlocked("market is not open -- refusing to submit orders")


def check_order_size(
    symbol: str, notional: float, sleeve_equity: float, max_single_instrument_weight: float | None,
) -> None:
    """Raise if ``notional`` (as a fraction of ``sleeve_equity``) exceeds the configured cap.

    ``max_single_instrument_weight is None`` means "no limit configured"
    (per :class:`atlas_quant.config.risk.RiskConfig`'s documented
    semantics), not "unlimited by policy" -- callers that want fail-closed
    behavior must pass an explicit cap.
    """
    if sleeve_equity <= 0:
        raise RiskGateBlocked(f"{symbol}: sleeve_equity={sleeve_equity!r} is not positive -- refusing to size an order")
    if max_single_instrument_weight is None:
        return
    weight = abs(notional) / sleeve_equity
    if weight > max_single_instrument_weight:
        raise RiskGateBlocked(
            f"{symbol}: order notional {notional:.2f} is {weight:.2%} of sleeve equity, "
            f"exceeds max_single_instrument_weight={max_single_instrument_weight:.2%}"
        )
