"""Corporate-action-aware economic return helpers.

Features consume raw close histories directly. This module is only for
economic return intervals: labels, portfolio accounting, benchmark returns,
and fallback ETF trailing returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from atlas_quant.data.records import CorporateActionRecord


@dataclass(frozen=True, slots=True)
class EconomicReturnBreakdown:
    raw_return: float
    split_factor: float
    dividend_cash: float
    split_count: int
    dividend_count: int


def actions_in_interval(
    actions: Sequence[CorporateActionRecord],
    entry_date: date,
    exit_date: date,
) -> tuple[CorporateActionRecord, ...]:
    """Actions with effective dates after entry and on/before exit."""
    return tuple(sorted(
        (a for a in actions if entry_date < a.effective_date <= exit_date),
        key=lambda a: (a.effective_date, a.action_type),
    ))


def compute_economic_return(
    entry_price: float,
    exit_price: float,
    entry_date: date,
    exit_date: date,
    actions: Sequence[CorporateActionRecord] = (),
    *,
    include_dividends: bool = True,
    apply_split_factor: bool = False,
) -> EconomicReturnBreakdown:
    """Return one-share economic performance over a holding interval.

    For split-normalized close series, split events are audit/count events
    only; applying them again would double-count the split. For genuinely
    split-discontinuous raw prices, callers can set ``apply_split_factor``
    so split events multiply the share count from their effective date
    forward. Cash dividends are paid once, on the shares held at that event
    date. No event after ``exit_date`` can affect the return.
    """
    shares = 1.0
    dividend_cash = 0.0
    split_count = 0
    dividend_count = 0
    for action in actions_in_interval(actions, entry_date, exit_date):
        if action.action_type == "split":
            if apply_split_factor:
                shares *= action.value
            split_count += 1
        elif action.action_type == "dividend" and include_dividends:
            dividend_cash += shares * action.value
            dividend_count += 1

    raw_return = ((exit_price * shares) + dividend_cash - entry_price) / entry_price
    return EconomicReturnBreakdown(
        raw_return=raw_return,
        split_factor=shares,
        dividend_cash=dividend_cash,
        split_count=split_count,
        dividend_count=dividend_count,
    )
