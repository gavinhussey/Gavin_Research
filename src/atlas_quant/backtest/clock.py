"""The deterministic historical-quarter event clock, report §5.5.

Report §5.5: "Each stock enters the portfolio the trading day after its
10-Q/10-K is filed (capped at day-42). All positions in a given quarter's
cohort exit simultaneously on the following quarter's buy date: buy_dt =
quarter_end + 42 calendar days, sell_dt = next_quarter_end + 42 calendar
days." This module turns that into one explicit, typed, deterministically
iterable period sequence — never an implicit DataFrame-index-driven loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from atlas_quant.config.identity import compute_config_identity


@dataclass(frozen=True, slots=True)
class BacktestPeriod:
    """One scoring quarter's complete timing, report §5.5.

    ``evaluation_timestamp`` and ``entry_timestamp`` are the same instant
    (this quarter's own ``buy_dt``) — kept as two separate, identically-
    named fields because they answer two different questions ("as of when
    do we evaluate/train/score" vs. "when does the cohort enter"), even
    though report §5.5 defines them as the same timestamp. ``training_cutoff``
    is also identical to ``evaluation_timestamp`` — a target quarter may
    only train on outcomes knowable strictly before its own evaluation
    moment.
    """

    quarter_end: date
    next_quarter_end: date
    evaluation_timestamp: datetime
    training_cutoff: datetime
    entry_timestamp: datetime
    exit_timestamp: datetime
    label_availability_cutoff: datetime

    def identity(self) -> str:
        return compute_config_identity(
            {
                "quarter_end": self.quarter_end.isoformat(),
                "next_quarter_end": self.next_quarter_end.isoformat(),
                "evaluation_timestamp": self.evaluation_timestamp,
                "entry_timestamp": self.entry_timestamp,
                "exit_timestamp": self.exit_timestamp,
            }
        )


def _next_quarter_end(quarter_end: date) -> date:
    month = quarter_end.month + 3
    year = quarter_end.year
    if month > 12:
        month -= 12
        year += 1
    # Quarter ends are always month-end dates (31/30/28-or-29); use the
    # last day of the target month rather than assuming day==quarter_end.day
    # carries over (handles Feb -> a 31-day month cleanly since we always
    # step by whole calendar quarters from an already-valid month-end).
    if month in (1, 3, 5, 7, 8, 10, 12):
        day = 31
    elif month == 2:
        day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    else:
        day = 30
    return date(year, month, day)


def build_period(quarter_end: date, *, earnings_lag_days: int = 42) -> BacktestPeriod:
    """Build one :class:`BacktestPeriod` for ``quarter_end``, report §5.5."""
    next_end = _next_quarter_end(quarter_end)
    buy_dt = datetime.combine(quarter_end, datetime.min.time()) + timedelta(days=earnings_lag_days)
    sell_dt = datetime.combine(next_end, datetime.min.time()) + timedelta(days=earnings_lag_days)
    return BacktestPeriod(
        quarter_end=quarter_end,
        next_quarter_end=next_end,
        evaluation_timestamp=buy_dt,
        training_cutoff=buy_dt,
        entry_timestamp=buy_dt,
        exit_timestamp=sell_dt,
        label_availability_cutoff=sell_dt,
    )


def generate_quarterly_periods(
    start_quarter_end: date, end_quarter_end: date, *, earnings_lag_days: int = 42
) -> tuple[BacktestPeriod, ...]:
    """Generate every quarter-end period from ``start_quarter_end`` to
    ``end_quarter_end`` inclusive, in ascending chronological order.

    Raises :class:`ValueError` if ``end_quarter_end`` is before
    ``start_quarter_end``, or if either is not itself a valid calendar
    quarter-end date (the 31st/30th/28th-or-29th of Mar/Jun/Sep/Dec) — an
    invalid range must fail clearly, not silently produce zero periods.
    """
    if end_quarter_end < start_quarter_end:
        raise ValueError(
            f"end_quarter_end ({end_quarter_end!r}) is before "
            f"start_quarter_end ({start_quarter_end!r})"
        )
    for label, q in (("start_quarter_end", start_quarter_end), ("end_quarter_end", end_quarter_end)):
        if q != _next_quarter_end(_previous_quarter_end(q)):
            raise ValueError(f"{label} ({q!r}) is not a valid calendar quarter-end date")

    periods = []
    current = start_quarter_end
    while current <= end_quarter_end:
        periods.append(build_period(current, earnings_lag_days=earnings_lag_days))
        current = _next_quarter_end(current)
    return tuple(periods)


def _previous_quarter_end(quarter_end: date) -> date:
    month = quarter_end.month - 3
    year = quarter_end.year
    if month < 1:
        month += 12
        year -= 1
    if month in (1, 3, 5, 7, 8, 10, 12):
        day = 31
    elif month == 2:
        day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    else:
        day = 30
    return date(year, month, day)
