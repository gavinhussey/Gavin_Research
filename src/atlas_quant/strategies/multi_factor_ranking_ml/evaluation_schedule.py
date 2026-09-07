"""This strategy's evaluation schedule: rank the universe on the first
calendar day of every quarter.

Replaces filing_momentum_ml's fixed post-quarter-end lag
(``earnings_lag_days`` = 42, a calendar-day approximation of "give
companies time to file"). This strategy instead has a real, per-row
``available_date`` on every fundamentals record, so no approximation is
needed: each quarterly ranking cycle's ``cutoff`` is exactly the day
before its ``quarter_start``, and only feature data with
``available_date <= cutoff`` may be used to produce that cycle's ranking
-- a genuine point-in-time selection, not a fixed-lag guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

_QUARTER_START_MONTHS = (1, 4, 7, 10)


@dataclass(frozen=True, slots=True)
class EvaluationCycle:
    """One quarterly ranking cycle: rank the universe as of ``quarter_start``,
    using only feature data with ``available_date <= cutoff``."""

    quarter_start: date
    cutoff: date

    def __post_init__(self) -> None:
        if self.cutoff >= self.quarter_start:
            raise ValueError(
                f"EvaluationCycle.cutoff ({self.cutoff!r}) must be strictly "
                f"before quarter_start ({self.quarter_start!r})"
            )


def quarterly_evaluation_cycles(start: date, end: date) -> tuple[EvaluationCycle, ...]:
    """Every quarter-start date (Jan 1, Apr 1, Jul 1, Oct 1) within
    ``[start, end]`` inclusive, each paired with its cutoff (the calendar
    day immediately before it).

    Returns cycles in ascending chronological order. Raises
    :class:`ValueError` if ``end`` is before ``start``.
    """
    if end < start:
        raise ValueError(f"end ({end!r}) must not be before start ({start!r})")

    cycles: list[EvaluationCycle] = []
    for year in range(start.year, end.year + 1):
        for month in _QUARTER_START_MONTHS:
            quarter_start = date(year, month, 1)
            if start <= quarter_start <= end:
                cycles.append(
                    EvaluationCycle(quarter_start=quarter_start, cutoff=quarter_start - timedelta(days=1))
                )
    return tuple(cycles)
