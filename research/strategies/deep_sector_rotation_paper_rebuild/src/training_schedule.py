"""Annual model training + weekly incremental update scheduler.

Source: paper p.3, "Model optimization and dynamic update" (annual cadence,
EXPLICIT) and p.4-5 step list (weekly update timing, EXPLICIT; mechanism,
MISSING). See ../docs/paper_source_audit.md #19-20.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .decisions import require_resolved


@dataclass
class AnnualScheduler:
    """Structural scaffold for the paper's training loop:

        for each trading year Y in 2012..2022:
            initialize annual model
            train initially using prior 2 years  [Y-2, Y)
            for each trading week in Y:
                predict
                execute the weekly trading cycle
                once the new outcome becomes known:
                    update the current annual model

    No future information may be used at any step (each week's update
    happens strictly after that week's own label is known, and strictly
    before the following week's prediction -- enforced by ordering, not by
    a runtime lookahead check, since the scheduler itself never receives
    data past the current week).
    """

    first_trading_year: int = 2012  # EXPLICIT, p.1 abstract / p.3
    last_trading_year: int = 2022  # EXPLICIT
    training_history_years: int = 2  # EXPLICIT, p.3 ("previous 2 years")

    loss_fn: object | None = None
    weekly_update_epochs: int | None = None

    def trading_years(self) -> list[int]:
        return list(range(self.first_trading_year, self.last_trading_year + 1))

    def training_window_for_year(self, year: int) -> tuple[int, int]:
        """Return [start_year, end_year) for the 2-year training window preceding `year`."""
        return (year - self.training_history_years, year)

    def initialize_annual_model(self, year: int):
        if self.loss_fn is None:
            require_resolved(
                "DECISION_REQUIRED_CUSTOM_FINANCIAL_LOSS",
                required_before=f"initializing the annual model for trading year {year}",
            )

    def weekly_update(self, year: int, week_id: int):
        """Perform the weekly incremental update step (last step of the weekly cycle).

        Blocked until DECISION_REQUIRED_WEEKLY_UPDATE_MECHANISM is resolved
        -- the paper states *when* this happens (after the week's label is
        known) but not *how* (single-example gradient step vs. expanding-
        window refit vs. rolling-window retrain).
        """
        require_resolved(
            "DECISION_REQUIRED_WEEKLY_UPDATE_MECHANISM",
            required_before=f"performing the weekly incremental model update for year {year}, week {week_id}",
        )
