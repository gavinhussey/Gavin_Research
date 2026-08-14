"""Dynamic per-ETF ROC-based buy thresholds.

Source: paper p.4, "Generation of `buy' signal". Threshold vector updated
weekly, estimated via ROC-curve cost optimization on recently observed
data. See ../docs/paper_source_audit.md #22.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .data import PAPER_UNIVERSE
from .decisions import require_resolved


@dataclass
class RocThresholdState:
    """State interface: threshold[ticker, week]. Populated weekly."""

    thresholds: dict[str, dict[int, float]] = field(
        default_factory=lambda: {ticker: {} for ticker in PAPER_UNIVERSE}
    )

    def get(self, ticker: str, week_id: int) -> float:
        if ticker not in self.thresholds or week_id not in self.thresholds[ticker]:
            require_resolved(
                "DECISION_REQUIRED_ROC_FALLBACK",
                required_before=(
                    f"reading a ROC threshold for {ticker!r} at week {week_id} "
                    f"before one has been estimated (insufficient history case)"
                ),
            )
        return self.thresholds[ticker][week_id]

    def set(self, ticker: str, week_id: int, threshold: float) -> None:
        self.thresholds.setdefault(ticker, {})[week_id] = threshold


def estimate_threshold(scores, labels, ticker: str, week_id: int) -> float:
    """Decision-gated entry point for estimating one asset's weekly ROC threshold.

    Blocked until DECISION_REQUIRED_ROC_OBJECTIVE (which operating-point
    criterion), DECISION_REQUIRED_ROC_LOOKBACK (history window), and
    DECISION_REQUIRED_ROC_MIN_SAMPLE (minimum data before estimation is
    valid) are all resolved. Youden's J is explicitly NOT used as a
    default per the task brief.
    """
    require_resolved(
        "DECISION_REQUIRED_ROC_OBJECTIVE",
        required_before=f"estimating the ROC threshold for {ticker!r} at week {week_id}",
    )


def apply_threshold(raw_scores: dict[str, float], state: RocThresholdState, week_id: int) -> dict[str, bool]:
    """Preliminary buy decision: raw_scores[ticker] > threshold[ticker, week_id]."""
    return {ticker: raw_scores[ticker] > state.get(ticker, week_id) for ticker in raw_scores}
