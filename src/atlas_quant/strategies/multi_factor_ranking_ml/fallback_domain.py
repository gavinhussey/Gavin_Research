"""Typed fallback-asset statistics — the auditable input to dynamic fallback weighting.

Report §5.4: fewer than ``min_positions`` qualifying picks falls back to
``ML_FALLBACK_TICKERS = ["SPY", "VGT"]``, weighted by trailing 12-quarter
average return when ``ML_FALLBACK_DYNAMIC_WEIGHT = True``. This module
represents the per-asset return history that weighting is computed from,
so a fallback weight is always traceable to concrete observations, never
an unexplained bare ``{ticker: weight}`` mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance


@dataclass(frozen=True, slots=True)
class FallbackAssetStatistics:
    """One fallback ticker's trailing quarterly-return history as of a cutoff.

    ``quarterly_returns`` is ordered oldest-to-newest (most-recent-last,
    matching every other sequence convention in this codebase), and may
    contain fewer than the configured lookback window if that much history
    isn't available -- this record does not pad or synthesize missing
    quarters. A non-finite (NaN/inf) entry represents a quarter whose
    return could not be computed (e.g. missing price data) and is excluded
    by :func:`~atlas_quant.strategies.multi_factor_ranking_ml.fallback_weighting
    .dynamic_fallback_weights`, not by this record.
    """

    instrument_id: InstrumentId
    measurement_cutoff: datetime
    quarterly_returns: tuple[float, ...]
    observation_count: int
    provenance: DataProvenance
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.observation_count != len(self.quarterly_returns):
            raise ValueError(
                "FallbackAssetStatistics.observation_count must equal "
                f"len(quarterly_returns) ({len(self.quarterly_returns)}), "
                f"got {self.observation_count!r}"
            )
