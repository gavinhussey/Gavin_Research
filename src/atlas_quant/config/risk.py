"""Portfolio-level risk limits — declared, not yet enforced.

No risk-gating logic exists in the platform yet (Stage 7+ will consume
this). This type exists now so strategy and portfolio code can be written
to accept a ``RiskConfig`` without every later stage needing to add the
parameter retroactively. Leaving a limit as ``None`` means "no limit
configured," not "unlimited by policy" — that distinction matters once
enforcement exists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_gross_exposure_pct: float | None = None
    max_single_instrument_weight: float | None = None
    max_sector_weight: float | None = None
