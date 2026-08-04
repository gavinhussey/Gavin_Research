"""Ranked Multi-Factor Rotation — typed strategy configuration.

Scaffold only: no factor formulas, weights, universe, or thresholds are
defined here yet. This module intentionally holds only the structural
fields every AtlasQuant strategy config needs (identifier, versioning,
``identity()``) plus placeholders for the fields this strategy's own
specification will eventually fix. Populate ``factor_names``,
``factor_weights``, ``rebalance_frequency``, ``universe_id``, and any
selection/construction parameters only once the user has supplied the
actual specification for them — do not guess values here, and do not add
a default that could pass validation while representing an undecided
value as if it were a real one; ``"unspecified"``/``None`` are used
below specifically so an unconfigured instance is still constructible
(for registry/test wiring) without looking like a real setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from atlas_quant.config.identity import compute_config_identity

STRATEGY_ID = "ranked_multi_factor_rotation"
DISPLAY_NAME = "Ranked Multi-Factor Rotation"

# Bumped whenever this module's formulas, defaults, or schema change in a
# way that could alter results. 0.0.1: scaffold only — no factor
# formulas, weights, universe, or selection/construction logic defined
# yet. Not comparable to any later version once real logic lands.
STRATEGY_VERSION = "0.0.1"


@dataclass(frozen=True, slots=True)
class RankedMultiFactorRotationConfig:
    """Structural configuration shell for Ranked Multi-Factor Rotation.

    Every field below is a placeholder pending the strategy's own
    specification — see
    ``research/strategies/ranked_multi_factor_rotation/docs/``. Nothing
    here should be read as a decided value until that document (and this
    docstring) says so with a cited rationale, the same way
    ``FilingMomentumMLConfig`` cites report sections for each of its
    fields.
    """

    strategy_id: str = STRATEGY_ID
    universe_id: str = "unspecified"
    rebalance_frequency: str = "unspecified"

    # Cross-sectional factors this strategy ranks on, and how they combine
    # into one composite score. Both empty until specified.
    factor_names: tuple[str, ...] = ()
    factor_weights: Mapping[str, float] = field(default_factory=dict)

    # How the ranked list becomes a portfolio (e.g. top-N equal weight,
    # top-N score-weighted, long/short). None until specified.
    top_n: int | None = None

    strategy_budget_pct: float = 1.0

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if not (0.0 <= self.strategy_budget_pct <= 1.0):
            raise ValueError(
                "strategy_budget_pct must be within [0.0, 1.0], got "
                f"{self.strategy_budget_pct!r}"
            )
        if self.top_n is not None and self.top_n < 1:
            raise ValueError(f"top_n must be >= 1 if set, got {self.top_n!r}")
        unknown_weight_keys = set(self.factor_weights) - set(self.factor_names)
        if unknown_weight_keys:
            raise ValueError(
                "factor_weights keys must be a subset of factor_names, got "
                f"unknown keys {sorted(unknown_weight_keys)!r}"
            )

    def identity(self) -> str:
        """Deterministic identity of this config's resolved values.

        Two configs with the same field values always produce the same
        identity; changing any field that could alter results changes it.
        """
        return compute_config_identity(self)
