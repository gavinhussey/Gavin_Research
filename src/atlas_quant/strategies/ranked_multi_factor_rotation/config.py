"""Ranked Multi-Factor Rotation — typed strategy configuration.

Every default value here is taken from
``research/strategies/ranked_multi_factor_rotation/docs/specification.md``
(the equivalent of ``report_current.html`` for Filing Momentum ML) as the
authoritative spec for this strategy. Field-by-field provenance:

- ``ranked_tickers`` = the 11-asset universe — spec §1 (confirmed
  original rule: matches the primary source's 7Twelve universe exactly,
  see spec preamble for citation)
- ``cash_ticker`` = "SHY" — spec preamble/§1/§5 (confirmed with the user
  2026-08-04: SHY is held as a real position, not a synthetic 0% return)
- ``momentum_lookback_days`` = 84 — spec §2.1. **Derived implementation
  convention, not source-confirmed**: the primary source states "4
  months momentum," never an exact trading-day count.
- ``ewma_lambda`` = 0.94, ``volatility_smoothing_window`` = 10 — spec
  §2.2. **Confirmed original rule**: the primary source names this exact
  RiskMetrics-EWMA construction (λ=0.94, 10-day smoothing) as its own
  volatility method, not an approximation of something else.
- ``correlation_lookback_days`` = 84 — spec §2.3. **Derived
  implementation convention, not source-confirmed**, same caveat as
  ``momentum_lookback_days`` above.
- ``trend_model`` = "canonical_source" — spec §2.4. Selects which
  Trend/Breakout construction is used: ``"canonical_source"`` (default)
  implements the primary source's literal formula
  (:func:`formulas.canonical_source_trend_bands`); ``"legacy_symmetric"``
  is a **noncanonical, deprecated** single-lookback construction kept
  only for existing research-artifact compatibility
  (:func:`formulas.legacy_symmetric_trend_bands`) and must not be used
  as the default for anything labeled canonical RAAM.
- ``atr_window`` = 42 — spec §2.4, confirmed original rule, shared by
  both trend models.
- ``trend_upper_lookback`` = 63, ``trend_lower_lookback`` = 105 — spec
  §2.4, confirmed original rule (primary source: "Highest Close of 63
  periods" / "Highest Low of 105 periods"). Only used when
  ``trend_model == "canonical_source"``.
- ``trend_lookback_n`` = 42 — spec §2.4 legacy note (``N`` confirmed
  with the user 2026-08-04; not paper-sourced, an explicit tunable
  default). Only used when ``trend_model == "legacy_symmetric"``.
- ``momentum_weight``/``volatility_weight``/``correlation_weight`` =
  1/3 each — spec §4. **Temporary unresolved placeholder, not a
  source-confirmed value**: the primary source defines these weights'
  existence and role but discloses no numeric defaults anywhere in the
  retrieved text. Equal-thirds is this repository's own placeholder.
- ``top_n`` = 5, ``position_weight`` = 0.20 — spec §5
- ``rebalance_frequency`` = "monthly" — spec §6
- ``strategy_budget_pct`` = 1.0 — standalone-backtest default, same
  convention as ``FilingMomentumMLConfig``

Deliberately *not* a field here yet (see spec §7/§8): any risk-weighted
allocation alternative to flat ``position_weight``, and transaction
cost/slippage assumptions -- both are explicitly open questions in the
spec, not decided rules, so no default is guessed for either.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas_quant.config.identity import compute_config_identity

STRATEGY_ID = "ranked_multi_factor_rotation"
DISPLAY_NAME = "Ranked Multi-Factor Rotation"

# 0.1.0: first version with real formulas/config values (spec sections
# above). Not comparable to 0.0.1's scaffold-only placeholders.
STRATEGY_VERSION = "0.1.0"

RANKED_TICKERS: tuple[str, ...] = (
    "VV", "IJH", "IJR", "EFA", "EEM", "RWR", "VAW", "DBC", "AGG", "TIP", "IGOV",
)


@dataclass(frozen=True, slots=True)
class RankedMultiFactorRotationConfig:
    """Ranked Multi-Factor Rotation's full strategy-level configuration.

    See this module's docstring for field-by-field spec provenance.
    """

    strategy_id: str = STRATEGY_ID
    universe_id: str = "rmfr_11_etf_plus_shy"
    rebalance_frequency: str = "monthly"

    ranked_tickers: tuple[str, ...] = RANKED_TICKERS
    cash_ticker: str = "SHY"

    momentum_lookback_days: int = 84
    ewma_lambda: float = 0.94
    volatility_smoothing_window: int = 10
    correlation_lookback_days: int = 84
    atr_window: int = 42
    trend_model: str = "canonical_source"
    trend_upper_lookback: int = 63
    trend_lower_lookback: int = 105
    trend_lookback_n: int = 42

    momentum_weight: float = 1.0 / 3.0
    volatility_weight: float = 1.0 / 3.0
    correlation_weight: float = 1.0 / 3.0

    top_n: int = 5
    position_weight: float = 0.20

    strategy_budget_pct: float = 1.0

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if len(self.ranked_tickers) < 2:
            raise ValueError(
                f"ranked_tickers must contain at least 2 tickers, got {self.ranked_tickers!r}"
            )
        if len(set(self.ranked_tickers)) != len(self.ranked_tickers):
            raise ValueError(f"ranked_tickers must not contain duplicates, got {self.ranked_tickers!r}")
        if not self.cash_ticker:
            raise ValueError("cash_ticker must be non-empty")
        if self.cash_ticker in self.ranked_tickers:
            raise ValueError(
                f"cash_ticker {self.cash_ticker!r} must not also appear in ranked_tickers "
                "-- it is the cash destination, never a ranking candidate"
            )
        if self.momentum_lookback_days <= 0:
            raise ValueError(
                f"momentum_lookback_days must be > 0, got {self.momentum_lookback_days!r}"
            )
        if not (0.0 < self.ewma_lambda < 1.0):
            raise ValueError(f"ewma_lambda must be within (0.0, 1.0), got {self.ewma_lambda!r}")
        if self.volatility_smoothing_window <= 0:
            raise ValueError(
                "volatility_smoothing_window must be > 0, got "
                f"{self.volatility_smoothing_window!r}"
            )
        if self.correlation_lookback_days <= 0:
            raise ValueError(
                f"correlation_lookback_days must be > 0, got {self.correlation_lookback_days!r}"
            )
        if self.atr_window <= 0:
            raise ValueError(f"atr_window must be > 0, got {self.atr_window!r}")
        if self.trend_model not in ("canonical_source", "legacy_symmetric"):
            raise ValueError(
                "trend_model must be 'canonical_source' or 'legacy_symmetric', got "
                f"{self.trend_model!r}"
            )
        if self.trend_model == "canonical_source":
            if self.trend_upper_lookback <= 0:
                raise ValueError(
                    f"trend_upper_lookback must be > 0, got {self.trend_upper_lookback!r}"
                )
            if self.trend_lower_lookback <= 0:
                raise ValueError(
                    f"trend_lower_lookback must be > 0, got {self.trend_lower_lookback!r}"
                )
        else:
            if self.trend_lookback_n <= 0:
                raise ValueError(f"trend_lookback_n must be > 0, got {self.trend_lookback_n!r}")
        weight_sum = self.momentum_weight + self.volatility_weight + self.correlation_weight
        if not (0.999 <= weight_sum <= 1.001):
            raise ValueError(
                "momentum_weight + volatility_weight + correlation_weight must sum to "
                f"1.0, got {weight_sum!r}"
            )
        if any(
            w < 0
            for w in (self.momentum_weight, self.volatility_weight, self.correlation_weight)
        ):
            raise ValueError("factor weights must each be >= 0.0")
        if not (1 <= self.top_n <= len(self.ranked_tickers)):
            raise ValueError(
                f"top_n must be within [1, len(ranked_tickers)]={len(self.ranked_tickers)!r}, "
                f"got {self.top_n!r}"
            )
        if not (0.0 < self.position_weight <= 1.0):
            raise ValueError(
                f"position_weight must be within (0.0, 1.0], got {self.position_weight!r}"
            )
        if not (0.0 <= self.strategy_budget_pct <= 1.0):
            raise ValueError(
                "strategy_budget_pct must be within [0.0, 1.0], got "
                f"{self.strategy_budget_pct!r}"
            )

    def identity(self) -> str:
        """Deterministic identity of this config's resolved values.

        Two configs with the same field values always produce the same
        identity; changing any field that could alter results changes it.
        """
        return compute_config_identity(self)
