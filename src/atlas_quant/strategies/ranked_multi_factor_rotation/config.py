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
- ``rank_direction_mode`` = "desirable_first" — spec §3. **Confirmed
  correction, 2026-08-05, not provisional**: because §4/§5 select the
  *lowest* Total Rank, rank 1 must be the most desirable value for every
  factor (highest M, lowest V, lowest C) for "lowest wins" to actually
  reward desirable assets. See the dedicated provenance note below this
  docstring for the full correction history, including why the
  previous, opposite convention (preserved as ``"legacy_desirable_last"``)
  is kept rather than deleted.

Deliberately *not* a field here yet (see spec §7/§8): any risk-weighted
allocation alternative to flat ``position_weight``, and transaction
cost/slippage assumptions -- both are explicitly open questions in the
spec, not decided rules, so no default is guessed for either.

## Provisional RAAM Total Rank fields (2026-08-05, research scaffolding only)

The fields below (``cash_proxy_symbol``, ``absolute_momentum_model``,
``absolute_momentum_lookback_sessions``, ``total_rank_divisor``,
``weight_model``, ``fixed_weight_artifact_id``) exist to carry a
**provisional, testable alternate Total Rank specification** documented in
full at spec §4A -- ``TotalRank = (wM*Rank(M) + wV*Rank(V) + wC*Rank(C) -
T + M) / X`` (the *entire* numerator divided by ``X``, never just ``M``).
This is a research assumption adopted 2026-08-05, not a confirmed
original-author parameter, and it is **not wired into any calculation
yet**: :func:`formulas.total_rank` and the rest of the pipeline are
unchanged by this stage, so every default below reproduces the exact
behavior already documented above with zero output difference. See spec
§4A and ``docs/reproducibility_findings.md`` for full provenance.

**2026-08-05 data-plumbing update:** the SHY-relative excess absolute
momentum this spec's ``M`` term requires now has a pure, tested
implementation -- :func:`formulas.excess_absolute_momentum_at` /
:func:`pipeline.compute_excess_momentum_snapshot` -- reachable via
``absolute_momentum_model="asset_minus_cash"`` and
``absolute_momentum_lookback_sessions``.

**2026-08-05 activation update:** ``absolute_momentum_model=
"asset_minus_cash"`` is now a selectable, implemented value --
``pipeline.compute_factor_snapshot`` routes its ``"momentum"`` column
(and therefore momentum ranking and cash-gate allocation) through
:func:`formulas.excess_absolute_momentum_at` when this mode is active,
per spec §4A's ``M_i,t = R_asset_i,4m - R_SHY,4m``. **Default remains
``"price_relative"``** (the original, unmodified
``P_t/P_{t-lookback}-1`` calculation, spec §2.1) -- this is an explicit
opt-in research mode, not a change to canonical/default output. Total
Rank's own formula (:func:`formulas.total_rank`) still does not include
an ``M`` term either way, and factor weights are unchanged -- see spec
§4A's still-open items.

## Factor rank direction correction (2026-08-05, confirmed -- not provisional)

``rank_direction_mode`` (default ``"desirable_first"``) is a **bug fix
to the existing §3/§4 ranking, not a §4A research assumption**: with
``select_top_n`` selecting the *lowest* Total Rank
(``wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T``, all weights positive), rank 1
must be each factor's most desirable value or "lowest wins" does not
actually reward desirable assets. Every rank direction in this codebase
before this correction did the opposite -- highest M got the *highest*
rank number, lowest V/C got the *highest* rank number -- see
:func:`pipeline.select_for_month_end`'s docstring and
``docs/reproducibility_findings.md`` for the full before/after
derivation and the real-data forensic re-check.

**This flips the pipeline's default selection output.** Confirmed with
the user 2026-08-05 after surfacing a material finding: applying the
corrected direction to the primary source's own published 2017-11-28
worked example changes the canonical selection's overlap with the
published holdings from 3/5 (the pre-correction direction) to 1/5 (the
corrected direction) -- i.e. the logically-required direction scores
*worse* on the only empirical check this repository has ever produced.
The user chose to proceed with the correction as the new default
regardless (the mechanism-design argument for "lowest Total Rank wins"
requiring rank 1 = desirable is not conditional on matching this one
worked example, and the unresolved weights/tie-breaker term mean an
exact match was never expected either way). The pre-correction
direction is fully preserved, not deleted, as
``rank_direction_mode="legacy_desirable_last"`` -- selectable explicitly
for forensic/backward comparison, the same preservation convention
already used for ``trend_model="legacy_symmetric"`` and
``formulas.legacy_highest_total_rank_select``.

## Full provisional Total Rank formula activation (2026-08-06)

``total_rank_formula`` (default ``"legacy"``) selects which Total Rank
calculation ``pipeline.select_for_month_end`` uses:

- ``"legacy"`` (**default, unchanged behavior**): the existing
  three-term :func:`formulas.total_rank`
  (``wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T``), exactly as before this
  update.
- ``"full_provisional"`` (opt-in research mode): the full spec §4A
  formula, :func:`formulas.provisional_total_rank_score` --
  ``(wM*Rank(M)+wV*Rank(V)+wC*Rank(C)-T+M) / X`` with the *entire*
  numerator divided by ``total_rank_divisor`` (never just ``M``).
  Requires ``absolute_momentum_model="asset_minus_cash"`` (validated
  below) since this formula's ``M`` term is specifically SHY-relative
  four-month absolute momentum, not §2.1's plain price-relative
  momentum. Factor weights remain equal-thirds (``weight_model="equal"``,
  unaffected by this field); as of the fixed-weight activation below,
  ``weight_model="fixed_estimated"`` may also supply factor weights here
  (still no ``+M``/``/X`` change to which formula is used, only which
  weights feed it).

This is still a provisional research assumption, not a confirmed
original-author parameter -- see spec §4A and
``docs/reproducibility_findings.md`` for the full derivation and the
required hand-calculation check.

- ``"faa_faithful_candidate"`` (2026-08-08, opt-in, bounded research
  experiment): :func:`formulas.faa_faithful_candidate_total_rank_score`
  -- the same whole-numerator/``/X`` construction as
  ``"full_provisional"``, but with two source-parity assumptions
  swapped in per an external forensic audit's highest-information
  candidate: factor weights fixed at ``wM=1.0, wV=0.5, wC=0.5``
  (overriding ``momentum_weight``/``volatility_weight``/
  ``correlation_weight``, not reading them), and ``M`` entered in
  percentage points rather than decimal (e.g. 8.5% enters as ``8.5``,
  not ``0.085``). Requires ``absolute_momentum_model="price_relative"``
  (validated below) -- this candidate's ``M`` is the existing plain
  4-month ROC, not SHY-relative excess momentum. **Result: NO-GO** --
  see ``docs/reproducibility_findings.md`` for the full evaluation. The
  percentage-point scaling was found to invert the momentum term's sign
  contribution (large positive M hurts a ticker's score, large negative
  M helps it, under lowest-Total-Rank selection) once its magnitude
  dominates the rank terms.
- ``"faa_faithful_candidate_decimal_m"`` (2026-08-08, opt-in, follow-up
  bounded test): :func:`formulas.faa_faithful_candidate_decimal_momentum_total_rank_score`
  -- isolates ``"faa_faithful_candidate"``'s two assumptions from each
  other after that candidate's rejection was traced specifically to its
  percentage-point ``M`` scaling, not its weights. Same fixed
  ``wM=1.0, wV=0.5, wC=0.5`` weights, but ``M`` stays in decimal
  (unscaled, same magnitude as ``"full_provisional"``'s own ``M``) --
  isolating the weight hypothesis alone. Also requires
  ``absolute_momentum_model="price_relative"``. See
  ``docs/reproducibility_findings.md`` for the evaluation.

## Fixed empirically-estimated weight mode: activated, then reverted (2026-08-06)

``weight_model="fixed_estimated"`` was briefly made selectable and
operational earlier this same day -- ``pipeline.resolve_factor_weights``
would load and compatibility-check a frozen weight artifact (path in
``fixed_weight_artifact_id``) at evaluation time, per
:mod:`atlas_quant.strategies.ranked_multi_factor_rotation.frozen_weight_artifact`.
**That activation has since been reverted**, after the empirical weight
investigation it enabled (spec §4D-§4H) concluded there is no reliable
evidence to support unequal factor weights:

- The first estimator (non-negative ridge, independent per-factor
  coefficients normalized after fitting) produced a **degenerate
  1/0/0 corner solution** on the real historical panel -- not a strong
  momentum signal, but an artifact of normalizing a near-null signal
  after a non-negativity clip.
- A second, structurally different estimator (separating overall
  signal strength from relative weight, spec §4H) **independently
  confirmed** the same conclusion on the same data: signal strength
  ``s=0.0``, mean out-of-sample MSE improvement over a null
  (constant-prediction) model ``0.0%``, classified ``"low_confidence"``,
  and reported equal-thirds itself, both by construction and empirically.

**Current, adopted provisional production decision**: ``wM=wV=wC=1/3``
(this config's own defaults) is used as a **neutral fallback**, not
because it is a confirmed original-author value (the primary source
discloses no numeric weights anywhere) but because empirical estimation
found no reliable evidence justifying a deviation from it. Both
``"fixed_estimated"`` and ``"walk_forward_estimated"`` are rejected at
construction (below) -- **only ``"equal"`` is selectable today.**
``momentum_weight``/``volatility_weight``/``correlation_weight`` remain
this config's own fields, always the active source of truth.

**Nothing about the two estimators, their artifacts, or their tests was
deleted.** Both remain in the repository, fully reproducible and
independently loadable/validatable as research diagnostics
(``frozen_weight_artifact.load_frozen_weight_artifact``/
``simplex_weight_estimation.load_simplex_weight_artifact``,
each callable directly without touching this config or
``pipeline.py`` at all) -- the config-level rejection above is what
keeps production scoring from reaching them, not their absence from the
codebase. See ``docs/reproducibility_findings.md`` for the full
investigation, both artifacts' exact values, and this decision's
complete rationale.
"""

from __future__ import annotations

import math
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

    rank_direction_mode: str = "desirable_first"

    # -- Provisional RAAM Total Rank scaffolding (spec §4A, 2026-08-05). --
    # Not consumed by any calculation yet -- see module docstring. Every
    # default below is inert / reproduces current behavior exactly.
    cash_proxy_symbol: str = "SHY"
    absolute_momentum_model: str = "price_relative"
    absolute_momentum_lookback_sessions: int = 84
    total_rank_divisor: float = 11.0
    weight_model: str = "equal"
    fixed_weight_artifact_id: str | None = None
    total_rank_formula: str = "legacy"

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
        for name, value in (
            ("momentum_weight", self.momentum_weight),
            ("volatility_weight", self.volatility_weight),
            ("correlation_weight", self.correlation_weight),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
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
        if self.rank_direction_mode not in ("desirable_first", "legacy_desirable_last"):
            raise ValueError(
                "rank_direction_mode must be 'desirable_first' or "
                f"'legacy_desirable_last', got {self.rank_direction_mode!r}"
            )
        if not self.cash_proxy_symbol:
            raise ValueError("cash_proxy_symbol must be non-empty")
        if self.cash_proxy_symbol in self.ranked_tickers:
            raise ValueError(
                f"cash_proxy_symbol {self.cash_proxy_symbol!r} must not also appear in "
                "ranked_tickers -- it is a momentum benchmark/cash proxy, never a ranking "
                "candidate"
            )
        if self.absolute_momentum_model not in ("price_relative", "asset_minus_cash"):
            raise ValueError(
                "absolute_momentum_model must be 'price_relative' or 'asset_minus_cash', "
                f"got {self.absolute_momentum_model!r}"
            )
        if self.absolute_momentum_lookback_sessions <= 0:
            raise ValueError(
                "absolute_momentum_lookback_sessions must be > 0, got "
                f"{self.absolute_momentum_lookback_sessions!r}"
            )
        if not math.isfinite(self.total_rank_divisor):
            raise ValueError(f"total_rank_divisor must be finite, got {self.total_rank_divisor!r}")
        if self.total_rank_divisor <= 0:
            raise ValueError(f"total_rank_divisor must be > 0, got {self.total_rank_divisor!r}")
        if self.weight_model not in ("equal", "fixed_estimated", "walk_forward_estimated"):
            raise ValueError(
                "weight_model must be one of 'equal', 'fixed_estimated', "
                f"'walk_forward_estimated', got {self.weight_model!r}"
            )
        if self.weight_model == "walk_forward_estimated":
            raise ValueError(
                "weight_model='walk_forward_estimated' is not available -- no "
                "walk-forward re-estimation-in-production stage has been implemented; "
                "only 'equal' is selectable today"
            )
        if self.weight_model == "fixed_estimated":
            raise ValueError(
                "weight_model='fixed_estimated' is not available in production -- the "
                "empirical weight investigation (spec §4D-§4H) found no reliable evidence "
                "for unequal factor weights (the ridge estimator produced a degenerate "
                "1/0/0 corner solution from a near-null signal; the signal-strength- "
                "separated estimator independently confirmed the signal is indistinguishable "
                "from null and reported equal-thirds itself). Only 'equal' is selectable "
                "today -- see docs/reproducibility_findings.md. Both estimated-weight "
                "artifacts, and the code that produced/loads/validates them, remain in the "
                "repository as reproducible research diagnostics "
                "(frozen_weight_artifact.py/simplex_weight_estimation.py); this gate is what "
                "keeps them out of production scoring, not their absence."
            )
        if self.fixed_weight_artifact_id is not None and self.weight_model != "fixed_estimated":
            raise ValueError(
                "fixed_weight_artifact_id may only be set when weight_model="
                f"'fixed_estimated', got weight_model={self.weight_model!r} with "
                f"fixed_weight_artifact_id={self.fixed_weight_artifact_id!r}"
            )
        if self.total_rank_formula not in (
            "legacy",
            "full_provisional",
            "faa_faithful_candidate",
            "faa_faithful_candidate_decimal_m",
        ):
            raise ValueError(
                "total_rank_formula must be 'legacy', 'full_provisional', "
                "'faa_faithful_candidate', or 'faa_faithful_candidate_decimal_m', got "
                f"{self.total_rank_formula!r}"
            )
        if self.total_rank_formula == "full_provisional" and self.absolute_momentum_model != "asset_minus_cash":
            raise ValueError(
                "total_rank_formula='full_provisional' requires absolute_momentum_model="
                "'asset_minus_cash' -- spec §4A's M term is SHY-relative four-month "
                f"absolute momentum, not 'price_relative', got absolute_momentum_model="
                f"{self.absolute_momentum_model!r}"
            )
        if (
            self.total_rank_formula
            in ("faa_faithful_candidate", "faa_faithful_candidate_decimal_m")
            and self.absolute_momentum_model != "price_relative"
        ):
            raise ValueError(
                f"total_rank_formula={self.total_rank_formula!r} requires "
                "absolute_momentum_model='price_relative' -- this candidate's M term is "
                "the existing plain 4-month ROC (spec §2.1), not SHY-relative excess "
                f"momentum, got absolute_momentum_model={self.absolute_momentum_model!r}"
            )

    def identity(self) -> str:
        """Deterministic identity of this config's resolved values.

        Two configs with the same field values always produce the same
        identity; changing any field that could alter results changes it.
        """
        return compute_config_identity(self)
