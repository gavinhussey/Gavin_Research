"""Typed configuration for the canonical Filing Momentum ML regime subsystem.

Every default here is taken directly from report_current.html §5b
(cross-checked against ``_markov_worker.py``/``_markov_batch_worker.py`` in
the separate legacy repository, read as supporting reference only). This
is deliberately a standalone, self-contained config — usable by
:class:`~atlas_quant.strategies.filing_momentum_ml.regime_evaluator
.RegimeEvaluator` on its own, not only nested inside
:class:`~atlas_quant.strategies.filing_momentum_ml.config
.FilingMomentumMLConfig` — because the regime subsystem is intended to be
reusable across historical/forward/single/batch/cached execution contexts
that do not all need the full strategy configuration.

``FilingMomentumMLConfig.regime_gate_mode`` (Stage 2) is unchanged and
remains the field a Filing Momentum ML strategy evaluator (Stage 5+)
should read for its own gate mode; ``RegimeConfig.gate_mode`` here is this
subsystem's own default when used standalone. Both default to ``"both"``,
the report's only actually-used production mode (§5.1: "This is the only
gate logic actually used").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from atlas_quant.config.identity import compute_config_identity

RegimeGateMode = Literal["both", "either", "markov", "hmm", "none"]
_VALID_GATE_MODES: tuple[RegimeGateMode, ...] = ("both", "either", "markov", "hmm", "none")

#: Policy for a component that could not produce a classification
#: (insufficient history, missing/invalid prices, or a numerical fit
#: failure). The report is silent on a general policy — it only states
#: the "both" mode's specific consequence when HMM is unavailable ("the
#: SPY gate's 'both Bear' condition then cannot be satisfied, since a
#: required input is absent", §5b.2). This platform generalizes that same
#: principle explicitly: an unavailable component is never itself
#: confirmed Bear, in any gate mode. There is currently only one policy;
#: the type exists so a future, different policy is a config change, not
#: a silent code change.
UnavailableComponentPolicy = Literal["never_confirms_bear"]


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Report §5b: the canonical Markov + HMM regime subsystem's parameters.

    Field-by-field report provenance:

    - ``gate_mode`` = "both" — report §5.1 ("the only gate logic actually
      used"); "either"/"markov"/"hmm"/"none" are exposed as genuine,
      real config values (not hardcoded) mirroring the same design
      decision already made for ``FilingMomentumMLConfig.regime_gate_mode``.
    - ``markov_windows`` = (63, 126, 252) — report §5b.1 (``WINDOWS``).
    - ``markov_persistence`` = 5 — report §5b.1 (``PERSISTENCE``).
    - ``markov_min_window_agreement`` = 2 — report §5b.1 ("at least 2 of
      the 3 windows independently confirm Bear").
    - ``markov_threshold_multiplier`` = 0.5, ``markov_threshold_floor`` =
      0.005 — report §5b.1: theta_w = max(0.5 * sigma_ann * sqrt(w/252), 0.005).
    - ``markov_annualization_factor`` = 252 — report §5b.1 (sqrt(252)).
    - ``markov_volatility_lookback_days`` = 63 — report §5b.1's own display
      writes sigma_ann from a fixed ``r_{-62},...,r_0`` (63 points)
      regardless of which window w is being thresholded — cross-checked
      against ``_markov_worker.py._vol_adj_threshold``, which always uses
      ``daily_ret.iloc[-63:]`` independent of ``window``. This is *not* a
      per-window volatility lookback.
    - ``hmm_state_count`` = 3, ``hmm_covariance_type`` = "diag",
      ``hmm_n_iter`` = 200, ``hmm_random_state`` = 42 — report §5b.2.
    - ``hmm_min_weekly_observations`` = 30 — report §5b.2.
    - ``hmm_weekly_anchor`` = "W-FRI" (Friday-anchored weekly close) —
      report §5b.2 ("weekly (Friday-close) observations"), cross-checked
      against ``_markov_worker.py``'s ``close.resample("W-FRI").last()``.
    - ``hmm_volatility_window_weeks`` = 4 — report §5b.2 ("4-week rolling
      volatility").

    Deliberately *not* copied from the legacy prototype: its
    ``_vol_adj_threshold`` also has an undocumented ``len(recent) < 10:
    return 0.02`` fallback and a ``len(close) < w + PERSISTENCE + 5``
    availability buffer, neither of which report_current.html states.
    Per this stage's explicit instruction to choose a conservative,
    explicit policy where the report is silent rather than invent
    historical-performance-shaped behavior, this config instead exposes
    ``markov_volatility_lookback_days`` as the explicit minimum daily-return
    sample required to compute ``sigma_ann`` at all — insufficient history
    is reported as :class:`~atlas_quant.strategies.filing_momentum_ml
    .regime_domain.ComponentAvailability.INSUFFICIENT_HISTORY`, not
    silently defaulted to an arbitrary threshold value.
    """

    gate_mode: RegimeGateMode = "both"

    markov_windows: tuple[int, ...] = (63, 126, 252)
    markov_persistence: int = 5
    markov_min_window_agreement: int = 2
    markov_threshold_multiplier: float = 0.5
    markov_threshold_floor: float = 0.005
    markov_annualization_factor: int = 252
    markov_volatility_lookback_days: int = 63

    hmm_state_count: int = 3
    hmm_covariance_type: str = "diag"
    hmm_n_iter: int = 200
    hmm_random_state: int = 42
    hmm_min_weekly_observations: int = 30
    hmm_weekly_anchor: str = "W-FRI"
    hmm_volatility_window_weeks: int = 4

    unavailable_component_policy: UnavailableComponentPolicy = "never_confirms_bear"

    def __post_init__(self) -> None:
        if self.gate_mode not in _VALID_GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {_VALID_GATE_MODES}, got {self.gate_mode!r}"
            )
        if not self.markov_windows or any(w <= 0 for w in self.markov_windows):
            raise ValueError("markov_windows must be a non-empty tuple of positive integers")
        if self.markov_persistence <= 0:
            raise ValueError(f"markov_persistence must be > 0, got {self.markov_persistence!r}")
        if not (1 <= self.markov_min_window_agreement <= len(self.markov_windows)):
            raise ValueError(
                "markov_min_window_agreement must be within "
                f"[1, {len(self.markov_windows)}], got {self.markov_min_window_agreement!r}"
            )
        if self.markov_threshold_multiplier <= 0:
            raise ValueError(
                f"markov_threshold_multiplier must be > 0, got {self.markov_threshold_multiplier!r}"
            )
        if self.markov_threshold_floor < 0:
            raise ValueError(
                f"markov_threshold_floor must be >= 0, got {self.markov_threshold_floor!r}"
            )
        if self.markov_annualization_factor <= 0:
            raise ValueError(
                f"markov_annualization_factor must be > 0, got {self.markov_annualization_factor!r}"
            )
        if self.markov_volatility_lookback_days <= 1:
            raise ValueError(
                "markov_volatility_lookback_days must be > 1, got "
                f"{self.markov_volatility_lookback_days!r}"
            )
        if self.hmm_state_count < 2:
            raise ValueError(f"hmm_state_count must be >= 2, got {self.hmm_state_count!r}")
        if self.hmm_n_iter <= 0:
            raise ValueError(f"hmm_n_iter must be > 0, got {self.hmm_n_iter!r}")
        if self.hmm_min_weekly_observations <= 0:
            raise ValueError(
                "hmm_min_weekly_observations must be > 0, got "
                f"{self.hmm_min_weekly_observations!r}"
            )
        if self.hmm_volatility_window_weeks <= 1:
            raise ValueError(
                "hmm_volatility_window_weeks must be > 1, got "
                f"{self.hmm_volatility_window_weeks!r}"
            )

    def identity(self) -> str:
        """Deterministic identity of this config's resolved values."""
        return compute_config_identity(self)
