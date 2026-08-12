"""Ranked Multi-Factor Rotation — one frozen, versioned, empirically
estimated ``wM``/``wV``/``wC`` artifact (spec §4F).

This module produces **one** artifact from a clearly bounded, documented
training window (never the full dataset — a later holdout period is
deliberately never touched here), using the previous two stages'
time-series-aware ridge-alpha selection and constrained-ridge estimator
unmodified. It also validates and loads such artifacts.

**2026-08-06 activation**: ``RankedMultiFactorRotationConfig.weight_model
="fixed_estimated"`` is now operational --
``pipeline.resolve_factor_weights`` calls
:func:`load_and_validate_fixed_weight_artifact` (this module) at
*evaluation* time to load and compatibility-check the configured
artifact. This module itself still never fits anything and still never
writes to ``RankedMultiFactorRotationConfig`` -- the separation between
*producing/validating* an artifact (this module, plus
``weight_estimation.py``/``alpha_selection.py`` for the fitting
machinery) and *consuming* one (``pipeline.py``, which imports only this
module's loading/validation functions, never the estimator modules) is
deliberate (task 3/4): the production scoring path can load a frozen
number, never re-derive one.

**Empirically estimated, not author-confirmed** (task 10): every
artifact this module produces carries an explicit
``provenance_label`` disclaiming it as a research estimate derived from
real historical data via a specific, documented statistical procedure —
never a claim about the primary source's own (still entirely
undisclosed) weight values, and never a claim that these weights
outperform equal-thirds or any other configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.reporting.serialization import write_json_atomic
from atlas_quant.strategies.ranked_multi_factor_rotation.alpha_selection import (
    AlphaSelectionConfig,
    select_ridge_alpha,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import generate_monthly_periods
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.panel import (
    RmfrPanelRow,
    RmfrWeightEstimationPanel,
    build_rmfr_weight_estimation_panel,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.temporal_eligibility import (
    build_temporal_eligibility_audit,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.weight_estimation import (
    WeightEstimationConfig,
    estimate_factor_weights,
)

#: Bumped whenever the artifact's own field set/meaning changes.
FROZEN_WEIGHT_ARTIFACT_SCHEMA_VERSION = "1"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1",)

#: Bumped whenever this module's generation *logic* changes in a way
#: that could produce a different artifact from identical inputs.
_ARTIFACT_GENERATOR_VERSION = "1"

#: Every field this stage's schema requires (task 6), plus the
#: provenance label this stage adds (task 10) -- validated at load time.
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version", "strategy", "weight_model", "estimator", "training_start", "training_end",
    "feature_definition", "target", "cash_proxy", "absolute_momentum_definition",
    "absolute_momentum_units", "total_rank_divisor", "ridge_alpha", "coefficients", "weights",
    "intercept", "observation_count", "generated_at", "data_fingerprint", "code_version",
    "diagnostics", "provenance_label",
)

#: Fields whose value legitimately differs between two otherwise-identical
#: regenerations (task 7) -- excluded from deterministic byte comparisons.
NON_DETERMINISTIC_FIELDS: tuple[str, ...] = ("generated_at",)

#: The strategy identifier / target / feature-definition / momentum-
#: definition every schema_version="1" artifact must carry -- shared
#: between generation (build_frozen_weight_artifact) and consumption
#: (validate_artifact_compatible_with_config) so the two can never drift
#: apart into silently-incompatible literal strings.
EXPECTED_STRATEGY_NAME = "Ranked_Multi_Factor_Rotation"
EXPECTED_TARGET = "next_month_asset_return_minus_next_month_SHY_return"
EXPECTED_ABSOLUTE_MOMENTUM_DEFINITION = "four_month_asset_return_minus_four_month_SHY_return"
EXPECTED_ABSOLUTE_MOMENTUM_UNITS = "decimal"
EXPECTED_FEATURE_DEFINITION: dict[str, str] = {
    "momentum": "12 - Rank(M)",
    "volatility": "12 - Rank(V)",
    "correlation": "12 - Rank(C)",
}
#: schema_version="1" artifacts were always built with panels ranked
#: under this rank direction (spec §3) -- not itself a field the
#: artifact schema carries (see validate_artifact_compatible_with_config's
#: docstring), so it is checked against the *config*, not the payload.
EXPECTED_RANK_DIRECTION_MODE = "desirable_first"
#: schema_version="1" artifacts assume this many ranked tickers (the
#: "12 - Rank" in EXPECTED_FEATURE_DEFINITION is literally "n+1").
EXPECTED_N_RANKED_TICKERS = 11

#: weight_estimation.FEATURE_NAMES ("Z_momentum", ...) -> this artifact's
#: plain schema keys ("momentum", ...), per the task's exact schema.
_FEATURE_NAME_TO_ARTIFACT_KEY: dict[str, str] = {
    "Z_momentum": "momentum",
    "Z_volatility": "volatility",
    "Z_correlation": "correlation",
}

_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class FrozenWeightArtifactConfig:
    """Every input needed to build one frozen artifact, bundled and
    validated together -- the chronological split (tasks 1-4) is
    explicit here, never implicit."""

    #: First rebalance month-end panel rows are built from (inclusive).
    training_start: date
    #: The estimation cutoff (task 2/4): only rows temporally eligible
    #: as of this date (``temporal_eligibility.build_temporal_eligibility_audit``)
    #: enter the final fit -- a row whose forward return is not yet
    #: resolved by this date is excluded, exactly as it would be for a
    #: real "generate weights now" run. Any panel row dated on/after this
    #: is never touched by this stage, reserved for a later evaluation
    #: stage (task 2).
    training_end: date
    rmfr_config: RankedMultiFactorRotationConfig
    calendar: TradingCalendar
    alpha_selection_config: AlphaSelectionConfig = field(default_factory=AlphaSelectionConfig)

    def __post_init__(self) -> None:
        if self.training_end <= self.training_start:
            raise ValueError(
                f"training_end ({self.training_end!r}) must be after training_start "
                f"({self.training_start!r})"
            )
        if self.rmfr_config.absolute_momentum_model != "asset_minus_cash":
            raise ValueError(
                "FrozenWeightArtifactConfig.rmfr_config.absolute_momentum_model must be "
                "'asset_minus_cash' -- the target/M definition this artifact estimates "
                f"weights for requires it, got {self.rmfr_config.absolute_momentum_model!r}"
            )
        if self.rmfr_config.rank_direction_mode != EXPECTED_RANK_DIRECTION_MODE:
            raise ValueError(
                "FrozenWeightArtifactConfig.rmfr_config.rank_direction_mode must be "
                f"{EXPECTED_RANK_DIRECTION_MODE!r} -- this artifact schema's feature_definition "
                f"('12 - Rank(...)') assumes it, got {self.rmfr_config.rank_direction_mode!r}"
            )


@dataclass(frozen=True, slots=True)
class FrozenWeightArtifact:
    """One versioned, deterministic (modulo ``generated_at``) estimated-
    weight artifact, task 6's exact schema plus a provenance label
    (task 10)."""

    schema_version: str
    strategy: str
    weight_model: str
    estimator: str
    training_start: date
    training_end: date
    feature_definition: dict[str, str]
    target: str
    cash_proxy: str
    absolute_momentum_definition: str
    absolute_momentum_units: str
    total_rank_divisor: float
    ridge_alpha: float
    coefficients: dict[str, float]
    weights: dict[str, float]
    intercept: float
    observation_count: int
    generated_at: str
    data_fingerprint: str
    code_version: str
    diagnostics: dict[str, object]
    provenance_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "weight_model": self.weight_model,
            "estimator": self.estimator,
            "training_start": self.training_start.isoformat(),
            "training_end": self.training_end.isoformat(),
            "feature_definition": dict(self.feature_definition),
            "target": self.target,
            "cash_proxy": self.cash_proxy,
            "absolute_momentum_definition": self.absolute_momentum_definition,
            "absolute_momentum_units": self.absolute_momentum_units,
            "total_rank_divisor": self.total_rank_divisor,
            "ridge_alpha": self.ridge_alpha,
            "coefficients": dict(self.coefficients),
            "weights": dict(self.weights),
            "intercept": self.intercept,
            "observation_count": self.observation_count,
            "generated_at": self.generated_at,
            "data_fingerprint": self.data_fingerprint,
            "code_version": self.code_version,
            "diagnostics": dict(self.diagnostics),
            "provenance_label": self.provenance_label,
        }

    def deterministic_dict(self) -> dict[str, object]:
        """``to_dict()`` with every field in :data:`NON_DETERMINISTIC_FIELDS`
        removed -- what two regenerations from identical inputs must
        agree on byte-for-byte (task 7)."""
        payload = self.to_dict()
        for field_name in NON_DETERMINISTIC_FIELDS:
            payload.pop(field_name, None)
        return payload


def build_frozen_weight_artifact(
    price_frames: Mapping[str, pd.DataFrame], config: FrozenWeightArtifactConfig
) -> FrozenWeightArtifact:
    """Build one :class:`FrozenWeightArtifact` from ``price_frames``
    (ticker -> OHLC ``DataFrame``, the same shape every other RMFR
    calculation uses) and ``config``'s explicit training window.

    Reuses, unmodified: :func:`backtest_clock.generate_monthly_periods`
    (the existing monthly rebalance convention),
    :func:`panel.build_rmfr_weight_estimation_panel` (bounded to
    ``[training_start, training_end]`` -- task 2: no row dated on/after
    ``training_end`` is ever generated, let alone fit on),
    :func:`temporal_eligibility.build_temporal_eligibility_audit` (the
    anti-look-ahead gate, applied with ``estimation_as_of=training_end``
    for the final fit's row selection), :func:`alpha_selection
    .select_ridge_alpha` (task 5: time-series-aware alpha selection, no
    random K-fold), and :func:`weight_estimation.estimate_factor_weights`
    (the constrained ridge estimator) -- no factor/target/eligibility
    logic is recomputed differently here.
    """
    periods = generate_monthly_periods(config.training_start, config.training_end, config.calendar)
    panel = build_rmfr_weight_estimation_panel(price_frames, periods, config.rmfr_config)

    alpha_result = select_ridge_alpha(panel.rows, config.alpha_selection_config)

    eligibility_audit = build_temporal_eligibility_audit(panel.rows, config.training_end)
    final_fit_rows = eligibility_audit.eligible_rows

    estimation_config = WeightEstimationConfig(
        ridge_alpha=alpha_result.selected_alpha,
        solver=config.alpha_selection_config.solver,
        fit_intercept=config.alpha_selection_config.fit_intercept,
        n_ranked_tickers=config.alpha_selection_config.n_ranked_tickers,
        min_observations=config.alpha_selection_config.min_train_observations,
    )
    fit_result = estimate_factor_weights(final_fit_rows, estimation_config)

    coefficients = {
        _FEATURE_NAME_TO_ARTIFACT_KEY[name]: value for name, value in fit_result.coefficients.items()
    }
    weights = {
        _FEATURE_NAME_TO_ARTIFACT_KEY[name]: value for name, value in fit_result.normalized_weights.items()
    }

    actual_training_start, actual_training_end = fit_result.date_range

    return FrozenWeightArtifact(
        schema_version=FROZEN_WEIGHT_ARTIFACT_SCHEMA_VERSION,
        strategy=EXPECTED_STRATEGY_NAME,
        weight_model="fixed_estimated",
        estimator="nonnegative_ridge",
        training_start=actual_training_start,
        training_end=actual_training_end,
        feature_definition=dict(EXPECTED_FEATURE_DEFINITION),
        target=EXPECTED_TARGET,
        cash_proxy=config.rmfr_config.cash_proxy_symbol,
        absolute_momentum_definition=EXPECTED_ABSOLUTE_MOMENTUM_DEFINITION,
        absolute_momentum_units=EXPECTED_ABSOLUTE_MOMENTUM_UNITS,
        total_rank_divisor=config.rmfr_config.total_rank_divisor,
        ridge_alpha=fit_result.ridge_alpha,
        coefficients=coefficients,
        weights=weights,
        intercept=fit_result.intercept,
        observation_count=fit_result.observation_count,
        generated_at=datetime.now(timezone.utc).isoformat(),
        data_fingerprint=panel.identity(),
        code_version=f"rmfr-{STRATEGY_VERSION}+weight_artifact_generator-{_ARTIFACT_GENERATOR_VERSION}",
        diagnostics={
            "requested_training_start": config.training_start.isoformat(),
            "requested_training_end_estimation_as_of": config.training_end.isoformat(),
            "panel_row_count": len(panel.rows),
            "panel_eligible_row_count": sum(1 for r in panel.rows if r.eligible),
            "temporally_eligible_row_count": len(final_fit_rows),
            "alpha_selection": {
                "selected_alpha": alpha_result.selected_alpha,
                "candidate_alphas": list(config.alpha_selection_config.candidate_alphas),
                "fold_count": len(alpha_result.fold_as_of_dates),
                "tie_tolerance": alpha_result.tie_tolerance,
                "selection_rule": alpha_result.selection_rule,
                "best_mse": alpha_result.diagnostics.get("best_mse"),
            },
            "final_fit_diagnostics": dict(fit_result.diagnostics),
            "held_out_period_note": (
                "No panel row dated on or after training_end was generated or fit on -- "
                "any later period is reserved for a separate future evaluation stage, "
                "never touched here."
            ),
        },
        provenance_label=(
            "EMPIRICALLY ESTIMATED, NOT AUTHOR-CONFIRMED -- AND A DEGENERATE CORNER "
            "SOLUTION FROM A NEAR-NULL PREDICTIVE SIGNAL, NOT A RELIABLE MOMENTUM "
            "FINDING. This estimator fits three independent non-negative ridge "
            "coefficients and normalizes them (w_j = beta_j / sum(beta)); out-of-sample "
            "evidence (docs/reproducibility_findings.md) shows the underlying signal is "
            "statistically indistinguishable from a null model, so the 100% momentum "
            "weight reflects which coefficient survived non-negativity clipping with the "
            "largest (economically negligible) residual value, not a real momentum "
            "effect. Superseded as the production candidate by a second, signal-strength- "
            "separated estimator (simplex_weight_estimation.py) that reports equal-thirds "
            "on the same data; this artifact is preserved unmodified as a reproducible "
            "diagnostic only. weight_model='fixed_estimated' is not selectable in "
            "production (config.py rejects it); this artifact makes no claim of superior "
            "performance versus equal-thirds or any other weighting, and is not a "
            "disclosed or confirmed parameter from the primary source paper (which "
            "discloses no numeric wM/wV/wC values at all)."
        ),
    )


def validate_frozen_weight_artifact_payload(payload: Mapping[str, object]) -> None:
    """Raise ``ValueError`` if ``payload`` (a plain ``dict``, e.g. freshly
    loaded from JSON) is not a valid frozen-weight artifact (task 8/9):
    missing required fields, an unsupported ``schema_version``, a
    negative weight, or normalized weights that do not sum to ``1.0``
    within tolerance.
    """
    missing = [f for f in REQUIRED_ARTIFACT_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"frozen weight artifact is missing required field(s): {missing!r}")

    schema_version = payload["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported frozen weight artifact schema_version {schema_version!r} -- "
            f"supported: {SUPPORTED_SCHEMA_VERSIONS!r}"
        )

    weights = payload["weights"]
    if not isinstance(weights, Mapping) or set(weights) != {"momentum", "volatility", "correlation"}:
        raise ValueError(
            f"weights must have exactly the keys 'momentum'/'volatility'/'correlation', "
            f"got {weights!r}"
        )
    for name, value in weights.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"weights[{name!r}] must be a finite number, got {value!r}")
        if value < -_WEIGHT_TOLERANCE:
            raise ValueError(f"weights[{name!r}] is negative: {value!r}")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"weights do not sum to 1.0 within tolerance: sum={weight_sum!r}, weights={weights!r}"
        )


def validate_artifact_compatible_with_config(
    payload: Mapping[str, object], config: RankedMultiFactorRotationConfig
) -> None:
    """Task 5/6: raise ``ValueError`` if ``payload`` (already schema-
    validated via :func:`validate_frozen_weight_artifact_payload`,
    called first here too, defense in depth) is not safe to apply to
    ``config``. **Fails closed** -- every check below either passes or
    raises; there is no fallback path.

    Checked: strategy name; schema version (via
    :func:`validate_frozen_weight_artifact_payload`); the exact factor
    definitions (``feature_definition``); the SHY cash proxy
    (``cash_proxy`` must equal ``config.cash_proxy_symbol``); ``M``'s
    decimal units and its exact SHY-relative definition; ``X=11``
    (``total_rank_divisor`` must equal ``config.total_rank_divisor``);
    rank direction (``config.rank_direction_mode`` must be
    ``"desirable_first"`` -- schema_version="1" artifacts do not carry
    this as an explicit payload field, since every schema_version="1"
    artifact was built assuming it; checked against the *config*
    instead, documented as a versioned assumption of this schema
    version rather than silently trusted); the applicable universe
    (``len(config.ranked_tickers)`` must equal the 11 tickers this
    schema version's ``"12 - Rank(...)"`` feature definition assumes --
    schema_version="1" does not carry the literal ticker list, only its
    size, implicitly, via the feature definition); and training
    metadata (``training_start``/``training_end`` must be valid ISO
    dates with ``training_start <= training_end``).
    """
    validate_frozen_weight_artifact_payload(payload)

    if payload["strategy"] != EXPECTED_STRATEGY_NAME:
        raise ValueError(
            f"artifact strategy {payload['strategy']!r} != {EXPECTED_STRATEGY_NAME!r} -- "
            "this artifact was not built for Ranked Multi-Factor Rotation"
        )
    if payload["feature_definition"] != EXPECTED_FEATURE_DEFINITION:
        raise ValueError(
            f"artifact feature_definition {payload['feature_definition']!r} != "
            f"{EXPECTED_FEATURE_DEFINITION!r} -- incompatible factor definitions"
        )
    if payload["target"] != EXPECTED_TARGET:
        raise ValueError(f"artifact target {payload['target']!r} != {EXPECTED_TARGET!r}")
    if payload["cash_proxy"] != config.cash_proxy_symbol:
        raise ValueError(
            f"artifact cash_proxy {payload['cash_proxy']!r} != "
            f"config.cash_proxy_symbol {config.cash_proxy_symbol!r}"
        )
    if payload["absolute_momentum_definition"] != EXPECTED_ABSOLUTE_MOMENTUM_DEFINITION:
        raise ValueError(
            f"artifact absolute_momentum_definition {payload['absolute_momentum_definition']!r} "
            f"!= {EXPECTED_ABSOLUTE_MOMENTUM_DEFINITION!r}"
        )
    if payload["absolute_momentum_units"] != EXPECTED_ABSOLUTE_MOMENTUM_UNITS:
        raise ValueError(
            f"artifact absolute_momentum_units {payload['absolute_momentum_units']!r} != "
            f"{EXPECTED_ABSOLUTE_MOMENTUM_UNITS!r} -- M must be a decimal return, not a "
            "whole percentage point"
        )
    try:
        artifact_divisor = float(payload["total_rank_divisor"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"artifact total_rank_divisor {payload['total_rank_divisor']!r} is not numeric"
        ) from exc
    if artifact_divisor != config.total_rank_divisor:
        raise ValueError(
            f"artifact total_rank_divisor {artifact_divisor!r} != "
            f"config.total_rank_divisor {config.total_rank_divisor!r} (X mismatch)"
        )
    if config.rank_direction_mode != EXPECTED_RANK_DIRECTION_MODE:
        raise ValueError(
            f"config.rank_direction_mode {config.rank_direction_mode!r} != "
            f"{EXPECTED_RANK_DIRECTION_MODE!r} -- this artifact schema version was fit "
            "assuming rank 1 = most desirable; applying it under a different rank "
            "direction would silently invert the intended factor weighting"
        )
    if len(config.ranked_tickers) != EXPECTED_N_RANKED_TICKERS:
        raise ValueError(
            f"config.ranked_tickers has {len(config.ranked_tickers)} ticker(s), but this "
            f"artifact schema version was fit assuming {EXPECTED_N_RANKED_TICKERS} "
            "(its feature_definition's '12 - Rank(...)' is only correct for that universe size)"
        )
    try:
        training_start = date.fromisoformat(str(payload["training_start"]))
        training_end = date.fromisoformat(str(payload["training_end"]))
    except ValueError as exc:
        raise ValueError(
            f"artifact training_start/training_end are not valid ISO dates: "
            f"{payload['training_start']!r}/{payload['training_end']!r}"
        ) from exc
    if training_start > training_end:
        raise ValueError(
            f"artifact training_start {training_start!r} is after training_end {training_end!r}"
        )
    if not isinstance(payload["observation_count"], int) or payload["observation_count"] <= 0:
        raise ValueError(
            f"artifact observation_count must be a positive integer, got "
            f"{payload['observation_count']!r}"
        )


def load_and_validate_fixed_weight_artifact(
    config: RankedMultiFactorRotationConfig,
) -> dict[str, object]:
    """The fail-closed loader ``pipeline.resolve_factor_weights`` calls
    for ``config.weight_model=="fixed_estimated"`` (task 6/7). Never
    falls back to equal weights on any failure -- every error path below
    raises ``ValueError`` instead.

    Requires ``config.fixed_weight_artifact_id`` to already be set (a
    filesystem path) -- ``RankedMultiFactorRotationConfig.__post_init__``
    already enforces this at config construction, so reaching this
    function with it unset would itself be a programming error, not a
    normal failure mode; still checked explicitly rather than assumed.
    """
    if config.fixed_weight_artifact_id is None:
        raise ValueError(
            "weight_model='fixed_estimated' requires fixed_weight_artifact_id to be set"
        )
    path = Path(config.fixed_weight_artifact_id)
    if not path.exists():
        raise ValueError(
            f"fixed_weight_artifact_id path does not exist: {path!r} -- cannot load weights "
            "for weight_model='fixed_estimated' (never silently falling back to equal weights)"
        )
    try:
        payload = load_frozen_weight_artifact(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"failed to read/parse frozen weight artifact at {path!r}: {exc}"
        ) from exc
    validate_artifact_compatible_with_config(payload, config)
    return payload


def load_frozen_weight_artifact(
    path: Path, *, expected_data_fingerprint: str | None = None
) -> dict[str, object]:
    """Load and validate one frozen-weight artifact JSON file (task 8).

    Raises ``ValueError`` if the payload fails
    :func:`validate_frozen_weight_artifact_payload`, or (task 9's
    "tampering/fingerprint mismatch") if ``expected_data_fingerprint``
    is supplied and does not match the artifact's own
    ``data_fingerprint`` field exactly.
    """
    payload = json.loads(Path(path).read_text())
    validate_frozen_weight_artifact_payload(payload)
    if expected_data_fingerprint is not None and payload["data_fingerprint"] != expected_data_fingerprint:
        raise ValueError(
            "data_fingerprint mismatch -- the artifact's recorded fingerprint "
            f"({payload['data_fingerprint']!r}) does not match the expected fingerprint "
            f"({expected_data_fingerprint!r}); the artifact may be stale or the "
            "underlying data may have changed since it was generated"
        )
    return payload


def write_frozen_weight_artifact(
    artifact: FrozenWeightArtifact, path: Path, *, overwrite: bool = True
) -> Path:
    """Write ``artifact`` as deterministic (sorted-key) JSON, reusing
    this repository's existing atomic-write primitive."""
    return write_json_atomic(path, artifact.to_dict(), overwrite=overwrite)
