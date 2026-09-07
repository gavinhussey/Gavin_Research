"""Multi-Factor Ranking ML — typed, validated strategy and model configuration.

Every default value here is taken directly from report_current.html
(source hash below) as the authoritative spec, cross-checked against a
separate legacy prototype repository's code (settings.py/ml_scorer.py,
read there only as supporting reference, not copied) where noted. Where
the report and that legacy code conflicted (ML_THRESHOLD/MIN_TRAIN_Q
duplicated across its settings.py and ml_scorer.py; fcf_mode default vs.
its on-disk feature-cache mismatch), this module is the single source of
truth going forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from atlas_quant.config.identity import compute_config_identity

STRATEGY_ID = "multi_factor_ranking_ml"
DISPLAY_NAME = "Multi-Factor Ranking ML"

# Bumped whenever this module's formulas, defaults, or schema change in a
# way that could alter results. Not the same as the platform version.
#
# 0.2.0: cloned from filing_momentum_ml, then diverged into a pure
# ranking system -- qualification threshold, position sizing, and the
# ETF fallback sleeve were all deleted entirely (see
# MultiFactorRankingMLConfig's docstring), not carried over in any form.
STRATEGY_VERSION = "0.2.0"

# sha256 of ~/Downloads/report_current.html at the time this config was
# written, so any future drift between this module and the report it was
# derived from is detectable rather than assumed away.
SOURCE_REPORT_SHA256 = (
    "c985c6ed2f85d4cfa7e4ea13449b5e05d571b2b004935eb997c42a975f787295"
)

# Bumped whenever the 17-feature schema (§3 of the report) changes shape,
# or FeatureObservation's own row shape/identity changes -- independent of
# STRATEGY_VERSION, since a cache built under one feature schema is never
# valid input for a model expecting a different one.
#
# v2 (this bump): FeatureObservation gained strategy_cohort_end/
# cohort_buy_timestamp, and build_feature_observation's inclusion rule
# changed from "the selected filing's quarter_end must equal the target
# cohort's calendar date" (an implementation bug -- synthetic fixtures are
# always calendar-aligned, so this was invisible until real data, where
# most issuers use 52/53-week or otherwise offset fiscal years) to the
# recovered report/legacy behavior: every ticker gets one row per shared
# cohort from its most-recently-knowable fiscal history, with exact
# calendar alignment used only to refine entry-timing precision, never as
# an inclusion requirement. A v1 cache reflects the old, buggy inclusion
# rule and must never be read as if it were a v2 cache -- this bump
# ensures FeatureCacheIdentity's own identity changes so v1 caches are
# rejected, not silently reused.
FEATURE_SCHEMA_VERSION = "2"

FcfMode = Literal["ratio", "raw"]

_VALID_FCF_MODES: tuple[FcfMode, ...] = ("ratio", "raw")


@dataclass(frozen=True, slots=True)
class MultiFactorRankingModelConfig:
    """HistGradientBoostingClassifier hyperparameters (report §4.5)."""

    max_iter: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.1
    class_weight: str = "balanced"
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter!r}")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {self.max_depth!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if self.max_leaf_nodes < 2:
            raise ValueError(f"max_leaf_nodes must be >= 2, got {self.max_leaf_nodes!r}")
        if self.min_samples_leaf <= 0:
            raise ValueError(
                f"min_samples_leaf must be > 0, got {self.min_samples_leaf!r}"
            )
        if self.l2_regularization < 0:
            raise ValueError(
                f"l2_regularization must be >= 0, got {self.l2_regularization!r}"
            )

    def identity(self) -> str:
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class MultiFactorRankingMLConfig:
    """Multi-Factor Ranking ML's full strategy-level configuration.

    Field-by-field report provenance:

    - ``ml_train_years`` = 3 — report §4.4 (``ML_TRAIN_YEARS``)
    - ``min_train_quarters`` = 8 — report §4.4 (``MIN_TRAIN_Q``)
    - ``n_winners`` = 10 — report §4.2 (``N_WINNERS``)
    - ``return_cap`` = 0.50 — report §5.5 (``RETURN_CAP``)
    - ``fcf_mode`` = "ratio" — report §3.1, the stated production default
    - ``exclude_sectors`` = ("Materials",) — report §5.2 (a sector-eligibility
      exclusion applied before ranking, independent of any qualification bar)

    This strategy is a **pure ranking system, not a portfolio-construction
    one**: given a set of scored candidates, it validates them, applies
    the sector exclusion, and ranks every survivor by descending score --
    it never qualifies a subset against a threshold, never sizes
    positions, never allocates capital, and never falls back to an ETF
    sleeve. Accordingly this config carries none of
    filing_momentum_ml's ``ml_threshold``/``min_positions``/
    ``max_positions``/``deployable_pct``/``earnings_lag_days``/
    ``fallback_tickers``/``fallback_dynamic_weight``/
    ``fallback_lookback_quarters``/``strategy_budget_pct`` fields --
    deleted entirely (not disabled/defaulted-away), since none of those
    concepts apply to a ranking-only strategy. See
    ``docs/reproducibility_findings.md`` and
    ``docs/strategy_decision_specification.md`` for the pure-ranking
    decision sequence this replaced those with.

    Evaluation timing is likewise not a fixed post-quarter-end lag (the
    report's ``earnings_lag_days`` = 42 day cap, approximating "give
    companies time to file"): this strategy ranks the full universe on
    the first calendar day of every quarter, using each instrument's own
    most recent fundamentals row with ``available_date`` on or before the
    prior day -- a genuine point-in-time cutoff, not an approximation. See
    ``evaluation_schedule.py``.

    Deliberately *not* a field here: a "minimum positive labels" threshold.
    ``report_current.html`` defines only ``N_WINNERS = 10`` (positive
    labels assigned per quarter, §4.2) and ``MIN_TRAIN_Q = 8`` (quarters of
    history required before a model is usable, §4.4) — no separate,
    independently-valued "minimum positive labels" parameter is named
    anywhere in the report. The legacy prototype's ``ml_scorer.py`` does
    gate model fitting on ``train["label"].sum() < N_WINNERS``
    (``fit_for_quarter``), but that reuses ``N_WINNERS`` rather than
    defining an independent constant — so this config schema already
    represents that same requirement via ``n_winners``; Stage 5's training
    gate should read ``config.n_winners`` for this check, not a new field.
    """

    strategy_id: str = STRATEGY_ID
    universe_id: str = "bloomberg_fundamentals_quarterly"

    fcf_mode: FcfMode = "ratio"
    ml_train_years: int = 3
    min_train_quarters: int = 8
    n_winners: int = 10
    return_cap: float = 0.50

    exclude_sectors: tuple[str, ...] = ("Materials",)

    model: MultiFactorRankingModelConfig = field(default_factory=MultiFactorRankingModelConfig)

    def __post_init__(self) -> None:
        if self.ml_train_years <= 0:
            raise ValueError(
                f"ml_train_years must be > 0, got {self.ml_train_years!r}"
            )
        if self.min_train_quarters <= 0:
            raise ValueError(
                f"min_train_quarters must be > 0, got {self.min_train_quarters!r}"
            )
        if self.n_winners <= 0:
            raise ValueError(f"n_winners must be > 0, got {self.n_winners!r}")
        if not (0.0 < self.return_cap <= 1.0):
            raise ValueError(
                f"return_cap must be within (0.0, 1.0], got {self.return_cap!r}"
            )
        if self.fcf_mode not in _VALID_FCF_MODES:
            raise ValueError(
                f"fcf_mode must be one of {_VALID_FCF_MODES}, got {self.fcf_mode!r}"
            )

    def identity(self) -> str:
        """Deterministic identity of this config's resolved values.

        Two configs with the same field values (including the nested model
        config) always produce the same identity; changing any field that
        could alter results changes it.
        """
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class FeatureCacheIdentity:
    """Identity metadata a Multi-Factor Ranking ML feature cache must record.

    This is a schema, not a cache implementation — Stage 3 will build the
    actual cache writer/reader against this shape. It exists now so the
    identity contract is fixed before any cache-writing code is written,
    directly addressing the fcf_mode/cache-file mismatch found in the
    legacy prototype's feature cache (Stage 1 conflict analysis, item C2):
    a cache missing this metadata cannot be safely reused across
    configurations.
    """

    strategy_id: str
    strategy_version: str
    feature_schema_version: str
    fcf_mode: FcfMode
    train_years: int
    min_train_quarters: int
    model_config_identity: str
    universe_id: str
    data_cutoff: date
    created_at: datetime
    sector_mapping_identity: str | None = None
    price_convention: str | None = None
    filing_timing_mode: str | None = None
    provider_identity: str | None = None

    @classmethod
    def compute(
        cls,
        config: MultiFactorRankingMLConfig,
        data_cutoff: date,
        created_at: datetime,
        *,
        sector_mapping_identity: str | None = None,
        price_convention: str | None = None,
        filing_timing_mode: str | None = None,
        provider_identity: str | None = None,
    ) -> "FeatureCacheIdentity":
        return cls(
            strategy_id=config.strategy_id,
            strategy_version=STRATEGY_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            fcf_mode=config.fcf_mode,
            train_years=config.ml_train_years,
            min_train_quarters=config.min_train_quarters,
            model_config_identity=config.model.identity(),
            universe_id=config.universe_id,
            data_cutoff=data_cutoff,
            created_at=created_at,
            sector_mapping_identity=sector_mapping_identity,
            price_convention=price_convention,
            filing_timing_mode=filing_timing_mode,
            provider_identity=provider_identity,
        )

    def cache_key(self) -> str:
        """A compact key suitable for a cache filename/dict key.

        Deliberately excludes ``created_at`` (a timestamp shouldn't be part
        of a lookup key) and ``data_cutoff`` is included because caches for
        different cutoffs are not interchangeable under strict
        point-in-time rules. ``sector_mapping_identity``/``price_convention``/
        ``filing_timing_mode``/``provider_identity`` were added in Stage 3
        so a cache built under one sector-consolidation mapping, price
        convention, filing-timing policy, or data-source cannot be silently
        reused under a different one of any of those (extending this
        dataclass rather than replacing it — see Stage 2's
        ``FeatureCacheIdentity`` docstring for why the identity contract
        was fixed ahead of any cache-writing code).
        """
        return compute_config_identity(
            {
                "strategy_id": self.strategy_id,
                "strategy_version": self.strategy_version,
                "feature_schema_version": self.feature_schema_version,
                "fcf_mode": self.fcf_mode,
                "train_years": self.train_years,
                "min_train_quarters": self.min_train_quarters,
                "model_config_identity": self.model_config_identity,
                "universe_id": self.universe_id,
                "data_cutoff": self.data_cutoff,
                "sector_mapping_identity": self.sector_mapping_identity,
                "price_convention": self.price_convention,
                "filing_timing_mode": self.filing_timing_mode,
                "provider_identity": self.provider_identity,
            }
        )
