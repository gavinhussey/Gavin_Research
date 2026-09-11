"""Multi-Factor Ranking ML — typed, validated strategy and model configuration.

Every default value here is taken directly from report_current.html
(source hash below) as the authoritative spec, cross-checked against a
separate legacy prototype repository's code (settings.py/ml_scorer.py,
read there only as supporting reference, not copied) where noted. Where
the report and that legacy code conflicted (ML_THRESHOLD/MIN_TRAIN_Q
duplicated across its settings.py and ml_scorer.py), this module is the
single source of truth going forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

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
# 0.3.0: the ML target changed from a binary "top-N performers this
# quarter" classification (HistGradientBoostingClassifier +
# predict_proba) to graded learning-to-rank (lightgbm LGBMRanker,
# objective="lambdarank") over the whole quarterly cross-section. Scores
# are now unbounded real-valued ranking margins, not calibrated
# probabilities. See docs/reproducibility_findings.md.
STRATEGY_VERSION = "0.3.0"

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
# v2: FeatureObservation gained strategy_cohort_end/
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
#
# v3 (this bump): 12 features were deleted from FEATURE_NAMES, taking the
# schema from 83 to 71 -- the 4 consensus/analyst columns
# (analyst_rating, consensus_sales_next_q, consensus_eps_next_q,
# analyst_eps_num_est; analyst_target_price was kept) and all 8
# free-cash-flow columns, after a measured importance/ablation study
# (see docs/reproducibility_findings.md, 2026-09-08). A v2 cache and a
# v2-trained model both carry 83 columns in a different order and must
# never be reused against this schema; this bump makes
# FeatureCacheIdentity.cache_key() and the model artifact identity
# change so they are rejected rather than silently misaligned.
# v4 (this bump): the five raw-level features listed in
# MultiFactorRankingMLConfig.cross_sectional_rank_features are now
# replaced, in the feature pipeline, by their percentile rank within their
# own quarterly cross-section (see
# strategies/multi_factor_ranking_ml/cross_sectional.py). The feature
# *set* is unchanged at 71 columns in the same order, but five columns'
# **values** now mean something different -- a [0, 1] relative rank rather
# than a dollar/share level. A v3 cache therefore holds raw levels and a
# v3-trained model expects raw levels; neither is valid input alongside
# this schema. This bump changes FeatureCacheIdentity.cache_key() and
# model_schema.compute_model_schema_identity(), so normalized and
# unnormalized features are rejected rather than silently mixed. See
# docs/reproducibility_findings.md (2026-09-08).
FEATURE_SCHEMA_VERSION = "4"


@dataclass(frozen=True, slots=True)
class MultiFactorRankingModelConfig:
    """``lightgbm.LGBMRanker`` (``objective="lambdarank"``) hyperparameters.

    This strategy's model is a **learning-to-rank** model, not a
    classifier: it is trained to order each quarter's entire
    cross-section by relative forward performance. The previous
    ``HistGradientBoostingClassifier`` configuration (``max_iter``,
    ``max_leaf_nodes``, ``min_samples_leaf``, ``l2_regularization``,
    ``class_weight``) is deleted, not disabled -- the fields below are the
    LightGBM equivalents, carrying the previous values forward as
    starting defaults. ``class_weight`` has no LGBMRanker analogue and is
    gone entirely: a ranker has no classes to weight. See
    ``docs/reproducibility_findings.md``.
    """

    #: How deep into each quarter's ranking LambdaRank's NDCG is computed.
    #: LightGBM's own default is 30 -- i.e. only the top 30 of a ~1,440-name
    #: cross-section contribute gradient, while this strategy is *evaluated*
    #: by whole-list rank correlation (IC). That is a direct objective/metric
    #: mismatch, so this is exposed as a tunable rather than left at an
    #: unexamined library default.
    lambdarank_truncation_level: int = 30
    #: Per-grade NDCG gains, index = relevance grade. ``None`` keeps
    #: LightGBM's exponential default (2^i - 1), which concentrates almost
    #: all gradient in the top grades; a linear schedule spreads it across
    #: the whole distribution. Length must cover the highest relevance grade
    #: produced by ``n_relevance_grades``.
    label_gain: tuple[float, ...] | None = None
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    reg_lambda: float = 0.1
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {self.n_estimators!r}")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {self.max_depth!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if self.num_leaves < 2:
            raise ValueError(f"num_leaves must be >= 2, got {self.num_leaves!r}")
        if self.min_child_samples <= 0:
            raise ValueError(
                f"min_child_samples must be > 0, got {self.min_child_samples!r}"
            )
        if self.reg_lambda < 0:
            raise ValueError(
                f"reg_lambda must be >= 0, got {self.reg_lambda!r}"
            )
        if self.lambdarank_truncation_level <= 0:
            raise ValueError(
                "lambdarank_truncation_level must be > 0, got "
                f"{self.lambdarank_truncation_level!r}"
            )
        if self.label_gain is not None:
            if len(self.label_gain) < 2:
                raise ValueError(
                    f"label_gain needs at least 2 entries, got {self.label_gain!r}"
                )
            if any(b < a for a, b in zip(self.label_gain, self.label_gain[1:])):
                raise ValueError(
                    "label_gain must be non-decreasing (a higher relevance grade "
                    f"can never be worth less), got {self.label_gain!r}"
                )

    def identity(self) -> str:
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class MultiFactorRankingMLConfig:
    """Multi-Factor Ranking ML's full strategy-level configuration.

    Field-by-field report provenance:

    - ``ml_train_years`` = 6 — **measured 2026-09-07**, not a report value
      (the report's ``ML_TRAIN_YEARS`` = 3 is superseded). A full-history
      sweep of 3/6/9/12/15/20 under the LambdaRank objective put 6y highest
      on both mean IC (0.0504) and IC information ratio (0.2951), but a
      paired per-cycle test shows 6y through 20y are **statistically
      indistinguishable** (p ≈ 0.35–0.83); only 6y-vs-3y approaches
      significance (p ≈ 0.052). 6 is therefore chosen on cost/parsimony —
      it ties the longer windows while training on the least data and
      carrying the least stale-regime exposure — not on demonstrated
      superiority. See ``docs/training_window_sweep_findings.md``.
    - ``min_train_quarters`` = 4 — **measured 2026-09-07**, not a report
      value (the report's ``MIN_TRAIN_Q`` = 8 is superseded). The same
      sweep varied this gate from 1 to 16 and every value produced
      byte-identical results for every window (same 147 measured cycles,
      same IC): the evaluation-side ``min_scored_count`` floor already
      excludes the early sparse-universe cycles this gate would block, so
      the gate is inert in practice. It is kept at a low value purely as a
      guard against a pathologically tiny training set — which matters
      only if that noise floor is ever lowered — and is explicitly **not**
      a tuned parameter.
    - ``n_relevance_grades`` = 10 — the number of rank-ordered relevance
      buckets the LambdaRank training target uses (``labeling.py``). Not a
      report value: it replaces the report's ``N_WINNERS`` = 10 binary
      top-N label entirely (see ``docs/reproducibility_findings.md``),
      keeping 10 only as a natural decile granularity.
    - ``return_cap`` = 0.50 — report §5.5 (``RETURN_CAP``)
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

    Deliberately *not* a field here: any "minimum positive labels"
    threshold. That gate (and the ``N_WINNERS`` binary top-N label it
    reused) existed only because the model was a binary classifier that
    needed both classes present to fit. Under LambdaRank there are no
    classes: the only training-viability requirement beyond
    ``min_train_quarters`` is that the relevance target is not constant
    across the training set, which ``training_dataset.py`` checks
    directly from the data and needs no configuration value.
    """

    strategy_id: str = STRATEGY_ID
    universe_id: str = "bloomberg_fundamentals_quarterly"

    ml_train_years: int = 6
    min_train_quarters: int = 4
    n_relevance_grades: int = 10
    return_cap: float = 0.50

    exclude_sectors: tuple[str, ...] = ("Materials",)

    #: Features replaced by their percentile rank within their own quarterly
    #: cross-section, scaled to [0, 1], by
    #: ``cross_sectional.apply_cross_sectional_rank_normalization`` at the end
    #: of ``feature_pipeline.run_feature_pipeline``. An explicit **per-feature
    #: policy**, never a global transform: these five are raw levels (dollars,
    #: share counts, a price) whose meaning drifts across a 6-year / ~24-quarter
    #: training window, which tree invariance to monotone transforms does not
    #: fix. Macro features are excluded by construction (they are broadcast
    #: identically to every instrument, so ranking them would collapse all six
    #: to a constant), as are the categorical, volatility, and growth/ratio
    #: blocks -- see ``cross_sectional.py`` and
    #: ``docs/reproducibility_findings.md``. Every entry must be a member of
    #: ``feature_domain.FEATURE_NAMES``; the empty tuple is legal and means
    #: "no normalization".
    cross_sectional_rank_features: tuple[str, ...] = (
        "market_cap",
        "volume",
        "net_debt",
        "adjusted_net_debt",
        "analyst_target_price",
    )

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
        if self.n_relevance_grades < 2:
            raise ValueError(
                f"n_relevance_grades must be >= 2, got {self.n_relevance_grades!r}"
            )
        if not (0.0 < self.return_cap <= 1.0):
            raise ValueError(
                f"return_cap must be within (0.0, 1.0], got {self.return_cap!r}"
            )
        # Imported locally so this module -- the base config every other
        # module in the strategy imports -- keeps no import-time dependency
        # on the feature schema module.
        from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import (
            FEATURE_NAMES,
        )

        unknown = [
            name for name in self.cross_sectional_rank_features if name not in FEATURE_NAMES
        ]
        if unknown:
            raise ValueError(
                "cross_sectional_rank_features contains name(s) not in "
                f"FEATURE_NAMES: {unknown!r} — a typo must fail loudly, never "
                "silently normalize nothing"
            )
        duplicates = sorted(
            {
                name
                for name in self.cross_sectional_rank_features
                if self.cross_sectional_rank_features.count(name) > 1
            }
        )
        if duplicates:
            raise ValueError(
                "cross_sectional_rank_features must not repeat a feature: "
                f"{duplicates!r}"
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
    directly addressing the config/cache-file identity mismatch found in
    the legacy prototype's feature cache (Stage 1 conflict analysis, item
    C2): a cache missing this metadata cannot be safely reused across
    configurations.
    """

    strategy_id: str
    strategy_version: str
    feature_schema_version: str
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
