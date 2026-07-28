"""Production feature/label build orchestration for Filing Momentum ML.

Builds :class:`~atlas_quant.strategies.filing_momentum_ml.feature_domain
.FeatureObservation` rows and labeled-quarter datasets from *normalized*
Stage 3 records over genuine historical periods. This module never
recomputes a feature or label formula itself — it validates the raw
inputs (:mod:`.validation`), then calls Stage 3's
:func:`~atlas_quant.strategies.filing_momentum_ml.feature_pipeline
.run_feature_pipeline` and Stage 3's feature cache verbatim for features,
and the same two public functions the Stage 7 backtest runner itself
calls internally
(:func:`~atlas_quant.strategies.filing_momentum_ml.forward_return
.build_forward_return_outcome`,
:func:`~atlas_quant.strategies.filing_momentum_ml.labeling
.assign_quarterly_labels`) for labels, in the same call pattern, so a
standalone label build and the backtest runner's own internal labeling
can never silently disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Mapping, Sequence

from atlas_quant.backtest.clock import BacktestPeriod
from atlas_quant.data.point_in_time import FilingTimingMode, TradingCalendar
from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.filing_momentum_ml.config import (
    FEATURE_SCHEMA_VERSION,
    FeatureCacheIdentity,
    FilingMomentumMLConfig,
)
from atlas_quant.strategies.filing_momentum_ml.feature_cache import write_feature_cache
from atlas_quant.strategies.filing_momentum_ml.feature_domain import FeatureObservation
from atlas_quant.strategies.filing_momentum_ml.feature_pipeline import FeaturePipelineResult, run_feature_pipeline
from atlas_quant.strategies.filing_momentum_ml.forward_return import build_forward_return_outcome
from atlas_quant.strategies.filing_momentum_ml.labeling import assign_quarterly_labels
from atlas_quant.strategies.filing_momentum_ml.production.validation import (
    DataValidationIssue,
    DataValidationSummary,
    validate_filings,
    validate_prices,
    validate_sectors,
)
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder
from atlas_quant.strategies.filing_momentum_ml.training_dataset import LabeledObservation


@dataclass(frozen=True, slots=True)
class ProductionFeatureBuildResult:
    """A production feature-build's complete, structured outcome."""

    feature_pipeline_result: FeaturePipelineResult
    validation_summary: DataValidationSummary
    cache_path: Path | None
    build_identity: str
    blocked: bool
    blocked_reason: str | None = None


def build_production_features(
    *,
    config: FilingMomentumMLConfig,
    calendar: TradingCalendar,
    sector_encoder: SectorEncoder,
    targets: Sequence[tuple[InstrumentId, date, datetime]],
    filings_by_instrument: Mapping[InstrumentId, Sequence[FilingFundamentals]],
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]],
    sector_by_instrument: Mapping[InstrumentId, SectorRecord],
    cache_identity: FeatureCacheIdentity,
    cache_root: Path | None = None,
    mode: FilingTimingMode = "training",
) -> ProductionFeatureBuildResult:
    """Validate raw inputs, then run Stage 3's feature pipeline over them unchanged.

    ``targets`` is ``(instrument_id, strategy_cohort_end,
    cohort_buy_timestamp)`` triples -- each shared cohort carries its own
    point-in-time cutoff (see :func:`~...feature_pipeline.run_feature_pipeline`),
    never one batch-wide cutoff shared across every cohort.

    A ``FATAL``-severity validation issue blocks the build entirely (no
    feature pipeline call, no cache write) — the caller must inspect
    ``validation_summary`` before treating an empty result as "no data."
    Passing ``cache_root`` persists the result via Stage 3's own
    ``write_feature_cache``; passing ``None`` (the default) never touches
    disk, which is what every non-production caller (tests, dry runs)
    should do.
    """
    target_instrument_ids = {instrument_id for instrument_id, _, _ in targets}

    issues: list[DataValidationIssue] = []
    for filings in filings_by_instrument.values():
        issues.extend(validate_filings(filings))
    # Validated per required target instrument (not merely per key present
    # in prices_by_instrument) so an instrument entirely missing from the
    # mapping is still reported as a FATAL empty-series issue, not silently
    # skipped.
    for instrument_id in target_instrument_ids:
        issues.extend(validate_prices(prices_by_instrument.get(instrument_id, ())))
    issues.extend(validate_sectors(list(sector_by_instrument.values())))
    summary = DataValidationSummary(issues=tuple(issues))

    if summary.has_fatal:
        empty_result = FeaturePipelineResult(
            observations=(),
            rejected=(),
            warnings=("blocked: fatal-severity data validation issue(s) present",),
            config_identity=config.identity(),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_cache_identity=cache_identity.cache_key(),
        )
        return ProductionFeatureBuildResult(
            feature_pipeline_result=empty_result,
            validation_summary=summary,
            cache_path=None,
            build_identity=cache_identity.cache_key(),
            blocked=True,
            blocked_reason="fatal-severity data validation issue(s) present",
        )

    result = run_feature_pipeline(
        config=config,
        calendar=calendar,
        sector_encoder=sector_encoder,
        targets=targets,
        filings_by_instrument=dict(filings_by_instrument),
        prices_by_instrument=dict(prices_by_instrument),
        sector_by_instrument=dict(sector_by_instrument),
        mode=mode,
        feature_cache_identity=cache_identity.cache_key(),
    )

    cache_path = None
    if cache_root is not None:
        cache_path = write_feature_cache(cache_root, cache_identity, result.observations)

    return ProductionFeatureBuildResult(
        feature_pipeline_result=result,
        validation_summary=summary,
        cache_path=cache_path,
        build_identity=cache_identity.cache_key(),
        blocked=False,
    )


@dataclass(frozen=True, slots=True)
class ProductionLabelBuildResult:
    """A production label-build's complete, structured outcome, one quarter's
    :class:`LabeledObservation` sequence per target quarter."""

    labeled_by_quarter: Mapping[date, tuple[LabeledObservation, ...]]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_production_labels(
    *,
    periods: Sequence[BacktestPeriod],
    observations_by_quarter: Mapping[date, Sequence[FeatureObservation]],
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]],
    n_winners: int,
) -> ProductionLabelBuildResult:
    """Compute each period's forward-return outcomes and quarterly labels.

    Mirrors, call-for-call, the same two Stage 6 functions and argument
    pattern the Stage 7 runner's own internal ``_build_labeled_quarters``
    uses — this exists so a standalone research label-build step (e.g.
    the ``build-labels`` CLI command or a research notebook) produces
    labels identical to what a full backtest run would compute internally,
    never a second, divergent labeling path.
    """
    labeled: dict[date, tuple[LabeledObservation, ...]] = {}
    warnings: list[str] = []

    for period in periods:
        observations = observations_by_quarter.get(period.quarter_end, ())
        outcomes = [
            build_forward_return_outcome(
                obs.instrument_id,
                period.quarter_end,
                obs.feature_timestamp,
                period.exit_timestamp.date(),
                prices_by_instrument.get(obs.instrument_id, ()),
                period.exit_timestamp,
            )
            for obs in observations
        ]
        labeling = assign_quarterly_labels(outcomes, period.quarter_end, n_winners=n_winners)
        label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
        labeled[period.quarter_end] = tuple(
            LabeledObservation(obs, label_by_id[obs.instrument_id], period.label_availability_cutoff)
            for obs in observations
        )
        if labeling.small_quarter:
            warnings.append(f"{period.quarter_end.isoformat()}: small quarter (fewer than n_winners valid outcomes), no positive labels assigned")
        missing_price_count = sum(1 for o in outcomes if o.missing_reason is not None)
        if missing_price_count:
            warnings.append(f"{period.quarter_end.isoformat()}: {missing_price_count} instrument(s) missing entry/exit price data")

    return ProductionLabelBuildResult(labeled_by_quarter=labeled, warnings=tuple(warnings))
