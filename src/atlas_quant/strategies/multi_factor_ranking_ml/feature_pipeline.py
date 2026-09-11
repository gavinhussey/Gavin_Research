"""Multi-Factor Ranking ML's point-in-time feature pipeline.

Turns :class:`~...production.normalization.FundamentalsFeatureRecord`
history into :class:`~...feature_domain.FeatureObservation` rows, one per
instrument per quarterly evaluation cycle (see ``evaluation_schedule.py``).

Timing model -- deliberately not filing_momentum_ml's fixed
``earnings_lag_days`` post-quarter-end approximation: this strategy ranks
the full universe on each cycle's ``quarter_start`` (the first calendar
day of a quarter), using each instrument's own most recent fundamentals
row with ``available_date <= cutoff`` (``cutoff`` = the day before
``quarter_start``). Different instruments naturally land on different
actual fiscal quarters depending on their own filing timing -- expected
and correct, never normalized away. An instrument with no row satisfying
that cutoff yet (newly public, or a long true reporting lag) is rejected
for that cycle, never fabricated.

``features`` is assembled from three sources, matching
``feature_domain.FEATURE_NAMES``'s three blocks: the selected record's own
``features`` bag (blocks 1-2, already keyed by name), ``quarter_num``/
``sector_enc`` computed locally (block 3), and the 6 macro series (block
4) looked up via ``macro_lookup`` at the *cycle's* ``cutoff`` -- the same
cutoff for every instrument in a cycle, since macro is market-wide and
every instrument ranked together should see the same macro snapshot.

One transform is applied *after* every instrument's row is assembled,
because it needs the whole cycle at once: the five raw-level features in
``config.cross_sectional_rank_features`` are replaced by their percentile
rank within that cycle's own cross-section (``cross_sectional.py``). It
lives in ``run_feature_pipeline``, never in ``build_feature_observation``,
which sees only one instrument and so has no cross-section to rank
against.

Explicitly out of scope here: qualification, model training/prediction,
ranking/rejection (``decision_pipeline.py``), and this strategy has no
position-weighting or portfolio-construction stage at all (pure ranking
system -- see ``strategy.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Mapping, Sequence

from atlas_quant.data.point_in_time import select_point_in_time_fundamentals
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.macro import MACRO_SERIES_NAMES, MacroSeriesLookup
from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_VERSION,
    MultiFactorRankingMLConfig,
)
from atlas_quant.strategies.multi_factor_ranking_ml.cross_sectional import (
    apply_cross_sectional_rank_normalization,
)
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import (
    FEATURE_NAMES,
    FeatureObservation,
    missing_feature_names,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import FundamentalsFeatureRecord
from atlas_quant.data.records import SectorRecord
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

_NAN = float("nan")
_TREND_WINDOW = 6


@dataclass(frozen=True, slots=True)
class RejectedObservation:
    """One instrument/cycle that could not produce a feature observation, and why."""

    instrument_id: InstrumentId
    quarter_start: date
    reason: str


@dataclass(frozen=True, slots=True)
class FeaturePipelineResult:
    """Structured output of a feature-pipeline batch run — never only a DataFrame."""

    observations: tuple[FeatureObservation, ...]
    rejected: tuple[RejectedObservation, ...]
    warnings: tuple[str, ...]
    config_identity: str
    feature_schema_version: str
    feature_cache_identity: str | None

    def to_dicts(self) -> list[dict[str, object]]:
        return [obs.to_dict() for obs in self.observations]

    def to_dataframe(self):
        """Convenience conversion. pandas is a presentation detail, not this
        result's authoritative domain model — ``observations`` is."""
        import pandas as pd

        return pd.DataFrame(self.to_dicts())

    def to_model_matrix(self):
        """Convenience conversion to a plain list-of-tuples model matrix,
        in :data:`~atlas_quant.strategies.multi_factor_ranking_ml.feature_domain
        .FEATURE_NAMES` column order."""
        return [obs.to_model_row() for obs in self.observations]


def _quarter_num(quarter_end: date) -> float:
    return float((quarter_end.month - 1) // 3 + 1)


def _macro_features(macro_lookup: MacroSeriesLookup | None, cutoff: date) -> dict[str, float]:
    if macro_lookup is None:
        return {name: _NAN for name in MACRO_SERIES_NAMES}
    return {
        name: (value if (value := macro_lookup.value_as_of(name, cutoff)) is not None else _NAN)
        for name in MACRO_SERIES_NAMES
    }


def build_feature_observation(
    *,
    config: MultiFactorRankingMLConfig,
    sector_encoder: SectorEncoder,
    instrument_id: InstrumentId,
    quarter_start: date,
    cutoff: date,
    fundamentals: Sequence[FundamentalsFeatureRecord],
    macro_lookup: MacroSeriesLookup | None = None,
    feature_cache_identity: str | None = None,
) -> FeatureObservation | RejectedObservation:
    """Build one instrument's :class:`FeatureObservation` for this cycle, or
    report why not.

    Selects this instrument's own most recent fundamentals row knowable
    as of ``cutoff`` (no requirement that its ``quarter_end`` equal any
    particular calendar date -- see module docstring). Rejected only for
    genuine data insufficiency: no fundamentals row knowable yet as of
    ``cutoff``.
    """
    cutoff_dt = datetime.combine(cutoff, time.min)
    selection = select_point_in_time_fundamentals(
        fundamentals, instrument_id, cutoff=cutoff_dt, max_periods=_TREND_WINDOW
    )
    if not selection.selected:
        return RejectedObservation(
            instrument_id=instrument_id,
            quarter_start=quarter_start,
            reason="no fundamentals row knowable as of cutoff",
        )

    target = selection.selected[-1]

    audit_trail = AuditTrail()
    audit_trail = audit_trail.append(
        AuditRecord(
            stage="point_in_time_selection",
            message=f"selected {len(selection.selected)} row(s), rejected {len(selection.rejected)}",
            timestamp=cutoff_dt,
            data={
                "rejected_reasons": [r.reason for r in selection.rejected],
                "selected_fiscal_quarter_ends": [f.quarter_end.isoformat() for f in selection.selected],
                "target_quarter_end": target.quarter_end.isoformat(),
                "cutoff": cutoff.isoformat(),
                "quarter_start": quarter_start.isoformat(),
            },
        )
    )

    sector_record = SectorRecord(
        instrument_id=instrument_id,
        raw_sector=target.gics_sector,
        as_of=target.filed_at,
        provenance=DataProvenance(source="fundamentals_quarterly", as_of=target.filed_at, retrieved_at=target.filed_at),
    )
    classification = sector_encoder.classify(sector_record)

    features: dict[str, float] = {
        name: (value if value is not None else _NAN) for name, value in target.features.items()
    }
    features["quarter_num"] = _quarter_num(target.quarter_end)
    features["sector_enc"] = float(classification.sector_enc)
    features.update(_macro_features(macro_lookup, cutoff))

    missing = missing_feature_names(features)
    if missing:
        audit_trail = audit_trail.append(
            AuditRecord(
                stage="missing_data",
                message=f"{len(missing)} of {len(FEATURE_NAMES)} features missing",
                timestamp=cutoff_dt,
                data={"missing_features": list(missing)},
            )
        )

    return FeatureObservation(
        strategy_id=config.strategy_id,
        strategy_version=STRATEGY_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        instrument_id=instrument_id,
        fiscal_period=target.fiscal_period,
        quarter_end=target.quarter_end,
        filing_timestamp=target.filed_at,
        feature_timestamp=cutoff,
        data_cutoff=cutoff_dt,
        sector=classification.normalized_sector,
        features=features,
        missing_features=missing,
        provenance=(target.provenance,),
        config_identity=config.identity(),
        feature_cache_identity=feature_cache_identity,
        strategy_cohort_end=quarter_start,
        cohort_buy_timestamp=datetime.combine(quarter_start, time.min),
        audit_trail=audit_trail,
    )


def run_feature_pipeline(
    *,
    config: MultiFactorRankingMLConfig,
    sector_encoder: SectorEncoder,
    universe: Sequence[InstrumentId],
    quarter_start: date,
    cutoff: date,
    fundamentals_by_instrument: Mapping[InstrumentId, Sequence[FundamentalsFeatureRecord]],
    macro_lookup: MacroSeriesLookup | None = None,
    feature_cache_identity: str | None = None,
) -> FeaturePipelineResult:
    """Build feature observations for every instrument in ``universe``, for
    one quarterly evaluation cycle (``quarter_start``/``cutoff`` -- see
    ``evaluation_schedule.quarterly_evaluation_cycles``).

    Deterministic: iterates ``universe`` in the given order and never
    depends on dict iteration order for its own output ordering.

    Cross-sectional normalization
    -----------------------------
    Once every observation for this cycle is assembled, the features named
    in ``config.cross_sectional_rank_features`` are replaced by their
    percentile rank within **this cycle's own cross-section only**, scaled
    to ``[0, 1]`` (see
    :func:`~...cross_sectional.apply_cross_sectional_rank_normalization`).
    No other quarter's data participates in any rank, past or future, so
    the transform introduces no lookahead: a later cycle's data cannot
    change this cycle's values. NaN stays NaN (never imputed), ties share
    the average rank, and a cross-section with fewer than 2 non-missing
    values for a feature leaves that feature untouched -- percentile rank
    is undefined there, a real edge case for live single-name scoring.

    This function is the single place the transform is applied, and it is
    the one every consumer goes through -- the IC backtest runner
    (``backtest.multi_factor_ranking_runner.build_feature_results``), live
    production scoring (``production.orchestration.run_current_ranking``,
    both its training-history and current-cycle calls), and the
    ``build-features`` CLI command -- so live and backtest normalize identically
    by construction. Keep it that way: never normalize in a caller.
    """
    observations: list[FeatureObservation] = []
    rejected: list[RejectedObservation] = []
    warnings: list[str] = []

    for instrument_id in universe:
        result = build_feature_observation(
            config=config,
            sector_encoder=sector_encoder,
            instrument_id=instrument_id,
            quarter_start=quarter_start,
            cutoff=cutoff,
            fundamentals=fundamentals_by_instrument.get(instrument_id, ()),
            macro_lookup=macro_lookup,
            feature_cache_identity=feature_cache_identity,
        )
        if isinstance(result, RejectedObservation):
            rejected.append(result)
        else:
            observations.append(result)
            if result.missing_features:
                warnings.append(
                    f"{result.instrument_id.symbol} {quarter_start.isoformat()}: "
                    f"{len(result.missing_features)} missing feature(s)"
                )

    normalized = apply_cross_sectional_rank_normalization(
        observations, config.cross_sectional_rank_features
    )

    return FeaturePipelineResult(
        observations=normalized,
        rejected=tuple(rejected),
        warnings=tuple(warnings),
        config_identity=config.identity(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_cache_identity=feature_cache_identity,
    )
