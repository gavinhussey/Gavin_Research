"""Multi-Factor Ranking ML's point-in-time feature pipeline.

Turns provider-neutral records (:mod:`atlas_quant.data.records`) into
:class:`~atlas_quant.strategies.multi_factor_ranking_ml.feature_domain
.FeatureObservation` rows. This module's job is exclusively: point-in-time
selection, timing resolution, missing-data handling, and assembling the
result — that plumbing is generic and kept unchanged from
filing_momentum_ml's architecture. What's removed is the *concrete*
feature computation: filing_momentum_ml's 17 filing-derived formulas are
gone (see ``formulas.py``), so ``build_feature_observation`` currently
assembles an empty ``features`` mapping (matching the empty
``FEATURE_NAMES`` in ``feature_domain.py``). Once this strategy's own
feature formulas exist, compute them here (from whatever raw records the
new Bloomberg-CSV acquisition path produces) and populate ``features``
before constructing the ``FeatureObservation`` — the point-in-time cutoff
enforcement below (``feature_date > data_cutoff.date()`` is rejected)
must keep applying to however that data arrives.

Explicitly out of scope here: qualification, model training/prediction,
position weighting (including ``score_proportional_weights``, which
belongs to the later strategy stage and is deliberately never called from
this module), portfolio construction, and backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping, Sequence

from atlas_quant.data.point_in_time import (
    FilingTimingMode,
    TradingCalendar,
    resolve_feature_timestamp,
    select_point_in_time_fundamentals,
    select_point_in_time_sector,
)
from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_VERSION,
    MultiFactorRankingMLConfig,
)
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import (
    FEATURE_NAMES,
    FeatureObservation,
    missing_feature_names,
)
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder

_NAN = float("nan")
_TREND_WINDOW = 6


@dataclass(frozen=True, slots=True)
class RejectedObservation:
    """One instrument/quarter that could not produce a feature observation, and why."""

    instrument_id: InstrumentId
    quarter_end: date
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


def build_feature_observation(
    *,
    config: MultiFactorRankingMLConfig,
    calendar: TradingCalendar,
    sector_encoder: SectorEncoder,
    instrument_id: InstrumentId,
    strategy_cohort_end: date,
    cohort_buy_timestamp: datetime,
    filings: Sequence[FilingFundamentals],
    prices: Sequence[DailyPriceObservation],
    sector_record: SectorRecord | None,
    data_cutoff: datetime,
    mode: FilingTimingMode = "training",
    feature_cache_identity: str | None = None,
) -> FeatureObservation | RejectedObservation:
    """Build one instrument/shared-cohort's :class:`FeatureObservation`, or report why not.

    Recovered report/legacy behavior (``ml_scorer.py``'s ``RollingMLScorer``
    + ``data_sec.py``'s ``get_available_as_of``): every ticker produces one
    candidate row per shared ``strategy_cohort_end``, built from whichever
    fiscal history is most recently knowable as of ``data_cutoff`` --
    ordinal ("most recent N filed quarters"), never requiring the
    selected filing's own ``quarter_end`` to equal ``strategy_cohort_end``.
    This function is rejected only for genuine data insufficiency (no
    knowable fundamental history at all, or a resolved feature date past
    ``data_cutoff``) -- never for a fiscal/calendar mismatch, which is not
    a rejection condition at all, only a timing-precision one (see below).

    An earlier implementation required exact equality between the
    selected filing's ``quarter_end`` and ``strategy_cohort_end`` as an
    *inclusion* condition -- this was an implementation bug (synthetic
    test fixtures are always calendar-aligned, so the bug was invisible
    until real data, where the majority of issuers use 52/53-week or
    otherwise offset fiscal calendars). The report/legacy implementation
    never had such a requirement; exact equality there is used only to
    refine ``feature_timestamp``'s entry-timing precision.
    """
    selection = select_point_in_time_fundamentals(
        filings, instrument_id, cutoff=data_cutoff, max_periods=_TREND_WINDOW
    )
    if not selection.selected:
        return RejectedObservation(
            instrument_id=instrument_id,
            quarter_end=strategy_cohort_end,
            reason="no fundamental history knowable as of data_cutoff",
        )

    target_filing = selection.selected[-1]
    has_exact_cohort_match = target_filing.quarter_end == strategy_cohort_end

    if has_exact_cohort_match:
        # Entry-timing refinement only (report §3/§5.5): filed_at + 1
        # trading day, capped at the shared cohort's own buy timestamp.
        # ``quarter_end=strategy_cohort_end`` here (not
        # ``target_filing.quarter_end``) is deliberate -- the day-42 cap is
        # always anchored to the shared cohort's clock, never replaced by
        # the issuer's own fiscal quarter-end.
        timing = resolve_feature_timestamp(
            filed_at=target_filing.filed_at,
            quarter_end=strategy_cohort_end,
            calendar=calendar,
            earnings_lag_days=config.earnings_lag_days,
            mode=mode,
        )
        feature_date = timing.feature_date
        timing_message = (
            "exact fiscal-cohort match: day-42 cap applied" if timing.capped
            else "exact fiscal-cohort match: natural filing_date + 1 trading day used"
        )
        timing_data = {
            "cohort_match": True,
            "filed_at": timing.filed_at.isoformat(),
            "natural_feature_date": timing.natural_feature_date.isoformat(),
            "day42_cutoff_date": timing.day42_cutoff_date.isoformat(),
            "feature_date": timing.feature_date.isoformat(),
            "mode": timing.mode,
            "capped": timing.capped,
        }
    else:
        # No exact fiscal-period match to this shared cohort -- recovered
        # report/legacy fallback (never a rejection, never nearest-date
        # matching): use the cohort's own buy timestamp directly.
        feature_date = cohort_buy_timestamp.date()
        timing_message = "no exact fiscal-cohort match: cohort_buy_timestamp used directly"
        timing_data = {
            "cohort_match": False,
            "issuer_fiscal_quarter_end": target_filing.quarter_end.isoformat(),
            "strategy_cohort_end": strategy_cohort_end.isoformat(),
            "feature_date": feature_date.isoformat(),
        }

    if feature_date > data_cutoff.date():
        return RejectedObservation(
            instrument_id=instrument_id,
            quarter_end=strategy_cohort_end,
            reason="resolved feature_date falls after data_cutoff",
        )

    audit_trail = AuditTrail()
    audit_trail = audit_trail.append(
        AuditRecord(
            stage="point_in_time_selection",
            message=f"selected {len(selection.selected)} filing(s), rejected {len(selection.rejected)}",
            timestamp=data_cutoff,
            data={
                "rejected_reasons": [r.reason for r in selection.rejected],
                "selected_fiscal_quarter_ends": [f.quarter_end.isoformat() for f in selection.selected],
            },
        )
    )
    audit_trail = audit_trail.append(
        AuditRecord(
            stage="timing_resolution",
            message=timing_message,
            timestamp=data_cutoff,
            data=timing_data,
        )
    )

    # This strategy's concrete feature formulas are not yet defined (see
    # module docstring) -- FEATURE_NAMES is empty, so the only valid
    # ``features`` mapping is also empty (FeatureObservation.__post_init__
    # rejects any name not in FEATURE_NAMES). Compute real features here
    # once they exist, keyed by whatever FEATURE_NAMES then contains.
    features: dict[str, float] = {}

    raw_sector = sector_record.raw_sector if sector_record else None
    classification = sector_encoder.classify(
        sector_record
        if sector_record is not None
        else SectorRecord(
            instrument_id=instrument_id,
            raw_sector=None,
            as_of=data_cutoff,
            provenance=DataProvenance(
                source="unavailable", as_of=data_cutoff, retrieved_at=data_cutoff
            ),
        )
    )

    provenance = [target_filing.provenance] + [p.provenance for p in prices[:1]]
    if sector_record is not None:
        provenance.append(sector_record.provenance)

    missing = missing_feature_names(features)
    if missing:
        audit_trail = audit_trail.append(
            AuditRecord(
                stage="missing_data",
                message=f"{len(missing)} of {len(FEATURE_NAMES)} features missing",
                timestamp=data_cutoff,
                data={"missing_features": list(missing)},
            )
        )

    return FeatureObservation(
        strategy_id=config.strategy_id,
        strategy_version=STRATEGY_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        instrument_id=instrument_id,
        fiscal_period=target_filing.fiscal_period,
        quarter_end=target_filing.quarter_end,
        filing_timestamp=target_filing.filed_at,
        feature_timestamp=feature_date,
        data_cutoff=data_cutoff,
        sector=classification.normalized_sector,
        features=features,
        missing_features=missing,
        provenance=tuple(provenance),
        config_identity=config.identity(),
        feature_cache_identity=feature_cache_identity,
        strategy_cohort_end=strategy_cohort_end,
        cohort_buy_timestamp=cohort_buy_timestamp,
        audit_trail=audit_trail,
    )


def run_feature_pipeline(
    *,
    config: MultiFactorRankingMLConfig,
    calendar: TradingCalendar,
    sector_encoder: SectorEncoder,
    targets: Sequence[tuple[InstrumentId, date, datetime]],
    filings_by_instrument: dict[InstrumentId, Sequence[FilingFundamentals]],
    prices_by_instrument: dict[InstrumentId, Sequence[DailyPriceObservation]],
    sector_by_instrument: Mapping[InstrumentId, Sequence[SectorRecord]],
    mode: FilingTimingMode = "training",
    feature_cache_identity: str | None = None,
) -> FeaturePipelineResult:
    """Build feature observations for every ``(instrument_id, strategy_cohort_end,
    cohort_buy_timestamp)`` triple.

    Each target carries its *own* point-in-time cutoff
    (``cohort_buy_timestamp``, used directly as that target's
    ``data_cutoff``) rather than one batch-wide cutoff shared across every
    cohort -- a single shared cutoff would let an early cohort in a
    multi-cohort batch see filings only knowable as of a *later* cohort's
    own buy date, a real lookahead bug distinct from (but previously
    masked by) the fiscal/calendar-equality bug this module also fixes.
    The same per-target cutoff selects each instrument's sector via
    :func:`atlas_quant.data.point_in_time.select_point_in_time_sector` --
    ``sector_by_instrument`` holds each instrument's *full* sector
    history (multiple point-in-time facts, not one snapshot), so an
    instrument whose real classification changed over the backtest window
    sees the classification that was actually knowable as of each
    cohort's own cutoff, not today's.

    Deterministic: iterates ``targets`` in the given order and never
    depends on dict iteration order for its own output ordering.
    """
    observations: list[FeatureObservation] = []
    rejected: list[RejectedObservation] = []
    warnings: list[str] = []

    for instrument_id, strategy_cohort_end, cohort_buy_timestamp in targets:
        result = build_feature_observation(
            config=config,
            calendar=calendar,
            sector_encoder=sector_encoder,
            instrument_id=instrument_id,
            strategy_cohort_end=strategy_cohort_end,
            cohort_buy_timestamp=cohort_buy_timestamp,
            filings=filings_by_instrument.get(instrument_id, ()),
            prices=prices_by_instrument.get(instrument_id, ()),
            sector_record=select_point_in_time_sector(
                sector_by_instrument.get(instrument_id, ()), instrument_id, cutoff=cohort_buy_timestamp,
            ),
            data_cutoff=cohort_buy_timestamp,
            mode=mode,
            feature_cache_identity=feature_cache_identity,
        )
        if isinstance(result, RejectedObservation):
            rejected.append(result)
        else:
            observations.append(result)
            if result.missing_features:
                warnings.append(
                    f"{result.instrument_id.symbol} {strategy_cohort_end.isoformat()}: "
                    f"{len(result.missing_features)} missing feature(s)"
                )

    return FeaturePipelineResult(
        observations=tuple(observations),
        rejected=tuple(rejected),
        warnings=tuple(warnings),
        config_identity=config.identity(),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_cache_identity=feature_cache_identity,
    )
