"""Filing Momentum ML's point-in-time feature pipeline.

Turns provider-neutral records (:mod:`atlas_quant.data.records`) into
:class:`~atlas_quant.strategies.filing_momentum_ml.feature_domain
.FeatureObservation` rows, using only the existing pure formulas in
:mod:`atlas_quant.strategies.filing_momentum_ml.formulas` — this module's
own job is exclusively: point-in-time selection, timing resolution,
missing-data handling, and assembling the result. It does not implement
any new mathematical formula itself.

Explicitly out of scope here (Stage 4+): qualification, model
training/prediction, position weighting (including
``score_proportional_weights``, which belongs to the later strategy stage
and is deliberately never called from this module), portfolio
construction, and backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
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
from atlas_quant.strategies.filing_momentum_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_VERSION,
    FilingMomentumMLConfig,
)
from atlas_quant.strategies.filing_momentum_ml.feature_domain import (
    FeatureObservation,
    missing_feature_names,
)
from atlas_quant.strategies.filing_momentum_ml.formulas import (
    annualized_vol,
    fiscal_quarter_number,
    margin_trend,
    ols_trend,
    price_momentum,
    qoq_acceleration,
    qoq_change,
    vol_ratio,
)
from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder

_NAN = float("nan")

#: report §3.2: 63/126/252 trading days ~= 3/6/12 months.
_PRICE_MOMENTUM_WINDOWS: tuple[tuple[str, int], ...] = (
    ("price_mom_3m", 63),
    ("price_mom_6m", 126),
    ("price_mom_12m", 252),
)
#: report §3.3: vol_20d/vol_63d need 20/63 daily returns respectively.
_VOL_20D_RETURNS = 20
_VOL_63D_RETURNS = 63
_TREND_WINDOW = 6
_DAILY_BAR_AVAILABLE_TIME = time(16, 0)


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
        in :data:`~atlas_quant.strategies.filing_momentum_ml.feature_domain
        .FEATURE_NAMES` column order."""
        return [obs.to_model_row() for obs in self.observations]


def _clean(values: Sequence[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def _clean_pairs(
    numerators: Sequence[float | None], denominators: Sequence[float | None]
) -> tuple[list[float], list[float]]:
    pairs = [
        (n, d) for n, d in zip(numerators, denominators) if n is not None and d is not None
    ]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _last_n(values: Sequence[float | None], n: int) -> list[float | None] | None:
    if len(values) < n:
        return None
    return list(values[-n:])


def _qoq_from_series(values: Sequence[float | None]) -> float:
    last2 = _last_n(values, 2)
    if last2 is None or any(v is None for v in last2):
        return _NAN
    previous, current = last2
    return qoq_change(current=current, previous=previous)


def _accel_from_series(values: Sequence[float | None]) -> float:
    last3 = _last_n(values, 3)
    if last3 is None or any(v is None for v in last3):
        return _NAN
    r2, r1, r0 = last3
    return qoq_acceleration(r0=r0, r1=r1, r2=r2)


def compute_fundamental_features(
    filings: Sequence[FilingFundamentals],
) -> dict[str, float]:
    """Compute the 9 fundamental-momentum features (report §3.1) from an
    ordered (oldest-to-newest, most-recent-last), point-in-time-selected
    filing history. Does not perform point-in-time selection itself —
    call :func:`atlas_quant.data.point_in_time.select_point_in_time_fundamentals`
    first.
    """
    revenue = [f.revenue for f in filings]
    gross_profit = [f.gross_profit for f in filings]
    operating_income = [f.operating_income for f in filings]
    net_income = [f.net_income for f in filings]
    eps = [f.diluted_eps for f in filings]
    equity = [f.stockholders_equity for f in filings]
    fcf = [
        (f.operating_cash_flow - f.capital_expenditure)
        if f.operating_cash_flow is not None and f.capital_expenditure is not None
        else None
        for f in filings
    ]

    gm_num, gm_den = _clean_pairs(gross_profit, revenue)
    om_num, om_den = _clean_pairs(operating_income, revenue)
    nm_num, nm_den = _clean_pairs(net_income, revenue)
    fcf_num, fcf_den = _clean_pairs(fcf, revenue)
    roe_num, roe_den = _clean_pairs(net_income, equity)

    return {
        "rev_qoq": _qoq_from_series(revenue),
        "rev_accel": _accel_from_series(revenue),
        "rev_trend": ols_trend(_clean(revenue), window=_TREND_WINDOW),
        "gm_trend": margin_trend(gm_num, gm_den, window=_TREND_WINDOW),
        "om_trend": margin_trend(om_num, om_den, window=_TREND_WINDOW),
        "nm_trend": margin_trend(nm_num, nm_den, window=_TREND_WINDOW),
        "eps_qoq": _qoq_from_series(eps),
        "fcf_trend": margin_trend(fcf_num, fcf_den, window=_TREND_WINDOW),
        "roe_trend": margin_trend(roe_num, roe_den, window=_TREND_WINDOW),
    }


def compute_raw_fcf_trend(filings: Sequence[FilingFundamentals]) -> float:
    """``fcf_trend`` under ``fcf_mode="raw"``: trend of raw FCF, not FCF/revenue."""
    fcf = [
        (f.operating_cash_flow - f.capital_expenditure)
        if f.operating_cash_flow is not None and f.capital_expenditure is not None
        else None
        for f in filings
    ]
    return ols_trend(_clean(fcf), window=_TREND_WINDOW)


def latest_completed_price_bar_date(
    prices: Sequence[DailyPriceObservation],
    feature_timestamp: date,
    as_of_timestamp: datetime,
) -> date | None:
    """Latest daily price bar usable for a decision at ``as_of_timestamp``.

    A daily close is modeled as available at 16:00 on its trading date.
    Eligibility is strict: ``available_at < as_of_timestamp``. The
    available price series itself determines the previous completed
    session, so weekends and holidays do not require calendar-day
    subtraction.
    """
    eligible = [
        p
        for p in prices
        if p.trading_date <= feature_timestamp
        and datetime.combine(p.trading_date, _DAILY_BAR_AVAILABLE_TIME) < as_of_timestamp
    ]
    if not eligible:
        return None
    return max(p.trading_date for p in eligible)


def compute_price_features(
    prices: Sequence[DailyPriceObservation],
    feature_timestamp: date,
    *,
    as_of_timestamp: datetime | None = None,
) -> dict[str, float]:
    """Compute the 6 price/volatility features (report §3.2-§3.3).

    Only completed daily bars whose availability timestamp is strictly
    before ``as_of_timestamp`` are eligible. Daily closes are modeled as
    becoming available at 16:00 on their own trading date; under the
    historical backtest clock's midnight entry timestamp, the buy-date
    close is therefore not eligible. If ``as_of_timestamp`` is omitted,
    the legacy ``trading_date <= feature_timestamp`` behavior is retained
    for direct callers that are not making a point-in-time decision.
    """
    cutoff_date = (
        latest_completed_price_bar_date(prices, feature_timestamp, as_of_timestamp)
        if as_of_timestamp is not None
        else feature_timestamp
    )
    eligible = sorted(
        (p for p in prices if cutoff_date is not None and p.trading_date <= cutoff_date),
        key=lambda p: p.trading_date,
    )
    closes = [p.close for p in eligible]

    features: dict[str, float] = {}
    for name, lookback in _PRICE_MOMENTUM_WINDOWS:
        if len(closes) > lookback:
            features[name] = price_momentum(closes[-1], closes[-1 - lookback])
        else:
            features[name] = _NAN

    returns = [
        closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] != 0
    ]

    vol_20d = annualized_vol(returns[-_VOL_20D_RETURNS:]) if len(returns) >= _VOL_20D_RETURNS else _NAN
    vol_63d = annualized_vol(returns[-_VOL_63D_RETURNS:]) if len(returns) >= _VOL_63D_RETURNS else _NAN

    features["vol_20d"] = vol_20d
    features["vol_63d"] = vol_63d
    features["vol_ratio"] = vol_ratio(vol_20d, vol_63d)
    return features


def build_feature_observation(
    *,
    config: FilingMomentumMLConfig,
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

    fundamentals = compute_fundamental_features(selection.selected)
    if config.fcf_mode == "raw":
        fundamentals["fcf_trend"] = compute_raw_fcf_trend(selection.selected)

    price_features = compute_price_features(
        prices,
        feature_date,
        as_of_timestamp=data_cutoff,
    )

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

    features: dict[str, float] = {
        **fundamentals,
        **price_features,
        "quarter_num": float(fiscal_quarter_number(target_filing.quarter_end)),
        "sector_enc": float(classification.sector_enc),
    }

    provenance = [target_filing.provenance] + [p.provenance for p in prices[:1]]
    if sector_record is not None:
        provenance.append(sector_record.provenance)

    missing = missing_feature_names(features)
    if missing:
        audit_trail = audit_trail.append(
            AuditRecord(
                stage="missing_data",
                message=f"{len(missing)} of 17 features missing",
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
    config: FilingMomentumMLConfig,
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
