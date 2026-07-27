"""Small, deterministic fixture generators for Filing Momentum ML Stage 3 tests.

Nothing here is real strategy performance evidence — these are synthetic
records sized only to exercise point-in-time and feature-calculation logic.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.filing_momentum_ml.fallback_domain import FallbackAssetStatistics
from atlas_quant.strategies.filing_momentum_ml.regime_domain import (
    ComponentAvailability,
    ComponentClassification,
    RegimeClassification,
    RegimeResult,
)
from atlas_quant.strategies.filing_momentum_ml.scoring_domain import ScoredCandidate


def instrument(symbol: str = "ACME", asset_class: AssetClass = AssetClass.EQUITY) -> InstrumentId:
    return InstrumentId(symbol=symbol, asset_class=asset_class)


def provenance(as_of: datetime, source: str = "fixture") -> DataProvenance:
    return DataProvenance(source=source, as_of=as_of, retrieved_at=as_of)


def weekday_calendar(start: date, end: date) -> ListTradingCalendar:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return ListTradingCalendar(tuple(days))


def make_filing(
    *,
    instrument_id: InstrumentId,
    quarter_end: date,
    fiscal_period: str,
    filed_days_after_quarter_end: int = 30,
    revenue: float | None = 100.0,
    gross_profit: float | None = 40.0,
    operating_income: float | None = 15.0,
    net_income: float | None = 10.0,
    diluted_eps: float | None = 1.0,
    stockholders_equity: float | None = 500.0,
    operating_cash_flow: float | None = 20.0,
    capital_expenditure: float | None = 5.0,
    accession_number: str | None = None,
    filed_at: datetime | None = None,
) -> FilingFundamentals:
    resolved_filed_at = filed_at or (
        datetime(quarter_end.year, quarter_end.month, quarter_end.day)
        + timedelta(days=filed_days_after_quarter_end)
    )
    return FilingFundamentals(
        instrument_id=instrument_id,
        fiscal_period=fiscal_period,
        fiscal_year=quarter_end.year,
        quarter_end=quarter_end,
        filed_at=resolved_filed_at,
        revenue=revenue,
        gross_profit=gross_profit,
        operating_income=operating_income,
        net_income=net_income,
        diluted_eps=diluted_eps,
        stockholders_equity=stockholders_equity,
        operating_cash_flow=operating_cash_flow,
        capital_expenditure=capital_expenditure,
        provenance=provenance(resolved_filed_at),
        accession_number=accession_number,
    )


def make_quarterly_filings(
    instrument_id: InstrumentId,
    quarter_ends: list[date],
    *,
    revenue_start: float = 100.0,
    revenue_step: float = 10.0,
) -> list[FilingFundamentals]:
    filings = []
    revenue = revenue_start
    for i, quarter_end in enumerate(quarter_ends):
        filings.append(
            make_filing(
                instrument_id=instrument_id,
                quarter_end=quarter_end,
                fiscal_period=f"Q{(quarter_end.month - 1) // 3 + 1}",
                revenue=revenue,
                gross_profit=revenue * 0.4,
                operating_income=revenue * 0.15,
                net_income=revenue * 0.10,
                diluted_eps=1.0 + i * 0.05,
            )
        )
        revenue += revenue_step
    return filings


def make_price_series(
    instrument_id: InstrumentId,
    trading_days: list[date],
    *,
    start_price: float = 50.0,
    daily_growth: float = 0.0,
) -> list[DailyPriceObservation]:
    prices = []
    price = start_price
    for day in trading_days:
        prices.append(
            DailyPriceObservation(
                instrument_id=instrument_id,
                trading_date=day,
                close=price,
                price_convention="split_dividend_adjusted",
                provenance=provenance(datetime(day.year, day.month, day.day)),
            )
        )
        price *= 1 + daily_growth
    return prices


def make_sector_record(
    instrument_id: InstrumentId, raw_sector: str | None, as_of: datetime
) -> SectorRecord:
    return SectorRecord(
        instrument_id=instrument_id,
        raw_sector=raw_sector,
        as_of=as_of,
        provenance=provenance(as_of),
    )


def make_daily_series(
    instrument_id: InstrumentId,
    trading_days: list[date],
    daily_growth_rates: list[float],
    *,
    start_price: float = 100.0,
) -> list[DailyPriceObservation]:
    """Build a deterministic close-price series from an explicit per-day growth-rate list.

    ``len(daily_growth_rates)`` must equal ``len(trading_days)``; the
    first day's own growth rate is applied to ``start_price`` to produce
    that day's close (there is no "day zero" close preceding the series).
    """
    if len(daily_growth_rates) != len(trading_days):
        raise ValueError("daily_growth_rates must have the same length as trading_days")
    prices = []
    price = start_price
    for day, rate in zip(trading_days, daily_growth_rates):
        price *= 1 + rate
        prices.append(
            DailyPriceObservation(
                instrument_id=instrument_id,
                trading_date=day,
                close=price,
                price_convention="split_dividend_adjusted",
                provenance=provenance(datetime(day.year, day.month, day.day)),
            )
        )
    return prices


class FakeHMMFitter:
    """A deterministic, injectable HMMFitter for tests -- never a real fit.

    ``state_means``/``predicted_states`` are returned exactly as given,
    letting a test control the fit outcome precisely rather than relying
    on probabilistic convergence.
    """

    def __init__(
        self,
        state_means: tuple[float, ...] = (-0.02, 0.0, 0.02),
        current_state: int = 2,
        converged: bool = True,
        error: str | None = None,
    ) -> None:
        self.state_means = state_means
        self.current_state = current_state
        self.converged = converged
        self.error = error

    def fit_predict(self, observations, **kwargs):
        from atlas_quant.strategies.filing_momentum_ml.regime_hmm import HMMFitResult

        if self.error is not None:
            return HMMFitResult(converged=False, state_means=(), predicted_states=(), error=self.error)
        n = len(observations)
        predicted = tuple([self.current_state] * n) if n else ()
        return HMMFitResult(
            converged=self.converged, state_means=self.state_means, predicted_states=predicted
        )


def make_component_classification(
    instrument_id: InstrumentId,
    *,
    component: str = "markov",
    is_bear: bool = False,
    availability: ComponentAvailability = ComponentAvailability.OK,
    evaluation_timestamp: datetime = datetime(2026, 1, 1),
    data_cutoff: datetime = datetime(2026, 1, 1),
) -> ComponentClassification:
    if availability != ComponentAvailability.OK:
        classification = RegimeClassification.UNKNOWN
        is_bear = False
    else:
        classification = RegimeClassification.BEAR if is_bear else RegimeClassification.BULL
    return ComponentClassification(
        component=component,
        instrument_id=instrument_id,
        evaluation_timestamp=evaluation_timestamp,
        data_cutoff=data_cutoff,
        classification=classification,
        is_bear=is_bear,
        availability=availability,
        confidence=None,
        observation_count=300,
        required_observation_count=100,
        windows=(),
        config_identity="a" * 64,
        provenance=(),
    )


def make_regime_result(
    instrument_id: InstrumentId,
    *,
    markov_bear: bool = False,
    hmm_bear: bool = False,
    markov_availability: ComponentAvailability = ComponentAvailability.OK,
    hmm_availability: ComponentAvailability = ComponentAvailability.OK,
    gate_mode: str = "both",
    evaluation_timestamp: datetime = datetime(2026, 1, 1),
    data_cutoff: datetime = datetime(2026, 1, 1),
) -> RegimeResult:
    markov = make_component_classification(
        instrument_id, component="markov", is_bear=markov_bear, availability=markov_availability,
        evaluation_timestamp=evaluation_timestamp, data_cutoff=data_cutoff,
    )
    hmm = make_component_classification(
        instrument_id, component="hmm", is_bear=hmm_bear, availability=hmm_availability,
        evaluation_timestamp=evaluation_timestamp, data_cutoff=data_cutoff,
    )
    if gate_mode == "both":
        is_blocked = markov.is_bear and hmm.is_bear
    elif gate_mode == "either":
        is_blocked = markov.is_bear or hmm.is_bear
    elif gate_mode == "markov":
        is_blocked = markov.is_bear
    elif gate_mode == "hmm":
        is_blocked = hmm.is_bear
    else:
        is_blocked = False
    warnings = tuple(
        f"{name} component unavailable: {component.availability.value}"
        for name, component in (("markov", markov), ("hmm", hmm))
        if component.availability != ComponentAvailability.OK
    )
    return RegimeResult(
        instrument_id=instrument_id,
        evaluation_timestamp=evaluation_timestamp,
        data_cutoff=data_cutoff,
        markov=markov,
        hmm=hmm,
        gate_mode=gate_mode,
        is_blocked=is_blocked,
        block_reason="test fixture" if is_blocked else None,
        warnings=warnings,
        config_identity="a" * 64,
        provenance=(),
    )


def make_scored_candidate(
    symbol: str,
    score: float,
    *,
    sector: str = "Tech & Media",
    strategy_id: str = "filing_momentum_ml",
    feature_schema_version: str = "1",
    model_identifier: str = "hgbc",
    model_version: str = "1",
    feature_timestamp: date = date(2026, 1, 1),
    data_cutoff: datetime = datetime(2026, 1, 1),
    asset_class: AssetClass = AssetClass.EQUITY,
) -> ScoredCandidate:
    return ScoredCandidate(
        instrument_id=instrument(symbol, asset_class),
        score=score,
        model_identifier=model_identifier,
        model_version=model_version,
        feature_observation_identity="f" * 64,
        feature_timestamp=feature_timestamp,
        data_cutoff=data_cutoff,
        sector=sector,
        strategy_id=strategy_id,
        feature_schema_version=feature_schema_version,
        provenance=provenance(datetime(2026, 1, 1)),
    )


def make_fallback_statistics(
    symbol: str,
    quarterly_returns: tuple[float, ...],
    *,
    asset_class: AssetClass = AssetClass.ETF,
    measurement_cutoff: datetime = datetime(2026, 1, 1),
) -> FallbackAssetStatistics:
    return FallbackAssetStatistics(
        instrument_id=instrument(symbol, asset_class),
        measurement_cutoff=measurement_cutoff,
        quarterly_returns=quarterly_returns,
        observation_count=len(quarterly_returns),
        provenance=provenance(measurement_cutoff),
    )
