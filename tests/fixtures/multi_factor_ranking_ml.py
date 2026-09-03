"""Small, deterministic fixture generators for Multi-Factor Ranking ML Stage 3 tests.

Nothing here is real strategy performance evidence — these are synthetic
records sized only to exercise point-in-time and feature-calculation logic.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.multi_factor_ranking_ml.fallback_domain import FallbackAssetStatistics
from atlas_quant.strategies.multi_factor_ranking_ml.scoring_domain import ScoredCandidate


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


def make_scored_candidate(
    symbol: str,
    score: float,
    *,
    sector: str = "Tech & Media",
    strategy_id: str = "multi_factor_ranking_ml",
    feature_schema_version: str = "2",
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


class FakeEstimator:
    """A deterministic, injectable Estimator for tests -- never a real fit.

    ``fixed_scores`` (if given) maps row index -> P(y=1) directly, letting
    a test control scoring output precisely. Otherwise ``predict_proba``
    returns a constant distribution for every row.
    """

    def __init__(
        self,
        classes: tuple[int, ...] = (0, 1),
        fixed_scores: dict | None = None,
        default_score: float = 0.5,
        fit_error: str | None = None,
    ) -> None:
        self.classes_ = classes
        self.fixed_scores = fixed_scores or {}
        self.default_score = default_score
        self.fit_error = fit_error
        self.fit_called_with = None

    def fit(self, X, y):
        if self.fit_error is not None:
            raise ValueError(self.fit_error)
        self.fit_called_with = (X, y)
        return self

    def predict_proba(self, X):
        rows = []
        pos_index = self.classes_.index(1) if 1 in self.classes_ else None
        for i in range(len(X)):
            score = self.fixed_scores.get(i, self.default_score)
            row = [0.0] * len(self.classes_)
            if pos_index is not None:
                row[pos_index] = score
                other = (1.0 - score) / max(1, len(self.classes_) - 1)
                for j in range(len(row)):
                    if j != pos_index:
                        row[j] = other
            rows.append(row)
        return rows


def make_backtest_universe(n: int = 15) -> list[InstrumentId]:
    return [instrument(f"T{i:02d}") for i in range(n)]


def make_backtest_price_source(
    universe: list, benchmark_ids: list, start: date, end: date
) -> dict:
    """A deterministic daily price history for every instrument in ``universe``
    plus ``benchmark_ids``, spanning ``start``..``end`` (weekdays only)."""
    import datetime as _dt

    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += _dt.timedelta(days=1)

    source = {}
    for i, iid in enumerate(universe + benchmark_ids):
        rate = 0.0002 * ((i % 7) - 3)
        source[iid] = tuple(make_price_series(iid, days, start_price=100.0, daily_growth=rate))
    return source


def make_backtest_feature_observation_source(
    universe: list, sector: str = "Tech & Media", *, point_in_time_cutoff: bool = False
):
    """Returns a callable(quarter_end) -> list[FeatureObservation] for a fixed universe.

    By default every observation carries a far-future ``data_cutoff``,
    which the Stage 5 evaluator rejects as ``FUTURE_DATA_CUTOFF`` -- so
    the default source exercises the "no candidate survives validation"
    path. Pass ``point_in_time_cutoff=True`` for a cutoff equal to the
    quarter end, producing candidates that actually survive into ranking
    and weighting.
    """
    from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import (
        FEATURE_NAMES,
        FeatureObservation,
        missing_feature_names,
    )

    def source(quarter_end: date) -> list:
        result = []
        for i, iid in enumerate(universe):
            features = {name: float((i + quarter_end.toordinal()) % 10) for name in FEATURE_NAMES}
            result.append(
                FeatureObservation(
                    strategy_id="multi_factor_ranking_ml", strategy_version="0.1.0", feature_schema_version="2",
                    instrument_id=iid, fiscal_period="Q", quarter_end=quarter_end,
                    filing_timestamp=datetime(quarter_end.year, quarter_end.month, quarter_end.day),
                    feature_timestamp=quarter_end,
                    data_cutoff=(
                        datetime(quarter_end.year, quarter_end.month, quarter_end.day)
                        if point_in_time_cutoff
                        else datetime(2035, 1, 1)
                    ),
                    sector=sector, features=features, missing_features=missing_feature_names(features),
                    provenance=(provenance(datetime(quarter_end.year, quarter_end.month, quarter_end.day)),),
                    config_identity="a" * 64, feature_cache_identity=None,
                    strategy_cohort_end=quarter_end,
                    cohort_buy_timestamp=datetime(quarter_end.year, quarter_end.month, quarter_end.day),
                )
            )
        return result

    return source


#: The ETF-sleeve tickers MultiFactorRankingMLConfig defaults to. Kept here so
#: fixtures and the config can never silently drift apart.
FALLBACK_TICKERS = ("VOO", "VTI")


def make_backtest_fallback_statistics_source(voo_return: float = 0.02, vti_return: float = 0.03):
    """A fallback_statistics_source covering the default VOO/VTI sleeve."""
    from atlas_quant.strategies.multi_factor_ranking_ml.fallback_domain import FallbackAssetStatistics

    voo = instrument("VOO", AssetClass.ETF)
    vti = instrument("VTI", AssetClass.ETF)

    def source(period) -> tuple:
        return (
            FallbackAssetStatistics(
                instrument_id=voo, measurement_cutoff=period.evaluation_timestamp,
                quarterly_returns=(voo_return,) * 12, observation_count=12,
                provenance=provenance(period.evaluation_timestamp),
            ),
            FallbackAssetStatistics(
                instrument_id=vti, measurement_cutoff=period.evaluation_timestamp,
                quarterly_returns=(vti_return,) * 12, observation_count=12,
                provenance=provenance(period.evaluation_timestamp),
            ),
        )

    return source
