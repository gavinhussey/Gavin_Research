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
