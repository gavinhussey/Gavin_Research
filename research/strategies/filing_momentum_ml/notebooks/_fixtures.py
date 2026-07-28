"""Synthetic fixture data shared by the numbered 00-10 production-research
notebooks in this directory.

Every value here is invented for demonstration only -- small enough to
run in seconds, shaped only to exercise the production code paths these
notebooks walk through. **Nothing computed from this data is a genuine
historical result** -- see each notebook's own "Data mode" banner and
`docs/reproducibility_findings.md`.

This is a plain local helper module (Jupyter automatically adds a
notebook's own directory to ``sys.path``) -- not a legacy path hack, and
not part of the installed ``atlas_quant`` package. It imports only
production AtlasQuant code, never anything from the legacy Arnold_Quant
repository.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from atlas_quant.backtest.clock import generate_quarterly_periods
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.filing_momentum_ml.production.normalization import (
    RawFilingRecord,
    RawPriceRecord,
    RawSectorRecord,
    RawUniverseRecord,
    normalize_filings,
    normalize_prices,
    normalize_sectors,
    normalize_universe,
)

SYNTHETIC_SYMBOLS = ("AAA", "BBB")
BENCHMARK_SYMBOL = "SPY"
#: The ETF-sleeve tickers FilingMomentumMLConfig now defaults to.
FALLBACK_SYMBOLS = ("VOO", "VTI")
QUARTER_ENDS = [date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31), date(2023, 3, 31)]
TARGET_QUARTER_END = date(2023, 3, 31)


def _all_symbols() -> tuple[str, ...]:
    return tuple(dict.fromkeys(SYNTHETIC_SYMBOLS + FALLBACK_SYMBOLS))


def build_raw_filings() -> list[RawFilingRecord]:
    records = []
    for symbol_index, symbol in enumerate(SYNTHETIC_SYMBOLS):
        revenue = 100.0 + symbol_index * 50.0
        for i, quarter_end in enumerate(QUARTER_ENDS):
            filed_at = datetime.combine(quarter_end, datetime.min.time()) + timedelta(days=30)
            records.append(
                RawFilingRecord(
                    symbol=symbol, asset_class="equity", fiscal_period=f"Q{(quarter_end.month - 1) // 3 + 1}",
                    fiscal_year=quarter_end.year, quarter_end=quarter_end, filed_at=filed_at,
                    revenue=revenue, gross_profit=revenue * 0.4, operating_income=revenue * 0.15,
                    net_income=revenue * 0.1, diluted_eps=1.0 + i * 0.05, stockholders_equity=500.0 + i * 10,
                    operating_cash_flow=revenue * 0.2, capital_expenditure=revenue * 0.05,
                    accession_number=f"synthetic-{symbol}-{i}", source="synthetic_fixture",
                    retrieved_at=datetime(2023, 6, 1),
                )
            )
            revenue += 10.0
    return records


def build_raw_prices() -> list[RawPriceRecord]:
    records = []
    for symbol in _all_symbols():
        current = date(2019, 1, 1)
        price = 100.0
        while current <= date(2023, 6, 30):
            if current.weekday() < 5:
                records.append(
                    RawPriceRecord(
                        symbol=symbol, asset_class="equity", trading_date=current, close=price,
                        price_convention="split_dividend_adjusted", source="synthetic_fixture",
                        retrieved_at=datetime(2023, 6, 1),
                    )
                )
                price *= 1.0003
            current += timedelta(days=1)
    return records


def build_raw_universe() -> list[RawUniverseRecord]:
    return [
        RawUniverseRecord(
            symbol=symbol, asset_class="equity", as_of=datetime(2023, 1, 1), source="synthetic_fixture",
            survivorship_biased=True, retrieved_at=datetime(2023, 6, 1),
        )
        for symbol in SYNTHETIC_SYMBOLS
    ]


def build_raw_sectors() -> list[RawSectorRecord]:
    sectors = {"AAA": "Technology", "BBB": "Healthcare"}
    return [
        RawSectorRecord(
            symbol=symbol, asset_class="equity", raw_sector=sectors.get(symbol), as_of=datetime(2023, 1, 1),
            source="synthetic_fixture", retrieved_at=datetime(2023, 6, 1),
        )
        for symbol in SYNTHETIC_SYMBOLS
    ]


class SyntheticBundle:
    """Normalized synthetic Stage 3 records, grouped the way production code expects."""

    def __init__(self):
        filings, self.filing_issues = normalize_filings(build_raw_filings())
        prices, self.price_issues = normalize_prices(build_raw_prices())
        universe_members, self.universe_issues = normalize_universe(build_raw_universe())
        sectors, self.sector_issues = normalize_sectors(build_raw_sectors())

        self.filings_by_instrument: dict[InstrumentId, list] = {}
        for f in filings:
            self.filings_by_instrument.setdefault(f.instrument_id, []).append(f)
        self.prices_by_instrument: dict[InstrumentId, list] = {}
        for p in prices:
            self.prices_by_instrument.setdefault(p.instrument_id, []).append(p)
        self.sector_by_instrument = {s.instrument_id: s for s in sectors}
        self.universe = tuple(InstrumentId(symbol=s, asset_class=AssetClass.EQUITY) for s in SYNTHETIC_SYMBOLS)
        self.benchmark_instrument_id = InstrumentId(symbol=BENCHMARK_SYMBOL, asset_class=AssetClass.EQUITY)

    @property
    def all_issues(self):
        return self.filing_issues + self.price_issues + self.universe_issues + self.sector_issues


def build_synthetic_bundle() -> SyntheticBundle:
    return SyntheticBundle()


def build_trading_calendar(bundle: SyntheticBundle) -> ListTradingCalendar:
    all_dates = sorted({p.trading_date for prices in bundle.prices_by_instrument.values() for p in prices})
    return ListTradingCalendar(tuple(all_dates))


def build_periods(config: FilingMomentumMLConfig | None = None):
    config = config or FilingMomentumMLConfig()
    return generate_quarterly_periods(TARGET_QUARTER_END, TARGET_QUARTER_END, earnings_lag_days=config.earnings_lag_days)


def build_synthetic_manifest() -> DataProvenanceManifest:
    """A DataProvenanceManifest describing exactly what it is: synthetic fixture
    data, never a real acquisition. Every downstream reproducibility check
    must treat a manifest built this way as NOT eligible for
    `fully_reproduced` -- see docs/reproducibility_findings.md."""
    return DataProvenanceManifest(
        dataset_identity_label="synthetic-notebook-fixture-v1",
        provider_name="synthetic_fixture", provider_version=None,
        retrieval_date=date(2023, 6, 1), data_cutoff=datetime(2023, 6, 1),
        universe_identity="synthetic-two-instrument-universe",
        universe_construction_method="hand-authored for notebook demonstration only",
        survivorship_biased=True, filing_source="synthetic_fixture",
        filing_point_in_time_status="synthetic, not SEC-verified",
        price_source="synthetic_fixture", price_convention="split_dividend_adjusted",
        sector_source="synthetic_fixture", sector_override_identity="none",
        trading_calendar_source="derived from synthetic price dates",
        coverage_start=date(2019, 1, 1), coverage_end=date(2023, 6, 30),
        row_counts={}, missing_data_summary={}, duplicate_summary={},
        corporate_action_treatment="none (synthetic)", delisting_treatment="none (synthetic)",
        data_corrections=(), source_file_hashes={},
        strategy_config_identity=FilingMomentumMLConfig().identity(),
        git_commit=None,
        notes=("SYNTHETIC FIXTURE DATA -- not a genuine historical dataset.",),
    )
