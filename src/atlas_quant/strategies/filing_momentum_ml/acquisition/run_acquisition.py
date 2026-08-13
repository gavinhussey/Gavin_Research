"""Top-level real-data acquisition orchestration for Filing Momentum ML.

Ties together the universe/sector, SEC EDGAR filing, and yfinance price
adapters into one acquisition run, and writes the result into the exact
JSON shape (and directory layout) the CLI's ``--raw-root`` already reads
-- acquisition produces input for the existing pipeline, never a second
one. A per-symbol failure (network hiccup, no CIK match, delisted
ticker) is caught and recorded as a warning; it never aborts the whole
run, and it never silently drops a symbol without saying why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from atlas_quant.strategies.filing_momentum_ml.acquisition.http_client import HttpClient
from atlas_quant.strategies.filing_momentum_ml.acquisition.sec_edgar import fetch_filings_for_symbol, fetch_ticker_to_cik_map
from atlas_quant.strategies.filing_momentum_ml.acquisition.universe import build_universe_records
from atlas_quant.strategies.filing_momentum_ml.acquisition.yfinance_provider import PriceHistoryProvider, fetch_price_history_for_symbol
from atlas_quant.strategies.filing_momentum_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.filing_momentum_ml.production.normalization import (
    RawFilingRecord,
    RawCorporateActionRecord,
    RawPriceRecord,
    RawUniverseRecord,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """The complete, structured outcome of one real-data acquisition run."""

    filings: tuple[RawFilingRecord, ...]
    prices: tuple[RawPriceRecord, ...]
    universe: tuple[RawUniverseRecord, ...]
    symbols_attempted: int
    symbols_with_filings: int
    symbols_with_prices: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    corporate_actions: tuple[RawCorporateActionRecord, ...] = field(default_factory=tuple)


def run_full_acquisition(
    http_client: HttpClient,
    price_provider: PriceHistoryProvider,
    *,
    sec_user_agent: str,
    retrieved_at: datetime,
    symbol_limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AcquisitionResult:
    """Acquire the present-day S&P 500 + Nasdaq 100 universe, real SEC
    EDGAR filing fundamentals, and real daily prices. Sector is *not*
    acquired here -- see ``acquire-sic-history``/``acquisition/sic_history.py``
    for the point-in-time SIC-derived sector source. ``symbol_limit`` (if
    given) caps how many universe members are actually fetched -- for a
    quick partial run, never for silently dropping symbols from the
    reported universe without disclosure (the returned ``universe`` is
    filtered to match exactly what was attempted).
    """
    universe_records = build_universe_records(http_client, as_of=retrieved_at, retrieved_at=retrieved_at)
    symbols = [r.symbol for r in universe_records]
    if symbol_limit is not None:
        symbols = symbols[:symbol_limit]
    symbol_set = set(symbols)

    ticker_to_cik = fetch_ticker_to_cik_map(http_client, user_agent=sec_user_agent)

    all_filings: list[RawFilingRecord] = []
    all_prices: list[RawPriceRecord] = []
    all_corporate_actions: list[RawCorporateActionRecord] = []
    warnings: list[str] = []
    symbols_with_filings = 0
    symbols_with_prices = 0

    for index, symbol in enumerate(symbols, start=1):
        if progress_callback is not None:
            progress_callback(symbol, index, len(symbols))

        try:
            filings = fetch_filings_for_symbol(
                http_client, symbol, ticker_to_cik, user_agent=sec_user_agent, retrieved_at=retrieved_at,
            )
        except Exception as exc:  # noqa: BLE001 - one symbol's failure must not abort the run
            warnings.append(f"{symbol}: SEC EDGAR fetch failed: {exc}")
        else:
            if filings:
                symbols_with_filings += 1
            else:
                warnings.append(f"{symbol}: no SEC filings acquired (no CIK match or no quarterly facts found)")
            all_filings.extend(filings)

        try:
            price_history = fetch_price_history_for_symbol(price_provider, symbol, source="yfinance", retrieved_at=retrieved_at)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{symbol}: yfinance fetch failed: {exc}")
        else:
            prices = price_history.prices
            if prices:
                symbols_with_prices += 1
            else:
                warnings.append(f"{symbol}: no price history acquired")
            all_prices.extend(prices)
            all_corporate_actions.extend(price_history.corporate_actions)

    return AcquisitionResult(
        filings=tuple(all_filings), prices=tuple(all_prices), corporate_actions=tuple(all_corporate_actions),
        universe=tuple(r for r in universe_records if r.symbol in symbol_set),
        symbols_attempted=len(symbols), symbols_with_filings=symbols_with_filings,
        symbols_with_prices=symbols_with_prices, warnings=tuple(warnings),
    )


def write_raw_data_files(result: AcquisitionResult, raw_root: Path) -> dict[str, Path]:
    """Write ``result`` into ``raw_root`` as ``filings.json``/``prices.json``/
    ``universe.json`` -- the exact shape the CLI's ``--raw-root`` already
    reads. Overwrites unconditionally (acquisition is expected to be
    re-run and refreshed); never call with a protected production path
    from a test. ``sic_history.json`` (the sector source) is written
    separately by ``acquire-sic-history``, not here -- it depends on
    ``filings.json`` already existing."""
    import json

    raw_root.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, records in (
        ("filings.json", result.filings), ("prices.json", result.prices),
        ("corporate_actions.json", result.corporate_actions),
        ("universe.json", result.universe),
    ):
        path = raw_root / name
        path.write_text(json.dumps([r.to_dict() for r in records], indent=2))
        written[name] = path
    return written


def build_acquisition_manifest(
    result: AcquisitionResult,
    *,
    dataset_identity_label: str,
    strategy_config_identity: str,
    retrieval_date: date,
    data_cutoff: datetime,
    git_commit: str | None,
) -> DataProvenanceManifest:
    """Build the real :class:`DataProvenanceManifest` describing exactly
    what this acquisition run produced."""
    coverage_start = min((p.trading_date for p in result.prices), default=retrieval_date)
    coverage_end = max((p.trading_date for p in result.prices), default=retrieval_date)
    return DataProvenanceManifest(
        dataset_identity_label=dataset_identity_label,
        provider_name="sec_edgar+yfinance+wikipedia", provider_version=None,
        retrieval_date=retrieval_date, data_cutoff=data_cutoff,
        universe_identity="sp500_nasdaq100_wikipedia_present_day_snapshot",
        universe_construction_method="present_day_snapshot_applied_retroactively",
        survivorship_biased=True, filing_source="sec_edgar",
        filing_point_in_time_status="filed_at taken directly from SEC's own 'filed' field",
        price_source="yfinance", price_convention="split_adjusted_dividend_unadjusted",
        sector_source="sec_edgar_sic_header_crosswalk (acquire-sic-history, run separately)",
        sector_override_identity="none",
        trading_calendar_source="derived from acquired price trading dates",
        coverage_start=coverage_start, coverage_end=coverage_end,
        row_counts={
            "filings": len(result.filings), "prices": len(result.prices),
            "corporate_actions": len(result.corporate_actions),
            "universe": len(result.universe),
        },
        missing_data_summary={
            "symbols_without_filings": result.symbols_attempted - result.symbols_with_filings,
            "symbols_without_prices": result.symbols_attempted - result.symbols_with_prices,
        },
        duplicate_summary={},
        corporate_action_treatment=(
            "split_adjusted_dividend_unadjusted_ohlc_plus_yfinance_effective_date_splits_dividends; "
            "no announcement-vintage timestamps"
        ),
        delisting_treatment="not handled -- present-day universe only, no delisted names included",
        data_corrections=(), source_file_hashes={},
        strategy_config_identity=strategy_config_identity,
        git_commit=git_commit,
        notes=tuple(result.warnings[:50]),  # first 50 warnings inline; full list belongs in run logs
    )
