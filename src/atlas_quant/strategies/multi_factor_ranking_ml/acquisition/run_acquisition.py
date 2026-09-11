"""Top-level real-data acquisition orchestration for Multi-Factor Ranking ML.

Ties together the universe and yfinance price adapters into one
acquisition run, and writes the result into the exact JSON shape (and
directory layout) the CLI's ``--raw-root`` already reads -- acquisition
produces input for the existing pipeline, never a second one. A
per-symbol failure (network hiccup, delisted ticker) is caught and
recorded as a warning; it never aborts the whole run, and it never
silently drops a symbol without saying why.

Fundamentals are *not* acquired here (unlike filing_momentum_ml, which
fetched them from SEC EDGAR) -- this strategy's fundamentals/feature
inputs come from locally supplied Bloomberg CSV exports instead, via
``acquisition/csv_import.py``, run separately from this live-provider
acquisition path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.http_client import HttpClient
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.universe import build_universe_records
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.yfinance_provider import PriceHistoryProvider, fetch_prices_for_symbol
from atlas_quant.strategies.multi_factor_ranking_ml.production.data_provenance import DataProvenanceManifest
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import (
    RawPriceRecord,
    RawUniverseRecord,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """The complete, structured outcome of one real-data acquisition run."""

    prices: tuple[RawPriceRecord, ...]
    universe: tuple[RawUniverseRecord, ...]
    symbols_attempted: int
    symbols_with_prices: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def run_full_acquisition(
    http_client: HttpClient,
    price_provider: PriceHistoryProvider,
    *,
    retrieved_at: datetime,
    symbol_limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AcquisitionResult:
    """Acquire the present-day S&P 500 + Nasdaq 100 universe and real daily
    prices. Fundamentals are *not* acquired here -- see
    ``acquisition/csv_import.py`` for the Bloomberg-CSV fundamentals
    source. ``symbol_limit`` (if given) caps how many universe members are
    actually fetched -- for a quick partial run, never for silently
    dropping symbols from the reported universe without disclosure (the
    returned ``universe`` is filtered to match exactly what was
    attempted).
    """
    universe_records = build_universe_records(http_client, as_of=retrieved_at, retrieved_at=retrieved_at)
    symbols = [r.symbol for r in universe_records]
    if symbol_limit is not None:
        symbols = symbols[:symbol_limit]
    symbol_set = set(symbols)

    all_prices: list[RawPriceRecord] = []
    warnings: list[str] = []
    symbols_with_prices = 0

    for index, symbol in enumerate(symbols, start=1):
        if progress_callback is not None:
            progress_callback(symbol, index, len(symbols))

        try:
            prices = fetch_prices_for_symbol(price_provider, symbol, source="yfinance", retrieved_at=retrieved_at)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{symbol}: yfinance fetch failed: {exc}")
        else:
            if prices:
                symbols_with_prices += 1
            else:
                warnings.append(f"{symbol}: no price history acquired")
            all_prices.extend(prices)

    return AcquisitionResult(
        prices=tuple(all_prices),
        universe=tuple(r for r in universe_records if r.symbol in symbol_set),
        symbols_attempted=len(symbols),
        symbols_with_prices=symbols_with_prices, warnings=tuple(warnings),
    )


def write_raw_data_files(result: AcquisitionResult, raw_root: Path) -> dict[str, Path]:
    """Write ``result`` into ``raw_root`` as ``prices.json``/``universe.json``
    -- the exact shape the CLI's ``--raw-root`` already reads. Overwrites
    unconditionally (acquisition is expected to be re-run and refreshed);
    never call with a protected production path from a test.
    Fundamentals (``filings.json``-equivalent) are written separately by
    ``acquisition/csv_import.py``, not here."""
    import json

    raw_root.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, records in (
        ("prices.json", result.prices),
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
        provider_name="yfinance+wikipedia", provider_version=None,
        retrieval_date=retrieval_date, data_cutoff=data_cutoff,
        universe_identity="sp500_nasdaq100_wikipedia_present_day_snapshot",
        universe_construction_method="present_day_snapshot_applied_retroactively",
        survivorship_biased=True, filing_source="not_applicable_bloomberg_csv_pending",
        filing_point_in_time_status="fundamentals sourced from locally supplied Bloomberg CSV, not acquired here",
        price_source="yfinance", price_convention="split_dividend_adjusted",
        sector_source="not_yet_wired_pending_bloomberg_csv",
        sector_override_identity="none",
        trading_calendar_source="derived from acquired price trading dates",
        coverage_start=coverage_start, coverage_end=coverage_end,
        row_counts={
            "prices": len(result.prices), "universe": len(result.universe),
        },
        missing_data_summary={
            "symbols_without_prices": result.symbols_attempted - result.symbols_with_prices,
        },
        duplicate_summary={},
        corporate_action_treatment="split_dividend_adjusted_close (yfinance auto_adjust=True)",
        delisting_treatment="not handled -- present-day universe only, no delisted names included",
        data_corrections=(), source_file_hashes={},
        strategy_config_identity=strategy_config_identity,
        git_commit=git_commit,
        notes=tuple(result.warnings[:50]),  # first 50 warnings inline; full list belongs in run logs
    )
