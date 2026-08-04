"""Top-level real-data acquisition orchestration for Ranked Multi-Factor
Rotation.

Much simpler than Filing Momentum ML's acquisition (no SEC filings, no
universe/sector construction) -- this strategy needs exactly one thing:
daily OHLC for its fixed 11-ticker universe plus SHY (spec §1). A
per-symbol failure (network hiccup, delisted ticker) is caught and
recorded as a warning; it never aborts the whole run, and it never
silently drops a symbol without saying why -- same convention as
``filing_momentum_ml/acquisition/run_acquisition.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from atlas_quant.data.records import DailyOHLCObservation
from atlas_quant.strategies.ranked_multi_factor_rotation.acquisition.yfinance_provider import (
    OHLCHistoryProvider,
    fetch_ohlc_for_symbol,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.config import RankedMultiFactorRotationConfig

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class RmfrDataProvenanceManifest:
    """Everything needed to know exactly what data one acquisition run
    used -- shaped after (not the same class as)
    ``filing_momentum_ml.production.data_provenance.DataProvenanceManifest``;
    this strategy's inputs (a fixed ETF universe, one price provider, no
    filings/sector data) don't need that manifest's SEC/universe/sector
    fields."""

    dataset_identity_label: str
    provider_name: str
    retrieval_date: date
    universe: tuple[str, ...]
    price_convention: str
    coverage_start: date | None
    coverage_end: date | None
    row_counts: dict[str, int]
    skipped_row_counts: dict[str, int]
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """The complete, structured outcome of one real-data acquisition run."""

    observations: tuple[DailyOHLCObservation, ...]
    manifest: RmfrDataProvenanceManifest
    symbols_attempted: int
    symbols_with_data: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def run_full_acquisition(
    price_provider: OHLCHistoryProvider,
    config: RankedMultiFactorRotationConfig,
    *,
    retrieved_at: datetime,
    progress_callback: ProgressCallback | None = None,
) -> AcquisitionResult:
    """Acquire real daily OHLC for every ticker in ``config`` (the ranked
    universe plus the cash ticker)."""
    symbols = list(config.ranked_tickers) + [config.cash_ticker]

    all_observations: list[DailyOHLCObservation] = []
    warnings: list[str] = []
    row_counts: dict[str, int] = {}
    skipped_row_counts: dict[str, int] = {}
    symbols_with_data = 0

    for index, symbol in enumerate(symbols, start=1):
        if progress_callback is not None:
            progress_callback(symbol, index, len(symbols))
        try:
            result = fetch_ohlc_for_symbol(
                price_provider, symbol, source="yfinance", retrieved_at=retrieved_at,
            )
        except Exception as exc:  # network hiccup, delisted ticker, etc. -- disclosed, never fatal
            warnings.append(f"{symbol}: yfinance fetch failed: {exc}")
            row_counts[symbol] = 0
            skipped_row_counts[symbol] = 0
            continue

        row_counts[symbol] = len(result.observations)
        skipped_row_counts[symbol] = len(result.skipped_rows)
        warnings.extend(result.skipped_rows)
        if result.observations:
            symbols_with_data += 1
            all_observations.extend(result.observations)
        else:
            warnings.append(f"{symbol}: no observations parsed")

    trading_dates = [obs.trading_date for obs in all_observations]
    manifest = RmfrDataProvenanceManifest(
        dataset_identity_label=f"rmfr_yfinance_{retrieved_at.date().isoformat()}",
        provider_name="yfinance",
        retrieval_date=retrieved_at.date(),
        universe=tuple(symbols),
        price_convention="split_dividend_adjusted",
        coverage_start=min(trading_dates) if trading_dates else None,
        coverage_end=max(trading_dates) if trading_dates else None,
        row_counts=row_counts,
        skipped_row_counts=skipped_row_counts,
        warnings=tuple(warnings),
    )

    return AcquisitionResult(
        observations=tuple(all_observations),
        manifest=manifest,
        symbols_attempted=len(symbols),
        symbols_with_data=symbols_with_data,
        warnings=tuple(warnings),
    )


def write_raw_data_files(result: AcquisitionResult, raw_root: Path) -> dict[str, Path]:
    """Write ``result`` into ``raw_root`` as ``ohlc.json``/``manifest.json``.
    Overwrites unconditionally (acquisition is expected to be re-run and
    refreshed); never call with a protected production path from a test."""
    raw_root.mkdir(parents=True, exist_ok=True)
    ohlc_path = raw_root / "ohlc.json"
    ohlc_path.write_text(json.dumps([o.to_dict() for o in result.observations], indent=2))

    manifest_path = raw_root / "manifest.json"
    manifest_dict = asdict(result.manifest)
    manifest_dict["retrieval_date"] = result.manifest.retrieval_date.isoformat()
    manifest_dict["coverage_start"] = (
        result.manifest.coverage_start.isoformat() if result.manifest.coverage_start else None
    )
    manifest_dict["coverage_end"] = (
        result.manifest.coverage_end.isoformat() if result.manifest.coverage_end else None
    )
    manifest_path.write_text(json.dumps(manifest_dict, indent=2, sort_keys=True))

    return {"ohlc.json": ohlc_path, "manifest.json": manifest_path}


def load_raw_observations(raw_root: Path) -> tuple[DailyOHLCObservation, ...]:
    """Load a previously-written ``ohlc.json`` back into observations."""
    raw = json.loads((raw_root / "ohlc.json").read_text())
    return tuple(DailyOHLCObservation.from_dict(row) for row in raw)
