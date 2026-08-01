"""One-time backfill: real point-in-time SIC history for already-acquired filings.

Separate from ``run_acquisition.py``'s regular per-symbol acquisition
loop (~500 requests) because this is a full order of magnitude larger
(~1 request per already-acquired ``(symbol, accession_number)`` pair,
tens of thousands total) and needs its own explicit rate limit to stay a
good citizen of SEC's fair-access policy. A per-filing failure (network
hiccup, missing/unparseable SIC line) is caught and recorded as a
warning; it never aborts the whole run, and the affected filing's
``sic_code``/``gics_sector`` are left ``None``, never fabricated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

from atlas_quant.strategies.filing_momentum_ml.acquisition.http_client import HttpClient
from atlas_quant.strategies.filing_momentum_ml.acquisition.sec_edgar import fetch_filing_sic
from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawFilingRecord, RawSicHistoryRecord
from atlas_quant.strategies.filing_momentum_ml.sic_gics_crosswalk import sic_to_gics_sector

#: Conservative relative to SEC's own stated fair-access ceiling (10
#: req/sec) -- this run is unattended and multi-hour, so headroom matters
#: more than raw throughput.
DEFAULT_REQUESTS_PER_SECOND = 4.0

SicHistoryProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class SicHistoryResult:
    """The complete, structured outcome of one SIC-history backfill run."""

    records: tuple[RawSicHistoryRecord, ...]
    pairs_attempted: int
    pairs_with_sic: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _unique_filing_pairs(filings: Sequence[RawFilingRecord]) -> list[RawFilingRecord]:
    """One row per ``(symbol, accession_number)`` -- a filing reports many
    concepts across many ``RawFilingRecord`` rows sharing one accession;
    its SIC only needs fetching once per accession, not once per row."""
    seen: dict[tuple[str, str], RawFilingRecord] = {}
    for filing in filings:
        if filing.accession_number is None:
            continue
        key = (filing.symbol, filing.accession_number)
        seen.setdefault(key, filing)
    return list(seen.values())


def fetch_sic_history(
    client: HttpClient,
    filings: Sequence[RawFilingRecord],
    ticker_to_cik: Mapping[str, str],
    *,
    user_agent: str,
    retrieved_at: datetime,
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
    progress_callback: SicHistoryProgressCallback | None = None,
) -> SicHistoryResult:
    """Fetch real point-in-time SIC for every unique already-acquired
    filing accession, and map each through the SIC→GICS crosswalk.

    ``filings`` is expected to be exactly what ``filings.json`` already
    holds (i.e. ``fetch_filings_for_symbol``'s prior output) -- this
    function acquires no new filings itself, only new per-filing SIC
    facts for filings already on disk.
    """
    pairs = _unique_filing_pairs(filings)
    delay_seconds = 1.0 / requests_per_second if requests_per_second > 0 else 0.0

    records: list[RawSicHistoryRecord] = []
    warnings: list[str] = []
    pairs_with_sic = 0

    for index, filing in enumerate(pairs, start=1):
        if progress_callback is not None:
            progress_callback(filing.symbol, index, len(pairs))

        cik10 = ticker_to_cik.get(filing.symbol.upper())
        if cik10 is None:
            warnings.append(f"{filing.symbol}/{filing.accession_number}: no CIK match")
            if index < len(pairs):
                time.sleep(delay_seconds)
            continue

        try:
            sic_code = fetch_filing_sic(client, cik10, filing.accession_number, user_agent=user_agent)
        except Exception as exc:  # noqa: BLE001 - one filing's failure must not abort the run
            warnings.append(f"{filing.symbol}/{filing.accession_number}: SIC fetch failed: {exc}")
            sic_code = None
        else:
            if sic_code is None:
                warnings.append(f"{filing.symbol}/{filing.accession_number}: no SIC found in filing header")

        gics_sector = sic_to_gics_sector(sic_code) if sic_code is not None else None
        if sic_code is not None:
            pairs_with_sic += 1

        records.append(
            RawSicHistoryRecord(
                symbol=filing.symbol, asset_class=filing.asset_class,
                accession_number=filing.accession_number, filed_at=filing.filed_at,
                sic_code=sic_code, gics_sector=gics_sector,
                source="sec_edgar_sic_header", retrieved_at=retrieved_at,
            )
        )

        if index < len(pairs):
            time.sleep(delay_seconds)

    return SicHistoryResult(
        records=tuple(records), pairs_attempted=len(pairs),
        pairs_with_sic=pairs_with_sic, warnings=tuple(warnings),
    )


def write_sic_history_file(result: SicHistoryResult, raw_root: Path) -> Path:
    """Write ``result`` into ``raw_root / 'sic_history.json'`` -- same
    JSON-array-of-``to_dict()`` shape as every other raw acquisition
    output in this package."""
    import json

    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    path = raw_root / "sic_history.json"
    path.write_text(json.dumps([r.to_dict() for r in result.records], indent=2))
    return path
