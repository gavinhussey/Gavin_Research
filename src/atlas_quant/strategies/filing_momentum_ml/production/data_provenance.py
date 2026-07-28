"""The typed data-provenance manifest for a real Filing Momentum ML production run.

Distinct from, and never collapsed with, the other two provenance
concepts this platform already tracks:

- **Strategy specification provenance** — ``report_current.html`` itself,
  controls trading logic (unchanged by this module).
- **Implementation provenance** — this repository's own Git history,
  tracks code changes (unchanged by this module).
- **Dataset and result provenance** (this module) — controls whether
  historical data, caches, models, backtests, and reports may be trusted
  or compared at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping

from atlas_quant.config.identity import compute_config_identity


@dataclass(frozen=True, slots=True)
class DataProvenanceManifest:
    """Everything needed to know exactly what data a production run used."""

    dataset_identity_label: str
    provider_name: str
    provider_version: str | None
    retrieval_date: date
    data_cutoff: datetime
    universe_identity: str
    universe_construction_method: str
    survivorship_biased: bool
    filing_source: str
    filing_point_in_time_status: str
    price_source: str
    price_convention: str
    sector_source: str
    sector_override_identity: str
    trading_calendar_source: str
    coverage_start: date
    coverage_end: date
    row_counts: Mapping[str, int]
    missing_data_summary: Mapping[str, int]
    duplicate_summary: Mapping[str, int]
    corporate_action_treatment: str
    delisting_treatment: str
    data_corrections: tuple[str, ...]
    source_file_hashes: Mapping[str, str]
    strategy_config_identity: str
    regime_config_identity: str
    git_commit: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.dataset_identity_label:
            raise ValueError("dataset_identity_label must be non-empty")
        if self.coverage_end < self.coverage_start:
            raise ValueError("coverage_end cannot be before coverage_start")

    def identity(self) -> str:
        """Deterministic identity — never includes credentials or secrets
        (only hashes/counts/labels/dates are hashed, never raw values that
        could carry a credential)."""
        return compute_config_identity(
            {
                "dataset_identity_label": self.dataset_identity_label,
                "provider_name": self.provider_name, "provider_version": self.provider_version,
                "retrieval_date": self.retrieval_date.isoformat(), "data_cutoff": self.data_cutoff,
                "universe_identity": self.universe_identity,
                "universe_construction_method": self.universe_construction_method,
                "survivorship_biased": self.survivorship_biased, "filing_source": self.filing_source,
                "filing_point_in_time_status": self.filing_point_in_time_status,
                "price_source": self.price_source, "price_convention": self.price_convention,
                "sector_source": self.sector_source, "sector_override_identity": self.sector_override_identity,
                "trading_calendar_source": self.trading_calendar_source,
                "coverage_start": self.coverage_start.isoformat(), "coverage_end": self.coverage_end.isoformat(),
                "row_counts": dict(sorted(self.row_counts.items())),
                "missing_data_summary": dict(sorted(self.missing_data_summary.items())),
                "duplicate_summary": dict(sorted(self.duplicate_summary.items())),
                "corporate_action_treatment": self.corporate_action_treatment,
                "delisting_treatment": self.delisting_treatment,
                "data_corrections": list(self.data_corrections),
                "source_file_hashes": dict(sorted(self.source_file_hashes.items())),
                "strategy_config_identity": self.strategy_config_identity,
                "regime_config_identity": self.regime_config_identity,
                "git_commit": self.git_commit,
            }
        )
