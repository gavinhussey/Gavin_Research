"""Deterministic sector normalization, consolidation, and encoding.

Report §3.4 defines ``sector_enc`` as a "label-encoded scoring-group
integer." The legacy prototype used ``sklearn.preprocessing.LabelEncoder``
fitted per-run on whatever sectors happened to appear in that run's
training data — meaning the same sector could encode to a different
integer across two runs that saw different tickers, and cache/config
identity could not meaningfully cover it. :class:`SectorEncoder` instead
fixes its category vocabulary at construction time from an explicit, known
list of raw GICS sectors plus the consolidation groups (report/legacy
``SCORING_GROUPS`` in ``settings.py``) — so the same sector always encodes
to the same integer, independent of run-to-run input order or which
instruments happen to appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.data.records import SectorRecord
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance

#: The eleven GICS sectors. Fixed and known in advance so the encoder's
#: vocabulary never depends on which sectors happen to appear in a given
#: run's instrument set.
RAW_GICS_SECTORS: tuple[str, ...] = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)

#: GICS sectors merged into a single scoring-group label for sector_enc,
#: read verbatim (cross-checked, not copied) from the legacy prototype's
#: settings.py: SCORING_GROUPS.
DEFAULT_SCORING_GROUPS: Mapping[str, tuple[str, ...]] = {
    "Tech & Media": ("Information Technology", "Communication Services"),
    "Consumer": ("Consumer Discretionary", "Consumer Staples"),
}

#: Per-ticker overrides applied *before* consolidation lookup, read
#: verbatim (cross-checked, not copied) from the legacy prototype's
#: settings.py: TICKER_SECTOR_OVERRIDES. These are Multi-Factor Ranking
#: ML-specific (a different strategy trading different names would not
#: necessarily want the same overrides), so they live here, not in any
#: platform-wide config.
DEFAULT_TICKER_OVERRIDES: Mapping[str, str] = {
    "T": "Utilities",
    "VZ": "Utilities",
    "TMUS": "Utilities",
}

UNKNOWN_SECTOR = "Unknown"


@dataclass(frozen=True, slots=True)
class SectorClassification:
    """The strategy-specific classification derived from a raw :class:`SectorRecord`."""

    instrument_id: InstrumentId
    raw_sector: str | None
    normalized_sector: str
    scoring_group: str
    sector_enc: int
    override_source: str | None
    as_of: datetime
    provenance: DataProvenance


@dataclass(frozen=True, slots=True)
class SectorEncoder:
    """Fixed-vocabulary sector normalizer/consolidator/encoder.

    ``consolidated_vocabulary()`` is the same regardless of which raw
    sectors or tickers are ever passed to :meth:`encode` — it is derived
    once from ``scoring_groups``/``raw_sector_vocabulary`` at construction,
    not accumulated from encoder calls.
    """

    scoring_groups: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_SCORING_GROUPS)
    )
    ticker_overrides: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TICKER_OVERRIDES)
    )
    raw_sector_vocabulary: tuple[str, ...] = RAW_GICS_SECTORS

    def __post_init__(self) -> None:
        vocabulary = self.consolidated_vocabulary()
        for ticker, target in self.ticker_overrides.items():
            if target not in vocabulary:
                raise ValueError(
                    f"ticker override {ticker!r} targets unknown sector "
                    f"{target!r}, not in {vocabulary!r}"
                )

    def consolidated_vocabulary(self) -> tuple[str, ...]:
        """The fixed, sorted set of possible normalized-sector labels, plus Unknown."""
        grouped_members = {
            member for members in self.scoring_groups.values() for member in members
        }
        ungrouped = sorted(set(self.raw_sector_vocabulary) - grouped_members)
        consolidated = set(self.scoring_groups.keys()) | set(ungrouped) | {UNKNOWN_SECTOR}
        return tuple(sorted(consolidated))

    def normalize(self, raw_sector: str | None, ticker: str | None = None) -> tuple[str, str | None]:
        """Return ``(normalized_sector, override_source)``.

        A ticker override always wins (``override_source`` set to the
        ticker). Otherwise a recognized raw sector is consolidated via
        ``scoring_groups`` if it belongs to one, else passed through
        unchanged. A missing (``None``/empty) or unrecognized raw sector
        normalizes to :data:`UNKNOWN_SECTOR` explicitly — this encoder
        never invents a new category for an unrecognized label.
        """
        if ticker and ticker in self.ticker_overrides:
            return self.ticker_overrides[ticker], ticker
        if not raw_sector:
            return UNKNOWN_SECTOR, None
        for group, members in self.scoring_groups.items():
            if raw_sector in members:
                return group, None
        if raw_sector in self.raw_sector_vocabulary:
            return raw_sector, None
        return UNKNOWN_SECTOR, None

    def encode(self, raw_sector: str | None, ticker: str | None = None) -> int:
        normalized, _ = self.normalize(raw_sector, ticker)
        vocabulary = self.consolidated_vocabulary()
        return vocabulary.index(normalized)

    def classify(self, record: SectorRecord) -> SectorClassification:
        ticker = record.instrument_id.symbol
        normalized, override_source = self.normalize(record.raw_sector, ticker)
        return SectorClassification(
            instrument_id=record.instrument_id,
            raw_sector=record.raw_sector,
            normalized_sector=normalized,
            scoring_group=normalized,
            sector_enc=self.encode(record.raw_sector, ticker),
            override_source=override_source,
            as_of=record.as_of,
            provenance=record.provenance,
        )

    def identity(self) -> str:
        """Deterministic identity of this encoder's mapping, for config/cache identity."""
        return compute_config_identity(
            {
                "scoring_groups": {
                    group: list(members)
                    for group, members in sorted(self.scoring_groups.items())
                },
                "ticker_overrides": dict(sorted(self.ticker_overrides.items())),
                "raw_sector_vocabulary": list(self.raw_sector_vocabulary),
            }
        )
