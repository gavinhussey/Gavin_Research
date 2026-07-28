"""Raw-data validation with explicit severity — never a warning string alone.

Operates only on already-typed Stage 3 records
(``FilingFundamentals``/``DailyPriceObservation``/
``UniverseMembershipRecord``/``SectorRecord``) — never raw provider
payloads. Severity determines behavior explicitly:

- ``INFO``: recorded, never blocks anything.
- ``WARNING``: recorded, permits feature generation for the affected
  instrument/quarter.
- ``ERROR``: rejects the single affected instrument or quarter, not the
  whole run.
- ``FATAL``: blocks the full production run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Sequence

from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import DailyPriceObservation, FilingFundamentals, SectorRecord, UniverseMembershipRecord
from atlas_quant.domain.identifiers import InstrumentId


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass(frozen=True, slots=True)
class DataValidationIssue:
    severity: ValidationSeverity
    category: str
    subject: str
    message: str


@dataclass(frozen=True, slots=True)
class DataValidationSummary:
    issues: tuple[DataValidationIssue, ...] = field(default_factory=tuple)

    @property
    def has_fatal(self) -> bool:
        return any(i.severity == ValidationSeverity.FATAL for i in self.issues)

    @property
    def has_error(self) -> bool:
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    def counts_by_severity(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ValidationSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts

    def by_category(self, category: str) -> tuple[DataValidationIssue, ...]:
        return tuple(i for i in self.issues if i.category == category)


def validate_filings(filings: Sequence[FilingFundamentals]) -> tuple[DataValidationIssue, ...]:
    """Filing validation: required fields, duplicates, zero-denominator cases.

    ``FilingFundamentals.__post_init__`` already rejects an empty
    ``fiscal_period`` and a ``filed_at`` before ``quarter_end`` at
    construction time — those cases cannot reach this function at all;
    only conditions the domain type itself permits are checked here.
    """
    issues: list[DataValidationIssue] = []
    seen: dict[tuple, list[FilingFundamentals]] = {}
    for filing in filings:
        key = (filing.instrument_id, filing.quarter_end)
        seen.setdefault(key, []).append(filing)

    for (instrument_id, quarter_end), group in seen.items():
        subject = f"{instrument_id.symbol}@{quarter_end.isoformat()}"
        if len(group) > 1:
            issues.append(
                DataValidationIssue(
                    ValidationSeverity.WARNING, "filing", subject,
                    f"{len(group)} filings for the same instrument/quarter (amendment/restatement or "
                    "duplicate) -- Stage 3's point-in-time selector resolves this by filed_at, not this "
                    "validator",
                )
            )
        for filing in group:
            if filing.revenue is not None and filing.revenue == 0:
                issues.append(DataValidationIssue(ValidationSeverity.WARNING, "filing", subject, "revenue is exactly zero"))
            if filing.stockholders_equity is not None and filing.stockholders_equity == 0:
                issues.append(DataValidationIssue(ValidationSeverity.WARNING, "filing", subject, "stockholders_equity is exactly zero"))
            for field_name in ("revenue", "gross_profit", "operating_income", "net_income", "diluted_eps",
                               "stockholders_equity", "operating_cash_flow", "capital_expenditure"):
                value = getattr(filing, field_name)
                if value is not None and not math.isfinite(value):
                    issues.append(
                        DataValidationIssue(ValidationSeverity.ERROR, "filing", subject, f"{field_name} is non-finite")
                    )
            if filing.accession_number is None:
                issues.append(DataValidationIssue(ValidationSeverity.INFO, "filing", subject, "no accession_number recorded"))

    return tuple(issues)


def validate_prices(
    prices: Sequence[DailyPriceObservation], *, max_gap_calendar_days: int = 10
) -> tuple[DataValidationIssue, ...]:
    """Price validation: duplicates, ordering, non-positive/non-finite closes, long gaps."""
    issues: list[DataValidationIssue] = []
    if not prices:
        issues.append(DataValidationIssue(ValidationSeverity.FATAL, "price", "<empty>", "no price observations supplied"))
        return tuple(issues)

    by_instrument: dict[InstrumentId, list[DailyPriceObservation]] = {}
    for p in prices:
        by_instrument.setdefault(p.instrument_id, []).append(p)

    for instrument_id, series in by_instrument.items():
        subject = instrument_id.symbol
        dates_seen: dict[date, int] = {}
        for p in series:
            dates_seen[p.trading_date] = dates_seen.get(p.trading_date, 0) + 1
            if p.close <= 0:
                issues.append(DataValidationIssue(ValidationSeverity.ERROR, "price", subject, f"non-positive close on {p.trading_date}"))
            if not math.isfinite(p.close):
                issues.append(DataValidationIssue(ValidationSeverity.ERROR, "price", subject, f"non-finite close on {p.trading_date}"))

        for d, count in dates_seen.items():
            if count > 1:
                issues.append(DataValidationIssue(ValidationSeverity.WARNING, "price", subject, f"{count} observations for {d}"))

        ordered_dates = sorted(dates_seen)
        input_dates = [p.trading_date for p in series]
        if input_dates != sorted(input_dates):
            issues.append(DataValidationIssue(ValidationSeverity.INFO, "price", subject, "input rows are not in chronological order"))

        for prev, curr in zip(ordered_dates, ordered_dates[1:]):
            gap = (curr - prev).days
            if gap > max_gap_calendar_days:
                issues.append(
                    DataValidationIssue(
                        ValidationSeverity.WARNING, "price", subject,
                        f"{gap}-calendar-day gap between {prev} and {curr} (possible halt/delisting/data gap)",
                    )
                )

    return tuple(issues)


def validate_universe(members: Sequence[UniverseMembershipRecord]) -> tuple[DataValidationIssue, ...]:
    issues: list[DataValidationIssue] = []
    if not members:
        issues.append(DataValidationIssue(ValidationSeverity.FATAL, "universe", "<empty>", "no universe membership records supplied"))
        return tuple(issues)

    seen: dict[InstrumentId, int] = {}
    for m in members:
        seen[m.instrument_id] = seen.get(m.instrument_id, 0) + 1
    for instrument_id, count in seen.items():
        if count > 1:
            issues.append(DataValidationIssue(ValidationSeverity.WARNING, "universe", instrument_id.symbol, f"{count} membership records for the same instrument"))

    survivorship_biased_flags = {m.survivorship_biased for m in members}
    if len(survivorship_biased_flags) > 1:
        issues.append(
            DataValidationIssue(ValidationSeverity.ERROR, "universe", "<all>", "inconsistent survivorship_biased flag across records")
        )
    elif True in survivorship_biased_flags:
        issues.append(
            DataValidationIssue(
                ValidationSeverity.INFO, "universe", "<all>",
                "universe is survivorship-biased (present-day snapshot) -- report_current.html's own "
                "documented approach, not a defect",
            )
        )

    return tuple(issues)


def validate_sectors(records: Sequence[SectorRecord]) -> tuple[DataValidationIssue, ...]:
    issues: list[DataValidationIssue] = []
    seen: dict[InstrumentId, set] = {}
    for r in records:
        seen.setdefault(r.instrument_id, set()).add(r.raw_sector)
        if r.raw_sector is None:
            issues.append(DataValidationIssue(ValidationSeverity.INFO, "sector", r.instrument_id.symbol, "missing raw_sector (will normalize to Unknown)"))
    for instrument_id, sectors in seen.items():
        if len(sectors) > 1:
            issues.append(DataValidationIssue(ValidationSeverity.WARNING, "sector", instrument_id.symbol, f"conflicting raw sectors recorded: {sorted(str(s) for s in sectors)}"))
    return tuple(issues)


def validate_calendar(calendar: TradingCalendar, start: date, end: date) -> tuple[DataValidationIssue, ...]:
    issues: list[DataValidationIssue] = []
    try:
        current = calendar.trading_day_on_or_before(end)
    except ValueError:
        issues.append(DataValidationIssue(ValidationSeverity.FATAL, "calendar", "<range>", f"no trading day on or before {end}"))
        return tuple(issues)
    if current < start:
        issues.append(DataValidationIssue(ValidationSeverity.FATAL, "calendar", "<range>", f"no trading day within [{start}, {end}]"))
    return tuple(issues)
