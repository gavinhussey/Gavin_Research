"""Generic report domain models — shared vocabulary for any future strategy's report.

Nothing here is Filing Momentum ML-specific; that lives in
``atlas_quant.strategies.filing_momentum_ml.reporting``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Mapping, Sequence

from atlas_quant.config.identity import compute_config_identity

#: Bumped whenever the generic report-domain shape changes in a way that
#: could affect a consuming report's structure.
REPORT_DOMAIN_SCHEMA_VERSION = "1"


class ReproducibilityStatus(str, Enum):
    """How trustworthy a report's results are as reproductions of a historical claim.

    A report built from synthetic fixtures (this stage's own tests) must
    never claim ``FULLY_REPRODUCED`` — it should self-report ``NOT_RUN``
    or a fixture-specific status instead.
    """

    FULLY_REPRODUCED = "fully_reproduced"
    STRUCTURALLY_REPRODUCED = "structurally_reproduced"
    PARTIALLY_REPRODUCED = "partially_reproduced"
    NOT_REPRODUCIBLE_MISSING_DATA = "not_reproducible_missing_data"
    NOT_REPRODUCIBLE_CONFIGURATION_MISMATCH = "not_reproducible_configuration_mismatch"
    NOT_REPRODUCIBLE_UNVERIFIED_CACHE = "not_reproducible_unverified_cache"
    NOT_RUN = "not_run"


class ComparisonStatus(str, Enum):
    MATCH = "match"
    WITHIN_TOLERANCE = "within_tolerance"
    DIFFERENT_EXPECTED = "different_expected"
    DIFFERENT_UNEXPLAINED = "different_unexplained"
    UNAVAILABLE_IN_SOURCE = "unavailable_in_source"
    UNAVAILABLE_IN_ATLASQUANT = "unavailable_in_atlasquant"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """One value-level comparison between the source report and AtlasQuant's own result."""

    key: str
    source_value: float | str | None
    atlasquant_value: float | str | None
    absolute_difference: float | None
    relative_difference: float | None
    tolerance: float | None
    status: ComparisonStatus
    explanation: str
    provenance: str


@dataclass(frozen=True, slots=True)
class ValidationCheckResult:
    """One structured, named validation check on a generated report."""

    name: str
    passed: bool
    message: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    checks: tuple[ValidationCheckResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> tuple[ValidationCheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """The deterministic identity of one generated report artifact (JSON, HTML, ...)."""

    artifact_type: str
    content_identity: str

    @classmethod
    def of_json(cls, content: str) -> "ArtifactIdentity":
        return cls(artifact_type="json", content_identity=compute_config_identity({"content": content}))

    @classmethod
    def of_html(cls, content: str) -> "ArtifactIdentity":
        return cls(artifact_type="html", content_identity=compute_config_identity({"content": content}))


@dataclass(frozen=True, slots=True)
class TableColumn:
    key: str
    label: str
    kind: str = "text"  # "text" | "number" | "percent" | "date"


@dataclass(frozen=True, slots=True)
class TableDefinition:
    """One report table — rows are plain, already-formatted-or-raw values keyed by column."""

    identifier: str
    title: str
    columns: tuple[TableColumn, ...]
    rows: tuple[Mapping[str, object], ...]
    row_limit: int | None = None


@dataclass(frozen=True, slots=True)
class ChartPoint:
    x_label: str
    y: float


@dataclass(frozen=True, slots=True)
class ChartSeriesDefinition:
    """Pure chart *data* — never a rendered image. Rendering (e.g. to SVG) is a
    separate, presentation-only step consuming this structure."""

    identifier: str
    title: str
    chart_type: str  # "line" | "bar" | "area" | "composition"
    series_labels: tuple[str, ...]
    points: tuple[tuple[ChartPoint, ...], ...]  # one tuple of points per series, aligned by index

    def is_empty(self) -> bool:
        return not any(self.points)


@dataclass(frozen=True, slots=True)
class SectionDefinition:
    """One report section: an identifier, title, and its already-built content payload."""

    identifier: str
    title: str
    available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    """Every report's shared identity/provenance metadata."""

    atlasquant_display_name: str
    strategy_id: str
    strategy_display_name: str
    strategy_version: str
    report_schema_version: str
    config_identity: str
    backtest_run_identity: str | None
    performance_analysis_identity: str | None
    report_identity: str
    provenance_notes: tuple[str, ...]
    warnings: tuple[str, ...]
    reproducibility_status: ReproducibilityStatus
    generated_at: datetime | None = None  # human-readable only; never part of report_identity
