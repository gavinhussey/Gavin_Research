"""Chronological rolling-window training-dataset construction, report §4.4.

"The model is retrained each quarter on the trailing ML_TRAIN_YEARS = 3
years of labelled data": D_train^(q) = {(x_i,q', y_i,q') : q - 3yr <= q' < q}
— note the strict ``q' < q``: the target quarter itself is always
excluded, and only *earlier* quarters may contribute. This module adds
one requirement the report's set-builder notation doesn't spell out but
that is essential for point-in-time correctness: a labeled row may only
be used once its own label is knowable (``label_available_at <
training_cutoff``), not merely because its quarter falls inside the
trailing window. The comparison is strict: a label whose availability
timestamp exactly equals ``training_cutoff`` (e.g. the immediately
preceding quarter's exit close, which lands on the same calendar day as
the current quarter's entry) is not yet knowable at the moment entry
decisions are made, since that price is only realized at market close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping, Sequence

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FeatureObservation
from atlas_quant.strategies.multi_factor_ranking_ml.model_schema import (
    FeatureMatrix,
    build_feature_matrix,
    compute_model_schema_identity,
)


@dataclass(frozen=True, slots=True)
class LabeledObservation:
    """One instrument/quarter's feature observation plus its report §4.2 label."""

    observation: FeatureObservation
    label: int
    label_available_at: datetime


@dataclass(frozen=True, slots=True)
class ExcludedQuarter:
    quarter_end: date
    reason: str


@dataclass(frozen=True, slots=True)
class TrainingDatasetResult:
    """The complete, structured rolling-window training dataset for one target quarter."""

    target_quarter_end: date
    training_cutoff: datetime
    candidate_quarters: tuple[date, ...]
    included_quarters: tuple[date, ...]
    excluded_quarters: tuple[ExcludedQuarter, ...]
    feature_matrix: FeatureMatrix
    labels: tuple[int, ...]
    instrument_ids: tuple[InstrumentId, ...]
    feature_timestamps: tuple[date, ...]
    label_available_timestamps: tuple[datetime, ...]
    positive_label_count: int
    negative_label_count: int
    total_row_count: int
    quarter_count: int
    model_schema_identity: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)


def _trailing_window_start(target_quarter_end: date, ml_train_years: int) -> date:
    """``target_quarter_end`` minus ``ml_train_years`` years (calendar-year
    arithmetic; a Feb-29 target in a leap year steps back to Feb-28 in a
    non-leap year rather than raising — an explicit, documented
    approximation of "3 years," not a trading-day-exact boundary, since
    the report expresses this window in whole years)."""
    try:
        return target_quarter_end.replace(year=target_quarter_end.year - ml_train_years)
    except ValueError:
        return target_quarter_end.replace(year=target_quarter_end.year - ml_train_years, day=28)


def build_training_dataset(
    target_quarter_end: date,
    training_cutoff: datetime,
    labeled_quarters: Mapping[date, Sequence[LabeledObservation]],
    *,
    strategy_id: str,
    feature_schema_version: str,
    ml_train_years: int,
    model_config_identity: str,
) -> TrainingDatasetResult:
    """Build the trailing ``ml_train_years``-window training dataset for ``target_quarter_end``.

    Excludes: the target quarter itself, any quarter on/after it, any
    quarter before the trailing-window start, and any individual labeled
    row whose ``label_available_at`` is on or after ``training_cutoff``
    (even if its quarter otherwise falls inside the window) — this is the
    leakage guard: a quarter can be partially included if some of its
    rows' outcomes were knowable by ``training_cutoff`` and others were
    not, though in practice every row sharing one quarter also shares one
    ``sell_timestamp``/``label_available_at``. The comparison is strict
    (``<``, not ``<=``) because a label available *exactly at*
    ``training_cutoff`` is not yet realized at the instant entry
    decisions are made — this matters at the boundary, since a quarter's
    ``sell_timestamp`` is defined to equal the following quarter's own
    ``training_cutoff``/``entry_timestamp``.
    """
    window_start = _trailing_window_start(target_quarter_end, ml_train_years)
    audit = AuditTrail()
    excluded: list[ExcludedQuarter] = []
    candidate_quarters = sorted(labeled_quarters.keys())

    included_rows: list[LabeledObservation] = []
    included_quarters: list[date] = []

    for quarter_end in candidate_quarters:
        if quarter_end >= target_quarter_end:
            excluded.append(ExcludedQuarter(quarter_end, "target quarter or later"))
            continue
        if quarter_end < window_start:
            excluded.append(ExcludedQuarter(quarter_end, "outside trailing ml_train_years window"))
            continue

        rows_for_quarter = labeled_quarters[quarter_end]
        knowable = [r for r in rows_for_quarter if r.label_available_at < training_cutoff]
        if not knowable:
            excluded.append(ExcludedQuarter(quarter_end, "label not yet available by training_cutoff"))
            continue

        included_rows.extend(knowable)
        included_quarters.append(quarter_end)

    observations = [r.observation for r in included_rows]
    matrix = build_feature_matrix(
        observations, strategy_id=strategy_id, feature_schema_version=feature_schema_version
    )

    surviving_keys = set(zip(matrix.instrument_ids, matrix.feature_timestamps))
    surviving_rows = [
        r
        for r in included_rows
        if (r.observation.instrument_id, r.observation.feature_timestamp) in surviving_keys
    ]

    labels = tuple(r.label for r in surviving_rows)
    label_available_timestamps = tuple(r.label_available_at for r in surviving_rows)

    warnings: list[str] = []
    if matrix.rejected:
        warnings.append(f"{len(matrix.rejected)} observation(s) rejected by feature-matrix validation")

    audit = audit.append(
        AuditRecord(
            stage="training_window",
            message=(
                f"{len(included_quarters)} of {len(candidate_quarters)} candidate quarter(s) "
                f"included, {len(matrix)} row(s) total"
            ),
            timestamp=training_cutoff,
            data={"excluded_reasons": [e.reason for e in excluded]},
        )
    )

    return TrainingDatasetResult(
        target_quarter_end=target_quarter_end,
        training_cutoff=training_cutoff,
        candidate_quarters=tuple(candidate_quarters),
        included_quarters=tuple(included_quarters),
        excluded_quarters=tuple(excluded),
        feature_matrix=matrix,
        labels=labels,
        instrument_ids=matrix.instrument_ids,
        feature_timestamps=matrix.feature_timestamps,
        label_available_timestamps=label_available_timestamps,
        positive_label_count=sum(labels),
        negative_label_count=len(labels) - sum(labels),
        total_row_count=len(matrix),
        quarter_count=len(included_quarters),
        model_schema_identity=compute_model_schema_identity(feature_schema_version, model_config_identity),
        warnings=tuple(warnings),
        audit_trail=audit,
    )


@dataclass(frozen=True, slots=True)
class TrainingEligibilityResult:
    """The two report-defined training gates (§4.4 quarters, §4.2 positive
    labels — verified separately, never conflated), plus basic validity checks."""

    eligible: bool
    quarter_count: int
    min_train_quarters: int
    quarter_gate_passed: bool
    positive_label_count: int
    n_winners: int
    positive_label_gate_passed: bool
    non_empty: bool
    matrix_label_length_match: bool
    has_finite_values: bool
    both_classes_present: bool
    reasons: tuple[str, ...]


def check_training_eligibility(
    dataset: TrainingDatasetResult, *, min_train_quarters: int, n_winners: int
) -> TrainingEligibilityResult:
    """Evaluate both report-defined training gates plus basic dataset validity.

    Report §4.4: ``quarter_count >= min_train_quarters`` (default 8).
    Report §4.2 (cross-checked, Stage 2.1 finding): the training-viability
    positive-label gate reuses ``n_winners`` (default 10) — there is no
    separate ``min_positive_labels`` value in the report, so this checks
    ``positive_label_count >= n_winners`` directly, not a distinct
    configuration field.
    """
    reasons: list[str] = []

    quarter_gate_passed = dataset.quarter_count >= min_train_quarters
    if not quarter_gate_passed:
        reasons.append(
            f"quarter_count={dataset.quarter_count} < min_train_quarters={min_train_quarters}"
        )

    positive_label_gate_passed = dataset.positive_label_count >= n_winners
    if not positive_label_gate_passed:
        reasons.append(
            f"positive_label_count={dataset.positive_label_count} < n_winners={n_winners}"
        )

    non_empty = dataset.total_row_count > 0
    if not non_empty:
        reasons.append("feature matrix is empty")

    matrix_label_length_match = len(dataset.feature_matrix) == len(dataset.labels)
    if not matrix_label_length_match:
        reasons.append(
            f"feature matrix row count ({len(dataset.feature_matrix)}) != "
            f"label count ({len(dataset.labels)})"
        )

    has_finite_values = all(
        math.isfinite(value) or math.isnan(value)
        for row in dataset.feature_matrix.rows
        for value in row
    )
    if not has_finite_values:
        reasons.append("a non-finite (infinite) feature value is present")

    both_classes_present = non_empty and 0 in dataset.labels and 1 in dataset.labels
    if non_empty and not both_classes_present:
        reasons.append("only one label class is present")

    eligible = (
        quarter_gate_passed
        and positive_label_gate_passed
        and non_empty
        and matrix_label_length_match
        and has_finite_values
        and both_classes_present
    )

    return TrainingEligibilityResult(
        eligible=eligible,
        quarter_count=dataset.quarter_count,
        min_train_quarters=min_train_quarters,
        quarter_gate_passed=quarter_gate_passed,
        positive_label_count=dataset.positive_label_count,
        n_winners=n_winners,
        positive_label_gate_passed=positive_label_gate_passed,
        non_empty=non_empty,
        matrix_label_length_match=matrix_label_length_match,
        has_finite_values=has_finite_values,
        both_classes_present=both_classes_present,
        reasons=tuple(reasons),
    )
