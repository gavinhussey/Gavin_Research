"""Ranked Multi-Factor Rotation — anti-look-ahead controls for the
weight-estimation panel (spec §4A/§4B).

This module answers exactly one question, reusably: **given a candidate
training row from ``panel.RmfrPanelRow`` and the date a future estimator
is generating weights "as of," is that row allowed to be used?** It does
not fit a regression or estimate ``wM``/``wV``/``wC`` -- it only decides,
before any fitting happens, which historical rows are safe to hand to
whatever future stage does that fitting.

Canonical eligibility condition (this stage's own specification):

    training_row.forward_return_end < current_estimation_as_of_date

Strict ``<``, matching this repository's one other cross-period
point-in-time training gate
(``filing_momentum_ml.training_dataset.build_training_dataset``'s
``label_available_at < training_cutoff``, same rationale: an outcome
whose availability timestamp exactly equals the cutoff is not yet
realized at the instant the estimation decision is made -- see that
module's docstring). No repository market-close convention was found
that requires a different comparison for this strategy -- see
``check_temporal_eligibility``'s docstring for the full derivation of
why strict ``<`` is correct (not merely a default) for RMFR's specific
monthly, simultaneous-rebalance timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Sequence

from atlas_quant.strategies.ranked_multi_factor_rotation.panel import RmfrPanelRow

#: Bumped whenever this module's eligibility rule set changes meaning.
ELIGIBILITY_SCHEMA_VERSION = "1"

# -- Closed, documented exclusion-reason vocabulary -- never free text. --
EXCLUSION_UPSTREAM_PANEL_ROW_INELIGIBLE = "upstream_panel_row_ineligible"
EXCLUSION_FEATURE_AFTER_ESTIMATION = "feature_as_of_after_estimation_as_of"
EXCLUSION_ROW_FROM_PREDICTION_PERIOD_OR_LATER = "row_rebalance_date_on_or_after_estimation_as_of"
EXCLUSION_TARGET_OVERLAPS_ESTIMATION = "target_window_overlaps_estimation_as_of"
EXCLUSION_TARGET_END_NOT_BEFORE_ESTIMATION = "target_end_not_before_estimation_as_of"
EXCLUSION_OBSERVATION_NOT_YET_AVAILABLE = "observation_not_yet_available"
EXCLUSION_WITHIN_EMBARGO = "within_embargo_window"


@dataclass(frozen=True, slots=True)
class TemporalObservationMetadata:
    """Panel-row-independent temporal metadata for one training
    candidate, this stage's task-2 fields:

    - ``feature_as_of``: the row's own point-in-time feature cutoff
      (``RmfrPanelRow.factor_data_as_of``) -- every feature value
      (momentum/volatility/correlation/trend, and the SHY-relative
      ``asset_return_4m``/``shy_return_4m`` legs) is already bounded by
      this date by construction of ``pipeline.compute_factor_snapshot``
      /``formulas.excess_absolute_momentum_at``'s own ``.loc[:as_of]``
      truncation -- this module does not recompute or re-verify that
      bound, it only re-exposes it under this stage's field name.
    - ``target_start``/``target_end``: the row's forward-return window
      (``RmfrPanelRow.forward_return_start``/``forward_return_end``).
    - ``observation_available_at``: see
      :func:`observation_metadata_for_row`'s docstring for the exact,
      conservative definition.
    """

    feature_as_of: date
    target_start: date
    target_end: date
    observation_available_at: datetime


def observation_metadata_for_row(row: RmfrPanelRow) -> TemporalObservationMetadata:
    """Derive :class:`TemporalObservationMetadata` from one
    ``panel.RmfrPanelRow`` -- a pure renaming/reshaping, never a
    recomputation: ``feature_as_of``/``target_start``/``target_end`` are
    read directly off the row's own ``factor_data_as_of``/
    ``forward_return_start``/``forward_return_end`` fields (already
    produced by the panel builder from this strategy's production
    functions, per the previous stage).

    ``observation_available_at`` is defined **conservatively** (task 3)
    as midnight of ``target_end`` --
    ``datetime.combine(row.forward_return_end, datetime.min.time())`` --
    the exact same "available at midnight of the day the outcome is
    realized" convention
    ``filing_momentum_ml.forward_return.ForwardReturnOutcome
    .label_available_at`` already uses for this repository's other
    forward-return outcome. This is conservative, not merely
    convenient: the return itself is only actually realized at that
    day's market *close*, not at its midnight/start -- using midnight
    means the strict ``<`` comparison against ``estimation_as_of``
    (also midnight-of-day, matching how this strategy's own
    ``month_end``/``data_cutoff`` are expressed throughout
    ``pipeline.py``/``backtest_clock.py``) is what actually enforces
    same-day exclusion, exactly as
    ``training_dataset.build_training_dataset`` documents for its own
    analogous boundary.
    """
    return TemporalObservationMetadata(
        feature_as_of=row.factor_data_as_of,
        target_start=row.forward_return_start,
        target_end=row.forward_return_end,
        observation_available_at=datetime.combine(row.forward_return_end, datetime.min.time()),
    )


@dataclass(frozen=True, slots=True)
class TemporalEligibilityResult:
    """One row's complete temporal-eligibility verdict, every applicable
    exclusion reason included (not just the first one found) so an
    audit can show every rule a row violates, not only the first."""

    rebalance_date: date
    ticker: str
    estimation_as_of: date
    metadata: TemporalObservationMetadata
    eligible: bool
    exclusion_reasons: tuple[str, ...] = field(default_factory=tuple)


def check_temporal_eligibility(
    row: RmfrPanelRow, estimation_as_of: date
) -> TemporalEligibilityResult:
    """**The reusable temporal eligibility validator** (task 1): may
    ``row`` be used to train an estimator generating weights for a
    rebalance dated ``estimation_as_of``?

    Canonical rule: ``row.forward_return_end < estimation_as_of``
    (strict). Derivation for *why* strict ``<`` is correct here, not
    merely the safe default: for two adjacent RMFR rebalance periods P
    (the row's own period) and R (the period being estimated for) where
    P immediately precedes R, P's ``forward_return_end`` equals R's own
    ``entry_timestamp`` date (``backtest_clock.RmfrBacktestPeriod``'s
    simultaneous-rebalance chaining: one period's ``exit_timestamp`` is
    always the next period's ``entry_timestamp``) -- i.e. the trading
    day *immediately after* R's own ``month_end``. That date can never
    equal ``estimation_as_of`` when ``estimation_as_of`` is R's own
    ``month_end`` (it is strictly later), so for RMFR's actual monthly
    cadence this stage's strict-vs-inclusive choice never changes which
    rows qualify in practice -- but strict is used regardless, both to
    match this repository's one other point-in-time training gate
    exactly (``training_dataset.build_training_dataset``) and as a
    deliberately conservative default should a future non-monthly
    variant ever produce a genuine same-day boundary.

    Every applicable check is independently evaluated and every
    violated rule is reported in ``exclusion_reasons`` (a row can fail
    more than one simultaneously):

    - ``EXCLUSION_UPSTREAM_PANEL_ROW_INELIGIBLE``: the panel itself
      already marked this row ineligible (spec §4B) -- a row excluded
      for a feature/target *data-quality* reason must not separately
      become "temporally eligible."
    - ``EXCLUSION_FEATURE_AFTER_ESTIMATION``: ``feature_as_of >
      estimation_as_of`` -- catches "feature timestamps after
      rebalance." This is the same gate that, transitively, also covers
      "future SHY data": every feature this row carries, including the
      SHY-relative ``asset_return_4m``/``shy_return_4m`` legs, is
      already bounded by ``feature_as_of`` at construction time (see
      ``excess_absolute_momentum_at``'s own no-forward-looking-fill
      contract) -- there is no separate SHY-specific temporal channel
      to check.
    - ``EXCLUSION_ROW_FROM_PREDICTION_PERIOD_OR_LATER``:
      ``rebalance_date >= estimation_as_of`` -- catches "training on
      rows from the prediction month" (and any later month) directly,
      by the row's own rebalance identity, not only via its target
      window.
    - ``EXCLUSION_TARGET_OVERLAPS_ESTIMATION``: ``target_start <=
      estimation_as_of <= target_end`` -- catches "target overlap with
      estimation date."
    - ``EXCLUSION_TARGET_END_NOT_BEFORE_ESTIMATION``: ``target_end >=
      estimation_as_of`` -- catches "target end after estimation date"
      (the direct negation of the canonical rule above; kept as its own
      named, independently-testable check even though it will always
      co-fire with the overlap check for this strategy's non-overlapping
      monthly windows).
    - ``EXCLUSION_OBSERVATION_NOT_YET_AVAILABLE``:
      ``observation_available_at >= estimation_as_of`` (both expressed
      at midnight) -- the ``observation_available_at``-based expression
      of the same canonical rule, kept separate so a future
      redefinition of "conservative" (e.g. requiring a same-day-close
      buffer) only has to change :func:`observation_metadata_for_row`,
      not this function's logic.

    "Future adjusted-price revisions" (also named in this stage's task
    list) is **not checked here and cannot be, from this data model**:
    ``pipeline.observations_to_price_frames`` discards each
    ``DailyOHLCObservation``'s ``provenance.retrieved_at`` when building
    the plain OHLC ``DataFrame``s every RMFR calculation (including this
    panel) consumes, so no "when was this historical price actually
    pulled" signal survives to check against. This is a disclosed, not
    silently ignored, gap -- see ``docs/reproducibility_findings.md``.
    """
    metadata = observation_metadata_for_row(row)
    reasons: list[str] = []

    if not row.eligible:
        reasons.append(EXCLUSION_UPSTREAM_PANEL_ROW_INELIGIBLE)
    if metadata.feature_as_of > estimation_as_of:
        reasons.append(EXCLUSION_FEATURE_AFTER_ESTIMATION)
    if row.rebalance_date >= estimation_as_of:
        reasons.append(EXCLUSION_ROW_FROM_PREDICTION_PERIOD_OR_LATER)
    if metadata.target_start <= estimation_as_of <= metadata.target_end:
        reasons.append(EXCLUSION_TARGET_OVERLAPS_ESTIMATION)
    if metadata.target_end >= estimation_as_of:
        reasons.append(EXCLUSION_TARGET_END_NOT_BEFORE_ESTIMATION)
    estimation_as_of_midnight = datetime.combine(estimation_as_of, datetime.min.time())
    if metadata.observation_available_at >= estimation_as_of_midnight:
        reasons.append(EXCLUSION_OBSERVATION_NOT_YET_AVAILABLE)

    return TemporalEligibilityResult(
        rebalance_date=row.rebalance_date,
        ticker=row.ticker,
        estimation_as_of=estimation_as_of,
        metadata=metadata,
        eligible=not reasons,
        exclusion_reasons=tuple(reasons),
    )


def find_duplicate_observations(
    rows: Sequence[RmfrPanelRow],
) -> tuple[tuple[date, str], ...]:
    """Every ``(rebalance_date, ticker)`` key that appears more than once
    in ``rows`` -- a candidate training set assembled from more than one
    source (e.g. two overlapping panel builds) can reintroduce
    duplicates even though a single ``panel.RmfrWeightEstimationPanel``
    already rejects them internally at construction. Returns each
    duplicated key once, in first-seen order; an empty tuple means no
    duplicates were found.
    """
    seen: set[tuple[date, str]] = set()
    duplicates: list[tuple[date, str]] = []
    duplicates_seen: set[tuple[date, str]] = set()
    for row in rows:
        key = (row.rebalance_date, row.ticker)
        if key in seen and key not in duplicates_seen:
            duplicates.append(key)
            duplicates_seen.add(key)
        seen.add(key)
    return tuple(duplicates)


def detect_overlapping_target_windows(
    rows: Sequence[RmfrPanelRow],
) -> tuple[tuple[RmfrPanelRow, RmfrPanelRow], ...]:
    """**Purging support** (task 5): every pair of *same-ticker* rows
    whose ``[target_start, target_end]`` windows genuinely overlap (share
    more than a single boundary date) -- the condition classic
    purged-cross-validation guards against, since two overlapping target
    windows are partly driven by the same underlying market moves.

    For RMFR's actual monthly, one-month-ahead, simultaneous-rebalance
    design this always returns empty (see
    :func:`embargo_days_required_for_monthly_targets`'s docstring for
    the full derivation of why) -- this function exists so a future
    config variant that *does* produce overlapping windows (e.g. a
    multi-month-forward target re-evaluated every month) has a ready,
    tested detector rather than silently trusting that overlap can't
    happen.

    Two windows sharing exactly one boundary date (one's ``target_end``
    equals another's ``target_start`` -- RMFR's own adjacent-period
    convention) are **not** reported as overlapping: that is the
    intended "no gap, no overlap" simultaneous-rebalance boundary, not a
    leakage condition.
    """
    overlaps: list[tuple[RmfrPanelRow, RmfrPanelRow]] = []
    by_ticker: dict[str, list[RmfrPanelRow]] = {}
    for row in rows:
        by_ticker.setdefault(row.ticker, []).append(row)

    for ticker_rows in by_ticker.values():
        ordered = sorted(ticker_rows, key=lambda r: r.forward_return_start)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if a.forward_return_start == b.forward_return_start and a.forward_return_end == b.forward_return_end:
                    overlaps.append((a, b))
                    continue
                overlap_start = max(a.forward_return_start, b.forward_return_start)
                overlap_end = min(a.forward_return_end, b.forward_return_end)
                if overlap_start < overlap_end:
                    overlaps.append((a, b))
    return tuple(overlaps)


def embargo_days_required_for_monthly_targets() -> int:
    """**Documented answer to task 6**: for RMFR's actual configuration
    -- monthly rebalances, a one-month-ahead forward target
    (``forward_return_start``/``forward_return_end`` = the very next
    holding period, spec §6) -- **no embargo is required, and this
    returns ``0``**.

    Consecutive rebalance periods' target windows are *contiguous, not
    overlapping*: period P's ``forward_return_end`` equals period P+1's
    own ``forward_return_start`` (``backtest_clock``'s simultaneous-
    rebalance chaining -- one period's exit is the next period's entry),
    so different periods' windows share at most a single boundary date,
    never a date *range*. Classic purging/embargo guidance (removing
    training observations near a test point, plus a buffer, because
    overlapping label windows share market-move information) applies
    when windows genuinely overlap; RMFR's one-month-ahead, monthly-
    cadence design structurally does not produce that condition --
    confirmed empirically by :func:`detect_overlapping_target_windows`
    always returning empty for panels built from
    ``backtest_clock.generate_monthly_periods`` (tested).

    This is *not* a claim that no future RMFR variant could need an
    embargo -- a config using a longer (e.g. 3-month) forward target
    re-evaluated every month, or a sub-monthly rebalance cadence with a
    monthly target, would reintroduce overlap and require one. This
    function documents today's answer for today's configuration, not a
    general guarantee; :func:`detect_overlapping_target_windows` is the
    mechanism a future stage should call to re-check this if the
    forward-target construction ever changes.
    """
    return 0


def apply_embargo(
    rows: Sequence[RmfrPanelRow], estimation_as_of: date, *, embargo_days: int = 0
) -> tuple[RmfrPanelRow, ...]:
    """Exclude any row whose ``forward_return_end`` falls within
    ``embargo_days`` calendar days *before* ``estimation_as_of``, beyond
    the strict inequality :func:`check_temporal_eligibility` already
    enforces -- an additional, optional buffer for a future config where
    :func:`detect_overlapping_target_windows` finds real overlap (see
    :func:`embargo_days_required_for_monthly_targets`). ``embargo_days=0``
    (the default, and the correct value for RMFR's current monthly,
    non-overlapping design) is a no-op beyond the eligibility check
    itself.
    """
    if embargo_days < 0:
        raise ValueError(f"embargo_days must be >= 0, got {embargo_days!r}")
    if embargo_days == 0:
        return tuple(rows)
    cutoff = estimation_as_of - timedelta(days=embargo_days)
    return tuple(r for r in rows if r.forward_return_end < cutoff)


@dataclass(frozen=True, slots=True)
class TemporalEligibilityAuditReport:
    """Task 9's audit report: eligible rows, excluded rows, exclusion
    reasons, latest target end used, and the estimation cutoff -- one
    structured, serializable, testable object. Building this report
    never fits anything (task 10)."""

    estimation_as_of: date
    total_candidate_rows: int
    eligible_row_count: int
    excluded_row_count: int
    exclusion_reason_counts: dict[str, int]
    latest_target_end_used: date | None
    eligible_rows: tuple[RmfrPanelRow, ...]
    excluded_results: tuple[TemporalEligibilityResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "estimation_as_of": self.estimation_as_of.isoformat(),
            "total_candidate_rows": self.total_candidate_rows,
            "eligible_row_count": self.eligible_row_count,
            "excluded_row_count": self.excluded_row_count,
            "exclusion_reason_counts": dict(self.exclusion_reason_counts),
            "latest_target_end_used": (
                self.latest_target_end_used.isoformat() if self.latest_target_end_used else None
            ),
        }


def build_temporal_eligibility_audit(
    rows: Sequence[RmfrPanelRow], estimation_as_of: date, *, embargo_days: int = 0
) -> TemporalEligibilityAuditReport:
    """Run :func:`check_temporal_eligibility` over every row in ``rows``
    and summarize the result (task 9). Raises ``ValueError`` if ``rows``
    contains a duplicated ``(rebalance_date, ticker)`` observation (task
    7's "duplicated observations" catch) -- a candidate-set construction
    error, never silently deduplicated, matching
    ``panel.RmfrWeightEstimationPanel``'s own duplicate-key policy.

    ``embargo_days`` (default 0, correct for RMFR's current monthly
    design -- see :func:`embargo_days_required_for_monthly_targets`) is
    applied via :func:`apply_embargo` before per-row eligibility
    checking; an embargoed row is reported with
    ``EXCLUSION_WITHIN_EMBARGO`` rather than silently vanishing from the
    report.
    """
    duplicates = find_duplicate_observations(rows)
    if duplicates:
        raise ValueError(
            f"duplicate (rebalance_date, ticker) observation(s) in candidate rows: {duplicates!r}"
        )

    embargoed = set(apply_embargo(rows, estimation_as_of, embargo_days=embargo_days))
    results: list[TemporalEligibilityResult] = []
    for row in rows:
        result = check_temporal_eligibility(row, estimation_as_of)
        if row not in embargoed and not result.exclusion_reasons:
            result = TemporalEligibilityResult(
                rebalance_date=result.rebalance_date,
                ticker=result.ticker,
                estimation_as_of=result.estimation_as_of,
                metadata=result.metadata,
                eligible=False,
                exclusion_reasons=(EXCLUSION_WITHIN_EMBARGO,),
            )
        results.append(result)

    eligible_rows = tuple(row for row, result in zip(rows, results) if result.eligible)
    excluded_results = tuple(result for result in results if not result.eligible)

    reason_counts: dict[str, int] = {}
    for result in excluded_results:
        for reason in result.exclusion_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    latest_target_end_used = max((r.forward_return_end for r in eligible_rows), default=None)

    return TemporalEligibilityAuditReport(
        estimation_as_of=estimation_as_of,
        total_candidate_rows=len(rows),
        eligible_row_count=len(eligible_rows),
        excluded_row_count=len(excluded_results),
        exclusion_reason_counts=reason_counts,
        latest_target_end_used=latest_target_end_used,
        eligible_rows=eligible_rows,
        excluded_results=excluded_results,
    )
