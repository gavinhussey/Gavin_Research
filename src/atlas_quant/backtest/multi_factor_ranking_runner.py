"""The standalone Multi-Factor Ranking ML historical backtest runner.

This strategy is a pure ranking system: no positions, no weights, no
capital, no orders (see ``strategy.py``'s module docstring). A P&L/
equity-curve backtest -- filing_momentum_ml's shape, and this module's
own shape before this rewrite -- has no meaning here: there is nothing to
hold, size, or realize a return on as a *portfolio*.

Instead this evaluates the *ranking's quality* directly, the standard
approach for a pure cross-sectional ranking system: for each quarterly
evaluation cycle (``evaluation_schedule.quarterly_evaluation_cycles``),
rank the universe, then measure the Information Coefficient (IC) --
Spearman rank correlation between each ranked instrument's score and its
realized forward return to the next cycle. No weights or capital are
implied anywhere in this computation; IC and the decile spread below are
descriptive statistics about the ranking's predictive quality, never a
simulated trading strategy.

Orchestrates, per cycle: build features (``feature_pipeline.py``) ->
train on the trailing window of prior cycles' labeled data
(``training_dataset.py``/``model_training.py``, unchanged from
filing_momentum_ml's lineage -- these are generic quarter/InstrumentId
machinery, not filing-specific) -> score this cycle's candidates
(``scoring.py``) -> rank via the Stage 5 evaluator (``strategy.py``) ->
measure IC against the *next* cycle's realized returns.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Callable, Mapping, Sequence

from atlas_quant.config.identity import compute_config_identity
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.base import StrategyEvaluationContext
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.macro import MacroSeriesLookup
from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_ID,
    STRATEGY_VERSION,
    MultiFactorRankingMLConfig,
)
from atlas_quant.strategies.multi_factor_ranking_ml.decision_domain import MultiFactorRankingDecisionSummary
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import Estimator, EstimatorBuildInfo
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import EvaluationCycle
from atlas_quant.strategies.multi_factor_ranking_ml.feature_pipeline import FeaturePipelineResult, run_feature_pipeline
from atlas_quant.strategies.multi_factor_ranking_ml.forward_return import ForwardReturnOutcome, build_forward_return_outcome
from atlas_quant.strategies.multi_factor_ranking_ml.labeling import assign_quarterly_labels
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import ModelIdentity, TrainingState, train_model
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import FundamentalsFeatureRecord
from atlas_quant.strategies.multi_factor_ranking_ml.scoring import ScoringResult, score_observations
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder
from atlas_quant.strategies.multi_factor_ranking_ml.strategy import (
    MultiFactorRankingEvaluationInputs,
    MultiFactorRankingMLStrategy,
)
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import (
    LabeledObservation,
    TrainingDatasetResult,
    build_training_dataset,
    check_training_eligibility,
)


@dataclass(frozen=True, slots=True)
class MultiFactorRankingBacktestConfig:
    """Every behavior-changing backtest-level configuration value, bundled and identified.

    No ``strategy_budget_pct``/``price_policy``/``transaction_costs``/
    ``instrument_return_cap`` here -- those are P&L-portfolio concepts
    that don't apply to an IC/rank-correlation evaluation of a pure
    ranking system.
    """

    strategy_config: MultiFactorRankingMLConfig = field(default_factory=MultiFactorRankingMLConfig)

    def identity(self) -> str:
        return compute_config_identity({"strategy_config_identity": self.strategy_config.identity()})


@dataclass(frozen=True, slots=True)
class RankingCycleResult:
    """One quarterly cycle's ranking plus its measured forward-looking IC."""

    cycle: EvaluationCycle
    training_state: TrainingState | None
    model_identity: ModelIdentity | None
    scoring_result: ScoringResult | None
    decision_summary: MultiFactorRankingDecisionSummary | None
    ic: float | None
    decile_spread: float | None
    auc: float | None
    ranked_count: int
    scored_for_ic_count: int
    scored_for_auc_count: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "quarter_start": self.cycle.quarter_start.isoformat(),
            "cutoff": self.cycle.cutoff.isoformat(),
            "training_state": self.training_state.value if self.training_state else None,
            "ic": self.ic,
            "decile_spread": self.decile_spread,
            "auc": self.auc,
            "ranked_count": self.ranked_count,
            "scored_for_ic_count": self.scored_for_ic_count,
            "scored_for_auc_count": self.scored_for_auc_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ICBacktestResult:
    """The complete, structured top-level result of one IC backtest run.

    Headline stats are rank-correlation/IC based -- there is deliberately
    no equity curve, total return, Sharpe ratio, or drawdown here: those
    are P&L-portfolio concepts, and this strategy never holds a position.
    """

    strategy_id: str
    strategy_version: str
    config_identity: str
    cycle_results: tuple[RankingCycleResult, ...]
    run_identity: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)
    #: Cycles scored against fewer than this many candidates are excluded
    #: from every headline stat below (mean/std/hit-rate/spread) -- a
    #: Spearman correlation or AUC computed over a handful of names (this
    #: strategy's real early history: 9-12 stocks from 1986-10 through
    #: 1989-04, before the universe's real growth to 30+ by 1989-07 and
    #: 100+ by 1990-01) is dominated by sampling noise, not signal. Still
    #: present in ``cycle_results`` and ``to_dict()``'s per-cycle list --
    #: this only affects aggregation, never hides or deletes a cycle's
    #: own recorded result.
    min_scored_count: int = 30

    def _is_headline_eligible(self, cycle: RankingCycleResult, *, for_auc: bool = False) -> bool:
        count = cycle.scored_for_auc_count if for_auc else cycle.scored_for_ic_count
        return count >= self.min_scored_count

    @property
    def _valid_ics(self) -> list[float]:
        return [c.ic for c in self.cycle_results if c.ic is not None and self._is_headline_eligible(c)]

    @property
    def cycle_count(self) -> int:
        return len(self.cycle_results)

    @property
    def measured_cycle_count(self) -> int:
        """Cycles with a computable IC (enough scored candidates with a
        realized forward return, and a model that actually trained)."""
        return len(self._valid_ics)

    @property
    def mean_ic(self) -> float | None:
        ics = self._valid_ics
        return statistics.mean(ics) if ics else None

    @property
    def ic_std(self) -> float | None:
        ics = self._valid_ics
        if len(ics) < 2:
            return 0.0 if ics else None
        return statistics.pstdev(ics)

    @property
    def ic_information_ratio(self) -> float | None:
        """mean_ic / ic_std -- undefined (``None``) when std is zero or
        there are no measured cycles, never a fabricated infinity."""
        mean, std = self.mean_ic, self.ic_std
        if mean is None or not std:
            return None
        return mean / std

    @property
    def hit_rate(self) -> float | None:
        """Fraction of measured cycles with a positive IC."""
        ics = self._valid_ics
        if not ics:
            return None
        return sum(1 for ic in ics if ic > 0) / len(ics)

    @property
    def mean_decile_spread(self) -> float | None:
        spreads = [
            c.decile_spread for c in self.cycle_results
            if c.decile_spread is not None and self._is_headline_eligible(c)
        ]
        return statistics.mean(spreads) if spreads else None

    @property
    def _valid_aucs(self) -> list[float]:
        return [
            c.auc for c in self.cycle_results
            if c.auc is not None and self._is_headline_eligible(c, for_auc=True)
        ]

    @property
    def measured_auc_cycle_count(self) -> int:
        """Cycles with a computable AUC (both a winner and a non-winner
        present among that cycle's scored candidates)."""
        return len(self._valid_aucs)

    @property
    def mean_auc(self) -> float | None:
        aucs = self._valid_aucs
        return statistics.mean(aucs) if aucs else None

    @property
    def auc_std(self) -> float | None:
        aucs = self._valid_aucs
        if len(aucs) < 2:
            return 0.0 if aucs else None
        return statistics.pstdev(aucs)

    @property
    def auc_above_half_rate(self) -> float | None:
        """Fraction of measured cycles with AUC > 0.5 -- AUC's own
        no-skill baseline (a coin flip scores 0.5, not 0.0, so this plays
        the same role ``hit_rate`` plays for IC, calibrated to AUC's
        actual midpoint)."""
        aucs = self._valid_aucs
        if not aucs:
            return None
        return sum(1 for auc in aucs if auc > 0.5) / len(aucs)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "config_identity": self.config_identity,
            "run_identity": self.run_identity,
            "cycle_count": self.cycle_count,
            "min_scored_count": self.min_scored_count,
            "measured_cycle_count": self.measured_cycle_count,
            "mean_ic": self.mean_ic,
            "ic_std": self.ic_std,
            "ic_information_ratio": self.ic_information_ratio,
            "hit_rate": self.hit_rate,
            "mean_decile_spread": self.mean_decile_spread,
            "measured_auc_cycle_count": self.measured_auc_cycle_count,
            "mean_auc": self.mean_auc,
            "auc_std": self.auc_std,
            "auc_above_half_rate": self.auc_above_half_rate,
            "cycles": [c.to_dict() for c in self.cycle_results],
            "warnings": list(self.warnings),
            "audit_trail": self.audit_trail.to_dict(),
        }


def _ranks(values: Sequence[float]) -> list[float]:
    """1-based ascending ranks, ties averaged (the standard mid-rank
    convention) -- shared by :func:`spearman_correlation` and
    :func:`roc_auc`, both of which are rank-based statistics."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation between ``x`` and ``y``.

    Ties are averaged (the standard mid-rank convention). Returns
    ``None`` (never a fabricated 0.0) when there are fewer than 2 pairs,
    or when either series has zero variance in rank (every value tied --
    correlation is undefined, not zero).
    """
    n = len(x)
    if n != len(y):
        raise ValueError(f"x and y must be the same length, got {n} and {len(y)}")
    if n < 2:
        return None

    rx = _ranks(list(x))
    ry = _ranks(list(y))
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


def decile_spread(score_return_pairs: Sequence[tuple[float, float]]) -> float | None:
    """Top-decile mean forward return minus bottom-decile mean forward
    return, ``score_return_pairs`` sorted descending by score first.

    Purely descriptive analytics about the ranking's quality -- never
    implies capital is deployed long the top decile / short the bottom
    one. Returns ``None`` if there are fewer than 10 pairs (deciles would
    be degenerate).
    """
    n = len(score_return_pairs)
    if n < 10:
        return None
    ordered = sorted(score_return_pairs, key=lambda pair: -pair[0])
    decile_size = n // 10
    top = ordered[:decile_size]
    bottom = ordered[-decile_size:]
    top_mean = statistics.mean(r for _, r in top)
    bottom_mean = statistics.mean(r for _, r in bottom)
    return top_mean - bottom_mean


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """ROC-AUC via the Mann-Whitney U / rank-sum formula: the probability
    that a random positive-labeled (``1``) instance is scored higher than
    a random negative-labeled (``0``) instance (a tie counts as 0.5).

    Unlike IC (which measures rank-correlation against the *continuous*
    forward return), this measures discrimination against the same
    *binary* win/loss label the model is trained on (``labeling.py``'s
    top-``n_winners`` global label) -- a direct read of how well the
    model's own training objective is being achieved, not a proxy for it.

    Returns ``None`` (never a fabricated 0.5) when ``labels`` contains no
    positives or no negatives -- AUC is undefined, not "average," when
    one class is entirely absent, exactly as :func:`spearman_correlation`
    returns ``None`` rather than 0.0 for a degenerate input.
    """
    n = len(labels)
    if n != len(scores):
        raise ValueError(f"labels and scores must be the same length, got {n} and {len(scores)}")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("roc_auc requires binary labels (0 or 1)")

    n_pos = sum(1 for label in labels if label == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = _ranks(list(scores))
    positive_rank_sum = sum(rank for label, rank in zip(labels, ranks) if label == 1)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _label_prior_cycle(
    cycle: EvaluationCycle,
    next_cycle: EvaluationCycle,
    observations,
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]],
    n_winners: int,
) -> tuple[LabeledObservation, ...]:
    exit_cutoff = datetime.combine(next_cycle.quarter_start, time.min)
    outcomes = [
        build_forward_return_outcome(
            obs.instrument_id, cycle.quarter_start, obs.feature_timestamp,
            next_cycle.quarter_start, prices_by_instrument.get(obs.instrument_id, ()), exit_cutoff,
        )
        for obs in observations
    ]
    labeling = assign_quarterly_labels(outcomes, cycle.quarter_start, n_winners=n_winners)
    label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
    return tuple(
        LabeledObservation(obs, label_by_id[obs.instrument_id], exit_cutoff)
        for obs in observations
        if obs.instrument_id in label_by_id
    )


def build_feature_results(
    *,
    config: MultiFactorRankingMLConfig,
    universe: Sequence[InstrumentId],
    fundamentals_by_instrument: Mapping[InstrumentId, Sequence[FundamentalsFeatureRecord]],
    sector_encoder: SectorEncoder,
    cycles: Sequence[EvaluationCycle],
    macro_lookup: MacroSeriesLookup | None = None,
) -> dict[date, FeaturePipelineResult]:
    """Build every cycle's :class:`FeaturePipelineResult`, keyed by ``quarter_start``.

    Independent of ``config.ml_train_years``/``config.n_winners`` and of
    which estimator will be used -- feature construction only depends on
    each cycle's own point-in-time cutoff. Callers sweeping several
    ``ml_train_years`` values (or several estimator configs) over the
    *same* universe/cycles/data should build this once and pass it to
    every :func:`run_ic_backtest` call via its ``feature_results``
    parameter, rather than paying this cost again per sweep point --
    this is real, expensive, per-instrument point-in-time selection work
    across the whole universe, not a cheap lookup.
    """
    feature_results: dict[date, FeaturePipelineResult] = {}
    for cycle in cycles:
        feature_results[cycle.quarter_start] = run_feature_pipeline(
            config=config, sector_encoder=sector_encoder, universe=universe,
            quarter_start=cycle.quarter_start, cutoff=cycle.cutoff,
            fundamentals_by_instrument=fundamentals_by_instrument, macro_lookup=macro_lookup,
        )
    return feature_results


def build_labeled_quarters(
    *,
    cycles: Sequence[EvaluationCycle],
    feature_results: Mapping[date, FeaturePipelineResult],
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]],
    n_winners: int,
) -> dict[date, tuple[LabeledObservation, ...]]:
    """Every cycle-but-the-last's realized top-``n_winners`` label, keyed by ``quarter_start``.

    Independent of ``config.ml_train_years`` -- a cycle's label (whether
    it was one of that quarter's top ``n_winners`` by realized forward
    return) is a pure function of that cycle, its next cycle, and prices,
    never of which later cycle is currently training or how far back its
    training window reaches. Callers sweeping several ``ml_train_years``
    values over the same ``feature_results``/prices/``n_winners`` should
    build this once and pass it to every :func:`run_ic_backtest` call via
    its ``labeled_by_quarter`` parameter.
    """
    return {
        cycles[i].quarter_start: _label_prior_cycle(
            cycles[i], cycles[i + 1], feature_results[cycles[i].quarter_start].observations,
            prices_by_instrument, n_winners,
        )
        for i in range(len(cycles) - 1)
    }


def run_ic_backtest(
    *,
    config: MultiFactorRankingMLConfig,
    universe: Sequence[InstrumentId],
    fundamentals_by_instrument: Mapping[InstrumentId, Sequence[FundamentalsFeatureRecord]],
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]],
    sector_encoder: SectorEncoder,
    cycles: Sequence[EvaluationCycle],
    estimator_factory: Callable[..., tuple[Estimator, EstimatorBuildInfo]],
    macro_lookup: MacroSeriesLookup | None = None,
    min_scored_count: int = 30,
    feature_results: dict[date, FeaturePipelineResult] | None = None,
    labeled_by_quarter: Mapping[date, tuple[LabeledObservation, ...]] | None = None,
) -> ICBacktestResult:
    """Run the full IC backtest across ``cycles`` (ascending chronological order).

    The last cycle in ``cycles`` never produces a measured IC (there is
    no next cycle to realize a forward return against) -- it still
    appears in ``cycle_results`` with ``ic=None``, so a caller always sees
    every requested cycle, not a silently shorter list.

    ``feature_results``/``labeled_by_quarter`` are computed internally
    (via :func:`build_feature_results`/:func:`build_labeled_quarters`)
    when omitted -- the default, and what every existing caller gets.
    Pass them explicitly (pre-built once, e.g. by a training-window
    sweep script trying several ``config.ml_train_years`` values) to
    skip rebuilding this ``ml_train_years``-independent work on every
    call; the caller is responsible for having built them with a
    matching ``n_winners``/universe/cycles, since this function has no
    way to detect a mismatch.
    """
    if len(cycles) < 2:
        raise ValueError("run_ic_backtest requires at least 2 cycles (the last has no next cycle to score against)")

    audit = AuditTrail()
    if feature_results is None:
        feature_results = build_feature_results(
            config=config, universe=universe, fundamentals_by_instrument=fundamentals_by_instrument,
            sector_encoder=sector_encoder, cycles=cycles, macro_lookup=macro_lookup,
        )
    if labeled_by_quarter is None:
        labeled_by_quarter = build_labeled_quarters(
            cycles=cycles, feature_results=feature_results,
            prices_by_instrument=prices_by_instrument, n_winners=config.n_winners,
        )
    cycle_results: list[RankingCycleResult] = []

    for i, cycle in enumerate(cycles):
        training_cutoff = datetime.combine(cycle.cutoff, time.min)
        # Safe to hand the whole precomputed dict to every cycle here (even
        # one built for the entire range, or one containing a quarter at/
        # after this cycle's own target) -- build_training_dataset itself
        # excludes any quarter with quarter_end >= target_quarter_end, so a
        # cycle can never see its own or a later quarter's label regardless
        # of what this dict happens to contain.
        dataset: TrainingDatasetResult = build_training_dataset(
            cycle.quarter_start, training_cutoff, labeled_by_quarter,
            strategy_id=config.strategy_id, feature_schema_version=FEATURE_SCHEMA_VERSION,
            ml_train_years=config.ml_train_years, model_config_identity=config.model.identity(),
        )
        eligibility = check_training_eligibility(
            dataset, min_train_quarters=config.min_train_quarters, n_winners=config.n_winners
        )
        training_result = train_model(
            dataset, eligibility, config.model, estimator_factory,
            strategy_id=config.strategy_id, strategy_version=STRATEGY_VERSION,
        )

        if training_result.state != TrainingState.TRAINED:
            cycle_results.append(
                RankingCycleResult(
                    cycle=cycle, training_state=training_result.state, model_identity=None,
                    scoring_result=None, decision_summary=None, ic=None, decile_spread=None, auc=None,
                    ranked_count=0, scored_for_ic_count=0, scored_for_auc_count=0,
                    warnings=(f"training skipped: {training_result.state.value}",),
                )
            )
            continue

        current_observations = feature_results[cycle.quarter_start].observations
        scoring_result = score_observations(
            training_result.fitted_estimator, training_result.model_identity, current_observations,
            training_cutoff, config_identity=config.identity(),
        )

        eval_context = StrategyEvaluationContext(
            strategy_id=config.strategy_id, evaluation_timestamp=training_cutoff, data_cutoff=training_cutoff,
            capital_budget_pct=0.0,
            strategy_config=MultiFactorRankingEvaluationInputs(
                config=config, scored_candidates=scoring_result.scored_candidates,
            ),
        )
        strategy_result = MultiFactorRankingMLStrategy().evaluate(eval_context)
        summary: MultiFactorRankingDecisionSummary = strategy_result.state_update

        ic, spread, scored_for_ic = None, None, 0
        auc, scored_for_auc = None, 0
        if i + 1 < len(cycles):
            next_cycle = cycles[i + 1]
            exit_cutoff = datetime.combine(next_cycle.quarter_start, time.min)
            outcomes: list[ForwardReturnOutcome] = []
            score_by_instrument: dict[InstrumentId, float] = {}
            pairs: list[tuple[float, float]] = []
            for ranked in summary.ranked_candidates:
                outcome: ForwardReturnOutcome = build_forward_return_outcome(
                    ranked.instrument_id, cycle.quarter_start, cycle.quarter_start,
                    next_cycle.quarter_start, prices_by_instrument.get(ranked.instrument_id, ()), exit_cutoff,
                )
                outcomes.append(outcome)
                score_by_instrument[ranked.instrument_id] = ranked.score
                if outcome.clipped_return is not None:
                    pairs.append((ranked.score, outcome.clipped_return))
            scored_for_ic = len(pairs)
            if pairs:
                ic = spearman_correlation([p[0] for p in pairs], [p[1] for p in pairs])
                spread = decile_spread(pairs)

            # Same binary win/loss label the model is trained on
            # (labeling.py's global top-n_winners rule), applied to this
            # cycle's own realized outcomes -- AUC measures how well the
            # model's own training objective is being achieved, a direct
            # complement to IC's continuous-return rank-correlation.
            #
            # Unlike training (where an uncomputable return is deliberately
            # labeled 0 -- see labeling.py), this evaluation excludes those
            # instruments entirely rather than counting them as confirmed
            # negatives: we have no real evidence they underperformed, only
            # that we don't know what they did, and treating "unknown" as
            # "lost" would bias AUC. This mirrors IC's own `pairs` filter
            # above, which excludes the same instruments for the same reason.
            labeling = assign_quarterly_labels(outcomes, cycle.quarter_start, n_winners=config.n_winners)
            label_pairs = [
                (assignment.label, score_by_instrument[assignment.instrument_id])
                for assignment in labeling.assignments
                if assignment.clipped_return is not None and assignment.instrument_id in score_by_instrument
            ]
            scored_for_auc = len(label_pairs)
            if label_pairs:
                auc = roc_auc([p[0] for p in label_pairs], [p[1] for p in label_pairs])

        cycle_results.append(
            RankingCycleResult(
                cycle=cycle, training_state=training_result.state, model_identity=training_result.model_identity,
                scoring_result=scoring_result, decision_summary=summary, ic=ic, decile_spread=spread, auc=auc,
                ranked_count=len(summary.ranked_candidates), scored_for_ic_count=scored_for_ic,
                scored_for_auc_count=scored_for_auc,
            )
        )

    audit = audit.append(
        AuditRecord(
            stage="ic_backtest", message=f"{len(cycle_results)} cycle(s) processed",
            timestamp=datetime.combine(cycles[-1].quarter_start, time.min),
        )
    )

    run_identity = compute_config_identity(
        {"config_identity": config.identity(), "cycles": [c.quarter_start.isoformat() for c in cycles]}
    )
    return ICBacktestResult(
        strategy_id=config.strategy_id, strategy_version=STRATEGY_VERSION, config_identity=config.identity(),
        cycle_results=tuple(cycle_results), run_identity=run_identity, audit_trail=audit,
        min_scored_count=min_scored_count,
    )
