"""The standalone Filing Momentum ML historical backtest runner, Stage 7.

Orchestrates, in order, Stage 6 (labeling/training/scoring) and Stage 5
(the strategy decision) for one quarter at a time — never reimplementing
any of their formulas. This is a single-strategy backtest: ``strategy_budget_pct`` defaults to 1.0 (100%
assigned capital) so the report's standalone behavior is reproduced;
cross-strategy allocation is out of scope entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Callable, Mapping, Sequence

from atlas_quant.backtest.accounting import (
    INSTRUMENT_RETURN_CAP,
    PositionOutcome,
    compute_period_return,
    resolve_position,
)
from atlas_quant.backtest.benchmark import BenchmarkResult, resolve_benchmark
from atlas_quant.backtest.clock import BacktestPeriod
from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.config.identity import compute_config_identity
from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.serialization import to_jsonable
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.base import StrategyEvaluationContext, StrategyResult
from atlas_quant.strategies.filing_momentum_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_ID,
    STRATEGY_VERSION,
    FilingMomentumMLConfig,
)
from atlas_quant.strategies.filing_momentum_ml.estimator import Estimator, EstimatorBuildInfo
from atlas_quant.strategies.filing_momentum_ml.fallback_domain import FallbackAssetStatistics
from atlas_quant.strategies.filing_momentum_ml.feature_domain import FeatureObservation
from atlas_quant.strategies.filing_momentum_ml.forward_return import build_forward_return_outcome
from atlas_quant.strategies.filing_momentum_ml.labeling import assign_quarterly_labels
from atlas_quant.strategies.filing_momentum_ml.model_training import (
    ModelIdentity,
    TrainingState,
    train_model,
)
from atlas_quant.strategies.filing_momentum_ml.scoring import ScoringResult, score_observations
from atlas_quant.strategies.filing_momentum_ml.strategy import (
    FilingMomentumEvaluationInputs,
    FilingMomentumMLStrategy,
)
from atlas_quant.strategies.filing_momentum_ml.training_dataset import (
    LabeledObservation,
    build_training_dataset,
    check_training_eligibility,
)


@dataclass(frozen=True, slots=True)
class TransactionCostPolicy:
    """Report §9: transaction costs are not modeled. Represented explicitly,
    not left as an undocumented frictionless-trading assumption."""

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    other_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_bps", "slippage_bps", "other_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative, got {getattr(self, name)!r}")

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps + self.other_bps

    def identity(self) -> str:
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class FilingMomentumBacktestConfig:
    """Every behavior-changing backtest-level configuration value, bundled and identified."""

    strategy_config: FilingMomentumMLConfig = field(default_factory=FilingMomentumMLConfig)
    price_policy: PriceResolutionPolicy = field(default_factory=PriceResolutionPolicy)
    transaction_costs: TransactionCostPolicy = field(default_factory=TransactionCostPolicy)
    strategy_budget_pct: float = 1.0
    instrument_return_cap: float = INSTRUMENT_RETURN_CAP

    def __post_init__(self) -> None:
        if not (0.0 <= self.strategy_budget_pct <= 1.0):
            raise ValueError(
                f"strategy_budget_pct must be within [0.0, 1.0], got {self.strategy_budget_pct!r}"
            )

    def identity(self) -> str:
        return compute_config_identity(
            {
                "strategy_config_identity": self.strategy_config.identity(),
                "price_policy": self.price_policy,
                "transaction_costs": self.transaction_costs.identity(),
                "strategy_budget_pct": self.strategy_budget_pct,
                "instrument_return_cap": self.instrument_return_cap,
            }
        )


@dataclass(frozen=True, slots=True)
class FilingMomentumBacktestDependencies:
    """Every injectable dependency the runner needs — a frozen container, no mutable globals."""

    feature_observation_source: Callable[[date], Sequence[FeatureObservation]]
    price_source: Mapping[InstrumentId, tuple[DailyPriceObservation, ...]]
    fallback_statistics_source: Callable[[BacktestPeriod], tuple[FallbackAssetStatistics, ...]]
    estimator_factory: Callable[[FilingMomentumMLConfig], tuple[Estimator, EstimatorBuildInfo]]
    trading_calendar: TradingCalendar
    universe: tuple[InstrumentId, ...]
    benchmark_instrument_id: InstrumentId


class QuarterOutcomeType(str, Enum):
    """One quarter's coarse capital-deployment bucket.

    ``FALLBACK`` is the blended partial-fill bucket: fewer than
    ``min_positions`` stocks qualified, so the quarter holds whatever
    stocks did qualify *plus* an ETF sleeve over the deployable capital
    they left unused (``FilingMomentumOutcome.BLENDED`` /
    ``StrategyStatus.FALLBACK``). It is deliberately not merged into
    ``PRIMARY`` -- a quarter with ETF exposure must stay distinguishable
    from a pure stock-selection quarter in every downstream statistic.

    ``CASH`` is now an edge case only. With the regime gate removed and
    the strategy no longer able to produce a 100%-cash decision by
    design, it is reachable only via ``StrategyStatus.MISSING_DATA``
    (fallback-ticker statistics genuinely unavailable) or
    ``StrategyStatus.DISABLED`` -- never as a routine outcome.
    """

    PRIMARY = "primary"
    FALLBACK = "fallback"
    CASH = "cash"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class BacktestQuarterResult:
    """One quarter's complete, structured backtest outcome."""

    period: BacktestPeriod
    outcome_type: QuarterOutcomeType
    training_state: TrainingState | None
    model_identity: ModelIdentity | None
    scoring_result: ScoringResult | None
    strategy_result: StrategyResult | None
    positions: tuple[PositionOutcome, ...]
    period_return: float | None
    benchmark: BenchmarkResult | None
    benchmark_return: float | None
    alpha: float | None
    cash_weight: float
    warnings: tuple[str, ...] = field(default_factory=tuple)
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    def to_dict(self) -> dict[str, object]:
        return {
            "quarter_end": self.period.quarter_end.isoformat(),
            "evaluation_timestamp": self.period.evaluation_timestamp.isoformat(),
            "outcome_type": self.outcome_type.value,
            "training_state": self.training_state.value if self.training_state else None,
            "period_return": self.period_return,
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha,
            "cash_weight": self.cash_weight,
            "position_count": len(self.positions),
            "warnings": list(self.warnings),
            "rejection_reasons": list(self.rejection_reasons),
            "audit_trail": self.audit_trail.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The complete, structured top-level result of one standalone backtest run."""

    strategy_id: str
    strategy_version: str
    config_identity: str
    backtest_start: date
    backtest_end: date
    quarter_results: tuple[BacktestQuarterResult, ...]
    run_identity: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    @property
    def completed_quarter_count(self) -> int:
        return sum(1 for q in self.quarter_results if q.outcome_type != QuarterOutcomeType.SKIPPED)

    @property
    def skipped_quarter_count(self) -> int:
        return sum(1 for q in self.quarter_results if q.outcome_type == QuarterOutcomeType.SKIPPED)

    @property
    def cash_quarter_count(self) -> int:
        return sum(1 for q in self.quarter_results if q.outcome_type == QuarterOutcomeType.CASH)

    @property
    def fallback_quarter_count(self) -> int:
        """Quarters that ran as a blended partial fill (stocks + ETF sleeve)."""
        return sum(1 for q in self.quarter_results if q.outcome_type == QuarterOutcomeType.FALLBACK)

    @property
    def primary_quarter_count(self) -> int:
        return sum(1 for q in self.quarter_results if q.outcome_type == QuarterOutcomeType.PRIMARY)

    def total_return(self) -> float:
        growth = 1.0
        for q in self.quarter_results:
            if q.period_return is not None:
                growth *= 1 + q.period_return
        return growth - 1.0

    def benchmark_total_return(self) -> float:
        growth = 1.0
        for q in self.quarter_results:
            if q.benchmark_return is not None:
                growth *= 1 + q.benchmark_return
        return growth - 1.0

    def equity_curve(self) -> tuple[tuple[date, float, float], ...]:
        strategy_growth = 1.0
        benchmark_growth = 1.0
        curve = []
        for q in self.quarter_results:
            if q.period_return is not None:
                strategy_growth *= 1 + q.period_return
            if q.benchmark_return is not None:
                benchmark_growth *= 1 + q.benchmark_return
            curve.append((q.period.quarter_end, strategy_growth, benchmark_growth))
        return tuple(curve)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "config_identity": self.config_identity,
            "backtest_start": self.backtest_start.isoformat(),
            "backtest_end": self.backtest_end.isoformat(),
            "run_identity": self.run_identity,
            "completed_quarter_count": self.completed_quarter_count,
            "skipped_quarter_count": self.skipped_quarter_count,
            "cash_quarter_count": self.cash_quarter_count,
            "fallback_quarter_count": self.fallback_quarter_count,
            "primary_quarter_count": self.primary_quarter_count,
            "total_return": self.total_return(),
            "benchmark_total_return": self.benchmark_total_return(),
            "quarter_results": [q.to_dict() for q in self.quarter_results],
            "warnings": list(self.warnings),
            "audit_trail": self.audit_trail.to_dict(),
        }

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([q.to_dict() for q in self.quarter_results])


def _compute_run_identity(
    periods: Sequence[BacktestPeriod],
    config: FilingMomentumBacktestConfig,
    dependencies: FilingMomentumBacktestDependencies,
) -> str:
    return compute_config_identity(
        {
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "config_identity": config.identity(),
            "period_identities": [p.identity() for p in periods],
            "universe": sorted(str(i) for i in dependencies.universe),
            "benchmark_instrument": str(dependencies.benchmark_instrument_id),
        }
    )


def _build_labeled_quarters(
    periods: Sequence[BacktestPeriod],
    dependencies: FilingMomentumBacktestDependencies,
    n_winners: int,
) -> dict[date, list[LabeledObservation]]:
    """Compute every period's own forward-return outcomes and quarterly labels once,
    upfront -- pure historical fact, independent of which quarter later trains on it."""
    labeled: dict[date, list[LabeledObservation]] = {}
    for period in periods:
        observations = dependencies.feature_observation_source(period.quarter_end)
        outcomes = [
            build_forward_return_outcome(
                obs.instrument_id, period.quarter_end, obs.feature_timestamp,
                period.exit_timestamp.date(), dependencies.price_source.get(obs.instrument_id, ()),
                period.exit_timestamp,
            )
            for obs in observations
        ]
        labeling = assign_quarterly_labels(outcomes, period.quarter_end, n_winners=n_winners)
        label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
        labeled[period.quarter_end] = [
            LabeledObservation(obs, label_by_id[obs.instrument_id], period.label_availability_cutoff)
            for obs in observations
        ]
    return labeled


def run_filing_momentum_backtest(
    periods: Sequence[BacktestPeriod],
    dependencies: FilingMomentumBacktestDependencies,
    config: FilingMomentumBacktestConfig | None = None,
) -> BacktestResult:
    """Run the standalone Filing Momentum ML backtest over ``periods``, in order.

    For each period: build/score via Stage 6, decide via Stage 5, then
    resolve positions/benchmark and compute the period's return. A model
    is retrained from scratch for every period (never reused across
    quarters) — the per-period fitted estimator lives only inside that
    period's ``TrainingResult`` momentarily and is never carried into a
    later period's training call.

    Exactly one piece of state crosses quarter boundaries:
    ``carried_reference_ratio``, the ``deployable_pct / sum(scores)``
    ratio recorded by the most recent *full-quota* quarter, which a later
    partial-fill quarter sizes its few picks against (see
    ``strategy.FilingMomentumMLStrategy``). It is deliberately not
    overwritten by a partial-fill quarter's ``None``, so "most recent
    full-quota quarter" is preserved across any number of intervening
    thin quarters. Nothing else — no model, no score, no position — is
    carried forward.
    """
    config = config or FilingMomentumBacktestConfig()
    strategy_config = config.strategy_config
    labeled_quarters = _build_labeled_quarters(periods, dependencies, strategy_config.n_winners)
    strategy = FilingMomentumMLStrategy()

    quarter_results: list[BacktestQuarterResult] = []
    run_audit = AuditTrail()
    carried_reference_ratio: float | None = None

    for period in periods:
        target_obs = dependencies.feature_observation_source(period.quarter_end)

        dataset = build_training_dataset(
            period.quarter_end, period.training_cutoff, labeled_quarters,
            strategy_id=STRATEGY_ID, feature_schema_version=FEATURE_SCHEMA_VERSION,
            ml_train_years=strategy_config.ml_train_years,
            model_config_identity=strategy_config.model.identity(),
        )
        eligibility = check_training_eligibility(
            dataset, min_train_quarters=strategy_config.min_train_quarters,
            n_winners=strategy_config.n_winners,
        )
        training_result = train_model(
            dataset, eligibility, strategy_config.model, dependencies.estimator_factory,
            strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION,
        )

        if training_result.state != TrainingState.TRAINED:
            quarter_results.append(
                BacktestQuarterResult(
                    period=period, outcome_type=QuarterOutcomeType.SKIPPED,
                    training_state=training_result.state, model_identity=None,
                    scoring_result=None, strategy_result=None, positions=(), period_return=None,
                    benchmark=None, benchmark_return=None, alpha=None, cash_weight=0.0,
                    warnings=(f"training skipped: {training_result.state.value}",),
                    rejection_reasons=tuple(eligibility.reasons),
                    audit_trail=training_result.audit_trail,
                )
            )
            continue

        scoring_result = score_observations(
            training_result.fitted_estimator, training_result.model_identity, target_obs,
            period.evaluation_timestamp, config_identity=strategy_config.identity(),
        )

        benchmark_prices = dependencies.price_source.get(dependencies.benchmark_instrument_id, ())

        inputs = FilingMomentumEvaluationInputs(
            config=strategy_config, scored_candidates=scoring_result.scored_candidates,
            fallback_statistics=dependencies.fallback_statistics_source(period),
            previous_reference_score_to_weight_ratio=carried_reference_ratio,
        )
        context = StrategyEvaluationContext(
            strategy_id=STRATEGY_ID, evaluation_timestamp=period.evaluation_timestamp,
            data_cutoff=period.evaluation_timestamp, capital_budget_pct=config.strategy_budget_pct,
            strategy_config=inputs,
        )
        strategy_result = strategy.evaluate(context)

        # Only a full-quota quarter produces a new reference ratio; every
        # other outcome leaves it None, which must NOT clear the carried
        # value -- otherwise a single thin quarter would erase the
        # conviction level the next thin quarter needs.
        new_reference = getattr(strategy_result.state_update, "reference_score_to_weight_ratio", None)
        if new_reference is not None:
            carried_reference_ratio = new_reference

        positions = tuple(
            resolve_position(
                rec, dependencies.price_source.get(rec.instrument_id, ()),
                period.entry_timestamp.date(), period.exit_timestamp.date(),
                config.price_policy, period.exit_timestamp, dependencies.trading_calendar,
                return_cap=config.instrument_return_cap,
            )
            for rec in strategy_result.recommendations
        )
        cash_weight = max(0.0, 1.0 - strategy_result.capital_requested_pct)
        # positions is empty for any status with no recommendations (cash-like
        # outcomes), in which case this naturally reduces to cash_weight * 0.0.
        period_return = compute_period_return(positions, cash_weight)

        benchmark = resolve_benchmark(
            dependencies.benchmark_instrument_id, benchmark_prices,
            period.entry_timestamp.date(), period.exit_timestamp.date(),
            config.price_policy, period.exit_timestamp, dependencies.trading_calendar,
        )
        benchmark_return = benchmark.raw_return
        alpha = (period_return - benchmark_return) if benchmark_return is not None else None

        if strategy_result.status == StrategyStatus.OK:
            outcome_type = QuarterOutcomeType.PRIMARY
        elif strategy_result.status == StrategyStatus.FALLBACK:
            outcome_type = QuarterOutcomeType.FALLBACK
        else:
            outcome_type = QuarterOutcomeType.CASH
            period_return = 0.0
            cash_weight = 1.0

        quarter_results.append(
            BacktestQuarterResult(
                period=period, outcome_type=outcome_type, training_state=training_result.state,
                model_identity=training_result.model_identity, scoring_result=scoring_result,
                strategy_result=strategy_result, positions=positions, period_return=period_return,
                benchmark=benchmark, benchmark_return=benchmark_return, alpha=alpha,
                cash_weight=cash_weight, warnings=strategy_result.warnings,
                rejection_reasons=strategy_result.rejection_reasons,
                audit_trail=strategy_result.audit_trail,
            )
        )

    run_audit = run_audit.append(
        AuditRecord(
            stage="backtest_run", message=f"completed {len(quarter_results)} period(s)",
            timestamp=periods[-1].evaluation_timestamp if periods else datetime.min,
        )
    )
    return BacktestResult(
        strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION, config_identity=config.identity(),
        backtest_start=periods[0].quarter_end if periods else date.min,
        backtest_end=periods[-1].quarter_end if periods else date.min,
        quarter_results=tuple(quarter_results),
        run_identity=_compute_run_identity(periods, config, dependencies),
        warnings=(), audit_trail=run_audit,
    )
