"""The standalone Ranked Multi-Factor Rotation historical backtest runner,
spec §§6-8.

Orchestrates the point-in-time monthly selection pipeline (already built
and tested in ``atlas_quant.strategies.ranked_multi_factor_rotation``)
across a sequence of months, resolving each recommendation into a real
position return via the same generic, asset-class-agnostic accounting
primitives ``filing_momentum_runner.py`` uses
(``atlas_quant.backtest.accounting``/``price_resolution``) -- neither of
those modules is Filing-Momentum-specific, so nothing here reimplements
position resolution or price staleness handling.

Unlike ``filing_momentum_runner.py``, this is a single-strategy backtest
with no model training step: each period is just
"evaluate -> resolve positions -> charge turnover-based transaction
costs". ``strategy_budget_pct`` defaults to 1.0 (100% assigned capital),
same convention as Filing Momentum ML's runner.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Mapping, Sequence

import pandas as pd

from atlas_quant.backtest.accounting import (
    INSTRUMENT_RETURN_CAP,
    PositionOutcome,
    compute_period_return,
    resolve_position,
)
from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.config.identity import compute_config_identity
from atlas_quant.data.point_in_time import TradingCalendar
from atlas_quant.data.records import DailyOHLCObservation, DailyPriceObservation
from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.base import StrategyEvaluationContext, StrategyResult
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import RmfrBacktestPeriod
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    STRATEGY_ID,
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.strategy import (
    RankedMultiFactorRotationStrategy,
)


@dataclass(frozen=True, slots=True)
class TransactionCostPolicy:
    """Spec §8: transaction costs and slippage must be explicitly modeled
    -- the original paper excluded them, which the user identified as
    flattering its result and asked not to be repeated here. Default
    (10 bps, confirmed with the user 2026-08-04) is a single round-trip
    assumption applied per unit of one-way monthly turnover; not
    calibrated per-instrument liquidity (spec leaves that unspecified)."""

    commission_bps: float = 0.0
    slippage_bps: float = 10.0
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
class RmfrBacktestConfig:
    """Every behavior-changing backtest-level configuration value, bundled and identified."""

    strategy_config: RankedMultiFactorRotationConfig = field(
        default_factory=RankedMultiFactorRotationConfig
    )
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
class RmfrBacktestDependencies:
    """Every injectable dependency the runner needs -- a frozen container,
    no mutable globals.

    ``ohlc_price_frames`` feeds the selection pipeline (factor
    computation); ``close_price_source`` feeds position resolution
    (``atlas_quant.backtest.accounting.resolve_position``, which only
    needs close, not the full OHLC range). Both are built from the same
    underlying acquired data -- see
    ``build_dependencies_from_observations`` below.
    """

    ohlc_price_frames: Mapping[str, pd.DataFrame]
    close_price_source: Mapping[InstrumentId, tuple[DailyPriceObservation, ...]]
    trading_calendar: TradingCalendar


def build_dependencies_from_observations(
    observations: Sequence[DailyOHLCObservation],
) -> tuple[dict[str, pd.DataFrame], dict[InstrumentId, tuple[DailyPriceObservation, ...]]]:
    """Derive both the OHLC price-frame mapping and the close-only
    ``DailyPriceObservation`` mapping this runner needs from one flat
    sequence of acquired observations -- one real acquisition run, two
    consuming shapes, never two separate data pulls."""
    from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
        observations_to_price_frames,
    )

    price_frames = observations_to_price_frames(observations)

    by_instrument: dict[InstrumentId, list[DailyPriceObservation]] = defaultdict(list)
    for obs in observations:
        by_instrument[obs.instrument_id].append(
            DailyPriceObservation(
                instrument_id=obs.instrument_id,
                trading_date=obs.trading_date,
                close=obs.close,
                price_convention=obs.price_convention,
                provenance=obs.provenance,
            )
        )
    close_price_source = {
        iid: tuple(sorted(rows, key=lambda r: r.trading_date)) for iid, rows in by_instrument.items()
    }
    return price_frames, close_price_source


class RmfrOutcomeType(str, Enum):
    """One month's coarse capital-deployment bucket."""

    OK = "ok"
    """Primary rotation exposure -- at least one non-cash ETF held."""

    CASH = "cash"
    """All selected assets had negative momentum, spec §5 step 3 -- 100% SHY."""

    SKIPPED = "skipped"
    """Could not be evaluated (e.g. insufficient point-in-time history for
    a ranked ticker -- see spec §1's still-open late-inception question).
    Contributes a 0.0 return and does not update turnover state; never
    silently treated as a genuine flat month."""


@dataclass(frozen=True, slots=True)
class RmfrBacktestMonthResult:
    """One month's complete, structured backtest outcome."""

    period: RmfrBacktestPeriod
    outcome_type: RmfrOutcomeType
    strategy_result: StrategyResult | None
    positions: tuple[PositionOutcome, ...]
    gross_return: float | None
    turnover: float
    cost_drag: float
    net_return: float | None
    weights: dict[str, float]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "month_end": self.period.month_end.isoformat(),
            "outcome_type": self.outcome_type.value,
            "gross_return": self.gross_return,
            "turnover": self.turnover,
            "cost_drag": self.cost_drag,
            "net_return": self.net_return,
            "weights": self.weights,
            "position_count": len(self.positions),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class RmfrBacktestResult:
    """The complete, structured top-level result of one standalone backtest run."""

    strategy_id: str
    strategy_version: str
    config_identity: str
    backtest_start: date
    backtest_end: date
    month_results: tuple[RmfrBacktestMonthResult, ...]
    run_identity: str
    audit_trail: AuditTrail = field(default_factory=AuditTrail)

    @property
    def completed_month_count(self) -> int:
        return sum(1 for m in self.month_results if m.outcome_type != RmfrOutcomeType.SKIPPED)

    @property
    def skipped_month_count(self) -> int:
        return sum(1 for m in self.month_results if m.outcome_type == RmfrOutcomeType.SKIPPED)

    @property
    def cash_month_count(self) -> int:
        return sum(1 for m in self.month_results if m.outcome_type == RmfrOutcomeType.CASH)

    def total_return(self) -> float:
        growth = 1.0
        for m in self.month_results:
            if m.net_return is not None:
                growth *= 1 + m.net_return
        return growth - 1.0

    def equity_curve(self) -> tuple[tuple[date, float], ...]:
        growth = 1.0
        curve = []
        for m in self.month_results:
            if m.net_return is not None:
                growth *= 1 + m.net_return
            curve.append((m.period.month_end, growth))
        return tuple(curve)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "config_identity": self.config_identity,
            "backtest_start": self.backtest_start.isoformat(),
            "backtest_end": self.backtest_end.isoformat(),
            "run_identity": self.run_identity,
            "completed_month_count": self.completed_month_count,
            "skipped_month_count": self.skipped_month_count,
            "cash_month_count": self.cash_month_count,
            "total_return": self.total_return(),
            "month_results": [m.to_dict() for m in self.month_results],
            "audit_trail": self.audit_trail.to_dict(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([m.to_dict() for m in self.month_results])


def _compute_run_identity(
    periods: Sequence[RmfrBacktestPeriod], config: RmfrBacktestConfig,
) -> str:
    return compute_config_identity(
        {
            "strategy_id": STRATEGY_ID,
            "strategy_version": STRATEGY_VERSION,
            "config_identity": config.identity(),
            "period_identities": [p.identity() for p in periods],
        }
    )


def _turnover(previous_weights: Mapping[str, float], new_weights: Mapping[str, float]) -> float:
    """One-way turnover: half the sum of absolute weight changes, so
    rotating 20% out of one ETF into another counts as 20% turnover, not
    40%."""
    tickers = set(previous_weights) | set(new_weights)
    return 0.5 * sum(abs(new_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) for t in tickers)


def run_ranked_multi_factor_rotation_backtest(
    periods: Sequence[RmfrBacktestPeriod],
    dependencies: RmfrBacktestDependencies,
    config: RmfrBacktestConfig | None = None,
) -> RmfrBacktestResult:
    """Run the standalone Ranked Multi-Factor Rotation backtest over
    ``periods``, in order.

    For each period: evaluate the strategy (spec §§3-5's rank/select/
    allocate, already implemented and tested in ``strategy.py``/
    ``pipeline.py``), resolve each recommendation into a real position
    return, and charge turnover-based transaction costs (spec §8). A
    period that cannot be evaluated (most commonly: a ranked ticker
    without enough point-in-time history yet, e.g. before IGOV's real
    2009-01-30 inception plus its own lookback requirement) is recorded
    as :attr:`RmfrOutcomeType.SKIPPED`, not silently treated as a flat
    month, and never aborts the rest of the run.
    """
    config = config or RmfrBacktestConfig()
    strategy = RankedMultiFactorRotationStrategy()

    month_results: list[RmfrBacktestMonthResult] = []
    run_audit = AuditTrail()
    previous_weights: dict[str, float] = {}

    for period in periods:
        context = StrategyEvaluationContext(
            strategy_id=config.strategy_config.strategy_id,
            evaluation_timestamp=period.evaluation_timestamp,
            data_cutoff=period.data_cutoff,
            capital_budget_pct=config.strategy_budget_pct,
            data_providers={"daily_ohlc": dependencies.ohlc_price_frames},
            strategy_config=config.strategy_config,
        )

        try:
            strategy_result = strategy.evaluate(context)
        except (ValueError, KeyError) as exc:
            month_results.append(
                RmfrBacktestMonthResult(
                    period=period, outcome_type=RmfrOutcomeType.SKIPPED, strategy_result=None,
                    positions=(), gross_return=None, turnover=0.0, cost_drag=0.0, net_return=0.0,
                    weights=dict(previous_weights), warnings=(f"evaluation failed: {exc}",),
                )
            )
            continue

        if strategy_result.status == StrategyStatus.MISSING_DATA:
            month_results.append(
                RmfrBacktestMonthResult(
                    period=period, outcome_type=RmfrOutcomeType.SKIPPED,
                    strategy_result=strategy_result, positions=(), gross_return=None,
                    turnover=0.0, cost_drag=0.0, net_return=0.0, weights=dict(previous_weights),
                    warnings=strategy_result.missing_data,
                )
            )
            continue

        positions = tuple(
            resolve_position(
                rec, dependencies.close_price_source.get(rec.instrument_id, ()),
                period.entry_timestamp.date(), period.exit_timestamp.date(),
                config.price_policy, period.exit_timestamp, dependencies.trading_calendar,
                return_cap=config.instrument_return_cap,
            )
            for rec in strategy_result.recommendations
        )
        cash_weight = max(0.0, 1.0 - strategy_result.capital_requested_pct)
        gross_return = compute_period_return(positions, cash_weight)

        new_weights = {rec.instrument_id.symbol: rec.weight for rec in strategy_result.recommendations}
        turnover = _turnover(previous_weights, new_weights)
        cost_drag = turnover * config.transaction_costs.total_bps / 10000.0
        net_return = gross_return - cost_drag

        outcome_type = RmfrOutcomeType.CASH if strategy_result.status == StrategyStatus.CASH else RmfrOutcomeType.OK

        month_results.append(
            RmfrBacktestMonthResult(
                period=period, outcome_type=outcome_type, strategy_result=strategy_result,
                positions=positions, gross_return=gross_return, turnover=turnover,
                cost_drag=cost_drag, net_return=net_return, weights=new_weights,
                warnings=strategy_result.warnings,
            )
        )
        previous_weights = new_weights

    run_audit = run_audit.append(
        AuditRecord(
            stage="backtest_run", message=f"completed {len(month_results)} month(s)",
            timestamp=periods[-1].evaluation_timestamp if periods else pd.Timestamp.min.to_pydatetime(),
        )
    )
    return RmfrBacktestResult(
        strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION, config_identity=config.identity(),
        backtest_start=periods[0].month_end if periods else date.min,
        backtest_end=periods[-1].month_end if periods else date.min,
        month_results=tuple(month_results), run_identity=_compute_run_identity(periods, config),
        audit_trail=run_audit,
    )
