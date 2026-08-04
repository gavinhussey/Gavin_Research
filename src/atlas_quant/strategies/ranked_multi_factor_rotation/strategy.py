"""The Ranked Multi-Factor Rotation strategy decision evaluator, spec §§3-6.

Implements the ``Strategy`` protocol's ``evaluate()``. Wraps
``pipeline.select_for_month_end`` -- this module owns translating a
``MonthlySelectionResult`` into a ``StrategyResult``; it never computes a
factor or rank itself.

Expects ``context.data_providers["daily_ohlc"]`` to be a mapping of
ticker -> a point-in-time-correct ``open``/``high``/``low``/``close``
DataFrame (see ``pipeline.compute_factor_snapshot``'s docstring) for
every ticker in ``context.strategy_config.ranked_tickers``. This
evaluator never fetches, caches, or acquires price data itself.
"""

from __future__ import annotations

import pandas as pd

from atlas_quant.domain.audit import AuditRecord, AuditTrail
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.signal import InstrumentRecommendation
from atlas_quant.domain.status import SignalKind, StrategyStatus
from atlas_quant.strategies.base import StrategyEvaluationContext, StrategyResult
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    DISPLAY_NAME,
    STRATEGY_VERSION,
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import select_for_month_end


class RankedMultiFactorRotationStrategy:
    """Constructing this does no I/O and loads no data -- all real inputs
    arrive per-evaluation via ``StrategyEvaluationContext``."""

    def evaluate(self, context: StrategyEvaluationContext) -> StrategyResult:
        config = context.strategy_config
        if not isinstance(config, RankedMultiFactorRotationConfig):
            raise TypeError(
                "RankedMultiFactorRotationStrategy requires "
                "context.strategy_config to be a RankedMultiFactorRotationConfig, "
                f"got {type(config).__name__}"
            )

        prices = context.data_providers.get("daily_ohlc")
        if prices is None:
            return StrategyResult(
                strategy_id=config.strategy_id,
                display_name=DISPLAY_NAME,
                strategy_version=STRATEGY_VERSION,
                config_identity=config.identity(),
                model_identity=None,
                evaluation_timestamp=context.evaluation_timestamp,
                data_cutoff=context.data_cutoff,
                status=StrategyStatus.MISSING_DATA,
                missing_data=("daily_ohlc",),
            )

        as_of = pd.Timestamp(context.data_cutoff.date())
        selection = select_for_month_end(prices, as_of, config)

        recommendations = tuple(
            InstrumentRecommendation(
                instrument_id=InstrumentId(symbol=ticker, asset_class=AssetClass.ETF),
                kind=(
                    SignalKind.FALLBACK if ticker == config.cash_ticker else SignalKind.PRIMARY
                ),
                weight=weight * context.capital_budget_pct,
                score=(
                    float(selection.total_rank_scores.loc[ticker])
                    if ticker in selection.total_rank_scores.index
                    else None
                ),
            )
            for ticker, weight in sorted(selection.weights.items())
        )

        cash_only = set(selection.weights) == {config.cash_ticker}
        has_cash = config.cash_ticker in selection.weights
        if cash_only:
            status = StrategyStatus.CASH
        elif has_cash:
            status = StrategyStatus.FALLBACK
        else:
            status = StrategyStatus.OK

        audit_trail = AuditTrail().append(
            AuditRecord(
                stage="selection",
                message="monthly rank/select/allocate",
                timestamp=context.evaluation_timestamp,
                data={
                    "as_of": as_of.isoformat(),
                    "selected_tickers": selection.selected_tickers,
                    "weights": selection.weights,
                },
            )
        )

        return StrategyResult(
            strategy_id=config.strategy_id,
            display_name=DISPLAY_NAME,
            strategy_version=STRATEGY_VERSION,
            config_identity=config.identity(),
            model_identity=None,
            evaluation_timestamp=context.evaluation_timestamp,
            data_cutoff=context.data_cutoff,
            status=status,
            recommendations=recommendations,
            capital_requested_pct=context.capital_budget_pct,
            audit_trail=audit_trail,
        )
