#!/usr/bin/env python
"""filing_momentum_ml's rolling (expanding-window) walk-forward check for
`min_positions` (the fallback-trigger threshold, report §5.4).

Sibling to `single_split.py` in this same folder, which does a single
select-then-test split. This script instead re-selects the best candidate
every year using an expanding window of all real quarters strictly before
that year, then checks how that specific choice actually performed that
year against every other candidate (the best and worst possible picks in
hindsight for that year). Run this when the question is "does the chosen
minimum drift over time / is it regime-dependent", not just "does one
particular split hold up."

Reuses the existing, tested atlas_quant functions unchanged, in the same
order and with the same shared-training-across-candidates approach as
`single_split.py` (see that file's docstring for the full rationale).
No strategy formula is reimplemented, and no production config default is
changed by running this.

## Finding (2026-07-28 run, RESELECT_YEARS=2015..2025)

Every single expanding-window re-selection from 2015 through 2025 picked
min_positions=6 -- no drift observed. That is consistent with, and now
explains, `single_split.py`'s single-split finding: the deciding
factor both times was the same handful of quarters where the fallback ETF
sleeve actually triggers (as few as 0, as many as ~13 out of 40+ quarters
depending on the candidate and window), most consequentially the 2011-03-31
quarter documented in `single_split.py`'s docstring. Because an expanding window
only ever *adds* quarters, once a triggering quarter like that one is
included it stays included for every later re-selection year too, which is
the mechanical reason this rolling check doesn't show drift here -- it is
evidence of stability given this specific realized history, not proof the
threshold is regime-invariant in general. A meaningfully different result
would require either a longer real history with more independent
fallback-triggering tail events, or a *non-expanding* (fixed-length,
sliding) rolling window that can actually lose an old triggering quarter as
newer ones arrive; this script deliberately uses an expanding window
(more data is strictly better for training a fresh point-in-time model
each quarter) so it cannot observe that kind of drift by construction.

Edit the constants below, then run:

    .venv/bin/python walkforward/filing_momentum_ml/rolling.py
"""
from __future__ import annotations

import dataclasses
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_ROOT = REPO_ROOT / "data" / "raw" / "filing_momentum_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "filing_momentum_ml" / "data_manifest.json"

# --- edit these to change what gets tested ---
MIN_POSITIONS_CANDIDATES = list(range(2, 11))
TRAIN_BUFFER_YEARS = 3            # matches ml_train_years -- warm-up quarters, excluded from all analysis
EVAL_START = date(2011, 3, 31)    # earliest quarter ever eligible to be evaluated or re-selected on
EVAL_END = date(2025, 12, 31)
RESELECT_YEARS = list(range(2015, 2026))  # re-select at the start of each of these years
SELECTION_METRIC = "sharpe"       # one of: sharpe, sortino, avg_yearly_alpha
# ------------------------------------------


def _shift_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def main() -> int:
    from atlas_quant.backtest.accounting import compute_period_return, resolve_position
    from atlas_quant.backtest.benchmark import resolve_benchmark
    from atlas_quant.backtest.clock import generate_quarterly_periods
    from atlas_quant.backtest.filing_momentum_runner import (
        BacktestQuarterResult, BacktestResult, FilingMomentumBacktestConfig, QuarterOutcomeType,
    )
    from atlas_quant.cli.filing_momentum import _build_trading_calendar, _load_manifest, load_normalized_bundle
    from atlas_quant.domain.audit import AuditTrail
    from atlas_quant.domain.identifiers import AssetClass, InstrumentId
    from atlas_quant.strategies.base import StrategyEvaluationContext
    from atlas_quant.strategies.filing_momentum_ml.config import (
        FEATURE_SCHEMA_VERSION, STRATEGY_ID, STRATEGY_VERSION, FeatureCacheIdentity, FilingMomentumMLConfig,
    )
    from atlas_quant.strategies.filing_momentum_ml.estimator import build_hgbc_estimator
    from atlas_quant.strategies.filing_momentum_ml.forward_return import build_forward_return_outcome
    from atlas_quant.strategies.filing_momentum_ml.labeling import assign_quarterly_labels
    from atlas_quant.strategies.filing_momentum_ml.model_training import TrainingState, train_model
    from atlas_quant.strategies.filing_momentum_ml.performance_analysis import analyze_backtest_result
    from atlas_quant.strategies.filing_momentum_ml.performance_domain import PerformanceAnalysisConfig
    from atlas_quant.strategies.filing_momentum_ml.production.feature_label_build import build_production_features
    from atlas_quant.strategies.filing_momentum_ml.production.orchestration import build_fallback_statistics_source
    from atlas_quant.strategies.filing_momentum_ml.scoring import score_observations
    from atlas_quant.strategies.filing_momentum_ml.sector_encoding import SectorEncoder
    from atlas_quant.strategies.filing_momentum_ml.strategy import FilingMomentumEvaluationInputs, FilingMomentumMLStrategy
    from atlas_quant.strategies.filing_momentum_ml.training_dataset import (
        LabeledObservation, build_training_dataset, check_training_eligibility,
    )

    log("loading bundle/calendar/manifest")
    bundle = load_normalized_bundle(RAW_ROOT)
    calendar = _build_trading_calendar(bundle)
    manifest = _load_manifest(MANIFEST)
    benchmark_id = InstrumentId(symbol="SPY", asset_class=AssetClass.EQUITY)
    base_config = FilingMomentumMLConfig()
    sector_encoder = SectorEncoder()
    price_source = {k: tuple(v) for k, v in bundle.prices_by_instrument.items()}

    buffer_start = _shift_years(EVAL_START, TRAIN_BUFFER_YEARS)
    periods = generate_quarterly_periods(buffer_start, EVAL_END, earnings_lag_days=base_config.earnings_lag_days)
    eval_qends = {p.quarter_end for p in periods if EVAL_START <= p.quarter_end <= EVAL_END}
    log(f"buffer_start={buffer_start} eval=[{EVAL_START}, {EVAL_END}] ({len(eval_qends)}q) "
        f"total_periods={len(periods)}")

    backtest_config = FilingMomentumBacktestConfig(strategy_config=base_config)

    # --- shared, min_positions-independent feature build ---
    targets = [
        (instrument_id, period.quarter_end, period.entry_timestamp)
        for period in periods for instrument_id in bundle.universe
    ]
    cache_identity = FeatureCacheIdentity(
        strategy_id=base_config.strategy_id, strategy_version="production",
        feature_schema_version=FEATURE_SCHEMA_VERSION, fcf_mode=base_config.fcf_mode,
        train_years=base_config.ml_train_years, min_train_quarters=base_config.min_train_quarters,
        model_config_identity=base_config.identity(), universe_id=manifest.universe_identity,
        data_cutoff=max(p.quarter_end for p in periods), created_at=datetime.now(),
    )
    feature_build = build_production_features(
        config=base_config, calendar=calendar, sector_encoder=sector_encoder, targets=targets,
        filings_by_instrument=bundle.filings_by_instrument, prices_by_instrument=bundle.prices_by_instrument,
        sector_by_instrument=bundle.sector_by_instrument, cache_identity=cache_identity,
        cache_root=None, mode="training",
    )
    if feature_build.blocked:
        raise RuntimeError(f"feature build blocked: {feature_build.blocked_reason}")
    log(f"features built ({len(feature_build.feature_pipeline_result.observations)} obs)")

    observations_by_quarter: dict[date, list] = {}
    for obs in feature_build.feature_pipeline_result.observations:
        observations_by_quarter.setdefault(obs.strategy_cohort_end, []).append(obs)

    fallback_source = build_fallback_statistics_source(
        fallback_tickers=base_config.fallback_tickers, asset_class=benchmark_id.asset_class,
        prices_by_instrument=bundle.prices_by_instrument, ordered_periods=periods,
        lookback_quarters=base_config.fallback_lookback_quarters,
    )
    benchmark_prices = price_source.get(benchmark_id, ())

    # --- shared labeling (pure historical fact, independent of min_positions) ---
    labeled_quarters: dict[date, list] = {}
    for period in periods:
        observations = tuple(observations_by_quarter.get(period.quarter_end, ()))
        outcomes = [
            build_forward_return_outcome(
                obs.instrument_id, period.quarter_end, obs.feature_timestamp, period.exit_timestamp.date(),
                price_source.get(obs.instrument_id, ()), period.exit_timestamp,
            )
            for obs in observations
        ]
        labeling = assign_quarterly_labels(outcomes, period.quarter_end, n_winners=base_config.n_winners)
        label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
        labeled_quarters[period.quarter_end] = [
            LabeledObservation(obs, label_by_id[obs.instrument_id], period.label_availability_cutoff)
            for obs in observations
        ]

    variant_configs = {n: dataclasses.replace(base_config, min_positions=n) for n in MIN_POSITIONS_CANDIDATES}
    variant_carried_ratio: dict[int, float | None] = {n: None for n in MIN_POSITIONS_CANDIDATES}
    eval_quarters: dict[int, list] = {n: [] for n in MIN_POSITIONS_CANDIDATES}  # full real series, one per candidate

    strategy = FilingMomentumMLStrategy()
    n_periods = len(periods)
    for i, period in enumerate(periods):
        target_obs = tuple(observations_by_quarter.get(period.quarter_end, ()))
        dataset = build_training_dataset(
            period.quarter_end, period.training_cutoff, labeled_quarters,
            strategy_id=STRATEGY_ID, feature_schema_version=FEATURE_SCHEMA_VERSION,
            ml_train_years=base_config.ml_train_years, model_config_identity=base_config.model.identity(),
        )
        eligibility = check_training_eligibility(
            dataset, min_train_quarters=base_config.min_train_quarters, n_winners=base_config.n_winners,
        )
        training_result = train_model(
            dataset, eligibility, base_config.model, build_hgbc_estimator,
            strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION,
        )

        in_eval = period.quarter_end in eval_qends

        if training_result.state != TrainingState.TRAINED:
            if in_eval:
                for n in MIN_POSITIONS_CANDIDATES:
                    eval_quarters[n].append(BacktestQuarterResult(
                        period=period, outcome_type=QuarterOutcomeType.SKIPPED,
                        training_state=training_result.state, model_identity=None,
                        scoring_result=None, strategy_result=None, positions=(), period_return=None,
                        benchmark=None, benchmark_return=None, alpha=None, cash_weight=0.0,
                        warnings=(f"training skipped: {training_result.state.value}",),
                        rejection_reasons=tuple(eligibility.reasons), audit_trail=training_result.audit_trail,
                    ))
            log(f"  [{i+1}/{n_periods}] {period.quarter_end} SKIPPED ({training_result.state.value})")
            continue

        scoring_result = score_observations(
            training_result.fitted_estimator, training_result.model_identity, target_obs,
            period.evaluation_timestamp, config_identity=base_config.identity(),
        )
        fallback_stats = fallback_source(period)
        benchmark = resolve_benchmark(
            benchmark_id, benchmark_prices, period.entry_timestamp.date(), period.exit_timestamp.date(),
            backtest_config.price_policy, period.exit_timestamp, calendar,
        )
        benchmark_return = benchmark.raw_return

        for n in MIN_POSITIONS_CANDIDATES:
            inputs = FilingMomentumEvaluationInputs(
                config=variant_configs[n], scored_candidates=scoring_result.scored_candidates,
                fallback_statistics=fallback_stats, previous_reference_score_to_weight_ratio=variant_carried_ratio[n],
            )
            context = StrategyEvaluationContext(
                strategy_id=STRATEGY_ID, evaluation_timestamp=period.evaluation_timestamp,
                data_cutoff=period.evaluation_timestamp, capital_budget_pct=backtest_config.strategy_budget_pct,
                strategy_config=inputs,
            )
            strategy_result = strategy.evaluate(context)

            new_reference = getattr(strategy_result.state_update, "reference_score_to_weight_ratio", None)
            if new_reference is not None:
                variant_carried_ratio[n] = new_reference

            positions = tuple(
                resolve_position(
                    rec, price_source.get(rec.instrument_id, ()), period.entry_timestamp.date(),
                    period.exit_timestamp.date(), backtest_config.price_policy, period.exit_timestamp,
                    calendar, return_cap=backtest_config.instrument_return_cap,
                )
                for rec in strategy_result.recommendations
            )
            cash_weight = max(0.0, 1.0 - strategy_result.capital_requested_pct)
            period_return = compute_period_return(positions, cash_weight)
            alpha = (period_return - benchmark_return) if benchmark_return is not None else None

            if in_eval:
                status_val = strategy_result.status.value if strategy_result.status else None
                outcome_type = QuarterOutcomeType.PRIMARY if status_val == "ok" else (
                    QuarterOutcomeType.FALLBACK if status_val == "fallback" else QuarterOutcomeType.CASH
                )
                eval_quarters[n].append(BacktestQuarterResult(
                    period=period, outcome_type=outcome_type, training_state=training_result.state,
                    model_identity=training_result.model_identity, scoring_result=scoring_result,
                    strategy_result=strategy_result, positions=positions, period_return=period_return,
                    benchmark=benchmark, benchmark_return=benchmark_return, alpha=alpha,
                    cash_weight=cash_weight, warnings=strategy_result.warnings,
                    rejection_reasons=(), audit_trail=AuditTrail(),
                ))

        log(f"  [{i+1}/{n_periods}] {period.quarter_end} trained+scored")

    metric_key = {"sharpe": "sharpe", "sortino": "sortino", "avg_yearly_alpha": "avg_yearly_alpha"}[SELECTION_METRIC]

    def _metric(quarter_results) -> float | None:
        qr = tuple(quarter_results)
        if not qr:
            return None
        backtest_result = BacktestResult(
            strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION,
            config_identity="rolling_walk_forward_check", backtest_start=qr[0].period.quarter_end,
            backtest_end=qr[-1].period.quarter_end, quarter_results=qr, run_identity="rolling_walk_forward_check",
        )
        performance = analyze_backtest_result(backtest_result, PerformanceAnalysisConfig())
        if metric_key == "avg_yearly_alpha":
            years = performance.invested.annual_summaries
            return sum(y.alpha for y in years if y.alpha is not None) / len(years) if years else None
        metric = performance.invested.sharpe if metric_key == "sharpe" else performance.invested.sortino
        return metric.value if metric else None

    def _year_compounded(quarter_results) -> float | None:
        rets = [q.period_return for q in quarter_results if q.period_return is not None]
        if not rets:
            return None
        v = 1.0
        for r in rets:
            v *= (1.0 + r)
        return v - 1.0

    def _year_alpha(quarter_results) -> float | None:
        alphas = [q.alpha for q in quarter_results if q.alpha is not None]
        return sum(alphas) / len(alphas) if alphas else None

    log("=== ROLLING RE-SELECTION ===")
    print(f"{'year':>6} {'selected':>9} {'sel_return':>11} {'sel_alpha':>10} {'best_that_yr':>13} "
          f"{'best_return':>11} {'worst_that_yr':>14} {'worst_return':>12} {'regret_vs_best':>15}")
    selection_history = []
    for year in RESELECT_YEARS:
        train_by_n = {n: [q for q in eval_quarters[n] if q.period.quarter_end.year < year] for n in MIN_POSITIONS_CANDIDATES}
        if not any(train_by_n.values()):
            continue
        scores = {n: _metric(train_by_n[n]) for n in MIN_POSITIONS_CANDIDATES}
        ranked = sorted((n for n in MIN_POSITIONS_CANDIDATES if scores[n] is not None), key=lambda n: scores[n], reverse=True)
        if not ranked:
            continue
        selected = ranked[0]
        selection_history.append((year, selected))

        year_by_n = {n: [q for q in eval_quarters[n] if q.period.quarter_end.year == year] for n in MIN_POSITIONS_CANDIDATES}
        year_returns = {n: _year_compounded(year_by_n[n]) for n in MIN_POSITIONS_CANDIDATES}
        year_returns = {n: v for n, v in year_returns.items() if v is not None}
        if not year_returns:
            continue
        best_n = max(year_returns, key=year_returns.get)
        worst_n = min(year_returns, key=year_returns.get)
        sel_return = year_returns.get(selected)
        sel_alpha = _year_alpha(year_by_n[selected])
        regret = (year_returns[best_n] - sel_return) if sel_return is not None else None

        print(f"{year:>6} {selected:>9} {(f'{sel_return:+.2%}' if sel_return is not None else 'n/a'):>11} "
              f"{(f'{sel_alpha:+.2%}' if sel_alpha is not None else 'n/a'):>10} {best_n:>13} "
              f"{year_returns[best_n]:+.2%} {worst_n:>14} {year_returns[worst_n]:+.2%} "
              f"{(f'{regret:+.2%}' if regret is not None else 'n/a'):>15}")

    log("Selection drift over time (best candidate as of each expanding window):")
    drifted = len({sel for _, sel in selection_history}) > 1
    for year, selected in selection_history:
        print(f"  as of end of {year - 1}: would select min_positions={selected}")
    log(f"drift observed: {drifted} ({'the selected minimum changed at least once' if drifted else 'the same minimum was selected every year'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
