#!/usr/bin/env python
"""multi_factor_ranking_ml's single-split walk-forward robustness check.

One subfolder per strategy lives under `walkforward/` (this is
multi_factor_ranking_ml's); add a sibling subfolder for each new strategy's own
walk-forward checks rather than growing this one to cover more than one
strategy. Within a strategy's subfolder, one file per *kind* of walk-forward
check -- this one does a single select-then-test split; see `rolling.py`
alongside it for the expanding-window, year-by-year variant.

This was originally used to sweep `min_positions` candidates and pick the
best one by in-sample performance, then check whether that choice held up
out-of-sample. That question is settled: `min_positions=6` is the strategy's
final, adopted fallback-trigger threshold (report §5.4's default of 3,
deliberately overridden -- see
`research/strategies/multi_factor_ranking_ml/docs/reproducibility_findings.md`).
This script no longer searches for a value; it runs the single production
config through a standard walk-forward split -- train/select window, then a
later, disjoint, out-of-sample test window -- to check whether the
strategy's real performance holds up out-of-sample, not to pick a parameter.

Reuses the existing, tested atlas_quant functions unchanged (feature build,
training-eligibility check, model training, scoring, the strategy decision
evaluator, position/benchmark resolution, and performance analysis) in the
same order the production runner (`multi_factor_ranking_runner.run_multi_factor_ranking_backtest`)
uses them. No strategy formula is reimplemented, and no production config
default is changed by running this.

Edit the constants below, then run:

    .venv/bin/python walkforward/multi_factor_ranking_ml/single_split.py
"""
from __future__ import annotations

import bisect
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_ROOT = REPO_ROOT / "data" / "raw" / "multi_factor_ranking_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "multi_factor_ranking_ml" / "data_manifest.json"

# --- edit these to change what gets tested ---
TRAIN_BUFFER_YEARS = 3          # matches ml_train_years -- warm-up quarters, excluded from both windows
SELECTION_START = date(2011, 3, 31)   # in-sample window
SELECTION_END = date(2020, 12, 31)
TEST_START = date(2021, 3, 31)        # out-of-sample window, blind to selection
TEST_END = date(2025, 12, 31)
SELECTION_METRIC = "sharpe"           # one of: sharpe, sortino, avg_yearly_alpha
# ------------------------------------------


def _shift_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


def _price_lookup_index(prices_by_instrument: dict) -> dict:
    index = {}
    for instrument_id, obs in prices_by_instrument.items():
        pairs = sorted((o.trading_date, o.close) for o in obs)
        if pairs:
            index[instrument_id] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return index


def _last_price_on_or_before(index: dict, instrument_id, d: date):
    entry = index.get(instrument_id)
    if not entry:
        return None
    dates, prices = entry
    i = bisect.bisect_right(dates, d) - 1
    return prices[i] if i >= 0 else None


def compute_daily_drawdown(quarter_results, price_index: dict, benchmark_id) -> dict:
    """Real, uncapped daily peak-to-trough drawdown from actual daily closes
    and each quarter's resolved position weights/entry prices -- not the
    quarterly-equity-point drawdown the built-in performance analysis
    reports. See performance_metrics.compute_drawdown for that quarterly
    series; this is a finer-grained, real-price reconstruction on top of it,
    not a replacement or a re-derivation of the report's own return math.
    """
    nav = 1.0
    points: list[tuple[date, float]] = []
    ordered = sorted(
        (q for q in quarter_results if q.outcome_type.value != "skipped" and q.period is not None),
        key=lambda q: q.period.quarter_end,
    )
    for q in ordered:
        period = q.period
        day_start, day_end = period.entry_timestamp.date(), period.exit_timestamp.date()
        bdates, _ = price_index.get(benchmark_id, ([], []))
        lo, hi = bisect.bisect_left(bdates, day_start), bisect.bisect_right(bdates, day_end)
        grid = bdates[lo:hi]
        if not grid or grid[0] != day_start:
            grid = [day_start] + grid
        if grid[-1] != day_end:
            grid = grid + [day_end]

        entry_prices = {p.instrument_id: p.entry_resolved.price for p in q.positions if p.entry_resolved and p.entry_resolved.price}
        quarter_start_nav = nav
        for d in grid:
            day_value = q.cash_weight * 1.0
            for p in q.positions:
                ep = entry_prices.get(p.instrument_id)
                if not ep:
                    continue
                px = _last_price_on_or_before(price_index, p.instrument_id, d) or ep
                day_value += p.target_weight * (px / ep)
            nav = quarter_start_nav * day_value
            points.append((d, nav))

    if not points:
        return {"max_drawdown": None, "peak_date": None, "trough_date": None}
    peak, peak_date = points[0][1], points[0][0]
    max_dd, max_dd_peak, max_dd_trough = 0.0, peak_date, peak_date
    for d, v in points:
        if v > peak:
            peak, peak_date = v, d
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd, max_dd_peak, max_dd_trough = dd, peak_date, d
    return {"max_drawdown": max_dd, "peak_date": max_dd_peak, "trough_date": max_dd_trough}


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def main() -> int:
    from atlas_quant.backtest.accounting import compute_period_return, resolve_position
    from atlas_quant.backtest.benchmark import resolve_benchmark
    from atlas_quant.backtest.clock import generate_quarterly_periods
    from atlas_quant.backtest.multi_factor_ranking_runner import (
        BacktestQuarterResult, BacktestResult, MultiFactorRankingBacktestConfig, QuarterOutcomeType,
    )
    from atlas_quant.cli.multi_factor_ranking import _build_trading_calendar, _load_manifest, load_normalized_bundle
    from atlas_quant.domain.audit import AuditTrail
    from atlas_quant.domain.identifiers import AssetClass, InstrumentId
    from atlas_quant.strategies.base import StrategyEvaluationContext
    from atlas_quant.strategies.multi_factor_ranking_ml.config import (
        FEATURE_SCHEMA_VERSION, STRATEGY_ID, STRATEGY_VERSION, FeatureCacheIdentity, MultiFactorRankingMLConfig,
    )
    from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_hgbc_estimator
    from atlas_quant.strategies.multi_factor_ranking_ml.forward_return import build_forward_return_outcome
    from atlas_quant.strategies.multi_factor_ranking_ml.labeling import assign_quarterly_labels
    from atlas_quant.strategies.multi_factor_ranking_ml.model_training import TrainingState, train_model
    from atlas_quant.strategies.multi_factor_ranking_ml.performance_analysis import analyze_backtest_result
    from atlas_quant.strategies.multi_factor_ranking_ml.performance_domain import PerformanceAnalysisConfig
    from atlas_quant.strategies.multi_factor_ranking_ml.production.feature_label_build import build_production_features
    from atlas_quant.strategies.multi_factor_ranking_ml.production.orchestration import build_fallback_statistics_source
    from atlas_quant.strategies.multi_factor_ranking_ml.scoring import score_observations
    from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder
    from atlas_quant.strategies.multi_factor_ranking_ml.strategy import MultiFactorRankingEvaluationInputs, MultiFactorRankingMLStrategy
    from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import (
        LabeledObservation, build_training_dataset, check_training_eligibility,
    )

    log("loading bundle/calendar/manifest")
    bundle = load_normalized_bundle(RAW_ROOT)
    calendar = _build_trading_calendar(bundle)
    manifest = _load_manifest(MANIFEST)
    benchmark_id = InstrumentId(symbol="SPY", asset_class=AssetClass.EQUITY)
    config = MultiFactorRankingMLConfig()
    sector_encoder = SectorEncoder()
    price_source = {k: tuple(v) for k, v in bundle.prices_by_instrument.items()}
    price_index = _price_lookup_index(price_source)

    buffer_start = _shift_years(SELECTION_START, TRAIN_BUFFER_YEARS)
    periods = generate_quarterly_periods(buffer_start, TEST_END, earnings_lag_days=config.earnings_lag_days)
    selection_qends = {p.quarter_end for p in periods if SELECTION_START <= p.quarter_end <= SELECTION_END}
    test_qends = {p.quarter_end for p in periods if TEST_START <= p.quarter_end <= TEST_END}
    log(f"buffer_start={buffer_start} selection=[{SELECTION_START}, {SELECTION_END}] "
        f"({len(selection_qends)}q) test=[{TEST_START}, {TEST_END}] ({len(test_qends)}q) "
        f"total_periods={len(periods)} min_positions={config.min_positions}")

    backtest_config = MultiFactorRankingBacktestConfig(strategy_config=config)

    targets = [
        (instrument_id, period.quarter_end, period.entry_timestamp)
        for period in periods for instrument_id in bundle.universe
    ]
    cache_identity = FeatureCacheIdentity(
        strategy_id=config.strategy_id, strategy_version="production",
        feature_schema_version=FEATURE_SCHEMA_VERSION, fcf_mode=config.fcf_mode,
        train_years=config.ml_train_years, min_train_quarters=config.min_train_quarters,
        model_config_identity=config.identity(), universe_id=manifest.universe_identity,
        data_cutoff=max(p.quarter_end for p in periods), created_at=datetime.now(),
    )
    feature_build = build_production_features(
        config=config, calendar=calendar, sector_encoder=sector_encoder, targets=targets,
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
        fallback_tickers=config.fallback_tickers, asset_class=benchmark_id.asset_class,
        prices_by_instrument=bundle.prices_by_instrument, ordered_periods=periods,
        lookback_quarters=config.fallback_lookback_quarters,
    )
    benchmark_prices = price_source.get(benchmark_id, ())

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
        labeling = assign_quarterly_labels(outcomes, period.quarter_end, n_winners=config.n_winners)
        label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
        labeled_quarters[period.quarter_end] = [
            LabeledObservation(obs, label_by_id[obs.instrument_id], period.label_availability_cutoff)
            for obs in observations
        ]

    reference_ratio: float | None = None
    selection_quarters: list = []
    test_quarters: list = []

    strategy = MultiFactorRankingMLStrategy()
    n_periods = len(periods)
    for i, period in enumerate(periods):
        target_obs = tuple(observations_by_quarter.get(period.quarter_end, ()))
        dataset = build_training_dataset(
            period.quarter_end, period.training_cutoff, labeled_quarters,
            strategy_id=STRATEGY_ID, feature_schema_version=FEATURE_SCHEMA_VERSION,
            ml_train_years=config.ml_train_years, model_config_identity=config.model.identity(),
        )
        eligibility = check_training_eligibility(
            dataset, min_train_quarters=config.min_train_quarters, n_winners=config.n_winners,
        )
        training_result = train_model(
            dataset, eligibility, config.model, build_hgbc_estimator,
            strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION,
        )

        target_bucket = None
        if period.quarter_end in selection_qends:
            target_bucket = selection_quarters
        elif period.quarter_end in test_qends:
            target_bucket = test_quarters

        if training_result.state != TrainingState.TRAINED:
            if target_bucket is not None:
                target_bucket.append(BacktestQuarterResult(
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
            period.evaluation_timestamp, config_identity=config.identity(),
        )
        fallback_stats = fallback_source(period)
        benchmark = resolve_benchmark(
            benchmark_id, benchmark_prices, period.entry_timestamp.date(), period.exit_timestamp.date(),
            backtest_config.price_policy, period.exit_timestamp, calendar,
        )
        benchmark_return = benchmark.raw_return

        inputs = MultiFactorRankingEvaluationInputs(
            config=config, scored_candidates=scoring_result.scored_candidates,
            fallback_statistics=fallback_stats, previous_reference_score_to_weight_ratio=reference_ratio,
        )
        context = StrategyEvaluationContext(
            strategy_id=STRATEGY_ID, evaluation_timestamp=period.evaluation_timestamp,
            data_cutoff=period.evaluation_timestamp, capital_budget_pct=backtest_config.strategy_budget_pct,
            strategy_config=inputs,
        )
        strategy_result = strategy.evaluate(context)

        new_reference = getattr(strategy_result.state_update, "reference_score_to_weight_ratio", None)
        if new_reference is not None:
            reference_ratio = new_reference

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

        if target_bucket is not None:
            status_val = strategy_result.status.value if strategy_result.status else None
            outcome_type = QuarterOutcomeType.PRIMARY if status_val == "ok" else (
                QuarterOutcomeType.FALLBACK if status_val == "fallback" else QuarterOutcomeType.CASH
            )
            target_bucket.append(BacktestQuarterResult(
                period=period, outcome_type=outcome_type, training_state=training_result.state,
                model_identity=training_result.model_identity, scoring_result=scoring_result,
                strategy_result=strategy_result, positions=positions, period_return=period_return,
                benchmark=benchmark, benchmark_return=benchmark_return, alpha=alpha,
                cash_weight=cash_weight, warnings=strategy_result.warnings,
                rejection_reasons=(), audit_trail=AuditTrail(),
            ))

        log(f"  [{i+1}/{n_periods}] {period.quarter_end} trained+scored")

    def _summarize(quarter_results, window_start, window_end):
        qr = tuple(quarter_results)
        backtest_result = BacktestResult(
            strategy_id=STRATEGY_ID, strategy_version=STRATEGY_VERSION,
            config_identity="walk_forward_check", backtest_start=window_start, backtest_end=window_end,
            quarter_results=qr, run_identity="walk_forward_check",
        )
        performance = analyze_backtest_result(backtest_result, PerformanceAnalysisConfig())
        years = performance.invested.annual_summaries
        avg_alpha = sum(y.alpha for y in years if y.alpha is not None) / len(years) if years else None
        daily_dd = compute_daily_drawdown(qr, price_index, benchmark_id)
        return {
            "sharpe": performance.invested.sharpe.value if performance.invested.sharpe else None,
            "sortino": performance.invested.sortino.value if performance.invested.sortino else None,
            "avg_yearly_alpha": avg_alpha,
            "fallback_quarters": sum(1 for q in qr if q.outcome_type == QuarterOutcomeType.FALLBACK),
            "n_quarters": len(qr),
            "daily_max_drawdown": daily_dd["max_drawdown"],
        }

    def _print_summary(label: str, s: dict) -> None:
        print(f"  {label:<28} sharpe={s['sharpe']:.3f}  sortino={('%.3f' % s['sortino']) if s['sortino'] is not None else 'n/a':>6}  "
              f"avg_yr_alpha={s['avg_yearly_alpha']:+.2%}  fallback={s['fallback_quarters']}/{s['n_quarters']}  "
              f"daily_maxDD={s['daily_max_drawdown']:.2%}")

    log(f"=== WALK-FORWARD CHECK (min_positions={config.min_positions}) ===")
    selection_summary = _summarize(selection_quarters, SELECTION_START, SELECTION_END)
    test_summary = _summarize(test_quarters, TEST_START, TEST_END)
    _print_summary(f"in-sample [{SELECTION_START}, {SELECTION_END}]", selection_summary)
    _print_summary(f"out-of-sample [{TEST_START}, {TEST_END}]", test_summary)

    metric_key = {"sharpe": "sharpe", "sortino": "sortino", "avg_yearly_alpha": "avg_yearly_alpha"}[SELECTION_METRIC]
    in_sample_metric = selection_summary[metric_key]
    oos_metric = test_summary[metric_key]
    if in_sample_metric is not None and oos_metric is not None:
        held_up = oos_metric >= 0.5 * in_sample_metric if in_sample_metric > 0 else oos_metric >= in_sample_metric
        log(f"out-of-sample {SELECTION_METRIC}={oos_metric:.3f} vs in-sample {in_sample_metric:.3f} "
            f"({'held up' if held_up else 'degraded materially'})")
    else:
        log(f"out-of-sample/in-sample {SELECTION_METRIC} not comparable (missing data in one window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
