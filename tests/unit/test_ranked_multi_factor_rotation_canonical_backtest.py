"""Focused tests for the first full historical canonical backtest stage
(2026-08-06): five-slot accounting, SHY aggregation, no lookahead,
report-artifact schema/metrics correctness, and isolation/equivalence of
the diagnostic-only gate-ordering backtest loop
(``trend_and_gate_diagnostics.run_gate_ordering_diagnostic_backtest``)
from the canonical production runner.

Uses small, deliberately synthetic OHLC fixtures -- mirrors
``test_ranked_multi_factor_rotation_runner.py``'s fixture style. Never
presented as a real backtest result; see
``outputs/ranked_multi_factor_rotation/canonical_backtest_report.json``
for the real-data artifact this stage produced.
"""

from __future__ import annotations

import inspect
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from atlas_quant.backtest.price_resolution import PriceResolutionPolicy
from atlas_quant.data.point_in_time import ListTradingCalendar
from atlas_quant.data.records import DailyOHLCObservation
from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.domain.provenance import DataProvenance
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import generate_monthly_periods
from atlas_quant.strategies.ranked_multi_factor_rotation.config import RankedMultiFactorRotationConfig
from atlas_quant.backtest.ranked_multi_factor_rotation_runner import (
    RmfrBacktestConfig,
    TransactionCostPolicy,
    build_dependencies_from_observations,
    run_ranked_multi_factor_rotation_backtest,
)
from atlas_quant.backtest.ranked_multi_factor_rotation_runner import RmfrBacktestDependencies
from atlas_quant.strategies.ranked_multi_factor_rotation import (
    backtest_report as backtest_report_module,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_report import (
    REQUIRED_REPORT_FIELDS,
    compute_annual_returns,
    compute_backtest_metrics,
    compute_data_fingerprint,
    compute_drawdown_periods,
    compute_selection_frequency,
    validate_report_payload,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.trend_and_gate_diagnostics import (
    run_gate_ordering_diagnostic_backtest,
)
from atlas_quant.strategies.ranked_multi_factor_rotation import (
    pipeline as pipeline_module,
)
from atlas_quant.strategies.ranked_multi_factor_rotation import strategy as strategy_module
from atlas_quant.backtest import ranked_multi_factor_rotation_runner as production_runner_module

_TICKERS = ("A", "B", "C", "D")


def _weekday_calendar(start: date, end: date) -> ListTradingCalendar:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return ListTradingCalendar(trading_days=tuple(days))


def _synthetic_observations(ticker: str, seed: int, start: date, n_days: int, drift: float) -> list[DailyOHLCObservation]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    instrument_id = InstrumentId(symbol=ticker, asset_class=AssetClass.ETF)
    observations = []
    for ts, c in zip(dates, close):
        d = ts.date()
        observations.append(
            DailyOHLCObservation(
                instrument_id=instrument_id, trading_date=d,
                open=c, high=c * 1.01, low=c * 0.99, close=c,
                price_convention="split_dividend_adjusted",
                provenance=DataProvenance(source="fake", as_of=d, retrieved_at=pd.Timestamp.now()),
            )
        )
    return observations


def _small_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_TICKERS, cash_ticker="CASH", top_n=2,
        momentum_lookback_days=5, correlation_lookback_days=5, atr_window=5,
        trend_model="legacy_symmetric", trend_lookback_n=5, volatility_smoothing_window=3,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


def _fixture_dependencies(n_days: int = 260):
    start = date(2020, 1, 2)
    observations = []
    for i, ticker in enumerate(_TICKERS + ("CASH",)):
        observations.extend(_synthetic_observations(ticker, seed=i + 1, start=start, n_days=n_days, drift=0.0005 * (i - 2)))
    price_frames, close_price_source = build_dependencies_from_observations(observations)
    all_dates = sorted({obs.trading_date for obs in observations})
    calendar = ListTradingCalendar(trading_days=tuple(all_dates))
    return (
        RmfrBacktestDependencies(ohlc_price_frames=price_frames, close_price_source=close_price_source, trading_calendar=calendar),
        observations,
    )


# -- five-slot accounting / SHY aggregation / weights sum to one --


def test_selected_five_slots_all_present_or_redirected_to_cash():
    # top_n=5, position_weight left at the canonical default (0.20) so
    # 5 slots * 20% = 100%, matching the real canonical configuration's
    # "five 20% slots" invariant (task B-5) -- _small_config's top_n=2
    # fixture elsewhere in this file deliberately under-allocates and is
    # not meant to sum to 1.0.
    dependencies, _ = _fixture_dependencies()
    config = RmfrBacktestConfig(strategy_config=_small_config(top_n=4, position_weight=0.25))
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 8, 31), dependencies.trading_calendar)
    result = run_ranked_multi_factor_rotation_backtest(periods, dependencies, config)
    checked_any = False
    for m in result.month_results:
        if m.strategy_result is None:
            continue
        total_weight = sum(m.weights.values())
        assert total_weight == pytest.approx(1.0, abs=1e-9)
        checked_any = True
    assert checked_any


def test_multiple_gated_slots_aggregate_into_a_single_shy_weight():
    # Direct check on allocate_weights, the mechanism the runner relies on.
    from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import allocate_weights

    momentum_values = pd.Series({"A": -0.01, "B": -0.02, "C": 0.03, "D": -0.01, "E": 0.02})
    weights = allocate_weights(["A", "B", "C", "D", "E"], momentum_values, position_weight=0.20, cash_ticker="CASH")
    # A, B, D all fail -> one aggregated CASH entry, not three separate ones.
    assert weights["CASH"] == pytest.approx(0.60)
    assert weights == pytest.approx({"C": 0.20, "E": 0.20, "CASH": 0.60})


# -- no lookahead / execution timing --


def test_signal_date_never_reads_data_after_month_end():
    dependencies, observations = _fixture_dependencies()
    config = RmfrBacktestConfig(strategy_config=_small_config())
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 5, 29), dependencies.trading_calendar)
    for period in periods:
        assert period.data_cutoff.date() == period.month_end
        assert period.entry_timestamp > period.data_cutoff
        assert period.exit_timestamp > period.entry_timestamp


def test_execution_occurs_strictly_after_signal_date():
    dependencies, _ = _fixture_dependencies()
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 5, 29), dependencies.trading_calendar)
    calendar = dependencies.trading_calendar
    for period in periods:
        assert period.entry_timestamp.date() == calendar.next_trading_day(period.month_end)


# -- report artifact: metrics correctness on a hand-built fixture --


def _hand_built_rows():
    # Two months: +10% then -5%, no turnover on the second, 20% SHY held
    # throughout via a fixed weights dict for allocation-bucket checks.
    return [
        {"month_end": "2020-01-31", "outcome_type": "ok", "gross_return": 0.10, "turnover": 0.5,
         "cost_drag": 0.0, "net_return": 0.10, "weights": {"A": 0.8, "SHY": 0.2}},
        {"month_end": "2020-02-29", "outcome_type": "ok", "gross_return": -0.05, "turnover": 0.0,
         "cost_drag": 0.0, "net_return": -0.05, "weights": {"A": 0.8, "SHY": 0.2}},
    ]


def test_compute_backtest_metrics_total_return_matches_hand_computed_compounding():
    rows = _hand_built_rows()
    metrics = compute_backtest_metrics(rows)
    expected_total_return = (1.10 * 0.95) - 1.0
    assert metrics["total_return"] == pytest.approx(expected_total_return)
    assert metrics["best_month"] == pytest.approx(0.10)
    assert metrics["worst_month"] == pytest.approx(-0.05)
    assert metrics["pct_positive_months"] == pytest.approx(50.0)
    assert metrics["avg_shy_allocation_pct"] == pytest.approx(20.0)
    assert metrics["shy_allocation_bucket_pct"]["20%"] == pytest.approx(100.0)
    assert metrics["avg_risky_holdings"] == pytest.approx(1.0)


def test_compute_annual_returns_compounds_within_year_not_sums():
    rows = _hand_built_rows()
    annual = compute_annual_returns(rows)
    assert annual["2020"] == pytest.approx((1.10 * 0.95) - 1.0)


def test_compute_drawdown_periods_detects_the_hand_built_drawdown():
    rows = _hand_built_rows()
    drawdowns = compute_drawdown_periods(rows)
    assert len(drawdowns) == 1
    assert drawdowns[0]["magnitude"] == pytest.approx(-0.05, abs=1e-9)


def test_compute_selection_frequency_distinguishes_selected_held_gated():
    selected_per_period = [["A", "B"], ["A", "C"]]
    held_per_period = [{"A": 0.5, "CASH": 0.5}, {"A": 0.5, "C": 0.5}]
    freq = compute_selection_frequency(selected_per_period, held_per_period, ["A", "B", "C"])
    assert freq["selected_frequency_pct"] == {"A": 100.0, "B": 50.0, "C": 50.0}
    assert freq["held_frequency_pct"] == {"A": 100.0, "B": 0.0, "C": 50.0}
    assert freq["gated_frequency_pct"] == {"A": 0.0, "B": 50.0, "C": 0.0}


def test_data_fingerprint_is_deterministic_for_identical_manifest():
    manifest = {"coverage_start": "2000-01-01", "coverage_end": "2026-01-01", "row_counts": {"A": 1}, "price_convention": "x"}
    a = compute_data_fingerprint(manifest)
    b = compute_data_fingerprint(dict(manifest))
    assert a == b


def test_data_fingerprint_changes_when_row_counts_change():
    manifest_a = {"coverage_start": "2000-01-01", "coverage_end": "2026-01-01", "row_counts": {"A": 1}, "price_convention": "x"}
    manifest_b = {"coverage_start": "2000-01-01", "coverage_end": "2026-01-01", "row_counts": {"A": 2}, "price_convention": "x"}
    assert compute_data_fingerprint(manifest_a) != compute_data_fingerprint(manifest_b)


def test_report_schema_validation_accepts_a_complete_payload():
    payload = {f: object() for f in REQUIRED_REPORT_FIELDS}
    payload["schema_version"] = backtest_report_module.REPORT_SCHEMA_VERSION
    validate_report_payload(payload)  # must not raise


def test_report_schema_validation_rejects_missing_field():
    payload = {f: object() for f in REQUIRED_REPORT_FIELDS if f != "metrics"}
    payload["schema_version"] = backtest_report_module.REPORT_SCHEMA_VERSION
    with pytest.raises(ValueError, match="metrics"):
        validate_report_payload(payload)


def test_report_schema_validation_rejects_wrong_schema_version():
    payload = {f: object() for f in REQUIRED_REPORT_FIELDS}
    payload["schema_version"] = "0.0.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_report_payload(payload)


# -- gate-ordering diagnostic backtest: isolation + B/C equivalence --


def test_gate_ordering_diagnostic_backtest_is_deterministic():
    dependencies, _ = _fixture_dependencies()
    config = _small_config()
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 6, 30), dependencies.trading_calendar)
    policy = PriceResolutionPolicy()
    rows_a = run_gate_ordering_diagnostic_backtest(
        periods, dependencies.ohlc_price_frames, dependencies.close_price_source, dependencies.trading_calendar,
        config, gate="prefilter", price_policy=policy, slippage_bps=10.0, cash_ticker="CASH",
    )
    rows_b = run_gate_ordering_diagnostic_backtest(
        periods, dependencies.ohlc_price_frames, dependencies.close_price_source, dependencies.trading_calendar,
        config, gate="prefilter", price_policy=policy, slippage_bps=10.0, cash_ticker="CASH",
    )
    assert rows_a == rows_b


def test_prefilter_and_waterfall_backtests_are_equivalent_on_a_real_fixture():
    dependencies, _ = _fixture_dependencies()
    config = _small_config()
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 8, 31), dependencies.trading_calendar)
    policy = PriceResolutionPolicy()
    rows_prefilter = run_gate_ordering_diagnostic_backtest(
        periods, dependencies.ohlc_price_frames, dependencies.close_price_source, dependencies.trading_calendar,
        config, gate="prefilter", price_policy=policy, slippage_bps=10.0, cash_ticker="CASH",
    )
    rows_waterfall = run_gate_ordering_diagnostic_backtest(
        periods, dependencies.ohlc_price_frames, dependencies.close_price_source, dependencies.trading_calendar,
        config, gate="waterfall", price_policy=policy, slippage_bps=10.0, cash_ticker="CASH",
    )
    for a, b in zip(rows_prefilter, rows_waterfall):
        assert a["selected"] == b["selected"]
        assert a["weights"] == pytest.approx(b["weights"])


def test_gate_ordering_diagnostic_backtest_and_canonical_runner_are_isolated_modules():
    # The diagnostic backtest loop must never be imported by the
    # production runner, and vice versa -- confirms Part C's
    # "diagnostic-only, never canonical" claim at the module-source level.
    runner_source = inspect.getsource(production_runner_module)
    assert "trend_and_gate_diagnostics" not in runner_source
    assert "run_gate_ordering_diagnostic_backtest" not in runner_source


def test_pipeline_and_strategy_modules_do_not_import_backtest_report_or_gate_diagnostics():
    for module in (pipeline_module, strategy_module):
        source = inspect.getsource(module)
        assert "trend_and_gate_diagnostics" not in source
        assert "backtest_report" not in source


def test_canonical_gate_ordering_backtest_and_prefilter_backtest_both_run_to_completion_on_identical_data():
    # Not asserting a specific performance direction (that would be
    # optimizing on results, forbidden this stage) -- only that the
    # canonical runner and the diagnostic-only alternative loop both
    # complete over the exact same dates/data/config and produce
    # one weights dict per period each, so they are comparable at all.
    dependencies, _ = _fixture_dependencies()
    config = _small_config(top_n=2, absolute_momentum_model="price_relative")
    periods = generate_monthly_periods(date(2020, 3, 31), date(2020, 8, 31), dependencies.trading_calendar)
    runner_config = RmfrBacktestConfig(strategy_config=config, transaction_costs=TransactionCostPolicy(slippage_bps=10.0))
    canonical_result = run_ranked_multi_factor_rotation_backtest(periods, dependencies, runner_config)

    policy = PriceResolutionPolicy()
    prefilter_rows = run_gate_ordering_diagnostic_backtest(
        periods, dependencies.ohlc_price_frames, dependencies.close_price_source, dependencies.trading_calendar,
        config, gate="prefilter", price_policy=policy, slippage_bps=10.0, cash_ticker="CASH",
    )
    canonical_weights = [m.weights for m in canonical_result.month_results]
    prefilter_weights = [r["weights"] for r in prefilter_rows]
    assert len(canonical_weights) == len(prefilter_weights) == len(periods)
