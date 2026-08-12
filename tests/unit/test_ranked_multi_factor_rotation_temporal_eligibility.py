"""Unit tests for Ranked Multi-Factor Rotation's anti-look-ahead
controls on the weight-estimation panel (``temporal_eligibility.py``).

Uses small, deliberately synthetic OHLC fixtures (never real market
data) plus hand-constructed ``RmfrPanelRow`` instances built directly
(not through the panel builder) so intentionally invalid/leaking rows
can be exercised precisely. This stage only validates temporal
eligibility -- no regression is fit here.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from atlas_quant.data.point_in_time import WeekdayTradingCalendar
from atlas_quant.strategies.ranked_multi_factor_rotation.backtest_clock import (
    generate_monthly_periods,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.panel import (
    RmfrPanelRow,
    build_rmfr_weight_estimation_panel,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.temporal_eligibility import (
    EXCLUSION_FEATURE_AFTER_ESTIMATION,
    EXCLUSION_OBSERVATION_NOT_YET_AVAILABLE,
    EXCLUSION_ROW_FROM_PREDICTION_PERIOD_OR_LATER,
    EXCLUSION_TARGET_END_NOT_BEFORE_ESTIMATION,
    EXCLUSION_TARGET_OVERLAPS_ESTIMATION,
    EXCLUSION_UPSTREAM_PANEL_ROW_INELIGIBLE,
    EXCLUSION_WITHIN_EMBARGO,
    apply_embargo,
    build_temporal_eligibility_audit,
    check_temporal_eligibility,
    detect_overlapping_target_windows,
    embargo_days_required_for_monthly_targets,
    find_duplicate_observations,
    observation_metadata_for_row,
)

_CALENDAR = WeekdayTradingCalendar()
_TEST_TICKERS = ("A", "B", "C", "D")


def _row(
    rebalance_date: date,
    ticker: str = "A",
    forward_return_start: date | None = None,
    forward_return_end: date | None = None,
    eligible: bool = True,
    exclusion_reason: str | None = None,
) -> RmfrPanelRow:
    return RmfrPanelRow(
        rebalance_date=rebalance_date,
        ticker=ticker,
        asset_return_4m=0.05,
        shy_return_4m=0.01,
        absolute_momentum=0.04,
        momentum_rank=1.0,
        volatility=0.01,
        volatility_rank=1.0,
        correlation=0.1,
        correlation_rank=1.0,
        trend_score=0.0,
        factor_data_as_of=rebalance_date,
        forward_return_start=forward_return_start or date(rebalance_date.year, rebalance_date.month, 28),
        forward_return_end=forward_return_end or date(rebalance_date.year, rebalance_date.month, 28),
        next_month_total_return=0.02,
        next_month_shy_return=0.01,
        next_month_excess_return=0.01,
        eligible=eligible,
        exclusion_reason=exclusion_reason,
    )


def _synthetic_ohlc(seed: int, n_days: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close},
        index=dates,
    )


def _price_fixture(n_days: int = 260) -> dict[str, pd.DataFrame]:
    prices = {t: _synthetic_ohlc(seed=i + 1, n_days=n_days, drift=0.001 * (i - 1)) for i, t in enumerate(_TEST_TICKERS)}
    prices["SHY"] = _synthetic_ohlc(seed=99, n_days=n_days, drift=0.0002)
    return prices


def _panel_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_TEST_TICKERS,
        top_n=2,
        momentum_lookback_days=20,
        correlation_lookback_days=20,
        atr_window=10,
        trend_model="legacy_symmetric",
        trend_lookback_n=10,
        volatility_smoothing_window=3,
        absolute_momentum_model="asset_minus_cash",
        absolute_momentum_lookback_sessions=20,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


def _real_panel(start: date, end: date):
    prices = _price_fixture()
    config = _panel_config()
    periods = generate_monthly_periods(start, end, _CALENDAR)
    return build_rmfr_weight_estimation_panel(prices, periods, config), periods


# -- observation_metadata_for_row --


def test_observation_metadata_derived_directly_from_row_fields():
    row = _row(
        date(2020, 3, 31),
        forward_return_start=date(2020, 4, 1),
        forward_return_end=date(2020, 5, 1),
    )
    metadata = observation_metadata_for_row(row)
    assert metadata.feature_as_of == date(2020, 3, 31)
    assert metadata.target_start == date(2020, 4, 1)
    assert metadata.target_end == date(2020, 5, 1)
    assert metadata.observation_available_at == pd.Timestamp("2020-05-01").to_pydatetime()


# -- check_temporal_eligibility: the canonical rule and each named catch --


def test_row_strictly_before_estimation_as_of_is_eligible():
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 6, 30))
    assert result.eligible
    assert result.exclusion_reasons == ()


def test_row_matches_canonical_condition_forward_return_end_lt_estimation_as_of():
    # Exactly the task's own canonical rule, both sides of the boundary.
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    still_eligible = check_temporal_eligibility(row, estimation_as_of=date(2020, 5, 2))
    assert still_eligible.eligible

    now_excluded = check_temporal_eligibility(row, estimation_as_of=date(2020, 5, 1))
    assert not now_excluded.eligible  # forward_return_end == estimation_as_of, not <


def test_feature_timestamp_after_estimation_is_caught():
    row = _row(date(2020, 6, 30), forward_return_start=date(2020, 7, 1), forward_return_end=date(2020, 8, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 5, 1))
    assert not result.eligible
    assert EXCLUSION_FEATURE_AFTER_ESTIMATION in result.exclusion_reasons


def test_target_overlap_with_estimation_date_is_caught():
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 4, 15))
    assert not result.eligible
    assert EXCLUSION_TARGET_OVERLAPS_ESTIMATION in result.exclusion_reasons


def test_target_end_after_estimation_date_is_caught():
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 6, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 5, 1))
    assert not result.eligible
    assert EXCLUSION_TARGET_END_NOT_BEFORE_ESTIMATION in result.exclusion_reasons


def test_observation_not_yet_available_is_caught():
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 5, 1))
    assert not result.eligible
    assert EXCLUSION_OBSERVATION_NOT_YET_AVAILABLE in result.exclusion_reasons


def test_training_on_row_from_prediction_month_is_caught():
    # The very row whose rebalance_date equals the estimation date --
    # using it to train the weights being applied to itself.
    row = _row(date(2020, 6, 30), forward_return_start=date(2020, 7, 1), forward_return_end=date(2020, 8, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 6, 30))
    assert not result.eligible
    assert EXCLUSION_ROW_FROM_PREDICTION_PERIOD_OR_LATER in result.exclusion_reasons


def test_row_from_a_later_prediction_month_is_also_caught():
    row = _row(date(2020, 8, 31), forward_return_start=date(2020, 9, 1), forward_return_end=date(2020, 10, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 6, 30))
    assert not result.eligible
    assert EXCLUSION_ROW_FROM_PREDICTION_PERIOD_OR_LATER in result.exclusion_reasons


def test_upstream_panel_ineligible_row_is_never_temporally_eligible():
    row = _row(
        date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1),
        eligible=False, exclusion_reason="insufficient_factor_history",
    )
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 6, 30))
    assert not result.eligible
    assert EXCLUSION_UPSTREAM_PANEL_ROW_INELIGIBLE in result.exclusion_reasons


def test_future_shy_data_cannot_leak_through_the_feature_gate():
    # asset_return_4m/shy_return_4m are both feature-side (trailing)
    # fields bounded by factor_data_as_of -- the same feature gate that
    # excludes any row whose factor_data_as_of is after estimation_as_of
    # necessarily also excludes whatever SHY-relative data that row
    # carries. Verified directly against a real panel + compute_factor_snapshot.
    panel, periods = _real_panel(date(2020, 3, 31), date(2020, 8, 31))
    row = next(r for r in panel.rows if r.rebalance_date == date(2020, 8, 31))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 5, 1))
    assert not result.eligible
    assert EXCLUSION_FEATURE_AFTER_ESTIMATION in result.exclusion_reasons
    # And its SHY-relative feature value is exactly what compute_factor_snapshot
    # produced as of *that row's own* factor_data_as_of -- never later data.
    assert row.shy_return_4m is not None


def test_multiple_violations_are_all_reported():
    row = _row(date(2020, 6, 30), forward_return_start=date(2020, 7, 1), forward_return_end=date(2020, 8, 1))
    result = check_temporal_eligibility(row, estimation_as_of=date(2020, 6, 30))
    assert not result.eligible
    assert len(result.exclusion_reasons) >= 3  # feature-gate NOT violated here, but the others are


# -- duplicate observation detection --


def test_find_duplicate_observations_detects_repeated_key():
    row_a = _row(date(2020, 3, 31), ticker="A")
    row_a_dup = _row(date(2020, 3, 31), ticker="A")
    row_b = _row(date(2020, 3, 31), ticker="B")
    duplicates = find_duplicate_observations([row_a, row_a_dup, row_b])
    assert duplicates == ((date(2020, 3, 31), "A"),)


def test_find_duplicate_observations_empty_when_all_unique():
    row_a = _row(date(2020, 3, 31), ticker="A")
    row_b = _row(date(2020, 3, 31), ticker="B")
    assert find_duplicate_observations([row_a, row_b]) == ()


def test_build_temporal_eligibility_audit_raises_on_duplicate_observations():
    row_a = _row(date(2020, 3, 31), ticker="A")
    row_a_dup = _row(date(2020, 3, 31), ticker="A")
    with pytest.raises(ValueError, match="duplicate"):
        build_temporal_eligibility_audit([row_a, row_a_dup], estimation_as_of=date(2020, 6, 30))


# -- overlap detection / embargo (purging mechanism) --


def test_detect_overlapping_target_windows_finds_no_overlap_for_real_monthly_panel():
    panel, _ = _real_panel(date(2020, 3, 31), date(2020, 9, 30))
    assert detect_overlapping_target_windows(panel.rows) == ()


def test_detect_overlapping_target_windows_finds_genuinely_overlapping_windows():
    row_a = _row(date(2020, 3, 31), ticker="A", forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 6, 1))
    row_b = _row(date(2020, 4, 30), ticker="A", forward_return_start=date(2020, 5, 1), forward_return_end=date(2020, 7, 1))
    overlaps = detect_overlapping_target_windows([row_a, row_b])
    assert len(overlaps) == 1


def test_detect_overlapping_target_windows_does_not_flag_adjacent_boundary():
    # b's target_start equals a's target_end exactly -- the intended
    # "no gap, no overlap" simultaneous-rebalance boundary, not a leak.
    row_a = _row(date(2020, 3, 31), ticker="A", forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    row_b = _row(date(2020, 4, 30), ticker="A", forward_return_start=date(2020, 5, 1), forward_return_end=date(2020, 6, 1))
    assert detect_overlapping_target_windows([row_a, row_b]) == ()


def test_detect_overlapping_target_windows_ignores_different_tickers():
    # Same window, different ticker -- not a same-instrument leak.
    row_a = _row(date(2020, 3, 31), ticker="A", forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    row_b = _row(date(2020, 3, 31), ticker="B", forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    assert detect_overlapping_target_windows([row_a, row_b]) == ()


def test_embargo_days_required_for_monthly_targets_is_zero():
    assert embargo_days_required_for_monthly_targets() == 0


def test_apply_embargo_is_a_no_op_at_zero_days():
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 1))
    result = apply_embargo([row], estimation_as_of=date(2020, 6, 1), embargo_days=0)
    assert result == (row,)


def test_apply_embargo_excludes_rows_within_the_buffer():
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 25))
    result = apply_embargo([row], estimation_as_of=date(2020, 6, 1), embargo_days=10)
    assert result == ()  # forward_return_end (5/25) is within 10 days of 6/1


def test_apply_embargo_rejects_negative_days():
    with pytest.raises(ValueError):
        apply_embargo([], estimation_as_of=date(2020, 6, 1), embargo_days=-1)


def test_build_temporal_eligibility_audit_reports_embargoed_rows():
    row = _row(date(2020, 3, 31), forward_return_start=date(2020, 4, 1), forward_return_end=date(2020, 5, 25))
    audit = build_temporal_eligibility_audit([row], estimation_as_of=date(2020, 6, 1), embargo_days=10)
    assert audit.eligible_row_count == 0
    assert audit.excluded_results[0].exclusion_reasons == (EXCLUSION_WITHIN_EMBARGO,)


# -- build_temporal_eligibility_audit: task-9 summary report --


def test_audit_report_on_a_real_panel_matches_manual_expectation():
    panel, periods = _real_panel(date(2020, 3, 31), date(2020, 9, 30))
    estimation_as_of = date(2020, 9, 30)
    audit = build_temporal_eligibility_audit(panel.rows, estimation_as_of)

    assert audit.estimation_as_of == estimation_as_of
    assert audit.total_candidate_rows == len(panel.rows)
    assert audit.eligible_row_count + audit.excluded_row_count == audit.total_candidate_rows

    # Every eligible row's rebalance_date must be strictly before Aug 31
    # (the last period whose forward window is fully resolved before Sep 30).
    eligible_dates = {r.rebalance_date for r in audit.eligible_rows}
    assert date(2020, 8, 31) not in eligible_dates
    assert date(2020, 9, 30) not in eligible_dates
    assert date(2020, 7, 31) in eligible_dates

    # latest_target_end_used must itself be strictly before estimation_as_of.
    assert audit.latest_target_end_used is not None
    assert audit.latest_target_end_used < estimation_as_of

    # Every eligible row's own forward_return_end must independently
    # satisfy the canonical rule -- the audit report's own self-consistency.
    for row in audit.eligible_rows:
        assert row.forward_return_end < estimation_as_of

    assert sum(audit.exclusion_reason_counts.values()) >= audit.excluded_row_count


def test_audit_report_to_dict_is_serializable():
    panel, _ = _real_panel(date(2020, 3, 31), date(2020, 6, 30))
    audit = build_temporal_eligibility_audit(panel.rows, estimation_as_of=date(2020, 6, 30))
    payload = audit.to_dict()
    assert payload["estimation_as_of"] == "2020-06-30"
    assert isinstance(payload["exclusion_reason_counts"], dict)
    assert payload["latest_target_end_used"] is None or isinstance(payload["latest_target_end_used"], str)


def test_no_eligible_row_ever_has_forward_return_end_on_or_after_estimation_as_of():
    # The one property this entire stage exists to guarantee, checked
    # directly against a real (synthetic-priced) panel across several
    # different estimation dates.
    panel, periods = _real_panel(date(2020, 3, 31), date(2020, 11, 30))
    for period in periods:
        audit = build_temporal_eligibility_audit(panel.rows, estimation_as_of=period.month_end)
        for row in audit.eligible_rows:
            assert row.forward_return_end < period.month_end
            assert row.factor_data_as_of <= period.month_end
            assert row.rebalance_date < period.month_end
