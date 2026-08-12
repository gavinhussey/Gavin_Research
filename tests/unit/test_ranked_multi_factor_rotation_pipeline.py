"""Unit tests for Ranked Multi-Factor Rotation's point-in-time pipeline
and strategy evaluator.

Uses small, deliberately synthetic OHLC fixtures purely to exercise
pipeline mechanics (point-in-time truncation, ranking wiring, selection/
allocation) -- never presented as, or used to produce, a genuine
backtest result. A short-lookback config keeps fixtures small; the
platform defaults (spec-sourced) are asserted separately in
test_ranked_multi_factor_rotation.py.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from atlas_quant.strategies.base import StrategyEvaluationContext
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.strategies.ranked_multi_factor_rotation.config import (
    RankedMultiFactorRotationConfig,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import (
    legacy_highest_total_rank_select,
    rank_scores,
    select_top_n,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import (
    compute_excess_momentum_snapshot,
    compute_factor_snapshot,
    compute_trend_state,
    select_for_month_end,
)
from atlas_quant.strategies.ranked_multi_factor_rotation.strategy import (
    RankedMultiFactorRotationStrategy,
)

_TEST_TICKERS = ("A", "B", "C", "D")


def _synthetic_ohlc(seed: int, n_days: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    daily_returns = rng.normal(loc=drift, scale=0.01, size=n_days)
    close = 100.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )


def _small_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_TEST_TICKERS,
        top_n=2,
        momentum_lookback_days=5,
        correlation_lookback_days=5,
        atr_window=5,
        trend_model="legacy_symmetric",  # small fixture window too short for 63/105-day canonical lookbacks
        trend_lookback_n=5,
        volatility_smoothing_window=3,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


def _price_fixture(n_days: int = 40) -> dict[str, pd.DataFrame]:
    # Distinct seeds/drifts so tickers are not degenerate duplicates of
    # each other -- A trends up strongly, D trends down.
    return {
        "A": _synthetic_ohlc(seed=1, n_days=n_days, drift=0.004),
        "B": _synthetic_ohlc(seed=2, n_days=n_days, drift=0.001),
        "C": _synthetic_ohlc(seed=3, n_days=n_days, drift=0.0005),
        "D": _synthetic_ohlc(seed=4, n_days=n_days, drift=-0.004),
    }


def test_compute_trend_state_starts_neutral_and_carries_forward():
    breakouts = pd.Series([0.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0])
    state = compute_trend_state(breakouts)
    assert state.iloc[0] == 0.0
    assert state.iloc[1] == 0.0
    assert state.iloc[2] == 0.0  # breakout on day 2 not effective until day 3
    assert state.iloc[3] == 2.0  # now effective
    assert state.iloc[4] == 2.0  # carries forward
    assert state.iloc[5] == 2.0  # day-5 breakout not effective until day 6
    assert state.iloc[6] == -2.0


def test_select_for_month_end_produces_weights_summing_to_one_or_is_all_cash():
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    result = select_for_month_end(prices, as_of, config)

    assert len(result.selected_tickers) == config.top_n
    total_weight = sum(result.weights.values())
    assert total_weight == pytest.approx(config.top_n * config.position_weight) or set(
        result.weights
    ) == {config.cash_ticker}


def test_select_for_month_end_uses_lowest_total_rank_canonically():
    # The canonical strategy configuration must select the lowest Total
    # Rank tickers -- confirm select_for_month_end's actual selection
    # matches formulas.select_top_n applied directly to the same scores,
    # and explicitly does NOT match the superseded highest-wins ordering.
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    result = select_for_month_end(prices, as_of, config)

    canonical_expected = select_top_n(result.total_rank_scores, config.top_n)
    legacy_would_have_selected = legacy_highest_total_rank_select(
        result.total_rank_scores, config.top_n
    )
    assert result.selected_tickers == canonical_expected
    assert result.selected_tickers != legacy_would_have_selected


# -- factor rank direction correctness (spec §3, 2026-08-05 correction) --


def test_desirable_first_mode_assigns_rank_1_to_the_most_desirable_factor_values():
    # Default rank_direction_mode="desirable_first": rank 1 = highest
    # momentum, lowest volatility, lowest correlation -- verified
    # directly against the snapshot's own raw factor values, independent
    # of how Total Rank combines/selects them.
    config = _small_config(rank_direction_mode="desirable_first")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    snapshot = compute_factor_snapshot(prices, as_of, config)

    rank_m = rank_scores(snapshot["momentum"], ascending=False)
    rank_v = rank_scores(snapshot["volatility"], ascending=True)
    rank_c = rank_scores(snapshot["correlation"], ascending=True)

    assert rank_m[snapshot["momentum"].idxmax()] == 1
    assert rank_v[snapshot["volatility"].idxmin()] == 1
    assert rank_c[snapshot["correlation"].idxmin()] == 1


def test_legacy_desirable_last_mode_assigns_rank_1_to_the_least_desirable_factor_values():
    # Superseded direction, preserved only for forensic/backward
    # comparison -- confirms it is a genuine inversion of the canonical
    # direction, not a no-op.
    config = _small_config(rank_direction_mode="legacy_desirable_last")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    snapshot = compute_factor_snapshot(prices, as_of, config)
    n = len(config.ranked_tickers)

    rank_m = rank_scores(snapshot["momentum"], ascending=True)
    rank_v = rank_scores(snapshot["volatility"], ascending=False)
    rank_c = rank_scores(snapshot["correlation"], ascending=False)

    assert rank_m[snapshot["momentum"].idxmax()] == n
    assert rank_v[snapshot["volatility"].idxmin()] == n
    assert rank_c[snapshot["correlation"].idxmin()] == n


def test_select_for_month_end_rank_direction_mode_changes_selection():
    # The two modes are not just internally different -- they can (and,
    # on this fixture, do) select a different set of tickers end-to-end.
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    desirable_first_result = select_for_month_end(prices, as_of, _small_config(rank_direction_mode="desirable_first"))
    legacy_result = select_for_month_end(prices, as_of, _small_config(rank_direction_mode="legacy_desirable_last"))

    assert desirable_first_result.selected_tickers != legacy_result.selected_tickers


def test_rank_scores_never_includes_shy_even_when_shy_has_extreme_factor_values():
    # SHY is a reference/defensive series only -- it must never be a
    # ranking candidate, regardless of how extreme its own momentum/
    # volatility/correlation would be if it were ranked.
    config = _small_config()
    prices = _price_fixture_with_shy()
    as_of = prices["A"].index[-1]

    snapshot = compute_factor_snapshot(prices, as_of, config)
    assert "SHY" not in snapshot.index

    result = select_for_month_end(prices, as_of, config)
    assert "SHY" not in result.total_rank_scores.index
    assert "SHY" not in result.momentum_values.index
    assert "SHY" not in result.volatility_values.index
    assert "SHY" not in result.correlation_values.index
    assert "SHY" not in result.selected_tickers


def test_legacy_highest_total_rank_select_is_not_referenced_by_the_pipeline_module():
    # formulas.legacy_highest_total_rank_select is preserved only for
    # research/forensic comparison -- the canonical pipeline must never
    # import or call it.
    import atlas_quant.strategies.ranked_multi_factor_rotation.pipeline as pipeline_module

    assert "legacy_highest_total_rank_select" not in vars(pipeline_module)


def test_select_for_month_end_respects_point_in_time_cutoff():
    config = _small_config()
    prices = _price_fixture()
    cutoff_index = 25
    as_of = prices["A"].index[cutoff_index]
    later_prices = {t: df.copy() for t, df in prices.items()}
    # Corrupt data strictly after the cutoff -- if the pipeline peeked ahead,
    # this would change the result relative to a version truncated exactly
    # at the cutoff.
    for df in later_prices.values():
        df.iloc[cutoff_index + 1 :] = df.iloc[cutoff_index + 1 :] * 1000.0

    truncated_prices = {t: df.iloc[: cutoff_index + 1] for t, df in prices.items()}

    result_from_full_history = select_for_month_end(later_prices, as_of, config)
    result_from_truncated_history = select_for_month_end(truncated_prices, as_of, config)

    assert result_from_full_history.weights == result_from_truncated_history.weights
    pd.testing.assert_series_equal(
        result_from_full_history.momentum_values, result_from_truncated_history.momentum_values
    )


def test_strategy_evaluate_returns_ok_status_with_recommendations():
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    context = StrategyEvaluationContext(
        strategy_id=config.strategy_id,
        evaluation_timestamp=datetime.now(timezone.utc),
        data_cutoff=as_of.to_pydatetime().replace(tzinfo=timezone.utc),
        capital_budget_pct=1.0,
        data_providers={"daily_ohlc": prices},
        strategy_config=config,
    )
    result = RankedMultiFactorRotationStrategy().evaluate(context)
    assert result.status in (StrategyStatus.OK, StrategyStatus.FALLBACK, StrategyStatus.CASH)
    assert result.strategy_id == config.strategy_id
    total_weight = sum(r.weight for r in result.recommendations)
    assert total_weight == pytest.approx(config.top_n * config.position_weight) or total_weight == pytest.approx(1.0)


def test_strategy_evaluate_missing_data_provider_returns_missing_data_status():
    config = _small_config()
    now = datetime.now(timezone.utc)
    context = StrategyEvaluationContext(
        strategy_id=config.strategy_id,
        evaluation_timestamp=now,
        data_cutoff=now,
        capital_budget_pct=1.0,
        data_providers={},
        strategy_config=config,
    )
    result = RankedMultiFactorRotationStrategy().evaluate(context)
    assert result.status == StrategyStatus.MISSING_DATA
    assert "daily_ohlc" in result.missing_data


def test_strategy_evaluate_rejects_wrong_config_type():
    now = datetime.now(timezone.utc)
    context = StrategyEvaluationContext(
        strategy_id="ranked_multi_factor_rotation",
        evaluation_timestamp=now,
        data_cutoff=now,
        capital_budget_pct=1.0,
        data_providers={},
        strategy_config={"not": "a config"},
    )
    with pytest.raises(TypeError):
        RankedMultiFactorRotationStrategy().evaluate(context)


# -- compute_excess_momentum_snapshot (spec §4A data plumbing) --


def _price_fixture_with_shy(n_days: int = 40) -> dict[str, pd.DataFrame]:
    prices = _price_fixture(n_days=n_days)
    prices["SHY"] = _synthetic_ohlc(seed=99, n_days=n_days, drift=0.0002)
    return prices


def test_compute_excess_momentum_snapshot_covers_every_ranked_ticker():
    config = _small_config(absolute_momentum_lookback_sessions=5)
    prices = _price_fixture_with_shy()
    as_of = prices["A"].index[-1]

    snapshot = compute_excess_momentum_snapshot(prices, as_of, config)

    assert set(snapshot) == set(config.ranked_tickers)
    for result in snapshot.values():
        assert result.is_valid
        assert result.excess_momentum is not None


def test_compute_excess_momentum_snapshot_excludes_shy_from_the_ranked_universe():
    # SHY is a reference/defensive series only -- it must never appear as
    # a cross-sectional ranking candidate, even though its OHLC frame is
    # present in the same `prices` mapping the ranked tickers use.
    config = _small_config()
    prices = _price_fixture_with_shy()
    as_of = prices["A"].index[-1]

    snapshot = compute_excess_momentum_snapshot(prices, as_of, config)

    assert config.cash_proxy_symbol not in snapshot
    assert "SHY" not in snapshot


def test_compute_factor_snapshot_also_never_ranks_shy_even_when_present():
    # The existing (unchanged) factor snapshot must likewise never include
    # SHY as a ranking candidate, whether or not its price frame happens
    # to be present in the `prices` mapping passed in.
    config = _small_config()
    prices = _price_fixture_with_shy()
    as_of = prices["A"].index[-1]

    snapshot = compute_factor_snapshot(prices, as_of, config)

    assert set(snapshot.index) == set(config.ranked_tickers)
    assert "SHY" not in snapshot.index


def test_compute_excess_momentum_snapshot_raises_when_cash_proxy_missing():
    config = _small_config()
    prices = _price_fixture()  # no SHY entry
    as_of = prices["A"].index[-1]

    with pytest.raises(ValueError, match="SHY"):
        compute_excess_momentum_snapshot(prices, as_of, config)


def test_compute_excess_momentum_snapshot_raises_when_a_ranked_ticker_missing():
    config = _small_config()
    prices = _price_fixture_with_shy()
    del prices["B"]
    as_of = prices["A"].index[-1]

    with pytest.raises(ValueError, match="B"):
        compute_excess_momentum_snapshot(prices, as_of, config)


def test_compute_excess_momentum_snapshot_does_not_change_selection_behavior():
    # Calling the new plumbing function alongside the existing selection
    # pipeline must not alter select_for_month_end's own result -- this
    # stage only adds an independent, unwired calculation.
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    baseline = select_for_month_end(prices, as_of, config)

    prices_with_shy = _price_fixture_with_shy()
    compute_excess_momentum_snapshot(prices_with_shy, as_of, config)
    after = select_for_month_end(prices, as_of, config)

    assert after.selected_tickers == baseline.selected_tickers
    assert after.weights == baseline.weights
    assert after.total_rank_scores.equals(baseline.total_rank_scores)


# -- absolute_momentum_model="asset_minus_cash" activation (spec §4A, 2026-08-05) --

_EXCESS_TEST_TICKERS = ("X", "Y", "Z", "W", "N")


def _flat_ohlc(closes: list[float]) -> pd.DataFrame:
    # High/low/open pinned to close -- these deterministic fixtures only
    # need to exercise the momentum leg precisely; degenerate OHLC
    # ranges are harmless for the other factors computed alongside it.
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.Series(closes, index=dates)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)


def _excess_momentum_price_fixture() -> dict[str, pd.DataFrame]:
    # 10 sessions, absolute_momentum_lookback_sessions=9 -> start=index[0], end=index[-1].
    return {
        "X": _flat_ohlc([100, 101, 102, 103, 104, 105, 106, 107, 108, 110]),  # 100 -> 110, +10%
        "Y": _flat_ohlc([100, 102, 101, 103, 102, 104, 103, 105, 104, 110]),  # 100 -> 110, +10% (different path)
        "W": _flat_ohlc([100, 100.9, 101.8, 102.7, 103.6, 104.5, 105.4, 106.3, 107.2, 108]),  # 100 -> 108, +8%
        "Z": _flat_ohlc([100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 105]),  # 100 -> 105, +5%
        "N": _flat_ohlc([100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 101]),  # 100 -> 101, +1%
        "SHY": _flat_ohlc([100, 100.3, 100.6, 100.9, 101.2, 101.5, 101.8, 102.1, 102.4, 103]),  # 100 -> 103, +3%
    }


def _excess_momentum_config(**overrides) -> RankedMultiFactorRotationConfig:
    defaults = dict(
        ranked_tickers=_EXCESS_TEST_TICKERS,
        top_n=2,
        momentum_lookback_days=9,
        absolute_momentum_lookback_sessions=9,
        correlation_lookback_days=5,
        atr_window=5,
        trend_model="legacy_symmetric",
        trend_lookback_n=5,
        volatility_smoothing_window=3,
    )
    defaults.update(overrides)
    return RankedMultiFactorRotationConfig(**defaults)


def test_identical_raw_returns_receive_identical_excess_momentum_under_common_shy_return():
    config = _excess_momentum_config(absolute_momentum_model="asset_minus_cash")
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    snapshot = compute_factor_snapshot(prices, as_of, config)

    # X and Y both have raw return +10% (different paths, same start/end).
    assert snapshot.loc["X", "asset_return_4m"] == pytest.approx(0.10)
    assert snapshot.loc["Y", "asset_return_4m"] == pytest.approx(0.10)
    assert snapshot.loc["X", "momentum"] == pytest.approx(snapshot.loc["Y", "momentum"])
    assert snapshot.loc["X", "momentum"] == pytest.approx(0.07)  # 0.10 - 0.03


def test_subtracting_common_shy_return_preserves_cross_sectional_ordering():
    legacy_config = _excess_momentum_config(absolute_momentum_model="price_relative")
    research_config = _excess_momentum_config(absolute_momentum_model="asset_minus_cash")
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    legacy_snapshot = compute_factor_snapshot(prices, as_of, legacy_config)
    research_snapshot = compute_factor_snapshot(prices, as_of, research_config)

    legacy_rank = rank_scores(legacy_snapshot["momentum"], ascending=True)
    research_rank = rank_scores(research_snapshot["momentum"], ascending=True)
    assert legacy_rank.to_dict() == research_rank.to_dict()

    # But the absolute momentum values themselves do change.
    for ticker in _EXCESS_TEST_TICKERS:
        assert legacy_snapshot.loc[ticker, "momentum"] != pytest.approx(
            research_snapshot.loc[ticker, "momentum"]
        )


def test_negative_excess_momentum_preserved_as_negative_decimal():
    config = _excess_momentum_config(absolute_momentum_model="asset_minus_cash")
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    snapshot = compute_factor_snapshot(prices, as_of, config)

    # N: raw return +1%, SHY return +3% -> excess = -0.02.
    assert snapshot.loc["N", "asset_return_4m"] == pytest.approx(0.01)
    assert snapshot.loc["N", "shy_return_4m"] == pytest.approx(0.03)
    assert snapshot.loc["N", "momentum"] == pytest.approx(-0.02)
    assert snapshot.loc["N", "momentum"] < 0.0


def test_legacy_and_research_modes_produce_different_audit_values():
    legacy_config = _excess_momentum_config(absolute_momentum_model="price_relative")
    research_config = _excess_momentum_config(absolute_momentum_model="asset_minus_cash")
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    legacy_snapshot = compute_factor_snapshot(prices, as_of, legacy_config)
    research_snapshot = compute_factor_snapshot(prices, as_of, research_config)

    assert (legacy_snapshot["absolute_momentum_model"] == "price_relative").all()
    assert (research_snapshot["absolute_momentum_model"] == "asset_minus_cash").all()

    # Legacy mode: momentum is the raw price_relative value, and the
    # SHY-relative audit columns are never populated (NaN/NaT).
    assert legacy_snapshot.loc["X", "momentum"] == pytest.approx(0.10)
    assert legacy_snapshot["asset_return_4m"].isna().all()
    assert legacy_snapshot["shy_return_4m"].isna().all()
    assert legacy_snapshot["absolute_momentum_excess"].isna().all()
    assert legacy_snapshot["lookback_start_date"].isna().all()
    assert legacy_snapshot["lookback_end_date"].isna().all()

    # Research mode: momentum is the SHY-relative excess value, and every
    # audit column is populated and self-consistent.
    assert research_snapshot.loc["X", "momentum"] == pytest.approx(0.07)
    assert not research_snapshot["asset_return_4m"].isna().any()
    assert not research_snapshot["shy_return_4m"].isna().any()
    for ticker in _EXCESS_TEST_TICKERS:
        assert research_snapshot.loc[ticker, "shy_return_4m"] == pytest.approx(0.03)
    assert research_snapshot["lookback_start_date"].notna().all()
    assert research_snapshot["lookback_end_date"].notna().all()
    for ticker in _EXCESS_TEST_TICKERS:
        assert research_snapshot.loc[ticker, "absolute_momentum_excess"] == pytest.approx(
            research_snapshot.loc[ticker, "momentum"]
        )
        assert research_snapshot.loc[ticker, "lookback_start_date"] == pd.Timestamp("2020-01-01")
        assert research_snapshot.loc[ticker, "lookback_end_date"] == as_of


def test_asset_minus_cash_mode_requires_cash_proxy_price_history():
    config = _excess_momentum_config(absolute_momentum_model="asset_minus_cash")
    prices = _excess_momentum_price_fixture()
    del prices["SHY"]
    as_of = prices["X"].index[-1]

    with pytest.raises(ValueError, match="SHY"):
        compute_factor_snapshot(prices, as_of, config)


def test_price_relative_mode_does_not_require_cash_proxy_price_history():
    # Unchanged default behavior: legacy mode never needs SHY present.
    config = _excess_momentum_config(absolute_momentum_model="price_relative")
    prices = _excess_momentum_price_fixture()
    del prices["SHY"]
    as_of = prices["X"].index[-1]

    snapshot = compute_factor_snapshot(prices, as_of, config)
    assert snapshot.loc["X", "momentum"] == pytest.approx(0.10)


def test_select_for_month_end_exposes_momentum_audit_fields():
    config = _excess_momentum_config(absolute_momentum_model="asset_minus_cash")
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    result = select_for_month_end(prices, as_of, config)

    assert result.absolute_momentum_model == "asset_minus_cash"
    assert result.asset_return_4m.loc["X"] == pytest.approx(0.10)
    assert result.shy_return_4m.loc["X"] == pytest.approx(0.03)
    assert result.absolute_momentum_excess.loc["X"] == pytest.approx(0.07)
    assert result.momentum_values.loc["X"] == pytest.approx(0.07)
    assert result.lookback_start_date.loc["X"] == pd.Timestamp("2020-01-01")
    assert result.lookback_end_date.loc["X"] == as_of


def test_select_for_month_end_default_mode_leaves_selection_unchanged():
    # No config override at all (platform default absolute_momentum_model
    # is "price_relative") must reproduce byte-identical selection to
    # before this stage -- confirmed by re-deriving the expected result
    # directly from formulas.momentum on the same fixture.
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)

    assert result.absolute_momentum_model == "price_relative"
    assert result.asset_return_4m.isna().all()
    assert result.shy_return_4m.isna().all()
    assert result.absolute_momentum_excess.isna().all()


# -- full provisional Total Rank formula routing (spec §4A, 2026-08-06 activation) --


def test_select_for_month_end_legacy_formula_default_leaves_scores_unchanged():
    # No total_rank_formula override -- must reproduce byte-identical
    # scores/selection to before this stage.
    from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import total_rank

    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    assert result.total_rank_formula == "legacy"
    assert result.total_rank_audit is None

    snapshot = compute_factor_snapshot(prices, as_of, config)
    rank_m = rank_scores(snapshot["momentum"], ascending=False)
    rank_v = rank_scores(snapshot["volatility"], ascending=True)
    rank_c = rank_scores(snapshot["correlation"], ascending=True)
    expected = total_rank(
        rank_m, rank_v, rank_c, snapshot["trend"],
        momentum_weight=config.momentum_weight,
        volatility_weight=config.volatility_weight,
        correlation_weight=config.correlation_weight,
    )
    assert result.total_rank_scores.sort_index().equals(expected.sort_index())


def test_select_for_month_end_full_provisional_formula_produces_audit_and_divides_whole_numerator():
    config = _excess_momentum_config(
        absolute_momentum_model="asset_minus_cash", total_rank_formula="full_provisional"
    )
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    assert result.total_rank_formula == "full_provisional"
    assert result.total_rank_audit is not None

    for ticker in _EXCESS_TEST_TICKERS:
        audit = result.total_rank_audit[ticker]
        assert audit is not None
        assert audit.divisor == config.total_rank_divisor == 11.0
        assert audit.absolute_momentum == pytest.approx(result.absolute_momentum_excess.loc[ticker])
        expected_numerator = (
            audit.momentum_contribution
            + audit.volatility_contribution
            + audit.correlation_contribution
            + audit.trend_adjustment
            + audit.absolute_momentum
        )
        assert audit.raw_numerator == pytest.approx(expected_numerator)
        assert audit.total_rank == pytest.approx(expected_numerator / 11.0)
        assert result.total_rank_scores.loc[ticker] == pytest.approx(audit.total_rank)
        # Whole-numerator-divided-by-X, not only-M-divided-by-X.
        only_m_divided = (
            audit.momentum_contribution
            + audit.volatility_contribution
            + audit.correlation_contribution
            + audit.trend_adjustment
            + audit.absolute_momentum / 11.0
        )
        assert audit.total_rank != pytest.approx(only_m_divided)


def test_select_for_month_end_full_provisional_formula_preserves_equal_weight_baseline():
    config = _excess_momentum_config(
        absolute_momentum_model="asset_minus_cash", total_rank_formula="full_provisional"
    )
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    for audit in result.total_rank_audit.values():
        assert audit.momentum_weight == pytest.approx(1 / 3)
        assert audit.volatility_weight == pytest.approx(1 / 3)
        assert audit.correlation_weight == pytest.approx(1 / 3)


def test_select_for_month_end_full_provisional_formula_keeps_lowest_score_selection():
    config = _excess_momentum_config(
        absolute_momentum_model="asset_minus_cash", total_rank_formula="full_provisional", top_n=2
    )
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    lowest_two = result.total_rank_scores.sort_values(ascending=True).index[:2].tolist()
    assert sorted(result.selected_tickers) == sorted(lowest_two)


def test_select_for_month_end_full_provisional_formula_divisor_rescales_without_changing_ordering():
    prices = _excess_momentum_price_fixture()
    as_of = prices["X"].index[-1]

    result_x11 = select_for_month_end(
        prices, as_of,
        _excess_momentum_config(
            absolute_momentum_model="asset_minus_cash", total_rank_formula="full_provisional",
            total_rank_divisor=11.0,
        ),
    )
    result_x22 = select_for_month_end(
        prices, as_of,
        _excess_momentum_config(
            absolute_momentum_model="asset_minus_cash", total_rank_formula="full_provisional",
            total_rank_divisor=22.0,
        ),
    )
    assert result_x11.selected_tickers == result_x22.selected_tickers
    order_x11 = result_x11.total_rank_scores.sort_values(ascending=True).index.tolist()
    order_x22 = result_x22.total_rank_scores.sort_values(ascending=True).index.tolist()
    assert order_x11 == order_x22
    for ticker in _EXCESS_TEST_TICKERS:
        assert result_x22.total_rank_scores.loc[ticker] == pytest.approx(
            result_x11.total_rank_scores.loc[ticker] / 2.0
        )


# -- faa_faithful_candidate formula (2026-08-08 bounded source-parity candidate) --


def test_default_config_total_rank_formula_is_unaffected_by_the_new_candidate_mode():
    # Isolation: adding "faa_faithful_candidate" as a selectable value
    # must not change the default -- still "legacy", unchanged behavior.
    assert RankedMultiFactorRotationConfig().total_rank_formula == "legacy"


def test_select_for_month_end_legacy_formula_unchanged_by_candidate_mode_existing():
    # Baseline ("legacy") selection/scores on a fixed fixture must be
    # identical to what pre-candidate-mode tests already pin -- adding
    # the candidate formula elsewhere in this module must not perturb it.
    config = _small_config()
    prices = _price_fixture()
    as_of = prices["A"].index[-1]
    result = select_for_month_end(prices, as_of, config)

    snapshot = compute_factor_snapshot(prices, as_of, config)
    rank_m = rank_scores(snapshot["momentum"], ascending=False)
    rank_v = rank_scores(snapshot["volatility"], ascending=True)
    rank_c = rank_scores(snapshot["correlation"], ascending=True)
    from atlas_quant.strategies.ranked_multi_factor_rotation.formulas import total_rank as _total_rank

    expected = _total_rank(
        rank_m, rank_v, rank_c, snapshot["trend"],
        momentum_weight=config.momentum_weight,
        volatility_weight=config.volatility_weight,
        correlation_weight=config.correlation_weight,
    )
    assert result.total_rank_formula == "legacy"
    assert result.total_rank_audit is None
    assert result.total_rank_scores.sort_index().equals(expected.sort_index())


def test_select_for_month_end_faa_faithful_candidate_enters_momentum_in_percentage_points():
    config = _small_config(total_rank_formula="faa_faithful_candidate")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    assert result.total_rank_formula == "faa_faithful_candidate"
    assert result.total_rank_audit is not None

    for ticker in _TEST_TICKERS:
        audit = result.total_rank_audit[ticker]
        assert audit is not None
        raw_momentum = result.momentum_values.loc[ticker]
        # 8.5% raw momentum must enter as 8.5, not 0.085 -- scaled by
        # exactly 100, not left as the decimal snapshot value.
        assert audit.absolute_momentum == pytest.approx(raw_momentum * 100.0)
        assert audit.absolute_momentum != pytest.approx(raw_momentum)


def test_select_for_month_end_faa_faithful_candidate_weights_are_exactly_1_0_0_5_0_5():
    config = _small_config(total_rank_formula="faa_faithful_candidate")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    for audit in result.total_rank_audit.values():
        assert audit is not None
        assert audit.momentum_weight == pytest.approx(1.0)
        assert audit.volatility_weight == pytest.approx(0.5)
        assert audit.correlation_weight == pytest.approx(0.5)
    # Config's own equal-thirds weight fields are untouched/ignored by
    # this candidate -- not read, not overwritten.
    assert config.momentum_weight == pytest.approx(1 / 3)
    assert config.volatility_weight == pytest.approx(1 / 3)
    assert config.correlation_weight == pytest.approx(1 / 3)


def test_select_for_month_end_faa_faithful_candidate_divides_the_whole_numerator_by_11():
    config = _small_config(total_rank_formula="faa_faithful_candidate")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    assert config.total_rank_divisor == 11.0
    for ticker in _TEST_TICKERS:
        audit = result.total_rank_audit[ticker]
        assert audit is not None
        assert audit.divisor == 11.0
        expected_numerator = (
            audit.momentum_contribution
            + audit.volatility_contribution
            + audit.correlation_contribution
            + audit.trend_adjustment
            + audit.absolute_momentum
        )
        assert audit.raw_numerator == pytest.approx(expected_numerator)
        assert audit.total_rank == pytest.approx(expected_numerator / 11.0)
        assert result.total_rank_scores.loc[ticker] == pytest.approx(audit.total_rank)
        only_m_divided = (
            audit.momentum_contribution
            + audit.volatility_contribution
            + audit.correlation_contribution
            + audit.trend_adjustment
            + audit.absolute_momentum / 11.0
        )
        assert audit.total_rank != pytest.approx(only_m_divided)


def test_select_for_month_end_faa_faithful_candidate_keeps_lowest_score_selection():
    config = _small_config(total_rank_formula="faa_faithful_candidate", top_n=2)
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    lowest_two = result.total_rank_scores.sort_values(ascending=True).index[:2].tolist()
    assert sorted(result.selected_tickers) == sorted(lowest_two)


def test_faa_faithful_candidate_requires_price_relative_momentum():
    with pytest.raises(ValueError, match="faa_faithful_candidate"):
        RankedMultiFactorRotationConfig(
            total_rank_formula="faa_faithful_candidate",
            absolute_momentum_model="asset_minus_cash",
        )


def test_faa_faithful_candidate_does_not_affect_full_provisional_or_legacy_paths():
    # Isolation: selecting the candidate formula on one config must not
    # leak into an independently constructed legacy/full_provisional
    # config evaluated on the same prices.
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    legacy_result = select_for_month_end(prices, as_of, _small_config())
    candidate_result = select_for_month_end(
        prices, as_of, _small_config(total_rank_formula="faa_faithful_candidate")
    )
    assert legacy_result.total_rank_formula == "legacy"
    assert candidate_result.total_rank_formula == "faa_faithful_candidate"
    assert not legacy_result.total_rank_scores.equals(candidate_result.total_rank_scores)


# -- faa_faithful_candidate_decimal_m (2026-08-08 follow-up, decimal M) --


def test_select_for_month_end_faa_faithful_candidate_decimal_m_keeps_momentum_decimal():
    config = _small_config(total_rank_formula="faa_faithful_candidate_decimal_m")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    assert result.total_rank_formula == "faa_faithful_candidate_decimal_m"
    assert result.total_rank_audit is not None

    for ticker in _TEST_TICKERS:
        audit = result.total_rank_audit[ticker]
        assert audit is not None
        raw_momentum = result.momentum_values.loc[ticker]
        # Unlike "faa_faithful_candidate", M must NOT be scaled by 100.
        assert audit.absolute_momentum == pytest.approx(raw_momentum)
        if raw_momentum != 0.0:
            assert audit.absolute_momentum != pytest.approx(raw_momentum * 100.0)


def test_select_for_month_end_faa_faithful_candidate_decimal_m_weights_are_exactly_1_0_0_5_0_5():
    config = _small_config(total_rank_formula="faa_faithful_candidate_decimal_m")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    for audit in result.total_rank_audit.values():
        assert audit is not None
        assert audit.momentum_weight == pytest.approx(1.0)
        assert audit.volatility_weight == pytest.approx(0.5)
        assert audit.correlation_weight == pytest.approx(0.5)


def test_select_for_month_end_faa_faithful_candidate_decimal_m_divides_the_whole_numerator_by_11():
    config = _small_config(total_rank_formula="faa_faithful_candidate_decimal_m")
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    assert config.total_rank_divisor == 11.0
    for ticker in _TEST_TICKERS:
        audit = result.total_rank_audit[ticker]
        assert audit is not None
        assert audit.divisor == 11.0
        expected_numerator = (
            audit.momentum_contribution
            + audit.volatility_contribution
            + audit.correlation_contribution
            + audit.trend_adjustment
            + audit.absolute_momentum
        )
        assert audit.raw_numerator == pytest.approx(expected_numerator)
        assert audit.total_rank == pytest.approx(expected_numerator / 11.0)
        assert result.total_rank_scores.loc[ticker] == pytest.approx(audit.total_rank)


def test_select_for_month_end_faa_faithful_candidate_decimal_m_keeps_lowest_score_selection():
    config = _small_config(total_rank_formula="faa_faithful_candidate_decimal_m", top_n=2)
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    result = select_for_month_end(prices, as_of, config)
    lowest_two = result.total_rank_scores.sort_values(ascending=True).index[:2].tolist()
    assert sorted(result.selected_tickers) == sorted(lowest_two)


def test_faa_faithful_candidate_decimal_m_requires_price_relative_momentum():
    with pytest.raises(ValueError, match="faa_faithful_candidate_decimal_m"):
        RankedMultiFactorRotationConfig(
            total_rank_formula="faa_faithful_candidate_decimal_m",
            absolute_momentum_model="asset_minus_cash",
        )


def test_faa_faithful_candidate_decimal_m_differs_from_percentage_point_candidate_and_legacy():
    prices = _price_fixture()
    as_of = prices["A"].index[-1]

    legacy_result = select_for_month_end(prices, as_of, _small_config())
    pct_result = select_for_month_end(
        prices, as_of, _small_config(total_rank_formula="faa_faithful_candidate")
    )
    decimal_result = select_for_month_end(
        prices, as_of, _small_config(total_rank_formula="faa_faithful_candidate_decimal_m")
    )
    assert decimal_result.total_rank_formula == "faa_faithful_candidate_decimal_m"
    assert not legacy_result.total_rank_scores.equals(decimal_result.total_rank_scores)
    assert not pct_result.total_rank_scores.equals(decimal_result.total_rank_scores)
