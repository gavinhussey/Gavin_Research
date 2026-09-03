"""Formula tests for Multi-Factor Ranking ML.

Cloned from filing_momentum_ml, minus the concrete feature formulas
(deleted along with the 17-feature set — this strategy's own feature
formulas are not yet defined). Only :func:`score_proportional_weights`
survives the clone: it is generic score-to-weight position sizing, not
a feature-engineering formula, and is unaffected by which features feed
the model.
"""

import math

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.formulas import score_proportional_weights


class TestScoreProportionalWeights:
    """w_i = P_i / sum(P_j for j in Q) * deployable_pct."""

    def test_one_instrument_receives_the_full_deployable_pct(self):
        weights = score_proportional_weights({"AAPL": 0.80}, deployable_pct=0.95)
        assert weights == {"AAPL": pytest.approx(0.95)}

    def test_multiple_instruments_equal_scores_split_evenly(self):
        weights = score_proportional_weights(
            {"AAPL": 0.5, "MSFT": 0.5, "NVDA": 0.5}, deployable_pct=0.95
        )
        assert weights["AAPL"] == pytest.approx(0.95 / 3)
        assert weights["MSFT"] == pytest.approx(0.95 / 3)
        assert weights["NVDA"] == pytest.approx(0.95 / 3)

    def test_unequal_scores_explicit_expected_values(self):
        # A=0.60, B=0.40, deployable_pct=0.95 -> A=0.57, B=0.38
        weights = score_proportional_weights({"A": 0.60, "B": 0.40}, deployable_pct=0.95)
        assert weights["A"] == pytest.approx(0.57, abs=1e-9)
        assert weights["B"] == pytest.approx(0.38, abs=1e-9)

    def test_default_deployable_pct_is_0_95(self):
        # Not a function default (this formula takes deployable_pct
        # explicitly, per its pure-function contract) -- this test pins the
        # value callers are expected to pass, matching
        # MultiFactorRankingMLConfig.deployable_pct's default.
        from atlas_quant.strategies.multi_factor_ranking_ml.config import (
            MultiFactorRankingMLConfig,
        )

        assert MultiFactorRankingMLConfig().deployable_pct == 0.95

    def test_weights_sum_to_deployable_pct(self):
        weights = score_proportional_weights(
            {"A": 0.9, "B": 0.4, "C": 0.1}, deployable_pct=0.95
        )
        assert sum(weights.values()) == pytest.approx(0.95, abs=1e-9)

    def test_empty_input_returns_empty_dict(self):
        assert score_proportional_weights({}, deployable_pct=0.95) == {}

    def test_zero_total_score_returns_zero_for_every_instrument_not_nan(self):
        weights = score_proportional_weights({"A": 0.0, "B": 0.0}, deployable_pct=0.95)
        assert weights == {"A": 0.0, "B": 0.0}
        assert all(not math.isnan(w) and math.isfinite(w) for w in weights.values())

    def test_negative_score_raises_value_error(self):
        with pytest.raises(ValueError):
            score_proportional_weights({"A": 0.5, "B": -0.1}, deployable_pct=0.95)

    def test_deployable_pct_below_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            score_proportional_weights({"A": 0.5}, deployable_pct=-0.01)

    def test_deployable_pct_above_one_raises_value_error(self):
        with pytest.raises(ValueError):
            score_proportional_weights({"A": 0.5}, deployable_pct=1.01)

    def test_deployable_pct_boundary_values_are_accepted(self):
        assert score_proportional_weights({"A": 0.5}, deployable_pct=0.0) == {"A": 0.0}
        assert score_proportional_weights({"A": 0.5}, deployable_pct=1.0) == {"A": 1.0}

    def test_output_is_deterministic_across_repeated_calls(self):
        scores = {"A": 0.7, "B": 0.2, "C": 0.1}
        first = score_proportional_weights(scores, deployable_pct=0.95)
        second = score_proportional_weights(scores, deployable_pct=0.95)
        assert first == second
        assert list(first.keys()) == list(second.keys()) == list(scores.keys())

    def test_weights_are_relative_to_strategy_budget_not_whole_portfolio(self):
        # deployable_pct=0.95 means 95% of *this strategy's* assigned
        # capital budget is deployed -- this function has no notion of
        # total portfolio capital and must not be interpreted as one. A
        # budget of 1.0 (standalone backtest default) or 0.25 (one of
        # several strategies) both produce weights that sum to
        # deployable_pct of *that budget*.
        full_budget_weights = score_proportional_weights(
            {"A": 0.6, "B": 0.4}, deployable_pct=0.95
        )
        assert sum(full_budget_weights.values()) == pytest.approx(0.95)
        # The function itself is budget-agnostic: the caller is
        # responsible for scaling by StrategyEvaluationContext
        # .capital_budget_pct separately, not by passing it in here.
