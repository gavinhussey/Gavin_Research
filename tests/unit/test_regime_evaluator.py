"""Unit tests for atlas_quant.strategies.filing_momentum_ml.regime_evaluator.

Covers gate-mode combinations, insufficient-data/fit-failure behavior,
single/batch parity (the critical regression test against future
live/backtest divergence), and result serialization.
"""

from datetime import date, datetime, timedelta

import pytest

from atlas_quant.domain.identifiers import AssetClass, InstrumentId
from atlas_quant.strategies.filing_momentum_ml.regime_config import RegimeConfig
from atlas_quant.strategies.filing_momentum_ml.regime_domain import (
    ComponentAvailability,
    ComponentClassification,
    RegimeClassification,
    RegimeResult,
    WindowClassification,
)
from atlas_quant.strategies.filing_momentum_ml.regime_evaluator import (
    RegimeEvaluationRequest,
    RegimeEvaluator,
)
from fixtures.filing_momentum_ml import FakeHMMFitter, instrument, make_daily_series


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _bull_prices(iid, n_days=1100):
    days = _weekdays(date(2022, 1, 3), n_days)
    return make_daily_series(iid, days, [0.0003] * n_days, start_price=100.0), days


def _bear_prices(iid, n_days=1100):
    days = _weekdays(date(2022, 1, 3), n_days)
    rates = [0.0002] * (n_days - 300) + [-0.002] * 300
    return make_daily_series(iid, days, rates, start_price=100.0), days


class TestComponentClassificationInvariant:
    def test_is_bear_requires_bear_classification(self):
        with pytest.raises(ValueError):
            ComponentClassification(
                component="markov",
                instrument_id=instrument(),
                evaluation_timestamp=datetime(2026, 1, 1),
                data_cutoff=datetime(2026, 1, 1),
                classification=RegimeClassification.BULL,
                is_bear=True,
                availability=ComponentAvailability.OK,
                confidence=None,
                observation_count=100,
                required_observation_count=10,
                windows=(),
                config_identity="a" * 64,
                provenance=(),
            )

    def test_unavailable_component_cannot_be_bear(self):
        with pytest.raises(ValueError):
            ComponentClassification(
                component="hmm",
                instrument_id=instrument(),
                evaluation_timestamp=datetime(2026, 1, 1),
                data_cutoff=datetime(2026, 1, 1),
                classification=RegimeClassification.BEAR,
                is_bear=True,
                availability=ComponentAvailability.INSUFFICIENT_HISTORY,
                confidence=None,
                observation_count=1,
                required_observation_count=30,
                windows=(),
                config_identity="a" * 64,
                provenance=(),
            )


class TestGateModes:
    """Test combinations of Markov Bear / HMM Bear across all 5 gate modes."""

    def _classification(self, is_bear, availability=ComponentAvailability.OK):
        classification = RegimeClassification.BEAR if is_bear else RegimeClassification.BULL
        if availability != ComponentAvailability.OK:
            classification = RegimeClassification.UNKNOWN
            is_bear = False
        return ComponentClassification(
            component="x",
            instrument_id=instrument(),
            evaluation_timestamp=datetime(2026, 1, 1),
            data_cutoff=datetime(2026, 1, 1),
            classification=classification,
            is_bear=is_bear,
            availability=availability,
            confidence=None,
            observation_count=100,
            required_observation_count=10,
            windows=(),
            config_identity="a" * 64,
            provenance=(),
        )

    @pytest.mark.parametrize(
        "mode,markov_bear,hmm_bear,expected_blocked",
        [
            ("both", True, True, True),
            ("both", True, False, False),
            ("both", False, True, False),
            ("both", False, False, False),
            ("either", True, True, True),
            ("either", True, False, True),
            ("either", False, True, True),
            ("either", False, False, False),
            ("markov", True, True, True),
            ("markov", True, False, True),
            ("markov", False, True, False),
            ("markov", False, False, False),
            ("hmm", True, True, True),
            ("hmm", True, False, False),
            ("hmm", False, True, True),
            ("hmm", False, False, False),
            ("none", True, True, False),
            ("none", False, False, False),
        ],
    )
    def test_gate_mode_truth_table(self, mode, markov_bear, hmm_bear, expected_blocked):
        evaluator = RegimeEvaluator(RegimeConfig(gate_mode=mode))
        markov = self._classification(markov_bear)
        hmm = self._classification(hmm_bear)
        result = evaluator.combine(markov, hmm, instrument(), datetime(2026, 1, 1), datetime(2026, 1, 1))
        assert result.is_blocked is expected_blocked

    def test_both_mode_never_blocks_when_one_component_unavailable(self):
        evaluator = RegimeEvaluator(RegimeConfig(gate_mode="both"))
        markov = self._classification(True)
        hmm = self._classification(False, availability=ComponentAvailability.INSUFFICIENT_HISTORY)
        result = evaluator.combine(markov, hmm, instrument(), datetime(2026, 1, 1), datetime(2026, 1, 1))
        assert result.is_blocked is False
        assert any("hmm component unavailable" in w for w in result.warnings)

    def test_either_mode_blocks_on_available_bear_even_if_other_unavailable(self):
        evaluator = RegimeEvaluator(RegimeConfig(gate_mode="either"))
        markov = self._classification(True)
        hmm = self._classification(False, availability=ComponentAvailability.NUMERICAL_FIT_FAILURE)
        result = evaluator.combine(markov, hmm, instrument(), datetime(2026, 1, 1), datetime(2026, 1, 1))
        assert result.is_blocked is True

    def test_both_components_unavailable_never_blocks_regardless_of_mode(self):
        for mode in ("both", "either", "markov", "hmm"):
            evaluator = RegimeEvaluator(RegimeConfig(gate_mode=mode))
            markov = self._classification(False, availability=ComponentAvailability.MISSING_PRICES)
            hmm = self._classification(False, availability=ComponentAvailability.INSUFFICIENT_HISTORY)
            result = evaluator.combine(markov, hmm, instrument(), datetime(2026, 1, 1), datetime(2026, 1, 1))
            assert result.is_blocked is False, mode


class TestMarkovComponentEvaluation:
    def test_bull_trend_classifies_bull_not_bear(self):
        iid = instrument()
        prices, days = _bull_prices(iid)
        evaluator = RegimeEvaluator(RegimeConfig())
        result = evaluator.evaluate_markov_component(iid, prices, datetime.combine(days[-1], datetime.min.time()), datetime.combine(days[-1], datetime.min.time()))
        assert result.is_bear is False
        assert result.availability == ComponentAvailability.OK

    def test_sustained_decline_confirms_bear(self):
        iid = instrument()
        prices, days = _bear_prices(iid)
        evaluator = RegimeEvaluator(RegimeConfig())
        result = evaluator.evaluate_markov_component(iid, prices, datetime.combine(days[-1], datetime.min.time()), datetime.combine(days[-1], datetime.min.time()))
        assert result.is_bear is True
        assert result.classification == RegimeClassification.BEAR

    def test_missing_prices_is_explicit_unavailable(self):
        iid = instrument()
        evaluator = RegimeEvaluator(RegimeConfig())
        result = evaluator.evaluate_markov_component(iid, [], datetime(2026, 1, 1), datetime(2026, 1, 1))
        assert result.availability == ComponentAvailability.MISSING_PRICES
        assert result.is_bear is False

    def test_short_history_is_insufficient_not_bull(self):
        iid = instrument()
        days = _weekdays(date(2026, 1, 5), 10)
        prices = make_daily_series(iid, days, [0.001] * 10, start_price=100.0)
        evaluator = RegimeEvaluator(RegimeConfig())
        result = evaluator.evaluate_markov_component(
            iid, prices, datetime.combine(days[-1], datetime.min.time()), datetime.combine(days[-1], datetime.min.time())
        )
        assert result.availability == ComponentAvailability.INSUFFICIENT_HISTORY
        assert result.classification == RegimeClassification.UNKNOWN


class TestHMMComponentEvaluation:
    def test_bear_state_prediction_yields_bear_classification(self):
        iid = instrument()
        days = _weekdays(date(2020, 1, 3), 300)
        prices = make_daily_series(iid, days, [0.0005] * 300, start_price=100.0)
        evaluator = RegimeEvaluator(RegimeConfig())
        fitter = FakeHMMFitter(state_means=(-0.02, 0.0, 0.02), current_state=0)
        result = evaluator.evaluate_hmm_component(
            iid, prices, datetime.combine(days[-1], datetime.min.time()), datetime.combine(days[-1], datetime.min.time()), fitter
        )
        assert result.classification == RegimeClassification.BEAR
        assert result.is_bear is True

    def test_insufficient_weekly_observations_is_explicit(self):
        iid = instrument()
        days = _weekdays(date(2026, 1, 5), 10)
        prices = make_daily_series(iid, days, [0.001] * 10, start_price=100.0)
        evaluator = RegimeEvaluator(RegimeConfig())
        result = evaluator.evaluate_hmm_component(
            iid, prices, datetime.combine(days[-1], datetime.min.time()), datetime.combine(days[-1], datetime.min.time()), FakeHMMFitter()
        )
        assert result.availability == ComponentAvailability.INSUFFICIENT_HISTORY

    def test_fit_error_is_numerical_fit_failure(self):
        iid = instrument()
        days = _weekdays(date(2020, 1, 3), 300)
        prices = make_daily_series(iid, days, [0.0005] * 300, start_price=100.0)
        evaluator = RegimeEvaluator(RegimeConfig())
        fitter = FakeHMMFitter(error="boom")
        result = evaluator.evaluate_hmm_component(
            iid, prices, datetime.combine(days[-1], datetime.min.time()), datetime.combine(days[-1], datetime.min.time()), fitter
        )
        assert result.availability == ComponentAvailability.NUMERICAL_FIT_FAILURE
        assert result.is_bear is False


class TestCanonicalParity:
    """The critical regression test: single and batch evaluation must agree exactly."""

    def test_single_and_batch_produce_identical_classifications(self):
        iid_a = instrument("AAA")
        iid_b = instrument("BBB")
        prices_a, days = _bear_prices(iid_a)
        prices_b, _ = _bull_prices(iid_b)
        ts = datetime.combine(days[-1], datetime.min.time())

        evaluator = RegimeEvaluator(RegimeConfig())
        fitter = FakeHMMFitter(state_means=(-0.02, 0.0, 0.02), current_state=1)

        single_a = evaluator.evaluate_one(iid_a, prices_a, ts, ts, fitter)
        single_b = evaluator.evaluate_one(iid_b, prices_b, ts, ts, fitter)

        batch = evaluator.evaluate_batch(
            [
                RegimeEvaluationRequest(iid_a, tuple(prices_a), ts, ts),
                RegimeEvaluationRequest(iid_b, tuple(prices_b), ts, ts),
            ],
            fitter,
        )

        assert batch[0].markov.classification == single_a.markov.classification
        assert batch[0].hmm.classification == single_a.hmm.classification
        assert batch[0].is_blocked == single_a.is_blocked
        assert batch[1].markov.classification == single_b.markov.classification
        assert batch[1].hmm.classification == single_b.hmm.classification
        assert batch[1].is_blocked == single_b.is_blocked

    def test_batch_preserves_input_order(self):
        iid_a = instrument("AAA")
        iid_b = instrument("BBB")
        prices_a, days = _bull_prices(iid_a)
        prices_b, _ = _bull_prices(iid_b)
        ts = datetime.combine(days[-1], datetime.min.time())
        evaluator = RegimeEvaluator(RegimeConfig())
        fitter = FakeHMMFitter()
        batch = evaluator.evaluate_batch(
            [
                RegimeEvaluationRequest(iid_b, tuple(prices_b), ts, ts),
                RegimeEvaluationRequest(iid_a, tuple(prices_a), ts, ts),
            ],
            fitter,
        )
        assert [r.instrument_id.symbol for r in batch] == ["BBB", "AAA"]


class TestRegimeResultSerialization:
    def _make_result(self):
        iid = instrument()
        prices, days = _bear_prices(iid)
        ts = datetime.combine(days[-1], datetime.min.time())
        evaluator = RegimeEvaluator(RegimeConfig())
        return evaluator.evaluate_one(iid, prices, ts, ts, FakeHMMFitter(current_state=0))

    def test_combined_result_round_trip(self):
        result = self._make_result()
        restored = RegimeResult.from_dict(result.to_dict())
        assert restored.instrument_id == result.instrument_id
        assert restored.is_blocked == result.is_blocked
        assert restored.gate_mode == result.gate_mode
        assert restored.markov.classification == result.markov.classification
        assert restored.hmm.classification == result.hmm.classification

    def test_component_result_round_trip(self):
        result = self._make_result()
        restored = ComponentClassification.from_dict(result.markov.to_dict())
        assert restored.classification == result.markov.classification
        assert restored.windows == result.markov.windows

    def test_window_result_round_trip_via_component(self):
        result = self._make_result()
        assert len(result.markov.windows) == 3
        restored = ComponentClassification.from_dict(result.markov.to_dict())
        for original, restored_window in zip(result.markov.windows, restored.windows):
            assert original.window_days == restored_window.window_days
            assert original.bear_confirmed == restored_window.bear_confirmed

    def test_audit_order_preserved_through_round_trip(self):
        result = self._make_result()
        restored = RegimeResult.from_dict(result.to_dict())
        assert [r.stage for r in restored.audit_trail] == [r.stage for r in result.audit_trail]
        assert [r.stage for r in restored.markov.audit_trail] == [
            r.stage for r in result.markov.audit_trail
        ]

    def test_timestamp_and_enum_preservation(self):
        result = self._make_result()
        as_dict = result.to_dict()
        assert as_dict["markov"]["classification"] == result.markov.classification.value
        assert as_dict["evaluation_timestamp"] == result.evaluation_timestamp.isoformat()

    def test_deterministic_output(self):
        result = self._make_result()
        assert result.to_dict() == result.to_dict()
