"""Unit tests for atlas_quant.strategies.filing_momentum_ml.regime_config."""

import pytest

from atlas_quant.strategies.filing_momentum_ml.regime_config import RegimeConfig


class TestRegimeConfigDefaults:
    def test_defaults_match_report_current_html(self):
        config = RegimeConfig()
        assert config.gate_mode == "both"  # report §5.1
        assert config.markov_windows == (63, 126, 252)  # report §5b.1 WINDOWS
        assert config.markov_persistence == 5  # report §5b.1 PERSISTENCE
        assert config.markov_min_window_agreement == 2  # report §5b.1
        assert config.markov_threshold_multiplier == 0.5  # report §5b.1
        assert config.markov_threshold_floor == 0.005  # report §5b.1
        assert config.markov_annualization_factor == 252  # report §5b.1
        assert config.markov_volatility_lookback_days == 63  # report §5b.1 sigma_ann window
        assert config.hmm_state_count == 3  # report §5b.2
        assert config.hmm_covariance_type == "diag"  # report §5b.2
        assert config.hmm_n_iter == 200  # report §5b.2
        assert config.hmm_random_state == 42  # report §5b.2
        assert config.hmm_min_weekly_observations == 30  # report §5b.2
        assert config.hmm_weekly_anchor == "W-FRI"  # report §5b.2
        assert config.hmm_volatility_window_weeks == 4  # report §5b.2


class TestRegimeConfigValidation:
    @pytest.mark.parametrize("mode", ["both", "either", "markov", "hmm", "none"])
    def test_accepts_all_documented_gate_modes(self, mode):
        RegimeConfig(gate_mode=mode)

    def test_rejects_unknown_gate_mode(self):
        with pytest.raises(ValueError):
            RegimeConfig(gate_mode="always")  # type: ignore[arg-type]

    def test_rejects_empty_markov_windows(self):
        with pytest.raises(ValueError):
            RegimeConfig(markov_windows=())

    def test_rejects_non_positive_markov_window(self):
        with pytest.raises(ValueError):
            RegimeConfig(markov_windows=(63, 0, 252))

    def test_rejects_non_positive_persistence(self):
        with pytest.raises(ValueError):
            RegimeConfig(markov_persistence=0)

    @pytest.mark.parametrize("agreement", [0, 4])
    def test_rejects_agreement_outside_window_count(self, agreement):
        with pytest.raises(ValueError):
            RegimeConfig(markov_min_window_agreement=agreement)  # only 3 windows by default

    def test_rejects_non_positive_threshold_multiplier(self):
        with pytest.raises(ValueError):
            RegimeConfig(markov_threshold_multiplier=0)

    def test_rejects_negative_threshold_floor(self):
        with pytest.raises(ValueError):
            RegimeConfig(markov_threshold_floor=-0.001)

    def test_rejects_hmm_state_count_below_two(self):
        with pytest.raises(ValueError):
            RegimeConfig(hmm_state_count=1)

    def test_rejects_non_positive_hmm_n_iter(self):
        with pytest.raises(ValueError):
            RegimeConfig(hmm_n_iter=0)

    def test_rejects_non_positive_min_weekly_observations(self):
        with pytest.raises(ValueError):
            RegimeConfig(hmm_min_weekly_observations=0)

    def test_rejects_volatility_window_weeks_of_one(self):
        with pytest.raises(ValueError):
            RegimeConfig(hmm_volatility_window_weeks=1)


class TestRegimeConfigIdentity:
    def test_deterministic_identity(self):
        assert RegimeConfig().identity() == RegimeConfig().identity()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("gate_mode", "either"),
            ("markov_windows", (63, 126)),
            ("markov_persistence", 4),
            ("markov_min_window_agreement", 1),
            ("markov_threshold_multiplier", 0.6),
            ("markov_threshold_floor", 0.01),
            ("markov_annualization_factor", 250),
            ("markov_volatility_lookback_days", 42),
            ("hmm_state_count", 4),
            ("hmm_covariance_type", "full"),
            ("hmm_n_iter", 100),
            ("hmm_random_state", 7),
            ("hmm_min_weekly_observations", 40),
            ("hmm_weekly_anchor", "W-MON"),
            ("hmm_volatility_window_weeks", 8),
        ],
    )
    def test_identity_changes_with_each_meaningful_parameter(self, field, value):
        default = RegimeConfig()
        changed = RegimeConfig(**{field: value})
        assert default.identity() != changed.identity()

    def test_identity_stable_for_equivalent_configs(self):
        a = RegimeConfig(gate_mode="either", markov_persistence=4)
        b = RegimeConfig(gate_mode="either", markov_persistence=4)
        assert a.identity() == b.identity()
