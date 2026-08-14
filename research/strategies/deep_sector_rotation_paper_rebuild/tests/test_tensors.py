"""Tensor chronology, shape, and no-lookahead tests -- task brief test
requirements #3, #4, #5."""
import numpy as np
import pandas as pd
import pytest

from src.data import PAPER_UNIVERSE
from src.tensors import build_paper_tensor, build_tensor
from src.decisions import PaperDecisionRequiredError


def _panel(n_weeks=20):
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(size=(n_weeks, 11)), columns=list(PAPER_UNIVERSE))


@pytest.mark.parametrize("n", [3, 5, 8])
def test_tensor_shape_parameterized_by_n(n):
    panel = _panel()
    x = build_tensor(panel, n=n, t_index=15, price_field="close", include_volume=False)
    assert x.shape == (n, 11)


def test_tensor_chronology_most_recent_row_last():
    panel = _panel()
    n = 5
    t_index = 10
    x = build_tensor(panel, n=n, t_index=t_index, price_field="close", include_volume=False)
    expected_last_row = panel.iloc[t_index].to_numpy()
    np.testing.assert_allclose(x[-1], expected_last_row)
    expected_first_row = panel.iloc[t_index - n + 1].to_numpy()
    np.testing.assert_allclose(x[0], expected_first_row)


def test_no_future_observations_enter_tensor():
    """Rows after t_index must never appear in X_t (no lookahead)."""
    panel = _panel()
    n = 5
    t_index = 10
    x = build_tensor(panel, n=n, t_index=t_index, price_field="close", include_volume=False)
    future_rows = panel.iloc[t_index + 1 :].to_numpy()
    for future_row in future_rows:
        for tensor_row in x:
            assert not np.allclose(tensor_row, future_row)


def test_insufficient_history_raises_rather_than_substitutes():
    panel = _panel()
    with pytest.raises(ValueError):
        build_tensor(panel, n=20, t_index=5, price_field="close", include_volume=False)


def test_volume_doubles_feature_width():
    panel = _panel()
    vol_panel = _panel()
    x_price_only = build_tensor(panel, n=5, t_index=10, price_field="close", include_volume=False)
    x_with_volume = build_tensor(
        panel, n=5, t_index=10, price_field="close", include_volume=True, volume_panel=vol_panel
    )
    assert x_price_only.shape[1] == 11
    assert x_with_volume.shape[1] == 22


def test_paper_tensor_blocked_on_lookback_n_decision():
    panel = _panel()
    with pytest.raises(PaperDecisionRequiredError) as exc_info:
        build_paper_tensor(panel, t_index=10)
    assert exc_info.value.decision_id == "DECISION_REQUIRED_LOOKBACK_N"
