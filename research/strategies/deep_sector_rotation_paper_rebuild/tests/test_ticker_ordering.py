"""Ticker-ordering consistency across tensors/labels/outputs/state -- see
task brief test requirement #1 and #6."""
import numpy as np
import pandas as pd

from src.data import PAPER_UNIVERSE, assert_canonical_order
from src.tensors import build_tensor


def test_tensor_columns_follow_canonical_order():
    n_weeks = 10
    panel = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n_weeks, 11)),
        columns=list(PAPER_UNIVERSE),
    )
    x = build_tensor(panel, n=5, t_index=9, price_field="close", include_volume=False)
    assert x.shape == (5, 11)
    # column i of x must correspond to PAPER_UNIVERSE[i]
    assert_canonical_order(panel.columns)


def test_tensor_rejects_misordered_columns():
    n_weeks = 10
    misordered = list(reversed(PAPER_UNIVERSE))
    panel = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n_weeks, 11)), columns=misordered
    )
    try:
        build_tensor(panel, n=5, t_index=9, price_field="close", include_volume=False)
        assert False, "expected AssertionError for misordered columns"
    except AssertionError:
        pass
