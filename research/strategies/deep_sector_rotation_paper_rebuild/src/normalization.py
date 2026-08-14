"""Zero-mean/unit-variance normalization of non-target inputs.

Source: paper p.3, §2.1, last line ("All non-target data were normalized to
have zero mean and unit variance."). See ../docs/paper_source_audit.md #12.

Hard constraint (non-negotiable regardless of which normalization-scope
decision is eventually chosen): normalization statistics must never be
computed using data at or after the point being normalized/predicted. This
module enforces that mechanically via the ``fit_on`` cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decisions import require_resolved


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    fit_on_end_index: int  # last row index (inclusive) used to fit these stats


def fit_normalization(data: np.ndarray, fit_on_end_index: int) -> NormalizationStats:
    """Fit zero-mean/unit-variance stats using only rows [0, fit_on_end_index]."""
    if fit_on_end_index < 0 or fit_on_end_index >= len(data):
        raise ValueError("fit_on_end_index out of range")
    fit_slice = data[: fit_on_end_index + 1]
    mean = fit_slice.mean(axis=0)
    std = fit_slice.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid divide-by-zero on constant columns
    return NormalizationStats(mean=mean, std=std, fit_on_end_index=fit_on_end_index)


def apply_normalization(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (data - stats.mean) / stats.std


def assert_no_lookahead(stats: NormalizationStats, predict_row_index: int) -> None:
    """Assert normalization stats were fit strictly before the row being predicted."""
    if stats.fit_on_end_index >= predict_row_index:
        raise AssertionError(
            f"Lookahead violation: normalization stats fit through row "
            f"{stats.fit_on_end_index} but being applied to predict row "
            f"{predict_row_index} (must be fit strictly before the predicted row)."
        )


def fit_paper_normalization(data: np.ndarray, current_week_index: int) -> NormalizationStats:
    """Decision-gated entry point for the paper-faithful normalization scope.

    Blocked until DECISION_REQUIRED_NORMALIZATION_SCOPE is resolved (fit-
    once-per-year vs. weekly-expanding-window vs. weekly-rolling-window are
    all mechanically different and produce different statistics).
    """
    require_resolved(
        "DECISION_REQUIRED_NORMALIZATION_SCOPE",
        required_before="fitting paper-faithful normalization statistics",
    )
