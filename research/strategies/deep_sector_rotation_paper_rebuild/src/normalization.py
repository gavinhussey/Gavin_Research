"""Zero-mean/unit-variance normalization of non-target inputs.

Source: paper p.3, §2.1, last line ("All non-target data were normalized to
have zero mean and unit variance."). See ../docs/paper_source_audit.md #12.

Hard constraint (non-negotiable regardless of which normalization-scope
decision is eventually chosen): normalization statistics must never be
computed using data at or after the point being normalized/predicted. This
module enforces that mechanically via the ``fit_on`` cutoff (generic
mechanics below) and via the calendar-year training-window filter in
``fit_annual_price_scaler`` (paper-faithful path).

DECISION_REQUIRED_NORMALIZATION_SCOPE is RESOLVED (see
../decisions/paper_decision_register.json): **per-ETF (column-wise)
z-score**, fit ONCE per annual trading model on that model's initial
two-year training history, then FROZEN for the entire corresponding
trading year -- no weekly refit, no expanding/rolling-window update. A
new annual model gets its own newly fit scaler. This is a USER-RESOLVED
reconstruction decision (the paper states the transform but not its
scope/cadence) -- see ``AnnualPriceScaler`` / ``fit_annual_price_scaler``
below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PAPER_UNIVERSE, assert_canonical_order
from .training_schedule import AnnualScheduler

# sklearn StandardScaler-equivalent population variance (ddof=0). This
# matches the pre-existing generic fit_normalization() convention below
# (numpy .std() default is also ddof=0) -- kept identical rather than
# introducing a second, inconsistent convention.
DDOF = 0

# Below this, a training window's per-ETF std is treated as zero/negligible
# and fitting raises rather than silently dividing by (near-)zero.
_ZERO_VARIANCE_EPS = 1e-10


class ZeroVarianceTrainingWindowError(ValueError):
    """Raised when an ETF's initial two-year training window has zero or
    numerically negligible price variance -- fitting a z-score scaler on
    it would silently divide by (near-)zero. Real ETF price history should
    make this extremely unlikely; if it ever fires, it indicates a genuine
    data-integrity problem (e.g. a stale/flat price feed), not a case to
    paper over with a default sigma.
    """

    def __init__(self, trading_year: int, ticker: str, sigma: float):
        self.trading_year = trading_year
        self.ticker = ticker
        self.sigma = sigma
        super().__init__(
            f"Zero/negligible variance in the initial two-year training "
            f"window for annual model {trading_year}, ticker {ticker!r}: "
            f"sigma={sigma!r}. Refusing to divide by (near-)zero; this is "
            f"a data-integrity problem, not a case for a silent sigma=1 "
            f"fallback."
        )


@dataclass(frozen=True)
class AnnualPriceScaler:
    """Frozen per-ETF z-score scaler for one annual trading model.

    Fit once, on the annual model's initial two-year training history
    only, then used unchanged (``transform``) for every weekly prediction
    and weekly update within that trading year -- see module docstring.
    Being a frozen dataclass, it cannot be mutated/refit in place; a new
    trading year gets an entirely new ``AnnualPriceScaler`` instance via
    ``fit_annual_price_scaler``.
    """

    trading_year: int
    training_start: pd.Timestamp
    training_end: pd.Timestamp  # exclusive
    ticker_order: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    price_field: str
    ddof: int
    n_training_weeks: int

    def transform(self, weekly_price_matrix: pd.DataFrame) -> np.ndarray:
        """Z[s,t] = (AdjustedClose[s,t] - mu[s,Y]) / sigma[s,Y], per ETF.

        ``weekly_price_matrix`` columns must be in PAPER_UNIVERSE canonical
        order (or a subset/reordering thereof matching ``self.ticker_order``
        exactly) -- no implicit reindexing/fabrication.
        """
        if tuple(weekly_price_matrix.columns) != self.ticker_order:
            raise AssertionError(
                f"weekly_price_matrix columns {tuple(weekly_price_matrix.columns)} "
                f"do not match this scaler's ticker_order {self.ticker_order}."
            )
        values = weekly_price_matrix.to_numpy(dtype=float)
        return (values - self.means) / self.stds

    def audit_rows(self) -> list[dict]:
        """One row per ticker, per the task brief's canonical audit schema."""
        return [
            {
                "trading_year": self.trading_year,
                "training_start_date": self.training_start.date().isoformat(),
                "training_end_date": self.training_end.date().isoformat(),
                "ticker": ticker,
                "mean": float(mean),
                "std": float(std),
                "number_of_training_weeks": self.n_training_weeks,
                "price_field": "Adjusted Close" if self.price_field == "adjusted_close" else self.price_field,
                "normalization": "zscore",
                "frozen": True,
            }
            for ticker, mean, std in zip(self.ticker_order, self.means, self.stds)
        ]


def fit_annual_price_scaler(
    weekly_price_matrix: pd.DataFrame,
    trading_year: int,
    price_field: str = "adjusted_close",
    ddof: int = DDOF,
    scheduler: AnnualScheduler | None = None,
) -> AnnualPriceScaler:
    """Fit the frozen per-ETF annual scaler for ``trading_year``.

    Training window = the annual model's own initial two-year training
    history, i.e. calendar years [trading_year - 2, trading_year) --
    reusing ``AnnualScheduler.training_window_for_year`` (already EXPLICIT,
    p.3 "previous 2 years") as the single source of truth for that window,
    rather than re-deriving "2 years" here.

    ``weekly_price_matrix`` must be date-indexed (index = each week's
    ``model_cutoff``, see tensors.build_weekly_price_matrix) with columns
    in PAPER_UNIVERSE canonical order. Only rows whose index date falls in
    [training_start, training_end) are used -- no week from trading_year
    itself, no later year, and no target-week information can influence
    mu/sigma (hard-asserted below).
    """
    assert_canonical_order(tuple(weekly_price_matrix.columns))
    scheduler = scheduler or AnnualScheduler()
    start_year, end_year = scheduler.training_window_for_year(trading_year)
    training_start = pd.Timestamp(year=start_year, month=1, day=1)
    training_end = pd.Timestamp(year=end_year, month=1, day=1)  # exclusive

    index = pd.DatetimeIndex(weekly_price_matrix.index)
    mask = (index >= training_start) & (index < training_end)
    training_slice = weekly_price_matrix.loc[mask]

    # Hard no-lookahead assertions: nothing at/after training_end, nothing
    # before training_start, made it into the fit.
    assert bool((pd.DatetimeIndex(training_slice.index) < training_end).all()), (
        f"Lookahead violation: a row at/after training_end={training_end.date()} "
        f"leaked into the {trading_year} annual scaler fit."
    )
    assert bool((pd.DatetimeIndex(training_slice.index) >= training_start).all()), (
        f"A row before training_start={training_start.date()} leaked into the "
        f"{trading_year} annual scaler fit."
    )

    if len(training_slice) == 0:
        raise ValueError(
            f"No weekly observations found in the initial two-year training "
            f"window [{training_start.date()}, {training_end.date()}) for "
            f"annual model {trading_year}; cannot fit AnnualPriceScaler."
        )

    means = training_slice.mean(axis=0).to_numpy(dtype=float)
    stds = training_slice.std(axis=0, ddof=ddof).to_numpy(dtype=float)

    for ticker, sigma in zip(weekly_price_matrix.columns, stds):
        if not np.isfinite(sigma) or sigma <= _ZERO_VARIANCE_EPS:
            raise ZeroVarianceTrainingWindowError(trading_year, ticker, float(sigma))

    return AnnualPriceScaler(
        trading_year=trading_year,
        training_start=training_start,
        training_end=training_end,
        ticker_order=tuple(weekly_price_matrix.columns),
        means=means,
        stds=stds,
        price_field=price_field,
        ddof=ddof,
        n_training_weeks=len(training_slice),
    )


# ---------------------------------------------------------------------------
# Generic normalization mechanics (pre-existing, kept for the lookahead-guard
# demonstration and for any future non-annual-scaler use). Not the
# paper-faithful entry point -- see AnnualPriceScaler / fit_annual_price_scaler
# above for that.
# ---------------------------------------------------------------------------


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
