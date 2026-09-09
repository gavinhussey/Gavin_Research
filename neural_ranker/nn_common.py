"""Shared helper for the ``neural_ranker/`` notebook series.

Per ``BUILD_SPEC.md`` §7 this is the *one* shared module the series is
allowed: path setup, provenance, device/seed policy, and (added by the
notebooks that first need them) tensor construction and the per-feature
normalization policy. It holds **no model code, no training loop, and no
thresholds** -- if a formula ever lands here it has outgrown this folder.

Real data only
--------------
Every number the series prints comes from the genuine Bloomberg exports,
read through the strategy's own ``_real_data.py`` helper (the only
sanctioned path). There is no synthetic fallback anywhere: when the raw
exports are absent ``rd.require_data()`` prints an explicit "cannot
produce a genuine result" notice and every computing cell is skipped.
"""

from __future__ import annotations

import math
import platform
import random
import sys
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

#: ``neural_ranker/`` -- where the notebooks and this module live.
NEURAL_RANKER_DIR = Path(__file__).resolve().parent
#: Repository root (``neural_ranker/`` sits directly under it).
REPO_ROOT = NEURAL_RANKER_DIR.parent
#: Home of ``_real_data.py``, the strategy's real-data helper.
STRATEGY_NOTEBOOKS_DIR = (
    REPO_ROOT / "research" / "strategies" / "multi_factor_ranking_ml" / "notebooks"
)

if str(STRATEGY_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_NOTEBOOKS_DIR))

import _real_data as rd  # noqa: E402  (needs the sys.path entry above)

from atlas_quant.domain.identifiers import InstrumentId  # noqa: E402
from atlas_quant.strategies.multi_factor_ranking_ml.cross_sectional import (  # noqa: E402
    apply_cross_sectional_rank_normalization,
)
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import EvaluationCycle  # noqa: E402
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import (  # noqa: E402
    FEATURE_NAMES,
    FeatureObservation,
)
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder  # noqa: E402
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import LabeledObservation  # noqa: E402

# ---------------------------------------------------------------------------
# Feature partition (BUILD_SPEC §4: use exactly this)
# ---------------------------------------------------------------------------

#: Market-wide series, broadcast identically to every instrument in a cycle.
#: Carried ONCE per cycle as ``m (6,)`` -- never tiled to ``(N, 6)``.
MACRO: tuple[str, ...] = (
    "fed_funds_rate", "hy_credit_oas", "ust_10y_yield", "ust_2y_yield", "vix", "yield_curve_10y_2y",
)
#: Labels, fed to embeddings; arithmetic on them is meaningless.
CATEGORICAL: tuple[str, ...] = ("sector_enc", "quarter_num")
#: Stock-level features that must NOT be rank-normalized (BUILD_SPEC §5,
#: rule 1: a measured negative result). Consumed by the NN_02 policy.
NEVER_RANK: tuple[str, ...] = (
    "volatility_20d", "volatility_30d", "volatility_63d", "volatility_90d",
    "vol_20d", "vol_63d", "vol_ratio", "beta",
)
#: Everything else, in ``FEATURE_NAMES`` order -- the columns of ``X``/``mask``.
STOCK_CONTINUOUS: tuple[str, ...] = tuple(f for f in FEATURE_NAMES if f not in MACRO + CATEGORICAL)

#: Fixed sector vocabulary; ``sector_enc`` is already the index into it.
SECTOR_VOCABULARY: tuple[str, ...] = SectorEncoder().consolidated_vocabulary()

# ---------------------------------------------------------------------------
# Normalization policy (BUILD_SPEC §4/§5; verified in NN_02, audited in NN_03)
# ---------------------------------------------------------------------------

#: Stock-level features replaced by their percentile rank within their own
#: cycle's cross-section: exactly the five **raw-level** features (dollars,
#: share counts, a price) whose cross-quarter scale drift is the actual
#: problem. Decided 2026-09-08 by honoring BUILD_SPEC §5 rule 2 over §4's
#: code block, which conflicted with it: growth, margin, ratio, momentum,
#: beta, volatility, and every other already scale-free or differenced
#: feature is passed through untouched, so the network keeps their
#: cross-quarter level and sign (NN_02 measured what ranking would erase).
#:
#: These are the same five names the production feature pipeline already
#: ranks (``config.cross_sectional_rank_features``), so on the cached slice
#: ``normalize_cycle`` is expected to be a no-op -- NN_02 verifies that.
RANK_NORMALIZED: tuple[str, ...] = (
    "market_cap", "volume", "net_debt", "adjusted_net_debt", "analyst_target_price",
)
#: Every other stock-level feature: byte-identical through ``normalize_cycle``.
PASS_THROUGH: tuple[str, ...] = tuple(f for f in STOCK_CONTINUOUS if f not in RANK_NORMALIZED)

#: Reporting-only grouping of the 58 pass-through features (NN_02, NN_04
#: tables). Not a policy: every family is passed through identically.
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "volatility / beta (NEVER_RANK)": NEVER_RANK,
    "growth rates": tuple(f for f in STOCK_CONTINUOUS if f.endswith(("_yoy_growth", "_qoq_growth"))),
    "growth acceleration": tuple(f for f in STOCK_CONTINUOUS if f.endswith("_growth_acceleration")),
    "margins": ("operating_margin", "net_margin", "operating_cash_flow_margin"),
    "margin changes (bps)": tuple(f for f in STOCK_CONTINUOUS if f.endswith("_change_bps")),
    "valuation ratios": ("pe_ratio", "price_to_book", "price_to_sales"),
    "other ratios": (
        "operating_cash_flow_to_net_income", "capex_to_revenue", "capex_to_depreciation",
        "debt_to_equity", "debt_to_assets", "ROA", "ROE",
    ),
    "legacy trends": ("fcf_trend", "roe_trend", "rev_accel", "rev_trend", "om_trend", "nm_trend"),
    "price momentum": ("price_mom_3m", "price_mom_6m", "price_mom_12m"),
}

# ---------------------------------------------------------------------------
# Fixed input transform for the neural encoder (decided 2026-09-08; NN_04)
# ---------------------------------------------------------------------------
#
# Fixed and non-fitted: no scaler is fit globally, no per-cycle statistic is
# computed, no cross-cycle statistic exists. Identity for the five
# RANK_NORMALIZED features (already in [0, 1]); deterministic signed log
# compression for every other stock-level feature, which keeps their
# cross-quarter level and sign while bounding the input scale. No clipping.

#: Columns of ``X`` that enter the encoder unchanged.
INPUT_IDENTITY: tuple[str, ...] = RANK_NORMALIZED
#: Columns of ``X`` that enter the encoder as ``signed_log1p``.
INPUT_SIGNED_LOG: tuple[str, ...] = PASS_THROUGH


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)`` elementwise; ``NaN`` stays ``NaN``."""
    return np.sign(x) * np.log1p(np.abs(x))


def transform_features(X: np.ndarray) -> np.ndarray:
    """Apply the fixed input transform to an ``(N, 63)`` ``X`` (columns in
    ``STOCK_CONTINUOUS`` order). Elementwise, so it reads no other row and no
    other cycle; ``NaN`` is preserved wherever it was."""
    if X.ndim != 2 or X.shape[1] != len(STOCK_CONTINUOUS):
        raise ValueError(f"expected an (N, {len(STOCK_CONTINUOUS)}) matrix, got {X.shape}")
    out = np.array(X, dtype=np.float64, copy=True)
    idx = [STOCK_CONTINUOUS.index(name) for name in INPUT_SIGNED_LOG]
    out[:, idx] = signed_log1p(out[:, idx])
    if not np.array_equal(np.isnan(out), np.isnan(X)):
        raise ValueError("the input transform must never create or remove a NaN")
    return out


@dataclass(frozen=True)
class ModelInput:
    """One cycle's encoder inputs on the device, plus the placeholder count.

    ``X`` is the transformed feature matrix with every missing entry set to
    ``0.0`` -- a **numeric placeholder for the tensor library only, not a
    statistical imputation**: ``mask`` is 1.0 at exactly those entries and
    is concatenated to ``X`` at the encoder input, so the model always knows
    which values were missing. ``n_nan_replaced`` records how many.
    """

    X: "object"
    mask: "object"
    sector: "object"
    quarter: "object"
    m: "object"
    y: "object | None"
    n_nan_replaced: int


def to_model_input(tensors: CycleTensors, device) -> ModelInput:
    """``transform_features`` -> ``NaN`` placeholder under the mask -> device.

    Order matters and is fixed: the mask was computed in
    :func:`build_cycle_tensors` from the untransformed ``X``; the transform
    is applied (``NaN`` preserved); only then are the ``NaN`` entries
    replaced by ``0.0`` for the tensor library. ``m`` enters as-is.
    """
    import torch

    transformed = transform_features(tensors.X)
    missing = np.isnan(transformed)
    if not np.array_equal(missing, tensors.mask):
        raise ValueError("NaN locations after the transform must equal the mask exactly")
    filled = np.where(missing, 0.0, transformed)
    y = None if tensors.y is None else torch.tensor(tensors.y, dtype=torch.int64, device=device)
    return ModelInput(
        X=torch.tensor(filled, dtype=torch.float32, device=device),
        mask=torch.tensor(tensors.mask, dtype=torch.float32, device=device),
        sector=torch.tensor(tensors.sector, dtype=torch.int64, device=device),
        quarter=torch.tensor(tensors.quarter, dtype=torch.int64, device=device),
        m=torch.tensor(tensors.m, dtype=torch.float32, device=device),
        y=y,
        n_nan_replaced=int(missing.sum()),
    )


def normalize_cycle(observations: Sequence[FeatureObservation]) -> tuple[FeatureObservation, ...]:
    """Apply the per-feature policy to exactly ONE cycle's cross-section.

    Delegates the rank transform entirely to the strategy's own
    ``apply_cross_sectional_rank_normalization`` (the only sanctioned
    implementation -- it enforces the single-cycle scope, preserves NaN, and
    records an audit record on every observation). Nothing else is done to
    the observations here: no standardization, no imputation, no clipping.
    """
    return apply_cross_sectional_rank_normalization(observations, RANK_NORMALIZED)

# ---------------------------------------------------------------------------
# Tensor construction (OUTLINE §3.1; BUILD_SPEC §7 row 01)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleTensors:
    """One evaluation cycle's cross-section as the six tensors of OUTLINE §3.1.

    Kept as numpy (float64 / bool / int64) so the data layer is exact;
    :func:`to_torch` casts to float32 for the device (MPS has no float64).

    Missingness is **masked, never imputed**: ``X`` keeps ``NaN`` wherever
    the pipeline reported the feature missing and ``mask`` is ``True``
    there. No value is invented at this layer; the encoder (NN_04) zero-
    fills *under the mask* at input time, so ``(0, mask=1)`` and
    ``(0, mask=0)`` remain distinguishable inputs.
    """

    quarter_start: date
    cutoff: date
    instrument_ids: tuple[InstrumentId, ...]
    X: np.ndarray            # (N, 63) float64, NaN where missing
    mask: np.ndarray         # (N, 63) bool, True where missing
    sector: np.ndarray       # (N,) int64, index into SECTOR_VOCABULARY
    quarter: np.ndarray      # (N,) int64, fiscal quarter 1..4
    m: np.ndarray            # (6,) float64, this cycle's macro vector, stored once
    y: np.ndarray | None     # (N,) int64 relevance grade, None if not yet realized
    label_available_at: datetime | None
    feature_names: tuple[str, ...] = STOCK_CONTINUOUS
    macro_names: tuple[str, ...] = MACRO

    @property
    def n(self) -> int:
        return len(self.instrument_ids)


def _feature_value(features: Mapping[str, float], name: str) -> float:
    value = features.get(name, float("nan"))
    if not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def build_cycle_tensors(
    cycle: EvaluationCycle,
    observations: Sequence[FeatureObservation],
    labeled: Sequence[LabeledObservation] | None,
    *,
    n_relevance_grades: int | None = None,
) -> CycleTensors:
    """Build :class:`CycleTensors` for exactly one cycle's cross-section.

    A pure function of its arguments: it sees one cycle's observations and
    (optionally) that same cycle's realized labels, and nothing else -- it
    cannot read another cycle. Every structural expectation is enforced
    with an explicit ``ValueError`` rather than silently repaired:

    - every observation's ``data_cutoff`` is this cycle's cutoff;
    - instrument ids are unique;
    - ``mask`` agrees exactly, row by row, with the pipeline's own
      ``missing_features`` restricted to the stock-level block;
    - ``sector_enc`` is integer-valued, in vocabulary, and names the
      observation's own ``sector``; ``quarter_num`` is in 1..4;
    - every macro feature has exactly one non-NaN value across the cycle
      (a missing macro value is an error, never filled);
    - when ``labeled`` is given, labels and observations cover the same
      instruments, and ``label_available_at`` is one value strictly after
      the cycle's entry date.
    """
    if n_relevance_grades is None:
        n_relevance_grades = rd.config().n_relevance_grades
    n = len(observations)
    if n == 0:
        raise ValueError(f"cycle {cycle.quarter_start}: no observations")

    bad_cutoff = [o for o in observations if o.data_cutoff.date() != cycle.cutoff]
    if bad_cutoff:
        raise ValueError(
            f"cycle {cycle.quarter_start}: {len(bad_cutoff)} observation(s) carry a "
            f"data_cutoff other than this cycle's {cycle.cutoff} (first: "
            f"{bad_cutoff[0].instrument_id.symbol} @ {bad_cutoff[0].data_cutoff.date()})"
        )
    ids = tuple(o.instrument_id for o in observations)
    if len(set(ids)) != n:
        raise ValueError(f"cycle {cycle.quarter_start}: duplicate instrument ids in the cross-section")

    X = np.array(
        [[_feature_value(o.features, name) for name in STOCK_CONTINUOUS] for o in observations],
        dtype=np.float64,
    )
    mask = np.isnan(X)

    stock_set = set(STOCK_CONTINUOUS)
    for i, o in enumerate(observations):
        reported = tuple(name for name in STOCK_CONTINUOUS if name in set(o.missing_features) & stock_set)
        found = tuple(name for name, is_missing in zip(STOCK_CONTINUOUS, mask[i]) if is_missing)
        if reported != found:
            raise ValueError(
                f"cycle {cycle.quarter_start}: mask disagrees with missing_features for "
                f"{o.instrument_id.symbol}: reported {reported}, found {found}"
            )

    sector = np.empty(n, dtype=np.int64)
    quarter = np.empty(n, dtype=np.int64)
    for i, o in enumerate(observations):
        raw_sector = _feature_value(o.features, "sector_enc")
        raw_quarter = _feature_value(o.features, "quarter_num")
        if math.isnan(raw_sector) or raw_sector != int(raw_sector) or not 0 <= int(raw_sector) < len(SECTOR_VOCABULARY):
            raise ValueError(f"{o.instrument_id.symbol}: sector_enc {raw_sector!r} is not a vocabulary index")
        if SECTOR_VOCABULARY[int(raw_sector)] != o.sector:
            raise ValueError(
                f"{o.instrument_id.symbol}: sector_enc {int(raw_sector)} names "
                f"{SECTOR_VOCABULARY[int(raw_sector)]!r} but the observation says {o.sector!r}"
            )
        if math.isnan(raw_quarter) or raw_quarter != int(raw_quarter) or not 1 <= int(raw_quarter) <= 4:
            raise ValueError(f"{o.instrument_id.symbol}: quarter_num {raw_quarter!r} is not in 1..4")
        sector[i] = int(raw_sector)
        quarter[i] = int(raw_quarter)

    m = np.empty(len(MACRO), dtype=np.float64)
    for j, name in enumerate(MACRO):
        column = np.array([_feature_value(o.features, name) for o in observations], dtype=np.float64)
        if np.isnan(column).any():
            raise ValueError(
                f"cycle {cycle.quarter_start}: macro feature {name!r} is missing on "
                f"{int(np.isnan(column).sum())} row(s) -- cannot produce a genuine macro vector"
            )
        if not np.all(column == column[0]):
            raise ValueError(
                f"cycle {cycle.quarter_start}: macro feature {name!r} is not constant across the "
                f"cross-section ({len(np.unique(column))} distinct values); it must be one value per cycle"
            )
        m[j] = column[0]

    y = None
    label_available_at = None
    if labeled is not None:
        by_id: dict[InstrumentId, LabeledObservation] = {}
        for row in labeled:
            if row.observation.instrument_id in by_id:
                raise ValueError(f"cycle {cycle.quarter_start}: duplicate label for {row.observation.instrument_id.symbol}")
            by_id[row.observation.instrument_id] = row
        if set(by_id) != set(ids):
            raise ValueError(
                f"cycle {cycle.quarter_start}: labels cover {len(by_id)} instruments, "
                f"observations {n}; they must be the same set"
            )
        availability = {row.label_available_at for row in labeled}
        if len(availability) != 1:
            raise ValueError(f"cycle {cycle.quarter_start}: labels carry {len(availability)} distinct label_available_at values")
        label_available_at = next(iter(availability))
        if label_available_at <= datetime.combine(cycle.quarter_start, time.min):
            raise ValueError(
                f"cycle {cycle.quarter_start}: label_available_at {label_available_at} is not after the "
                "cycle's entry date -- a target would be knowable when the ranking is made"
            )
        y = np.empty(n, dtype=np.int64)
        for i, o in enumerate(observations):
            row = by_id[o.instrument_id]
            if (
                row.observation.feature_timestamp != o.feature_timestamp
                or row.observation.strategy_cohort_end != o.strategy_cohort_end
            ):
                raise ValueError(f"{o.instrument_id.symbol}: label belongs to a different cohort than the observation")
            if not 0 <= row.relevance < n_relevance_grades:
                raise ValueError(f"{o.instrument_id.symbol}: relevance {row.relevance} outside 0..{n_relevance_grades - 1}")
            y[i] = row.relevance

    return CycleTensors(
        quarter_start=cycle.quarter_start,
        cutoff=cycle.cutoff,
        instrument_ids=ids,
        X=X,
        mask=mask,
        sector=sector,
        quarter=quarter,
        m=m,
        y=y,
        label_available_at=label_available_at,
    )


def bundle_cycle_tensors(
    bundle: "rd.RealDataSlice",
    transform: Callable[[Sequence[FeatureObservation]], Sequence[FeatureObservation]] | None = None,
) -> dict[date, CycleTensors]:
    """:func:`build_cycle_tensors` for every cycle of the slice, keyed by ``quarter_start``.

    ``transform`` (NN_02's normalization policy) is applied to **one
    cycle's observations at a time** before tensor construction -- it is
    never handed more than a single cross-section, which is what keeps the
    normalization scope point-in-time safe by construction.
    """
    out: dict[date, CycleTensors] = {}
    for cycle in bundle.cycles:
        observations = bundle.feature_results[cycle.quarter_start].observations
        if transform is not None:
            observations = transform(observations)
        out[cycle.quarter_start] = build_cycle_tensors(
            cycle, observations, bundle.labeled_quarters.get(cycle.quarter_start)
        )
    return out


def to_torch(tensors: CycleTensors, device) -> dict[str, "object"]:
    """Cast one cycle's tensors for the device.

    Floats become float32 (MPS has no float64) with ``NaN`` still in place
    under the mask; ``mask`` becomes a 0/1 float32 tensor ready to be
    concatenated to ``X``; ``sector``/``quarter``/``y`` stay int64 for the
    embeddings and the loss. ``y`` is omitted when the cycle has no labels.
    """
    import torch

    out = {
        "X": torch.tensor(tensors.X, dtype=torch.float32, device=device),
        "mask": torch.tensor(tensors.mask, dtype=torch.float32, device=device),
        "sector": torch.tensor(tensors.sector, dtype=torch.int64, device=device),
        "quarter": torch.tensor(tensors.quarter, dtype=torch.int64, device=device),
        "m": torch.tensor(tensors.m, dtype=torch.float32, device=device),
    }
    if tensors.y is not None:
        out["y"] = torch.tensor(tensors.y, dtype=torch.int64, device=device)
    return out

# ---------------------------------------------------------------------------
# Seed policy (BUILD_SPEC §8: five seeds minimum, never a single-run number)
# ---------------------------------------------------------------------------

#: The fixed seed set every training notebook runs over. Any reported
#: number is the mean over these seeds, with the per-seed values shown
#: alongside it. Five is the BUILD_SPEC §8 minimum, not a tuned choice.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU and MPS) identically.

    Not covered, deliberately: Python's per-process string-hash seed
    (fixed at interpreter start, so it cannot be set here). No code in
    this series may therefore depend on the iteration order of a set of
    strings -- sort first.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# Device policy (BUILD_SPEC §1: MPS with a CPU fallback, never assume CUDA)
# ---------------------------------------------------------------------------


def resolve_device():
    """``torch.device("mps")`` when Metal is available, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Provenance banner (BUILD_SPEC §7: cell 1 of every notebook prints this)
# ---------------------------------------------------------------------------


def library_versions() -> dict[str, str]:
    """Version of every library whose behavior could change a result."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "numpy", "pandas", "scipy", "lightgbm"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            versions[name] = f"NOT IMPORTABLE ({type(exc).__name__})"
    return versions


def provenance_banner(notebook: str) -> bool:
    """Print the honest provenance banner and return ``rd.data_available()``.

    Prints, in order: the real-data / data-absent notice from
    ``_real_data``, the SHA-256 identity of every raw CSV, this checkout's
    git commit, the strategy config identity, the seed policy, and the
    resolved torch device. Every notebook's first code cell calls this and
    stores the returned flag; every computing cell is then guarded by
    ``rd.require_data()``.
    """
    import torch

    print(f"neural_ranker / {notebook}")
    print("=" * 78)
    print(rd.banner())
    available = rd.data_available()

    print("SOURCE MANIFEST (SHA-256 of every raw CSV the pipeline reads)")
    print(f"  raw root: {rd.RAW_ROOT}")
    for entry in rd.source_manifest():
        if not entry.present:
            print(f"  {entry.name:<28} ABSENT")
            continue
        print(
            f"  {entry.name:<28} {entry.size_bytes / 1e6:8.1f} MB  "
            f"{entry.modified:%Y-%m-%d %H:%M:%S}  {entry.sha256}"
        )
    print()
    print(f"git commit        : {rd.git_commit() or 'unavailable'}")
    print(f"repo root         : {REPO_ROOT}")
    print(f"config identity   : {rd.config().identity()}")
    print(f"model identity    : {rd.config().model.identity()}  (LightGBM baseline config)")
    print(f"seed policy       : SEEDS={SEEDS}  ({len(SEEDS)} seeds; every reported number is a mean over them)")
    device = resolve_device()
    print(f"torch device      : {device}  (mps available={torch.backends.mps.is_available()})")
    print(f"platform          : {platform.platform()}")
    print("libraries         : " + ", ".join(f"{k}={v}" for k, v in library_versions().items()))
    print("=" * 78)
    return available
