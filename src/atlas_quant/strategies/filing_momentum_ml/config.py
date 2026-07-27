"""Filing Momentum ML — typed, validated strategy and model configuration.

Every default value here is taken directly from report_current.html
(source hash below) as the authoritative spec, cross-checked against a
separate legacy prototype repository's code (settings.py/ml_scorer.py,
read there only as supporting reference, not copied) where noted. Where
the report and that legacy code conflicted (ML_THRESHOLD/MIN_TRAIN_Q
duplicated across its settings.py and ml_scorer.py; fcf_mode default vs.
its on-disk feature-cache mismatch), this module is the single source of
truth going forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from atlas_quant.config.identity import compute_config_identity

STRATEGY_ID = "filing_momentum_ml"
DISPLAY_NAME = "Filing Momentum ML"

# Bumped whenever this module's formulas, defaults, or schema change in a
# way that could alter results. Not the same as the platform version.
STRATEGY_VERSION = "0.1.0"

# sha256 of ~/Downloads/report_current.html at the time this config was
# written, so any future drift between this module and the report it was
# derived from is detectable rather than assumed away.
SOURCE_REPORT_SHA256 = (
    "c985c6ed2f85d4cfa7e4ea13449b5e05d571b2b004935eb997c42a975f787295"
)

# Bumped whenever the 17-feature schema (§3 of the report) changes shape —
# independent of STRATEGY_VERSION, since a cache built under one feature
# schema is never valid input for a model expecting a different one.
FEATURE_SCHEMA_VERSION = "1"

FcfMode = Literal["ratio", "raw"]
RegimeGateMode = Literal["both", "either", "markov", "hmm", "none"]

_VALID_FCF_MODES: tuple[FcfMode, ...] = ("ratio", "raw")
_VALID_REGIME_GATE_MODES: tuple[RegimeGateMode, ...] = (
    "both",
    "either",
    "markov",
    "hmm",
    "none",
)


@dataclass(frozen=True, slots=True)
class FilingMomentumModelConfig:
    """HistGradientBoostingClassifier hyperparameters (report §4.5)."""

    max_iter: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.1
    class_weight: str = "balanced"
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter!r}")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {self.max_depth!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if self.max_leaf_nodes < 2:
            raise ValueError(f"max_leaf_nodes must be >= 2, got {self.max_leaf_nodes!r}")
        if self.min_samples_leaf <= 0:
            raise ValueError(
                f"min_samples_leaf must be > 0, got {self.min_samples_leaf!r}"
            )
        if self.l2_regularization < 0:
            raise ValueError(
                f"l2_regularization must be >= 0, got {self.l2_regularization!r}"
            )

    def identity(self) -> str:
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class FilingMomentumMLConfig:
    """Filing Momentum ML's full strategy-level configuration.

    Field-by-field report provenance:

    - ``ml_threshold`` = 0.35 — report §4.3 (``ML_THRESHOLD``)
    - ``ml_train_years`` = 3 — report §4.4 (``ML_TRAIN_YEARS``)
    - ``min_train_quarters`` = 8 — report §4.4 (``MIN_TRAIN_Q``)
    - ``n_winners`` = 10 — report §4.2 (``N_WINNERS``)
    - ``max_positions`` = 10, ``min_positions`` = 3 — report §5.4
    - ``deployable_pct`` = 0.95 — report §5.3 (``ML_DEPLOYABLE_PCT``)
    - ``return_cap`` = 0.50 — report §5.5 (``RETURN_CAP``)
    - ``earnings_lag_days`` = 42 — report §5.5
    - ``fcf_mode`` = "ratio" — report §3.1, the stated production default
    - ``exclude_sectors`` = ("Materials",) — report §5.2
    - ``fallback_tickers`` = ("SPY", "VGT"), ``fallback_dynamic_weight`` =
      True, ``fallback_lookback_quarters`` = 12 — report §5.4
    - ``regime_gate_mode`` = "both" — report §5.1/§5b.1 (the documented
      live default; "either"/"markov"/"hmm"/"none" are also valid,
      real, exercised modes — exposed here as a genuine config value
      rather than hardcoded, per the Stage 1 conflict analysis)
    - ``markov_years`` = 3 — legacy prototype's settings.py ``MARKOV_YEARS``
      (not contradicted by the report; used by the regime gate, Stage 4)

    ``strategy_budget_pct`` is new relative to the report: the report
    assumed 100% of portfolio capital and had no concept of a "strategy
    budget." Default 1.0 reproduces that assumption for a standalone
    backtest; a multi-strategy allocator (Stage 7+) is expected to override
    it per-run, not by editing this default.
    """

    strategy_id: str = STRATEGY_ID
    universe_id: str = "sp500_nasdaq100_dedup"

    fcf_mode: FcfMode = "ratio"
    ml_threshold: float = 0.35
    ml_train_years: int = 3
    min_train_quarters: int = 8
    n_winners: int = 10

    max_positions: int = 10
    min_positions: int = 3
    deployable_pct: float = 0.95
    return_cap: float = 0.50
    earnings_lag_days: int = 42

    exclude_sectors: tuple[str, ...] = ("Materials",)

    fallback_tickers: tuple[str, ...] = ("SPY", "VGT")
    fallback_dynamic_weight: bool = True
    fallback_lookback_quarters: int = 12

    regime_gate_mode: RegimeGateMode = "both"
    markov_years: int = 3

    strategy_budget_pct: float = 1.0

    model: FilingMomentumModelConfig = field(default_factory=FilingMomentumModelConfig)

    def __post_init__(self) -> None:
        if not (0.0 < self.ml_threshold < 1.0):
            raise ValueError(
                f"ml_threshold must be within (0.0, 1.0), got {self.ml_threshold!r}"
            )
        if self.ml_train_years <= 0:
            raise ValueError(
                f"ml_train_years must be > 0, got {self.ml_train_years!r}"
            )
        if self.min_train_quarters <= 0:
            raise ValueError(
                f"min_train_quarters must be > 0, got {self.min_train_quarters!r}"
            )
        if self.n_winners <= 0:
            raise ValueError(f"n_winners must be > 0, got {self.n_winners!r}")
        if self.max_positions < 1:
            raise ValueError(
                f"max_positions must be >= 1, got {self.max_positions!r}"
            )
        if self.min_positions < 1:
            raise ValueError(
                f"min_positions must be >= 1, got {self.min_positions!r}"
            )
        if self.min_positions > self.max_positions:
            raise ValueError(
                "min_positions cannot exceed max_positions, got "
                f"min_positions={self.min_positions!r}, max_positions={self.max_positions!r}"
            )
        if not (0.0 < self.deployable_pct <= 1.0):
            raise ValueError(
                f"deployable_pct must be within (0.0, 1.0], got {self.deployable_pct!r}"
            )
        if not (0.0 < self.return_cap <= 1.0):
            raise ValueError(
                f"return_cap must be within (0.0, 1.0], got {self.return_cap!r}"
            )
        if self.earnings_lag_days < 0:
            raise ValueError(
                f"earnings_lag_days must be >= 0, got {self.earnings_lag_days!r}"
            )
        if self.fcf_mode not in _VALID_FCF_MODES:
            raise ValueError(
                f"fcf_mode must be one of {_VALID_FCF_MODES}, got {self.fcf_mode!r}"
            )
        if self.regime_gate_mode not in _VALID_REGIME_GATE_MODES:
            raise ValueError(
                f"regime_gate_mode must be one of {_VALID_REGIME_GATE_MODES}, "
                f"got {self.regime_gate_mode!r}"
            )
        if self.fallback_lookback_quarters <= 0:
            raise ValueError(
                "fallback_lookback_quarters must be > 0, got "
                f"{self.fallback_lookback_quarters!r}"
            )
        if self.markov_years <= 0:
            raise ValueError(f"markov_years must be > 0, got {self.markov_years!r}")
        if not (0.0 <= self.strategy_budget_pct <= 1.0):
            raise ValueError(
                "strategy_budget_pct must be within [0.0, 1.0], got "
                f"{self.strategy_budget_pct!r}"
            )

    def identity(self) -> str:
        """Deterministic identity of this config's resolved values.

        Two configs with the same field values (including the nested model
        config) always produce the same identity; changing any field that
        could alter results changes it.
        """
        return compute_config_identity(self)


@dataclass(frozen=True, slots=True)
class FeatureCacheIdentity:
    """Identity metadata a Filing Momentum ML feature cache must record.

    This is a schema, not a cache implementation — Stage 3 will build the
    actual cache writer/reader against this shape. It exists now so the
    identity contract is fixed before any cache-writing code is written,
    directly addressing the fcf_mode/cache-file mismatch found in the
    legacy prototype's feature cache (Stage 1 conflict analysis, item C2):
    a cache missing this metadata cannot be safely reused across
    configurations.
    """

    strategy_id: str
    strategy_version: str
    feature_schema_version: str
    fcf_mode: FcfMode
    train_years: int
    min_train_quarters: int
    model_config_identity: str
    universe_id: str
    data_cutoff: date
    created_at: datetime

    @classmethod
    def compute(
        cls,
        config: FilingMomentumMLConfig,
        data_cutoff: date,
        created_at: datetime,
    ) -> "FeatureCacheIdentity":
        return cls(
            strategy_id=config.strategy_id,
            strategy_version=STRATEGY_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            fcf_mode=config.fcf_mode,
            train_years=config.ml_train_years,
            min_train_quarters=config.min_train_quarters,
            model_config_identity=config.model.identity(),
            universe_id=config.universe_id,
            data_cutoff=data_cutoff,
            created_at=created_at,
        )

    def cache_key(self) -> str:
        """A compact key suitable for a cache filename/dict key.

        Deliberately excludes ``created_at`` (a timestamp shouldn't be part
        of a lookup key) and ``data_cutoff`` is included because caches for
        different cutoffs are not interchangeable under strict
        point-in-time rules.
        """
        return compute_config_identity(
            {
                "strategy_id": self.strategy_id,
                "strategy_version": self.strategy_version,
                "feature_schema_version": self.feature_schema_version,
                "fcf_mode": self.fcf_mode,
                "train_years": self.train_years,
                "min_train_quarters": self.min_train_quarters,
                "model_config_identity": self.model_config_identity,
                "universe_id": self.universe_id,
                "data_cutoff": self.data_cutoff,
            }
        )
