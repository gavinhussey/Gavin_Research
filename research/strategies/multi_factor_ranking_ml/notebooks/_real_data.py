"""Real-data access helper shared by the numbered 00-10 Multi-Factor
Ranking ML research notebooks.

**There is no synthetic fixture data anywhere in this series.** Every
number any of these notebooks prints is computed from the genuine
Bloomberg CSV exports under this repository's (gitignored) raw-data root,
through the same production code paths the CLI and the live runner use.

Because that raw-data root is gitignored, it is simply absent in a fresh
clone. In that case :func:`data_available` returns ``False`` and every
notebook prints :data:`DATA_UNAVAILABLE_MESSAGE` and stops -- it never
falls back to invented numbers. Producing a plausible-looking figure from
made-up inputs would be worse than producing none, and is forbidden
outright by ``CLAUDE.md``.

Bounded slice
-------------
A full-history run (1980-2026) takes minutes. These notebooks are meant to
be read and re-run, so they work on a small, explicitly bounded window of
**real** quarters (:data:`SLICE_START` .. :data:`SLICE_END`) -- a genuine
subset, never a downsample or an approximation.

Slice cache
-----------
Loading and normalizing the raw CSVs takes ~55s (9.2M daily price rows),
and building the slice's features/labels/IC backtest takes ~2min more.
Paying that in all ten notebooks would make the series unusable, so
:func:`slice_bundle` computes it **once** and caches the derived result
under :data:`CACHE_ROOT` (gitignored). The cache is keyed by a fingerprint
of the source CSVs' SHA-256 digests plus every input that could change the
result (strategy config identity, feature schema version, slice bounds),
so a stale cache is rebuilt rather than silently reused. Deleting
:data:`CACHE_ROOT` is always safe.

This is a plain local helper module (Jupyter puts a notebook's own
directory on ``sys.path``) -- not a path hack, and not part of the
installed ``atlas_quant`` package. It imports only production AtlasQuant
code, never anything from the legacy Arnold_Quant repository.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from atlas_quant.backtest.multi_factor_ranking_runner import (
    ICBacktestResult,
    build_feature_results,
    build_labeled_quarters,
    run_ic_backtest,
)
from atlas_quant.data.records import DailyPriceObservation
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_ID,
    STRATEGY_VERSION,
    MultiFactorRankingMLConfig,
)
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import build_lgbm_ranker_estimator
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import (
    EvaluationCycle,
    quarterly_evaluation_cycles,
)
from atlas_quant.strategies.multi_factor_ranking_ml.feature_pipeline import FeaturePipelineResult
from atlas_quant.strategies.multi_factor_ranking_ml.production.orchestration import LoadedData, load_raw_data
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import LabeledObservation

REPO_ROOT = Path(__file__).resolve().parents[4]

#: The genuine Bloomberg CSV exports. Gitignored -- absent in a fresh
#: clone. ``ATLASQUANT_MFR_RAW_ROOT`` repoints this at a copy of the same
#: exports held elsewhere; it is read once at import, so a notebook picks
#: it up from the environment its kernel was launched in. It only ever
#: selects *which real export directory* to read -- there is no setting
#: that makes this module produce data it did not read from disk.
RAW_ROOT = Path(os.environ.get("ATLASQUANT_MFR_RAW_ROOT") or (REPO_ROOT / "data" / "raw" / STRATEGY_ID))

#: Where the derived slice cache lives. Gitignored; safe to delete.
CACHE_ROOT = REPO_ROOT / "data" / "cache" / STRATEGY_ID / "notebook_slice"

#: Required by ``load_raw_data``; cannot be inferred from Close_Price.csv
#: itself (see ``acquisition/close_price.py``).
PRICE_CONVENTION = "split_dividend_adjusted"

#: The CSVs the pipeline reads. Only the first is strictly required.
SOURCE_FILES: tuple[str, ...] = (
    "fundamentals_quarterly.csv",
    "Close_Price.csv",
    "Daily_Macro.csv",
    "Fed_Funds_Rate.csv",
)

#: The bounded window of real quarterly evaluation cycles these notebooks
#: work on. 14 cycles: with ``ml_train_years=3``/``min_train_quarters=8``
#: the first trainable cycle is the 10th (a target quarter needs 8 prior
#: quarters whose labels were already knowable at its own cutoff), so this
#: yields 5 trained cycles and 4 with a measurable forward IC.
SLICE_START = date(2020, 10, 1)
SLICE_END = date(2024, 1, 1)

#: Notebook 08 re-runs the IC backtest live over just the last few cycles
#: (reusing the cached, slice-wide features/labels) so it stays fast while
#: still exercising the real orchestration path end to end.
BACKTEST_DEMO_CYCLE_COUNT = 3

#: Daily prices are cached only over the window notebook 08's live re-run
#: can actually reach (entry at the third-from-last cycle, exit at the
#: last). Caching all 9.2M rows would make the cache enormous for no gain;
#: the slice-wide ``ICBacktestResult`` in the cache was computed from the
#: complete, untrimmed price history.
PRICE_CACHE_START = date(2023, 5, 1)
PRICE_CACHE_END = date(2024, 3, 1)

DATA_UNAVAILABLE_MESSAGE = f"""\
REAL DATA NOT PRESENT -- CANNOT PRODUCE A GENUINE RESULT.

This notebook computes every figure it shows from the real Bloomberg CSV
exports for {STRATEGY_ID}, which are gitignored and are not
present in this checkout:

    {RAW_ROOT}

Nothing is computed below. This notebook deliberately does NOT substitute
synthetic or approximated data to fill the gap: a plausible-looking number
derived from invented inputs is not a result, and presenting one as though
it were is forbidden by this repository's CLAUDE.md.

To run this notebook, restore the raw exports at the path above (see
docs/data_provenance_manifest.md) and re-run from the top.
"""

_REAL_DATA_BANNER = f"""\
DATA MODE: REAL DATA.

Source: Bloomberg CSV exports under this repository's gitignored raw-data
root for {STRATEGY_ID} -- 1,520 tickers, quarterly fundamentals
from 1980 onward, plus daily closes and macro series.

Every number below is computed from that real data through production
AtlasQuant code. No synthetic, simulated, imputed, or approximated values
are used anywhere in this notebook series.

Bounded slice: quarterly evaluation cycles {SLICE_START.isoformat()} .. {SLICE_END.isoformat()}
(a genuine subset of the history, chosen so each notebook runs quickly --
not a downsample or an approximation of a longer run).
"""


def data_available() -> bool:
    """True when the real raw exports are present and readable."""
    return RAW_ROOT.is_dir() and (RAW_ROOT / SOURCE_FILES[0]).is_file()


def banner() -> str:
    """The honest provenance banner every notebook prints in its first cell."""
    if not data_available():
        return DATA_UNAVAILABLE_MESSAGE
    return _REAL_DATA_BANNER


def require_data() -> bool:
    """Print the unavailability notice and return ``False`` when the real
    data is absent; return ``True`` when it is present.

    Notebooks guard every computing cell with this so that a checkout
    without the data still executes top to bottom -- printing an explicit,
    honest "no genuine result possible" notice instead of a fabricated one.
    """
    if data_available():
        return True
    print(DATA_UNAVAILABLE_MESSAGE)
    return False


def config() -> MultiFactorRankingMLConfig:
    """The strategy config these notebooks run under (all defaults)."""
    return MultiFactorRankingMLConfig()


def slice_cycles() -> tuple[EvaluationCycle, ...]:
    """The bounded slice's quarterly evaluation cycles, ascending."""
    return quarterly_evaluation_cycles(SLICE_START, SLICE_END)


def git_commit() -> str | None:
    """This checkout's HEAD commit, or ``None`` if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One raw CSV's identity, for the provenance manifest in notebook 00."""

    name: str
    present: bool
    size_bytes: int | None
    modified: datetime | None
    sha256: str | None


@lru_cache(maxsize=1)
def source_manifest() -> tuple[SourceFile, ...]:
    """SHA-256 identity of every raw CSV this pipeline reads.

    Hashing ~110MB takes a second or two; cached per process.
    """
    entries: list[SourceFile] = []
    for name in SOURCE_FILES:
        path = RAW_ROOT / name
        if not path.is_file():
            entries.append(SourceFile(name, False, None, None, None))
            continue
        stat = path.stat()
        entries.append(
            SourceFile(
                name=name,
                present=True,
                size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                sha256=_sha256(path),
            )
        )
    return tuple(entries)


@lru_cache(maxsize=1)
def load_raw() -> LoadedData:
    """Load and normalize the complete real raw dataset (~55s, ~9.2M price rows).

    Notebooks 02 and 03 use this directly, because inspecting the real
    load/normalization boundary is exactly what they are for. Every other
    notebook goes through :func:`slice_bundle`, which is cached.
    """
    if not data_available():
        raise FileNotFoundError(DATA_UNAVAILABLE_MESSAGE)
    return load_raw_data(RAW_ROOT, price_convention=PRICE_CONVENTION)


def _fingerprint() -> str:
    """Identity of everything that could change the cached slice's contents."""
    parts = [
        f"strategy={STRATEGY_ID}",
        f"strategy_version={STRATEGY_VERSION}",
        f"feature_schema_version={FEATURE_SCHEMA_VERSION}",
        f"config_identity={config().identity()}",
        f"slice={SLICE_START.isoformat()}..{SLICE_END.isoformat()}",
        f"prices={PRICE_CACHE_START.isoformat()}..{PRICE_CACHE_END.isoformat()}",
        f"price_convention={PRICE_CONVENTION}",
        "cache_layout=2",
    ]
    parts += [f"{f.name}={f.sha256}:{f.size_bytes}" for f in source_manifest()]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class RealDataSlice:
    """Everything the notebooks need from the bounded real-data slice.

    Every field was computed from the genuine raw CSVs by
    :func:`_build_slice`; nothing here is synthetic, and nothing is an
    approximation of a value that was too expensive to compute.
    """

    fingerprint: str
    built_at: datetime
    source_manifest: tuple[SourceFile, ...]
    cycles: tuple[EvaluationCycle, ...]
    universe: tuple[InstrumentId, ...]
    #: Row counts of the full (untrimmed) real dataset the slice came from.
    total_fundamentals_rows: int
    total_price_rows: int
    load_issue_count: int
    feature_results: dict[date, FeaturePipelineResult]
    labeled_quarters: dict[date, tuple[LabeledObservation, ...]]
    #: Real daily closes, trimmed to :data:`PRICE_CACHE_START` ..
    #: :data:`PRICE_CACHE_END` -- enough for notebook 08's live re-run.
    prices_by_instrument: dict[InstrumentId, tuple[DailyPriceObservation, ...]]
    #: The slice-wide IC backtest, computed over the COMPLETE price
    #: history (not the trimmed copy above).
    ic_result: ICBacktestResult

    @property
    def demo_cycles(self) -> tuple[EvaluationCycle, ...]:
        """The last few cycles notebook 08 re-runs the backtest over."""
        return self.cycles[-BACKTEST_DEMO_CYCLE_COUNT:]


def _build_slice() -> RealDataSlice:
    cfg = config()
    cycles = slice_cycles()
    data = load_raw()

    feature_results = build_feature_results(
        config=cfg,
        universe=data.universe,
        fundamentals_by_instrument=data.fundamentals_by_instrument,
        sector_encoder=SectorEncoder(),
        cycles=cycles,
        macro_lookup=data.macro_lookup,
    )
    labeled_quarters = build_labeled_quarters(
        cycles=cycles,
        feature_results=feature_results,
        prices_by_instrument=data.prices_by_instrument,
        n_relevance_grades=cfg.n_relevance_grades,
    )
    ic_result = run_ic_backtest(
        config=cfg,
        universe=data.universe,
        fundamentals_by_instrument=data.fundamentals_by_instrument,
        prices_by_instrument=data.prices_by_instrument,
        sector_encoder=SectorEncoder(),
        cycles=cycles,
        estimator_factory=build_lgbm_ranker_estimator,
        macro_lookup=data.macro_lookup,
        feature_results=feature_results,
        labeled_by_quarter=labeled_quarters,
    )

    trimmed_prices = {
        instrument_id: tuple(
            p for p in prices if PRICE_CACHE_START <= p.trading_date <= PRICE_CACHE_END
        )
        for instrument_id, prices in data.prices_by_instrument.items()
    }
    trimmed_prices = {k: v for k, v in trimmed_prices.items() if v}

    return RealDataSlice(
        fingerprint=_fingerprint(),
        built_at=datetime.now(),
        source_manifest=source_manifest(),
        cycles=cycles,
        universe=data.universe,
        total_fundamentals_rows=sum(len(v) for v in data.fundamentals_by_instrument.values()),
        total_price_rows=sum(len(v) for v in data.prices_by_instrument.values()),
        load_issue_count=len(data.issues),
        feature_results=feature_results,
        labeled_quarters=labeled_quarters,
        prices_by_instrument=trimmed_prices,
        ic_result=ic_result,
    )


def cache_path() -> Path:
    return CACHE_ROOT / f"slice-{_fingerprint()}.pickle"


@lru_cache(maxsize=1)
def slice_bundle(rebuild: bool = False) -> RealDataSlice:
    """The bounded real-data slice, built once and cached on disk.

    Raises :class:`FileNotFoundError` when the raw data is absent -- call
    :func:`require_data` first (every notebook does).
    """
    if not data_available():
        raise FileNotFoundError(DATA_UNAVAILABLE_MESSAGE)

    path = cache_path()
    if path.is_file() and not rebuild:
        try:
            bundle = pickle.loads(path.read_bytes())
        except Exception:  # a truncated/incompatible cache is never trusted
            bundle = None
        if isinstance(bundle, RealDataSlice) and bundle.fingerprint == _fingerprint():
            return bundle

    bundle = _build_slice()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pickle.tmp")
    tmp.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)
    return bundle
