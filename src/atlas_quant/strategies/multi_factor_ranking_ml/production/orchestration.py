"""Orchestrates one full pass of Multi-Factor Ranking ML: load already-
acquired raw data -> build this cycle's features -> train on the trailing
window of prior real cycles -> score -> rank -> record via
``decision_log.py``.

This is a pure ranking system (no portfolio, no positions, no orders --
see ``strategy.py``'s module docstring), so there is no rebalancing step
here: ``run_current_ranking`` produces and locks in one quarterly cycle's
full ranking, nothing else. Historical performance evaluation (measuring
the ranking's Information Coefficient against realized forward returns)
is a separate concern, handled by
``atlas_quant.backtest.multi_factor_ranking_runner.run_ic_backtest`` over
a range of *past* cycles -- this module never measures IC, since the
current/most-recent cycle has no realized "next cycle" return yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from atlas_quant.data.records import DailyPriceObservation, PriceConvention
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.strategies.base import StrategyEvaluationContext
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.close_price import read_close_price
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.fundamentals_quarterly import (
    read_fundamentals_quarterly,
)
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.legacy_features import (
    join_legacy_features_onto_fundamentals,
    read_legacy_features,
)
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.macro import MacroSeriesLookup, read_macro_csv
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.universe import universe_records_from_fundamentals
from atlas_quant.strategies.multi_factor_ranking_ml.config import (
    FEATURE_SCHEMA_VERSION,
    STRATEGY_VERSION,
    MultiFactorRankingMLConfig,
)
from atlas_quant.strategies.multi_factor_ranking_ml.decision_domain import MultiFactorRankingDecisionSummary
from atlas_quant.strategies.multi_factor_ranking_ml.estimator import Estimator, EstimatorBuildInfo
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import (
    EvaluationCycle,
    quarterly_evaluation_cycles,
)
from atlas_quant.strategies.multi_factor_ranking_ml.feature_pipeline import FeaturePipelineResult, run_feature_pipeline
from atlas_quant.strategies.multi_factor_ranking_ml.forward_return import build_forward_return_outcome
from atlas_quant.strategies.multi_factor_ranking_ml.labeling import assign_quarterly_labels
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import ModelIdentity, TrainingState, train_model
from atlas_quant.strategies.multi_factor_ranking_ml.production.decision_log import (
    DecisionLogEntry,
    DecisionRanking,
    read_decision,
    write_decision_if_absent,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.normalization import (
    FundamentalsFeatureRecord,
    normalize_fundamentals_batch,
    normalize_prices,
    normalize_universe,
    sector_records_from_fundamentals,
)
from atlas_quant.strategies.multi_factor_ranking_ml.production.validation import (
    DataValidationIssue,
    validate_filings,
    validate_prices,
)
from atlas_quant.strategies.multi_factor_ranking_ml.scoring import score_observations
from atlas_quant.strategies.multi_factor_ranking_ml.sector_encoding import SectorEncoder
from atlas_quant.strategies.multi_factor_ranking_ml.strategy import (
    MultiFactorRankingEvaluationInputs,
    MultiFactorRankingMLStrategy,
)
from atlas_quant.strategies.multi_factor_ranking_ml.training_dataset import (
    LabeledObservation,
    build_training_dataset,
    check_training_eligibility,
)

_FUNDAMENTALS_FILE = "fundamentals_quarterly.csv"
_LEGACY_FEATURES_FILE = "filing_momentum_features.csv"
_FED_FUNDS_FILE = "Fed_Funds_Rate.csv"
_DAILY_MACRO_FILE = "Daily_Macro.csv"
_CLOSE_PRICE_FILE = "Close_Price.csv"


@dataclass(frozen=True, slots=True)
class LoadedData:
    """Everything :func:`run_current_ranking`/the IC backtest need, already
    normalized into Stage 3 domain types."""

    universe: tuple[InstrumentId, ...]
    fundamentals_by_instrument: dict[InstrumentId, tuple[FundamentalsFeatureRecord, ...]]
    prices_by_instrument: dict[InstrumentId, tuple[DailyPriceObservation, ...]]
    macro_lookup: MacroSeriesLookup
    issues: tuple[DataValidationIssue, ...]


def load_raw_data(raw_root: Path, *, price_convention: PriceConvention) -> LoadedData:
    """Load and normalize every raw CSV under ``raw_root`` (a local copy of
    ``data/raw/multi_factor_ranking_ml/``'s Bloomberg exports).

    Never fetches anything over the network -- ``raw_root``'s files must
    already exist. ``price_convention`` is required (no default; see
    ``acquisition/close_price.py``'s module docstring) since it can't be
    inferred from ``Close_Price.csv`` itself.
    """
    raw_rows = read_fundamentals_quarterly(raw_root / _FUNDAMENTALS_FILE)
    legacy_path = raw_root / _LEGACY_FEATURES_FILE
    if legacy_path.exists():
        raw_rows = join_legacy_features_onto_fundamentals(raw_rows, read_legacy_features(legacy_path))

    fundamentals, fundamentals_issues = normalize_fundamentals_batch(raw_rows)
    filing_issues = validate_filings(fundamentals)

    universe_raw = universe_records_from_fundamentals(raw_rows, retrieved_at=datetime.now())
    universe_members, universe_issues = normalize_universe(universe_raw)
    universe = tuple(dict.fromkeys(m.instrument_id for m in universe_members))

    fundamentals_by_instrument: dict[InstrumentId, list[FundamentalsFeatureRecord]] = {}
    for record in fundamentals:
        fundamentals_by_instrument.setdefault(record.instrument_id, []).append(record)

    prices_by_instrument: dict[InstrumentId, tuple[DailyPriceObservation, ...]] = {}
    price_issues: tuple[DataValidationIssue, ...] = ()
    close_price_path = raw_root / _CLOSE_PRICE_FILE
    if close_price_path.exists():
        raw_prices = read_close_price(close_price_path, price_convention=price_convention)
        prices, price_norm_issues = normalize_prices(raw_prices)
        price_issues = price_norm_issues + validate_prices(prices)
        for price in prices:
            prices_by_instrument.setdefault(price.instrument_id, []).append(price)
        prices_by_instrument = {k: tuple(v) for k, v in prices_by_instrument.items()}

    macro_lookup = MacroSeriesLookup()
    fed_funds_path = raw_root / _FED_FUNDS_FILE
    if fed_funds_path.exists():
        macro_lookup = macro_lookup.merge(read_macro_csv(fed_funds_path))
    daily_macro_path = raw_root / _DAILY_MACRO_FILE
    if daily_macro_path.exists():
        macro_lookup = macro_lookup.merge(read_macro_csv(daily_macro_path))

    return LoadedData(
        universe=universe,
        fundamentals_by_instrument={k: tuple(v) for k, v in fundamentals_by_instrument.items()},
        prices_by_instrument=prices_by_instrument,
        macro_lookup=macro_lookup,
        issues=fundamentals_issues + filing_issues + universe_issues + price_issues,
    )


def most_recent_cycle(as_of: date, *, earliest_start: date = date(2000, 1, 1)) -> EvaluationCycle:
    """The most recent quarterly evaluation cycle whose ``quarter_start``
    is on or before ``as_of`` -- "the ranking currently in effect" for a
    live run on a given day."""
    cycles = quarterly_evaluation_cycles(earliest_start, as_of)
    if not cycles:
        raise ValueError(f"no evaluation cycle starts on or before {as_of!r}")
    return cycles[-1]


def _label_prior_cycle(
    cycle: EvaluationCycle,
    next_cycle: EvaluationCycle,
    observations,
    prices_by_instrument: Mapping[InstrumentId, Sequence[DailyPriceObservation]],
    n_winners: int,
) -> tuple[LabeledObservation, ...]:
    exit_cutoff = datetime.combine(next_cycle.quarter_start, time.min)
    outcomes = [
        build_forward_return_outcome(
            obs.instrument_id, cycle.quarter_start, obs.feature_timestamp,
            next_cycle.quarter_start, prices_by_instrument.get(obs.instrument_id, ()), exit_cutoff,
        )
        for obs in observations
    ]
    labeling = assign_quarterly_labels(outcomes, cycle.quarter_start, n_winners=n_winners)
    label_by_id = {a.instrument_id: a.label for a in labeling.assignments}
    return tuple(
        LabeledObservation(obs, label_by_id[obs.instrument_id], exit_cutoff)
        for obs in observations
        if obs.instrument_id in label_by_id
    )


@dataclass(frozen=True, slots=True)
class RankingRunResult:
    """The outcome of one :func:`run_current_ranking` call."""

    cycle: EvaluationCycle
    training_state: TrainingState
    model_identity: ModelIdentity | None
    decision_summary: MultiFactorRankingDecisionSummary | None
    decision_log_entry: DecisionLogEntry | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def run_current_ranking(
    *,
    config: MultiFactorRankingMLConfig,
    data: LoadedData,
    cycle: EvaluationCycle,
    estimator_factory: Callable[..., tuple[Estimator, EstimatorBuildInfo]],
    decision_log_root: Path,
) -> RankingRunResult:
    """Produce (and permanently record, via ``decision_log.py``)
    ``cycle``'s full ranking of ``data.universe``.

    Trains on every real quarterly cycle strictly before ``cycle`` (an
    expanding window, same as the IC backtest's per-cycle training step),
    scores ``cycle``'s own feature observations, and ranks every
    surviving candidate -- no threshold, no cap, no weight. If ``cycle``
    was already decided (see ``decision_log.write_decision_if_absent``),
    that existing, locked-in record is returned unchanged rather than
    recomputed.
    """
    sector_encoder = SectorEncoder()
    already_decided = read_decision(decision_log_root, cycle.quarter_start)
    if already_decided is not None:
        return RankingRunResult(
            cycle=cycle, training_state=TrainingState.TRAINED if already_decided.rankings else TrainingState.SKIPPED,
            model_identity=None, decision_summary=None, decision_log_entry=already_decided,
            warnings=("cycle already decided; returning existing locked-in record",),
        )

    prior_cycles = quarterly_evaluation_cycles(date(2000, 1, 1), cycle.quarter_start)
    prior_cycles = tuple(c for c in prior_cycles if c.quarter_start < cycle.quarter_start)

    feature_results: dict[date, FeaturePipelineResult] = {}
    for prior in prior_cycles:
        feature_results[prior.quarter_start] = run_feature_pipeline(
            config=config, sector_encoder=sector_encoder, universe=data.universe,
            quarter_start=prior.quarter_start, cutoff=prior.cutoff,
            fundamentals_by_instrument=data.fundamentals_by_instrument, macro_lookup=data.macro_lookup,
        )
    current_features = run_feature_pipeline(
        config=config, sector_encoder=sector_encoder, universe=data.universe,
        quarter_start=cycle.quarter_start, cutoff=cycle.cutoff,
        fundamentals_by_instrument=data.fundamentals_by_instrument, macro_lookup=data.macro_lookup,
    )

    labeled_by_quarter: dict[date, tuple[LabeledObservation, ...]] = {}
    for i, prior in enumerate(prior_cycles):
        next_cycle = prior_cycles[i + 1] if i + 1 < len(prior_cycles) else cycle
        labeled_by_quarter[prior.quarter_start] = _label_prior_cycle(
            prior, next_cycle, feature_results[prior.quarter_start].observations,
            data.prices_by_instrument, config.n_winners,
        )

    training_cutoff = datetime.combine(cycle.cutoff, time.min)
    dataset = build_training_dataset(
        cycle.quarter_start, training_cutoff, labeled_by_quarter,
        strategy_id=config.strategy_id, feature_schema_version=FEATURE_SCHEMA_VERSION,
        ml_train_years=config.ml_train_years, model_config_identity=config.model.identity(),
    )
    eligibility = check_training_eligibility(
        dataset, min_train_quarters=config.min_train_quarters, n_winners=config.n_winners
    )
    training_result = train_model(
        dataset, eligibility, config.model, estimator_factory,
        strategy_id=config.strategy_id, strategy_version=STRATEGY_VERSION,
    )

    if training_result.state != TrainingState.TRAINED:
        return RankingRunResult(
            cycle=cycle, training_state=training_result.state, model_identity=None,
            decision_summary=None, decision_log_entry=None,
            warnings=(f"training skipped: {training_result.state.value}",),
        )

    scoring_result = score_observations(
        training_result.fitted_estimator, training_result.model_identity, current_features.observations,
        training_cutoff, config_identity=config.identity(),
    )
    eval_context = StrategyEvaluationContext(
        strategy_id=config.strategy_id, evaluation_timestamp=training_cutoff, data_cutoff=training_cutoff,
        capital_budget_pct=0.0,
        strategy_config=MultiFactorRankingEvaluationInputs(
            config=config, scored_candidates=scoring_result.scored_candidates,
        ),
    )
    strategy_result = MultiFactorRankingMLStrategy().evaluate(eval_context)
    summary: MultiFactorRankingDecisionSummary = strategy_result.state_update

    entry = DecisionLogEntry(
        quarter_start=cycle.quarter_start, cutoff=cycle.cutoff, decided_at=training_cutoff,
        outcome=summary.outcome.value, model_identity_hash=training_result.model_identity.identity(),
        rankings=tuple(
            DecisionRanking(instrument_id=r.instrument_id, score=r.score, rank=r.rank)
            for r in summary.ranked_candidates
        ),
    )
    locked_entry = write_decision_if_absent(decision_log_root, entry)

    return RankingRunResult(
        cycle=cycle, training_state=training_result.state, model_identity=training_result.model_identity,
        decision_summary=summary, decision_log_entry=locked_entry,
    )
