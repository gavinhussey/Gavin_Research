from __future__ import annotations

from datetime import date, datetime

from atlas_quant.backtest.multi_factor_ranking_runner import ICBacktestResult, RankingCycleResult
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import EvaluationCycle
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import TrainingState
from atlas_quant.strategies.multi_factor_ranking_ml.production.decision_log import DecisionLogEntry, DecisionRanking
from atlas_quant.strategies.multi_factor_ranking_ml.reporting.report_builder import (
    build_backtest_report,
    build_ranking_report,
)

from fixtures.multi_factor_ranking_ml import instrument


def _entry():
    return DecisionLogEntry(
        quarter_start=date(2020, 10, 1), cutoff=date(2020, 9, 30), decided_at=datetime(2020, 10, 1, 6),
        outcome="ranked", model_identity_hash="deadbeef",
        rankings=(
            DecisionRanking(instrument_id=instrument("BBB"), score=0.9, rank=1),
            DecisionRanking(instrument_id=instrument("AAA"), score=0.1, rank=2),
        ),
    )


def test_ranking_report_text_contains_symbols_and_ranks():
    text = build_ranking_report(_entry()).to_text()
    assert "BBB" in text and "AAA" in text
    assert "2 instrument(s) ranked" in text
    assert "2020-10-01" in text


def test_ranking_report_respects_top_n():
    text = build_ranking_report(_entry()).to_text(top_n=1)
    assert "BBB" in text
    assert "AAA" not in text
    assert "showing top 1" in text


def test_ranking_report_to_dict_matches_entry():
    entry = _entry()
    assert build_ranking_report(entry).to_dict() == entry.to_dict()


def test_ranking_report_sorted_by_rank_regardless_of_input_order():
    entry = DecisionLogEntry(
        quarter_start=date(2020, 10, 1), cutoff=date(2020, 9, 30), decided_at=datetime(2020, 10, 1),
        outcome="ranked", model_identity_hash=None,
        rankings=(
            DecisionRanking(instrument_id=instrument("AAA"), score=0.1, rank=2),
            DecisionRanking(instrument_id=instrument("BBB"), score=0.9, rank=1),
        ),
    )
    text = build_ranking_report(entry).to_text()
    assert text.index("BBB") < text.index("AAA")


def _ic_result():
    cycle = EvaluationCycle(quarter_start=date(2020, 1, 1), cutoff=date(2019, 12, 31))
    cycle_result = RankingCycleResult(
        cycle=cycle, training_state=TrainingState.TRAINED, model_identity=None, scoring_result=None,
        decision_summary=None, ic=0.25, decile_spread=0.05, auc=0.6, ranked_count=10,
        scored_for_ic_count=10, scored_for_auc_count=10,
    )
    return ICBacktestResult(
        strategy_id="multi_factor_ranking_ml", strategy_version="0.2.0", config_identity="abc",
        cycle_results=(cycle_result,), run_identity="run-1",
    )


def test_backtest_report_text_contains_headline_stats():
    text = build_backtest_report(_ic_result()).to_text()
    assert "mean IC:" in text
    assert "0.2500" in text
    assert "no equity curve, Sharpe ratio, or drawdown" in text
    assert "mean AUC" in text
    assert "0.6000" in text


def test_backtest_report_to_dict_matches_result():
    result = _ic_result()
    assert build_backtest_report(result).to_dict() == result.to_dict()
