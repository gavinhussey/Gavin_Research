from __future__ import annotations

from datetime import date, datetime

import pytest

from atlas_quant.strategies.multi_factor_ranking_ml.config import MultiFactorRankingMLConfig
from atlas_quant.strategies.multi_factor_ranking_ml.evaluation_schedule import (
    EvaluationCycle,
    quarterly_evaluation_cycles,
)
from atlas_quant.strategies.multi_factor_ranking_ml.model_training import TrainingState
from atlas_quant.strategies.multi_factor_ranking_ml.production import orchestration
from atlas_quant.strategies.multi_factor_ranking_ml.production.decision_log import read_decision

from fixtures.multi_factor_ranking_ml import FakeEstimator, instrument, make_daily_series

pytestmark = pytest.mark.filterwarnings("ignore")


def _estimator_factory(*args, **kwargs):
    from atlas_quant.strategies.multi_factor_ranking_ml.estimator import EstimatorBuildInfo

    return FakeEstimator(default_score=0.5), EstimatorBuildInfo(
        estimator_type="FakeEstimator", library="test", library_version=None, parameters={},
    )


def _write_fundamentals_quarterly(path, symbols, quarter_ends):
    header = (
        "ticker,period_end,fiscal_year,fiscal_quarter,available_date,available_date_is_estimated,"
        "gics_sector_name,market_cap,consensus_eps_next_q,volatility_30d,analyst_target_price,pe_ratio,"
        "price_to_book,volatility_63d,volatility_20d,volume,consensus_sales_next_q,price_to_sales,beta,"
        "analyst_rating,volatility_90d,analyst_eps_num_est,free_cash_flow,operating_margin,net_margin,"
        "operating_cash_flow_margin,free_cash_flow_margin,revenue_yoy_growth,revenue_qoq_growth,"
        "operating_income_yoy_growth,operating_income_qoq_growth,net_income_yoy_growth,net_income_qoq_growth,"
        "diluted_eps_yoy_growth,diluted_eps_qoq_growth,operating_cash_flow_yoy_growth,"
        "operating_cash_flow_qoq_growth,free_cash_flow_yoy_growth,free_cash_flow_qoq_growth,"
        "total_assets_yoy_growth,total_assets_qoq_growth,total_debt_yoy_growth,total_debt_qoq_growth,"
        "stockholders_equity_yoy_growth,stockholders_equity_qoq_growth,diluted_share_count_yoy_growth,"
        "diluted_share_count_qoq_growth,shares_outstanding_yoy_growth,shares_outstanding_qoq_growth,"
        "revenue_growth_acceleration,operating_income_growth_acceleration,eps_growth_acceleration,"
        "operating_cash_flow_growth_acceleration,free_cash_flow_growth_acceleration,"
        "operating_margin_yoy_change_bps,operating_margin_qoq_change_bps,net_margin_yoy_change_bps,"
        "net_margin_qoq_change_bps,free_cash_flow_margin_yoy_change_bps,free_cash_flow_margin_qoq_change_bps,"
        "operating_cash_flow_to_net_income,free_cash_flow_to_net_income,capex_to_revenue,"
        "capex_to_depreciation,net_debt,adjusted_net_debt,debt_to_equity,debt_to_assets,ROA,ROE\n"
    )
    lines = [header]
    for symbol in symbols:
        for i, quarter_end in enumerate(quarter_ends):
            available_date = date.fromordinal(quarter_end.toordinal() + 45)
            values = ",".join(str(0.01 * (i + 1)) for _ in range(63))
            fiscal_quarter = (quarter_end.month - 1) // 3 + 1
            lines.append(
                f"{symbol} UN Equity,{quarter_end.isoformat()},{quarter_end.year},{fiscal_quarter},"
                f"{available_date.isoformat()},False,Information Technology,{values}\n"
            )
    path.write_text("".join(lines))


@pytest.fixture
def raw_root(tmp_path):
    symbols = ["AAA", "BBB", "CCC"]
    quarter_ends = [date(2019, 12, 31), date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30)]
    _write_fundamentals_quarterly(tmp_path / "fundamentals_quarterly.csv", symbols, quarter_ends)

    days = []
    d = date(2019, 1, 1)
    while d <= date(2021, 1, 1):
        if d.weekday() < 5:
            days.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    lines = ["date," + ",".join(f"{s} UN Equity" for s in symbols) + "\n"]
    for day in days:
        lines.append(day.isoformat() + "," + ",".join("100.0" for _ in symbols) + "\n")
    (tmp_path / "Close_Price.csv").write_text("".join(lines))
    return tmp_path


def test_load_raw_data_builds_universe_and_records(raw_root):
    data = orchestration.load_raw_data(raw_root, price_convention="unadjusted")
    assert len(data.universe) == 3
    assert all(len(v) == 4 for v in data.fundamentals_by_instrument.values())
    assert all(len(v) > 0 for v in data.prices_by_instrument.values())


def test_load_raw_data_missing_optional_files_is_not_fatal(tmp_path):
    _write_fundamentals_quarterly(tmp_path / "fundamentals_quarterly.csv", ["AAA"], [date(2020, 3, 31)])
    data = orchestration.load_raw_data(tmp_path, price_convention="unadjusted")
    assert data.prices_by_instrument == {}
    assert data.macro_lookup.value_as_of("fed_funds_rate", date(2020, 1, 1)) is None


def test_most_recent_cycle_picks_last_cycle_on_or_before_as_of():
    cycle = orchestration.most_recent_cycle(date(2020, 5, 15))
    assert cycle.quarter_start == date(2020, 4, 1)


def test_run_current_ranking_trains_and_records_when_enough_history(raw_root, tmp_path):
    data = orchestration.load_raw_data(raw_root, price_convention="unadjusted")
    config = MultiFactorRankingMLConfig(min_train_quarters=1, n_winners=1)
    cycle = EvaluationCycle(quarter_start=date(2020, 10, 1), cutoff=date(2020, 9, 30))
    decision_log_root = tmp_path / "decisions"

    result = orchestration.run_current_ranking(
        config=config, data=data, cycle=cycle, estimator_factory=_estimator_factory,
        decision_log_root=decision_log_root,
    )

    assert result.decision_log_entry is not None
    logged = read_decision(decision_log_root, cycle.quarter_start)
    assert logged is not None
    assert logged.quarter_start == cycle.quarter_start
    assert len(logged.rankings) == len(result.decision_summary.ranked_candidates)


def test_run_current_ranking_returns_existing_entry_if_already_decided(raw_root, tmp_path):
    data = orchestration.load_raw_data(raw_root, price_convention="unadjusted")
    config = MultiFactorRankingMLConfig(min_train_quarters=1, n_winners=1)
    cycle = EvaluationCycle(quarter_start=date(2020, 10, 1), cutoff=date(2020, 9, 30))
    decision_log_root = tmp_path / "decisions"

    first = orchestration.run_current_ranking(
        config=config, data=data, cycle=cycle, estimator_factory=_estimator_factory,
        decision_log_root=decision_log_root,
    )
    second = orchestration.run_current_ranking(
        config=config, data=data, cycle=cycle, estimator_factory=_estimator_factory,
        decision_log_root=decision_log_root,
    )
    assert second.decision_log_entry == first.decision_log_entry
    assert "already decided" in second.warnings[0]


def test_run_current_ranking_skips_when_insufficient_training_history(raw_root, tmp_path):
    data = orchestration.load_raw_data(raw_root, price_convention="unadjusted")
    config = MultiFactorRankingMLConfig(min_train_quarters=50, n_winners=1)
    cycle = EvaluationCycle(quarter_start=date(2020, 10, 1), cutoff=date(2020, 9, 30))

    result = orchestration.run_current_ranking(
        config=config, data=data, cycle=cycle, estimator_factory=_estimator_factory,
        decision_log_root=tmp_path / "decisions",
    )
    assert result.training_state != TrainingState.TRAINED
    assert result.decision_log_entry is None
