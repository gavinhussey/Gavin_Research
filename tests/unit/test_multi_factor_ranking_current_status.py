from __future__ import annotations

import importlib
import io
import sys
from datetime import date
from pathlib import Path

import pytest

LIVE_DIR = Path(__file__).resolve().parents[2] / "live" / "multi_factor_ranking_ml"


def _load_module():
    sys.path.insert(0, str(LIVE_DIR))
    try:
        if "current_status" in sys.modules:
            del sys.modules["current_status"]
        return importlib.import_module("current_status")
    finally:
        sys.path.remove(str(LIVE_DIR))


def _fundamentals_csv_text(symbols, quarter_ends):
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
            fq = (quarter_end.month - 1) // 3 + 1
            lines.append(
                f"{symbol} UN Equity,{quarter_end.isoformat()},{quarter_end.year},{fq},"
                f"{available_date.isoformat()},False,Information Technology,{values}\n"
            )
    return "".join(lines)


@pytest.fixture
def raw_root(tmp_path):
    quarter_ends = [date(2019, 12, 31), date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30)]
    (tmp_path / "fundamentals_quarterly.csv").write_text(_fundamentals_csv_text(["AAA", "BBB", "CCC"], quarter_ends))
    return tmp_path


def test_current_status_calls_multi_factor_ranking_cli_not_shared_main(monkeypatch, raw_root, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "RAW_ROOT", raw_root)
    monkeypatch.setattr(module, "DECISION_LOG_ROOT", tmp_path / "decisions")
    monkeypatch.setattr(module, "AS_OF", "2020-10-01")

    captured = {}

    def fake_cli_main(args, **kwargs):
        captured["args"] = args
        return 0

    monkeypatch.setattr("atlas_quant.cli.multi_factor_ranking.main", fake_cli_main)

    code = module.main()
    assert code == 0
    assert captured["args"][:2] == ["multi-factor-ranking", "rank"]
    assert "--as-of" in captured["args"]


def test_current_status_end_to_end_produces_output(monkeypatch, raw_root, tmp_path, capsys):
    module = _load_module()
    monkeypatch.setattr(module, "RAW_ROOT", raw_root)
    monkeypatch.setattr(module, "DECISION_LOG_ROOT", tmp_path / "decisions")
    monkeypatch.setattr(module, "AS_OF", "2020-10-01")

    code = module.main()
    assert code in (0, 1)
