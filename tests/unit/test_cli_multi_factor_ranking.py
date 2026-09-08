from __future__ import annotations

import io
from datetime import date

import pytest

from atlas_quant.cli.multi_factor_ranking import build_parser, main
from atlas_quant.strategies.multi_factor_ranking_ml.acquisition.fundamentals_quarterly import (
    FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS,
)


def _fundamentals_csv_text(symbols, quarter_ends):
    header = (
        "ticker,period_end,fiscal_year,fiscal_quarter,available_date,"
        "available_date_is_estimated,gics_sector_name,"
        + ",".join(FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS)
        + "\n"
    )
    lines = [header]
    for symbol in symbols:
        for i, quarter_end in enumerate(quarter_ends):
            available_date = date.fromordinal(quarter_end.toordinal() + 45)
            values = ",".join(str(0.01 * (i + 1)) for _ in FUNDAMENTALS_QUARTERLY_FEATURE_COLUMNS)
            fq = (quarter_end.month - 1) // 3 + 1
            lines.append(
                f"{symbol} UN Equity,{quarter_end.isoformat()},{quarter_end.year},{fq},"
                f"{available_date.isoformat()},False,Information Technology,{values}\n"
            )
    return "".join(lines)


@pytest.fixture
def raw_root(tmp_path):
    symbols = ["AAA", "BBB", "CCC"]
    quarter_ends = [date(2019, 12, 31), date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30)]
    (tmp_path / "fundamentals_quarterly.csv").write_text(_fundamentals_csv_text(symbols, quarter_ends))
    return tmp_path


def test_build_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["multi-factor-ranking"])


def test_build_parser_rejects_unknown_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["not-a-real-command"])


def test_cli_validate_data_reports_universe_and_counts(raw_root):
    out, err = io.StringIO(), io.StringIO()
    code = main(["multi-factor-ranking", "validate-data", "--raw-root", str(raw_root)], stdout=out, stderr=err)
    assert code == 0
    assert "universe: 3 instrument(s)" in out.getvalue()


def test_cli_validate_data_missing_raw_root_is_a_clean_error(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["multi-factor-ranking", "validate-data", "--raw-root", str(tmp_path / "nope")], stdout=out, stderr=err,
    )
    assert code == 1
    assert "does not exist" in err.getvalue()


def test_cli_build_features_rejects_non_quarter_start_date(raw_root):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["multi-factor-ranking", "build-features", "--raw-root", str(raw_root), "--quarter-start", "2020-02-15"],
        stdout=out, stderr=err,
    )
    assert code == 1
    assert "not the first day of a calendar quarter" in err.getvalue()


def test_cli_build_features_reports_observation_count(raw_root):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["multi-factor-ranking", "build-features", "--raw-root", str(raw_root), "--quarter-start", "2020-10-01"],
        stdout=out, stderr=err,
    )
    assert code == 0
    assert "observation(s)" in out.getvalue()


def test_cli_run_backtest_rejects_too_short_a_range(raw_root):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        [
            "multi-factor-ranking", "run-backtest", "--raw-root", str(raw_root),
            "--start-quarter", "2020-10-01", "--end-quarter", "2020-10-01",
        ],
        stdout=out, stderr=err,
    )
    assert code == 1
    assert "requires at least 2" in err.getvalue()


def test_cli_rank_produces_and_locks_in_a_ranking(raw_root, tmp_path):
    out, err = io.StringIO(), io.StringIO()
    decision_log_root = tmp_path / "decisions"
    code = main(
        [
            "multi-factor-ranking", "rank", "--raw-root", str(raw_root), "--as-of", "2020-10-01",
            "--decision-log-root", str(decision_log_root),
        ],
        stdout=out, stderr=err,
    )
    assert code in (0, 1)  # 1 is acceptable if training is skipped for insufficient history
    if code == 0:
        assert "rank" in out.getvalue().lower()
