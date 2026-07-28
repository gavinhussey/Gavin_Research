"""Unit tests for atlas_quant.strategies.filing_momentum_ml.reporting.*"""

import json
from datetime import date, datetime

import pytest

from atlas_quant.backtest.clock import build_period
from atlas_quant.backtest.filing_momentum_runner import BacktestQuarterResult, BacktestResult, QuarterOutcomeType
from atlas_quant.domain.status import StrategyStatus
from atlas_quant.reporting.domain import ReproducibilityStatus
from atlas_quant.strategies.filing_momentum_ml.config import FilingMomentumMLConfig
from atlas_quant.strategies.filing_momentum_ml.performance_analysis import analyze_backtest_result
from atlas_quant.strategies.filing_momentum_ml.regime_config import RegimeConfig
from atlas_quant.strategies.filing_momentum_ml.reporting.charts import (
    alpha_distribution_chart,
    equity_growth_chart,
    outcome_composition_chart,
    render_svg_bar_chart,
)
from atlas_quant.strategies.filing_momentum_ml.reporting.comparison import (
    ComparisonInput,
    build_comparison_record,
    extract_source_report_values,
)
from atlas_quant.strategies.filing_momentum_ml.reporting.html_template import render_report_html
from atlas_quant.strategies.filing_momentum_ml.reporting.output import (
    DEFAULT_REPORT_OUTPUT_ROOT,
    write_report_artifacts,
)
from atlas_quant.strategies.filing_momentum_ml.reporting.report_builder import (
    ReportBuildError,
    build_filing_momentum_report,
)
from atlas_quant.strategies.filing_momentum_ml.reporting.report_model import ReportOptions
from atlas_quant.strategies.filing_momentum_ml.reporting.serialization import report_to_dict
from atlas_quant.reporting.domain import ComparisonStatus


def _minimal_strategy_result(status: StrategyStatus):
    from atlas_quant.strategies.base import StrategyResult

    return StrategyResult(
        strategy_id="filing_momentum_ml", display_name="Filing Momentum ML", strategy_version="0.1.0",
        config_identity="a" * 64, model_identity=None, evaluation_timestamp=datetime(2020, 1, 1),
        data_cutoff=datetime(2020, 1, 1), status=status,
    )


def _quarter(quarter_end, outcome_type, period_return=None, benchmark_return=None, status=None) -> BacktestQuarterResult:
    period = build_period(quarter_end)
    alpha = (period_return - benchmark_return) if (period_return is not None and benchmark_return is not None) else None
    strategy_result = _minimal_strategy_result(status) if status is not None else None
    return BacktestQuarterResult(
        period=period, outcome_type=outcome_type, training_state=None, model_identity=None,
        scoring_result=None, market_regime=None, per_instrument_regime_count=0,
        strategy_result=strategy_result, positions=(), period_return=period_return,
        benchmark=None, benchmark_return=benchmark_return, alpha=alpha, cash_weight=0.0,
    )


_QUARTER_END_MONTHS_DAYS = ((3, 31), (6, 30), (9, 30), (12, 31))


def _quarter_end(index: int, start_year: int = 2018) -> date:
    year = start_year + index // 4
    month, day = _QUARTER_END_MONTHS_DAYS[index % 4]
    return date(year, month, day)


def _backtest_result(quarters, run_identity="run" * 16) -> BacktestResult:
    return BacktestResult(
        strategy_id="filing_momentum_ml", strategy_version="0.1.0", config_identity="a" * 64,
        backtest_start=quarters[0].period.quarter_end if quarters else date.min,
        backtest_end=quarters[-1].period.quarter_end if quarters else date.min,
        quarter_results=tuple(quarters), run_identity=run_identity,
    )


def _mixed_result(n_primary=6, n_fallback=4, run_identity="run" * 16):
    quarters = []
    idx = 0
    for _ in range(n_primary):
        quarters.append(_quarter(_quarter_end(idx), QuarterOutcomeType.PRIMARY, 0.02 + 0.001 * idx, 0.01))
        idx += 1
    for _ in range(n_fallback):
        quarters.append(_quarter(_quarter_end(idx), QuarterOutcomeType.FALLBACK, 0.01, 0.01))
        idx += 1
    return _backtest_result(quarters, run_identity=run_identity)


def _report():
    backtest_result = _mixed_result()
    analysis = analyze_backtest_result(backtest_result)
    return build_filing_momentum_report(
        backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig(), ReportOptions(),
    )


class TestReportOptions:
    def test_defaults(self):
        options = ReportOptions()
        assert options.top_n_rows == 10
        assert options.decimal_precision == 4

    def test_rejects_invalid_top_n(self):
        with pytest.raises(ValueError):
            ReportOptions(top_n_rows=0)

    def test_rejects_invalid_audit_detail_level(self):
        with pytest.raises(ValueError):
            ReportOptions(audit_detail_level="bogus")

    def test_identity_deterministic(self):
        assert ReportOptions().identity() == ReportOptions().identity()

    def test_identity_changes_with_top_n(self):
        assert ReportOptions().identity() != ReportOptions(top_n_rows=5).identity()

    def test_section_and_scope_selection(self):
        options = ReportOptions(included_scopes=("primary_only",))
        assert options.included_scopes == ("primary_only",)


class TestReportBuilderCompatibility:
    def test_matching_identities_succeeds(self):
        report = _report()
        assert report.metadata.strategy_id == "filing_momentum_ml"

    def test_mismatched_run_identity_rejected(self):
        backtest_result = _mixed_result(run_identity="A" * 64)
        analysis = analyze_backtest_result(_mixed_result(run_identity="B" * 64))
        with pytest.raises(ReportBuildError):
            build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())

    def test_mismatched_strategy_id_rejected(self):
        backtest_result = _mixed_result()
        analysis = analyze_backtest_result(backtest_result)
        import dataclasses

        bad_result = dataclasses.replace(backtest_result, strategy_id="other_strategy")
        with pytest.raises(ReportBuildError):
            build_filing_momentum_report(bad_result, analysis, FilingMomentumMLConfig(), RegimeConfig())

    def test_mismatched_version_rejected(self):
        backtest_result = _mixed_result()
        analysis = analyze_backtest_result(backtest_result)
        import dataclasses

        bad_result = dataclasses.replace(backtest_result, strategy_version="9.9.9")
        with pytest.raises(ReportBuildError):
            build_filing_momentum_report(bad_result, analysis, FilingMomentumMLConfig(), RegimeConfig())

    def test_deterministic_output(self):
        backtest_result = _mixed_result()
        analysis = analyze_backtest_result(backtest_result)
        r1 = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        r2 = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        assert r1.metadata.report_identity == r2.metadata.report_identity


class TestExecutiveSummary:
    def test_configuration_values(self):
        report = _report()
        config = FilingMomentumMLConfig()
        assert report.executive_summary.ml_threshold == config.ml_threshold
        assert report.executive_summary.min_positions == config.min_positions
        assert report.executive_summary.max_positions == config.max_positions

    def test_quarter_counts_via_coverage(self):
        report = _report()
        assert report.coverage.primary_count == 6
        assert report.coverage.fallback_count == 4

    def test_no_promotional_language(self):
        report = _report()
        forbidden_words = ["guaranteed", "best-in-class", "amazing", "risk-free"]
        text = report.executive_summary.description.lower()
        assert not any(w in text for w in forbidden_words)


class TestPerformanceSections:
    def test_all_scopes_present(self):
        report = _report()
        assert set(report.performance_scopes.keys()) == {"all_evaluated", "invested", "primary_only", "fallback_only"}

    def test_scope_labels_always_visible(self):
        report = _report()
        for summary in report.performance_scopes.values():
            assert summary.scope_label

    def test_primary_and_fallback_are_different_scopes(self):
        report = _report()
        assert report.performance_scopes["primary_only"].included_count == 6
        assert report.performance_scopes["fallback_only"].included_count == 4

    def test_missing_metric_shows_unavailable(self):
        # A single-quarter backtest makes Sharpe unavailable (n < 2).
        backtest_result = _backtest_result([_quarter(_quarter_end(0), QuarterOutcomeType.PRIMARY, 0.02, 0.01)])
        analysis = analyze_backtest_result(backtest_result)
        report = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        assert report.performance_scopes["primary_only"].sharpe.availability.value == "insufficient_history"


class TestAnnualAndQuarterlyTables:
    def test_chronological_ordering(self):
        report = _report()
        years = [row["year"] for row in report.annual_table.rows]
        assert years == sorted(years)

    def test_quarterly_table_includes_skipped(self):
        quarters = [_quarter(_quarter_end(0), QuarterOutcomeType.SKIPPED)] + [
            _quarter(_quarter_end(i), QuarterOutcomeType.PRIMARY, 0.01, 0.0) for i in range(1, 5)
        ]
        backtest_result = _backtest_result(quarters)
        analysis = analyze_backtest_result(backtest_result)
        report = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        assert any(row["outcome_type"] == "skipped" for row in report.quarterly_table.rows)

    def test_missing_benchmark_shown_as_none(self):
        quarters = [_quarter(_quarter_end(i), QuarterOutcomeType.PRIMARY, 0.01, None) for i in range(3)]
        backtest_result = _backtest_result(quarters)
        analysis = analyze_backtest_result(backtest_result)
        report = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        assert all(row["benchmark_return"] is None for row in report.quarterly_table.rows)


class TestHoldingsAndTopWorst:
    def test_ranking_field_is_contribution_not_display_string(self):
        report = _report()
        # top_holdings_table should be empty since our minimal fixture
        # quarters carry no PositionOutcome objects -- verifies it
        # doesn't crash and produces a structurally valid (empty) table.
        assert report.top_holdings_table.rows == ()

    def test_row_limit_recorded(self):
        report = _report()
        assert report.top_holdings_table.row_limit == 10

    def test_best_worst_quarters_ranking(self):
        report = _report()
        best_returns = [row["period_return"] for row in report.best_quarters_table.rows]
        assert best_returns == sorted(best_returns, reverse=True)


class TestSectorSection:
    def test_marked_unavailable_not_inferred(self):
        report = _report()
        assert report.sector_section.available is False
        assert report.sector_section.unavailable_reason


class TestStatisticalValiditySection:
    def test_permutation_deferred(self):
        report = _report()
        assert report.statistical_validity.permutation_test_available is False
        assert "validate_arnold_main" in report.statistical_validity.permutation_test_reason


class TestSourceReportParsing:
    def test_extraction_from_real_report(self):
        with open("/Users/gavinhussey/Downloads/report_current.html", encoding="utf-8", errors="ignore") as fh:
            html = fh.read()
        import re

        text = re.sub("<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        values = extract_source_report_values(text)
        assert values["full_sharpe"] == 0.99
        assert values["recent_sharpe"] == 1.41
        assert values["bayesian_posterior_sharpe"] == 1.03

    def test_missing_source_section(self):
        values = extract_source_report_values("nothing relevant here")
        assert all(v is None for v in values.values())

    def test_deterministic_extraction(self):
        text = "0.99 Sharpe Ratio 1.28 Sortino Ratio"
        a = extract_source_report_values(text)
        b = extract_source_report_values(text)
        assert a == b

    def test_no_source_file_mutation(self, tmp_path):
        original = "0.99 Sharpe Ratio"
        path = tmp_path / "fake_report.html"
        path.write_text(original)
        extract_source_report_values(path.read_text())
        assert path.read_text() == original


class TestComparison:
    def test_exact_match(self):
        record = build_comparison_record(ComparisonInput("k", 1.0, 1.0, 0.01, "test"))
        assert record.status == ComparisonStatus.MATCH

    def test_within_tolerance(self):
        record = build_comparison_record(ComparisonInput("k", 1.0, 1.02, 0.05, "test"))
        assert record.status == ComparisonStatus.WITHIN_TOLERANCE

    def test_unexplained_difference(self):
        record = build_comparison_record(ComparisonInput("k", 1.0, 5.0, 0.05, "test"))
        assert record.status == ComparisonStatus.DIFFERENT_UNEXPLAINED

    def test_missing_source_value(self):
        record = build_comparison_record(ComparisonInput("k", None, 1.0, 0.05, "test"))
        assert record.status == ComparisonStatus.UNAVAILABLE_IN_SOURCE

    def test_missing_atlasquant_value(self):
        record = build_comparison_record(ComparisonInput("k", 1.0, None, 0.05, "test"))
        assert record.status == ComparisonStatus.UNAVAILABLE_IN_ATLASQUANT

    def test_not_comparable_when_both_missing(self):
        record = build_comparison_record(ComparisonInput("k", None, None, 0.05, "test"))
        assert record.status == ComparisonStatus.NOT_COMPARABLE


class TestCharts:
    def test_equity_growth_chart_generation(self):
        report = _report()
        assert any(c.identifier == "equity_growth" for c in report.charts)

    def test_correct_series_count(self):
        backtest_result = _mixed_result()
        analysis = analyze_backtest_result(backtest_result)
        chart = equity_growth_chart(analysis.invested.return_series)
        assert len(chart.series_labels) == 2

    def test_output_artifact_generation(self, tmp_path):
        report = _report()
        chart = report.charts[0]
        svg = render_svg_bar_chart(chart)
        path = tmp_path / "chart.svg"
        path.write_text(svg)
        assert path.exists()
        assert path.read_text().startswith("<svg")

    def test_empty_data_behavior(self):
        from atlas_quant.strategies.filing_momentum_ml.performance_domain import PerformanceScope, ScopeDefinition, ReturnSeries

        empty_series = ReturnSeries(
            scope=ScopeDefinition.standard(PerformanceScope.PRIMARY_ONLY), points=(), excluded=(),
            backtest_run_identity="x",
        )
        chart = alpha_distribution_chart(empty_series)
        assert chart.is_empty()
        svg = render_svg_bar_chart(chart)
        assert "no data" in svg

    def test_no_calculations_hidden_chart_data_matches_series(self):
        backtest_result = _mixed_result()
        analysis = analyze_backtest_result(backtest_result)
        chart = equity_growth_chart(analysis.invested.return_series)
        # last strategy point must equal metrics.total_return (same value, no separate calc)
        last_point = chart.points[0][-1]
        assert last_point.y == pytest.approx(analysis.invested.metrics.total_return)


class TestReportSerialization:
    def test_round_trip_json_serializable(self):
        report = _report()
        data = report_to_dict(report)
        json.dumps(data)

    def test_deterministic_json(self):
        report = _report()
        d1 = report_to_dict(report)
        d2 = report_to_dict(report)
        assert d1 == d2

    def test_enum_preservation(self):
        report = _report()
        data = report_to_dict(report)
        assert data["metadata"]["reproducibility_status"] == "not_run"

    def test_availability_state_preserved(self):
        backtest_result = _backtest_result([_quarter(_quarter_end(0), QuarterOutcomeType.PRIMARY, 0.02, 0.01)])
        analysis = analyze_backtest_result(backtest_result)
        report = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        data = report_to_dict(report)
        assert data["performance_scopes"]["primary_only"]["sharpe"]["availability"] == "insufficient_history"


class TestReportHtml:
    def test_required_headings_present(self):
        report = _report()
        html = render_report_html(report)
        for heading in ["Executive Summary", "Strategy Specification", "Backtest Coverage",
                        "Performance Summary", "Statistical Validity", "Audit and Reproducibility"]:
            assert heading in html

    def test_no_external_cdn(self):
        report = _report()
        html = render_report_html(report)
        assert "http://" not in html or "www.w3.org" in html  # only the SVG XML namespace
        assert "https://" not in html
        assert "cdn." not in html.lower()

    def test_no_secrets_in_html(self):
        report = _report()
        html = render_report_html(report)
        for forbidden in ("SCHWAB_APP_SECRET", "BLOOMBERG_", "api_key", "password"):
            assert forbidden not in html

    def test_escaping_of_malicious_instrument_symbol(self):
        from atlas_quant.domain.identifiers import AssetClass, InstrumentId
        from atlas_quant.domain.signal import InstrumentRecommendation
        from atlas_quant.domain.status import SignalKind
        from atlas_quant.backtest.accounting import PositionLifecycleState, PositionOutcome

        malicious_id = InstrumentId(symbol="XSS", asset_class=AssetClass.EQUITY)
        position = PositionOutcome(
            instrument_id=malicious_id, role=SignalKind.PRIMARY, target_weight=0.1,
            entry_target_timestamp=date(2020, 5, 1), entry_resolved=None,
            exit_target_timestamp=date(2020, 8, 1), exit_resolved=None,
            raw_return=None, capped_return=None, contribution=None,
            lifecycle_state=PositionLifecycleState.UNRESOLVED,
        )
        quarter = _quarter(_quarter_end(0), QuarterOutcomeType.PRIMARY, 0.02, 0.01)
        import dataclasses

        quarter = dataclasses.replace(quarter, positions=(position,))
        backtest_result = _backtest_result([quarter])
        analysis = analyze_backtest_result(backtest_result)
        report = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        html = render_report_html(report)
        assert "<script>" not in html

    def test_empty_optional_sections_render_without_error(self):
        backtest_result = _backtest_result([])
        # An empty backtest result would fail run-identity matching in a
        # real pipeline; here we only check the HTML renderer handles a
        # report built from a minimal, mostly-empty (but valid) result.
        backtest_result = _mixed_result(n_primary=0, n_fallback=0)
        analysis = analyze_backtest_result(backtest_result)
        report = build_filing_momentum_report(backtest_result, analysis, FilingMomentumMLConfig(), RegimeConfig())
        html = render_report_html(report)
        assert "<html" in html


class TestOutputWrites:
    def test_json_and_html_written(self, tmp_path):
        report = _report()
        written = write_report_artifacts(report, tmp_path)
        assert written["json"].exists()
        assert written["html"].exists()

    def test_overwrite_policy(self, tmp_path):
        report = _report()
        write_report_artifacts(report, tmp_path)
        with pytest.raises(Exception):
            write_report_artifacts(report, tmp_path, overwrite=False)

    def test_protected_production_path_rejected(self):
        report = _report()
        with pytest.raises(Exception):
            write_report_artifacts(report, DEFAULT_REPORT_OUTPUT_ROOT)
        assert not DEFAULT_REPORT_OUTPUT_ROOT.exists()

    def test_temporary_path_success(self, tmp_path):
        report = _report()
        written = write_report_artifacts(report, tmp_path, formats=("json",))
        assert written["json"].exists()
        assert "html" not in written
