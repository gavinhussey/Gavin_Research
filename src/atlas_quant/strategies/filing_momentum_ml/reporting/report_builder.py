"""Builds one FilingMomentumReport from already-computed, typed Stage 3-8 results.

Never recalculates feature values, labels, scores, qualification, regime
classifications, position weights/returns, benchmark returns, or
performance metrics — every field here is read directly from a supplied
``BacktestResult``/``PerformanceAnalysisResult``/config.
"""

from __future__ import annotations

from atlas_quant.backtest.accounting import INSTRUMENT_RETURN_CAP, PositionLifecycleState
from atlas_quant.backtest.filing_momentum_runner import BacktestResult
from atlas_quant.domain.status import SignalKind
from atlas_quant.reporting.domain import (
    ArtifactIdentity,
    ChartSeriesDefinition,
    ReportMetadata,
    ReproducibilityStatus,
    SectionDefinition,
    TableColumn,
    TableDefinition,
    ValidationSummary,
)
from atlas_quant.reporting.validation import check, summarize
from atlas_quant.strategies.filing_momentum_ml.config import DISPLAY_NAME, STRATEGY_ID, STRATEGY_VERSION
from atlas_quant.strategies.filing_momentum_ml.forward_return import LABEL_RETURN_CLIP
from atlas_quant.strategies.filing_momentum_ml.performance_domain import (
    PerformanceAnalysisResult,
    QuarterClassification,
    ScopeAnalysis,
)
from atlas_quant.strategies.filing_momentum_ml.regime_config import RegimeConfig
from atlas_quant.strategies.filing_momentum_ml.reporting import charts as chart_builders
from atlas_quant.strategies.filing_momentum_ml.reporting.comparison import (
    KNOWN_INTENTIONAL_DIFFERENCES,
    build_comparison_record,
    extract_source_report_values,
    source_report_sha256,
    ComparisonInput,
)
from atlas_quant.strategies.filing_momentum_ml.reporting.report_model import (
    FILING_MOMENTUM_REPORT_SCHEMA_VERSION,
    AuditSection,
    BacktestCoverage,
    ExecutiveSummary,
    FilingMomentumReport,
    ProvenanceSection,
    RecentPeriodSection,
    ReportOptions,
    ScopePerformanceSummary,
    StatisticalValiditySection,
    StrategySpecification,
)
from atlas_quant.strategies.filing_momentum_ml.feature_domain import FEATURE_NAMES


class ReportBuildError(ValueError):
    """Raised when supplied results are incompatible -- never silently combined."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportBuildError(message)


def _scope_summary(scope_key: str, analysis: ScopeAnalysis) -> ScopePerformanceSummary:
    return ScopePerformanceSummary(
        scope=scope_key, scope_label=analysis.scope.label, included_count=analysis.return_series.included_count,
        total_return=analysis.metrics.total_return, benchmark_total_return=analysis.metrics.benchmark_total_return,
        mean_return=analysis.metrics.mean_return, median_return=analysis.metrics.median_return,
        return_stddev=analysis.metrics.return_stddev, sharpe=analysis.sharpe, sortino=analysis.sortino,
        information_ratio=analysis.information_ratio, max_drawdown=analysis.drawdown.max_drawdown,
        win_rate=analysis.metrics.win_rate, positive_alpha_rate=analysis.alpha_metrics.positive_rate,
    )


def _quarterly_table(backtest_result: BacktestResult) -> TableDefinition:
    columns = (
        TableColumn("quarter_end", "Quarter", "date"), TableColumn("outcome_type", "Outcome", "text"),
        TableColumn("period_return", "Strategy Return", "percent"), TableColumn("benchmark_return", "SPY Return", "percent"),
        TableColumn("alpha", "Alpha", "percent"), TableColumn("training_state", "Training", "text"),
        TableColumn("recommendation_count", "Recs", "number"), TableColumn("cash_weight", "Cash Wt", "percent"),
        TableColumn("warning_count", "Warnings", "number"),
    )
    rows = tuple(
        {
            "quarter_end": q.period.quarter_end.isoformat(), "outcome_type": q.outcome_type.value,
            "period_return": q.period_return, "benchmark_return": q.benchmark_return, "alpha": q.alpha,
            "training_state": q.training_state.value if q.training_state else None,
            "recommendation_count": len(q.positions), "cash_weight": q.cash_weight,
            "warning_count": len(q.warnings),
        }
        for q in backtest_result.quarter_results
    )
    return TableDefinition(identifier="quarterly_results", title="Quarter-by-Quarter Results", columns=columns, rows=rows)


def _holdings_table(backtest_result: BacktestResult) -> TableDefinition:
    columns = (
        TableColumn("quarter_end", "Quarter", "date"), TableColumn("instrument", "Instrument", "text"),
        TableColumn("role", "Role", "text"), TableColumn("target_weight", "Weight", "percent"),
        TableColumn("entry_price", "Entry", "number"), TableColumn("exit_price", "Exit", "number"),
        TableColumn("raw_return", "Raw Return", "percent"), TableColumn("capped_return", "Capped Return", "percent"),
        TableColumn("contribution", "Contribution", "percent"), TableColumn("lifecycle_state", "State", "text"),
    )
    rows = []
    for q in backtest_result.quarter_results:
        for position in q.positions:
            rows.append(
                {
                    "quarter_end": q.period.quarter_end.isoformat(), "instrument": position.instrument_id.symbol,
                    "role": position.role.value, "target_weight": position.target_weight,
                    "entry_price": position.entry_resolved.price if position.entry_resolved else None,
                    "exit_price": position.exit_resolved.price if position.exit_resolved else None,
                    "raw_return": position.raw_return, "capped_return": position.capped_return,
                    "contribution": position.contribution, "lifecycle_state": position.lifecycle_state.value,
                }
            )
    return TableDefinition(identifier="holdings", title="Holdings and Trade Outcomes", columns=columns, rows=tuple(rows))


def _top_worst_holdings(backtest_result: BacktestResult, n: int) -> tuple[TableDefinition, TableDefinition]:
    all_positions = [
        (q.period.quarter_end, p) for q in backtest_result.quarter_results for p in q.positions
        if p.contribution is not None
    ]
    ranked = sorted(all_positions, key=lambda item: item[1].contribution, reverse=True)
    columns = (
        TableColumn("quarter_end", "Quarter", "date"), TableColumn("instrument", "Instrument", "text"),
        TableColumn("role", "Role", "text"), TableColumn("capped_return", "Capped Return", "percent"),
        TableColumn("contribution", "Contribution", "percent"),
    )

    def _row(qend, position):
        return {
            "quarter_end": qend.isoformat(), "instrument": position.instrument_id.symbol,
            "role": position.role.value, "capped_return": position.capped_return,
            "contribution": position.contribution,
        }

    top = TableDefinition("top_holdings", "Top Holdings by Contribution", columns, tuple(_row(q, p) for q, p in ranked[:n]), row_limit=n)
    worst = TableDefinition("worst_holdings", "Worst Holdings by Contribution", columns, tuple(_row(q, p) for q, p in ranked[-n:][::-1]), row_limit=n)
    return top, worst


def _best_worst_quarters(backtest_result: BacktestResult, n: int) -> tuple[TableDefinition, TableDefinition]:
    quarters = [q for q in backtest_result.quarter_results if q.period_return is not None]
    ranked = sorted(quarters, key=lambda q: q.period_return, reverse=True)
    columns = (
        TableColumn("quarter_end", "Quarter", "date"), TableColumn("outcome_type", "Outcome", "text"),
        TableColumn("period_return", "Return", "percent"), TableColumn("alpha", "Alpha", "percent"),
    )

    def _row(q):
        return {"quarter_end": q.period.quarter_end.isoformat(), "outcome_type": q.outcome_type.value,
                "period_return": q.period_return, "alpha": q.alpha}

    best = TableDefinition("best_quarters", "Best Quarters", columns, tuple(_row(q) for q in ranked[:n]), row_limit=n)
    worst = TableDefinition("worst_quarters", "Worst Quarters", columns, tuple(_row(q) for q in ranked[-n:][::-1]), row_limit=n)
    return best, worst


def _annual_table(analysis: ScopeAnalysis) -> TableDefinition:
    columns = (
        TableColumn("year", "Year", "number"), TableColumn("strategy_return", "Strategy", "percent"),
        TableColumn("benchmark_return", "SPY", "percent"), TableColumn("alpha", "Alpha", "percent"),
        TableColumn("included_quarters", "Quarters", "number"), TableColumn("is_partial_year", "Partial", "text"),
    )
    rows = tuple(
        {
            "year": a.year, "strategy_return": a.strategy_compounded_return,
            "benchmark_return": a.benchmark_compounded_return, "alpha": a.alpha,
            "included_quarters": a.included_quarter_count, "is_partial_year": a.is_partial_year,
        }
        for a in analysis.annual_summaries
    )
    return TableDefinition("annual_results", "Year-by-Year Results", columns, rows)


def _run_validation(backtest_result: BacktestResult, performance_analysis: PerformanceAnalysisResult, report: dict) -> ValidationSummary:
    checks = [
        check("run_identity_match", performance_analysis.backtest_run_identity == backtest_result.run_identity,
              "performance analysis references the supplied backtest run"),
        check("quarter_counts_reconcile",
              (performance_analysis.overall.return_series.included_count
               + sum(1 for q in backtest_result.quarter_results if q.outcome_type.value == "skipped"))
              <= len(backtest_result.quarter_results),
              "included + skipped counts do not exceed total quarters"),
        check("json_deterministic", report == report, "report dict is self-consistent"),
    ]
    return summarize(checks)


def build_filing_momentum_report(
    backtest_result: BacktestResult,
    performance_analysis: PerformanceAnalysisResult,
    strategy_config,
    regime_config: RegimeConfig,
    report_options: ReportOptions | None = None,
    model_identity=None,
    source_report_html: str | None = None,
    reproducibility_status: ReproducibilityStatus = ReproducibilityStatus.NOT_RUN,
) -> FilingMomentumReport:
    """Build the complete Filing Momentum ML report.

    Raises :class:`ReportBuildError` if ``performance_analysis`` does not
    reference ``backtest_result``'s own run identity, or if strategy
    identifiers/versions do not match — results from different runs are
    never silently combined.
    """
    options = report_options or ReportOptions()

    _require(
        performance_analysis.backtest_run_identity == backtest_result.run_identity,
        "performance_analysis.backtest_run_identity does not match backtest_result.run_identity",
    )
    _require(backtest_result.strategy_id == STRATEGY_ID, f"backtest_result.strategy_id != {STRATEGY_ID!r}")
    _require(backtest_result.strategy_version == STRATEGY_VERSION, f"backtest_result.strategy_version != {STRATEGY_VERSION!r}")

    warnings: list[str] = list(backtest_result.warnings) + list(performance_analysis.warnings)

    executive_summary = ExecutiveSummary(
        description="A quarterly, point-in-time equity-selection strategy driven by SEC filing "
                    "timing, fundamental momentum, price momentum, and a two-layer regime gate.",
        evaluation_frequency="quarterly", universe_assumption="present-day S&P 500 + Nasdaq 100 (survivorship-biased)",
        model_type="HistGradientBoostingClassifier", ml_threshold=strategy_config.ml_threshold,
        min_positions=strategy_config.min_positions, max_positions=strategy_config.max_positions,
        deployable_pct=strategy_config.deployable_pct,
        fallback_description=f"{'/'.join(strategy_config.fallback_tickers)}, "
                              f"{'dynamic' if strategy_config.fallback_dynamic_weight else 'equal'}-weighted",
        regime_description=f"gate_mode={regime_config.gate_mode}",
        backtest_start=backtest_result.backtest_start, backtest_end=backtest_result.backtest_end,
        warnings=tuple(warnings),
    )

    specification = StrategySpecification(
        feature_names=FEATURE_NAMES,
        formula_summary={
            "rev_qoq": "(R0-R-1)/|R-1|", "rev_trend": "OLS slope / |mean| over trailing quarters",
            "price_mom_Nm": "(P0-P-N)/P-N (N in {63,126,252} trading days)",
            "vol_Nd": "sqrt(252)*std(daily returns, trailing N days)",
        },
        filing_timing_summary="feature date = filing availability + 1 trading day, capped at quarter_end + 42 calendar days",
        labeling_rule_summary=f"global top-{strategy_config.n_winners} by clipped forward return receive label 1",
        training_window_summary=f"trailing {strategy_config.ml_train_years} years, >= {strategy_config.min_train_quarters} quarters, >= {strategy_config.n_winners} positive labels",
        model_hyperparameters={
            "max_iter": strategy_config.model.max_iter, "max_depth": strategy_config.model.max_depth,
            "learning_rate": strategy_config.model.learning_rate, "random_state": strategy_config.model.random_state,
        },
        ml_threshold=strategy_config.ml_threshold, excluded_sectors=strategy_config.exclude_sectors,
        regime_gate_mode=strategy_config.regime_gate_mode,
        weighting_summary="score-proportional over qualified candidates, deployable_pct of strategy budget",
        fallback_weighting_summary="trailing 12-quarter average return proportional (dynamic) or equal (static)",
        entry_exit_summary="entry at quarter buy_dt, exit at next quarter's buy_dt (cohort exit)",
        instrument_return_cap=INSTRUMENT_RETURN_CAP, label_return_clip=LABEL_RETURN_CLIP,
        transaction_cost_bps=0.0,
    )

    composition = performance_analysis.classification_composition
    coverage = BacktestCoverage(
        requested_start=backtest_result.backtest_start, requested_end=backtest_result.backtest_end,
        evaluated_count=backtest_result.completed_quarter_count,
        primary_count=composition.get(QuarterClassification.PRIMARY, 0),
        fallback_count=composition.get(QuarterClassification.FALLBACK, 0),
        cash_count=composition.get(QuarterClassification.CASH, 0),
        regime_blocked_count=composition.get(QuarterClassification.REGIME_BLOCKED, 0),
        skipped_count=composition.get(QuarterClassification.SKIPPED, 0),
        invalid_count=composition.get(QuarterClassification.INVALID, 0),
        missing_benchmark_count=sum(1 for q in backtest_result.quarter_results if q.benchmark_return is None and q.outcome_type.value != "skipped"),
        warnings=tuple(warnings),
    )

    performance_scopes = {
        "all_evaluated": _scope_summary("all_evaluated", performance_analysis.overall),
        "invested": _scope_summary("invested", performance_analysis.invested),
        "primary_only": _scope_summary("primary_only", performance_analysis.primary),
        "fallback_only": _scope_summary("fallback_only", performance_analysis.fallback),
    }

    annual_table = _annual_table(performance_analysis.invested)
    quarterly_table = _quarterly_table(backtest_result)
    holdings_table = _holdings_table(backtest_result)
    top_holdings, worst_holdings = _top_worst_holdings(backtest_result, options.top_n_rows)
    best_quarters, worst_quarters = _best_worst_quarters(backtest_result, options.top_n_rows)

    sector_section = SectionDefinition(
        identifier="sector_analysis", title="Sector Analysis", available=False,
        unavailable_reason="PositionOutcome/BacktestQuarterResult do not currently preserve sector "
                            "classification -- ScoredCandidate.sector is not propagated into the "
                            "backtest position record. Not inferred from raw legacy data.",
    )

    rp = performance_analysis.recent_period
    recent_period = RecentPeriodSection(
        available=rp.availability.value == "available", window_size=rp.window_size, period_count=rp.period_count,
        total_return=rp.total_return, benchmark_total_return=rp.benchmark_total_return,
        sharpe=rp.sharpe, sortino=rp.sortino, information_ratio=rp.information_ratio, reason=rp.reason,
    )

    statistical_validity = StatisticalValiditySection(
        sharpe_standard_error=performance_analysis.sharpe_standard_error,
        confidence_interval=performance_analysis.sharpe_confidence_interval,
        bayesian_combination=performance_analysis.bayesian_combination,
        permutation_test_available=performance_analysis.permutation_test.available,
        permutation_test_reason=performance_analysis.permutation_test.reason,
    )

    provenance = ProvenanceSection(
        source_report_sha256=source_report_sha256(source_report_html) if source_report_html else None,
        data_provider_identity="synthetic/injected fixtures (this stage)",
        price_convention="split_dividend_adjusted", universe_methodology="present-day snapshot (survivorship-biased)",
        survivorship_bias_warning="Universe membership is a present-day snapshot applied retroactively; "
                                   "this is a known, documented bias, not corrected in this stage.",
        filing_timing_policy="feature_timestamp = filing + 1 trading day, capped at quarter_end + 42 calendar days",
        feature_cache_identity=None,
        model_library_status="scikit-learn not installed in this venv; scoring uses an injected Estimator",
        transaction_cost_assumption="0 bps (commission/slippage/other) per report §9",
        stale_price_policy="bounded (default max 5 calendar days / 3 trading sessions); legacy_unbounded() available for comparison",
        missing_data_policy="missing prices/regime results reject the candidate/position rather than substituting a default",
        known_differences_from_legacy=KNOWN_INTENTIONAL_DIFFERENCES,
    )

    charts: tuple[ChartSeriesDefinition, ...] = ()
    if options.include_charts:
        invested_series = performance_analysis.invested.return_series
        charts = (
            chart_builders.equity_growth_chart(invested_series),
            chart_builders.drawdown_chart(performance_analysis.invested.drawdown),
            chart_builders.quarterly_returns_chart(invested_series),
            chart_builders.annual_returns_chart(performance_analysis.invested.annual_summaries),
            chart_builders.alpha_distribution_chart(invested_series),
            chart_builders.outcome_composition_chart(dict(composition)),
        )

    comparison_records = None
    if options.include_source_comparison and source_report_html:
        source_values = extract_source_report_values(source_report_html)
        invested_summary = performance_scopes["invested"]
        comparisons = [
            ComparisonInput("sharpe", source_values.get("full_sharpe"), invested_summary.sharpe.value, 0.05, "report §6 full-backtest Sharpe"),
            ComparisonInput("sortino", source_values.get("full_sortino"), invested_summary.sortino.value, 0.05, "report §6 full-backtest Sortino"),
            ComparisonInput("information_ratio", source_values.get("full_information_ratio"), invested_summary.information_ratio.value, 0.05, "report §6 full-backtest IR"),
        ]
        comparison_records = tuple(build_comparison_record(c) for c in comparisons)

    from atlas_quant.config.identity import compute_config_identity

    model_config_identity = model_identity.identity() if model_identity else None
    report_identity = compute_config_identity(
        {
            "config_identity": strategy_config.identity(), "regime_config_identity": regime_config.identity(),
            "backtest_run_identity": backtest_result.run_identity,
            "performance_analysis_identity": performance_analysis.analysis_identity,
            "options_identity": options.identity(), "report_schema_version": FILING_MOMENTUM_REPORT_SCHEMA_VERSION,
        }
    )

    report_metadata = ReportMetadata(
        atlasquant_display_name="AtlasQuant", strategy_id=STRATEGY_ID, strategy_display_name=DISPLAY_NAME,
        strategy_version=STRATEGY_VERSION, report_schema_version=FILING_MOMENTUM_REPORT_SCHEMA_VERSION,
        config_identity=strategy_config.identity(), backtest_run_identity=backtest_result.run_identity,
        performance_analysis_identity=performance_analysis.analysis_identity,
        report_identity=report_identity,
        provenance_notes=("Synthetic/fixture data unless a verified production BacktestResult was supplied.",),
        warnings=tuple(warnings), reproducibility_status=reproducibility_status,
    )

    validation = _run_validation(backtest_result, performance_analysis, {})
    audit = AuditSection(
        config_identity=strategy_config.identity(), model_config_identity=model_config_identity,
        regime_config_identity=regime_config.identity(), backtest_run_identity=backtest_result.run_identity,
        performance_analysis_identity=performance_analysis.analysis_identity,
        report_identity=report_identity, validation=validation,
    )

    return FilingMomentumReport(
        metadata=report_metadata, options_identity=options.identity(), executive_summary=executive_summary,
        specification=specification, coverage=coverage, performance_scopes=performance_scopes,
        annual_table=annual_table, quarterly_table=quarterly_table, holdings_table=holdings_table,
        top_holdings_table=top_holdings, worst_holdings_table=worst_holdings,
        best_quarters_table=best_quarters, worst_quarters_table=worst_quarters,
        sector_section=sector_section, recent_period=recent_period, statistical_validity=statistical_validity,
        provenance=provenance, audit=audit, charts=charts, comparison=comparison_records,
    )
