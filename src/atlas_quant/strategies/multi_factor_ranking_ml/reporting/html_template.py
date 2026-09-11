"""Renders a MultiFactorRankingReport into a self-contained HTML document.

No remote CDN, no network access, no external stylesheet/script/font.
Every untrusted text field (instrument symbols, warnings, provenance
notes, labels) is passed through :func:`atlas_quant.reporting.html.escape`
before being embedded.
"""

from __future__ import annotations

from atlas_quant.reporting.html import escape, wrap_page
from atlas_quant.strategies.multi_factor_ranking_ml.reporting.charts import render_svg_bar_chart
from atlas_quant.strategies.multi_factor_ranking_ml.reporting.report_model import MultiFactorRankingReport

_STYLE = (
    "body{font-family:sans-serif;margin:2em;color:#1a202c}"
    "h1,h2{color:#2b6cb0}table{border-collapse:collapse;margin:1em 0}"
    "th,td{border:1px solid #cbd5e0;padding:4px 8px;font-size:0.9em}"
    "th{background:#edf2f7}.warn{color:#c05621}.unavailable{color:#a0aec0;font-style:italic}"
)


def _fmt(value, kind: str, percent_precision: int = 2, decimal_precision: int = 4) -> str:
    if value is None:
        return "—"
    if kind == "percent":
        return f"{value * 100:.{percent_precision}f}%"
    if kind == "number" and isinstance(value, float):
        return f"{value:.{decimal_precision}f}"
    return escape(value)


def _render_table(table, percent_precision: int, decimal_precision: int) -> str:
    header = "".join(f"<th>{escape(c.label)}</th>" for c in table.columns)
    body_rows = []
    for row in table.rows:
        cells = "".join(
            f"<td>{_fmt(row.get(c.key), c.kind, percent_precision, decimal_precision)}</td>"
            for c in table.columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f"<h3>{escape(table.title)}</h3>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def _metric_inline(result) -> str:
    if result is None:
        return "unavailable"
    if result.availability.value != "available":
        return f'<span class="unavailable">unavailable ({escape(result.reason)})</span>'
    return f"{result.value:.4f} (n={result.observation_count})"


def _render_metric(name: str, result) -> str:
    return f"<li>{escape(name)}: {_metric_inline(result)}</li>"


def render_report_html(report: MultiFactorRankingReport, *, include_charts: bool = True) -> str:
    """Render ``report`` into a self-contained HTML string. Never recalculates
    any value — only formats fields already present on ``report``."""
    pp = 2
    dp = 4

    sections = []

    sections.append(
        f"<h1>{escape(report.metadata.strategy_display_name)} — Strategy Report</h1>"
        f"<p><em>{escape(report.metadata.atlasquant_display_name)} · "
        f"schema v{escape(report.metadata.report_schema_version)} · "
        f"reproducibility: {escape(report.metadata.reproducibility_status.value)}</em></p>"
    )
    if report.metadata.warnings:
        warn_items = "".join(f"<li class='warn'>{escape(w)}</li>" for w in report.metadata.warnings)
        sections.append(f"<ul>{warn_items}</ul>")

    es = report.executive_summary
    sections.append(
        "<h2>1. Executive Summary</h2>"
        f"<p>{escape(es.description)}</p>"
        f"<ul><li>Evaluation frequency: {escape(es.evaluation_frequency)}</li>"
        f"<li>Universe: {escape(es.universe_assumption)}</li>"
        f"<li>Model: {escape(es.model_type)}</li>"
        f"<li>Threshold: {es.ml_threshold:.2f}</li>"
        f"<li>Positions: {es.min_positions}-{es.max_positions}</li>"
        f"<li>Deployable: {_fmt(es.deployable_pct, 'percent', pp)}</li>"
        f"<li>Fallback: {escape(es.fallback_description)}</li>"
        f"<li>Backtest period: {escape(es.backtest_start)} – {escape(es.backtest_end)}</li></ul>"
    )

    spec = report.specification
    sections.append(
        "<h2>2. Strategy Specification</h2>"
        f"<p>17 features: {escape(', '.join(spec.feature_names))}</p>"
        f"<p>{escape(spec.filing_timing_summary)}</p>"
        f"<p>{escape(spec.labeling_rule_summary)}</p>"
        f"<p>{escape(spec.training_window_summary)}</p>"
        f"<p>Sector exclusion: {escape(', '.join(spec.excluded_sectors))}</p>"
        f"<p>{escape(spec.weighting_summary)}</p>"
        f"<p>{escape(spec.fallback_weighting_summary)}</p>"
        f"<p>{escape(spec.entry_exit_summary)}</p>"
        f"<p>Instrument return cap: {_fmt(spec.instrument_return_cap, 'percent', pp)} "
        f"(label clip: {_fmt(spec.label_return_clip, 'percent', pp)})</p>"
        f"<p>Transaction costs: {spec.transaction_cost_bps} bps</p>"
    )

    cov = report.coverage
    sections.append(
        "<h2>3. Backtest Coverage</h2>"
        f"<ul><li>Evaluated: {cov.evaluated_count}</li><li>Primary: {cov.primary_count}</li>"
        f"<li>Fallback: {cov.fallback_count}</li><li>Cash: {cov.cash_count}</li>"
        f"<li>Skipped: {cov.skipped_count}</li>"
        f"<li>Invalid: {cov.invalid_count}</li><li>Missing benchmark: {cov.missing_benchmark_count}</li></ul>"
    )

    perf_rows = []
    for key, summary in report.performance_scopes.items():
        perf_rows.append(
            f"<tr><td>{escape(summary.scope_label)}</td><td>{summary.included_count}</td>"
            f"<td>{_fmt(summary.total_return, 'percent', pp)}</td>"
            f"<td>{_fmt(summary.benchmark_total_return, 'percent', pp)}</td>"
            f"<td>{_metric_inline(summary.sharpe)}</td>"
            f"<td>{_fmt(summary.win_rate, 'percent', pp)}</td></tr>"
        )
    sections.append(
        "<h2>4. Performance Summary</h2>"
        "<table><thead><tr><th>Scope</th><th>N</th><th>Total Return</th><th>SPY Return</th>"
        "<th>Sharpe</th><th>Win Rate</th></tr></thead><tbody>"
        + "".join(perf_rows) + "</tbody></table>"
        "<p><em>Fallback-scope results are SPY/VGT outcomes, never stock-selection alpha.</em></p>"
    )

    if include_charts and report.charts:
        chart_html = "".join(f"<div>{render_svg_bar_chart(c)}</div>" for c in report.charts)
        sections.append(f"<h2>5-6. Equity, Drawdown, and Distribution Charts</h2>{chart_html}")

    sections.append(f"<h2>7. Annual Results</h2>{_render_table(report.annual_table, pp, dp)}")
    sections.append(f"<h2>8. Quarter-by-Quarter Results</h2>{_render_table(report.quarterly_table, pp, dp)}")
    sections.append(f"<h2>9. Holdings and Trade Outcomes</h2>{_render_table(report.holdings_table, pp, dp)}")
    sections.append(
        "<h2>10. Best and Worst Outcomes</h2>"
        + _render_table(report.top_holdings_table, pp, dp)
        + _render_table(report.worst_holdings_table, pp, dp)
        + _render_table(report.best_quarters_table, pp, dp)
        + _render_table(report.worst_quarters_table, pp, dp)
    )

    if report.sector_section.available:
        sections.append("<h2>11. Sector Analysis</h2><p>Available.</p>")
    else:
        sections.append(
            f"<h2>11. Sector Analysis</h2><p class='unavailable'>Unavailable: "
            f"{escape(report.sector_section.unavailable_reason)}</p>"
        )

    rp = report.recent_period
    if rp.available:
        sections.append(
            "<h2>12. Recent 8-Quarter Analysis</h2>"
            f"<ul>{_render_metric('Sharpe', rp.sharpe)}{_render_metric('Sortino', rp.sortino)}"
            f"{_render_metric('Information Ratio', rp.information_ratio)}</ul>"
            f"<p>Total return: {_fmt(rp.total_return, 'percent', pp)}</p>"
        )
    else:
        sections.append(f"<h2>12. Recent 8-Quarter Analysis</h2><p class='unavailable'>{escape(rp.reason)}</p>")

    sv = report.statistical_validity
    ci = sv.confidence_interval
    ci_text = (
        f"{ci.confidence_level:.0%} CI [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]"
        if ci.availability.value == "available" else f"unavailable ({escape(ci.reason)})"
    )
    sections.append(
        "<h2>13. Statistical Validity</h2>"
        f"<ul>{_render_metric('Sharpe SE', sv.sharpe_standard_error)}"
        f"<li>Confidence interval: {ci_text}</li>"
        f"<li>Permutation test: {'available' if sv.permutation_test_available else 'deferred — ' + escape(sv.permutation_test_reason)}</li></ul>"
        "<p><em>Analytical approximations, not proof of statistical significance.</em></p>"
    )

    prov = report.provenance
    sections.append(
        "<h2>14. Data Provenance and Limitations</h2>"
        f"<ul><li>Price convention: {escape(prov.price_convention)}</li>"
        f"<li>Universe: {escape(prov.universe_methodology)}</li>"
        f"<li class='warn'>{escape(prov.survivorship_bias_warning)}</li>"
        f"<li>{escape(prov.filing_timing_policy)}</li>"
        f"<li>{escape(prov.model_library_status)}</li>"
        f"<li>{escape(prov.transaction_cost_assumption)}</li>"
        f"<li>{escape(prov.stale_price_policy)}</li>"
        f"<li>{escape(prov.missing_data_policy)}</li></ul>"
        "<p>Known intentional differences from the legacy implementation:</p><ul>"
        + "".join(f"<li>{escape(d)}</li>" for d in prov.known_differences_from_legacy)
        + "</ul>"
    )

    audit = report.audit
    validation_items = "".join(
        f"<li>{'✓' if c.passed else '✗'} {escape(c.name)}: {escape(c.message)}</li>" for c in audit.validation.checks
    )
    sections.append(
        "<h2>15. Audit and Reproducibility</h2>"
        f"<ul><li>Config identity: {escape(audit.config_identity)}</li>"
        f"<li>Backtest run identity: {escape(audit.backtest_run_identity)}</li>"
        f"<li>Performance analysis identity: {escape(audit.performance_analysis_identity)}</li>"
        f"<li>Report identity: {escape(audit.report_identity)}</li></ul>"
        f"<p>Validation ({'all passed' if audit.validation.all_passed else 'FAILURES PRESENT'}):</p>"
        f"<ul>{validation_items}</ul>"
    )

    if report.comparison:
        comparison_rows = "".join(
            f"<tr><td>{escape(c.key)}</td><td>{escape(c.source_value)}</td>"
            f"<td>{escape(c.atlasquant_value)}</td><td>{escape(c.status.value)}</td>"
            f"<td>{escape(c.explanation)}</td></tr>"
            for c in report.comparison
        )
        sections.append(
            "<h2>Comparison with report_current.html</h2>"
            "<table><thead><tr><th>Key</th><th>Source</th><th>AtlasQuant</th><th>Status</th><th>Explanation</th></tr></thead>"
            f"<tbody>{comparison_rows}</tbody></table>"
        )

    return wrap_page(f"{report.metadata.strategy_display_name} Report", "\n".join(sections), style=_STYLE)
