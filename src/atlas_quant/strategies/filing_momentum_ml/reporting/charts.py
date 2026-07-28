"""Chart *data* extraction (pure, no rendering) plus a dependency-free inline
SVG renderer.

matplotlib is confirmed absent from this repository's venv (same status
as scikit-learn/hmmlearn) and is never required to import this module or
generate a report — chart data is plain structured
:class:`~atlas_quant.reporting.domain.ChartSeriesDefinition` values;
rendering to SVG (below) uses only the Python standard library.
"""

from __future__ import annotations

from atlas_quant.reporting.domain import ChartPoint, ChartSeriesDefinition
from atlas_quant.reporting.html import escape
from atlas_quant.strategies.filing_momentum_ml.performance_domain import (
    DrawdownMetrics,
    QuarterClassification,
    ReturnSeries,
)


def equity_growth_chart(series: ReturnSeries) -> ChartSeriesDefinition:
    """Report §6.1: strategy vs. benchmark cumulative growth."""
    strategy_growth = 1.0
    benchmark_growth = 1.0
    strategy_points = []
    benchmark_points = []
    for point in series.points:
        strategy_growth *= 1 + point.strategy_return
        strategy_points.append(ChartPoint(point.quarter_end.isoformat(), strategy_growth - 1.0))
        if point.benchmark_return is not None:
            benchmark_growth *= 1 + point.benchmark_return
        benchmark_points.append(ChartPoint(point.quarter_end.isoformat(), benchmark_growth - 1.0))
    return ChartSeriesDefinition(
        identifier="equity_growth", title="Strategy vs SPY — Cumulative Return",
        chart_type="line", series_labels=("Strategy", "SPY"),
        points=(tuple(strategy_points), tuple(benchmark_points)),
    )


def drawdown_chart(drawdown: DrawdownMetrics) -> ChartSeriesDefinition:
    """Strategy drawdown curve."""
    points = tuple(ChartPoint(p.quarter_end.isoformat(), p.drawdown) for p in drawdown.series)
    return ChartSeriesDefinition(
        identifier="drawdown", title="Strategy Drawdown", chart_type="area",
        series_labels=("Drawdown",), points=(points,),
    )


def quarterly_returns_chart(series: ReturnSeries) -> ChartSeriesDefinition:
    points = tuple(ChartPoint(p.quarter_end.isoformat(), p.strategy_return) for p in series.points)
    return ChartSeriesDefinition(
        identifier="quarterly_returns", title="Quarterly Strategy Returns", chart_type="bar",
        series_labels=("Return",), points=(points,),
    )


def annual_returns_chart(annual_summaries) -> ChartSeriesDefinition:
    strategy_points = tuple(ChartPoint(str(a.year), a.strategy_compounded_return) for a in annual_summaries)
    benchmark_points = tuple(
        ChartPoint(str(a.year), a.benchmark_compounded_return if a.benchmark_compounded_return is not None else 0.0)
        for a in annual_summaries
    )
    return ChartSeriesDefinition(
        identifier="annual_returns", title="Annual Returns — Strategy vs SPY", chart_type="bar",
        series_labels=("Strategy", "SPY"), points=(strategy_points, benchmark_points),
    )


def alpha_distribution_chart(series: ReturnSeries, bin_count: int = 10) -> ChartSeriesDefinition:
    """Report §6.3: quarterly alpha distribution, binned into ``bin_count`` histogram buckets."""
    alphas = series.alphas()
    if not alphas:
        return ChartSeriesDefinition(
            identifier="alpha_distribution", title="Quarterly Alpha Distribution",
            chart_type="bar", series_labels=("Count",), points=((),),
        )
    lo, hi = min(alphas), max(alphas)
    if lo == hi:
        return ChartSeriesDefinition(
            identifier="alpha_distribution", title="Quarterly Alpha Distribution",
            chart_type="bar", series_labels=("Count",),
            points=((ChartPoint(f"{lo:.2%}", float(len(alphas))),),),
        )
    width = (hi - lo) / bin_count
    counts = [0] * bin_count
    for a in alphas:
        idx = min(bin_count - 1, int((a - lo) / width))
        counts[idx] += 1
    points = tuple(
        ChartPoint(f"{lo + i * width:.2%}..{lo + (i + 1) * width:.2%}", float(counts[i]))
        for i in range(bin_count)
    )
    return ChartSeriesDefinition(
        identifier="alpha_distribution", title="Quarterly Alpha Distribution", chart_type="bar",
        series_labels=("Count",), points=(points,),
    )


def outcome_composition_chart(composition: dict) -> ChartSeriesDefinition:
    """Outcome composition by classification (counts) — report's own
    "27 stock-pick / 33 fallback / 0 cash" style breakdown."""
    points = tuple(
        ChartPoint(classification.value, float(count))
        for classification, count in sorted(composition.items(), key=lambda kv: kv[0].value)
    )
    return ChartSeriesDefinition(
        identifier="outcome_composition", title="Quarter Outcome Composition", chart_type="composition",
        series_labels=("Count",), points=(points,),
    )


def render_svg_bar_chart(chart: ChartSeriesDefinition, *, width: int = 480, height: int = 220) -> str:
    """Render a single-series bar/line/area/composition chart as a minimal, deterministic
    inline SVG string. Pure standard-library string formatting -- no external library."""
    if chart.is_empty():
        return f'<svg width="{width}" height="{height}"><text x="10" y="20">{escape(chart.title)} — no data</text></svg>'

    all_values = [p.y for series in chart.points for p in series]
    y_min, y_max = min(all_values + [0.0]), max(all_values + [0.0])
    y_range = (y_max - y_min) or 1.0
    plot_height = height - 40
    plot_width = width - 20

    bars = []
    n = max(len(series) for series in chart.points)
    bar_width = plot_width / max(n, 1) / max(len(chart.points), 1)
    colors = ("#2b6cb0", "#c05621", "#2f855a", "#805ad5")
    for series_index, series in enumerate(chart.points):
        color = colors[series_index % len(colors)]
        for i, point in enumerate(series):
            bar_height = abs(point.y - 0.0) / y_range * plot_height
            x = 10 + (i * len(chart.points) + series_index) * bar_width
            y_zero = 20 + (y_max / y_range) * plot_height
            y = y_zero - bar_height if point.y >= 0 else y_zero
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 1, 1):.1f}" '
                f'height="{max(bar_height, 0.5):.1f}" fill="{color}">'
                f"<title>{escape(point.x_label)}: {point.y:.4f}</title></rect>"
            )
    title = f'<text x="10" y="14" font-size="12">{escape(chart.title)}</text>'
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{title}{"".join(bars)}</svg>'
