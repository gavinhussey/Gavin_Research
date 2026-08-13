#!/usr/bin/env python
"""filing_momentum_ml's standard backtest: runs the same production backtest
as `atlas-quant filing-momentum run-backtest`, then prints the performance
statistics (Sharpe, Sortino, drawdown, alpha, ...) that command itself
doesn't surface -- `run-backtest` only prints run state/identity; the stats
come from a separate call into the strategy's own analysis code
(`analyze_backtest_result`), the same call `build-report` makes internally.

One subfolder per strategy lives under `backtest/` (this is
filing_momentum_ml's); add a sibling subfolder for each new strategy rather
than growing this one to cover more than one. Within a strategy's subfolder,
add one file per kind of backtest as needed (this is the standard one).

Edit the constants below to change the backtest window or flags, then run:

    .venv/bin/python backtest/filing_momentum_ml/run_backtest.py

Mirrors (via the same internal helpers, not a CLI subprocess) the CLI
command documented in
research/strategies/filing_momentum_ml/docs/production_backtest_specification.md.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- edit these to change what gets run ---
START_QUARTER = "2015-03-31"
END_QUARTER = "2026-06-30"
BENCHMARK = "SPY"
JSON_OUTPUT = True
# ------------------------------------------

RAW_ROOT = REPO_ROOT / "data" / "raw" / "filing_momentum_ml"
MANIFEST = REPO_ROOT / "data" / "manifests" / "filing_momentum_ml" / "data_manifest.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STATS_OUTPUT = OUTPUT_DIR / "strategy_statistics.txt"
EQUITY_CURVE_OUTPUT = OUTPUT_DIR / "equity_curve.jpg"

_WHITE = (255, 255, 255)
_BLACK = (30, 30, 30)
_GRID = (222, 226, 232)
_STRATEGY = (20, 96, 176)
_BENCHMARK = (220, 115, 35)


def _print_stats(performance, *, as_json: bool, stream) -> None:
    from atlas_quant.strategies.filing_momentum_ml.performance_analysis import performance_analysis_to_dict

    if as_json:
        stream.write(json.dumps(performance_analysis_to_dict(performance), indent=2, sort_keys=True) + "\n")
        return

    def _fmt_scope(label: str, scope) -> None:
        sharpe = scope.sharpe.value
        sortino = scope.sortino.value
        stream.write(
            f"\n{label} ({scope.return_series.included_count} quarters):\n"
            f"  total_return={scope.metrics.total_return:.1%} "
            f"benchmark_total_return={scope.metrics.benchmark_total_return:.1%}\n"
            f"  sharpe={sharpe:.2f} sortino={'n/a' if sortino is None else f'{sortino:.2f}'} "
            f"information_ratio={scope.information_ratio.value:.2f}\n"
            f"  max_drawdown={scope.drawdown.max_drawdown:.1%} "
            f"({scope.drawdown.max_drawdown_peak_date} -> {scope.drawdown.max_drawdown_trough_date})\n"
            f"  win_rate={scope.metrics.win_rate:.1%}\n"
        )

    _fmt_scope("invested", performance.invested)
    _fmt_scope("primary-only", performance.primary)
    _fmt_scope("fallback-only", performance.fallback)
    stream.write(
        f"\nsharpe_confidence_interval (95%): "
        f"[{performance.sharpe_confidence_interval.lower_bound:.2f}, "
        f"{performance.sharpe_confidence_interval.upper_bound:.2f}]\n"
    )
    if performance.warnings:
        for w in performance.warnings:
            stream.write(f"warning: {w}\n")


def _write_statistics_file(result, performance) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    buffer.write("Filing Momentum ML Backtest Statistics\n")
    buffer.write("=" * 42 + "\n\n")
    buffer.write(f"window: {START_QUARTER} through {END_QUARTER}\n")
    buffer.write(f"benchmark: {BENCHMARK}\n")
    buffer.write(f"state: {result.state.value}\n")
    buffer.write(f"run_identity: {result.run_identity}\n")
    buffer.write(f"analysis_identity: {performance.analysis_identity}\n\n")
    _print_stats(performance, as_json=False, stream=buffer)
    STATS_OUTPUT.write_text(buffer.getvalue())


def _blank_canvas(width: int, height: int, color=_WHITE) -> list[list[tuple[int, int, int]]]:
    return [[color for _ in range(width)] for _ in range(height)]


def _set_pixel(canvas, x: int, y: int, color) -> None:
    if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
        canvas[y][x] = color


def _draw_line(canvas, x0: int, y0: int, x1: int, y1: int, color, width: int = 2) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for ox in range(-(width // 2), width // 2 + 1):
            for oy in range(-(width // 2), width // 2 + 1):
                _set_pixel(canvas, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


_FONT = {
    " ": ("000", "000", "000", "000", "000", "000", "000"),
    ".": ("0", "0", "0", "0", "0", "0", "1"),
    "$": ("01110", "10100", "10100", "01110", "00101", "00101", "11110"),
    "&": ("0110", "1001", "1001", "0110", "1011", "1001", "0111"),
    "%": ("10001", "00010", "00100", "01000", "10000", "00000", "10001"),
    "-": ("000", "000", "000", "111", "000", "000", "000"),
    "/": ("0001", "0010", "0010", "0100", "0100", "1000", "1000"),
    "0": ("111", "101", "101", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "010", "010", "111"),
    "2": ("111", "001", "001", "111", "100", "100", "111"),
    "3": ("111", "001", "001", "111", "001", "001", "111"),
    "4": ("101", "101", "101", "111", "001", "001", "001"),
    "5": ("111", "100", "100", "111", "001", "001", "111"),
    "6": ("111", "100", "100", "111", "101", "101", "111"),
    "7": ("111", "001", "001", "010", "010", "100", "100"),
    "8": ("111", "101", "101", "111", "101", "101", "111"),
    "9": ("111", "101", "101", "111", "001", "001", "111"),
    "A": ("010", "101", "101", "111", "101", "101", "101"),
    "B": ("110", "101", "101", "110", "101", "101", "110"),
    "C": ("011", "100", "100", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "101", "101", "110"),
    "E": ("111", "100", "100", "110", "100", "100", "111"),
    "F": ("111", "100", "100", "110", "100", "100", "100"),
    "G": ("011", "100", "100", "101", "101", "101", "011"),
    "H": ("101", "101", "101", "111", "101", "101", "101"),
    "I": ("111", "010", "010", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "100", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "100", "100", "111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("1001", "1101", "1011", "1001", "1001", "1001", "1001"),
    "O": ("010", "101", "101", "101", "101", "101", "010"),
    "P": ("110", "101", "101", "110", "100", "100", "100"),
    "Q": ("010", "101", "101", "101", "101", "010", "001"),
    "R": ("110", "101", "101", "110", "110", "101", "101"),
    "S": ("011", "100", "100", "010", "001", "001", "110"),
    "T": ("111", "010", "010", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "101", "101", "010"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("101", "101", "101", "010", "101", "101", "101"),
    "Y": ("101", "101", "101", "010", "010", "010", "010"),
    "Z": ("111", "001", "001", "010", "100", "100", "111"),
}


def _draw_text(canvas, x: int, y: int, text: str, color=_BLACK, scale: int = 2) -> None:
    cursor = x
    for ch in text.upper():
        glyph = _FONT.get(ch, _FONT[" "])
        for row_i, row in enumerate(glyph):
            for col_i, bit in enumerate(row):
                if bit == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            _set_pixel(canvas, cursor + col_i * scale + sx, y + row_i * scale + sy, color)
        cursor += (len(glyph[0]) + 1) * scale


def _write_ppm(canvas, path: Path) -> None:
    height = len(canvas)
    width = len(canvas[0])
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        for row in canvas:
            for r, g, b in row:
                f.write(bytes((r, g, b)))


def _price_on_or_before(prices, day):
    eligible = [p for p in prices if p.trading_date <= day]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p.trading_date)


def _daily_equity_curve(backtest_result, price_source, benchmark_instrument_id):
    from atlas_quant.backtest.accounting import INSTRUMENT_RETURN_CAP, apply_return_cap

    strategy_base = 1.0
    benchmark_base = 1.0
    daily: list[tuple[object, float, float]] = []
    benchmark_prices = tuple(price_source.get(benchmark_instrument_id, ()))

    for quarter in backtest_result.quarter_results:
        start = quarter.period.entry_timestamp.date()
        end = quarter.period.exit_timestamp.date()
        benchmark_entry = _price_on_or_before(benchmark_prices, start)
        period_days = sorted(
            {
                p.trading_date
                for p in benchmark_prices
                if start <= p.trading_date <= end
            }
        )
        if not period_days:
            period_days = [quarter.period.quarter_end]

        for day in period_days:
            if daily and day <= daily[-1][0]:
                continue

            strategy_period_return = 0.0
            for position in quarter.positions:
                entry_price = position.entry_resolved.price if position.entry_resolved else None
                if not entry_price or entry_price <= 0:
                    continue
                position_prices = tuple(price_source.get(position.instrument_id, ()))
                current = _price_on_or_before(position_prices, day)
                if current is None:
                    continue
                raw_return = current.close / entry_price - 1
                strategy_period_return += position.target_weight * apply_return_cap(
                    raw_return, INSTRUMENT_RETURN_CAP
                )

            benchmark_period_return = 0.0
            benchmark_current = _price_on_or_before(benchmark_prices, day)
            if benchmark_entry is not None and benchmark_entry.close > 0 and benchmark_current is not None:
                benchmark_period_return = benchmark_current.close / benchmark_entry.close - 1

            daily.append(
                (
                    day,
                    strategy_base * (1 + strategy_period_return),
                    benchmark_base * (1 + benchmark_period_return),
                )
            )

        if quarter.period_return is not None:
            strategy_base *= 1 + quarter.period_return
        if quarter.benchmark_return is not None:
            benchmark_base *= 1 + quarter.benchmark_return

    return tuple(daily)


def _write_equity_curve_jpg(backtest_result, price_source, benchmark_instrument_id) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = _daily_equity_curve(backtest_result, price_source, benchmark_instrument_id)
    if not curve:
        return

    width, height = 1200, 720
    left, right, top, bottom = 90, 40, 80, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    canvas = _blank_canvas(width, height)

    dates = [row[0] for row in curve]
    strategy = [row[1] for row in curve]
    benchmark = [row[2] for row in curve]
    y_min = min(1.0, min(strategy), min(benchmark))
    y_max = max(strategy + benchmark)
    y_range = y_max - y_min if y_max > y_min else 1.0

    def xy(index: int, value: float) -> tuple[int, int]:
        x = left + round(index / max(1, len(curve) - 1) * plot_w)
        y = top + round((y_max - value) / y_range * plot_h)
        return x, y

    for i in range(6):
        y = top + round(i / 5 * plot_h)
        _draw_line(canvas, left, y, width - right, y, _GRID, width=1)
    for i in range(0, len(curve), max(1, len(curve) // 8)):
        x, _ = xy(i, y_min)
        _draw_line(canvas, x, top, x, height - bottom, _GRID, width=1)

    _draw_line(canvas, left, top, left, height - bottom, _BLACK, width=1)
    _draw_line(canvas, left, height - bottom, width - right, height - bottom, _BLACK, width=1)

    for series, color in ((benchmark, _BENCHMARK), (strategy, _STRATEGY)):
        points = [xy(i, value) for i, value in enumerate(series)]
        for a, b in zip(points, points[1:]):
            _draw_line(canvas, a[0], a[1], b[0], b[1], color, width=3)

    _draw_text(canvas, 90, 28, "FILING MOMENTUM ML EQUITY CURVE", _BLACK, scale=3)
    _draw_line(canvas, 90, 58, 135, 58, _STRATEGY, width=4)
    _draw_text(canvas, 145, 51, f"STRATEGY {strategy[-1]:.2f}X", _BLACK, scale=2)
    _draw_line(canvas, 345, 58, 390, 58, _BENCHMARK, width=4)
    _draw_text(canvas, 400, 51, f"S&P 500 {benchmark[-1]:.2f}X", _BLACK, scale=2)
    _draw_text(canvas, 90, height - 48, f"{dates[0].isoformat()} / {dates[-1].isoformat()}", _BLACK, scale=2)
    _draw_text(canvas, width - 250, height - 48, "GROWTH OF $1", _BLACK, scale=2)

    for i, value in enumerate((y_min, y_max)):
        label = f"{value:.1f}X"
        y = height - bottom if i == 0 else top
        _draw_text(canvas, 18, y - 7, label, _BLACK, scale=2)

    sips = shutil.which("sips")
    if sips is None:
        raise RuntimeError("cannot write JPG equity curve because macOS 'sips' was not found")
    with tempfile.TemporaryDirectory() as tmp:
        ppm_path = Path(tmp) / "equity_curve.ppm"
        _write_ppm(canvas, ppm_path)
        subprocess.run(
            [sips, "-s", "format", "jpeg", str(ppm_path), "--out", str(EQUITY_CURVE_OUTPUT)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> int:
    from atlas_quant.cli.filing_momentum import (
        _COMPLETED_STATES,
        _build_run_inputs,
        _build_trading_calendar,
        _exit_code_for_state,
        _periods_from_args,
        _print_run_result,
        load_normalized_bundle,
    )
    from atlas_quant.strategies.filing_momentum_ml.production.orchestration import (
        run_filing_momentum_production_backtest,
    )

    # dry_run=False matches this script's prior behavior: real checkpoint
    # manifests get written to DEFAULT_CHECKPOINT_ROOT, same as before this
    # script printed stats.
    args = argparse.Namespace(
        raw_root=RAW_ROOT, manifest=MANIFEST,
        start_quarter=START_QUARTER, end_quarter=END_QUARTER,
        benchmark=BENCHMARK, earnings_lag_days=None,
        dry_run=False, checkpoint_root=None, source_report_html=None,
    )

    bundle = load_normalized_bundle(args.raw_root)
    calendar = _build_trading_calendar(bundle)
    periods = _periods_from_args(args, required=True)
    inputs = _build_run_inputs(args, bundle, calendar, periods, with_report=False)
    result = run_filing_momentum_production_backtest(inputs)

    _print_run_result(result, as_json=JSON_OUTPUT, stream=sys.stdout)
    if result.performance_analysis is not None:
        _print_stats(result.performance_analysis, as_json=JSON_OUTPUT, stream=sys.stdout)
    if result.backtest_result is not None and result.performance_analysis is not None:
        _write_statistics_file(result, result.performance_analysis)
        _write_equity_curve_jpg(result.backtest_result, inputs.prices_by_instrument, inputs.benchmark_instrument_id)
        sys.stdout.write(f"wrote: {STATS_OUTPUT}\n")
        sys.stdout.write(f"wrote: {EQUITY_CURVE_OUTPUT}\n")

    return 0 if result.state in _COMPLETED_STATES else _exit_code_for_state(result.state)


if __name__ == "__main__":
    sys.exit(main())
