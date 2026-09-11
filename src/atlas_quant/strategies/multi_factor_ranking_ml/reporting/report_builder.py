"""Human-readable reports for this pure ranking system: one quarterly
cycle's full ranking, and an IC backtest run's summary statistics.

Deliberately not built on this package's older ``report_model.py``/
``charts.py``/``html_template.py``/``performance_domain.py`` machinery --
those model an executive-summary/equity-curve/drawdown P&L report, which
has no meaning for a strategy with no positions, no weights, and no
capital (see ``strategy.py``'s module docstring). Those modules are left
in place (still individually correct, still tested) but are no longer
wired to this pipeline; nothing here imports them. Plain text output only
-- no HTML/SVG rendering, since there is no chart data
(equity growth, drawdown, ...) meaningful to this strategy to render.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas_quant.backtest.multi_factor_ranking_runner import ICBacktestResult
from atlas_quant.strategies.multi_factor_ranking_ml.production.decision_log import DecisionLogEntry, DecisionRanking


def _fmt_pct(value: float | None, precision: int = 2) -> str:
    return "—" if value is None else f"{value * 100:.{precision}f}%"


def _fmt_num(value: float | None, precision: int = 4) -> str:
    return "—" if value is None else f"{value:.{precision}f}"


@dataclass(frozen=True, slots=True)
class RankingReport:
    """One quarterly cycle's full ranking, ready to print or serialize."""

    entry: DecisionLogEntry

    def to_text(self, *, top_n: int | None = None) -> str:
        rankings = sorted(self.entry.rankings, key=lambda r: r.rank)
        shown = rankings[:top_n] if top_n is not None else rankings
        lines = [
            f"Multi-Factor Ranking ML -- quarter {self.entry.quarter_start.isoformat()} "
            f"(cutoff {self.entry.cutoff.isoformat()}, decided {self.entry.decided_at.isoformat()})",
            f"outcome: {self.entry.outcome}  |  {len(rankings)} instrument(s) ranked"
            + (f"  |  showing top {len(shown)}" if top_n is not None and top_n < len(rankings) else ""),
            "",
            f"{'rank':>5}  {'symbol':<12}  {'score':>10}",
        ]
        for r in shown:
            lines.append(f"{r.rank:>5}  {r.instrument_id.symbol:<12}  {r.score:>10.6f}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, object]:
        return self.entry.to_dict()


def build_ranking_report(entry: DecisionLogEntry) -> RankingReport:
    return RankingReport(entry=entry)


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """One IC backtest run's headline stats plus a per-cycle breakdown."""

    result: ICBacktestResult

    def to_text(self) -> str:
        r = self.result
        lines = [
            f"Multi-Factor Ranking ML -- IC backtest ({r.strategy_id} {r.strategy_version})",
            (
                "This measures the ranking's predictive quality (Information "
                "Coefficient / rank correlation) -- there is no equity curve, "
                "Sharpe ratio, or drawdown: this strategy holds no positions."
            ),
            "",
            (
                f"cycles: {r.cycle_count}  |  measured (next-cycle return + "
                f">= {r.min_scored_count} scored candidates): {r.measured_cycle_count}"
            ),
            f"mean IC:            {_fmt_num(r.mean_ic)}",
            f"IC std dev:         {_fmt_num(r.ic_std)}",
            f"IC information ratio: {_fmt_num(r.ic_information_ratio)}",
            f"hit rate (IC > 0):  {_fmt_pct(r.hit_rate)}",
            (
                f"mean decile spread: {_fmt_num(r.mean_decile_spread)}  "
                "(descriptive only -- top-decile minus bottom-decile mean forward "
                "return; does not imply capital deployed long/short either decile)"
            ),
            "",
            (
                f"{'quarter_start':<14}  {'training':<12}  {'ranked':>7}  {'scored_ic':>10}  "
                f"{'ic':>8}  {'decile_spread':>13}"
            ),
        ]
        for c in r.cycle_results:
            training = c.training_state.value if c.training_state else "—"
            lines.append(
                f"{c.cycle.quarter_start.isoformat():<14}  {training:<12}  {c.ranked_count:>7}  "
                f"{c.scored_for_ic_count:>10}  {_fmt_num(c.ic, 4):>8}  {_fmt_num(c.decile_spread, 4):>13}"
            )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, object]:
        return self.result.to_dict()


def build_backtest_report(result: ICBacktestResult) -> BacktestReport:
    return BacktestReport(result=result)
