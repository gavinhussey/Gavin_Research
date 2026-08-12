"""Read-only per-ETF audit table for one month-end selection.

Combines :func:`pipeline.select_for_month_end`'s already-computed
``MonthlySelectionResult`` into a single flat table exposing every
intermediate value the Total Rank decision depends on -- not just the
final weights -- for implementation-audit and historical-behavior review
(2026-08-06 audit stage). This module performs no new computation of its
own; it only reads fields already produced by :mod:`pipeline` and
:mod:`formulas`, so it carries no lookahead or strategy-logic risk of its
own and is never called from live selection/execution paths.

Requires ``total_rank_formula="full_provisional"`` (the only mode that
populates ``total_rank_audit`` with per-term contributions) -- raises
otherwise rather than silently returning a partial table.
"""

from __future__ import annotations

import pandas as pd

from atlas_quant.strategies.ranked_multi_factor_rotation.pipeline import MonthlySelectionResult

_NAN = float("nan")

ELIGIBLE = "eligible"
INELIGIBLE_MISSING_DATA = "ineligible_missing_data"

SELECTED_LONG = "selected_long"
SELECTED_CASH_GATED = "selected_cash_gated"
NOT_SELECTED = "not_selected"

EXCLUSION_MISSING_INPUT_DATA = "missing_input_data"
EXCLUSION_NEGATIVE_OR_ZERO_MOMENTUM = "negative_or_zero_momentum"
EXCLUSION_TOTAL_RANK_NOT_IN_LOWEST_N = "total_rank_not_in_lowest_n"

DIAGNOSTIC_TABLE_COLUMNS = (
    "asset_return_4m",
    "shy_return_4m",
    "absolute_momentum",
    "momentum_rank",
    "volatility",
    "volatility_rank",
    "correlation",
    "correlation_rank",
    "trend_score",
    "momentum_contribution",
    "volatility_contribution",
    "correlation_contribution",
    "raw_numerator",
    "divisor",
    "total_rank",
    "eligibility",
    "selection_status",
    "exclusion_reason",
)


def build_diagnostic_table(result: MonthlySelectionResult) -> pd.DataFrame:
    """One row per ``config.ranked_tickers`` entry, columns per
    :data:`DIAGNOSTIC_TABLE_COLUMNS` -- every named field an implementation
    audit needs to see per ETF per rebalance date (spec §4-audit task 5).

    ``result.total_rank_audit`` must be populated (i.e. ``result`` was
    produced with ``total_rank_formula="full_provisional"``); a ``None``
    per-ticker entry there means that ticker's rank/trend/M inputs were
    not all finite that month, and this table reports it as
    ``eligibility="ineligible_missing_data"`` with every numeric audit
    column ``NaN``, rather than raising.
    """
    if result.total_rank_audit is None:
        raise ValueError(
            "build_diagnostic_table requires a MonthlySelectionResult built with "
            "total_rank_formula='full_provisional' (total_rank_audit is None -- "
            "this result was built under 'legacy', which has no per-term audit)"
        )

    selected_set = set(result.selected_tickers)
    positive_momentum_selected = {
        t for t in result.selected_tickers if result.momentum_values.loc[t] > 0
    }

    rows: dict[str, dict[str, object]] = {}
    for ticker in result.total_rank_audit.keys():
        audit = result.total_rank_audit[ticker]
        is_selected = ticker in selected_set

        if audit is None:
            eligibility = INELIGIBLE_MISSING_DATA
            selection_status = NOT_SELECTED
            exclusion_reason = EXCLUSION_MISSING_INPUT_DATA
            rows[ticker] = {
                "asset_return_4m": result.asset_return_4m.get(ticker, _NAN),
                "shy_return_4m": result.shy_return_4m.get(ticker, _NAN),
                "absolute_momentum": result.absolute_momentum_excess.get(ticker, _NAN),
                "momentum_rank": _NAN,
                "volatility": result.volatility_values.get(ticker, _NAN),
                "volatility_rank": _NAN,
                "correlation": result.correlation_values.get(ticker, _NAN),
                "correlation_rank": _NAN,
                "trend_score": result.trend_values.get(ticker, _NAN),
                "momentum_contribution": _NAN,
                "volatility_contribution": _NAN,
                "correlation_contribution": _NAN,
                "raw_numerator": _NAN,
                "divisor": _NAN,
                "total_rank": _NAN,
                "eligibility": eligibility,
                "selection_status": selection_status,
                "exclusion_reason": exclusion_reason,
            }
            continue

        eligibility = ELIGIBLE
        if not is_selected:
            selection_status = NOT_SELECTED
            exclusion_reason = EXCLUSION_TOTAL_RANK_NOT_IN_LOWEST_N
        elif ticker in positive_momentum_selected:
            selection_status = SELECTED_LONG
            exclusion_reason = None
        else:
            selection_status = SELECTED_CASH_GATED
            exclusion_reason = EXCLUSION_NEGATIVE_OR_ZERO_MOMENTUM

        rows[ticker] = {
            "asset_return_4m": result.asset_return_4m.get(ticker, _NAN),
            "shy_return_4m": result.shy_return_4m.get(ticker, _NAN),
            "absolute_momentum": audit.absolute_momentum,
            "momentum_rank": audit.momentum_rank,
            "volatility": result.volatility_values.get(ticker, _NAN),
            "volatility_rank": audit.volatility_rank,
            "correlation": result.correlation_values.get(ticker, _NAN),
            "correlation_rank": audit.correlation_rank,
            "trend_score": audit.trend_score,
            "momentum_contribution": audit.momentum_contribution,
            "volatility_contribution": audit.volatility_contribution,
            "correlation_contribution": audit.correlation_contribution,
            "raw_numerator": audit.raw_numerator,
            "divisor": audit.divisor,
            "total_rank": audit.total_rank,
            "eligibility": eligibility,
            "selection_status": selection_status,
            "exclusion_reason": exclusion_reason,
        }

    table = pd.DataFrame.from_dict(rows, orient="index", columns=list(DIAGNOSTIC_TABLE_COLUMNS))
    return table.sort_values("total_rank", na_position="last")
