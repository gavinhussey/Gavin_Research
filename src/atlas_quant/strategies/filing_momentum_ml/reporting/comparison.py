"""Deterministic, targeted extraction of a handful of known KPI values from
report_current.html, plus the structured comparison-record builder.

This is deliberately **not** a generic web scraper — only the specific,
known KPI phrases report_current.html displays (§6 "Backtest Results"
and §7 "Recent Performance") are matched, via a small, fixed set of
regex patterns. The source file is never modified, and no network access
or browser automation is used — this operates on a local HTML string.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from atlas_quant.reporting.domain import ComparisonRecord, ComparisonStatus

#: (extraction_key, regex) pairs -- each captures a signed number
#: (optionally comma-grouped, optionally a percentage) immediately
#: preceding its known label text in report_current.html's plain-text
#: rendering (HTML tags stripped, whitespace collapsed).
_KPI_PATTERNS: tuple[tuple[str, str], ...] = (
    ("full_sharpe", r"([+-]?[\d,]+\.?\d*)\s*Sharpe Ratio"),
    ("full_sortino", r"([+-]?[\d,]+\.?\d*)\s*Sortino Ratio"),
    ("full_information_ratio", r"([+-]?[\d,]+\.?\d*)\s*Info Ratio"),
    ("full_cumulative_return_pct", r"([+-]?[\d,]+\.?\d*)%\s*Cumulative Return"),
    ("full_max_drawdown_pct", r"([+-]?[\d,]+\.?\d*)%\s*Max Drawdown"),
    ("full_win_rate_pct", r"([+-]?[\d,]+\.?\d*)%\s*Win Rate vs SPY"),
    ("full_avg_alpha_pct", r"([+-]?[\d,]+\.?\d*)%\s*Avg Alpha\s*/\s*Q(?:uarter|tr)"),
    ("full_median_alpha_pct", r"([+-]?[\d,]+\.?\d*)%\s*Median Alpha"),
    ("recent_sharpe", r"([+-]?[\d,]+\.?\d*)\s*Sharpe \(n=8\)"),
    ("recent_sortino", r"([+-]?[\d,]+\.?\d*)\s*Sortino \(n=8"),
    ("recent_total_return_pct", r"([+-]?[\d,]+\.?\d*)%\s*Total Portfolio Return"),
    ("recent_information_ratio", r"([+-]?[\d,]+\.?\d*)\s*Info Ratio"),
    ("recent_avg_alpha_pct", r"([+-]?[\d,]+\.?\d*)%\s*Avg Alpha\s*/\s*Quarter"),
    ("recent_max_drawdown_pct", r"([+-]?[\d,]+\.?\d*)%\s*Max Drawdown"),
    ("bayesian_posterior_sharpe", r"posterior Sharpe = ([+-]?[\d.]+)"),
)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def extract_source_report_values(html_text: str) -> dict[str, float | None]:
    """Extract the known KPI values report_current.html displays, by exact label match.

    Returns a value of ``None`` for any key whose pattern is not found —
    never a guessed or interpolated value. Two identically-labeled values
    (e.g. "Info Ratio" appears in both §6 and §7) are disambiguated by
    scanning the full-backtest section (before the "07 Recent Performance"
    heading) and the recent-period section (after it) separately. The
    numbered-heading form ("07 Recent Performance") is required, not the
    bare phrase "Recent Performance" -- the document's own table of
    contents lists "7. Recent Performance" long before the abstract's KPI
    tiles or §6, and matching the bare phrase would find that instead.
    """
    split_marker = "07 Recent Performance"
    split_index = html_text.find(split_marker)
    full_section = html_text if split_index == -1 else html_text[:split_index]
    recent_section = "" if split_index == -1 else html_text[split_index:]

    # bayesian_posterior_sharpe (report §8.2) is unambiguous (appears once)
    # and falls after the "07 Recent Performance" heading despite not being
    # a "recent_"-prefixed key -- searched against the full original text
    # rather than either half.
    unambiguous_keys = {"bayesian_posterior_sharpe"}

    results: dict[str, float | None] = {}
    for key, pattern in _KPI_PATTERNS:
        if key in unambiguous_keys:
            section = html_text
        else:
            section = recent_section if key.startswith("recent_") else full_section
        match = re.search(pattern, section)
        results[key] = _to_float(match.group(1)) if match else None
    return results


def source_report_sha256(html_text: str) -> str:
    return hashlib.sha256(html_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ComparisonInput:
    key: str
    source_value: float | None
    atlasquant_value: float | None
    tolerance: float | None
    explanation: str


def build_comparison_record(item: ComparisonInput) -> ComparisonRecord:
    """Compare one (source_value, atlasquant_value) pair into a typed :class:`ComparisonRecord`.

    Never compares raw strings — both sides must already be resolved to
    numeric values (or ``None``) by the caller.
    """
    if item.source_value is None and item.atlasquant_value is None:
        return ComparisonRecord(
            key=item.key, source_value=None, atlasquant_value=None, absolute_difference=None,
            relative_difference=None, tolerance=item.tolerance, status=ComparisonStatus.NOT_COMPARABLE,
            explanation=item.explanation, provenance="neither value available",
        )
    if item.source_value is None:
        return ComparisonRecord(
            key=item.key, source_value=None, atlasquant_value=item.atlasquant_value,
            absolute_difference=None, relative_difference=None, tolerance=item.tolerance,
            status=ComparisonStatus.UNAVAILABLE_IN_SOURCE, explanation=item.explanation,
            provenance="report_current.html did not contain this value under the expected label",
        )
    if item.atlasquant_value is None:
        return ComparisonRecord(
            key=item.key, source_value=item.source_value, atlasquant_value=None,
            absolute_difference=None, relative_difference=None, tolerance=item.tolerance,
            status=ComparisonStatus.UNAVAILABLE_IN_ATLASQUANT, explanation=item.explanation,
            provenance="AtlasQuant result did not produce this metric (unavailable state)",
        )

    absolute_difference = abs(item.atlasquant_value - item.source_value)
    relative_difference = (
        absolute_difference / abs(item.source_value) if item.source_value != 0 else None
    )
    tolerance = item.tolerance if item.tolerance is not None else 0.0
    if absolute_difference == 0:
        status = ComparisonStatus.MATCH
    elif absolute_difference <= tolerance:
        status = ComparisonStatus.WITHIN_TOLERANCE
    else:
        status = ComparisonStatus.DIFFERENT_UNEXPLAINED

    return ComparisonRecord(
        key=item.key, source_value=item.source_value, atlasquant_value=item.atlasquant_value,
        absolute_difference=absolute_difference, relative_difference=relative_difference,
        tolerance=item.tolerance, status=status, explanation=item.explanation,
        provenance="compared against a synthetic/fixture AtlasQuant result, not a verified production run",
    )


#: Documented, expected differences between this platform's canonical
#: implementation and the legacy prototype's behavior -- surfaced
#: alongside any numeric comparison so a real difference is never
#: silently treated as "unexplained" when it is, in fact, an intentional
#: design improvement.
KNOWN_INTENTIONAL_DIFFERENCES: tuple[str, ...] = (
    "Deterministic label tie-breaking (ascending instrument symbol) replaces the legacy "
    "prototype's incidental DataFrame.nlargest(keep='first') row-order tie-break.",
    "Bounded stale-price resolution (default max 5 calendar days / 3 trading sessions) "
    "replaces the legacy/Stage 6 unlimited backward price search.",
    "The report's two-layer HMM+Markov regime gate (market-level block and "
    "per-instrument Bear filter) is removed entirely -- this platform deliberately "
    "and permanently does not gate on regime. See docs/reproducibility_findings.md.",
    "Deterministic, fixed-vocabulary sector encoding replaces sklearn.LabelEncoder's "
    "run-order-dependent category assignment.",
    "A quarter with fewer than min_positions qualifying stocks keeps those stocks "
    "(sized off the most recent full-quota quarter's score-to-weight ratio) and puts "
    "only the unused deployable capital into the VOO/VTI ETF sleeve, replacing the "
    "report's all-or-nothing SPY/VGT fallback.",
    "Sharpe/Sortino/Information Ratio explicitly report an unavailable state (never "
    "infinity or a silently-computed value) for zero variance or insufficient "
    "observations -- including a documented divergence from the report's own "
    "single-negative-quarter Sortino example (report §7), which this platform's "
    "compute_sortino treats as insufficient history rather than reproducing.",
    "Point-in-time feature/price evaluation is enforced structurally (typed "
    "cutoffs, no-lookahead validation) rather than by convention.",
)
