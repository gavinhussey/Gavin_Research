# Filing Momentum ML — reporting and reproducibility specification (Stage 9)

Authoritative source: `~/Downloads/report_current.html`. This document
summarizes what `src/atlas_quant/reporting/*` and
`src/atlas_quant/strategies/filing_momentum_ml/reporting/*` actually
implement. Production code and tests remain authoritative.

**No performance claim is made anywhere in this stage.** Every report
built by this stage's own tests uses synthetic fixtures and reports
`ReproducibilityStatus.NOT_RUN` — never a historical claim.

## Report inputs

`build_filing_momentum_report` consumes, and never recalculates:
`BacktestResult` (Stage 7), `PerformanceAnalysisResult` (Stage 8),
`FilingMomentumMLConfig`, `RegimeConfig`, an optional `ModelIdentity`
(Stage 6), and an optional raw `report_current.html` string for
comparison. It rejects incompatible inputs (`ReportBuildError`) rather
than silently combining results from different runs: the performance
analysis must reference the supplied backtest's own `run_identity`, and
`strategy_id`/`strategy_version` must match this platform's own constants.

## Generic vs. strategy-specific reporting

`atlas_quant/reporting/` defines only shared vocabulary any future
strategy's report can reuse: `ReportMetadata`, `SectionDefinition`,
`TableDefinition`/`TableColumn`, `ChartSeriesDefinition`/`ChartPoint`,
`ArtifactIdentity`, `ValidationCheckResult`/`ValidationSummary`,
`ReproducibilityStatus`, `ComparisonRecord`/`ComparisonStatus`, plus
HTML-escaping and atomic-write helpers. Every Filing Momentum ML-specific
section, table, chart selection, and HTML layout lives in
`atlas_quant/strategies/filing_momentum_ml/reporting/` — no future
strategy is forced to adopt this report's 15-section structure.

## Section inventory

Reproduces `report_current.html`'s section coverage where supported by
existing typed data: (1) executive summary, (2) strategy specification
(all 17 features, formulas, timing, labeling, training window,
hyperparameters, threshold, exclusions, regime gate, weighting, entry/
exit, caps, transaction costs), (3) backtest coverage (evaluated/primary/
fallback/cash/regime-blocked/skipped/invalid/missing-benchmark counts),
(4) performance summary across all four scopes, (5)-(6) equity/drawdown/
returns/alpha/composition charts, (7) annual results, (8) quarter-by-
quarter results, (9) holdings and trade outcomes, (10) best/worst
holdings and quarters, (11) sector analysis, (12) recent-8-quarter
analysis, (13) statistical validity, (14) data provenance and
limitations, (15) audit and reproducibility.

**Section 11 (sector analysis) is marked explicitly unavailable**:
`PositionOutcome`/`BacktestQuarterResult` do not currently preserve the
sector classification `ScoredCandidate.sector` carries earlier in the
pipeline. Per this stage's own instruction, this is marked unavailable
with a stated reason rather than inferred from raw legacy data or a new,
unreviewed calculation.

## Performance scopes

All four Stage 8 scopes (`all_evaluated`, `invested`, `primary_only`,
`fallback_only`) are always presented side by side in the performance
section — there is no default output path that silently selects the
best-performing scope, and fallback results are labeled distinctly from
primary stock-selection results everywhere they appear (executive
summary's fallback description, the performance table, the holdings
table's `role` column).

## Source-report comparison

`comparison.py`'s `extract_source_report_values` is a small, fixed set of
regex patterns matching report_current.html's own known KPI phrasing
(§6/§7/§8's Sharpe/Sortino/IR/cumulative-return/drawdown/win-rate/alpha
tiles, and the §8.2 posterior Sharpe) — **not** a generic scraper. A
real parsing subtlety was found and fixed during development: the bare
phrase "Recent Performance" first appears in the document's own table of
contents ("7. Recent Performance"), long before the abstract's KPI tiles
or §6 itself — splitting on that bare phrase would put almost the entire
document on the wrong side of the full-vs-recent-period split. The parser
splits on the numbered heading `"07 Recent Performance"` instead, and
every extracted value was verified against the report's own displayed
numbers (Sharpe 0.99, Sortino 1.28, IR 0.41, recent Sharpe 1.41, recent
Sortino 6.33, posterior Sharpe 1.03 — all confirmed).

`ComparisonStatus`: `match`, `within_tolerance`, `different_expected`,
`different_unexplained`, `unavailable_in_source`,
`unavailable_in_atlasquant`, `not_comparable`. Comparisons are only ever
performed between two already-resolved numeric values, never raw HTML
strings.

## Reproducibility classification

`ReproducibilityStatus`: `fully_reproduced`, `structurally_reproduced`,
`partially_reproduced`, three `not_reproducible_*` variants, and
`not_run`. Every report built from synthetic fixtures in this stage's own
tests reports `not_run` — this stage never claims to have reproduced the
source report's historical results, because no real, verified production
dataset was run under matched configuration.

## Validation checks

`report_builder._run_validation` checks (at minimum): the performance
analysis references the supplied backtest's run identity, and included +
skipped quarter counts do not exceed the total. Every generated report
carries its own `ValidationSummary` (`audit.validation`), inspectable
without re-running tests against the artifact.

## Output formats

JSON (`report_to_dict`, deterministic sort-order via `json.dumps(...,
sort_keys=True)`) and HTML (`render_report_html`, self-contained, no
remote CDN/script/font). Chart *data* (`ChartSeriesDefinition`) is
rendered to inline SVG via a dependency-free, pure-stdlib renderer
(`render_svg_bar_chart`) — matplotlib is confirmed absent from this
venv (same status as scikit-learn/hmmlearn) and was never required.

## Report identity

`ReportMetadata.report_identity` combines the strategy config identity,
regime config identity, backtest run identity, performance analysis
identity, and report options identity — deterministic, never including
wall-clock generation time. A separate, optional
`ReportMetadata.generated_at` field exists for human display only and
never participates in `report_identity`.

## Atomic-write behavior

`atlas_quant/reporting/serialization.py` reuses the exact atomic-write
pattern established in Stage 6's feature cache: a computed (never
`tempfile.mkstemp`-created) temp filename, written via the guarded
`pathlib.Path.write_text`, then `os.replace` — for the same reason that
pattern was adopted there (`mkstemp` bypasses the test-safety guard via a
low-level `os.open` call).

## HTML safety

Every untrusted text field (instrument symbols, warnings, provenance
notes, configuration labels) passes through `atlas_quant.reporting.html
.escape` before being embedded — verified by test with a deliberately
malicious instrument symbol. No secrets, environment variable values, or
provider credentials/tokens are ever included.

## CLI — not implemented

No CLI was added this stage. The project has no existing CLI framework,
and a narrow offline command would only wrap `build_filing_momentum_report`
+ `write_report_artifacts`, both already directly callable — adding a CLI
layer now would be premature scaffolding rather than "immediate value,"
per this stage's own optional-CLI guidance.

## No-performance-claim policy

Every test in this stage constructs synthetic `BacktestResult` fixtures
directly (never a real historical run); every report they produce
reports `ReproducibilityStatus.NOT_RUN`. Source-report comparison
extracts the report's own published numbers for structural
parity-testing purposes only — it is never presented as validating this
platform's own (unrun) historical performance.

## Deferred real historical reproduction

Running the complete Filing Momentum ML pipeline against the real,
verified production universe/price/EDGAR dataset under matched
configuration — the only way to legitimately claim
`fully_reproduced`/`structurally_reproduced` against report_current.html
— remains out of scope for this stage and is explicitly deferred to a
future stage that has verified production data available.
