# Multi-Factor Ranking ML — research workspace

This is the **research** home for the Multi-Factor Ranking ML strategy —
an independent strategy cloned from `filing_momentum_ml`'s pipeline
*architecture* (point-in-time feature/label plumbing, model training,
scoring, backtest/reporting orchestration), not from its results, feature
set, or data provenance. It's a separate strategy the user is building on
top of this scaffold: a larger universe than S&P 500 + Nasdaq 100, and its
own features (not filing_momentum_ml's 17 filing-derived ones), fed from
Bloomberg CSV exports rather than SEC EDGAR. See
`docs/reproducibility_findings.md` for exactly what is and isn't real yet.

This is not production code and contains no authoritative strategy logic
— every notebook here must `import` from
`atlas_quant.strategies.multi_factor_ranking_ml` (the production package
under `src/`) rather than reimplementing formulas, thresholds, or rules
locally. A rule explored here does not become part of the strategy until
it is implemented in production code, configured explicitly, tested,
versioned, and reflected in the strategy's audit output.

## Layout

- `notebooks/research/` — hypothesis development and exploratory analysis.
- `notebooks/experiments/` — parameter experiments, model comparisons,
  feature investigations, alternate methods.
- `notebooks/validation/` — robustness testing, out-of-sample checks,
  parity work, methodology verification.
- `notebooks/diagnostics/` — investigating specific trades, model
  decisions, data problems, unexpected outcomes, failed assumptions.
- `fixtures/` — small, checked-in data snapshots used by notebooks (not
  production caches).
- `reports/` — generated research reports/exports specific to this
  strategy.
- `docs/` — strategy-specific research notes (distinct from `docs/` at the
  repo root, which covers the platform).

## What belongs here vs. `src/atlas_quant/strategies/multi_factor_ranking_ml/`

If it's a formula, threshold, qualification rule, or weighting rule that
defines this strategy's own behavior — it belongs in the production
package, tested, not here. (Unlike filing_momentum_ml, there is no
`report_current.html` this strategy is trying to match — its rules are
whatever gets implemented and tested here, full stop.)

If it's "what if we tried a different threshold," "why did this specific
trade lose money," or "does this hold up out-of-sample" — it belongs here.
