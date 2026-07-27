# Filing Momentum ML — research workspace

This is the **research** home for the Filing Momentum ML strategy. It is not
production code and contains no authoritative strategy logic — every
notebook here must `import` from `atlas_quant.strategies.filing_momentum_ml`
(the production package under `src/`) rather than reimplementing formulas,
thresholds, or rules locally. A rule explored here does not become part of
the strategy until it is implemented in production code, configured
explicitly, tested, versioned, and reflected in the strategy's audit output.

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

## What belongs here vs. `src/atlas_quant/strategies/filing_momentum_ml/`

If it's a formula, threshold, qualification rule, weighting rule, or
anything the report (`report_current.html`) specifies as strategy
behavior — it belongs in the production package, tested, not here.

If it's "what if we tried a different threshold," "why did this specific
trade lose money," or "does this hold up out-of-sample" — it belongs here.
