# Ranked Multi-Factor Rotation — research workspace

Scaffold only, mirroring `research/strategies/filing_momentum_ml/`'s
layout per `docs/adding_a_strategy.md`. No specification, formula, or
result in this directory is authoritative until it also exists as tested
code under
`src/atlas_quant/strategies/ranked_multi_factor_rotation/` — this
directory is never authoritative on its own.

- `docs/` — this strategy's own specification documents (the equivalent
  of `report_current.html` for Filing Momentum ML): factor definitions,
  ranking/combination formulas, rebalance cadence, portfolio construction
  rules, and a `reproducibility_findings.md` once there's a real result
  to record provenance for.
- `notebooks/{research,experiments,validation,diagnostics}/` — exploratory
  work. If it's a formula, threshold, or rule the specification defines,
  it belongs in the production package (`src/`), tested — not only in a
  notebook.
- `fixtures/` — small checked-in data snapshots for tests, never
  production caches.
- `reports/` — generated exports (backtest reports, performance
  summaries) once a real backtest runner exists.

## Status

No factors, formulas, universe, or rebalance cadence have been specified
yet. See
`src/atlas_quant/strategies/ranked_multi_factor_rotation/config.py` for
the current (placeholder-only) configuration shell.
