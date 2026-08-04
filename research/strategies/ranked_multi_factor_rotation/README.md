# Ranked Multi-Factor Rotation — research workspace

Mirrors `research/strategies/filing_momentum_ml/`'s layout per
`docs/adding_a_strategy.md`. No specification, formula, or result in
this directory is authoritative until it also exists as tested code
under `src/atlas_quant/strategies/ranked_multi_factor_rotation/` — this
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

`docs/specification.md` §§1–6 (universe, momentum/volatility/
correlation/trend factors, ranking, total rank, selection/allocation,
rebalance cadence) are specified and implemented as tested code in
`src/atlas_quant/strategies/ranked_multi_factor_rotation/` (config,
formulas, point-in-time monthly pipeline, strategy evaluator). Not yet
built: a backtest runner, real data acquisition, or a genuine historical
backtest — nothing in this directory should be read as a strategy
performance claim until one exists. §7 (risk/sizing beyond flat
per-slot weighting) and §8 (transaction cost/slippage assumptions)
remain open questions, not decided rules.
