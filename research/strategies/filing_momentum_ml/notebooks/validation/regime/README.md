# Regime subsystem validation notebooks

Placeholder for Stage 4+ regime-validation notebooks (e.g. comparing the
canonical `RegimeEvaluator`'s classifications against SPY's historical
regime history, or visualizing Markov window persistence/HMM state
transitions).

No authoritative calculation belongs here. Any notebook added under this
directory must import the canonical implementation from
`atlas_quant.strategies.filing_momentum_ml.regime_evaluator` /
`regime_markov` / `regime_hmm` — it must never reimplement Markov or HMM
math inline. See `../../docs/regime_specification.md` for the canonical
formulas and configuration this subsystem implements.
