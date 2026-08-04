# Ranked Multi-Factor Rotation — strategy specification

This is the source-of-truth document for this strategy's own design, the
role `report_current.html` plays for Filing Momentum ML (see
`CLAUDE.md`'s reproducibility-classification policy note). Nothing here
yet — sections below are placeholders to be filled in as the strategy is
specified, one piece at a time.

Once a section below is filled in with a real, decided formula or rule,
the corresponding implementation in
`src/atlas_quant/strategies/ranked_multi_factor_rotation/` should cite
this document's section the way `filing_momentum_ml/config.py` and
`formulas.py` cite `report_current.html` section numbers.

## 1. Universe

*Not yet specified.*

## 2. Factors

*Not yet specified.* For each factor: exact definition/formula, data
inputs required, lookback window, and any cross-sectional transform
(z-score, percentile rank, etc.) applied before combination.

## 3. Composite score

*Not yet specified.* How individual factor scores combine into one
ranking score (equal-weighted, factor-weighted, sequential screens, etc.).

## 4. Rebalance cadence

*Not yet specified.*

## 5. Portfolio construction

*Not yet specified.* How the ranked list becomes position weights
(top-N equal weight, score-weighted, long/short, position limits,
sector/exposure constraints, etc.).

## 6. Risk / sizing rules

*Not yet specified.*
