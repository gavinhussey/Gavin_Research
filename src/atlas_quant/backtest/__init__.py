"""The standalone Filing Momentum ML historical backtest subsystem, Stage 7.

Orchestrates Stage 3 (features), Stage 5 (strategy
decision), and Stage 6 (labeling/training/scoring) for one strategy in
isolation — this package must never reimplement any of their logic, only
call it in the report-defined sequence with a deterministic clock and
explicit price-resolution/accounting policies.
"""
