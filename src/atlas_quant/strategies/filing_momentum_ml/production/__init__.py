"""Filing Momentum ML's production research workflow: data provenance,
legacy-cache audit, raw-data validation, normalization, and orchestration.

Everything here consumes or produces existing Stage 3-9 typed records and
services — it never reimplements feature/label/model/strategy/
backtest/performance/report logic. It only decides *whether* those
services may run (dependency and data-provenance gating) and *what data*
they run against (acquisition, normalization).
"""
