"""Ranked Multi-Factor Rotation — pure formulas, report/spec §TBD.

Empty by design. Every calculation this strategy's own specification
defines (per-factor cross-sectional transform, composite score
combination, portfolio construction weights, anything else the user
supplies as an equation) belongs here as a pure function — no I/O, no
state, and a docstring citing the section of
``research/strategies/ranked_multi_factor_rotation/docs/`` it came from,
exactly like ``filing_momentum_ml/formulas.py`` cites report sections.
Every other module in this package should call functions defined here
rather than re-deriving a formula inline.

No functions exist yet because no formula has been specified yet.
"""

from __future__ import annotations
