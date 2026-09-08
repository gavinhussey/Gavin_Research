"""Cross-sectional percentile-rank normalization of selected raw-level features.

Why this exists
---------------
Each model is fit on a rolling ``ml_train_years`` = 6 window, i.e. ~24
quarterly cross-sections. A feature expressed as a **raw level** (dollars,
share counts) does not mean the same thing at the start and the end of
that window: a $10B market cap was large-cap in 1990 and mid-cap in 2026.
Tree models' invariance to monotone transforms does **not** protect
against this — it is a change in the feature's *meaning across time*, not
a change of scale within a single cross-section.

Replacing each such value with its percentile rank *within its own
quarter's cross-section* makes it a stable relative statement ("this name
is in the 87th percentile of market cap among the names ranked this
quarter") that carries the same meaning in every quarter of the window.
Percentile rank rather than z-score because financial cross-sections are
heavily fat-tailed — a single extreme value distorts the mean/SD of every
other name — and because rank matches what the training label already is
(a decile-rank relevance grade).

Which features, and why not the rest
------------------------------------
The set is a **per-feature policy**, carried as
``MultiFactorRankingMLConfig.cross_sectional_rank_features``, not a global
transform. Deliberately excluded, by construction:

- **Macro** (``fed_funds_rate``, ``hy_credit_oas``, ``ust_10y_yield``,
  ``ust_2y_yield``, ``vix``, ``yield_curve_10y_2y``) is broadcast
  identically to every instrument in a cycle, so ranking it
  cross-sectionally would map all six to the same constant every quarter,
  annihilating the block entirely.
- **Categorical** (``sector_enc``, ``quarter_num``) are labels; arithmetic
  on them is meaningless.
- **Volatility** features are already scale-free, and ranking them within
  a quarter discards their absolute level, which carries information — a
  measured negative result, see ``docs/reproducibility_findings.md``.
- **Growth/ratio** features are already differenced or already scale-free,
  and ranking growth destroys sign information (a recession's least-bad
  shrinker would look like a boom-time leader).

Lookahead safety
----------------
:func:`apply_cross_sectional_rank_normalization` reads **only** the
observations handed to it, which are exactly one evaluation cycle's own
cross-section (``feature_pipeline.run_feature_pipeline`` builds them for a
single ``quarter_start``/``cutoff`` pair). No other quarter's data — past
or future — participates in any rank. Adding, removing, or altering data
in any other cycle therefore cannot change this cycle's normalized values.
This is what keeps the transform point-in-time safe by construction, and
it must stay true: never widen this function's inputs to a multi-cycle
panel.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from atlas_quant.domain.audit import AuditRecord
from atlas_quant.strategies.multi_factor_ranking_ml.feature_domain import FeatureObservation

_AUDIT_STAGE = "cross_sectional_rank_normalization"


def _is_missing(value: object) -> bool:
    return not isinstance(value, (int, float)) or (
        isinstance(value, float) and math.isnan(value)
    )


def percentile_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Percentile-rank ``values`` within themselves, scaled to ``[0.0, 1.0]``.

    - Missing entries (``NaN``, or a non-numeric) are returned as ``NaN``:
      missingness is **never** imputed here. Only non-missing values
      participate in the ranking, and the scaling denominator counts only
      them. LightGBM handles NaN natively; the pipeline deliberately
      preserves it.
    - Ties share the **average** of the ranks they span.
    - Scaling is ``(rank - 1) / (n - 1)`` over the ``n`` non-missing
      values, so the minimum is exactly ``0.0`` and the maximum exactly
      ``1.0``.
    - With ``n < 2`` the percentile rank is **undefined** (the denominator
      is zero and a lone value has no cross-section to be relative to), so
      the input is returned unchanged. This is a real live-scoring edge
      case: a single-name cross-section keeps its raw level.
    """
    indexed = [(i, float(v)) for i, v in enumerate(values) if not _is_missing(v)]
    if len(indexed) < 2:
        return tuple(
            float("nan") if _is_missing(v) else float(v) for v in values
        )

    indexed.sort(key=lambda pair: pair[1])
    n = len(indexed)
    out = [float("nan")] * len(values)

    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and indexed[stop + 1][1] == indexed[start][1]:
            stop += 1
        # 1-based ranks of this tie group are start+1 .. stop+1; the shared
        # rank is their average.
        average_rank = (start + 1 + stop + 1) / 2.0
        scaled = (average_rank - 1.0) / (n - 1.0)
        for position in range(start, stop + 1):
            out[indexed[position][0]] = scaled
        start = stop + 1

    return tuple(out)


def apply_cross_sectional_rank_normalization(
    observations: Sequence[FeatureObservation],
    feature_names: Sequence[str],
) -> tuple[FeatureObservation, ...]:
    """Return ``observations`` with each name in ``feature_names`` replaced by
    its percentile rank within **this cross-section only**.

    ``observations`` must be exactly one evaluation cycle's cross-section
    (see the module docstring's lookahead note). ``feature_names`` is the
    configured per-feature policy — anything not listed is passed through
    byte-identically, including the macro block, which would be destroyed
    by a global transform.

    An empty ``feature_names`` is legal and means "no normalization": the
    observations are returned unchanged.
    """
    names = tuple(feature_names)
    if not names or not observations:
        return tuple(observations)

    normalized: dict[str, tuple[float, ...]] = {}
    for name in names:
        column = [obs.features.get(name, float("nan")) for obs in observations]
        ranks = percentile_ranks(column)
        # A cross-section with <2 non-missing values is left untouched;
        # percentile_ranks already returns the column unchanged there.
        normalized[name] = ranks

    non_missing_counts = {
        name: sum(1 for obs in observations if not _is_missing(obs.features.get(name, float("nan"))))
        for name in names
    }

    out: list[FeatureObservation] = []
    for row, obs in enumerate(observations):
        features = dict(obs.features)
        for name in names:
            if name in features:
                features[name] = normalized[name][row]
        audit_trail = obs.audit_trail.append(
            AuditRecord(
                stage=_AUDIT_STAGE,
                message=(
                    f"percentile-rank normalized {len(names)} feature(s) within this "
                    f"cycle's {len(observations)}-name cross-section"
                ),
                timestamp=obs.data_cutoff,
                data={
                    "normalized_features": list(names),
                    "cross_section_size": len(observations),
                    "non_missing_counts": non_missing_counts,
                },
            )
        )
        out.append(replace(obs, features=features, audit_trail=audit_trail))

    return tuple(out)
