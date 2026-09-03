"""Forward-return calculation and outcome domain, report §4.2.

Report: "all tickers are ranked by their forward return from feature date
to the quarter's sell date, clipped to ±150%." This is the *label-input*
return — never confused with the later backtest portfolio-return cap
(``RETURN_CAP = ±50%``, report §5.5), which belongs to portfolio-return
aggregation and is entirely out of scope for this stage.

Forward returns are never stored back onto ``FeatureObservation`` — a
feature observation must remain a point-in-time input with no future
information; a forward-return outcome is a separate, later-knowable fact
about the same instrument/quarter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from atlas_quant.data.records import CANONICAL_PRICE_CONVENTION, DailyPriceObservation, PriceConvention
from atlas_quant.domain.identifiers import InstrumentId
from atlas_quant.domain.provenance import DataProvenance

#: Report §4.2: label-input forward returns are clipped to ±150% -- distinct
#: from RETURN_CAP (±50%, report §5.5), which is a portfolio-return
#: aggregation concept and does not apply here.
LABEL_RETURN_CLIP = 1.50


@dataclass(frozen=True, slots=True)
class ForwardReturnOutcome:
    """One instrument/quarter's forward-return outcome, report §4.2.

    ``raw_return``/``clipped_return`` are ``None`` (not NaN) when the
    outcome could not be determined at all (see ``missing_reason``) --
    ``None`` distinguishes "not computable" from "computed as exactly
    0.0" more explicitly than a float sentinel would.

    ``label_available_at`` is the timestamp at which this outcome becomes
    knowable -- always ``sell_timestamp`` itself, since the forward return
    cannot be computed before the exit price exists. A training row may
    only be used when this is ``<=`` the target scoring cutoff (see
    ``training_dataset.py``).
    """

    instrument_id: InstrumentId
    quarter_end: date
    feature_timestamp: date
    sell_timestamp: date
    entry_price: float | None
    exit_price: float | None
    raw_return: float | None
    clipped_return: float | None
    label_available_at: datetime
    price_convention: PriceConvention
    data_cutoff: datetime
    provenance: tuple[DataProvenance, ...]
    missing_reason: str | None = None
    label: int | None = None


def _price_on_or_before(
    prices: Sequence[DailyPriceObservation], cutoff: date, data_cutoff: datetime
) -> DailyPriceObservation | None:
    """The last price with ``trading_date <= cutoff``, never after ``data_cutoff``.

    Report/legacy convention (``ml_scorer.py``: ``ps[ps.index <=
    feature_dt].iloc[-1]`` / ``ps[ps.index <= sell_dt].iloc[-1]``): the
    *last* price on or before the target date, never the first price on
    or after it -- an "on or after" convention would look ahead past the
    target date itself.
    """
    eligible = [
        p
        for p in prices
        if p.trading_date <= cutoff and datetime.combine(p.trading_date, datetime.min.time()) <= data_cutoff
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p.trading_date)


# Design note: because sell_timestamp is always >= feature_timestamp, any
# price that successfully resolves the entry lookup also, trivially,
# satisfies the exit lookup's own "<= sell_timestamp and <= data_cutoff"
# condition (falling back to that same stale price if nothing more recent
# qualifies). "Entry resolves, exit is missing" is therefore not a
# reachable state under this convention -- matching the legacy
# prototype's identical ps[ps.index <= X].iloc[-1] behavior exactly. Both
# resolve together or fail together (missing entry is checked first).


def compute_forward_return(entry_price: float, exit_price: float) -> tuple[float, float]:
    """Report §4.2: raw and ±150%-clipped forward return.

    ``raw_return = (exit_price - entry_price) / entry_price``. Caller must
    ensure ``entry_price > 0`` -- this pure function does not itself guard
    against a zero/negative entry price (that is a "missing/invalid entry
    price" case handled by :func:`build_forward_return_outcome`, which
    never calls this function in that case).
    """
    raw = (exit_price - entry_price) / entry_price
    clipped = max(-LABEL_RETURN_CLIP, min(LABEL_RETURN_CLIP, raw))
    return raw, clipped


def build_forward_return_outcome(
    instrument_id: InstrumentId,
    quarter_end: date,
    feature_timestamp: date,
    sell_timestamp: date,
    prices: Sequence[DailyPriceObservation],
    data_cutoff: datetime,
) -> ForwardReturnOutcome:
    """Build one instrument's forward-return outcome, report §4.2/§5.5.

    Entry price: the last available close with ``trading_date <=
    feature_timestamp``. Exit price: the last available close with
    ``trading_date <= sell_timestamp``. Neither ever uses a price dated
    after ``data_cutoff`` -- a price observed only in hindsight cannot
    resolve an outcome. Missing entry/exit price, or a non-positive entry
    price (division would be undefined or economically meaningless),
    produce ``missing_reason`` and ``None`` returns rather than NaN or an
    exception.
    """
    entry_obs = _price_on_or_before(prices, feature_timestamp, data_cutoff)
    exit_obs = _price_on_or_before(prices, sell_timestamp, data_cutoff)
    label_available_at = datetime.combine(sell_timestamp, datetime.min.time())
    provenance = tuple(p.provenance for p in (entry_obs, exit_obs) if p is not None)

    if entry_obs is None:
        return ForwardReturnOutcome(
            instrument_id=instrument_id, quarter_end=quarter_end,
            feature_timestamp=feature_timestamp, sell_timestamp=sell_timestamp,
            entry_price=None, exit_price=exit_obs.close if exit_obs else None,
            raw_return=None, clipped_return=None, label_available_at=label_available_at,
            price_convention=CANONICAL_PRICE_CONVENTION, data_cutoff=data_cutoff,
            provenance=provenance, missing_reason="missing entry price",
        )
    if exit_obs is None:
        return ForwardReturnOutcome(
            instrument_id=instrument_id, quarter_end=quarter_end,
            feature_timestamp=feature_timestamp, sell_timestamp=sell_timestamp,
            entry_price=entry_obs.close, exit_price=None,
            raw_return=None, clipped_return=None, label_available_at=label_available_at,
            price_convention=CANONICAL_PRICE_CONVENTION, data_cutoff=data_cutoff,
            provenance=provenance, missing_reason="missing exit price",
        )
    if entry_obs.close <= 0:
        return ForwardReturnOutcome(
            instrument_id=instrument_id, quarter_end=quarter_end,
            feature_timestamp=feature_timestamp, sell_timestamp=sell_timestamp,
            entry_price=entry_obs.close, exit_price=exit_obs.close,
            raw_return=None, clipped_return=None, label_available_at=label_available_at,
            price_convention=CANONICAL_PRICE_CONVENTION, data_cutoff=data_cutoff,
            provenance=provenance, missing_reason="non-positive entry price",
        )

    raw, clipped = compute_forward_return(entry_obs.close, exit_obs.close)
    return ForwardReturnOutcome(
        instrument_id=instrument_id, quarter_end=quarter_end,
        feature_timestamp=feature_timestamp, sell_timestamp=sell_timestamp,
        entry_price=entry_obs.close, exit_price=exit_obs.close,
        raw_return=raw, clipped_return=clipped, label_available_at=label_available_at,
        price_convention=CANONICAL_PRICE_CONVENTION, data_cutoff=data_cutoff,
        provenance=provenance, missing_reason=None,
    )
