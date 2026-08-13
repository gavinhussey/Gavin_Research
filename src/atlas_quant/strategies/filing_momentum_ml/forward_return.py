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

from atlas_quant.backtest.price_resolution import (
    daily_close_available_at,
    normalize_price_request_timestamp,
)
from atlas_quant.backtest.corporate_actions import compute_economic_return
from atlas_quant.data.records import CANONICAL_PRICE_CONVENTION, CorporateActionRecord, DailyPriceObservation, PriceConvention
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
    knowable -- the later of the modeled sell timestamp and the resolved
    exit close's availability timestamp. A training row may only be used
    when this is strictly before the target scoring cutoff (see
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
    split_count: int = 0
    dividend_count: int = 0
    dividend_cash: float = 0.0


def _price_on_or_before(
    prices: Sequence[DailyPriceObservation], cutoff: date | datetime, data_cutoff: datetime
) -> DailyPriceObservation | None:
    """The last completed close with ``trading_date <= cutoff``, never after ``data_cutoff``.

    Report/legacy convention (``ml_scorer.py``: ``ps[ps.index <=
    feature_dt].iloc[-1]`` / ``ps[ps.index <= sell_dt].iloc[-1]``): the
    *last* price on or before the target date, never the first price on
    or after it -- an "on or after" convention would look ahead past the
    target date itself. Daily closes are eligible only after their 16:00
    availability timestamp, so a midnight target cannot consume that
    same day's close.
    """
    requested_at = normalize_price_request_timestamp(cutoff)
    requested_date = requested_at.date()
    eligible = [
        p
        for p in prices
        if p.trading_date <= requested_date
        and daily_close_available_at(p.trading_date) < requested_at
        and daily_close_available_at(p.trading_date) < data_cutoff
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
    feature_timestamp: date | datetime,
    sell_timestamp: date | datetime,
    prices: Sequence[DailyPriceObservation],
    data_cutoff: datetime,
    corporate_actions: Sequence[CorporateActionRecord] = (),
) -> ForwardReturnOutcome:
    """Build one instrument's forward-return outcome, report §4.2/§5.5.

    Entry price: the last completed daily close with ``trading_date <=
    feature_timestamp`` and 16:00 close availability strictly before the
    modeled feature timestamp. Exit price uses the same rule against
    ``sell_timestamp``. Neither ever uses a close whose availability is on
    or after ``data_cutoff`` -- a price observed only in hindsight cannot
    resolve an outcome. Missing entry/exit price, or a non-positive entry
    price (division would be undefined or economically meaningless),
    produce ``missing_reason`` and ``None`` returns rather than NaN or an
    exception.
    """
    entry_at = normalize_price_request_timestamp(feature_timestamp)
    sell_at = normalize_price_request_timestamp(sell_timestamp)
    entry_obs = _price_on_or_before(prices, entry_at, data_cutoff)
    exit_obs = _price_on_or_before(prices, sell_at, data_cutoff)
    exit_available_at = daily_close_available_at(exit_obs.trading_date) if exit_obs else sell_at
    label_available_at = max(sell_at, exit_available_at)
    provenance = tuple(p.provenance for p in (entry_obs, exit_obs) if p is not None)

    if entry_obs is None:
        return ForwardReturnOutcome(
            instrument_id=instrument_id, quarter_end=quarter_end,
            feature_timestamp=entry_at.date(), sell_timestamp=sell_at.date(),
            entry_price=None, exit_price=exit_obs.close if exit_obs else None,
            raw_return=None, clipped_return=None, label_available_at=label_available_at,
            price_convention=CANONICAL_PRICE_CONVENTION, data_cutoff=data_cutoff,
            provenance=provenance, missing_reason="missing entry price",
        )
    if exit_obs is None:
        return ForwardReturnOutcome(
            instrument_id=instrument_id, quarter_end=quarter_end,
            feature_timestamp=entry_at.date(), sell_timestamp=sell_at.date(),
            entry_price=entry_obs.close, exit_price=None,
            raw_return=None, clipped_return=None, label_available_at=label_available_at,
            price_convention=CANONICAL_PRICE_CONVENTION, data_cutoff=data_cutoff,
            provenance=provenance, missing_reason="missing exit price",
        )
    if entry_obs.close <= 0:
        return ForwardReturnOutcome(
            instrument_id=instrument_id, quarter_end=quarter_end,
            feature_timestamp=entry_at.date(), sell_timestamp=sell_at.date(),
            entry_price=entry_obs.close, exit_price=exit_obs.close,
            raw_return=None, clipped_return=None, label_available_at=label_available_at,
            price_convention=CANONICAL_PRICE_CONVENTION, data_cutoff=data_cutoff,
            provenance=provenance, missing_reason="non-positive entry price",
        )

    economic = compute_economic_return(
        entry_obs.close, exit_obs.close, entry_obs.trading_date, exit_obs.trading_date,
        corporate_actions, include_dividends=True,
    )
    raw = economic.raw_return
    clipped = max(-LABEL_RETURN_CLIP, min(LABEL_RETURN_CLIP, raw))
    return ForwardReturnOutcome(
        instrument_id=instrument_id, quarter_end=quarter_end,
        feature_timestamp=entry_at.date(), sell_timestamp=sell_at.date(),
        entry_price=entry_obs.close, exit_price=exit_obs.close,
        raw_return=raw, clipped_return=clipped, label_available_at=label_available_at,
        price_convention=entry_obs.price_convention, data_cutoff=data_cutoff,
        provenance=provenance, missing_reason=None,
        split_count=economic.split_count, dividend_count=economic.dividend_count,
        dividend_cash=economic.dividend_cash,
    )
