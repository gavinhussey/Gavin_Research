"""SEC EDGAR XBRL filing-fundamentals acquisition.

Uses SEC's free, public ``data.sec.gov`` XBRL "company facts" API -- no
API key. SEC's own fair-access policy requires every request identify
its requester via a ``User-Agent`` header (a real name and contact,
never a fake identity) -- this module never hardcodes one; it must be
supplied by the caller (see ``SEC_EDGAR_USER_AGENT`` below).

Known, disclosed limitation: SEC XBRL data reports quarterly (~90-day
duration) facts from 10-Qs for Q1-Q3, but a fiscal Q4 is not filed as its
own 10-Q -- only as part of the annual 10-K's full-year figures. For the
additive (flow) concepts in :data:`_ADDITIVE_CONCEPTS`, this module
derives Q4 = FY - (Q1+Q2+Q3) when all four are disclosed for the same
fiscal year -- real arithmetic on real disclosed numbers, added as a
separately-sourced row (``source`` suffixed ``_derived_q4``,
``accession_number`` suffixed ``#derived_q4``) rather than merged into
any one filing's own row, so it is never mistaken for a directly-reported
fact. ``diluted_eps`` (not additive -- a per-share ratio, not a flow) and
``stockholders_equity`` (a balance-sheet snapshot, not a flow -- its real
Q4 value would need to be read directly off the FY 10-K's own
balance-sheet date, not derived by subtraction; not yet implemented) are
excluded from this derivation and remain absent for Q4 when no filing
itself reports them, never fabricated. A Q4 quarter with insufficient
underlying facts to derive from (any of the four missing) is left
absent, which downstream validation (``production.validation``) will
surface as missing history, never silently interpolated.

Second known, disclosed limitation, confirmed against real Apple data:
one 10-Q's accession number commonly reports both the current quarter
and the prior-year comparative quarter for the same concept -- this
module keeps both as separate ``(accession_number, quarter_end)`` rows
(point-in-time correct: a comparative figure genuinely becomes known
when the filing containing it is published), but this can leave one
quarter's row with a partially or fully missing set of fields for that
specific accession when a concept's value only appears attached to a
different accession/period pairing this module does not reconcile
against (reported as ``None``, never fabricated; surfaced by downstream
validation/missing-feature handling, never silently interpolated).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from atlas_quant.strategies.filing_momentum_ml.acquisition.http_client import HttpClient
from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawFilingRecord

TICKER_TO_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

#: One real filing's own full submission text -- its SGML header (the
#: first few KB) carries that filing's own point-in-time SIC code, unlike
#: the company-facts/submissions endpoints above, which only ever expose
#: the *current* SIC. Verified against real filings in development: the
#: same company's SIC differs between an old and a recent filing.
FILING_SUBMISSION_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_nodash}/{accession}.txt"

#: Generous enough to cover the full SGML header even for filings with
#: several co-filers (each with their own SIC/company-data block) --
#: verified sufficient against real filings; the header is a fixed-size
#: prefix of the document, well before the first <DOCUMENT> body starts.
_HEADER_RANGE_BYTES = "bytes=0-8000"

_SIC_HEADER_PATTERN = re.compile(r"STANDARD INDUSTRIAL CLASSIFICATION:\s*.*\[(\d+)\]")

#: Read once per real acquisition run -- SEC requires a real requester
#: identity ("Company Name contact@example.com"), never a fake one.
SEC_EDGAR_USER_AGENT_ENV_VAR = "SEC_EDGAR_USER_AGENT"

_QUARTERLY_MIN_DAYS = 80
_QUARTERLY_MAX_DAYS = 100
_FULL_YEAR_MIN_DAYS = 350
_FULL_YEAR_MAX_DAYS = 380
_QUARTERLY_FISCAL_PERIODS = ("Q1", "Q2", "Q3", "Q4")
_ACCEPTED_FORMS = ("10-Q", "10-Q/A", "10-K", "10-K/A")
_ANNUAL_FORMS = ("10-K", "10-K/A")

#: SEC XBRL never reports fiscal Q4 as its own quarterly (~90-day) fact --
#: only Q1-Q3 get a standalone 10-Q. These concepts are additive across a
#: fiscal year, so Q4 can be derived exactly as FY - (Q1+Q2+Q3) when all
#: four are disclosed -- real arithmetic on real disclosed numbers, never
#: an estimate. ``diluted_eps`` is deliberately excluded: EPS is a
#: per-share ratio with a weighted-average share count that changes
#: quarter to quarter, so it is *not* additive and this identity would
#: silently produce a wrong number, not an approximate one.
#: ``stockholders_equity`` is handled separately (see
#: ``_derive_q4_instant_facts``) -- it is a balance-sheet snapshot, not a
#: flow, so its real Q4 value is simply the FY 10-K's own balance-sheet
#: date fact, with no subtraction needed or valid.
_ADDITIVE_CONCEPTS = frozenset({
    "revenue", "gross_profit", "operating_income", "net_income",
    "operating_cash_flow", "capital_expenditure",
})

#: Marks a Q4 row as computed (FY minus the three disclosed quarters)
#: rather than read directly off one filing's own tag -- propagated
#: through ``RawFilingRecord.source`` -> ``DataProvenance.source`` so a
#: derived value is never presented with the same provenance confidence
#: as a directly-reported one.
DERIVED_Q4_SOURCE_SUFFIX = "_derived_q4"

#: One or more candidate us-gaap tags per concept, tried in priority
#: order -- different companies/years report the same concept under
#: different tags; the first match for a given (accession, quarter_end)
#: is kept.
_TAG_CANDIDATES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "diluted_eps": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "stockholders_equity": (
        "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireOtherProductiveAssets",
        "PaymentsToAcquireMachineryAndEquipment",
    ),
}

#: Balance-sheet ("instant") concepts have only an `end` date, no `start`.
_INSTANT_CONCEPTS = frozenset({"stockholders_equity"})


def resolve_user_agent(explicit: str | None = None) -> str:
    """Resolve the SEC EDGAR User-Agent string.

    Never hardcodes a default -- raises :class:`ValueError` if neither
    ``explicit`` nor the ``SEC_EDGAR_USER_AGENT`` environment variable is
    set, so a real acquisition run can never silently identify itself
    with a made-up requester.
    """
    value = explicit or os.environ.get(SEC_EDGAR_USER_AGENT_ENV_VAR)
    if not value:
        raise ValueError(
            f"SEC EDGAR requires a real User-Agent identifying the requester "
            f"(e.g. 'Your Name your@email.example') -- set the "
            f"{SEC_EDGAR_USER_AGENT_ENV_VAR} environment variable or pass "
            "user_agent= explicitly; SEC's own fair-access policy is not optional."
        )
    return value


def fetch_ticker_to_cik_map(client: HttpClient, *, user_agent: str) -> dict[str, str]:
    """Fetch SEC's own ticker -> 10-digit zero-padded CIK mapping."""
    data = client.get_json(TICKER_TO_CIK_URL, headers={"User-Agent": user_agent})
    result: dict[str, str] = {}
    for entry in data.values():
        ticker = entry.get("ticker")
        cik = entry.get("cik_str")
        if ticker is None or cik is None:
            continue
        result[ticker.upper()] = f"{int(cik):010d}"
    return result


def fetch_company_facts(client: HttpClient, cik10: str, *, user_agent: str) -> dict:
    """Fetch one company's full XBRL company-facts payload."""
    url = COMPANY_FACTS_URL_TEMPLATE.format(cik10=cik10)
    return client.get_json(url, headers={"User-Agent": user_agent})


def fetch_filing_sic(client: HttpClient, cik10: str, accession_number: str, *, user_agent: str) -> int | None:
    """Fetch one specific filing's own point-in-time SIC code from its
    real SEC submission header -- the only place a filing's SIC as of its
    own filing date is available (the company-facts/submissions endpoints
    only expose today's SIC). Returns ``None`` if the header's SIC line
    isn't found (never fabricated)."""
    accession_nodash = accession_number.replace("-", "")
    url = FILING_SUBMISSION_URL_TEMPLATE.format(
        cik_nozero=int(cik10), accession_nodash=accession_nodash, accession=accession_number,
    )
    header_text = client.get_text(url, headers={"User-Agent": user_agent, "Range": _HEADER_RANGE_BYTES})
    match = _SIC_HEADER_PATTERN.search(header_text)
    if match is None:
        return None
    return int(match.group(1))


@dataclass(frozen=True, slots=True)
class _FactPoint:
    accn: str
    quarter_end: date
    fiscal_period: str
    fiscal_year: int
    filed_at: date
    value: float


def _extract_concept_facts(facts_json: dict, concept: str) -> list[_FactPoint]:
    us_gaap = (facts_json.get("facts") or {}).get("us-gaap") or {}
    points: list[_FactPoint] = []

    for tag in _TAG_CANDIDATES[concept]:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            continue
        for unit_entries in (tag_data.get("units") or {}).values():
            for entry in unit_entries:
                if entry.get("form") not in _ACCEPTED_FORMS:
                    continue
                fp = entry.get("fp")
                if fp not in _QUARTERLY_FISCAL_PERIODS:
                    continue
                end = entry.get("end")
                if not end:
                    continue

                if concept not in _INSTANT_CONCEPTS:
                    start = entry.get("start")
                    if not start:
                        continue
                    duration_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                    if not (_QUARTERLY_MIN_DAYS <= duration_days <= _QUARTERLY_MAX_DAYS):
                        continue

                filed = entry.get("filed")
                accn = entry.get("accn")
                fy = entry.get("fy")
                val = entry.get("val")
                if filed is None or accn is None or val is None or fy is None:
                    continue

                points.append(
                    _FactPoint(
                        accn=accn, quarter_end=date.fromisoformat(end), fiscal_period=fp, fiscal_year=int(fy),
                        filed_at=date.fromisoformat(filed), value=float(val),
                    )
                )
    return points


def _extract_full_year_facts(facts_json: dict, concept: str) -> list[_FactPoint]:
    """The FY-tagged (10-K, ~365-day-duration) facts for ``concept`` --
    never returned by :func:`_extract_concept_facts`, which only keeps
    quarterly-duration facts. Used solely to derive Q4 (see
    :func:`_derive_q4_additive_facts`), never surfaced as a filing row on
    its own."""
    us_gaap = (facts_json.get("facts") or {}).get("us-gaap") or {}
    points: list[_FactPoint] = []

    for tag in _TAG_CANDIDATES[concept]:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            continue
        for unit_entries in (tag_data.get("units") or {}).values():
            for entry in unit_entries:
                if entry.get("form") not in _ANNUAL_FORMS or entry.get("fp") != "FY":
                    continue
                end = entry.get("end")
                start = entry.get("start")
                if not end or not start:
                    continue
                duration_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
                if not (_FULL_YEAR_MIN_DAYS <= duration_days <= _FULL_YEAR_MAX_DAYS):
                    continue

                filed = entry.get("filed")
                accn = entry.get("accn")
                fy = entry.get("fy")
                val = entry.get("val")
                if filed is None or accn is None or val is None or fy is None:
                    continue

                points.append(
                    _FactPoint(
                        accn=accn, quarter_end=date.fromisoformat(end), fiscal_period="FY", fiscal_year=int(fy),
                        filed_at=date.fromisoformat(filed), value=float(val),
                    )
                )
    return points


def _canonical_by_fiscal_year_and_period(points: Sequence[_FactPoint]) -> dict[tuple[int, str], _FactPoint]:
    """The earliest-filed point per (fiscal_year, fiscal_period) -- the
    original filing's own value for that period, not a later filing's
    comparative-period fragment (see ``select_point_in_time_fundamentals``
    for the analogous, point-in-time-safe choice made at selection time;
    this is a separate, acquisition-time choice about which raw value
    feeds the Q4 subtraction below)."""
    best: dict[tuple[int, str], _FactPoint] = {}
    for point in points:
        key = (point.fiscal_year, point.fiscal_period)
        existing = best.get(key)
        if existing is None or point.filed_at < existing.filed_at:
            best[key] = point
    return best


def _derive_q4_additive_facts(facts_json: dict, concept: str) -> list[_FactPoint]:
    """Q4 = FY - (Q1+Q2+Q3) for one additive ``concept``, one fiscal year
    at a time -- only when all four are disclosed; never a partial
    estimate. See :data:`_ADDITIVE_CONCEPTS` for why this is restricted
    to flow concepts."""
    quarterly = _canonical_by_fiscal_year_and_period(_extract_concept_facts(facts_json, concept))
    full_year = _canonical_by_fiscal_year_and_period(_extract_full_year_facts(facts_json, concept))

    derived: list[_FactPoint] = []
    for (fiscal_year, fp), fy_point in full_year.items():
        if fp != "FY":
            continue
        # A company that already files its own Q4 10-Q (rare, but real)
        # already has a directly-reported Q4 fact from
        # _extract_concept_facts -- never overwritten by a derived one.
        if (fiscal_year, "Q4") in quarterly:
            continue
        q1 = quarterly.get((fiscal_year, "Q1"))
        q2 = quarterly.get((fiscal_year, "Q2"))
        q3 = quarterly.get((fiscal_year, "Q3"))
        if q1 is None or q2 is None or q3 is None:
            continue
        derived.append(
            _FactPoint(
                accn=fy_point.accn, quarter_end=fy_point.quarter_end, fiscal_period="Q4",
                fiscal_year=fiscal_year, filed_at=fy_point.filed_at,
                value=fy_point.value - q1.value - q2.value - q3.value,
            )
        )
    return derived


def parse_company_facts_to_filings(
    symbol: str, facts_json: dict, *, source: str, retrieved_at: datetime,
) -> list[RawFilingRecord]:
    """Turn one company's XBRL company-facts payload into raw filing records.

    Groups facts by ``(accession_number, quarter_end)`` -- one accession
    number is one real filing, which reports many concepts (revenue, net
    income, etc.) together; this never merges values from two different
    filings into one record. A concept absent from a specific filing
    (not every 10-Q reports every tag) is left ``None``, never defaulted.

    Fiscal Q4 is never filed as its own 10-Q (see module docstring), so
    for :data:`_ADDITIVE_CONCEPTS` it is derived as FY - (Q1+Q2+Q3) and
    added as a separate, distinctly-sourced synthetic row (accession
    number suffixed ``#derived_q4``, ``source`` suffixed
    ``_derived_q4``) rather than merged into a real accession's row --
    it was never disclosed by any single filing, so it must never look
    like one was.
    """
    rows: dict[tuple[str, date], dict] = {}
    for concept in _TAG_CANDIDATES:
        for point in _extract_concept_facts(facts_json, concept):
            key = (point.accn, point.quarter_end)
            row = rows.setdefault(
                key,
                {"fiscal_period": point.fiscal_period, "fiscal_year": point.fiscal_year, "filed_at": point.filed_at},
            )
            row.setdefault(concept, point.value)

    derived_rows: dict[tuple[int, date], dict] = {}
    for concept in _ADDITIVE_CONCEPTS:
        for point in _derive_q4_additive_facts(facts_json, concept):
            key = (point.fiscal_year, point.quarter_end)
            row = derived_rows.setdefault(
                key,
                {
                    "fiscal_period": point.fiscal_period, "fiscal_year": point.fiscal_year,
                    "filed_at": point.filed_at, "accn": point.accn,
                },
            )
            row.setdefault(concept, point.value)

    records = []
    for (accn, quarter_end), row in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        records.append(
            RawFilingRecord(
                symbol=symbol, asset_class="equity",
                fiscal_period=row["fiscal_period"], fiscal_year=row["fiscal_year"],
                quarter_end=quarter_end, filed_at=datetime.combine(row["filed_at"], datetime.min.time()),
                revenue=row.get("revenue"), gross_profit=row.get("gross_profit"),
                operating_income=row.get("operating_income"), net_income=row.get("net_income"),
                diluted_eps=row.get("diluted_eps"), stockholders_equity=row.get("stockholders_equity"),
                operating_cash_flow=row.get("operating_cash_flow"), capital_expenditure=row.get("capital_expenditure"),
                accession_number=accn, source=source, retrieved_at=retrieved_at,
            )
        )
    for (_, quarter_end), row in sorted(derived_rows.items(), key=lambda kv: kv[0][1]):
        records.append(
            RawFilingRecord(
                symbol=symbol, asset_class="equity",
                fiscal_period=row["fiscal_period"], fiscal_year=row["fiscal_year"],
                quarter_end=quarter_end, filed_at=datetime.combine(row["filed_at"], datetime.min.time()),
                revenue=row.get("revenue"), gross_profit=row.get("gross_profit"),
                operating_income=row.get("operating_income"), net_income=row.get("net_income"),
                diluted_eps=None, stockholders_equity=None,
                operating_cash_flow=row.get("operating_cash_flow"), capital_expenditure=row.get("capital_expenditure"),
                accession_number=f"{row['accn']}#derived_q4", source=f"{source}{DERIVED_Q4_SOURCE_SUFFIX}",
                retrieved_at=retrieved_at,
            )
        )
    return records


def fetch_filings_for_symbol(
    client: HttpClient, symbol: str, ticker_to_cik: dict[str, str], *, user_agent: str, retrieved_at: datetime,
) -> list[RawFilingRecord]:
    """Fetch and parse one symbol's real SEC EDGAR filing history end-to-end."""
    cik10 = ticker_to_cik.get(symbol.upper())
    if cik10 is None:
        return []
    facts = fetch_company_facts(client, cik10, user_agent=user_agent)
    return parse_company_facts_to_filings(symbol, facts, source="sec_edgar", retrieved_at=retrieved_at)
