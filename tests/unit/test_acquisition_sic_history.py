"""Unit tests for the point-in-time SIC-history backfill acquisition path.

Every test uses FakeHttpClient with hand-crafted SGML header text -- never
a real network call.
"""

from datetime import date, datetime

from atlas_quant.strategies.filing_momentum_ml.acquisition.sic_history import fetch_sic_history
from atlas_quant.strategies.filing_momentum_ml.production.normalization import RawFilingRecord

from tests.fixtures.acquisition import FakeHttpClient

_RETRIEVED_AT = datetime(2026, 7, 29)


def _filing(symbol="AAPL", accession="0001-09-000001", fy=2009) -> RawFilingRecord:
    return RawFilingRecord(
        symbol=symbol, asset_class="equity", fiscal_period="Q1", fiscal_year=fy,
        quarter_end=date(2009, 3, 31), filed_at=datetime(2009, 4, 15),
        revenue=1.0, gross_profit=None, operating_income=None, net_income=None, diluted_eps=None,
        stockholders_equity=None, operating_cash_flow=None, capital_expenditure=None,
        accession_number=accession, source="sec_edgar", retrieved_at=_RETRIEVED_AT,
    )


def _header(sic_code: int) -> str:
    return (
        "<SEC-HEADER>\n"
        "\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\tSOME DESCRIPTION "
        f"[{sic_code}]\n"
        "</SEC-HEADER>\n"
    )


def _url(cik_nozero: int, accession: str) -> str:
    accession_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_nodash}/{accession}.txt"


def test_dedupes_multiple_fact_rows_sharing_one_accession():
    filings = [_filing(), _filing(fy=2009), _filing(fy=2009)]  # same symbol+accession, 3 fact rows
    client = FakeHttpClient(text_responses={_url(320193, "0001-09-000001"): _header(7372)})
    result = fetch_sic_history(
        client, filings, {"AAPL": "0000320193"}, user_agent="Test test@example.com",
        retrieved_at=_RETRIEVED_AT, requests_per_second=0,
    )
    assert result.pairs_attempted == 1
    assert len(result.records) == 1
    assert result.records[0].sic_code == 7372
    assert result.records[0].gics_sector == "Information Technology"


def test_missing_cik_recorded_as_warning_not_raised():
    filings = [_filing(symbol="NOPE")]
    result = fetch_sic_history(
        FakeHttpClient(), filings, {}, user_agent="Test test@example.com",
        retrieved_at=_RETRIEVED_AT, requests_per_second=0,
    )
    assert result.pairs_attempted == 1
    assert result.pairs_with_sic == 0
    assert result.records == ()
    assert any("no CIK match" in w for w in result.warnings)


def test_fetch_failure_recorded_as_warning_never_aborts_run():
    filings = [_filing(symbol="AAPL", accession="a1"), _filing(symbol="MSFT", accession="a2")]
    # Only AAPL's URL is canned; MSFT's fetch will KeyError inside FakeHttpClient.
    client = FakeHttpClient(text_responses={_url(320193, "a1"): _header(7372)})
    result = fetch_sic_history(
        client, filings, {"AAPL": "0000320193", "MSFT": "0000789019"}, user_agent="Test test@example.com",
        retrieved_at=_RETRIEVED_AT, requests_per_second=0,
    )
    assert result.pairs_attempted == 2
    assert result.pairs_with_sic == 1
    assert any("SIC fetch failed" in w for w in result.warnings)


def test_multiple_symbols_each_get_own_sic():
    filings = [_filing(symbol="AAPL", accession="a1"), _filing(symbol="MSFT", accession="a2")]
    client = FakeHttpClient(text_responses={
        _url(320193, "a1"): _header(7372),
        _url(789019, "a2"): _header(7371),
    })
    result = fetch_sic_history(
        client, filings, {"AAPL": "0000320193", "MSFT": "0000789019"}, user_agent="Test test@example.com",
        retrieved_at=_RETRIEVED_AT, requests_per_second=0,
    )
    by_symbol = {r.symbol: r.sic_code for r in result.records}
    assert by_symbol == {"AAPL": 7372, "MSFT": 7371}
