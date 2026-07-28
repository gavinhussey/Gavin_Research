"""Unit tests for the SEC EDGAR filing-fundamentals acquisition adapter.

Every test uses FakeHttpClient with hand-crafted, small XBRL-shaped
payloads -- never a real network call.
"""

from datetime import date, datetime

import pytest

from atlas_quant.strategies.filing_momentum_ml.acquisition.sec_edgar import (
    SEC_EDGAR_USER_AGENT_ENV_VAR,
    fetch_company_facts,
    fetch_filings_for_symbol,
    fetch_ticker_to_cik_map,
    parse_company_facts_to_filings,
    resolve_user_agent,
)

from tests.fixtures.acquisition import FakeHttpClient

_RETRIEVED_AT = datetime(2024, 6, 1)


def test_resolve_user_agent_raises_without_explicit_or_env(monkeypatch):
    monkeypatch.delenv(SEC_EDGAR_USER_AGENT_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="User-Agent"):
        resolve_user_agent()


def test_resolve_user_agent_uses_explicit_value():
    assert resolve_user_agent("Test Researcher test@example.com") == "Test Researcher test@example.com"


def test_resolve_user_agent_uses_env_var(monkeypatch):
    monkeypatch.setenv(SEC_EDGAR_USER_AGENT_ENV_VAR, "Env Researcher env@example.com")
    assert resolve_user_agent() == "Env Researcher env@example.com"


def test_fetch_ticker_to_cik_map_parses_and_pads():
    client = FakeHttpClient(json_responses={
        "https://www.sec.gov/files/company_tickers.json": {
            "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
            "1": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
        }
    })
    result = fetch_ticker_to_cik_map(client, user_agent="Test test@example.com")
    assert result["AAPL"] == "0000320193"
    assert result["GOOGL"] == "0001652044"
    assert client.requested_headers[0]["User-Agent"] == "Test test@example.com"


def _facts_json() -> dict:
    return {
        "cik": 320193,
        "entityName": "Test Co",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            # Genuine quarterly 10-Q fact.
                            {"start": "2023-01-01", "end": "2023-03-31", "val": 1000.0, "accn": "acc-1",
                             "fy": 2023, "fp": "Q1", "form": "10-Q", "filed": "2023-04-28"},
                            # Annual 10-K figure sharing the same tag -- must be excluded (fp="FY").
                            {"start": "2022-01-01", "end": "2022-12-31", "val": 4000.0, "accn": "acc-fy",
                             "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-01"},
                            # A half-year (non-quarterly duration) figure -- must be excluded.
                            {"start": "2023-01-01", "end": "2023-06-30", "val": 2100.0, "accn": "acc-h1",
                             "fy": 2023, "fp": "Q2", "form": "10-Q", "filed": "2023-07-28"},
                        ]
                    }
                },
                "GrossProfit": {
                    "units": {
                        "USD": [
                            {"start": "2023-01-01", "end": "2023-03-31", "val": 400.0, "accn": "acc-1",
                             "fy": 2023, "fp": "Q1", "form": "10-Q", "filed": "2023-04-28"},
                        ]
                    }
                },
                "StockholdersEquity": {
                    "units": {
                        "USD": [
                            {"end": "2023-03-31", "val": 5000.0, "accn": "acc-1",
                             "fy": 2023, "fp": "Q1", "form": "10-Q", "filed": "2023-04-28"},
                        ]
                    }
                },
            }
        },
    }


def test_parse_company_facts_extracts_one_quarterly_filing():
    filings = parse_company_facts_to_filings("TEST", _facts_json(), source="sec_edgar", retrieved_at=_RETRIEVED_AT)
    assert len(filings) == 1
    row = filings[0]
    assert row.quarter_end == date(2023, 3, 31)
    assert row.fiscal_period == "Q1"
    assert row.fiscal_year == 2023
    assert row.filed_at == datetime(2023, 4, 28)
    assert row.revenue == 1000.0
    assert row.gross_profit == 400.0
    assert row.stockholders_equity == 5000.0
    assert row.operating_income is None  # not present in this accession -- never defaulted
    assert row.accession_number == "acc-1"


def test_annual_and_non_quarterly_durations_are_excluded():
    filings = parse_company_facts_to_filings("TEST", _facts_json(), source="sec_edgar", retrieved_at=_RETRIEVED_AT)
    quarter_ends = {f.quarter_end for f in filings}
    assert date(2022, 12, 31) not in quarter_ends  # the 10-K annual figure
    assert date(2023, 6, 30) not in quarter_ends  # the half-year figure


def test_revenue_tag_fallback_priority():
    facts = _facts_json()
    facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "units": {"USD": [
            {"start": "2023-04-01", "end": "2023-06-30", "val": 1200.0, "accn": "acc-2",
             "fy": 2023, "fp": "Q2", "form": "10-Q", "filed": "2023-07-28"},
        ]}
    }
    filings = parse_company_facts_to_filings("TEST", facts, source="sec_edgar", retrieved_at=_RETRIEVED_AT)
    by_accn = {f.accession_number: f for f in filings}
    assert by_accn["acc-2"].revenue == 1200.0


def test_fetch_company_facts_uses_correct_url_and_headers():
    client = FakeHttpClient(json_responses={
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json": _facts_json(),
    })
    result = fetch_company_facts(client, "0000320193", user_agent="Test test@example.com")
    assert result["cik"] == 320193
    assert client.requested_headers[0]["User-Agent"] == "Test test@example.com"


def test_fetch_filings_for_symbol_returns_empty_for_unknown_symbol():
    client = FakeHttpClient()
    result = fetch_filings_for_symbol(
        client, "NOPE", {}, user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT,
    )
    assert result == []


def test_fetch_filings_for_symbol_end_to_end():
    client = FakeHttpClient(json_responses={
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json": _facts_json(),
    })
    result = fetch_filings_for_symbol(
        client, "AAPL", {"AAPL": "0000320193"}, user_agent="Test test@example.com", retrieved_at=_RETRIEVED_AT,
    )
    assert len(result) == 1
    assert result[0].symbol == "AAPL"
