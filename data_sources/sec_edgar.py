from __future__ import annotations

import gzip
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)


def _get_user_agent() -> str:
    """
    SEC requests should identify the automated client.

    Set locally with:
    export SEC_USER_AGENT="Financial Data Intelligence your-email@example.com"
    """
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()

    if not user_agent:
        raise ValueError(
            "SEC_USER_AGENT is not configured. "
            "Set it before requesting SEC EDGAR data."
        )

    return user_agent


def _request_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": _get_user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read()
            content_encoding = response.headers.get(
                "Content-Encoding",
                "",
            ).lower()

            if content_encoding == "gzip" or raw.startswith(b"\x1f\x8b"):
                raw = gzip.decompress(raw)

            return json.loads(raw.decode("utf-8"))

    except HTTPError as exc:
        raise RuntimeError(
            f"SEC request failed with HTTP {exc.code}: {url}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to SEC EDGAR: {exc.reason}"
        ) from exc


def resolve_ticker(ticker: str) -> dict[str, Any]:
    """
    Resolve a ticker such as AAPL or NVDA to SEC company metadata.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    companies = _request_json(SEC_TICKERS_URL)

    for company in companies.values():
        if str(company.get("ticker", "")).upper() == ticker:
            cik = str(company["cik_str"]).zfill(10)

            return {
                "ticker": ticker,
                "cik": cik,
                "company_name": company.get("title"),
            }

    raise ValueError(f"Ticker '{ticker}' was not found in SEC company data.")


def fetch_company_facts(ticker: str) -> dict[str, Any]:
    """
    Fetch all standard XBRL company facts for a public company.
    """
    company = resolve_ticker(ticker)

    url = SEC_COMPANY_FACTS_URL.format(cik=company["cik"])
    payload = _request_json(url)

    facts = payload.get("facts", {})

    return {
        "ticker": company["ticker"],
        "cik": company["cik"],
        "company_name": payload.get(
            "entityName",
            company["company_name"],
        ),
        "facts": facts,
        "taxonomies": list(facts.keys()),
    }
