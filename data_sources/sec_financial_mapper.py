from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from data_sources.sec_edgar import fetch_company_facts


# Standard metrics used by the Financial Intelligence Engine.
# Multiple XBRL concepts are provided because companies can change
# taxonomy concepts over time.
METRIC_CANDIDATES = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
    ],
    "inventory": [
        "InventoryNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
    ],
    "total_debt": [
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligations",
    ],
}


DURATION_METRICS = {
    "revenue",
    "net_income",
    "operating_cash_flow",
}


INSTANT_METRICS = {
    "accounts_receivable",
    "inventory",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "current_assets",
    "current_liabilities",
    "cash_and_equivalents",
    "total_debt",
}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _annual_duration_row(row: dict[str, Any]) -> bool:
    """
    Identify full-year 10-K duration facts.

    A fiscal year can contain 52 or 53 weeks, so use a broad
    duration range rather than assuming exactly 365 days.
    """
    if row.get("form") != "10-K":
        return False

    start = _parse_date(row.get("start"))
    end = _parse_date(row.get("end"))

    if start is None or end is None:
        return False

    days = (end - start).days

    return 300 <= days <= 380


def _annual_instant_row(row: dict[str, Any]) -> bool:
    """
    Identify balance-sheet facts reported in a 10-K.
    """
    return (
        row.get("form") == "10-K"
        and row.get("end") is not None
        and row.get("start") is None
    )


def _concept_observations(
    facts: dict[str, Any],
    concept: str,
    metric: str,
) -> dict[str, dict[str, Any]]:
    item = facts.get(concept)

    if not item:
        return {}

    units = item.get("units", {})
    rows = units.get("USD", [])

    if not rows:
        return {}

    if metric in DURATION_METRICS:
        rows = [
            row
            for row in rows
            if _annual_duration_row(row)
        ]
    else:
        rows = [
            row
            for row in rows
            if _annual_instant_row(row)
        ]

    # SEC company facts often contain repeated comparative values.
    # Keep one observation per reporting end date, preferring the
    # most recently filed version.
    by_end: dict[str, dict[str, Any]] = {}

    for row in rows:
        end = row.get("end")

        if not end:
            continue

        existing = by_end.get(end)

        if (
            existing is None
            or row.get("filed", "") > existing.get("filed", "")
        ):
            by_end[end] = row

    return by_end


def _select_best_concept(
    facts: dict[str, Any],
    metric: str,
    candidates: list[str],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """
    Choose the candidate with the freshest annual SEC coverage.

    This avoids assuming one XBRL tag is used forever. For example,
    a company can migrate from one revenue concept to another.
    """
    choices = []

    for concept in candidates:
        observations = _concept_observations(
            facts,
            concept,
            metric,
        )

        if not observations:
            continue

        latest_end = max(observations)

        choices.append(
            (
                latest_end,
                len(observations),
                concept,
                observations,
            )
        )

    if not choices:
        return None, {}

    # Freshest coverage first; number of annual periods breaks ties.
    choices.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    _, _, concept, observations = choices[0]

    return concept, observations


def _build_debt_fallback(
    facts: dict[str, Any],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """
    If a direct total long-term debt concept is unavailable,
    derive debt from current + non-current long-term debt.
    """
    current_candidates = [
        "LongTermDebtCurrent",
        "DebtCurrent",
    ]

    noncurrent_candidates = [
        "LongTermDebtNoncurrent",
    ]

    current_concept, current_rows = _select_best_concept(
        facts,
        "total_debt",
        current_candidates,
    )

    noncurrent_concept, noncurrent_rows = _select_best_concept(
        facts,
        "total_debt",
        noncurrent_candidates,
    )

    if not current_rows or not noncurrent_rows:
        return None, {}

    common_ends = set(current_rows) & set(noncurrent_rows)

    combined = {}

    for end in common_ends:
        current_value = current_rows[end].get("val")
        noncurrent_value = noncurrent_rows[end].get("val")

        if current_value is None or noncurrent_value is None:
            continue

        combined[end] = {
            "end": end,
            "val": current_value + noncurrent_value,
            "form": "10-K",
            "filed": max(
                current_rows[end].get("filed", ""),
                noncurrent_rows[end].get("filed", ""),
            ),
        }

    if not combined:
        return None, {}

    label = f"{current_concept} + {noncurrent_concept}"

    return label, combined


def build_sec_annual_financial_statement(
    ticker: str,
) -> dict[str, Any]:
    """
    Convert SEC Company Facts into the standard two-period dataframe
    consumed by the Financial Intelligence Engine.

    Version 1 deliberately uses annual 10-K data only.
    """
    company = fetch_company_facts(ticker)

    facts = company["facts"].get("us-gaap", {})

    if not facts:
        raise ValueError(
            f"No US-GAAP company facts were found for {ticker.upper()}."
        )

    metric_rows: dict[str, dict[str, dict[str, Any]]] = {}
    selected_concepts: dict[str, str | None] = {}
    warnings: list[str] = []

    for metric, candidates in METRIC_CANDIDATES.items():
        concept, observations = _select_best_concept(
            facts,
            metric,
            candidates,
        )

        if metric == "total_debt" and not observations:
            concept, observations = _build_debt_fallback(facts)

        selected_concepts[metric] = concept
        metric_rows[metric] = observations

        if not observations:
            warnings.append(
                f"No suitable annual SEC fact found for {metric}."
            )

    # Total assets is a reliable balance-sheet anchor for annual periods.
    reference_rows = metric_rows.get("total_assets", {})

    if len(reference_rows) < 2:
        raise ValueError(
            "At least two annual balance-sheet periods are required."
        )

    latest_periods = sorted(
        reference_rows.keys(),
        reverse=True,
    )[:2]

    latest_periods = sorted(latest_periods)

    records = []

    for period_end in latest_periods:
        record: dict[str, Any] = {
            "period": period_end,
        }

        for metric in METRIC_CANDIDATES:
            observation = metric_rows.get(metric, {}).get(period_end)

            record[metric] = (
                observation.get("val")
                if observation is not None
                else None
            )

        records.append(record)

    dataframe = pd.DataFrame(records)

    missing_by_period = {}

    for _, row in dataframe.iterrows():
        period = row["period"]

        missing = [
            metric
            for metric in METRIC_CANDIDATES
            if pd.isna(row[metric])
        ]

        missing_by_period[period] = missing

    return {
        "ticker": company["ticker"],
        "cik": company["cik"],
        "company_name": company["company_name"],
        "dataframe": dataframe,
        "selected_concepts": selected_concepts,
        "missing_by_period": missing_by_period,
        "warnings": warnings,
        "methodology": (
            "Annual 10-K facts only. Duration metrics use full-year "
            "observations; balance-sheet metrics use reporting-date "
            "instant facts. Repeated SEC facts are deduplicated by "
            "reporting end date using the most recently filed value."
        ),
    }
