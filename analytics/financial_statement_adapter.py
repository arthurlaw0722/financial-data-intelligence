from __future__ import annotations

from typing import Any

import pandas as pd


FINANCIAL_METRICS = [
    "revenue",
    "accounts_receivable",
    "inventory",
    "net_income",
    "operating_cash_flow",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "current_assets",
    "current_liabilities",
    "total_debt",
    "cash_and_equivalents",
]


def detect_financial_statement_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect whether a dataframe resembles standardized
    period-based financial statement data.
    """

    available_metrics = [
        column
        for column in FINANCIAL_METRICS
        if column in df.columns
    ]

    missing_metrics = [
        column
        for column in FINANCIAL_METRICS
        if column not in df.columns
    ]

    has_period = "period" in df.columns

    # Require enough financial metrics to avoid treating an unrelated
    # dataset as a financial-statement dataset.
    is_financial_statement = (
        has_period
        and len(df) >= 2
        and len(available_metrics) >= 5
    )

    return {
        "is_financial_statement": is_financial_statement,
        "has_period": has_period,
        "available_metrics": available_metrics,
        "missing_metrics": missing_metrics,
        "metric_count": len(available_metrics),
    }


def _clean_value(value: Any) -> float | None:
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prepare_financial_periods(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Convert a standardized financial-statement dataframe into the
    current/previous dictionaries expected by the Financial
    Intelligence Engine.

    Required structure:

        period | revenue | net_income | total_assets | ...

    The latest two valid periods are used.
    """

    schema = detect_financial_statement_schema(df)

    if not schema["has_period"]:
        raise ValueError(
            "Financial statement data requires a 'period' column."
        )

    if len(df) < 2:
        raise ValueError(
            "At least two financial periods are required for comparison."
        )

    working = df.copy()

    parsed_period = pd.to_datetime(
        working["period"],
        errors="coerce",
    )

    if parsed_period.notna().sum() < 2:
        raise ValueError(
            "The 'period' column must contain at least two valid dates."
        )

    working["_parsed_period"] = parsed_period
    working = working.loc[
        working["_parsed_period"].notna()
    ].sort_values("_parsed_period")

    if len(working) < 2:
        raise ValueError(
            "At least two valid financial periods are required."
        )

    previous_row = working.iloc[-2]
    current_row = working.iloc[-1]

    def row_to_metrics(row: pd.Series) -> dict[str, float | None]:
        return {
            metric: (
                _clean_value(row[metric])
                if metric in working.columns
                else None
            )
            for metric in FINANCIAL_METRICS
        }

    previous = row_to_metrics(previous_row)
    current = row_to_metrics(current_row)

    history = [
        {
            "period": str(row["period"]),
            **row_to_metrics(row),
        }
        for _, row in working.iterrows()
    ]

    return {
        "previous_period": str(previous_row["period"]),
        "current_period": str(current_row["period"]),
        "previous": previous,
        "current": current,
        "history": history,
        "history_period_count": len(history),
        "available_metrics": schema["available_metrics"],
        "missing_metrics": schema["missing_metrics"],
        "schema": schema,
    }
