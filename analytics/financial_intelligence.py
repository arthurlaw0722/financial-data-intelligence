from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    """Safely convert a financial value to float."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator = _number(numerator)
    denominator = _number(denominator)

    if numerator is None or denominator is None or denominator == 0:
        return None

    return numerator / denominator


def _growth(current: Any, previous: Any) -> float | None:
    """
    Return percentage growth.

    Example:
    previous = 100
    current = 120
    result = 20.0
    """
    current = _number(current)
    previous = _number(previous)

    if current is None or previous is None or previous == 0:
        return None

    return ((current - previous) / abs(previous)) * 100


def _signal(
    area: str,
    status: str,
    title: str,
    interpretation: str,
    action: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "area": area,
        "status": status,
        "title": title,
        "interpretation": interpretation,
        "analyst_action": action,
        "metrics": metrics,
    }


def accounting_integrity_signal(
    current: dict[str, Any],
    tolerance_pct: float = 0.5,
) -> dict[str, Any]:
    """
    Check whether:
    Assets ≈ Liabilities + Equity
    """

    assets = _number(current.get("total_assets"))
    liabilities = _number(current.get("total_liabilities"))
    equity = _number(current.get("total_equity"))

    if None in (assets, liabilities, equity):
        return _signal(
            area="Accounting Integrity",
            status="Review",
            title="Insufficient balance-sheet data",
            interpretation=(
                "Assets, liabilities, and equity are required to test "
                "the accounting equation."
            ),
            action=(
                "Confirm that total assets, total liabilities, and "
                "total equity were extracted correctly."
            ),
            metrics={},
        )

    expected_assets = liabilities + equity
    difference = assets - expected_assets

    denominator = max(abs(assets), 1.0)
    difference_pct = abs(difference) / denominator * 100

    if difference_pct <= tolerance_pct:
        status = "Normal"
        interpretation = (
            "The balance sheet reconciles within the configured tolerance."
        )
        action = (
            "No immediate reconciliation issue detected. Continue with "
            "the remaining financial checks."
        )
    elif difference_pct <= 2.0:
        status = "Review"
        interpretation = (
            "The accounting equation shows a small reconciliation difference."
        )
        action = (
            "Review XBRL tags, rounding differences, minority interests, "
            "and extraction consistency."
        )
    else:
        status = "High Attention"
        interpretation = (
            "Reported assets do not reconcile with liabilities plus equity "
            "within a reasonable tolerance."
        )
        action = (
            "Validate the source filing and extracted accounting facts "
            "before relying on downstream analysis."
        )

    return _signal(
        area="Accounting Integrity",
        status=status,
        title="Balance-sheet reconciliation",
        interpretation=interpretation,
        action=action,
        metrics={
            "assets": assets,
            "liabilities_plus_equity": expected_assets,
            "difference": difference,
            "difference_pct": round(difference_pct, 4),
        },
    )



def _build_derived_financial_metrics(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """
    Build reusable analyst-oriented financial ratios from standardized
    financial-statement data.

    These metrics are deterministic calculations intended to support
    downstream diagnostics, peer comparison and evidence-grounded AI
    explanations.
    """

    def number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def safe_ratio(
        numerator: Any,
        denominator: Any,
    ) -> float | None:
        numerator_value = number(numerator)
        denominator_value = number(denominator)

        if (
            numerator_value is None
            or denominator_value is None
            or denominator_value == 0
        ):
            return None

        return numerator_value / denominator_value

    def average_balance(
        current_value: Any,
        previous_value: Any,
    ) -> float | None:
        current_number = number(current_value)
        previous_number = number(previous_value)

        if current_number is None or previous_number is None:
            return None

        return (current_number + previous_number) / 2

    def pct(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value * 100, 2)

    def multiple(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 2)

    revenue = current.get("revenue")
    net_income = current.get("net_income")
    operating_cash_flow = current.get("operating_cash_flow")

    total_assets = current.get("total_assets")
    total_equity = current.get("total_equity")

    average_assets = average_balance(
        total_assets,
        previous.get("total_assets"),
    )

    average_equity = average_balance(
        total_equity,
        previous.get("total_equity"),
    )

    net_margin = safe_ratio(
        net_income,
        revenue,
    )

    operating_cash_flow_margin = safe_ratio(
        operating_cash_flow,
        revenue,
    )

    cash_conversion_of_earnings = safe_ratio(
        operating_cash_flow,
        net_income,
    )

    return_on_assets = safe_ratio(
        net_income,
        average_assets,
    )

    return_on_equity = safe_ratio(
        net_income,
        average_equity,
    )

    inventory_intensity = safe_ratio(
        current.get("inventory"),
        revenue,
    )

    receivables_intensity = safe_ratio(
        current.get("accounts_receivable"),
        revenue,
    )

    current_ratio = safe_ratio(
        current.get("current_assets"),
        current.get("current_liabilities"),
    )

    debt_to_assets = safe_ratio(
        current.get("total_debt"),
        total_assets,
    )

    debt_to_equity = safe_ratio(
        current.get("total_debt"),
        total_equity,
    )

    cash_to_debt = safe_ratio(
        current.get("cash_and_equivalents"),
        current.get("total_debt"),
    )

    metrics = {
        "net_margin_pct": pct(net_margin),
        "operating_cash_flow_margin_pct": pct(
            operating_cash_flow_margin
        ),
        "cash_conversion_of_earnings": multiple(
            cash_conversion_of_earnings
        ),
        "return_on_assets_pct": pct(return_on_assets),
        "return_on_equity_pct": pct(return_on_equity),
        "inventory_intensity_pct": pct(inventory_intensity),
        "receivables_intensity_pct": pct(receivables_intensity),
        "current_ratio": multiple(current_ratio),
        "debt_to_assets_pct": pct(debt_to_assets),
        "debt_to_equity": multiple(debt_to_equity),
        "cash_to_debt": multiple(cash_to_debt),
    }

    available_metrics = [
        key
        for key, value in metrics.items()
        if value is not None
    ]

    missing_metrics = [
        key
        for key, value in metrics.items()
        if value is None
    ]

    return {
        "metrics": metrics,
        "available_metrics": available_metrics,
        "missing_metrics": missing_metrics,
        "methodology": {
            "net_margin": "Net income / revenue",
            "operating_cash_flow_margin": (
                "Operating cash flow / revenue"
            ),
            "cash_conversion_of_earnings": (
                "Operating cash flow / net income"
            ),
            "return_on_assets": (
                "Net income / average total assets"
            ),
            "return_on_equity": (
                "Net income / average total equity"
            ),
            "inventory_intensity": "Inventory / revenue",
            "receivables_intensity": (
                "Accounts receivable / revenue"
            ),
            "current_ratio": (
                "Current assets / current liabilities"
            ),
            "debt_to_assets": "Total debt / total assets",
            "debt_to_equity": "Total debt / total equity",
            "cash_to_debt": (
                "Cash and cash equivalents / total debt"
            ),
        },
        "notes": [
            (
                "ROA and ROE use average opening and closing balance-sheet "
                "values when both periods are available."
            ),
            (
                "Metrics are analytical ratios derived from standardized "
                "financial-statement data and are not investment advice."
            ),
        ],
    }


def working_capital_signal(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    revenue_growth = _growth(
        current.get("revenue"),
        previous.get("revenue"),
    )

    receivables_growth = _growth(
        current.get("accounts_receivable"),
        previous.get("accounts_receivable"),
    )

    inventory_growth = _growth(
        current.get("inventory"),
        previous.get("inventory"),
    )

    ar_gap = (
        receivables_growth - revenue_growth
        if revenue_growth is not None
        and receivables_growth is not None
        else None
    )

    inventory_gap = (
        inventory_growth - revenue_growth
        if revenue_growth is not None
        and inventory_growth is not None
        else None
    )

    status = "Normal"
    reasons = []

    if ar_gap is not None:
        if ar_gap > 25:
            status = "High Attention"
            reasons.append(
                "Accounts receivable is growing much faster than revenue."
            )
        elif ar_gap > 10:
            status = "Review"
            reasons.append(
                "Accounts receivable is growing faster than revenue."
            )

    if inventory_gap is not None:
        if inventory_gap > 30:
            status = "High Attention"
            reasons.append(
                "Inventory growth materially exceeds revenue growth."
            )
        elif inventory_gap > 15:
            if status == "Normal":
                status = "Review"
            reasons.append(
                "Inventory growth is running ahead of revenue growth."
            )

    if not reasons:
        interpretation = (
            "Receivables and inventory movements do not show a material "
            "divergence from revenue under the current heuristic rules."
        )
        action = (
            "Continue monitoring working-capital trends across future periods."
        )
    else:
        interpretation = " ".join(reasons)
        action = (
            "Review days sales outstanding, receivables ageing, inventory "
            "turnover, customer payment terms, and related management commentary."
        )

    return _signal(
        area="Working Capital",
        status=status,
        title="Revenue vs working-capital growth",
        interpretation=interpretation,
        action=action,
        metrics={
            "revenue_growth_pct": (
                round(revenue_growth, 2)
                if revenue_growth is not None
                else None
            ),
            "receivables_growth_pct": (
                round(receivables_growth, 2)
                if receivables_growth is not None
                else None
            ),
            "inventory_growth_pct": (
                round(inventory_growth, 2)
                if inventory_growth is not None
                else None
            ),
            "receivables_vs_revenue_gap_pp": (
                round(ar_gap, 2)
                if ar_gap is not None
                else None
            ),
            "inventory_vs_revenue_gap_pp": (
                round(inventory_gap, 2)
                if inventory_gap is not None
                else None
            ),
        },
    )



def _build_working_capital_diagnostic(
    signals: list[dict[str, Any]],
    trend_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build a structured analyst-style diagnostic for working-capital signals.

    The diagnostic separates observed evidence from possible explanations.
    It does not infer management intent or claim causality from financial
    statement movements alone.
    """

    working_signal = next(
        (
            signal
            for signal in signals
            if signal.get("area") == "Working Capital"
        ),
        None,
    )

    earnings_signal = next(
        (
            signal
            for signal in signals
            if signal.get("area") == "Earnings Quality"
        ),
        None,
    )

    if not working_signal:
        return {
            "status": "Unavailable",
            "observed_issue": (
                "Working-capital diagnostic is unavailable because the "
                "underlying signal was not produced."
            ),
            "supporting_evidence": [],
            "historical_pattern": (
                "Insufficient information to assess the historical pattern."
            ),
            "historical_pattern_label": "Unavailable",
            "interpretation": (
                "No diagnostic interpretation is available."
            ),
            "alternative_explanations": [],
            "evidence_gaps": [],
            "analyst_action": (
                "Confirm that revenue, receivables and inventory data are "
                "available for comparable fiscal periods."
            ),
            "metrics": {},
        }

    metrics = working_signal.get("metrics", {})
    earnings_metrics = (
        earnings_signal.get("metrics", {})
        if earnings_signal
        else {}
    )

    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    revenue_growth = metrics.get("revenue_growth_pct")
    receivables_growth = metrics.get("receivables_growth_pct")
    inventory_growth = metrics.get("inventory_growth_pct")
    receivables_gap = metrics.get("receivables_vs_revenue_gap_pp")
    inventory_gap = metrics.get("inventory_vs_revenue_gap_pp")
    operating_cash_flow_growth = earnings_metrics.get(
        "operating_cash_flow_growth_pct"
    )

    inventory_flagged = (
        is_number(inventory_gap)
        and inventory_gap > 15
    )
    receivables_flagged = (
        is_number(receivables_gap)
        and receivables_gap > 10
    )

    if inventory_flagged and receivables_flagged:
        dominant_metric = (
            "inventory"
            if inventory_gap >= receivables_gap
            else "accounts_receivable"
        )
    elif inventory_flagged:
        dominant_metric = "inventory"
    elif receivables_flagged:
        dominant_metric = "accounts_receivable"
    else:
        dominant_metric = None

    # ------------------------------------------------------------
    # Observed issue
    # ------------------------------------------------------------
    issue_parts = []

    if inventory_flagged and is_number(inventory_growth):
        issue_parts.append(
            "Inventory growth exceeded revenue growth by "
            f"{inventory_gap:.2f} percentage points "
            f"({inventory_growth:.2f}% vs "
            f"{revenue_growth:.2f}%)."
        )

    if receivables_flagged and is_number(receivables_growth):
        issue_parts.append(
            "Receivables growth exceeded revenue growth by "
            f"{receivables_gap:.2f} percentage points "
            f"({receivables_growth:.2f}% vs "
            f"{revenue_growth:.2f}%)."
        )

    if issue_parts:
        observed_issue = " ".join(issue_parts)
    else:
        observed_issue = (
            "No material receivables or inventory divergence from revenue "
            "was identified under the current working-capital rules."
        )

    # ------------------------------------------------------------
    # Supporting cross-statement evidence
    # ------------------------------------------------------------
    supporting_evidence = []

    if is_number(revenue_growth):
        supporting_evidence.append(
            f"Revenue grew {revenue_growth:.2f}% in the latest fiscal year."
        )

    if is_number(receivables_gap):
        if abs(receivables_gap) <= 10:
            supporting_evidence.append(
                "Receivables growth remained broadly aligned with revenue, "
                f"with a {receivables_gap:+.2f} percentage-point gap."
            )
        elif receivables_gap > 10:
            supporting_evidence.append(
                "Receivables growth ran ahead of revenue by "
                f"{receivables_gap:.2f} percentage points."
            )
        else:
            supporting_evidence.append(
                "Receivables growth was below revenue growth by "
                f"{abs(receivables_gap):.2f} percentage points."
            )

    if is_number(inventory_gap):
        if abs(inventory_gap) <= 15:
            supporting_evidence.append(
                "Inventory growth remained reasonably aligned with revenue, "
                f"with a {inventory_gap:+.2f} percentage-point gap."
            )
        elif inventory_gap > 15:
            supporting_evidence.append(
                "Inventory growth ran ahead of revenue by "
                f"{inventory_gap:.2f} percentage points."
            )
        else:
            supporting_evidence.append(
                "Inventory growth was below revenue growth by "
                f"{abs(inventory_gap):.2f} percentage points."
            )

    if is_number(operating_cash_flow_growth):
        direction = (
            "increased"
            if operating_cash_flow_growth >= 0
            else "declined"
        )
        supporting_evidence.append(
            "Operating cash flow "
            f"{direction} {abs(operating_cash_flow_growth):.2f}% "
            "over the same period."
        )

    # ------------------------------------------------------------
    # Historical intensity analysis
    # ------------------------------------------------------------
    metric_trends = (
        (trend_context or {}).get("metric_trends", {})
    )

    revenue_series = (
        metric_trends
        .get("revenue", {})
        .get("series", [])
    )

    dominant_series = (
        metric_trends
        .get(dominant_metric, {})
        .get("series", [])
        if dominant_metric
        else []
    )

    revenue_by_period = {
        str(item.get("period")): item.get("value")
        for item in revenue_series
        if item.get("period") is not None
    }

    intensity_series = []

    for item in dominant_series:
        period = str(item.get("period"))
        driver_value = item.get("value")
        revenue_value = revenue_by_period.get(period)

        if (
            is_number(driver_value)
            and is_number(revenue_value)
            and revenue_value > 0
            and driver_value >= 0
        ):
            intensity_series.append(
                {
                    "period": period,
                    "ratio_pct": round(
                        driver_value / revenue_value * 100,
                        2,
                    ),
                }
            )

    historical_pattern_label = "Insufficient history"
    historical_pattern = (
        "There is not enough aligned multi-year history to determine "
        "whether the latest working-capital divergence is persistent."
    )

    historical_level_context = {
        "label": "Insufficient history",
        "latest_ratio_pct": None,
        "prior_median_pct": None,
        "prior_min_pct": None,
        "prior_max_pct": None,
        "latest_vs_median_pct": None,
        "interpretation": (
            "There is not enough prior history to assess whether the "
            "latest working-capital intensity is unusual."
        ),
    }

    if len(intensity_series) >= 3:
        ratios = [
            item["ratio_pct"]
            for item in intensity_series
        ]

        latest_ratio = ratios[-1]
        previous_ratio = ratios[-2]
        recent_ratios = ratios[-4:]

        prior_ratios = ratios[:-1]

        if len(prior_ratios) >= 2:
            sorted_prior = sorted(prior_ratios)
            midpoint = len(sorted_prior) // 2

            if len(sorted_prior) % 2 == 0:
                prior_median = (
                    sorted_prior[midpoint - 1]
                    + sorted_prior[midpoint]
                ) / 2
            else:
                prior_median = sorted_prior[midpoint]

            prior_min = min(prior_ratios)
            prior_max = max(prior_ratios)

            latest_vs_median_pct = (
                ((latest_ratio - prior_median) / abs(prior_median)) * 100
                if prior_median != 0
                else None
            )

            if latest_ratio > prior_max:
                level_label = "Above historical range"
                level_interpretation = (
                    f"The latest intensity of {latest_ratio:.2f}% is above "
                    f"the prior historical maximum of {prior_max:.2f}%."
                )

            elif latest_ratio < prior_min:
                level_label = "Below historical range"
                level_interpretation = (
                    f"The latest intensity of {latest_ratio:.2f}% is below "
                    f"the prior historical minimum of {prior_min:.2f}%."
                )

            else:
                level_label = "Within historical range"
                level_interpretation = (
                    f"The latest intensity of {latest_ratio:.2f}% remains "
                    f"within the prior historical range of "
                    f"{prior_min:.2f}% to {prior_max:.2f}% and compares "
                    f"with a prior median of {prior_median:.2f}%."
                )

            historical_level_context = {
                "label": level_label,
                "latest_ratio_pct": round(latest_ratio, 2),
                "prior_median_pct": round(prior_median, 2),
                "prior_min_pct": round(prior_min, 2),
                "prior_max_pct": round(prior_max, 2),
                "latest_vs_median_pct": (
                    round(latest_vs_median_pct, 2)
                    if latest_vs_median_pct is not None
                    else None
                ),
                "interpretation": level_interpretation,
            }

        latest_change_pp = latest_ratio - previous_ratio

        relative_change_pct = None
        if previous_ratio != 0:
            relative_change_pct = (
                latest_change_pp
                / abs(previous_ratio)
                * 100
            )

        increasing_steps = sum(
            current_value > prior_value
            for prior_value, current_value
            in zip(recent_ratios, recent_ratios[1:])
        )

        decreasing_steps = sum(
            current_value < prior_value
            for prior_value, current_value
            in zip(recent_ratios, recent_ratios[1:])
        )

        driver_label = (
            "Inventory intensity"
            if dominant_metric == "inventory"
            else "Receivables intensity"
        )

        if (
            len(recent_ratios) >= 4
            and increasing_steps == len(recent_ratios) - 1
        ):
            historical_pattern_label = "Persistent build"
            historical_pattern = (
                f"{driver_label} has increased across each of the latest "
                f"{len(recent_ratios)} available fiscal periods, reaching "
                f"{latest_ratio:.2f}% of revenue in the latest period."
            )

        elif (
            len(recent_ratios) >= 4
            and decreasing_steps == len(recent_ratios) - 1
        ):
            historical_pattern_label = "Persistent easing"
            historical_pattern = (
                f"{driver_label} has declined across each of the latest "
                f"{len(recent_ratios)} available fiscal periods, reaching "
                f"{latest_ratio:.2f}% of revenue in the latest period."
            )

        elif (
            latest_change_pp >= 1.5
            and relative_change_pct is not None
            and relative_change_pct >= 15
        ):
            historical_pattern_label = "Recent step-up"
            historical_pattern = (
                f"{driver_label} increased from "
                f"{previous_ratio:.2f}% to {latest_ratio:.2f}% of revenue "
                "in the latest fiscal period, indicating a recent step-up "
                "rather than a clearly persistent multi-year build."
            )

        elif (
            abs(latest_change_pp) <= 1.0
            and max(ratios[-3:]) - min(ratios[-3:]) <= 2.0
        ):
            historical_pattern_label = "Broadly stable"
            historical_pattern = (
                f"{driver_label} has remained broadly stable across the "
                f"latest three fiscal periods and is currently "
                f"{latest_ratio:.2f}% of revenue."
            )

        else:
            historical_pattern_label = "Mixed trend"
            historical_pattern = (
                f"{driver_label} shows a mixed multi-year pattern. "
                f"The latest level is {latest_ratio:.2f}% of revenue, "
                f"compared with {previous_ratio:.2f}% in the prior period."
            )

    # ------------------------------------------------------------
    # Diagnostic interpretation
    # ------------------------------------------------------------
    earnings_status = (
        earnings_signal.get("status")
        if earnings_signal
        else None
    )

    if working_signal.get("status") == "Normal":
        interpretation = (
            "The available evidence does not currently indicate a material "
            "working-capital divergence from revenue."
        )

    elif inventory_flagged and not receivables_flagged:
        if (
            is_number(operating_cash_flow_growth)
            and operating_cash_flow_growth >= 0
            and earnings_status == "Normal"
        ):
            interpretation = (
                "The available evidence suggests that the working-capital "
                "divergence is concentrated in inventory rather than a "
                "broader deterioration across receivables and cash "
                "generation."
            )
        else:
            interpretation = (
                "The main working-capital divergence is concentrated in "
                "inventory. Cash-generation evidence should be reviewed "
                "alongside the inventory build before drawing a stronger "
                "conclusion."
            )

    elif receivables_flagged and not inventory_flagged:
        interpretation = (
            "The main working-capital divergence is concentrated in "
            "receivables, which may indicate changing customer mix, payment "
            "terms or collection performance and warrants further review."
        )

    elif inventory_flagged and receivables_flagged:
        interpretation = (
            "Both inventory and receivables are expanding faster than "
            "revenue, indicating a broader working-capital build rather "
            "than a single-account divergence."
        )

    else:
        interpretation = working_signal.get(
            "interpretation",
            "Working-capital movements require further review.",
        )

    # ------------------------------------------------------------
    # Historical level context for the diagnostic interpretation
    # ------------------------------------------------------------
    level_label = historical_level_context.get("label")
    latest_ratio_pct = historical_level_context.get("latest_ratio_pct")
    prior_median_pct = historical_level_context.get("prior_median_pct")
    prior_max_pct = historical_level_context.get("prior_max_pct")
    latest_vs_median_pct = historical_level_context.get(
        "latest_vs_median_pct"
    )

    driver_name = (
        "inventory intensity"
        if dominant_metric == "inventory"
        else "receivables intensity"
        if dominant_metric == "accounts_receivable"
        else "working-capital intensity"
    )

    if working_signal.get("status") != "Normal":
        if level_label == "Within historical range":
            context_sentence = (
                f"However, the latest {driver_name} of "
                f"{latest_ratio_pct:.2f}% remains within the company's "
                f"prior historical range"
            )

            if isinstance(prior_median_pct, (int, float)):
                context_sentence += (
                    f" and compares with a prior median of "
                    f"{prior_median_pct:.2f}%"
                )

            context_sentence += ". "

            if isinstance(latest_vs_median_pct, (int, float)):
                context_sentence += (
                    f"The latest level is {latest_vs_median_pct:.2f}% "
                    "above the prior median. "
                )

            context_sentence += (
                "This tempers the severity of the movement signal and "
                "does not by itself indicate persistent structural "
                "deterioration."
            )

            interpretation = (
                interpretation.rstrip()
                + " "
                + context_sentence
            )

        elif level_label == "Above historical range":
            context_sentence = (
                f"The latest {driver_name} of {latest_ratio_pct:.2f}% "
                "is also above the company's prior historical range"
            )

            if isinstance(prior_max_pct, (int, float)):
                context_sentence += (
                    f", exceeding the previous maximum of "
                    f"{prior_max_pct:.2f}%"
                )

            context_sentence += (
                ". This strengthens the case for further analyst review."
            )

            interpretation = (
                interpretation.rstrip()
                + " "
                + context_sentence
            )

        elif level_label == "Below historical range":
            interpretation = (
                interpretation.rstrip()
                + " "
                + (
                    f"Despite the latest growth divergence, the current "
                    f"{driver_name} remains below its prior historical "
                    "range, which weakens the evidence of structural "
                    "working-capital deterioration."
                )
            )

    # ------------------------------------------------------------
    # Alternative explanations and evidence gaps
    # ------------------------------------------------------------
    alternative_explanations = []
    evidence_gaps = []

    if inventory_flagged:
        alternative_explanations.extend(
            [
                (
                    "Planned inventory build ahead of expected demand, "
                    "product launches or capacity expansion."
                ),
                (
                    "Supply-chain buffering or deliberate inventory "
                    "positioning."
                ),
                (
                    "Slower inventory conversion or weaker-than-expected "
                    "sell-through."
                ),
            ]
        )

        evidence_gaps.extend(
            [
                "Inventory turnover or days inventory outstanding.",
                "Cost of revenue and gross-margin movement.",
                "Inventory ageing, write-down or obsolescence disclosures.",
                "Management commentary explaining the inventory build.",
            ]
        )

    if receivables_flagged:
        alternative_explanations.extend(
            [
                (
                    "Sales mix shifting toward customers or channels with "
                    "longer payment terms."
                ),
                "Temporary collection timing effects.",
                (
                    "Looser credit terms or deterioration in collection "
                    "performance."
                ),
            ]
        )

        evidence_gaps.extend(
            [
                "Days sales outstanding.",
                "Receivables ageing and bad-debt allowance information.",
                "Customer payment terms and channel mix.",
                "Management commentary on collection performance.",
            ]
        )

    if not alternative_explanations:
        alternative_explanations.append(
            "No material working-capital divergence currently requires a "
            "specific causal hypothesis."
        )

    if not evidence_gaps:
        evidence_gaps.append(
            "Continue monitoring future revenue, receivables and inventory "
            "movements for emerging divergence."
        )

    if inventory_flagged and receivables_flagged:
        analyst_action = (
            "Review inventory conversion, receivables collection, margin "
            "movement and management commentary together to determine "
            "whether the broader working-capital build is operationally "
            "supported."
        )
    elif inventory_flagged:
        analyst_action = (
            "Review inventory turnover, cost of revenue, gross-margin "
            "movement and management commentary. Escalate only if the "
            "inventory build persists, margins weaken, or cash conversion "
            "deteriorates."
        )
    elif receivables_flagged:
        analyst_action = (
            "Review days sales outstanding, receivables ageing, customer "
            "payment terms and management commentary. Escalate if collection "
            "performance deteriorates or cash conversion weakens."
        )
    else:
        analyst_action = (
            "Continue monitoring working-capital trends across future "
            "fiscal periods."
        )

    return {
        "status": working_signal.get("status"),
        "observed_issue": observed_issue,
        "supporting_evidence": supporting_evidence,
        "historical_pattern": historical_pattern,
        "historical_pattern_label": historical_pattern_label,
        "historical_level_context": historical_level_context,
        "interpretation": interpretation,
        "alternative_explanations": alternative_explanations,
        "evidence_gaps": evidence_gaps,
        "analyst_action": analyst_action,
        "metrics": {
            "revenue_growth_pct": revenue_growth,
            "receivables_growth_pct": receivables_growth,
            "inventory_growth_pct": inventory_growth,
            "receivables_vs_revenue_gap_pp": receivables_gap,
            "inventory_vs_revenue_gap_pp": inventory_gap,
            "operating_cash_flow_growth_pct": operating_cash_flow_growth,
            "dominant_metric": dominant_metric,
            "intensity_series": intensity_series,
        },
    }


def earnings_quality_signal(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    net_income_growth = _growth(
        current.get("net_income"),
        previous.get("net_income"),
    )

    cfo_growth = _growth(
        current.get("operating_cash_flow"),
        previous.get("operating_cash_flow"),
    )

    status = "Normal"

    if net_income_growth is None or cfo_growth is None:
        return _signal(
            area="Earnings Quality",
            status="Review",
            title="Earnings vs operating cash flow",
            interpretation=(
                "There is not enough comparable data to assess earnings "
                "and operating cash-flow growth."
            ),
            action=(
                "Check that comparable net income and operating cash-flow "
                "periods are available."
            ),
            metrics={
                "net_income_growth_pct": net_income_growth,
                "operating_cash_flow_growth_pct": cfo_growth,
            },
        )

    divergence = net_income_growth - cfo_growth

    if net_income_growth > 15 and cfo_growth < 0:
        status = "High Attention"
        interpretation = (
            "Reported earnings are increasing while operating cash flow "
            "is declining."
        )
        action = (
            "Review working-capital movements, non-cash adjustments, "
            "receivables, and revenue-recognition disclosures."
        )
    elif divergence > 25:
        status = "Review"
        interpretation = (
            "Net income growth is materially stronger than operating "
            "cash-flow growth."
        )
        action = (
            "Investigate cash conversion and the main reconciliation "
            "items between earnings and operating cash flow."
        )
    else:
        interpretation = (
            "Earnings growth and operating cash-flow growth appear "
            "reasonably aligned."
        )
        action = (
            "No major earnings-quality divergence detected under the "
            "current heuristic rules."
        )

    return _signal(
        area="Earnings Quality",
        status=status,
        title="Earnings vs operating cash flow",
        interpretation=interpretation,
        action=action,
        metrics={
            "net_income_growth_pct": round(net_income_growth, 2),
            "operating_cash_flow_growth_pct": round(cfo_growth, 2),
            "growth_divergence_pp": round(divergence, 2),
        },
    )



def _build_earnings_quality_diagnostic(
    signals: list[dict[str, Any]],
    derived_financial_metrics: dict[str, Any],
    trend_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build a structured earnings-quality diagnostic using profit growth,
    operating cash-flow growth, cash conversion and multi-year context.

    The diagnostic treats ratios as analytical evidence rather than proof
    of accounting quality or management intent.
    """

    earnings_signal = next(
        (
            signal
            for signal in signals
            if signal.get("area") == "Earnings Quality"
        ),
        None,
    )

    if not earnings_signal:
        return {
            "status": "Unavailable",
            "cash_conversion_posture": "Unavailable",
            "observed_evidence": [],
            "historical_context": {},
            "interpretation": (
                "Earnings-quality diagnostic is unavailable because the "
                "underlying signal was not produced."
            ),
            "evidence_gaps": [],
            "analyst_action": (
                "Confirm that comparable net income and operating cash-flow "
                "data are available."
            ),
            "metrics": {},
        }

    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    signal_metrics = earnings_signal.get("metrics", {})
    derived_metrics = derived_financial_metrics.get("metrics", {})

    net_income_growth = signal_metrics.get("net_income_growth_pct")
    operating_cash_flow_growth = signal_metrics.get(
        "operating_cash_flow_growth_pct"
    )

    growth_gap = (
        net_income_growth - operating_cash_flow_growth
        if is_number(net_income_growth)
        and is_number(operating_cash_flow_growth)
        else None
    )

    net_margin = derived_metrics.get("net_margin_pct")
    operating_cash_flow_margin = derived_metrics.get(
        "operating_cash_flow_margin_pct"
    )
    cash_conversion = derived_metrics.get(
        "cash_conversion_of_earnings"
    )

    margin_gap = (
        net_margin - operating_cash_flow_margin
        if is_number(net_margin)
        and is_number(operating_cash_flow_margin)
        else None
    )

    # ------------------------------------------------------------
    # Latest-period evidence
    # ------------------------------------------------------------
    observed_evidence = []

    if is_number(net_income_growth):
        observed_evidence.append(
            f"Net income grew {net_income_growth:.2f}% in the latest "
            "fiscal year."
        )

    if is_number(operating_cash_flow_growth):
        observed_evidence.append(
            f"Operating cash flow grew "
            f"{operating_cash_flow_growth:.2f}% over the same period."
        )

    if is_number(growth_gap):
        observed_evidence.append(
            "Net income growth exceeded operating cash-flow growth by "
            f"{growth_gap:.2f} percentage points."
        )

    if is_number(cash_conversion):
        observed_evidence.append(
            "Operating cash flow represented "
            f"{cash_conversion:.2f}x reported net income."
        )

    if (
        is_number(net_margin)
        and is_number(operating_cash_flow_margin)
    ):
        observed_evidence.append(
            f"Net margin was {net_margin:.2f}% versus an operating "
            f"cash-flow margin of {operating_cash_flow_margin:.2f}%."
        )

    # ------------------------------------------------------------
    # Multi-year cash-conversion history
    # ------------------------------------------------------------
    metric_trends = (
        (trend_context or {}).get("metric_trends", {})
    )

    net_income_series = (
        metric_trends
        .get("net_income", {})
        .get("series", [])
    )

    cash_flow_series = (
        metric_trends
        .get("operating_cash_flow", {})
        .get("series", [])
    )

    net_income_by_period = {
        str(item.get("period")): item.get("value")
        for item in net_income_series
        if item.get("period") is not None
    }

    cash_conversion_series = []

    for item in cash_flow_series:
        period = str(item.get("period"))
        cash_flow_value = item.get("value")
        net_income_value = net_income_by_period.get(period)

        if (
            is_number(cash_flow_value)
            and is_number(net_income_value)
            and net_income_value > 0
        ):
            cash_conversion_series.append(
                {
                    "period": period,
                    "ratio": round(
                        cash_flow_value / net_income_value,
                        2,
                    ),
                }
            )

    historical_context = {
        "label": "Insufficient history",
        "latest_ratio": (
            round(cash_conversion, 2)
            if is_number(cash_conversion)
            else None
        ),
        "prior_median_ratio": None,
        "prior_min_ratio": None,
        "prior_max_ratio": None,
        "interpretation": (
            "There is not enough aligned history to assess whether the "
            "latest cash-conversion level is unusual."
        ),
    }

    if len(cash_conversion_series) >= 3:
        ratios = [
            item["ratio"]
            for item in cash_conversion_series
        ]

        latest_ratio = ratios[-1]
        prior_ratios = ratios[:-1]
        sorted_prior = sorted(prior_ratios)
        midpoint = len(sorted_prior) // 2

        if len(sorted_prior) % 2 == 0:
            prior_median = (
                sorted_prior[midpoint - 1]
                + sorted_prior[midpoint]
            ) / 2
        else:
            prior_median = sorted_prior[midpoint]

        prior_min = min(prior_ratios)
        prior_max = max(prior_ratios)

        if latest_ratio < prior_min:
            historical_label = "Below historical range"
            historical_interpretation = (
                f"The latest cash-conversion ratio of {latest_ratio:.2f}x "
                f"is below the prior historical range of "
                f"{prior_min:.2f}x to {prior_max:.2f}x."
            )

        elif latest_ratio > prior_max:
            historical_label = "Above historical range"
            historical_interpretation = (
                f"The latest cash-conversion ratio of {latest_ratio:.2f}x "
                f"is above the prior historical range of "
                f"{prior_min:.2f}x to {prior_max:.2f}x."
            )

        else:
            historical_label = "Within historical range"
            historical_interpretation = (
                f"The latest cash-conversion ratio of {latest_ratio:.2f}x "
                f"remains within the prior historical range of "
                f"{prior_min:.2f}x to {prior_max:.2f}x and compares with "
                f"a prior median of {prior_median:.2f}x."
            )

        historical_context = {
            "label": historical_label,
            "latest_ratio": round(latest_ratio, 2),
            "prior_median_ratio": round(prior_median, 2),
            "prior_min_ratio": round(prior_min, 2),
            "prior_max_ratio": round(prior_max, 2),
            "interpretation": historical_interpretation,
        }

    # ------------------------------------------------------------
    # Historical materiality and recent cash-conversion pattern
    # ------------------------------------------------------------
    recent_cash_conversion_pattern = {
        "label": "Insufficient history",
        "interpretation": (
            "There is not enough recent history to assess the direction "
            "of cash conversion."
        ),
    }

    if len(cash_conversion_series) >= 4:
        recent_ratios = [
            item["ratio"]
            for item in cash_conversion_series[-4:]
        ]

        decreasing_steps = sum(
            current_ratio < previous_ratio
            for previous_ratio, current_ratio
            in zip(recent_ratios, recent_ratios[1:])
        )

        increasing_steps = sum(
            current_ratio > previous_ratio
            for previous_ratio, current_ratio
            in zip(recent_ratios, recent_ratios[1:])
        )

        if decreasing_steps == len(recent_ratios) - 1:
            recent_cash_conversion_pattern = {
                "label": "Persistent recent decline",
                "interpretation": (
                    "Cash conversion has declined across each of the "
                    f"latest {len(recent_ratios)} available fiscal periods, "
                    f"from {recent_ratios[0]:.2f}x to "
                    f"{recent_ratios[-1]:.2f}x."
                ),
            }

        elif increasing_steps == len(recent_ratios) - 1:
            recent_cash_conversion_pattern = {
                "label": "Persistent recent improvement",
                "interpretation": (
                    "Cash conversion has improved across each of the "
                    f"latest {len(recent_ratios)} available fiscal periods, "
                    f"from {recent_ratios[0]:.2f}x to "
                    f"{recent_ratios[-1]:.2f}x."
                ),
            }

        else:
            recent_cash_conversion_pattern = {
                "label": "Mixed recent trend",
                "interpretation": (
                    "Cash conversion has shown a mixed pattern across the "
                    f"latest {len(recent_ratios)} available fiscal periods."
                ),
            }

    if historical_context.get("label") == "Below historical range":
        latest_ratio = historical_context.get("latest_ratio")
        prior_min_ratio = historical_context.get("prior_min_ratio")

        if (
            is_number(latest_ratio)
            and is_number(prior_min_ratio)
            and prior_min_ratio > 0
        ):
            shortfall_pct = (
                (prior_min_ratio - latest_ratio)
                / prior_min_ratio
                * 100
            )

            if shortfall_pct < 10:
                historical_context["label"] = (
                    "Slightly below historical range"
                )
                historical_context[
                    "shortfall_vs_prior_min_pct"
                ] = round(shortfall_pct, 2)
                historical_context["interpretation"] = (
                    f"The latest cash-conversion ratio of "
                    f"{latest_ratio:.2f}x is only modestly below the "
                    f"prior historical minimum of "
                    f"{prior_min_ratio:.2f}x, a shortfall of "
                    f"{shortfall_pct:.2f}%."
                )

    # ------------------------------------------------------------
    # Cash-conversion posture
    # ------------------------------------------------------------
    if not is_number(cash_conversion):
        cash_conversion_posture = "Unavailable"
    elif cash_conversion >= 1.0:
        cash_conversion_posture = "Strong"
    elif cash_conversion >= 0.80:
        cash_conversion_posture = "Broadly supportive"
    elif cash_conversion >= 0.60:
        cash_conversion_posture = "Moderate"
    else:
        cash_conversion_posture = "Weak"

    # ------------------------------------------------------------
    # Diagnostic interpretation
    # ------------------------------------------------------------
    if (
        earnings_signal.get("status") == "Normal"
        and is_number(cash_conversion)
        and cash_conversion >= 0.80
        and (
            not is_number(growth_gap)
            or abs(growth_gap) <= 15
        )
    ):
        interpretation = (
            "Earnings growth and operating cash-flow growth remain broadly "
            "aligned, while cash conversion is supportive of reported "
            "earnings. The available evidence does not currently indicate "
            "a material deterioration in earnings quality."
        )

    elif (
        is_number(cash_conversion)
        and cash_conversion < 0.60
    ):
        interpretation = (
            "Operating cash flow is materially below reported net income, "
            "which weakens cash support for earnings and warrants further "
            "review of working-capital movements and non-cash adjustments."
        )

    elif (
        is_number(growth_gap)
        and growth_gap > 25
    ):
        interpretation = (
            "Net income growth is materially stronger than operating "
            "cash-flow growth. The divergence warrants review even though "
            "a single period does not establish poor earnings quality."
        )

    else:
        interpretation = (
            "The available evidence presents a mixed earnings-quality "
            "picture. Profitability and cash-generation metrics should be "
            "reviewed together before drawing a stronger conclusion."
        )

    historical_label = historical_context.get("label")

    if historical_label == "Within historical range":
        interpretation += (
            " The latest cash-conversion level also remains within the "
            "company's prior historical range, which reduces the evidence "
            "of an unusual structural deterioration."
        )

    elif historical_label == "Slightly below historical range":
        interpretation += (
            " The latest cash-conversion level is only modestly below the "
            "company's prior historical range, which does not by itself "
            "represent a material break from historical experience."
        )

    elif historical_label == "Below historical range":
        interpretation += (
            " The latest cash-conversion level is materially below the "
            "company's prior historical range, strengthening the case for "
            "further analyst review."
        )

    elif historical_label == "Above historical range":
        interpretation += (
            " The latest cash-conversion level is above the company's "
            "prior historical range, providing stronger cash support for "
            "reported earnings in the current period."
        )

    recent_pattern_label = recent_cash_conversion_pattern.get("label")

    if recent_pattern_label == "Persistent recent decline":
        interpretation += (
            " However, cash conversion has declined across several "
            "consecutive fiscal periods, so the direction of travel "
            "warrants continued monitoring."
        )

    elif recent_pattern_label == "Persistent recent improvement":
        interpretation += (
            " The recent multi-year direction is improving, which provides "
            "additional support for the current earnings-quality assessment."
        )

    # ------------------------------------------------------------
    # Evidence gaps and analyst action
    # ------------------------------------------------------------
    evidence_gaps = [
        (
            "Working-capital reconciliation explaining the bridge between "
            "net income and operating cash flow."
        ),
        (
            "Material non-cash adjustments such as depreciation, "
            "stock-based compensation and deferred taxes."
        ),
        (
            "One-off timing effects or other operating cash-flow "
            "reclassification items."
        ),
    ]

    if is_number(margin_gap) and margin_gap > 10:
        evidence_gaps.append(
            "Drivers of the gap between net margin and operating "
            "cash-flow margin."
        )

    if cash_conversion_posture in {"Weak", "Moderate"}:
        analyst_action = (
            "Review the operating cash-flow reconciliation, working-capital "
            "movements and material non-cash adjustments. Escalate if cash "
            "conversion remains weak or the profit-to-cash-flow divergence "
            "widens across future periods."
        )
    else:
        analyst_action = (
            "Continue monitoring cash conversion, working-capital movements "
            "and non-cash adjustments. Escalate only if operating cash flow "
            "begins to materially lag reported earnings."
        )

    return {
        "status": earnings_signal.get("status"),
        "cash_conversion_posture": cash_conversion_posture,
        "observed_evidence": observed_evidence,
        "historical_context": historical_context,
        "recent_cash_conversion_pattern": recent_cash_conversion_pattern,
        "interpretation": interpretation,
        "evidence_gaps": evidence_gaps,
        "analyst_action": analyst_action,
        "metrics": {
            "net_income_growth_pct": net_income_growth,
            "operating_cash_flow_growth_pct": operating_cash_flow_growth,
            "growth_divergence_pp": (
                round(growth_gap, 2)
                if is_number(growth_gap)
                else None
            ),
            "net_margin_pct": net_margin,
            "operating_cash_flow_margin_pct": (
                operating_cash_flow_margin
            ),
            "margin_gap_pp": (
                round(margin_gap, 2)
                if is_number(margin_gap)
                else None
            ),
            "cash_conversion_of_earnings": cash_conversion,
            "cash_conversion_series": cash_conversion_series,
        },
    }


def liquidity_leverage_signal(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """
    Assess latest-period liquidity and leverage using a combination of
    balance-sheet ratios and period-on-period financing trends.

    Thresholds are generic screening heuristics for non-financial
    corporates. They are not industry-specific covenant thresholds.
    """

    current_ratio = _ratio(
        current.get("current_assets"),
        current.get("current_liabilities"),
    )

    debt_to_assets = _ratio(
        current.get("total_debt"),
        current.get("total_assets"),
    )

    debt_to_equity = _ratio(
        current.get("total_debt"),
        current.get("total_equity"),
    )

    cash_to_debt = _ratio(
        current.get("cash_and_equivalents"),
        current.get("total_debt"),
    )

    debt_growth = _growth(
        current.get("total_debt"),
        previous.get("total_debt"),
    )

    cash_growth = _growth(
        current.get("cash_and_equivalents"),
        previous.get("cash_and_equivalents"),
    )

    review_reasons = []
    high_attention_reasons = []

    # ------------------------------------------------------------
    # Liquidity screening
    # ------------------------------------------------------------
    if current_ratio is not None:
        if current_ratio < 0.75:
            high_attention_reasons.append(
                "Current assets are materially below current liabilities."
            )
        elif current_ratio < 1.0:
            review_reasons.append(
                "Current assets are below current liabilities."
            )

    if cash_to_debt is not None:
        if cash_to_debt < 0.10:
            high_attention_reasons.append(
                "Cash coverage of debt is very limited."
            )
        elif cash_to_debt < 0.25:
            review_reasons.append(
                "Cash coverage of debt is relatively limited."
            )

    # ------------------------------------------------------------
    # Leverage screening
    # ------------------------------------------------------------
    if debt_to_assets is not None:
        if debt_to_assets > 0.75:
            high_attention_reasons.append(
                "Debt represents a very high share of total assets."
            )
        elif debt_to_assets > 0.60:
            review_reasons.append(
                "Debt represents a high share of total assets."
            )

    total_equity = current.get("total_equity")

    if (
        isinstance(total_equity, (int, float))
        and total_equity <= 0
    ):
        high_attention_reasons.append(
            "Reported equity is non-positive, which weakens leverage coverage."
        )
    elif debt_to_equity is not None:
        if debt_to_equity > 3.0:
            high_attention_reasons.append(
                "Debt is more than three times reported equity."
            )
        elif debt_to_equity > 2.0:
            review_reasons.append(
                "Debt is more than twice reported equity."
            )

    # ------------------------------------------------------------
    # Change-based financing signals
    # ------------------------------------------------------------
    if debt_growth is not None and debt_growth > 25:
        review_reasons.append(
            "Debt has increased materially compared with the previous period."
        )

    if cash_growth is not None and cash_growth < -20:
        review_reasons.append(
            "Cash and cash equivalents have declined materially."
        )

    # ------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------
    if high_attention_reasons:
        status = "High Attention"

    elif review_reasons:
        status = "Review"

    else:
        status = "Normal"

    # ------------------------------------------------------------
    # Analyst interpretation
    # ------------------------------------------------------------
    if status == "Normal":
        evidence_parts = []

        if current_ratio is not None:
            evidence_parts.append(
                f"The current ratio is {current_ratio:.2f}x."
            )

        if debt_to_assets is not None:
            evidence_parts.append(
                f"Debt represents {debt_to_assets * 100:.1f}% of total assets."
            )

        if debt_to_equity is not None:
            evidence_parts.append(
                f"Debt-to-equity is {debt_to_equity:.2f}x."
            )

        if cash_to_debt is not None:
            evidence_parts.append(
                f"Cash covers total debt at {cash_to_debt:.2f}x."
            )

        evidence_parts.append(
            "No major liquidity or leverage pressure was detected "
            "under the current screening rules."
        )

        interpretation = " ".join(evidence_parts)

        action = (
            "Continue monitoring liquidity, debt, cash coverage, "
            "and financing trends."
        )

    else:
        interpretation = " ".join(
            high_attention_reasons + review_reasons
        )

        if status == "High Attention":
            action = (
                "Prioritise review of debt maturity, liquidity facilities, "
                "cash generation, covenant headroom, and financing commentary."
            )
        else:
            action = (
                "Review debt maturity, liquidity facilities, current liabilities, "
                "cash generation, and financing commentary."
            )

    return _signal(
        area="Liquidity & Leverage",
        status=status,
        title="Liquidity and financing pressure",
        interpretation=interpretation,
        action=action,
        metrics={
            "current_ratio": (
                round(current_ratio, 2)
                if current_ratio is not None
                else None
            ),
            "debt_to_assets_pct": (
                round(debt_to_assets * 100, 2)
                if debt_to_assets is not None
                else None
            ),
            "debt_to_equity": (
                round(debt_to_equity, 2)
                if debt_to_equity is not None
                else None
            ),
            "cash_to_debt": (
                round(cash_to_debt, 2)
                if cash_to_debt is not None
                else None
            ),
            "debt_growth_pct": (
                round(debt_growth, 2)
                if debt_growth is not None
                else None
            ),
            "cash_growth_pct": (
                round(cash_growth, 2)
                if cash_growth is not None
                else None
            ),
        },
    )

def _build_executive_assessment(
    signals: list[dict[str, Any]],
    trend_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a concise, deterministic analyst-level executive assessment."""

    severity_rank = {
        "Normal": 0,
        "Review": 1,
        "High Attention": 2,
    }

    trend_summary = None

    if trend_context:
        core_metrics = [
            ("revenue", "Revenue"),
            ("net_income", "Net Income"),
            ("operating_cash_flow", "Operating Cash Flow"),
        ]

        metric_trends = trend_context.get("metric_trends", {})

        yoy_parts = []
        accelerating = []
        moderating = []

        for metric_key, display_name in core_metrics:
            metric_trend = metric_trends.get(metric_key, {})

            latest_yoy = metric_trend.get("latest_yoy_pct")
            recent_cagr = metric_trend.get("recent_cagr_pct")
            full_cagr = metric_trend.get("full_history_cagr_pct")

            if isinstance(latest_yoy, (int, float)):
                yoy_parts.append(
                    f"{display_name} {latest_yoy:.2f}%"
                )

            if (
                isinstance(recent_cagr, (int, float))
                and isinstance(full_cagr, (int, float))
            ):
                trend_gap = recent_cagr - full_cagr

                if trend_gap >= 10:
                    accelerating.append(display_name)
                elif trend_gap <= -10:
                    moderating.append(display_name)

        trend_parts = []

        if yoy_parts:
            trend_parts.append(
                "Latest YoY growth across the core metrics is: "
                + ", ".join(yoy_parts)
                + "."
            )

        if len(accelerating) == len(core_metrics):
            trend_parts.append(
                "Recent 3-year CAGR is above the full-period CAGR across all three "
                "core metrics, indicating broad-based acceleration relative "
                "to the longer-term history."
            )
        elif accelerating:
            trend_parts.append(
                "Recent growth is running above the longer-term trend for "
                + ", ".join(accelerating)
                + "."
            )

        if moderating:
            trend_parts.append(
                "Recent growth is below the longer-term trend for "
                + ", ".join(moderating)
                + ", indicating some moderation."
            )

        if trend_parts:
            period_count = trend_context.get("period_count", 0)

            if period_count:
                trend_summary = (
                    f"Across the available {period_count}-fiscal-year history, "
                    + " ".join(trend_parts)
                )
            else:
                trend_summary = " ".join(trend_parts)

    flagged = [
        signal
        for signal in signals
        if signal.get("status") != "Normal"
    ]

    normal = [
        signal
        for signal in signals
        if signal.get("status") == "Normal"
    ]

    primary = (
        max(
            flagged,
            key=lambda signal: severity_rank.get(
                signal.get("status"),
                0,
            ),
        )
        if flagged
        else None
    )

    supporting_evidence = " ".join(
        signal.get("interpretation", "").strip()
        for signal in normal[:3]
        if signal.get("interpretation")
    )

    if not supporting_evidence:
        supporting_evidence = (
            "No additional positive evidence was identified under the "
            "current analytical rules."
        )

    if primary is None:
        return {
            "overall_finding": (
                "No material financial watchpoints were identified across "
                "the monitored areas under the current analytical rules."
            ),
            "primary_review_point": None,
            "trend_summary": trend_summary,
            "supporting_evidence": supporting_evidence,
            "follow_up_analysis": None,
            "evidence_gap": (
                "This rule-based review is limited to the standardized "
                "metrics available in the dataset and does not replace "
                "filing-note or management-commentary review."
            ),
            "recommended_next_step": (
                "No immediate escalation is indicated. Continue trend "
                "monitoring and refresh the analysis when the next reporting "
                "period becomes available."
            ),
        }

    area = primary.get("area", "Financial signal")

    interpretation = primary.get(
        "interpretation",
        "A financial signal requires analyst review.",
    )

    metrics = primary.get("metrics", {})

    other_flagged = max(len(flagged) - 1, 0)
    stable_count = len(normal)

    overall_parts = [
        f"The principal watchpoint is {area.lower()}: {interpretation}"
    ]

    if stable_count:
        overall_parts.append(
            f"{stable_count} of the other monitored areas remain within "
            "normal ranges under the current analytical rules."
        )

    if other_flagged:
        overall_parts.append(
            f"{other_flagged} additional monitored area"
            f"{' also requires' if other_flagged == 1 else 's also require'} "
            "review."
        )

    overall_finding = " ".join(overall_parts)

    def metric_value(*names):
        for name in names:
            value = metrics.get(name)

            if isinstance(value, (int, float)):
                return float(value)

        return None

    follow_up_analysis = interpretation

    evidence_gap = (
        "The current standardized dataset does not contain enough detail to "
        "determine the underlying operating driver of this signal."
    )

    recommended_next_step = primary.get(
        "analyst_action",
        "Review the underlying filing detail before relying on this signal "
        "for a downstream decision.",
    )

    # --------------------------------------------------------
    # Working Capital
    # --------------------------------------------------------

    if area == "Working Capital":
        revenue_growth = metric_value(
            "revenue_growth_pct",
        )

        inventory_growth = metric_value(
            "inventory_growth_pct",
        )

        receivables_growth = metric_value(
            "accounts_receivable_growth_pct",
            "receivables_growth_pct",
        )

        follow_up_parts = []

        if (
            revenue_growth is not None
            and inventory_growth is not None
        ):
            follow_up_parts.append(
                "Inventory growth exceeded revenue growth by "
                f"{inventory_growth - revenue_growth:.2f} percentage points."
            )

        if (
            revenue_growth is not None
            and receivables_growth is not None
            and receivables_growth > revenue_growth
        ):
            follow_up_parts.append(
                "Receivables also grew faster than revenue by "
                f"{receivables_growth - revenue_growth:.2f} percentage points."
            )

        if follow_up_parts:
            follow_up_analysis = (
                " ".join(follow_up_parts)
                + " The divergence is confirmed by the current data, but its "
                "commercial cause cannot be established from these metrics alone."
            )

        evidence_gap = (
            "The standardized dataset does not include inventory turnover, "
            "cost of revenue, ageing detail, or management commentary, so it "
            "cannot distinguish planned inventory build from slower conversion."
        )

        recommended_next_step = (
            "Review inventory turnover, cost of revenue and gross-margin "
            "movement alongside management commentary. Escalate only if the "
            "inventory build is persistent, margin-dilutive, or unsupported "
            "by demand indicators."
        )

    # --------------------------------------------------------
    # Accounting Integrity
    # --------------------------------------------------------

    elif area == "Accounting Integrity":
        difference_pct = metric_value(
            "difference_pct",
            "balance_sheet_difference_pct",
        )

        if difference_pct is not None:
            follow_up_analysis = (
                "The balance-sheet reconciliation difference is "
                f"{difference_pct:.2f}% under the current mapping."
            )

        evidence_gap = (
            "This check does not validate footnote classifications, "
            "off-balance-sheet commitments, lease detail, or filing-specific "
            "taxonomy judgements."
        )

        recommended_next_step = (
            "Trace any reconciliation exception back to the source filing, "
            "confirm the mapped XBRL concepts, and review material footnote "
            "classifications before escalation."
        )

    # --------------------------------------------------------
    # Earnings Quality
    # --------------------------------------------------------

    elif area == "Earnings Quality":
        net_income_growth = metric_value(
            "net_income_growth_pct",
        )

        cash_flow_growth = metric_value(
            "operating_cash_flow_growth_pct",
        )

        divergence = metric_value(
            "growth_divergence_pp",
        )

        if divergence is not None:
            follow_up_analysis = (
                "Net-income and operating-cash-flow growth differ by "
                f"{divergence:.2f} percentage points."
            )

        elif (
            net_income_growth is not None
            and cash_flow_growth is not None
        ):
            follow_up_analysis = (
                "Net income grew "
                f"{net_income_growth:.2f}% versus operating cash flow at "
                f"{cash_flow_growth:.2f}%."
            )

        evidence_gap = (
            "The current view does not decompose accruals, non-cash items, "
            "one-off gains or losses, or working-capital contributions to "
            "cash flow."
        )

        recommended_next_step = (
            "Review the cash-flow bridge, accrual movements, non-cash items "
            "and material one-offs to determine whether the earnings/cash "
            "divergence is temporary or persistent."
        )

    # --------------------------------------------------------
    # Liquidity & Leverage
    # --------------------------------------------------------

    elif area == "Liquidity & Leverage":
        current_ratio = metric_value(
            "current_ratio",
        )

        debt_growth = metric_value(
            "debt_growth_pct",
        )

        cash_growth = metric_value(
            "cash_growth_pct",
        )

        follow_up_parts = []

        if current_ratio is not None:
            follow_up_parts.append(
                f"The current ratio is {current_ratio:.2f}x."
            )

        if debt_growth is not None:
            follow_up_parts.append(
                f"Debt changed by {debt_growth:.2f}%."
            )

        if cash_growth is not None:
            follow_up_parts.append(
                f"Cash changed by {cash_growth:.2f}%."
            )

        if follow_up_parts:
            follow_up_analysis = " ".join(
                follow_up_parts
            )

        evidence_gap = (
            "The standardized dataset does not include debt maturity profile, "
            "interest coverage, covenant headroom, committed facilities, or "
            "near-term contractual cash requirements."
        )

        recommended_next_step = (
            "Review debt maturities, interest coverage, available liquidity "
            "and covenant headroom before drawing a firm conclusion on "
            "financing risk."
        )

    return {
        "overall_finding": overall_finding,
        "primary_review_point": interpretation,
        "trend_summary": trend_summary,
        "supporting_evidence": supporting_evidence,
        "follow_up_analysis": follow_up_analysis,
        "evidence_gap": evidence_gap,
        "recommended_next_step": recommended_next_step,
    }



def _build_multi_year_trend_context(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build analyst-oriented multi-year trend statistics from annual history.

    Missing observations are ignored rather than treated as zero.
    CAGR uses the actual fiscal-year span between valid observations.
    """

    metrics = [
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

    ordered_history = sorted(
        history,
        key=lambda row: str(row.get("period", "")),
    )

    def period_year(period: str) -> int | None:
        try:
            return int(str(period)[:4])
        except (TypeError, ValueError):
            return None

    def growth_pct(current_value, previous_value):
        current_number = _number(current_value)
        previous_number = _number(previous_value)

        if (
            current_number is None
            or previous_number is None
            or previous_number == 0
        ):
            return None

        return (
            (current_number - previous_number)
            / abs(previous_number)
            * 100
        )

    metric_trends: dict[str, Any] = {}

    for metric in metrics:
        observations = []

        for row in ordered_history:
            value = _number(row.get(metric))
            period = str(row.get("period", ""))

            if value is not None and period:
                observations.append(
                    {
                        "period": period,
                        "year": period_year(period),
                        "value": value,
                    }
                )

        trend = {
            "observations": len(observations),
            "first_period": None,
            "last_period": None,
            "latest_value": None,
            "latest_yoy_pct": None,
            "recent_cagr_pct": None,
            "full_history_cagr_pct": None,
            "series": observations,
        }

        if observations:
            trend["first_period"] = observations[0]["period"]
            trend["last_period"] = observations[-1]["period"]
            trend["latest_value"] = observations[-1]["value"]

        if len(observations) >= 2:
            latest = observations[-1]
            previous = observations[-2]

            latest_year = latest["year"]
            previous_year = previous["year"]

            # Only call it YoY when the observations are consecutive fiscal years.
            if (
                latest_year is not None
                and previous_year is not None
                and latest_year - previous_year == 1
            ):
                latest_yoy = growth_pct(
                    latest["value"],
                    previous["value"],
                )

                trend["latest_yoy_pct"] = (
                    round(latest_yoy, 2)
                    if latest_yoy is not None
                    else None
                )

            first = observations[0]
            first_year = first["year"]

            if (
                first_year is not None
                and latest_year is not None
                and latest_year > first_year
                and first["value"] > 0
                and latest["value"] > 0
            ):
                year_span = latest_year - first_year

                full_cagr = (
                    (latest["value"] / first["value"])
                    ** (1 / year_span)
                    - 1
                ) * 100

                trend["full_history_cagr_pct"] = round(
                    full_cagr,
                    2,
                )

            if latest_year is not None:
                recent_candidates = [
                    obs
                    for obs in observations
                    if (
                        obs["year"] is not None
                        and obs["year"] >= latest_year - 3
                    )
                ]

                if len(recent_candidates) >= 2:
                    recent_first = recent_candidates[0]
                    recent_last = recent_candidates[-1]

                    recent_span = (
                        recent_last["year"]
                        - recent_first["year"]
                    )

                    if (
                        recent_span > 0
                        and recent_first["value"] > 0
                        and recent_last["value"] > 0
                    ):
                        recent_cagr = (
                            (
                                recent_last["value"]
                                / recent_first["value"]
                            )
                            ** (1 / recent_span)
                            - 1
                        ) * 100

                        trend["recent_cagr_pct"] = round(
                            recent_cagr,
                            2,
                        )

        metric_trends[metric] = trend

    periods = [
        str(row.get("period"))
        for row in ordered_history
        if row.get("period")
    ]

    return {
        "period_count": len(ordered_history),
        "start_period": periods[0] if periods else None,
        "end_period": periods[-1] if periods else None,
        "metric_trends": metric_trends,
    }



def _build_next_fiscal_year_outlook(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a deterministic one-year-ahead financial outlook.

    This is a trend-based analytical scenario, not company guidance,
    an analyst price target, or investment advice.

    The base growth assumption gives greater weight to the recent trend
    while retaining the longer-term history as an anchor.
    """

    trend_context = _build_multi_year_trend_context(history)

    core_metrics = [
        ("revenue", "Revenue"),
        ("net_income", "Net Income"),
        ("operating_cash_flow", "Operating Cash Flow"),
    ]

    items = []

    for metric, display_name in core_metrics:
        trend = trend_context.get("metric_trends", {}).get(metric, {})

        latest_value = _number(trend.get("latest_value"))
        recent_growth = _number(trend.get("recent_cagr_pct"))
        long_term_growth = _number(trend.get("full_history_cagr_pct"))

        observations = int(
            trend.get("observations") or 0
        )

        if (
            latest_value is None
            or latest_value <= 0
            or observations < 3
        ):
            items.append(
                {
                    "metric": metric,
                    "display_name": display_name,
                    "status": "Insufficient history",
                    "latest_value": latest_value,
                    "base_growth_pct": None,
                    "projected_value": None,
                    "lower_value": None,
                    "upper_value": None,
                    "recent_growth_pct": recent_growth,
                    "long_term_growth_pct": long_term_growth,
                    "observations": observations,
                }
            )
            continue

        if recent_growth is not None and long_term_growth is not None:
            blended_growth = (
                0.35 * recent_growth
                + 0.65 * long_term_growth
            )

            # Moderate extraordinary historical growth before using it
            # as a one-year analytical scenario.
            base_growth = 0.50 * blended_growth

            uncertainty = max(
                5.0,
                0.15 * abs(recent_growth - long_term_growth),
            )

            lower_growth = base_growth - uncertainty
            upper_growth = base_growth + uncertainty

        elif recent_growth is not None:
            base_growth = 0.50 * recent_growth
            uncertainty = max(
                5.0,
                0.15 * abs(base_growth),
            )
            lower_growth = base_growth - uncertainty
            upper_growth = base_growth + uncertainty

        elif long_term_growth is not None:
            base_growth = 0.50 * long_term_growth
            uncertainty = max(
                5.0,
                0.15 * abs(base_growth),
            )
            lower_growth = base_growth - uncertainty
            upper_growth = base_growth + uncertainty

        else:
            items.append(
                {
                    "metric": metric,
                    "display_name": display_name,
                    "status": "Insufficient trend data",
                    "latest_value": latest_value,
                    "base_growth_pct": None,
                    "projected_value": None,
                    "lower_value": None,
                    "upper_value": None,
                    "recent_growth_pct": None,
                    "long_term_growth_pct": None,
                    "observations": observations,
                }
            )
            continue

        projected_value = max(
            latest_value * (1 + base_growth / 100),
            0.0,
        )

        lower_value = max(
            latest_value * (1 + lower_growth / 100),
            0.0,
        )

        upper_value = max(
            latest_value * (1 + upper_growth / 100),
            0.0,
        )

        if base_growth > 5:
            direction = "Expansion"
        elif base_growth < -5:
            direction = "Contraction"
        else:
            direction = "Broadly stable"

        items.append(
            {
                "metric": metric,
                "display_name": display_name,
                "status": "Available",
                "direction": direction,
                "latest_value": round(latest_value, 2),
                "base_growth_pct": round(base_growth, 2),
                "projected_value": round(projected_value, 2),
                "lower_value": round(lower_value, 2),
                "upper_value": round(upper_value, 2),
                "recent_growth_pct": (
                    round(recent_growth, 2)
                    if recent_growth is not None
                    else None
                ),
                "long_term_growth_pct": (
                    round(long_term_growth, 2)
                    if long_term_growth is not None
                    else None
                ),
                "observations": observations,
            }
        )

    return {
        "label": "Next Fiscal Year Outlook",
        "period_count": trend_context.get("period_count", 0),
        "history_start": trend_context.get("start_period"),
        "history_end": trend_context.get("end_period"),
        "methodology": (
            "Moderated trend scenario using 35% recent 3-year CAGR and "
            "65% full-history CAGR, followed by a 50% growth dampening "
            "factor to reduce extrapolation risk."
        ),
        "items": items,
        "disclaimer": (
            "This outlook is an analytical trend scenario based on "
            "historical financial statements. It is not company guidance, "
            "a valuation target, or investment advice."
        ),
    }



def _build_forward_looking_executive_conclusion(
    executive_assessment: dict[str, Any],
    trend_context: dict[str, Any],
    outlook: dict[str, Any],
) -> str:
    """
    Synthesize multi-year momentum, current financial signals and the
    moderated next-fiscal-year scenario into one executive conclusion.

    This is analytical scenario synthesis, not company guidance or
    an investment forecast.
    """

    core_metrics = [
        ("revenue", "Revenue"),
        ("net_income", "Net Income"),
        ("operating_cash_flow", "Operating Cash Flow"),
    ]

    metric_trends = trend_context.get("metric_trends", {})

    accelerating = []
    moderating = []

    for metric_key, display_name in core_metrics:
        trend = metric_trends.get(metric_key, {})

        recent_cagr = trend.get("recent_cagr_pct")
        full_cagr = trend.get("full_history_cagr_pct")

        if not isinstance(recent_cagr, (int, float)):
            continue

        if not isinstance(full_cagr, (int, float)):
            continue

        gap = recent_cagr - full_cagr

        if gap >= 10:
            accelerating.append(display_name)
        elif gap <= -10:
            moderating.append(display_name)

    parts = []

    # --------------------------------------------------------
    # Historical momentum
    # --------------------------------------------------------

    if len(accelerating) == len(core_metrics):
        parts.append(
            "Recent 3-year growth is running above the full-period trend "
            "across Revenue, Net Income and Operating Cash Flow, indicating "
            "broad-based acceleration in the historical record."
        )

    elif accelerating:
        parts.append(
            "Recent 3-year growth is running above the longer-term trend for "
            + ", ".join(accelerating)
            + "."
        )

    elif moderating:
        parts.append(
            "Recent 3-year growth is running below the longer-term trend for "
            + ", ".join(moderating)
            + ", indicating some moderation in recent momentum."
        )

    # --------------------------------------------------------
    # Forward scenario
    # --------------------------------------------------------

    available_items = [
        item
        for item in outlook.get("items", [])
        if item.get("status") == "Available"
    ]

    if available_items:
        directions = [
            item.get("direction")
            for item in available_items
        ]

        names = [
            item.get("display_name", item.get("metric", "Metric"))
            for item in available_items
        ]

        if all(direction == "Expansion" for direction in directions):
            if len(names) == 3:
                parts.append(
                    "The moderated historical scenario therefore points to "
                    "continued expansion across all three core metrics in the "
                    "next fiscal year."
                )
            else:
                parts.append(
                    "The moderated historical scenario points to continued "
                    "expansion across "
                    + ", ".join(names)
                    + " in the next fiscal year."
                )

        elif all(
            direction == "Broadly stable"
            for direction in directions
        ):
            parts.append(
                "The moderated historical scenario points to broadly stable "
                "performance across the available core metrics in the next "
                "fiscal year."
            )

        else:
            expanding = [
                item.get("display_name", item.get("metric", "Metric"))
                for item in available_items
                if item.get("direction") == "Expansion"
            ]

            contracting = [
                item.get("display_name", item.get("metric", "Metric"))
                for item in available_items
                if item.get("direction") == "Contraction"
            ]

            if expanding:
                parts.append(
                    "The moderated historical scenario points to expansion in "
                    + ", ".join(expanding)
                    + "."
                )

            if contracting:
                parts.append(
                    "At the same time, the scenario indicates contraction in "
                    + ", ".join(contracting)
                    + "."
                )

    # --------------------------------------------------------
    # Current watchpoint
    # --------------------------------------------------------

    primary_review_point = executive_assessment.get(
        "primary_review_point"
    )

    if primary_review_point:
        clean_watchpoint = str(primary_review_point).strip().rstrip(".")

        parts.append(
            "However, the principal near-term watchpoint remains: "
            + clean_watchpoint
            + "."
        )

        if executive_assessment.get("evidence_gap"):
            parts.append(
                "Confidence in the scenario should therefore remain "
                "conditional on the recommended follow-up review and "
                "additional operating evidence."
            )

    else:
        parts.append(
            "No material financial watchpoint is currently identified under "
            "the analytical rules, although continued monitoring remains "
            "appropriate."
        )

    # ------------------------------------------------------------
    # Balance-sheet context
    # ------------------------------------------------------------
    balance_sheet_summary = executive_assessment.get(
        "balance_sheet_summary"
    )

    if balance_sheet_summary:
        balance_status = balance_sheet_summary.get("status")
        balance_interpretation = balance_sheet_summary.get(
            "interpretation"
        )

        ratio_parts = []

        current_ratio = balance_sheet_summary.get("current_ratio")
        if isinstance(current_ratio, (int, float)):
            ratio_parts.append(
                f"a {current_ratio:.2f}x current ratio"
            )

        debt_to_assets_pct = balance_sheet_summary.get(
            "debt_to_assets_pct"
        )
        if isinstance(debt_to_assets_pct, (int, float)):
            ratio_parts.append(
                f"debt equal to {debt_to_assets_pct:.1f}% of assets"
            )

        debt_to_equity = balance_sheet_summary.get(
            "debt_to_equity"
        )
        if isinstance(debt_to_equity, (int, float)):
            ratio_parts.append(
                f"debt-to-equity of {debt_to_equity:.2f}x"
            )

        cash_to_debt = balance_sheet_summary.get(
            "cash_to_debt"
        )
        if isinstance(cash_to_debt, (int, float)):
            ratio_parts.append(
                f"cash-to-debt of {cash_to_debt:.2f}x"
            )

        if balance_status == "Normal":
            if ratio_parts:
                if len(ratio_parts) == 1:
                    ratio_text = ratio_parts[0]
                else:
                    ratio_text = (
                        ", ".join(ratio_parts[:-1])
                        + ", and "
                        + ratio_parts[-1]
                    )

                parts.append(
                    "Liquidity and leverage screening remains normal, "
                    f"supported by {ratio_text}."
                )
            else:
                parts.append(
                    "Liquidity and leverage screening remains normal "
                    "under the current analytical rules."
                )

        elif balance_status == "Review":
            sentence = (
                "Liquidity and leverage screening requires review."
            )
            if balance_interpretation:
                sentence += f" {balance_interpretation}"
            parts.append(sentence)

        elif balance_status == "High Attention":
            sentence = (
                "Liquidity and leverage screening requires high attention."
            )
            if balance_interpretation:
                sentence += f" {balance_interpretation}"
            parts.append(sentence)

    parts.append(
        "This conclusion is a scenario-based synthesis of historical "
        "financial data rather than company guidance or an investment forecast."
    )

    return " ".join(parts)


def analyze_financial_intelligence(
    current: dict[str, Any],
    previous: dict[str, Any],
    trend_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the first Financial Intelligence rule engine.

    The rules identify analyst-review signals; they do not claim fraud,
    manipulation, or accounting misconduct.
    """

    derived_financial_metrics = _build_derived_financial_metrics(
        current,
        previous,
    )

    signals = [
        accounting_integrity_signal(current),
        working_capital_signal(current, previous),
        earnings_quality_signal(current, previous),
        liquidity_leverage_signal(current, previous),
    ]

    status_counts = {
        "Normal": sum(s["status"] == "Normal" for s in signals),
        "Review": sum(s["status"] == "Review" for s in signals),
        "High Attention": sum(
            s["status"] == "High Attention"
            for s in signals
        ),
    }

    if status_counts["High Attention"] > 0:
        decision = "HIGH ATTENTION"
        summary = (
            "One or more material financial signals require analyst review "
            "before relying on the data for downstream decisions."
        )
    elif status_counts["Review"] > 0:
        decision = "REVIEW REQUIRED"
        summary = (
            "The financial data is broadly usable, but one or more signals "
            "should be investigated before drawing strong conclusions."
        )
    else:
        decision = "PROCEED"
        summary = (
            "No material issues were identified by the current financial "
            "integrity and analytical rules."
        )

    priorities = [
        {
            "area": signal["area"],
            "status": signal["status"],
            "action": signal["analyst_action"],
        }
        for signal in signals
        if signal["status"] != "Normal"
    ]

    working_capital_diagnostic = _build_working_capital_diagnostic(
        signals,
        trend_context,
    )

    earnings_quality_diagnostic = _build_earnings_quality_diagnostic(
        signals,
        derived_financial_metrics,
        trend_context,
    )

    executive_assessment = _build_executive_assessment(
        signals,
        trend_context=trend_context,
    )

    liquidity_signal = next(
        (
            signal
            for signal in signals
            if signal.get("area") == "Liquidity & Leverage"
        ),
        None,
    )

    balance_sheet_summary = None

    if liquidity_signal:
        liquidity_metrics = liquidity_signal.get("metrics", {})

        balance_sheet_summary = {
            "status": liquidity_signal.get("status"),
            "current_ratio": liquidity_metrics.get("current_ratio"),
            "debt_to_assets_pct": liquidity_metrics.get("debt_to_assets_pct"),
            "debt_to_equity": liquidity_metrics.get("debt_to_equity"),
            "cash_to_debt": liquidity_metrics.get("cash_to_debt"),
            "debt_growth_pct": liquidity_metrics.get("debt_growth_pct"),
            "cash_growth_pct": liquidity_metrics.get("cash_growth_pct"),
            "interpretation": liquidity_signal.get("interpretation"),
        }

    executive_assessment["balance_sheet_summary"] = balance_sheet_summary

    if balance_sheet_summary:
        balance_status = balance_sheet_summary.get("status")
        balance_interpretation = balance_sheet_summary.get("interpretation")

        ratio_parts = []

        current_ratio = balance_sheet_summary.get("current_ratio")
        if isinstance(current_ratio, (int, float)):
            ratio_parts.append(
                f"a {current_ratio:.2f}x current ratio"
            )

        debt_to_assets_pct = balance_sheet_summary.get("debt_to_assets_pct")
        if isinstance(debt_to_assets_pct, (int, float)):
            ratio_parts.append(
                f"debt equal to {debt_to_assets_pct:.1f}% of assets"
            )

        debt_to_equity = balance_sheet_summary.get("debt_to_equity")
        if isinstance(debt_to_equity, (int, float)):
            ratio_parts.append(
                f"debt-to-equity of {debt_to_equity:.2f}x"
            )

        cash_to_debt = balance_sheet_summary.get("cash_to_debt")
        if isinstance(cash_to_debt, (int, float)):
            ratio_parts.append(
                f"cash-to-debt of {cash_to_debt:.2f}x"
            )

        if balance_status == "Normal":
            if ratio_parts:
                balance_sheet_sentence = (
                    "Liquidity and leverage screening remains normal, supported by "
                    + ", ".join(ratio_parts[:-1])
                    + (
                        f", and {ratio_parts[-1]}."
                        if len(ratio_parts) > 1
                        else f"{ratio_parts[-1]}."
                    )
                )
            else:
                balance_sheet_sentence = (
                    "Liquidity and leverage screening remains normal under the "
                    "current analytical rules."
                )
        elif balance_status in {"Review", "High Attention"}:
            balance_sheet_sentence = (
                f"Liquidity and leverage screening is {balance_status.lower()}"
            )

            if balance_interpretation:
                balance_sheet_sentence += (
                    f": {balance_interpretation}"
                )
            else:
                balance_sheet_sentence += (
                    " and warrants additional analyst review."
                )
        else:
            balance_sheet_sentence = None

        existing_conclusion = executive_assessment.get(
            "forward_looking_conclusion"
        )

        if existing_conclusion and balance_sheet_sentence:
            disclaimer = (
                "This conclusion is a scenario-based synthesis of historical "
                "financial data rather than company guidance or an investment forecast."
            )

            if existing_conclusion.endswith(disclaimer):
                conclusion_body = existing_conclusion[
                    :-len(disclaimer)
                ].rstrip()

                executive_assessment["forward_looking_conclusion"] = (
                    f"{conclusion_body} "
                    f"{balance_sheet_sentence} "
                    f"{disclaimer}"
                )
            else:
                executive_assessment["forward_looking_conclusion"] = (
                    f"{existing_conclusion} {balance_sheet_sentence}"
                )

    return {
        "decision": decision,
        "summary": summary,
        "executive_assessment": executive_assessment,
        "status_counts": status_counts,
        "signals": signals,
        "derived_financial_metrics": derived_financial_metrics,
        "working_capital_diagnostic": working_capital_diagnostic,
        "earnings_quality_diagnostic": earnings_quality_diagnostic,
        "review_priorities": priorities,
        "disclaimer": (
            "Signals are analytical heuristics intended to support review. "
            "They are not evidence of fraud, misconduct, or investment advice."
        ),
    }
