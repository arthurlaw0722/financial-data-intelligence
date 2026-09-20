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
        },
    )


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

    executive_assessment = _build_executive_assessment(
        signals,
        trend_context=trend_context,
    )

    return {
        "decision": decision,
        "summary": summary,
        "executive_assessment": executive_assessment,
        "status_counts": status_counts,
        "signals": signals,
        "review_priorities": priorities,
        "disclaimer": (
            "Signals are analytical heuristics intended to support review. "
            "They are not evidence of fraud, misconduct, or investment advice."
        ),
    }
