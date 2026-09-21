from __future__ import annotations

from typing import Any


_SIGNAL_KEY_MAP = {
    "Accounting Integrity": "accounting_integrity",
    "Working Capital": "working_capital",
    "Earnings Quality": "earnings_quality",
    "Liquidity & Leverage": "liquidity_and_leverage",
}


def _compact_signal(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": signal.get("status"),
        "title": signal.get("title"),
        "interpretation": signal.get("interpretation"),
        "analyst_action": signal.get("analyst_action"),
        "metrics": signal.get("metrics", {}),
    }


def _compact_working_capital(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    if not diagnostic:
        return {}

    metrics = dict(diagnostic.get("metrics", {}))
    metrics.pop("intensity_series", None)

    return {
        "status": diagnostic.get("status"),
        "observed_issue": diagnostic.get("observed_issue"),
        "historical_pattern": diagnostic.get("historical_pattern"),
        "historical_pattern_label": diagnostic.get(
            "historical_pattern_label"
        ),
        "historical_level_context": diagnostic.get(
            "historical_level_context"
        ),
        "interpretation": diagnostic.get("interpretation"),
        "supporting_evidence": diagnostic.get(
            "supporting_evidence",
            [],
        ),
        "alternative_explanations": diagnostic.get(
            "alternative_explanations",
            [],
        ),
        "evidence_gaps": diagnostic.get("evidence_gaps", []),
        "analyst_action": diagnostic.get("analyst_action"),
        "metrics": metrics,
    }


def _compact_earnings_quality(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    if not diagnostic:
        return {}

    metrics = dict(diagnostic.get("metrics", {}))
    metrics.pop("cash_conversion_series", None)

    return {
        "status": diagnostic.get("status"),
        "cash_conversion_posture": diagnostic.get(
            "cash_conversion_posture"
        ),
        "historical_context": diagnostic.get(
            "historical_context"
        ),
        "recent_cash_conversion_pattern": diagnostic.get(
            "recent_cash_conversion_pattern"
        ),
        "interpretation": diagnostic.get("interpretation"),
        "observed_evidence": diagnostic.get(
            "observed_evidence",
            [],
        ),
        "evidence_gaps": diagnostic.get("evidence_gaps", []),
        "analyst_action": diagnostic.get("analyst_action"),
        "metrics": metrics,
    }


def _compact_trend_context(
    trend_context: dict[str, Any],
) -> dict[str, Any]:
    metric_trends = {}

    for metric, trend in trend_context.get(
        "metric_trends",
        {},
    ).items():
        metric_trends[metric] = {
            "observations": trend.get("observations"),
            "first_period": trend.get("first_period"),
            "last_period": trend.get("last_period"),
            "latest_value": trend.get("latest_value"),
            "latest_yoy_pct": trend.get("latest_yoy_pct"),
            "recent_cagr_pct": trend.get("recent_cagr_pct"),
            "full_history_cagr_pct": trend.get(
                "full_history_cagr_pct"
            ),
        }

    return {
        "period_count": trend_context.get("period_count"),
        "start_period": trend_context.get("start_period"),
        "end_period": trend_context.get("end_period"),
        "metric_trends": metric_trends,
    }


def _append_unique(
    destination: list[str],
    values: Any,
) -> None:
    if values is None:
        return

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return

    for value in values:
        if (
            isinstance(value, str)
            and value.strip()
            and value not in destination
        ):
            destination.append(value)


def build_financial_intelligence_context(
    *,
    result: dict[str, Any],
    periods: dict[str, Any],
    trend_context: dict[str, Any],
    next_fiscal_year_outlook: dict[str, Any],
    forward_looking_conclusion: str | None,
) -> dict[str, Any]:
    """
    Build a compact, deterministic context package for downstream
    AI-assisted financial explanation.

    This function does not call an LLM and does not create new
    financial conclusions. It reorganises outputs already produced
    by the Financial Intelligence engine.
    """

    executive = result.get("executive_assessment", {})
    cross_statement = result.get(
        "cross_statement_assessment",
        {},
    )

    working_capital = result.get(
        "working_capital_diagnostic",
        {},
    )
    earnings_quality = result.get(
        "earnings_quality_diagnostic",
        {},
    )
    derived_metrics = result.get(
        "derived_financial_metrics",
        {},
    )

    signal_context: dict[str, Any] = {}

    for signal in result.get("signals", []):
        area = signal.get("area")
        key = _SIGNAL_KEY_MAP.get(
            area,
            str(area or "unknown")
            .lower()
            .replace(" & ", "_and_")
            .replace(" ", "_"),
        )
        signal_context[key] = _compact_signal(signal)

    evidence_gaps: list[str] = []

    _append_unique(
        evidence_gaps,
        working_capital.get("evidence_gaps"),
    )
    _append_unique(
        evidence_gaps,
        earnings_quality.get("evidence_gaps"),
    )
    _append_unique(
        evidence_gaps,
        executive.get("evidence_gap"),
    )

    recommended_actions: list[str] = []

    _append_unique(
        recommended_actions,
        working_capital.get("analyst_action"),
    )
    _append_unique(
        recommended_actions,
        earnings_quality.get("analyst_action"),
    )
    _append_unique(
        recommended_actions,
        cross_statement.get("analyst_action"),
    )
    _append_unique(
        recommended_actions,
        executive.get("recommended_next_step"),
    )

    current = periods.get("current", {})
    previous = periods.get("previous", {})

    return {
        "schema_version": "1.0",
        "context_type": "financial_intelligence",
        "purpose": (
            "Grounded context for evidence-based financial "
            "analysis and AI-assisted explanation."
        ),
        "period": {
            "current": current.get("period"),
            "previous": previous.get("period"),
        },
        "financial_statements": {
            "current": current,
            "previous": previous,
        },
        "overall_assessment": {
            "decision": result.get("decision"),
            "summary": result.get("summary"),
            "signals_requiring_review": (
                result.get("status_counts", {}).get(
                    "Review",
                    0,
                )
                + result.get("status_counts", {}).get(
                    "High Attention",
                    0,
                )
            ),
            "cross_statement_posture": (
                cross_statement.get("overall_posture")
            ),
        },
        "executive_assessment": {
            "overall_finding": executive.get(
                "overall_finding"
            ),
            "primary_review_point": executive.get(
                "primary_review_point"
            ),
            "trend_summary": executive.get(
                "trend_summary"
            ),
            "supporting_evidence": executive.get(
                "supporting_evidence"
            ),
            "follow_up_analysis": executive.get(
                "follow_up_analysis"
            ),
            "evidence_gap": executive.get(
                "evidence_gap"
            ),
            "recommended_next_step": executive.get(
                "recommended_next_step"
            ),
            "forward_looking_conclusion": (
                forward_looking_conclusion
            ),
        },
        "financial_diagnostics": {
            "working_capital": (
                _compact_working_capital(
                    working_capital
                )
            ),
            "earnings_quality": (
                _compact_earnings_quality(
                    earnings_quality
                )
            ),
            "signals": signal_context,
        },
        "derived_metrics": {
            "metrics": derived_metrics.get("metrics", {}),
            "available_metrics": derived_metrics.get(
                "available_metrics",
                [],
            ),
            "missing_metrics": derived_metrics.get(
                "missing_metrics",
                [],
            ),
        },
        "historical_context": _compact_trend_context(
            trend_context
        ),
        "cross_statement_reasoning": {
            "overall_posture": cross_statement.get(
                "overall_posture"
            ),
            "component_statuses": cross_statement.get(
                "component_statuses",
                {},
            ),
            "reinforcing_evidence": cross_statement.get(
                "reinforcing_evidence",
                [],
            ),
            "offsetting_evidence": cross_statement.get(
                "offsetting_evidence",
                [],
            ),
            "watchpoints": cross_statement.get(
                "watchpoints",
                [],
            ),
            "synthesis": cross_statement.get(
                "synthesis"
            ),
            "analyst_action": cross_statement.get(
                "analyst_action"
            ),
        },
        "evidence_gaps": evidence_gaps,
        "recommended_actions": recommended_actions,
        "scenario_outlook": next_fiscal_year_outlook,
        "ai_guardrails": [
            (
                "Use only the supplied financial evidence "
                "when explaining company-specific findings."
            ),
            (
                "Distinguish observed financial facts from "
                "analytical scenarios and possible explanations."
            ),
            (
                "Do not present possible explanations as "
                "confirmed causes unless supporting evidence "
                "is available."
            ),
            (
                "Explicitly mention material evidence gaps "
                "when they limit a conclusion."
            ),
            (
                "Treat the next-fiscal-year outlook as a "
                "historical trend scenario, not company guidance "
                "or an investment forecast."
            ),
        ],
    }
