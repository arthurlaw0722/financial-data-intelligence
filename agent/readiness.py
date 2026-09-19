def assess_readiness(
    analysis: dict,
    leakage: dict,
    score: dict,
    workflow: str = "generic",
    context: dict | None = None,
) -> dict:
    """
    Convert technical verification results into business-friendly readiness decisions.
    """

    trust_score = score.get("trust_score", 0)
    leakage_count = leakage.get("risk_count", 0)

    imbalance = analysis.get("class_imbalance", {})
    is_imbalanced = imbalance.get("is_imbalanced", False)

    profile = analysis.get("profile", {})
    missing_values = profile.get("missing_values_ratio", {})
    max_missing = max(missing_values.values()) if missing_values else 0

    reasons = []
    next_steps = []

    context = context or {}

    if workflow == "financial_statement":
        period_count = int(
            context.get("period_count")
            or profile.get("rows")
            or 0
        )
        financial_metric_count = int(
            context.get("financial_metric_count")
            or 0
        )
        has_contextual_evidence = bool(
            context.get("has_contextual_evidence", False)
        )

        if leakage_count > 0 or trust_score < 70:
            business_decision_readiness = "Low"

        elif (
            trust_score >= 90
            and period_count >= 5
            and financial_metric_count >= 8
            and has_contextual_evidence
        ):
            business_decision_readiness = "High"

        elif (
            trust_score >= 80
            and period_count >= 3
            and financial_metric_count >= 5
        ):
            business_decision_readiness = "Medium"

        else:
            business_decision_readiness = "Low"

        if business_decision_readiness == "High":
            decision_readiness_basis = (
                "The financial dataset has strong integrity, sufficient "
                f"multi-year coverage ({period_count} fiscal periods), broad "
                f"metric coverage ({financial_metric_count} standardized metrics), "
                "and supporting contextual evidence for analyst-level review."
            )
            decision_next_step = (
                "Proceed with financial analysis while continuing normal "
                "filing and market-context validation."
            )

        elif business_decision_readiness == "Medium":
            decision_readiness_basis = (
                "The structured financial data is reliable and provides "
                f"{period_count} fiscal periods across "
                f"{financial_metric_count} standardized financial metrics. "
                "However, qualitative filing context such as management "
                "commentary, detailed notes, or forward guidance is not "
                "available in the current standardized dataset."
            )
            decision_next_step = (
                "Use Financial Intelligence to review the quantitative signals, "
                "then validate material findings against filing notes, management "
                "commentary, and other relevant company disclosures."
            )

        else:
            decision_readiness_basis = (
                "The current financial dataset does not yet provide sufficient "
                "integrity or analytical coverage for decision-support use."
            )
            decision_next_step = (
                "Resolve the identified data-quality or coverage limitations "
                "before relying on the dataset for financial analysis."
            )

        if leakage_count > 0:
            reasons.append("Possible target leakage detected.")
        elif trust_score < 70:
            reasons.append(
                "Dataset trust is below the minimum decision-support threshold."
            )
        else:
            reasons.append("No major blocking data-integrity issue detected.")

        next_steps.append(decision_next_step)

        return {
            "ml_readiness": "Not Applicable",
            "business_decision_readiness": business_decision_readiness,
            "should_train_model": False,
            "main_reasons": reasons,
            "recommended_next_steps": next_steps,
            "decision_readiness_basis": decision_readiness_basis,
            "decision_readiness_next_step": decision_next_step,
        }

    if trust_score >= 85:
        ml_readiness = "High"
    elif trust_score >= 70:
        ml_readiness = "Medium"
    elif trust_score >= 50:
        ml_readiness = "Low"
    else:
        ml_readiness = "Not Ready"

    if trust_score >= 85 and leakage_count == 0:
        business_decision_readiness = "Medium"
    else:
        business_decision_readiness = "Low"

    if is_imbalanced:
        reasons.append("Severe class imbalance detected.")
        next_steps.append("Use stratified split, resampling, class weights, and recall/F1-focused evaluation.")

    if leakage_count > 0:
        reasons.append("Possible target leakage detected.")
        next_steps.append("Review suspicious columns before training any downstream model.")

    if max_missing > 0.1:
        reasons.append("Meaningful missing values detected.")
        next_steps.append("Apply missing value treatment and document the imputation strategy.")

    if not reasons:
        reasons.append("No major blocking issue detected.")
        next_steps.append("Proceed with model experimentation, but validate before production use.")

    should_train_model = trust_score >= 70 and leakage_count == 0

    return {
        "ml_readiness": ml_readiness,
        "business_decision_readiness": business_decision_readiness,
        "should_train_model": should_train_model,
        "main_reasons": reasons,
        "recommended_next_steps": next_steps
    }
