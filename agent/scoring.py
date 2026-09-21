def calculate_trust_score(
    analysis: dict,
    leakage: dict,
    *,
    workflow: str = "generic",
) -> dict:
    score = 100
    penalties = []

    profile = analysis["profile"]

    duplicate_ratio = profile.get("duplicate_ratio", 0)
    if duplicate_ratio > 0.05:
        penalty = 10
        score -= penalty
        penalties.append(f"High duplicate ratio: -{penalty}")

    missing_values = profile.get("missing_values_ratio", {})
    latest_period_missing_ratio = profile.get(
        "latest_period_missing_ratio"
    )

    if (
        workflow == "financial_statement"
        and latest_period_missing_ratio is not None
    ):
        if latest_period_missing_ratio > 0.3:
            penalty = 15
            score -= penalty
            penalties.append(
                f"Severe latest-period missing values: -{penalty}"
            )

        elif latest_period_missing_ratio > 0.1:
            penalty = 8
            score -= penalty
            penalties.append(
                f"Moderate latest-period missing values: -{penalty}"
            )

    elif missing_values:
        max_missing = max(missing_values.values())

        if max_missing > 0.3:
            penalty = 15
            score -= penalty
            penalties.append(f"Severe missing values: -{penalty}")

        elif max_missing > 0.1:
            penalty = 8
            score -= penalty
            penalties.append(f"Moderate missing values: -{penalty}")

    outliers = analysis.get("outliers", {})

    high_outlier_cols = [
        col
        for col, info in outliers.items()
        if info["outlier_ratio"] > 0.05
    ]

    # For ordinary modelling datasets, unusually high outlier ratios can
    # indicate data-quality risk. For annual financial statements, however,
    # rapid company growth can legitimately produce statistical outliers in
    # a small number of fiscal-year observations. These are therefore
    # interpreted downstream by Financial Intelligence rather than treated
    # as a verification-quality penalty.
    if workflow != "financial_statement" and high_outlier_cols:
        penalty = min(15, len(high_outlier_cols) * 3)
        score -= penalty
        penalties.append(f"Outlier-heavy columns: -{penalty}")

    imbalance = analysis.get("class_imbalance", {})

    if imbalance.get("is_imbalanced"):
        penalty = 10
        score -= penalty
        penalties.append(f"Class imbalance: -{penalty}")

    leakage_count = leakage.get("risk_count", 0)

    if leakage_count > 0:
        penalty = min(25, leakage_count * 8)
        score -= penalty
        penalties.append(f"Possible target leakage: -{penalty}")

        # Target leakage is a hard safety blocker.
        score = min(score, 49)
        penalties.append("Safety gate: target leakage caps trust score at 49")

    score = max(score, 0)

    if score >= 85:
        grade = "High Trust"
    elif score >= 70:
        grade = "Medium Trust"
    elif score >= 50:
        grade = "Low Trust"
    else:
        grade = "High Risk"

    return {
        "trust_score": score,
        "trust_grade": grade,
        "penalties": penalties
    }
