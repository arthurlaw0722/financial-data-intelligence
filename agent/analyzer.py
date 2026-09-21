import pandas as pd
import numpy as np


def basic_profile(df: pd.DataFrame) -> dict:
    rows, cols = df.shape

    missing = df.isnull().mean().sort_values(ascending=False)
    missing_dict = {
        col: round(float(val), 4)
        for col, val in missing.items()
        if val > 0
    }

    duplicate_rows = int(df.duplicated().sum())

    dtypes = df.dtypes.astype(str).to_dict()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    latest_period_missing_ratio = None
    latest_period_missing_columns = []

    if "period" in df.columns and len(df) > 0:
        period_values = pd.to_datetime(df["period"], errors="coerce")

        if period_values.notna().any():
            latest_index = period_values.idxmax()
        else:
            latest_index = df.index[-1]

        financial_value_columns = list(numeric_cols)

        if financial_value_columns:
            latest_row = df.loc[
                latest_index,
                financial_value_columns,
            ]

            latest_period_missing_columns = [
                col
                for col in financial_value_columns
                if pd.isna(latest_row[col])
            ]

            latest_period_missing_ratio = round(
                len(latest_period_missing_columns)
                / len(financial_value_columns),
                4,
            )

    return {
        "rows": rows,
        "columns": cols,
        "dtypes": dtypes,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values_ratio": missing_dict,
        "latest_period_missing_ratio": latest_period_missing_ratio,
        "latest_period_missing_columns": latest_period_missing_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": round(duplicate_rows / rows, 4) if rows > 0 else 0,
    }


def detect_outliers(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((series < lower) | (series > upper)).sum()
        ratio = outliers / len(series)

        if ratio > 0:
            outlier_report[col] = {
                "outlier_count": int(outliers),
                "outlier_ratio": round(float(ratio), 4),
            }

    return outlier_report


def detect_class_imbalance(df: pd.DataFrame, target_column: str | None) -> dict:
    if not target_column or target_column not in df.columns:
        return {"status": "no_target_column_provided"}

    counts = df[target_column].value_counts(dropna=False)
    ratios = df[target_column].value_counts(normalize=True, dropna=False)

    return {
        "target_column": target_column,
        "class_counts": counts.to_dict(),
        "class_ratios": {str(k): round(float(v), 4) for k, v in ratios.to_dict().items()},
        "minority_class_ratio": round(float(ratios.min()), 4),
        "is_imbalanced": bool(ratios.min() < 0.1),
    }


def analyse_dataset(csv_path: str, target_column: str | None = None) -> dict:
    df = pd.read_csv(csv_path)

    profile = basic_profile(df)
    outliers = detect_outliers(df)
    imbalance = detect_class_imbalance(df, target_column)

    return {
        "profile": profile,
        "outliers": outliers,
        "class_imbalance": imbalance,
    }
