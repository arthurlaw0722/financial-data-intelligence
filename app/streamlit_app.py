import sys
import os
import tempfile
import hashlib
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data_sources.sec_financial_mapper import (
    build_sec_annual_financial_statement,
)

from agent.analyzer import analyse_dataset
from agent.leakage import detect_possible_leakage
from agent.scoring import calculate_trust_score
from agent.report_generator import generate_markdown_report
from agent.proof import create_proof
from agent.readiness import assess_readiness
from analytics.profiler import profile_dataset
from analytics.target_analysis import analyze_target
from analytics.ml_pipeline import train_binary_models, model_comparison_table
from analytics.financial_intelligence import (
    analyze_financial_intelligence,
    _build_multi_year_trend_context,
    _build_next_fiscal_year_outlook,
    _build_forward_looking_executive_conclusion,
)
from analytics.financial_context import (
    build_financial_intelligence_context,
)
from analytics.ai_financial_analyst import ask_financial_analyst
from analytics.financial_statement_adapter import (
    detect_financial_statement_schema,
    prepare_financial_periods,
)
from analytics.statistics import (
    numeric_statistics,
    categorical_statistics,
)
from analytics.visualizations import (
    histogram_chart,
    box_chart,
    scatter_chart,
    correlation_heatmap,
    missing_values_chart,
)


st.set_page_config(
    page_title="Financial Data Intelligence",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.1rem;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #e8f5ee;
        color: #0f7a4f;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .section-note {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 0.5rem;
    }

    .st-key-run_verification button {
        min-height: 3rem;
        font-weight: 700;
        border-radius: 0.6rem;
        transition: all 0.15s ease;
    }

    .st-key-run_verification button:not(:disabled) {
        background-color: #D62828 !important;
        border-color: #D62828 !important;
        color: white !important;
    }

    .st-key-run_verification button:not(:disabled):hover {
        background-color: #B71C1C !important;
        border-color: #B71C1C !important;
        color: white !important;
    }

    .st-key-run_verification button:disabled {
        background-color: #FDECEC !important;
        border: 1px solid #DFA3A3 !important;
        color: #A63A3A !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header():
    st.markdown(
        '<div class="main-title">Financial Data Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="subtitle">'
            "Retrieve standardized annual 10-K financials from SEC EDGAR, "
            "verify the data, surface multi-year financial signals, and investigate the evidence with an AI financial analyst."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class="badge">SEC EDGAR Integration</span>
    <span class="badge">Financial Data Verification</span>
        <span class="badge">Multi-Year Financial Intelligence</span>
        <span class="badge">AI Financial Analyst</span>
    <span class="badge">SHA256 Evidence Trail</span>
        """,
        unsafe_allow_html=True,
    )


class InMemoryDatasetFile(io.BytesIO):
    """File-like object for API-generated datasets."""

    def __init__(self, raw_bytes, name, file_id):
        super().__init__(raw_bytes)
        self.name = name
        self.size = len(raw_bytes)
        self.file_id = file_id


def dataframe_as_dataset_file(df, name, source_id):
    """Create a deterministic CSV representation for verification."""
    raw_bytes = df.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")

    digest = hashlib.sha256(raw_bytes).hexdigest()

    return InMemoryDatasetFile(
        raw_bytes,
        name=name,
        file_id=f"{source_id}:{digest}",
    )


def uploaded_file_signature(uploaded_file):
    return (
        uploaded_file.name,
        getattr(uploaded_file, "size", None),
        getattr(uploaded_file, "file_id", None),
    )


def load_dataset(uploaded_file):
    signature = uploaded_file_signature(uploaded_file)

    if st.session_state.get("loaded_file_signature") != signature:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        st.session_state["dataset_df"] = df
        st.session_state["dataset_profile"] = profile_dataset(df)
        st.session_state["loaded_file_signature"] = signature

        st.session_state.pop("verification_result", None)
        st.session_state.pop("verification_signature", None)

    return (
        st.session_state["dataset_df"],
        st.session_state["dataset_profile"],
    )


def render_configuration():
    st.subheader("SEC Filing Lookup")

    # Public portfolio mode currently focuses on the SEC filing workflow.
    # The CSV workflow remains implemented below and can be re-enabled later.
    data_source = "SEC Public Filing"

    uploaded_file = None
    df = None
    dataset_profile = None
    target_column = ""
    dataset_name = ""

    # =====================================================
    # CSV SOURCE
    # =====================================================
    if data_source == "Upload CSV":
        st.session_state.pop("active_sec_metadata", None)

        config_col1, config_col2 = st.columns([1.35, 1])

        with config_col1:
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key="dataset_uploader",
            )

        if uploaded_file is not None:
            with st.spinner("Loading dataset..."):
                df, dataset_profile = load_dataset(uploaded_file)

        with config_col2:
            if df is None:
                target_column = ""

                st.selectbox(
                    "Target column",
                    ["Upload a CSV first"],
                    disabled=True,
                    key="target_column_disabled",
                )

            else:
                target_column = st.selectbox(
                    "Target column",
                    options=[""] + df.columns.tolist(),
                    index=0,
                    format_func=lambda value: (
                        "None / No target"
                        if value == ""
                        else value
                    ),
                    help=(
                        "Select the column you want to predict or analyse."
                    ),
                    key="target_column_selector",
                )

            if uploaded_file is not None:
                current_name_signature = uploaded_file_signature(
                    uploaded_file
                )

                if (
                    st.session_state.get(
                        "dataset_name_file_signature"
                    )
                    != current_name_signature
                ):
                    file_stem = os.path.splitext(
                        uploaded_file.name
                    )[0]

                    special_dataset_names = {
                        "creditcard": (
                            "Credit Card Fraud Detection"
                        ),
                        "financial_statement_demo": (
                            "Financial Statement Demo"
                        ),
                    }

                    suggested_name = special_dataset_names.get(
                        file_stem.lower(),
                        file_stem
                        .replace("_", " ")
                        .replace("-", " ")
                        .title(),
                    )

                    st.session_state[
                        "dataset_name_input"
                    ] = suggested_name

                    st.session_state[
                        "dataset_name_file_signature"
                    ] = current_name_signature

            elif "dataset_name_input" not in st.session_state:
                st.session_state["dataset_name_input"] = ""

            dataset_name = st.text_input(
                "Dataset name",
                key="dataset_name_input",
            )

        return (
            uploaded_file,
            df,
            dataset_profile,
            target_column,
            dataset_name,
        )

    # =====================================================
    # SEC EDGAR SOURCE
    # =====================================================
    sec_col1 = st.container()

    with sec_col1:
        ticker = st.text_input(
            "Company ticker",
            placeholder="e.g. NVDA, AAPL, MSFT",
            key="sec_ticker_input",
        ).strip().upper()

        load_sec = st.button(
            "Load SEC filing",
            type="primary",
            use_container_width=True,
            key="load_sec_filing",
        )

        st.caption(
            "Retrieve and standardize annual 10-K financial statement data "
            "from SEC EDGAR Company Facts."
        )

    if load_sec:
        if not ticker:
            st.warning("Enter a company ticker first.")

        else:
            try:
                with st.spinner(
                    f"Loading {ticker} from SEC EDGAR..."
                ):
                    sec_result = (
                        build_sec_annual_financial_statement(
                            ticker
                        )
                    )

                st.session_state[
                    "sec_financial_result"
                ] = sec_result

                st.session_state[
                    "sec_loaded_ticker"
                ] = ticker

                # A newly loaded source must be verified again.
                st.session_state.pop(
                    "verification_result",
                    None,
                )
                st.session_state.pop(
                    "verification_signature",
                    None,
                )

                st.rerun()

            except Exception as exc:
                st.session_state.pop(
                    "sec_financial_result",
                    None,
                )
                st.session_state.pop(
                    "sec_loaded_ticker",
                    None,
                )

                st.error(
                    "Unable to load this SEC filing. "
                    f"{type(exc).__name__}: {exc}"
                )

    sec_result = st.session_state.get(
        "sec_financial_result"
    )
    loaded_ticker = st.session_state.get(
        "sec_loaded_ticker"
    )

    # Do not silently keep showing data for an old ticker.
    if (
        sec_result is not None
        and ticker
        and ticker != loaded_ticker
    ):
        sec_result = None
        st.info(
            f"Ticker changed to {ticker}. "
            "Click Load SEC filing to retrieve the new company."
        )

    target_column = ""

    dataset_name = (
        f"{sec_result['company_name']} — SEC Annual Financials"
        if sec_result is not None
        else ""
    )

    if sec_result is not None:
        sec_df = sec_result["dataframe"].copy()

        virtual_name = (
            f"{sec_result['ticker'].lower()}"
            "_sec_annual_financials.csv"
        )

        uploaded_file = dataframe_as_dataset_file(
            df=sec_df,
            name=virtual_name,
            source_id=(
                f"sec:{sec_result['ticker']}:"
                f"{sec_result['cik']}"
            ),
        )

        df, dataset_profile = load_dataset(
            uploaded_file
        )

        st.session_state[
            "active_sec_metadata"
        ] = sec_result

        periods = (
            df["period"]
            .astype(str)
            .tolist()
            if "period" in df.columns
            else []
        )

        st.success(
            f"Loaded {sec_result['company_name']} "
            f"({sec_result['ticker']})"
        )

        if len(periods) >= 2:
            st.caption(
                "SEC EDGAR · Annual 10-K data · "
                f"FY{pd.to_datetime(periods[0]).year}–FY{pd.to_datetime(periods[-1]).year} · "
                f"{len(periods)} fiscal years · "
                f"{len(sec_result['selected_concepts'])} "
                "standardized financial metrics"
            )
        else:
            st.caption(
                "Source: SEC EDGAR Company Facts"
            )

    else:
        st.session_state.pop(
            "active_sec_metadata",
            None,
        )

        if not ticker:
            st.info(
                "Enter a US-listed company ticker to retrieve "
                "its annual SEC financial data."
            )

    return (
        uploaded_file,
        df,
        dataset_profile,
        target_column,
        dataset_name,
    )


def render_overview(df, dataset_profile, target_column):
    st.subheader("Dataset Overview")
    st.caption(
        "A compact view of dataset size, structure, quality, and a sample preview."
    )

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric("Rows", f"{dataset_profile['rows']:,}")
    overview_col2.metric("Columns", f"{dataset_profile['columns']:,}")
    overview_col3.metric("Numeric", dataset_profile["numeric_columns"])
    overview_col4.metric("Categorical", dataset_profile["categorical_columns"])

    overview_col5, overview_col6, overview_col7, overview_col8 = st.columns(4)
    overview_col5.metric(
        "Missing Cells",
        f"{dataset_profile['missing_cells']:,}",
    )
    overview_col6.metric(
        "Missing %",
        f"{dataset_profile['missing_ratio'] * 100:.2f}%",
    )
    overview_col7.metric(
        "Duplicate Rows",
        f"{dataset_profile['duplicate_rows']:,}",
    )
    overview_col8.metric(
        "Memory",
        f"{dataset_profile['memory_mb']:.2f} MB",
    )

    st.divider()
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    preview_col1, preview_col2, preview_col3 = st.columns(3)
    preview_col1.metric("Rows", f"{df.shape[0]:,}")
    preview_col2.metric("Columns", f"{df.shape[1]:,}")
    preview_col3.metric(
        "Target Column",
        target_column if target_column else "Not provided",
    )


def render_target_analysis(df, target_column):
    st.subheader("Target Analysis")
    st.caption(
        "Understand the prediction target, class balance, and strongest "
        "numeric associations."
    )

    if not target_column:
        st.info(
            "Select a target column above to unlock target-aware analysis."
        )
        return

    if target_column not in df.columns:
        st.warning("The selected target column is not available in this dataset.")
        return

    target_summary = analyze_target(df, target_column)

    target_col1, target_col2, target_col3, target_col4 = st.columns(
        [1.0, 1.8, 0.8, 0.9]
    )
    target_col1.metric("Target", target_summary["target_column"])
    target_col2.metric("Task", target_summary["task_type"])
    target_col3.metric("Unique Values", target_summary["unique_values"])
    target_col4.metric(
        "Missing %",
        f'{target_summary["missing_pct"]:.2f}%',
    )

    if "Classification" in target_summary["task_type"]:
        class_col1, class_col2, class_col3 = st.columns(3)

        class_col1.metric(
            "Majority Class",
            target_summary["majority_class"],
            help=f'{target_summary["majority_count"]:,} rows',
        )
        class_col2.metric(
            "Minority Class",
            target_summary["minority_class"],
            help=f'{target_summary["minority_count"]:,} rows',
        )

        imbalance_value = target_summary["imbalance_ratio"]
        class_col3.metric(
            "Imbalance Ratio",
            (
                f"{imbalance_value:.2f}:1"
                if imbalance_value is not None
                else "N/A"
            ),
        )

        detail_col1, detail_col2 = st.columns([1, 1.25])

        with detail_col1:
            st.markdown("### Class Distribution")
            class_distribution_df = pd.DataFrame(
                target_summary["class_distribution"]
            )
            st.dataframe(
                class_distribution_df.rename(columns={"class": "Class", "count": "Count", "percentage": "Percentage (%)"}),
                use_container_width=True,
                hide_index=True,
            )

        with detail_col2:
            associations = target_summary["top_numeric_associations"]
            st.markdown("### Top Numeric Associations")
            if associations:
                association_df = pd.DataFrame(associations)
                st.dataframe(
                    association_df.rename(columns={"feature": "Feature", "correlation": "Correlation", "absolute_correlation": "Strength"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No numeric target associations are available.")
    else:
        associations = target_summary["top_numeric_associations"]
        st.markdown("### Top Numeric Associations")
        if associations:
            association_df = pd.DataFrame(associations)
            st.dataframe(
                association_df.rename(columns={"feature": "Feature", "correlation": "Correlation", "absolute_correlation": "Strength"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No numeric target associations are available.")


def render_explorer(df, dataset_profile):
    st.subheader("Data Explorer")
    st.caption(
        "Explore descriptive statistics and interactive visual analysis "
        "without leaving this workspace."
    )

    numeric_stats = numeric_statistics(df)
    categorical_stats = categorical_statistics(df)

    numeric_tab, categorical_tab, visual_tab = st.tabs(
        [
            "Numeric Statistics",
            "Categorical Statistics",
            "Visual Analysis",
        ]
    )

    with numeric_tab:
        if numeric_stats.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(
                numeric_stats.round(4),
                use_container_width=True,
            )

    with categorical_tab:
        if categorical_stats.empty:
            st.info("No categorical columns found.")
        else:
            st.dataframe(
                categorical_stats,
                use_container_width=True,
            )

    with visual_tab:
        numeric_columns = dataset_profile["numeric_column_names"]
        group_columns = (
            dataset_profile["categorical_column_names"]
            + dataset_profile["low_cardinality_numeric_column_names"]
        )

        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Correlation Heatmap",
                "Missing Values",
            ],
            key="visual_chart_type",
        )

        if chart_type in {"Histogram", "Box Plot", "Scatter Plot"} and not numeric_columns:
            st.info("No numeric columns are available for this chart.")
            return

        if chart_type == "Histogram":
            column = st.selectbox(
                "Column",
                numeric_columns,
                key="histogram_column",
            )
            group = st.selectbox(
                "Group / colour",
                ["None"] + group_columns,
                key="histogram_group",
            )

            figure = histogram_chart(
                df,
                column,
                None if group == "None" else group,
            )
            st.plotly_chart(figure, use_container_width=True)

        elif chart_type == "Box Plot":
            column = st.selectbox(
                "Column",
                numeric_columns,
                key="box_column",
            )
            group = st.selectbox(
                "Group by",
                ["None"] + group_columns,
                key="box_group",
            )

            figure = box_chart(
                df,
                column,
                None if group == "None" else group,
            )
            st.plotly_chart(figure, use_container_width=True)

        elif chart_type == "Scatter Plot":
            selector_col1, selector_col2 = st.columns(2)

            with selector_col1:
                x_column = st.selectbox(
                    "X-axis",
                    numeric_columns,
                    key="scatter_x",
                )

            with selector_col2:
                default_y_index = 1 if len(numeric_columns) > 1 else 0
                y_column = st.selectbox(
                    "Y-axis",
                    numeric_columns,
                    index=default_y_index,
                    key="scatter_y",
                )

            group = st.selectbox(
                "Group / colour",
                ["None"] + group_columns,
                key="scatter_group",
            )

            figure = scatter_chart(
                df,
                x_column,
                y_column,
                None if group == "None" else group,
            )

            st.caption(
                "Large datasets are sampled to a maximum of 30,000 points "
                "for responsive visualisation. Statistical calculations "
                "still use the full dataset."
            )
            st.plotly_chart(figure, use_container_width=True)

        elif chart_type == "Correlation Heatmap":
            figure = correlation_heatmap(df)

            if figure is None:
                st.info(
                    "No numeric columns are available for correlation analysis."
                )
            else:
                st.plotly_chart(figure, use_container_width=True)

        elif chart_type == "Missing Values":
            figure = missing_values_chart(df)

            if figure is None:
                st.success("No missing values were found in this dataset.")
            else:
                st.plotly_chart(figure, use_container_width=True)


def build_verification_result(uploaded_file, df, target_column, dataset_name):
    target = target_column if target_column.strip() else None

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    csv_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as tmp:
            tmp.write(raw_bytes)
            csv_path = tmp.name

        analysis = analyse_dataset(csv_path, target)
        leakage = detect_possible_leakage(df, target)
        is_financial_statement = detect_financial_statement_schema(df).get(
            "is_financial_statement",
            False,
        )

        score = calculate_trust_score(
            analysis,
            leakage,
            workflow=(
                "financial_statement"
                if is_financial_statement
                else "generic"
            ),
        )
        financial_schema = detect_financial_statement_schema(df)
        is_financial_statement = financial_schema.get(
            "is_financial_statement",
            False,
        )

        normalized_columns = {
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            for column in df.columns
        }

        contextual_evidence_columns = {
            "management_commentary",
            "management_discussion",
            "mda",
            "filing_notes",
            "financial_notes",
            "notes",
            "forward_guidance",
            "guidance",
        }

        has_contextual_evidence = bool(
            normalized_columns & contextual_evidence_columns
        )

        readiness = assess_readiness(
            analysis,
            leakage,
            score,
            workflow=(
                "financial_statement"
                if is_financial_statement
                else "generic"
            ),
            context=(
                {
                    "period_count": len(df),
                    "financial_metric_count": len(
                        financial_schema.get("available_metrics", [])
                    ),
                    "has_contextual_evidence": has_contextual_evidence,
                }
                if is_financial_statement
                else None
            ),
        )
        temp_report = generate_markdown_report(
            dataset_name,
            analysis,
            leakage,
            score,
            {
                "dataset_fingerprint": "pending",
                "report_hash": "pending",
                "execution_timestamp": "pending",
            },
        )

        proof = create_proof(csv_path, temp_report)

        final_report = generate_markdown_report(
            dataset_name,
            analysis,
            leakage,
            score,
            proof,
        )

        return {
            "analysis": analysis,
            "leakage": leakage,
            "score": score,
            "readiness": readiness,
            "proof": proof,
            "final_report": final_report,
            "dataset_name": dataset_name,
        }
    finally:
        if csv_path and os.path.exists(csv_path):
            os.remove(csv_path)


def render_verification_result(
    result,
    is_financial_statement=False,
):
    analysis = result["analysis"]
    leakage = result["leakage"]
    score = result["score"]
    readiness = result["readiness"]
    proof = result["proof"]
    final_report = result["final_report"]
    dataset_name = result["dataset_name"]

    leakage_count = leakage.get("risk_count", 0)

    if leakage_count == 0:
        if is_financial_statement:
            st.success(
                "Verification completed — no blocking data-integrity "
                "issues were detected."
            )
        else:
            st.success(
                "Verification passed — no target leakage risks were detected."
            )
    else:
        st.error(
            f"Verification completed — {leakage_count} possible target "
            "leakage risk(s) detected. Machine Learning is blocked."
        )

    if is_financial_statement:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric(
            "Trust Score",
            f"{score['trust_score']}/100",
        )
        metric_col2.metric(
            "Trust Grade",
            score["trust_grade"],
        )
        metric_col3.metric(
            "Decision Readiness",
            readiness["business_decision_readiness"],
        )
        metric_col4.metric(
            "Workflow",
            "Financial Statement",
        )
    else:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric(
            "Trust Score",
            f"{score['trust_score']}/100",
        )
        metric_col2.metric(
            "Trust Grade",
            score["trust_grade"],
        )
        metric_col3.metric(
            "ML Readiness",
            readiness["ml_readiness"],
        )
        metric_col4.metric(
            "Business Readiness",
            readiness["business_decision_readiness"],
        )

    summary_tab, risks_tab, proof_tab = st.tabs(
        [
            "Summary",
            "Risks & Readiness",
            "SHA256 Proof & Report",
        ]
    )

    with summary_tab:
        st.markdown("### Key Findings")
        findings_col1, findings_col2 = st.columns([1, 1])

        with findings_col1:
            st.warning("**Main reasons**")
            for reason in readiness["main_reasons"]:
                st.markdown(f"- {reason}")

        with findings_col2:
            st.info("**Recommended next steps**")

            if is_financial_statement:
                st.markdown(
                    "- Open **Financial Intelligence** and review any "
                    "flagged accounting, working-capital, earnings-quality, "
                    "and liquidity signals."
                )
            else:
                for step in readiness["recommended_next_steps"]:
                    st.markdown(f"- {step}")

    with risks_tab:
        st.markdown("### Dataset Risk Breakdown")

        risk_col1, risk_col2, risk_col3 = st.columns(3)
        profile = analysis["profile"]
        imbalance = analysis.get("class_imbalance", {})

        risk_col1.metric("Duplicate Rows", profile["duplicate_rows"])
        risk_col2.metric("Duplicate Ratio", profile["duplicate_ratio"])
        risk_col3.metric(
            "Leakage Risks",
            leakage.get("risk_count", 0),
        )

        risk_detail_col1, risk_detail_col2 = st.columns([1, 1])

        with risk_detail_col1:
            st.markdown("### Score Penalties")
            if score["penalties"]:
                for penalty in score["penalties"]:
                    st.markdown(f"- {penalty}")
            else:
                st.success("No major penalties detected.")

        with risk_detail_col2:
            if is_financial_statement:
                st.markdown("### Financial Statement Context")
                decision_readiness = readiness.get(
                    "business_decision_readiness",
                    "Unknown",
                )

                if decision_readiness == "High":
                    readiness_explanation = (
                        "The financial data and available contextual evidence support "
                        "a stronger business-review assessment."
                    )
                elif decision_readiness == "Medium":
                    readiness_explanation = (
                        "The financial data is reliable, but standardized SEC metrics "
                        "alone do not provide enough contextual evidence for "
                        "high-confidence business conclusions."
                    )
                else:
                    readiness_explanation = (
                        "Additional financial or contextual evidence is recommended "
                        "before relying on the dataset for business conclusions."
                    )

                st.info(
                    f"**Decision Readiness is {decision_readiness}:** "
                    f"{readiness_explanation}\n\n"
                    "Target-class and class-imbalance checks are not applicable to "
                    "this financial-statement workflow."
                )
            else:
                st.markdown("### Class Imbalance")

                if imbalance.get("status") == "no_target_column_provided":
                    st.warning("No target column provided.")
                else:
                    st.metric(
                        "Target Column",
                        imbalance["target_column"],
                    )
                    imbalance_col1, imbalance_col2 = st.columns(2)
                    imbalance_col1.metric(
                        "Minority Class Ratio",
                        imbalance["minority_class_ratio"],
                    )
                    imbalance_col2.metric(
                        "Is Imbalanced",
                        str(imbalance["is_imbalanced"]),
                    )

    with proof_tab:
        st.markdown("### SHA256 Verification Proof")
        st.caption(
            "These SHA256 values let you verify the integrity of the "
            "uploaded dataset and the generated report."
        )

        st.markdown("**SHA256 Dataset Fingerprint**")
        st.code(proof["dataset_fingerprint"])

        st.markdown("**SHA256 Canonical Report Hash**")
        st.code(proof["report_hash"])

        st.markdown("**Execution Timestamp**")
        st.code(proof["execution_timestamp"])

        with st.expander("View full markdown report"):
            st.markdown(final_report)

        with st.expander("View JSON-style summary"):
            st.json(
                {
                    "dataset_name": dataset_name,
                    "trust_score": score["trust_score"],
                    "trust_grade": score["trust_grade"],
                    "readiness": readiness,
                    "leakage": leakage,
                    "proof": proof,
                }
            )

        st.download_button(
            "Download report.md",
            final_report,
            file_name="report.md",
        )


def render_verification(
    uploaded_file,
    df,
    target_column,
    dataset_name,
):
    st.subheader("Verification")
    is_financial_statement = detect_financial_statement_schema(df).get(
        "is_financial_statement",
        False,
    )

    if is_financial_statement:
        st.caption(
            "Check financial-data quality, statement structure, decision readiness, "
            "and SHA256 integrity before analyst review."
        )
    else:
        st.caption(
            "Check dataset quality, target leakage, ML readiness, and SHA256 "
            "integrity before model experimentation."
        )

    if uploaded_file is None or df is None:
        st.info("Upload a CSV dataset first.")
        return

    current_signature = (
        uploaded_file_signature(uploaded_file),
        target_column,
        dataset_name,
    )

    if (
        st.session_state.get("verification_signature") is not None
        and st.session_state.get("verification_signature") != current_signature
    ):
        st.session_state.pop("verification_result", None)
        st.session_state.pop("verification_signature", None)

    run_button = st.button(
        "Run Verification",
        type="primary",
        use_container_width=True,
        key="run_verification",
    )

    if run_button:
        with st.status(
            "Running dataset verification...",
            expanded=True,
        ) as status:
            st.write("Checking dataset quality and class imbalance...")
            st.write("Checking possible target leakage...")
            st.write("Calculating trust and readiness scores...")
            st.write("Generating SHA256 integrity proof and report...")

            result = build_verification_result(
                uploaded_file,
                df,
                target_column,
                dataset_name,
            )

            st.session_state["verification_result"] = result
            st.session_state["verification_signature"] = current_signature

            status.update(
                label="Verification completed",
                state="complete",
                expanded=False,
            )

            # Refresh the page so the Machine Learning lock updates immediately.
            st.rerun()

    result = st.session_state.get("verification_result")

    if (
        result is not None
        and st.session_state.get("verification_signature") == current_signature
    ):
        render_verification_result(
            result,
            is_financial_statement=detect_financial_statement_schema(
                df
            ).get("is_financial_statement", False),
        )
    else:
        is_financial_statement = detect_financial_statement_schema(df).get(
            "is_financial_statement",
            False,
        )

        if is_financial_statement:
            st.markdown(
                """
**What this check includes**
- Financial dataset quality and duplicate-row checks
- Financial-statement structure and metric coverage
- Dataset trust score and decision-readiness guidance
- SHA256 dataset fingerprint and canonical report hash
"""
            )
        else:
            st.markdown(
                """
**What this check includes**
- Dataset quality and duplicate-row checks
- Outlier-heavy column detection
- Class imbalance analysis
- Possible target leakage detection
- Dataset trust score and readiness guidance
- SHA256 dataset fingerprint and canonical report hash
"""
            )


def render_getting_started():
    st.subheader("Getting started")
    st.markdown(
        """
1. Enter a **US-listed company ticker** to retrieve standardized annual 10-K financial data from **SEC EDGAR**.
2. Review **Overview** and **Explore** to inspect the standardized financial statements and multi-year history.
3. Open **Verification** to assess data quality, financial-statement structure, trust, and SHA256 integrity.
4. Open **Financial Intelligence** to review accounting integrity, working capital, earnings quality, liquidity & leverage, multi-year trends, and the next-year scenario.
5. Use the **AI Financial Analyst** to investigate questions grounded in the verified financial analysis and supporting evidence.
6. Review the **SHA256 Proof & Report** for the verification record and evidence trail.

        """
    )


def render_financial_intelligence(
    uploaded_file,
    df,
    target_column,
    dataset_name,
):
    """Render financial-statement intelligence and analyst review signals."""

    st.subheader("Financial Intelligence")
    st.caption(
        "Translate verified financial-statement data into accounting, "
        "working-capital, earnings-quality, and liquidity signals."
    )

    current_verification_signature = (
        uploaded_file_signature(uploaded_file),
        target_column,
        dataset_name,
    )

    verification_result = st.session_state.get("verification_result")
    verification_signature = st.session_state.get("verification_signature")

    if (
        verification_result is None
        or verification_signature != current_verification_signature
    ):
        st.warning(
            "**Verification required**\n\n"
            "Run Verification before using Financial Intelligence for "
            "this dataset."
        )
        return

    leakage_count = (
        verification_result
        .get("leakage", {})
        .get("risk_count", 0)
    )

    if leakage_count > 0:
        st.error(
            "Financial Intelligence is blocked because Verification "
            "detected possible target leakage."
        )
        return

    schema = detect_financial_statement_schema(df)

    if not schema["is_financial_statement"]:
        st.info(
            "**Financial statement analysis is not applicable to this dataset.**\n\n"
            "This dataset does not match the standardized multi-period "
            "financial-statement structure. Transaction-level or modelling "
            "datasets can still use Verification and Machine Learning."
        )

        with st.expander("Expected financial statement format"):
            st.markdown(
                """
A compatible dataset should contain:

- a `period` column with at least two reporting dates
- at least five recognised financial statement metrics
- one row per reporting period

Examples of recognised metrics include `revenue`,
`accounts_receivable`, `inventory`, `net_income`,
`operating_cash_flow`, `total_assets`, `total_liabilities`,
`total_equity`, `current_assets`, `current_liabilities`,
`total_debt`, and `cash_and_equivalents`.
                """
            )

            detected = schema.get("available_metrics", [])

            if detected:
                st.caption(
                    "Recognised financial metrics detected: "
                    + ", ".join(detected)
                )
            else:
                st.caption(
                    "No recognised financial-statement metrics were detected."
                )

        return

    try:
        periods = prepare_financial_periods(df)

        trend_context = _build_multi_year_trend_context(
            periods.get("history", [])
        )

        result = analyze_financial_intelligence(
            current=periods["current"],
            previous=periods["previous"],
            trend_context=trend_context,
        )

        next_fiscal_year_outlook = _build_next_fiscal_year_outlook(
            periods.get("history", [])
        )

        forward_looking_conclusion = (
            _build_forward_looking_executive_conclusion(
                result.get("executive_assessment", {}),
                trend_context,
                next_fiscal_year_outlook,
            )
        )

        financial_intelligence_context = (
            build_financial_intelligence_context(
                result=result,
                periods=periods,
                trend_context=trend_context,
                next_fiscal_year_outlook=next_fiscal_year_outlook,
                forward_looking_conclusion=(
                    forward_looking_conclusion
                ),
            )
        )

    except ValueError as exc:
        st.error(str(exc))
        return

    diagnostic = result.get("working_capital_diagnostic", {})

    if diagnostic:
        st.markdown("### Working Capital Diagnostic")

        historical_level = diagnostic.get(
            "historical_level_context",
            {},
        )

        movement_col, pattern_col, level_col = st.columns(3)

        with movement_col:
            st.metric(
                "Movement signal",
                diagnostic.get("status", "Unavailable"),
            )

        with pattern_col:
            st.metric(
                "Historical pattern",
                diagnostic.get(
                    "historical_pattern_label",
                    "Unavailable",
                ),
            )

        with level_col:
            st.metric(
                "Historical level",
                historical_level.get(
                    "label",
                    "Unavailable",
                ),
            )

        st.caption(
            "The movement signal identifies unusual recent change, while "
            "historical context shows whether the current level is unusual "
            "relative to the company's own history."
        )

        observed_issue = diagnostic.get("observed_issue")
        diagnostic_status = diagnostic.get("status")

        if observed_issue:
            if diagnostic_status == "High Attention":
                st.warning(observed_issue)
            elif diagnostic_status == "Review":
                st.info(observed_issue)
            else:
                st.success(observed_issue)

        interpretation = diagnostic.get("interpretation")

        if interpretation:
            st.markdown("##### Diagnostic interpretation")
            st.write(interpretation)

        with st.expander("Diagnostic evidence and next steps"):
            history_col, evidence_col = st.columns(2)

            with history_col:
                st.markdown("**Historical pattern**")
                historical_pattern = diagnostic.get(
                    "historical_pattern"
                )

                if historical_pattern:
                    st.write(historical_pattern)
                else:
                    st.caption(
                        "Historical pattern is unavailable."
                    )

                st.markdown("**Historical level context**")
                level_interpretation = historical_level.get(
                    "interpretation"
                )

                if level_interpretation:
                    st.write(level_interpretation)
                else:
                    st.caption(
                        "Historical level context is unavailable."
                    )

            with evidence_col:
                st.markdown(
                    "**Evidence supporting the assessment**"
                )

                supporting_evidence = diagnostic.get(
                    "supporting_evidence",
                    [],
                )

                if supporting_evidence:
                    st.markdown(
                        "\n".join(
                            f"- {item}"
                            for item in supporting_evidence
                        )
                    )
                else:
                    st.caption(
                        "No supporting evidence is available."
                    )

                st.markdown("**Evidence still required**")

                evidence_gaps = diagnostic.get(
                    "evidence_gaps",
                    [],
                )

                if evidence_gaps:
                    st.markdown(
                        "\n".join(
                            f"- {item}"
                            for item in evidence_gaps
                        )
                    )
                else:
                    st.caption(
                        "No material evidence gaps identified."
                    )

            alternative_explanations = diagnostic.get(
                "alternative_explanations",
                [],
            )

            if alternative_explanations:
                st.markdown("**Possible explanations**")
                st.markdown(
                    "\n".join(
                        f"- {item}"
                        for item in alternative_explanations
                    )
                )

            analyst_action = diagnostic.get("analyst_action")

            if analyst_action:
                st.markdown("**Recommended analyst action**")
                st.info(analyst_action)

    earnings_diagnostic = result.get(
        "earnings_quality_diagnostic",
        {},
    )

    if earnings_diagnostic:
        st.markdown("### Earnings Quality Diagnostic")

        historical_context = earnings_diagnostic.get(
            "historical_context",
            {},
        )

        recent_pattern = earnings_diagnostic.get(
            "recent_cash_conversion_pattern",
            {},
        )

        earnings_metrics = earnings_diagnostic.get(
            "metrics",
            {},
        )

        current_quality = earnings_diagnostic.get(
            "status",
            "Unavailable",
        )

        cash_conversion_value = earnings_metrics.get(
            "cash_conversion_of_earnings"
        )

        cash_conversion_posture = earnings_diagnostic.get(
            "cash_conversion_posture",
            "Unavailable",
        )

        trend_label = recent_pattern.get(
            "label",
            "Unavailable",
        )

        display_trend = {
            "Persistent recent decline": "Persistent decline",
            "Persistent recent improvement": "Persistent improvement",
            "Mixed recent trend": "Mixed trend",
        }.get(trend_label, trend_label)

        quality_col, conversion_col, trend_col = st.columns(3)

        with quality_col:
            st.metric(
                "Current quality",
                current_quality,
            )

        with conversion_col:
            conversion_display = (
                f"{cash_conversion_value:.2f}x"
                if isinstance(cash_conversion_value, (int, float))
                else "Unavailable"
            )

            st.metric(
                "Cash conversion",
                conversion_display,
            )

            st.caption(cash_conversion_posture)

        with trend_col:
            st.metric(
                "Trend watch",
                display_trend,
            )

        historical_label = historical_context.get(
            "label",
            "Unavailable",
        )
        latest_ratio = historical_context.get("latest_ratio")
        prior_min = historical_context.get("prior_min_ratio")

        context_parts = [f"Historical context: {historical_label}"]

        if (
            isinstance(latest_ratio, (int, float))
            and isinstance(prior_min, (int, float))
        ):
            context_parts.append(
                f"{latest_ratio:.2f}x vs prior minimum {prior_min:.2f}x"
            )

        st.caption(" · ".join(context_parts))

        if (
            current_quality == "Normal"
            and cash_conversion_posture == "Broadly supportive"
            and trend_label == "Persistent recent decline"
        ):
            st.info(
                "Current earnings quality remains broadly supportive, "
                "but cash conversion has weakened across four consecutive "
                "fiscal periods and warrants continued monitoring."
            )

        interpretation = earnings_diagnostic.get("interpretation")

        with st.expander("Earnings quality analysis and evidence"):
            if interpretation:
                st.markdown("**Diagnostic interpretation**")
                st.write(interpretation)

            st.markdown("**Historical cash-conversion context**")

            historical_interpretation = historical_context.get(
                "interpretation"
            )

            if historical_interpretation:
                st.write(historical_interpretation)

            recent_interpretation = recent_pattern.get(
                "interpretation"
            )

            if recent_interpretation:
                st.write(recent_interpretation)

            st.markdown("**Evidence supporting the assessment**")

            observed_evidence = earnings_diagnostic.get(
                "observed_evidence",
                [],
            )

            if observed_evidence:
                st.markdown(
                    "\n".join(
                        f"- {item}"
                        for item in observed_evidence
                    )
                )

            st.markdown("**Evidence still required**")

            evidence_gaps = earnings_diagnostic.get(
                "evidence_gaps",
                [],
            )

            if evidence_gaps:
                st.markdown(
                    "\n".join(
                        f"- {item}"
                        for item in evidence_gaps
                    )
                )

            analyst_action = earnings_diagnostic.get("analyst_action")

            if analyst_action:
                st.markdown("**Recommended analyst action**")
                st.info(analyst_action)

    # Floating AI Financial Analyst chat
    st.markdown(
        """
        <style>
        .st-key-ai_financial_chat_launcher {
            position: fixed;
            right: 1.25rem;
            top: 50%;
            transform: translateY(-50%);
            z-index: 1000;
            width: 190px;
        }

        .st-key-ai_financial_chat_launcher button {
            border-radius: 999px !important;
            font-weight: 650 !important;
            padding-top: 0.65rem !important;
            padding-bottom: 0.65rem !important;
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.14);
        }

        div[data-testid="stPopoverBody"] {
            width: 460px;
            min-width: 360px;
            max-width: min(80vw, 900px);

            height: min(68vh, 640px);
            min-height: 420px;
            max-height: 85vh;

            resize: both;
            overflow: auto;

            /* Keep the right edge fixed so resizing expands leftward. */
            position: fixed !important;
            right: 1.5rem !important;
            left: auto !important;
            transform: none !important;

            /* Chrome native resize handle at bottom-left. */
            direction: rtl;
        }

        div[data-testid="stPopoverBody"] > * {
            direction: ltr;
        }

        /* Hide only the Enter instruction; Enter still submits. */
        div[data-testid="stPopoverBody"]
        div[data-testid="InputInstructions"] {
            display: none !important;
        }

        /* Visible resize-grip hint */
        div[data-testid="stPopoverBody"]::after {
            content: "Resize";
            position: absolute;
            left: 13px;
            bottom: 9px;

            color: #8b93a1;
            font-size: 11px;
            font-weight: 600;
            line-height: 1;
            letter-spacing: 0.02em;

            background: transparent;
            border: none;
            padding: 0;
            box-shadow: none;

            pointer-events: none;
            user-select: none;
            z-index: 10;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.45rem 0.65rem;
            margin-bottom: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    ai_context_marker = repr(
        (
            financial_intelligence_context.get("overall_assessment"),
            financial_intelligence_context.get("executive_assessment"),
            financial_intelligence_context.get("financial_diagnostics"),
            financial_intelligence_context.get("historical_context"),
            financial_intelligence_context.get("cross_statement_reasoning"),
        )
    )

    if st.session_state.get("ai_financial_context_marker") != ai_context_marker:
        st.session_state["ai_financial_context_marker"] = ai_context_marker
        st.session_state["ai_financial_chat"] = []

    with st.container(key="ai_financial_chat_launcher"):
        with st.popover("💬 AI Financial Analyst"):
            st.markdown("### AI Financial Analyst")
            st.caption(
                "Ask questions about the financial analysis on this page. "
                "Responses are grounded in the verified Financial Intelligence context."
            )

            chat_history = st.session_state.setdefault(
                "ai_financial_chat",
                [],
            )

            if not chat_history:
                st.info(
                    "Try asking: Why is inventory the main watchpoint?"
                )

            for message in chat_history[-8:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            with st.form(
                "ai_financial_chat_form",
                clear_on_submit=True,
            ):
                ai_question = st.text_input(
                    "Message",
                    placeholder="Ask about earnings, cash flow, risks...",
                    label_visibility="collapsed",
                )
                ai_send = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                )

            def _set_ai_financial_error(exc, question):
                error_text = str(exc)
                error_lower = error_text.lower()

                if (
                    "503" in error_text
                    or "unavailable" in error_lower
                    or "high demand" in error_lower
                ):
                    error_type = "busy"
                    title = "AI Analyst is temporarily busy"
                    message = (
                        "Gemini is experiencing high demand right now. "
                        "Your question has been kept, so you can retry it."
                    )
                elif "gemini_api_key" in error_lower:
                    error_type = "configuration"
                    title = "AI Analyst is not configured"
                    message = (
                        "Gemini API access is not available to this "
                        "Streamlit process."
                    )
                else:
                    error_type = "request"
                    title = "AI Analyst could not complete the request"
                    message = (
                        "The request could not be completed. "
                        "Please try again."
                    )

                st.session_state["ai_financial_error"] = {
                    "type": error_type,
                    "title": title,
                    "message": message,
                    "question": question,
                }


            if ai_send:
                clean_ai_question = ai_question.strip()

                if not clean_ai_question:
                    st.warning("Enter a question first.")
                else:
                    chat_history.append(
                        {
                            "role": "user",
                            "content": clean_ai_question,
                        }
                    )

                    try:
                        with st.spinner("Analysing financial context..."):
                            ai_answer = ask_financial_analyst(
                                financial_intelligence_context,
                                clean_ai_question,
                            )

                    except Exception as exc:
                        _set_ai_financial_error(
                            exc,
                            clean_ai_question,
                        )
                    else:
                        chat_history.append(
                            {
                                "role": "assistant",
                                "content": ai_answer,
                            }
                        )
                        st.session_state.pop(
                            "ai_financial_error",
                            None,
                        )

                    st.session_state["ai_financial_chat"] = chat_history
                    st.rerun()


            ai_error = st.session_state.get("ai_financial_error")

            if ai_error:
                error_type = ai_error.get("type", "request")
                error_title = ai_error.get(
                    "title",
                    "AI Analyst unavailable",
                )
                error_message = ai_error.get(
                    "message",
                    "Please try again.",
                )

                if error_type == "busy":
                    st.warning(
                        f"**{error_title}**\n\n{error_message}"
                    )
                else:
                    st.error(
                        f"**{error_title}**\n\n{error_message}"
                    )

                retry_question = (
                    ai_error.get("question") or ""
                ).strip()

                if retry_question:
                    st.caption(
                        f'Ready to retry: "{retry_question}"'
                    )

                    if st.button(
                        "↻ Retry",
                        key="retry_ai_financial_chat",
                        use_container_width=True,
                    ):
                        try:
                            with st.spinner(
                                "Retrying financial analysis..."
                            ):
                                retry_answer = ask_financial_analyst(
                                    financial_intelligence_context,
                                    retry_question,
                                )

                        except Exception as retry_exc:
                            _set_ai_financial_error(
                                retry_exc,
                                retry_question,
                            )
                            st.rerun()

                        else:
                            chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": retry_answer,
                                }
                            )

                            st.session_state[
                                "ai_financial_chat"
                            ] = chat_history

                            st.session_state.pop(
                                "ai_financial_error",
                                None,
                            )

                            st.rerun()
            if chat_history:
                if st.button(
                    "Clear conversation",
                    key="clear_ai_financial_chat",
                    use_container_width=True,
                ):
                    st.session_state["ai_financial_chat"] = []
                    st.rerun()

            st.caption(
                "AI explanations are grounded in the deterministic analysis. "
                "They do not replace analyst review."
            )

    st.markdown("### Overall Assessment")

    decision = result["decision"]
    summary = result["summary"]

    if decision == "HIGH ATTENTION":
        st.error(f"**{decision}** — {summary}")
    elif decision == "REVIEW REQUIRED":
        st.warning(f"**{decision}** — {summary}")
    else:
        st.success(f"**{decision}** — {summary}")

    status_counts = result["status_counts"]
    review_count = (
        status_counts.get("Review", 0)
        + status_counts.get("High Attention", 0)
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    previous_period_date = pd.to_datetime(
        periods["previous_period"],
        errors="coerce",
    )
    current_period_date = pd.to_datetime(
        periods["current_period"],
        errors="coerce",
    )

    if pd.notna(previous_period_date) and pd.notna(current_period_date):
        period_comparison_label = (
            f"FY{previous_period_date.year} → FY{current_period_date.year}"
        )
    else:
        period_comparison_label = (
            f"{periods['previous_period']} → {periods['current_period']}"
        )

    summary_col1.metric(
        "Period Comparison",
        period_comparison_label,
    )
    summary_col2.metric(
        "Financial Metrics",
        len(periods["available_metrics"]),
    )
    summary_col3.metric(
        "Signals Requiring Review",
        review_count,
    )

    executive = result.get(
        "executive_assessment",
        {},
    )

    st.markdown("### Executive Assessment")

    st.write(
        executive.get(
            "overall_finding",
            summary,
        )
    )

    cross_statement = (
        result.get("cross_statement_assessment")
        or {}
    )

    cross_statement_posture = cross_statement.get(
        "overall_posture"
    )

    cross_statement_synthesis = cross_statement.get(
        "synthesis"
    )

    if cross_statement_posture:
        st.markdown(
            f"**Cross-statement posture:** "
            f"{cross_statement_posture}"
        )

    trend_summary = executive.get("trend_summary")
    primary_review_point = executive.get("primary_review_point")

    trend_summary_lower = (trend_summary or "").lower()

    if "broad-based acceleration" in trend_summary_lower:
        momentum_label = "Broad-based acceleration"
    elif (
        "below the longer-term trend" in trend_summary_lower
        or "moderation" in trend_summary_lower
    ):
        momentum_label = "Moderating"
    elif trend_summary:
        momentum_label = "Mixed trend"
    else:
        momentum_label = "Insufficient history"

    available_outlook_items = [
        item
        for item in next_fiscal_year_outlook.get("items", [])
        if item.get("status") == "Available"
    ]

    outlook_directions = [
        item.get("direction")
        for item in available_outlook_items
        if item.get("direction")
    ]

    if outlook_directions and all(
        direction == "Expansion"
        for direction in outlook_directions
    ):
        scenario_label = "Expansion"
        scenario_caption = (
            f"Across {len(outlook_directions)} core metrics"
        )
    elif outlook_directions and all(
        direction == "Contraction"
        for direction in outlook_directions
    ):
        scenario_label = "Contraction"
        scenario_caption = (
            f"Across {len(outlook_directions)} core metrics"
        )
    elif outlook_directions and all(
        direction == "Broadly stable"
        for direction in outlook_directions
    ):
        scenario_label = "Broadly stable"
        scenario_caption = (
            f"Across {len(outlook_directions)} core metrics"
        )
    elif outlook_directions:
        scenario_label = "Mixed outlook"
        scenario_caption = "Core metrics point in different directions"
    else:
        scenario_label = "Insufficient history"
        scenario_caption = "No usable next-year trend scenario"

    if primary_review_point:
        review_label = "Conditional"
        review_caption = "Material watchpoint remains"
    else:
        review_label = "Clear"
        review_caption = "No material watchpoint under current rules"

    balance_sheet_summary = executive.get(
        "balance_sheet_summary"
    ) or {}

    balance_sheet_status = balance_sheet_summary.get("status")

    if balance_sheet_status in {
        "Normal",
        "Review",
        "High Attention",
    }:
        balance_sheet_label = balance_sheet_status
    else:
        balance_sheet_label = "Limited data"

    balance_caption_parts = []

    balance_current_ratio = balance_sheet_summary.get(
        "current_ratio"
    )
    if isinstance(balance_current_ratio, (int, float)):
        balance_caption_parts.append(
            f"Current ratio {balance_current_ratio:.2f}x"
        )

    balance_debt_assets = balance_sheet_summary.get(
        "debt_to_assets_pct"
    )
    if isinstance(balance_debt_assets, (int, float)):
        balance_caption_parts.append(
            f"Debt / Assets {balance_debt_assets:.1f}%"
        )

    balance_debt_growth = balance_sheet_summary.get(
        "debt_growth_pct"
    )
    if (
        balance_sheet_status in {"Review", "High Attention"}
        and isinstance(balance_debt_growth, (int, float))
    ):
        balance_caption_parts.append(
            f"Debt {balance_debt_growth:+.1f}% YoY"
        )

    balance_sheet_driver = balance_sheet_summary.get(
        "trigger_summary"
    )

    if (
        balance_sheet_status in {"Review", "High Attention"}
        and balance_sheet_driver
    ):
        balance_sheet_caption = balance_sheet_driver
    else:
        balance_sheet_caption = (
            " · ".join(balance_caption_parts)
            if balance_caption_parts
            else "Balance-sheet coverage is limited"
        )

    st.markdown("#### Executive Outlook")

    (
        outlook_col1,
        outlook_col2,
        outlook_col3,
        outlook_col4,
    ) = st.columns(4)

    with outlook_col1:
        with st.container(border=True):
            st.caption("Momentum")
            st.markdown(f"**{momentum_label}**")
            st.caption(
                "Recent 3-year CAGR vs full-period history"
            )

    with outlook_col2:
        with st.container(border=True):
            st.caption("Next-year scenario")
            st.markdown(f"**{scenario_label}**")
            st.caption(scenario_caption)

    with outlook_col3:
        with st.container(border=True):
            st.caption("Review posture")
            st.markdown(f"**{review_label}**")
            st.caption(review_caption)

    with outlook_col4:
        with st.container(border=True):
            st.caption("Balance-sheet posture")
            st.markdown(f"**{balance_sheet_label}**")
            st.caption(balance_sheet_caption)

    if (
        trend_summary
        or cross_statement_synthesis
        or forward_looking_conclusion
    ):
        with st.expander(
            "Executive synthesis",
            expanded=False,
        ):
            if trend_summary:
                st.markdown("**Multi-year context**")
                st.write(trend_summary)

            if cross_statement_synthesis:
                st.markdown(
                    "**Cross-statement context**"
                )
                st.write(cross_statement_synthesis)

            if forward_looking_conclusion:
                st.markdown(
                    "**Forward-looking executive conclusion**"
                )
                st.write(forward_looking_conclusion)

    executive_col1, executive_col2 = st.columns(
        [1, 1]
    )

    with executive_col1:
        st.markdown("**Primary watchpoint**")

        primary_review_point = executive.get(
            "primary_review_point"
        )

        if primary_review_point:
            st.warning(
                primary_review_point
            )
        else:
            st.success(
                "No material watchpoint was identified under the current "
                "analytical rules."
            )

    with executive_col2:
        st.markdown("**Recommended next step**")

        st.info(
            executive.get(
                "recommended_next_step",
                "Continue standard financial review.",
            )
        )

    supporting_evidence = executive.get(
        "supporting_evidence"
    )

    if supporting_evidence:
        st.markdown(
            "**Supporting evidence**"
        )

        st.write(
            supporting_evidence
        )

    follow_up_analysis = executive.get(
        "follow_up_analysis"
    )

    evidence_gap = executive.get(
        "evidence_gap"
    )

    if follow_up_analysis or evidence_gap:
        with st.expander(
            "Follow-up analysis",
            expanded=False,
        ):
            if follow_up_analysis:
                st.markdown(
                    "**What the current data indicates**"
                )

                st.write(
                    follow_up_analysis
                )

            if evidence_gap:
                st.markdown(
                    "**Evidence gap**"
                )

                st.write(
                    evidence_gap
                )

    def format_financial_value(value):
        if value is None:
            return "N/A"

        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        absolute = abs(number)

        if absolute >= 1_000_000_000:
            return f"{number / 1_000_000_000:,.2f}B"
        if absolute >= 1_000_000:
            return f"{number / 1_000_000:,.2f}M"
        if absolute >= 1_000:
            return f"{number / 1_000:,.2f}K"

        return f"{number:,.2f}"


    def format_percentage(value):
        if value is None:
            return "N/A"

        try:
            return f"{float(value):,.2f}%"
        except (TypeError, ValueError):
            return "N/A"


    st.markdown("### Multi-Year Trend Analysis")

    trend_period_count = trend_context.get("period_count", 0)
    trend_start_period = trend_context.get("start_period")
    trend_end_period = trend_context.get("end_period")

    try:
        trend_start_fy = pd.to_datetime(trend_start_period).year
        trend_end_fy = pd.to_datetime(trend_end_period).year
        trend_period_label = (
            f"{trend_period_count} fiscal years analysed · "
            f"FY{trend_start_fy}–FY{trend_end_fy}"
        )
    except Exception:
        trend_period_label = (
            f"{trend_period_count} fiscal periods analysed"
        )

    st.caption(trend_period_label)

    core_trend_metric_labels = {
        "revenue": "Revenue",
        "net_income": "Net Income",
        "operating_cash_flow": "Operating Cash Flow",
    }

    supporting_trend_metric_labels = {
        "accounts_receivable": "Accounts Receivable",
        "inventory": "Inventory",
        "cash_and_equivalents": "Cash & Equivalents",
        "total_debt": "Total Debt",
        "total_assets": "Total Assets",
    }

    def build_trend_rows(metric_labels):
        rows = []

        for metric_name, display_name in metric_labels.items():
            trend = trend_context.get(
                "metric_trends",
                {},
            ).get(
                metric_name,
                {},
            )

            if trend.get("latest_value") is None:
                continue

            rows.append(
                {
                    "Metric": display_name,
                    "Latest": format_financial_value(
                        trend.get("latest_value")
                    ),
                    "Latest YoY": format_percentage(
                        trend.get("latest_yoy_pct")
                    ),
                    "Recent CAGR (3Y)": format_percentage(
                        trend.get("recent_cagr_pct")
                    ),
                    "Full-period CAGR": format_percentage(
                        trend.get("full_history_cagr_pct")
                    ),
                }
            )

        return rows

    core_trend_rows = build_trend_rows(
        core_trend_metric_labels
    )

    st.markdown("**Core Performance Trends**")

    st.dataframe(
        pd.DataFrame(core_trend_rows),
        hide_index=True,
        use_container_width=True,
    )

    supporting_trend_rows = build_trend_rows(
        supporting_trend_metric_labels
    )

    if supporting_trend_rows:
        st.markdown("**Financial Position & Operating Trends**")
        st.caption(
            "Supporting balance-sheet and working-capital trends. "
            "These metrics inform financial review but are not used "
            "as direct next-year forecast targets."
        )

        st.dataframe(
            pd.DataFrame(supporting_trend_rows),
            hide_index=True,
            use_container_width=True,
        )

    # Financial-position ratios derived from the latest available period.
    latest_metric_trends = trend_context.get("metric_trends", {})

    def latest_metric_value(metric_name):
        value = latest_metric_trends.get(
            metric_name,
            {},
        ).get("latest_value")

        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    current_assets = latest_metric_value("current_assets")
    current_liabilities = latest_metric_value("current_liabilities")
    total_debt = latest_metric_value("total_debt")
    total_assets = latest_metric_value("total_assets")
    total_equity = latest_metric_value("total_equity")
    cash = latest_metric_value("cash_and_equivalents")

    signals = result["signals"]

    liquidity_signal = next(
        (
            signal
            for signal in signals
            if signal.get("area") == "Liquidity & Leverage"
        ),
        {},
    )

    liquidity_metrics = liquidity_signal.get("metrics", {})

    current_ratio = liquidity_metrics.get("current_ratio")
    debt_to_assets_pct = liquidity_metrics.get("debt_to_assets_pct")
    debt_to_equity = liquidity_metrics.get("debt_to_equity")
    cash_to_debt = liquidity_metrics.get("cash_to_debt")

    # The engine stores Debt / Assets as percentage points
    # (for example 4.1 means 4.1%), while the existing UI formatter
    # expects a decimal ratio.
    debt_to_assets = (
        debt_to_assets_pct / 100
        if debt_to_assets_pct is not None
        else None
    )

    if any(
        ratio is not None
        for ratio in [
            current_ratio,
            debt_to_assets,
            debt_to_equity,
            cash_to_debt,
        ]
    ):
        st.markdown("### Financial Position Ratios")
        st.caption(
            "Latest-period balance-sheet ratios used to support "
            "liquidity and leverage review."
        )

        ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)

        ratio_col1.metric(
            "Current Ratio",
            (
                f"{current_ratio:.2f}x"
                if current_ratio is not None
                else "N/A"
            ),
        )

        ratio_col2.metric(
            "Debt / Assets",
            (
                f"{debt_to_assets:.1%}"
                if debt_to_assets is not None
                else "N/A"
            ),
        )

        ratio_col3.metric(
            "Debt / Equity",
            (
                f"{debt_to_equity:.2f}x"
                if debt_to_equity is not None
                else "N/A"
            ),
        )

        ratio_col4.metric(
            "Cash / Debt",
            (
                f"{cash_to_debt:.2f}x"
                if cash_to_debt is not None
                else "N/A"
            ),
        )

    st.markdown("### Next Fiscal Year Outlook")

    st.caption(
        "Historical trend scenario based on the available annual financial "
        "record. The outlook is not company guidance or an investment forecast."
    )

    outlook_rows = []

    for item in next_fiscal_year_outlook.get("items", []):
        lower_value = item.get("lower_value")
        upper_value = item.get("upper_value")

        if lower_value is not None and upper_value is not None:
            indicative_range = (
                f"{format_financial_value(lower_value)} – "
                f"{format_financial_value(upper_value)}"
            )
        else:
            indicative_range = "N/A"

        outlook_rows.append(
            {
                "Metric": item.get(
                    "display_name",
                    "Financial Metric",
                ),
                "Direction": item.get(
                    "direction",
                    "N/A",
                ),
                "Latest": format_financial_value(
                    item.get("latest_value")
                ),
                "Moderated Growth Assumption": format_percentage(
                    item.get("base_growth_pct")
                ),
                "Base Outlook": format_financial_value(
                    item.get("projected_value")
                ),
                "Indicative Range": indicative_range,
            }
        )

    if outlook_rows:
        st.dataframe(
            pd.DataFrame(outlook_rows),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "Insufficient historical observations are available to produce "
            "a next-fiscal-year trend outlook."
        )

    st.markdown("### Financial Signals")

    signal_columns = st.columns(len(signals))

    for column, signal in zip(signal_columns, signals):
        column.metric(
            signal["area"],
            signal["status"],
        )

    st.markdown("### Review Details")

    for signal in signals:
        label = f"{signal['area']} — {signal['status']}"

        with st.expander(label):
            st.markdown("**Why it matters**")
            st.write(signal["interpretation"])

            st.markdown("**What to investigate**")
            st.write(signal["analyst_action"])

            metrics = signal.get("metrics", {})

            if metrics:
                rows = []

                for metric_name, value in metrics.items():
                    display_name = (
                        metric_name
                        .replace("_pct", "")
                        .replace("_pp", "")
                        .replace("_", " ")
                        .title()
                    )

                    if value is None:
                        display_value = "N/A"
                    elif metric_name.endswith("_pct"):
                        display_value = f"{value:,.2f}%"
                    elif metric_name.endswith("_pp"):
                        display_value = f"{value:,.2f} pp"
                    elif isinstance(value, (int, float)):
                        display_value = f"{value:,.2f}"
                    else:
                        display_value = str(value)

                    rows.append(
                        {
                            "Metric": display_name,
                            "Value": display_value,
                        }
                    )

                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                )

    st.caption(result["disclaimer"])


def render_machine_learning(uploaded_file, df, target_column, dataset_name):
    """Render the portfolio machine-learning benchmark workspace."""

    st.subheader("Machine Learning")
    st.caption(
        "Train and compare baseline classification models only after "
        "the dataset passes the leakage safety gate."
    )

    if not target_column or target_column not in df.columns:
        st.info(
            "Select a binary target column in Dataset configuration "
            "before running the machine-learning benchmark."
        )
        return

    current_verification_signature = (
        uploaded_file_signature(uploaded_file),
        target_column,
        dataset_name,
    )

    verification_result = st.session_state.get("verification_result")
    verification_signature = st.session_state.get("verification_signature")

    if (
        verification_result is None
        or verification_signature != current_verification_signature
    ):
        st.warning(
            "**Verification required**\n\n"
            "Run Verification to unlock Machine Learning for this dataset and target. "
            "Access is granted only if no target leakage is detected."
        )
        return

    verification_leakage = verification_result.get("leakage", {})
    verification_leakage_count = verification_leakage.get("risk_count", 0)

    if verification_leakage_count > 0:
        st.error(
            "Machine Learning is blocked because Verification detected "
            "possible target leakage."
        )
        st.caption(
            "Review Risks & Readiness in Verification, remove or validate "
            "the leakage-related features, then rerun Verification."
        )
        return

    target_values = df[target_column].dropna()

    if target_values.nunique() != 2:
        st.warning(
            "The current ML benchmark supports binary classification only. "
            f"The selected target contains {target_values.nunique()} unique values."
        )
        return

    class_counts = target_values.value_counts()
    minority_pct = (
        class_counts.min() / class_counts.sum() * 100
        if class_counts.sum()
        else 0
    )

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

    overview_col1.metric(
        "Dataset Rows",
        f"{len(df):,}",
    )
    overview_col2.metric(
        "Features",
        f"{max(len(df.columns) - 1, 0):,}",
    )
    overview_col3.metric(
        "Target",
        target_column,
    )
    overview_col4.metric(
        "Minority Class",
        f"{minority_pct:.3f}%",
    )

    st.divider()

    st.markdown("### Verification Gate")

    st.success(
        "Passed — this exact dataset and target were verified, "
        "and no target leakage risks were detected."
    )

    gate_col1, gate_col2 = st.columns(2)
    gate_col1.metric("Verification Status", "Passed")
    gate_col2.metric("Leakage Risks", verification_leakage_count)

    if minority_pct < 10:
        st.info(
            "This is an imbalanced classification problem. "
            "Model recommendation therefore prioritises PR-AUC rather "
            "than accuracy alone."
        )

    st.divider()

    st.markdown("### Model Benchmark")

    current_signature = (
        f"{target_column}|{len(df)}|{len(df.columns)}"
    )

    if st.session_state.get("ml_signature") != current_signature:
        st.session_state.pop("ml_result", None)
        st.session_state["ml_signature"] = current_signature

    run_ml = st.button(
        "Run ML Benchmark",
        type="primary",
        use_container_width=True,
        key="run_ml_benchmark",
    )

    if run_ml:
        with st.spinner(
            "Training Logistic Regression and Random Forest..."
        ):
            result = train_binary_models(
                df,
                target_column=target_column,
            )

        st.session_state["ml_result"] = result

    result = st.session_state.get("ml_result")

    if result is None:
        st.caption(
            "Run the benchmark to compare Logistic Regression and "
            "Random Forest on a stratified train/test split."
        )
        return

    if result.get("status") != "completed":
        validation = result.get("validation", {})
        reason = validation.get(
            "reason",
            "The machine-learning benchmark could not be completed.",
        )
        st.error(reason)
        return

    models = result.get("models", {})

    if not models:
        st.warning("No model results were returned.")
        return

    # PR-AUC is particularly informative for highly imbalanced problems.
    recommended_model = max(
        models,
        key=lambda name: models[name].get("pr_auc", -1),
    )
    recommended_metrics = models[recommended_model]

    st.success("Machine-learning benchmark completed.")

    recommendation_col1, recommendation_col2, recommendation_col3 = st.columns(
        [1.5, 1, 1]
    )

    recommendation_col1.metric(
        "Top Benchmark Model",
        recommended_model,
    )
    recommendation_col2.metric(
        "PR-AUC",
        f"{recommended_metrics.get('pr_auc', 0):.4f}",
    )
    recommendation_col3.metric(
        "F1 Score",
        f"{recommended_metrics.get('f1', 0):.4f}",
    )

    st.caption(
        "Benchmark ranking is based on PR-AUC, which is more informative "
        "than raw accuracy when the positive class is rare."
    )

    st.markdown("#### Model Comparison")

    comparison_df = model_comparison_table(result).copy()

    rename_map = {
        "Model": "Model",
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1",
        "ROC-AUC": "ROC-AUC",
        "PR-AUC": "PR-AUC",
    }

    comparison_df = comparison_df.rename(columns=rename_map)

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Model Performance Curves")
    st.caption(
        "Precision–Recall is the primary comparison for this highly "
        "imbalanced classification problem. ROC is shown as a complementary view."
    )

    pr_figure = go.Figure()
    roc_figure = go.Figure()

    for model_name, metrics in models.items():
        pr_data = metrics.get("pr_curve", {})
        roc_data = metrics.get("roc_curve", {})

        recall_values = pr_data.get("recall", [])
        precision_values = pr_data.get("precision", [])

        if recall_values and precision_values:
            pr_figure.add_trace(
                go.Scatter(
                    x=recall_values,
                    y=precision_values,
                    mode="lines",
                    name=model_name,
                )
            )

        fpr_values = roc_data.get("fpr", [])
        tpr_values = roc_data.get("tpr", [])

        if fpr_values and tpr_values:
            roc_figure.add_trace(
                go.Scatter(
                    x=fpr_values,
                    y=tpr_values,
                    mode="lines",
                    name=model_name,
                )
            )

    pr_figure.update_layout(
        title="Precision–Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        legend_title_text="Model",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    roc_figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash"),
        )
    )

    roc_figure.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        legend_title_text="Model",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        st.plotly_chart(
            pr_figure,
            use_container_width=True,
            key="ml_precision_recall_curve",
        )

    with curve_col2:
        st.plotly_chart(
            roc_figure,
            use_container_width=True,
            key="ml_roc_curve",
        )

    st.markdown("#### Top Benchmark Model Diagnostics")

    confusion = recommended_metrics.get("confusion_matrix", {})

    confusion_df = pd.DataFrame(
        [
            [
                confusion.get("true_negative", 0),
                confusion.get("false_positive", 0),
            ],
            [
                confusion.get("false_negative", 0),
                confusion.get("true_positive", 0),
            ],
        ],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )

    diagnostic_col1, diagnostic_col2 = st.columns([1.15, 1])

    with diagnostic_col1:
        st.markdown("##### Confusion Matrix")
        st.dataframe(
            confusion_df,
            use_container_width=True,
        )

    with diagnostic_col2:
        st.markdown("##### Performance Summary")

        st.metric(
            "Precision",
            f"{recommended_metrics.get('precision', 0):.4f}",
        )
        st.metric(
            "Recall",
            f"{recommended_metrics.get('recall', 0):.4f}",
        )
        st.metric(
            "ROC-AUC",
            f"{recommended_metrics.get('roc_auc', 0):.4f}",
        )

    st.markdown("#### How to interpret the benchmark")

    st.markdown(
        """
        - **Precision** shows how many predicted positive cases were correct.
        - **Recall** shows how many actual positive cases were detected.
        - **F1** balances precision and recall.
        - **ROC-AUC** measures ranking performance across thresholds.
        - **PR-AUC** is especially useful when the positive class is rare.
        """
    )

render_header()
st.divider()

uploaded_file, df, dataset_profile, target_column, dataset_name = (
    render_configuration()
)

st.divider()

if df is None:
    render_getting_started()
else:
    current_verification_signature = (
        uploaded_file_signature(uploaded_file),
        target_column,
        dataset_name,
    )

    current_verification_result = st.session_state.get("verification_result")

    verification_unlocked = (
        current_verification_result is not None
        and st.session_state.get("verification_signature")
        == current_verification_signature
        and current_verification_result
        .get("leakage", {})
        .get("risk_count", 0)
        == 0
    )

    financial_schema = detect_financial_statement_schema(df)
    is_financial_statement = financial_schema.get(
        "is_financial_statement",
        False,
    )

    if is_financial_statement:
        workspace_options = [
            "Overview",
            "Explore",
            "Verification",
            "Financial Intelligence",
        ]
        gated_workspaces = {"Financial Intelligence"}
    else:
        workspace_options = [
            "Overview",
            "Explore",
            "Target Analysis",
            "Verification",
            "Machine Learning",
        ]
        gated_workspaces = {"Machine Learning"}

    if st.session_state.get("workspace_nav") not in workspace_options:
        st.session_state["workspace_nav"] = "Overview"

    def workspace_label(name):
        if name in gated_workspaces and not verification_unlocked:
            return f"{name} 🔒"
        return name

    workspace = st.radio(
        "Workspace",
        workspace_options,
        horizontal=True,
        label_visibility="collapsed",
        key="workspace_nav",
        format_func=workspace_label,
    )

    st.divider()

    if workspace == "Overview":
        render_overview(
            df,
            dataset_profile,
            target_column,
        )
    elif workspace == "Explore":
        render_explorer(
            df,
            dataset_profile,
        )
    elif workspace == "Target Analysis":
        render_target_analysis(
            df,
            target_column,
        )
    elif workspace == "Verification":
        render_verification(
            uploaded_file,
            df,
            target_column,
            dataset_name,
        )
    elif workspace == "Financial Intelligence":
        render_financial_intelligence(
            uploaded_file,
            df,
            target_column,
            dataset_name,
        )
    elif workspace == "Machine Learning":
        render_machine_learning(
            uploaded_file,
            df,
            target_column,
            dataset_name,
        )
