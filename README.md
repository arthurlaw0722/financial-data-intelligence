# Financial Data Intelligence

A financial analysis application that retrieves standardized annual 10-K financial data from SEC EDGAR, verifies the underlying dataset, surfaces multi-year financial signals, and supports evidence-grounded investigation with an AI Financial Analyst.

**Live Application:**  
https://financial-data-intelligence.streamlit.app

---

## Overview

Financial statements contain much more information than a single year's revenue or profit figure.

The challenge is turning raw filing data into a structured analytical workflow that can answer questions such as:

- Is the underlying financial dataset reliable?
- Are revenue, earnings, and operating cash flow moving consistently?
- Is working capital showing unusual behaviour?
- Is liquidity or leverage becoming a concern?
- Is a recent movement unusual relative to the company's own history?
- What additional evidence would an analyst need before drawing a stronger conclusion?

Financial Data Intelligence combines deterministic financial analysis with an AI explanation layer while keeping the supporting evidence visible.

---

## Workflow

```text
US-listed company ticker
        ↓
SEC EDGAR Company Facts
        ↓
Standardized annual 10-K financial dataset
        ↓
Financial Data Verification
        ↓
Multi-Year Financial Intelligence
        ↓
Deterministic Financial Signals
        ↓
AI Financial Analyst
        ↓
Evidence-Grounded Analyst Review
```

---

## Core Features

### 1. SEC EDGAR Integration

Enter a US-listed company ticker to retrieve and standardize annual financial statement data from SEC EDGAR Company Facts.

The pipeline supports up to 10 fiscal years of history and standardizes key financial metrics including:

- Revenue
- Net income
- Operating cash flow
- Accounts receivable
- Inventory
- Cash and equivalents
- Total assets
- Total liabilities
- Total equity
- Current assets
- Current liabilities
- Total debt

The SEC mapping layer handles multiple XBRL concepts and converts them into a consistent analytical schema.

---

### 2. Financial Data Verification

Before financial analysis begins, the application verifies the retrieved dataset.

Checks include:

- Dataset completeness
- Duplicate-row detection
- Financial statement structure
- Standardized metric coverage
- Dataset trust assessment
- Decision-readiness guidance
- SHA256 dataset fingerprint
- Canonical report hash

This creates a verifiable evidence layer before downstream financial analysis is performed.

---

### 3. Multi-Year Financial Intelligence

The application analyses both recent movements and longer-term financial patterns.

It evaluates four core analytical areas.

#### Accounting Integrity

Checks whether the accounting relationship between assets, liabilities, and equity reconciles within the configured tolerance.

#### Working Capital

Compares movements in:

- Revenue
- Accounts receivable
- Inventory

This helps identify situations where receivables or inventory are moving materially differently from revenue.

#### Earnings Quality

Compares net income growth with operating cash flow growth to identify potential earnings and cash-flow divergence.

#### Liquidity & Leverage

Evaluates indicators including:

- Current ratio
- Debt / assets
- Debt / equity
- Cash / debt
- Debt growth
- Cash growth

These are general analytical screening rules rather than company-specific covenant tests.

---

## Historical Context

A large one-year movement does not automatically imply structural deterioration.

The system therefore compares recent movements with the company's historical financial record.

Historical analysis includes:

- Latest year-over-year growth
- Recent 3-year CAGR
- Full-period CAGR
- Historical range
- Historical median
- Recent step-up or step-down patterns

This helps distinguish between a significant recent movement and a genuine break from historical behaviour.

---

## Executive Assessment

Financial signals are translated into an executive-level analytical view.

The application provides:

- Overall assessment
- Principal financial watchpoint
- Cross-statement posture
- Multi-year momentum
- Next-year historical scenario
- Review posture
- Balance-sheet posture
- Supporting evidence
- Recommended analyst next step

Signal statuses include:

- `Normal`
- `Review`
- `High Attention`

These statuses are analytical screening outputs, not investment recommendations.

---

## Working Capital Diagnostic

When a material working-capital signal is detected, the application performs a deeper diagnostic.

For example, if inventory growth materially exceeds revenue growth, the system can evaluate:

- Inventory vs. revenue growth divergence
- Inventory intensity
- Historical inventory range
- Historical median
- Receivables alignment
- Operating cash-flow generation
- Evidence still required for further review

This helps distinguish a concentrated working-capital issue from broader financial deterioration.

---

## Financial Position Analysis

The application also evaluates the latest balance-sheet position using financial ratios including:

- Current ratio
- Debt / assets
- Debt / equity
- Cash / debt

These ratios support the liquidity and leverage assessment and provide additional context for interpreting other financial signals.

---

## Next Fiscal Year Outlook

The application generates a moderated historical scenario for:

- Revenue
- Net income
- Operating cash flow

For each metric, it displays:

- Latest value
- Direction
- Moderated growth assumption
- Base outlook
- Indicative range

The outlook is derived from historical financial trends.

It is **not company guidance and not an investment forecast**.

---

## AI Financial Analyst

The application includes a Gemini-powered AI Financial Analyst.

Users can ask questions about the financial analysis currently displayed, for example:

> What is the main financial watchpoint and what evidence supports it?

The AI receives structured context generated by the deterministic financial intelligence engine, including:

- Overall assessment
- Financial signals
- Historical context
- Diagnostic evidence
- Cross-statement reasoning
- Balance-sheet context
- Evidence gaps

The AI layer is designed to explain and investigate the existing analysis rather than independently invent financial conclusions.

When the available SEC data is insufficient to support a stronger conclusion, the analyst identifies the additional evidence required for further review.

---

## Deterministic Analysis + Generative AI

The project deliberately separates two analytical layers.

### Deterministic Financial Engine

Financial calculations, ratios, historical comparisons, thresholds, and signal classifications are calculated directly from structured financial data.

### AI Explanation Layer

The language model receives the deterministic analysis as structured context and converts it into an interactive analyst-style explanation.

This design reduces the risk of allowing a generative AI model to replace the underlying financial logic.

The workflow is therefore:

```text
Retrieve
→ Standardize
→ Verify
→ Calculate
→ Detect
→ Contextualize
→ Explain
```

---

## Evidence and Auditability

Each verified dataset produces cryptographic evidence including:

- SHA256 dataset fingerprint
- Canonical verification report hash

This provides a reproducible link between the financial data analysed and the verification output produced by the application.

---

## Technology Stack

- Python
- Streamlit
- pandas
- NumPy
- SEC EDGAR Company Facts API
- XBRL financial concepts
- Google Gemini
- SHA256 cryptographic hashing
- Git / GitHub

---

## Project Structure

```text
app/
    streamlit_app.py

analytics/
    financial_intelligence.py
    financial_statement_adapter.py
    financial_context.py
    ai_financial_analyst.py
    profiler.py
    statistics.py
    visualizations.py

data_sources/
    sec_edgar.py
    sec_financial_mapper.py

agent/
    analyzer.py
    scoring.py
    readiness.py
```

---

## Running Locally

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables:

```bash
export SEC_USER_AGENT="Your Name your-email@example.com"
export GEMINI_API_KEY="your_gemini_api_key"
```

Then run:

```bash
streamlit run app/streamlit_app.py
```

Open the local Streamlit URL and enter a US-listed company ticker such as:

```text
NVDA
AAPL
GOOGL
MSFT
```

---

## Application Workflow

1. Enter a **US-listed company ticker**.
2. Retrieve standardized annual 10-K financial data from **SEC EDGAR**.
3. Review **Overview** and **Explore** to inspect the standardized financial statements and multi-year history.
4. Open **Verification** to assess data quality, financial statement structure, trust, and SHA256 integrity.
5. Open **Financial Intelligence** to review accounting integrity, working capital, earnings quality, liquidity & leverage, multi-year trends, and the next-year historical scenario.
6. Use the **AI Financial Analyst** to investigate questions grounded in verified financial analysis and supporting evidence.
7. Review the **SHA256 Proof & Report** for the verification record and evidence trail.

---

## Design Principle

The purpose of this project is not to send financial numbers directly to a language model and ask it to generate an unsupported opinion.

Instead, financial data first passes through structured retrieval, standardization, verification, deterministic calculations, historical contextualization, and signal detection.

Generative AI is then used as an interactive explanation and investigation layer on top of that evidence.

This keeps the core financial logic inspectable while still allowing users to interact naturally with the analysis.

---

## Limitations

The application currently focuses on standardized annual SEC Company Facts data.

Important limitations include:

- Financial signals use general analytical heuristics rather than industry-specific thresholds.
- SEC XBRL concepts may differ between issuers or reporting periods.
- Standardized Company Facts data does not contain all qualitative information available in a full 10-K filing.
- Management commentary, segment disclosures, debt maturity schedules, inventory ageing, cost of revenue, covenant details, and other contextual evidence may be required for a complete review.
- Historical relationships do not necessarily continue into future periods.
- The next-year outlook is a scenario based on historical financial data rather than a prediction of future company performance.
- AI-generated explanations remain grounded in the available analytical context and should not replace professional financial analysis.

The application is designed for analytical exploration and portfolio demonstration and does not provide investment advice.

---

## Author

**Arthur Law**

BSc Finance and Technology — **First Class Honours**  
MSc Business Analytics — **University College London**

Interests include:

- Financial Technology
- Data & AI Applications
- Financial Analytics
- Digital Products
- Technology Transformation
