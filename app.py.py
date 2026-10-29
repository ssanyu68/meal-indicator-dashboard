"""MEAL Indicator Performance and Data Quality Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="MEAL Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

STATUS_COLORS = {
    "On Track": "green",
    "Needs Improvement": "orange",
    "Off Track": "red",
    "Review Required": "gray",
    "Data Unavailable": "black"
}

REQUIRED_COLUMNS = [
    "indicator_name",
    "reporting_period",
    "quarterly_target",
    "actual",
    "annual_target"
]

# --------------------------------------------------
# Sample dataset used when no file is uploaded
# --------------------------------------------------
def create_sample_data():
    indicators = {
        "Children enrolled in school": {
            "annual_target": 2000,
            "targets": [500, 500, 500, 500],
            "actuals": [450, 480, 520, 490]
        },
        "Households receiving support": {
            "annual_target": 1200,
            "targets": [300, 300, 300, 300],
            "actuals": [315, 280, 310, 320]
        },
        "Farmers trained": {
            "annual_target": 800,
            "targets": [200, 200, 200, 200],
            "actuals": [120, 180, 210, 190]
        },
        "Water points rehabilitated": {
            "annual_target": 80,
            "targets": [20, 20, 20, 20],
            "actuals": [18, 22, 15, 20]
        },
        "Children receiving nutrition services": {
            "annual_target": 1600,
            "targets": [400, 400, 400, 400],
            "actuals": [380, 410, 390, 420]
        },
        "Teachers trained": {
            "annual_target": 400,
            "targets": [100, 100, 100, 100],
            "actuals": [95, 105, 90, 110]
        },
        "Households accessing clean water": {
            "annual_target": 1000,
            "targets": [250, 250, 250, 250],
            "actuals": [200, 240, 260, 230]
        },
        "Community groups established": {
            "annual_target": 60,
            "targets": [15, 15, 15, 15],
            "actuals": [12, 16, 14, 18]
        }
    }

    records = []
    for name, details in indicators.items():
        for i, quarter in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            records.append({
                "indicator_name": name,
                "reporting_period": quarter,
                "quarterly_target": details["targets"][i],
                "actual": details["actuals"][i],
                "annual_target": details["annual_target"]
            })

    return pd.DataFrame(records)


def load_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def classify_performance(value):
    if pd.isna(value) or not np.isfinite(value):
        return "Data Unavailable"
    if value >= 90:
        return "On Track"
    if value >= 70:
        return "Needs Improvement"
    return "Off Track"


def prepare_data(raw_data):
    data = raw_data.copy()
    issues = []

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in data.columns
    ]

    if missing_columns:
        return None, [
            "Missing required columns: " + ", ".join(missing_columns)
        ]

    # Check missing values in required fields
    for col in REQUIRED_COLUMNS:
        missing_count = int(data[col].isna().sum())
        if missing_count:
            issues.append(
                f"{col}: {missing_count} missing value(s)"
            )

    # Check duplicate indicator-quarter records
    duplicates = data.duplicated(
        subset=["indicator_name", "reporting_period"],
        keep=False
    )
    if duplicates.any():
        issues.append(
            f"{int(duplicates.sum())} row(s) belong to duplicate "
            "indicator-quarter records"
        )

    # Convert targets and actuals to numeric
    numeric_columns = [
        "quarterly_target", "actual", "annual_target"
    ]

    for col in numeric_columns:
        original = data[col].copy()
        converted = pd.to_numeric(original, errors="coerce")

        invalid_count = int(
            (original.notna() & converted.isna()).sum()
        )
        if invalid_count:
            issues.append(
                f"{col}: {invalid_count} non-numeric value(s)"
            )

        data[col] = converted

        nonfinite = data[col].notna() & ~np.isfinite(data[col])
        if nonfinite.any():
            issues.append(
                f"{col}: {int(nonfinite.sum())} non-finite value(s)"
            )
            data.loc[nonfinite, col] = np.nan

        negative_count = int((data[col] < 0).sum())
        if negative_count:
            issues.append(
                f"{col}: {negative_count} negative value(s)"
            )

    for col in ["quarterly_target", "annual_target"]:
        zero_count = int((data[col] == 0).sum())
        if zero_count:
            issues.append(
                f"{col}: {zero_count} zero target value(s)"
            )

    invalid_period = ~data["reporting_period"].isin(
        ["Q1", "Q2", "Q3", "Q4"]
    )
    if invalid_period.any():
        issues.append(
            f"{int(invalid_period.sum())} invalid reporting period(s)"
        )

    blank_names = (
        data["indicator_name"].isna()
        | data["indicator_name"].astype(str).str.strip().eq("")
    )
    if blank_names.any():
        issues.append(
            f"{int(blank_names.sum())} missing or blank indicator name(s)"
        )

    # Check annual target consistency within each indicator
    inconsistent = data.groupby("indicator_name")[
        "annual_target"
    ].nunique(dropna=True)
    inconsistent_names = inconsistent[inconsistent > 1]

    if len(inconsistent_names):
        issues.append(
            f"{len(inconsistent_names)} indicator(s) have inconsistent "
            "annual targets"
        )

    # Exclude records unsuitable for performance calculations
    valid = (
        ~blank_names
        & data["reporting_period"].isin(["Q1", "Q2", "Q3", "Q4"])
        & data[numeric_columns].notna().all(axis=1)
        & data["quarterly_target"].gt(0)
        & data["annual_target"].gt(0)
        & data["actual"].ge(0)
        & data["quarterly_target"].ge(0)
        & data["annual_target"].ge(0)
    )

    clean = data.loc[valid].copy()

    # Avoid double-counting duplicate indicator-quarter records
    clean = clean.drop_duplicates(
        subset=["indicator_name", "reporting_period"],
        keep=False
    )

    return clean, issues


def analyse_data(data):
    quarterly = data.copy()

    quarterly["achievement_pct"] = (
        quarterly["actual"] / quarterly["quarterly_target"]
    ) * 100

    quarterly["performance_gap"] = (
        quarterly["quarterly_target"] - quarterly["actual"]
    )

    quarterly["performance_status"] = (
        quarterly["achievement_pct"].apply(classify_performance)
    )

    annual = (
        data.groupby("indicator_name", as_index=False)
        .agg(
            annual_target=("annual_target", "first"),
            annual_actual=("actual", "sum"),
            reported_quarters=("reporting_period", "nunique"),
            target_versions=("annual_target", "nunique")
        )
    )

    annual["data_status"] = np.where(
        (annual["reported_quarters"] == 4)
        & (annual["target_versions"] == 1),
        "Complete",
        "Review Required"
    )

    annual["annual_achievement_pct"] = (
        annual["annual_actual"] / annual["annual_target"]
    ) * 100

    annual["annual_performance_gap"] = (
        annual["annual_target"] - annual["annual_actual"]
    )

    annual["performance_status"] = (
        annual["annual_achievement_pct"].apply(classify_performance)
    )

    annual.loc[
        annual["data_status"] != "Complete",
        "performance_status"
    ] = "Review Required"

    return quarterly, annual


# --------------------------------------------------
# Sidebar: dataset selection and navigation
# --------------------------------------------------
st.sidebar.title("MEAL Dashboard")
st.sidebar.caption("Indicator Performance and Data Quality")

uploaded_file = st.sidebar.file_uploader(
    "Upload indicator data",
    type=["csv", "xlsx"]
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Quarterly Performance",
        "Annual Performance",
        "Data Quality",
        "Follow-up Actions",
        "Download Reports"
    ]
)

if uploaded_file is None:
    raw_data = create_sample_data()
    st.sidebar.info(
        "Showing illustrative sample data. Upload a CSV or Excel file "
        "to analyse your own data."
    )
else:
    try:
        raw_data = load_uploaded_file(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the uploaded file: {error}")
        st.stop()

data, quality_issues = prepare_data(raw_data)

st.title("MEAL Indicator Performance Dashboard")
st.caption(
    "Monitor indicator achievement, identify data-quality concerns, "
    "and support evidence-based programme decisions."
)

if data is None:
    st.error("The uploaded dataset does not have the required columns.")
    st.write("Required columns:", REQUIRED_COLUMNS)
    st.stop()

if data.empty:
    st.error(
        "No valid records are available for analysis. "
        "Review the uploaded data and required columns."
    )
    st.stop()

quarterly, annual = analyse_data(data)

follow_up = quarterly[
    quarterly["performance_status"].isin(
        ["Needs Improvement", "Off Track"]
    )
].copy()

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    complete_annual = annual[
        annual["data_status"] == "Complete"
    ]

    average_annual = (
        complete_annual["annual_achievement_pct"].mean()
        if not complete_annual.empty else np.nan
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Indicators monitored",
        int(data["indicator_name"].nunique())
    )
    col2.metric(
        "Quarterly records",
        int(len(quarterly))
    )
    col3.metric(
        "Average annual achievement",
        f"{average_annual:.1f}%"
        if pd.notna(average_annual) else "N/A"
    )
    col4.metric(
        "Records requiring follow-up",
        int(len(follow_up))
    )

    st.subheader("Quarterly performance trend")

    quarterly_summary = (
        quarterly.groupby("reporting_period", as_index=False)
        .agg(
            average_achievement_pct=("achievement_pct", "mean")
        )
    )

    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    quarterly_summary["reporting_period"] = pd.Categorical(
        quarterly_summary["reporting_period"],
        categories=quarter_order,
        ordered=True
    )
    quarterly_summary = quarterly_summary.sort_values(
        "reporting_period"
    )

    fig = px.line(
        quarterly_summary,
        x="reporting_period",
        y="average_achievement_pct",
        markers=True,
        title="Average Achievement by Quarter",
        labels={
            "reporting_period": "Quarter",
            "average_achievement_pct": "Average Achievement (%)"
        }
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quarterly performance status")

    status_summary = (
        quarterly.groupby(
            ["reporting_period", "performance_status"]
        )
        .size()
        .reset_index(name="number_of_records")
    )

    fig = px.bar(
        status_summary,
        x="reporting_period",
        y="number_of_records",
        color="performance_status",
        barmode="stack",
        color_discrete_map=STATUS_COLORS,
        category_orders={
            "reporting_period": quarter_order,
            "performance_status": [
                "On Track", "Needs Improvement", "Off Track"
            ]
        },
        labels={
            "reporting_period": "Quarter",
            "number_of_records": "Number of Records",
            "performance_status": "Performance Status"
        }
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Annual performance snapshot")
    st.dataframe(
        annual[
            [
                "indicator_name",
                "annual_target",
                "annual_actual",
                "annual_achievement_pct",
                "data_status",
                "performance_status"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# Quarterly Performance
# --------------------------------------------------
elif page == "Quarterly Performance":
    st.subheader("Quarterly Indicator Performance")

    selected_quarters = st.multiselect(
        "Select reporting quarters",
        ["Q1", "Q2", "Q3", "Q4"],
        default=["Q1", "Q2", "Q3", "Q4"]
    )

    view = quarterly[
        quarterly["reporting_period"].isin(selected_quarters)
    ].copy()

    fig = px.bar(
        view,
        x="indicator_name",
        y="achievement_pct",
        color="performance_status",
        color_discrete_map=STATUS_COLORS,
        title="Quarterly Achievement by Indicator",
        hover_data=[
            "reporting_period",
            "quarterly_target",
            "actual",
            "performance_gap"
        ],
        labels={
            "indicator_name": "Indicator",
            "achievement_pct": "Achievement (%)",
            "performance_status": "Status"
        }
    )
    fig.add_hline(
        y=90,
        line_dash="dash",
        line_color="green",
        annotation_text="90% threshold"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_tickangle=-35
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        view[
            [
                "indicator_name", "reporting_period",
                "quarterly_target", "actual", "achievement_pct",
                "performance_gap", "performance_status"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# Annual Performance
# --------------------------------------------------
elif page == "Annual Performance":
    st.subheader("Annual Indicator Performance")

    fig = px.bar(
        annual,
        x="indicator_name",
        y="annual_achievement_pct",
        color="performance_status",
        color_discrete_map=STATUS_COLORS,
        hover_data=[
            "annual_target",
            "annual_actual",
            "annual_performance_gap",
            "data_status"
        ],
        title="Annual Achievement Against Targets",
        labels={
            "indicator_name": "Indicator",
            "annual_achievement_pct": "Achievement (%)",
            "performance_status": "Status"
        }
    )
    fig.add_hline(
        y=90,
        line_dash="dash",
        line_color="green",
        annotation_text="90% threshold"
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_tickangle=-35
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        annual.drop(columns=["target_versions"]).round(2),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# Data Quality
# --------------------------------------------------
elif page == "Data Quality":
    st.subheader("Data Quality Checks")

    col1, col2 = st.columns(2)
    col1.metric("Uploaded records", len(raw_data))
    col2.metric("Records available for analysis", len(data))

    if quality_issues:
        st.warning(
            "Potential data-quality issues were detected. "
            "Review the findings below before interpreting results."
        )
        for issue in quality_issues:
            st.write(f"- {issue}")
    else:
        st.success(
            "No issues detected by the configured data-quality checks."
        )

    st.markdown(
        "Checks include missing values, duplicate indicator-quarter "
        "records, invalid periods, invalid numeric values, non-positive "
        "targets, negative values, and inconsistent annual targets."
    )

    st.subheader("Data used for analysis")
    st.dataframe(data, use_container_width=True, hide_index=True)

# --------------------------------------------------
# Follow-up Actions
# --------------------------------------------------
elif page == "Follow-up Actions":
    st.subheader("Indicators Requiring Follow-up")

    if follow_up.empty:
        st.success(
            "No quarterly indicators are classified as Needs Improvement "
            "or Off Track in the records available for analysis."
        )
    else:
        st.write(
            "Review these records and document any action taken."
        )
        st.dataframe(
            follow_up[
                [
                    "indicator_name", "reporting_period",
                    "quarterly_target", "actual", "achievement_pct",
                    "performance_gap", "performance_status"
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# Download Reports
# --------------------------------------------------
elif page == "Download Reports":
    st.subheader("Download MEAL Reports")

    quarterly_csv = quarterly.to_csv(index=False).encode("utf-8")
    annual_csv = annual.to_csv(index=False).encode("utf-8")
    follow_up_csv = follow_up.to_csv(index=False).encode("utf-8")

    summary = pd.DataFrame([
        ["Indicators monitored", data["indicator_name"].nunique()],
        ["Quarterly records analysed", len(quarterly)],
        [
            "Quarterly records On Track",
            int((quarterly["performance_status"] == "On Track").sum())
        ],
        [
            "Quarterly records Needs Improvement",
            int((quarterly["performance_status"] == "Needs Improvement").sum())
        ],
        [
            "Quarterly records Off Track",
            int((quarterly["performance_status"] == "Off Track").sum())
        ],
        ["Data-quality issues detected", len(quality_issues)]
    ], columns=["Measure", "Result"])

    summary_csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Quarterly Report",
        quarterly_csv,
        file_name="quarterly_meal_report.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Annual Report",
        annual_csv,
        file_name="annual_meal_report.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Follow-up Report",
        follow_up_csv,
        file_name="meal_follow_up_report.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Overall Summary",
        summary_csv,
        file_name="meal_overall_summary.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption(
    "Performance rules: On Track ≥90%; Needs Improvement ≥70% "
    "and <90%; Off Track <70%."
)
st.caption(
    "Demonstration dashboard. The performance thresholds are project-defined "
    "and should be agreed with programme stakeholders before operational use."
)
