import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Employee Workforce Segmentation",
    page_icon="📊",
    layout="wide"
)

RESULTS_DIR = "employee_workforce_segmentation/results"

st.title("Employee Workforce Segmentation")
st.caption(
    "Workforce analytics | Clustering | PCA | Department insights | Retention"
)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

results_path = os.path.join(
    RESULTS_DIR, "employee_segmentation_results.csv"
)
pca_path = os.path.join(
    RESULTS_DIR, "pca_employee_visualization.csv"
)
comparison_path = os.path.join(
    RESULTS_DIR, "clustering_model_comparison.csv"
)

if not os.path.exists(results_path):
    st.error(
        "Employee segmentation results were not found. "
        "Place employee_segmentation_results.csv inside "
        "employee_workforce_segmentation/results/."
    )
    st.stop()

df = load_csv(results_path)

if df.empty:
    st.error("The employee segmentation file is empty.")
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

if "Department" in df.columns:
    departments = sorted(
        df["Department"].dropna().astype(str).unique().tolist()
    )
    selected_departments = st.sidebar.multiselect(
        "Select department",
        departments,
        default=departments
    )
    filtered = df[
        df["Department"].astype(str).isin(selected_departments)
    ].copy()
else:
    filtered = df.copy()
    st.sidebar.info("Department field is not available.")

cluster_options = []
if "KMeans_Cluster" in filtered.columns:
    cluster_options = sorted(
        filtered["KMeans_Cluster"].dropna().unique().tolist()
    )

if cluster_options:
    selected_clusters = st.sidebar.multiselect(
        "Select K-Means clusters",
        cluster_options,
        default=cluster_options
    )
    filtered = filtered[
        filtered["KMeans_Cluster"].isin(selected_clusters)
    ].copy()

if filtered.empty:
    st.warning("No employees match the selected filters.")
    st.stop()

# KPI metrics
st.header("Workforce Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(filtered):,}")

if "Department" in filtered.columns:
    c2.metric("Departments", filtered["Department"].nunique())
else:
    c2.metric("Departments", "N/A")

if "KMeans_Cluster" in filtered.columns:
    c3.metric("K-Means Clusters", filtered["KMeans_Cluster"].nunique())
else:
    c3.metric("K-Means Clusters", "N/A")

if "MonthlyIncome" in filtered.columns:
    c4.metric(
        "Average Monthly Income",
        f"{filtered['MonthlyIncome'].mean():,.0f}"
    )
else:
    c4.metric("Average Monthly Income", "N/A")

# Cluster distribution
st.header("Employee Segmentation")

left, right = st.columns(2)

with left:
    if "KMeans_Cluster" in filtered.columns:
        cluster_counts = (
            filtered["KMeans_Cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Employees")
        )
        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Employees",
            title="K-Means Cluster Distribution",
            text="Employees"
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    if "Hierarchical_Cluster" in filtered.columns:
        hierarchy_counts = (
            filtered["Hierarchical_Cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Employees")
        )
        fig = px.bar(
            hierarchy_counts,
            x="Cluster",
            y="Employees",
            title="Hierarchical Cluster Distribution",
            text="Employees"
        )
        st.plotly_chart(fig, use_container_width=True)

# Department segmentation
if "Department" in filtered.columns and "KMeans_Cluster" in filtered.columns:
    st.header("Department-Level Workforce Segments")

    department_counts = pd.crosstab(
        filtered["Department"],
        filtered["KMeans_Cluster"]
    ).reset_index()

    department_long = department_counts.melt(
        id_vars="Department",
        var_name="Cluster",
        value_name="Employees"
    )

    fig = px.bar(
        department_long,
        x="Department",
        y="Employees",
        color="Cluster",
        barmode="stack",
        title="Department Distribution by K-Means Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

# PCA visualization
st.header("PCA Cluster Visualization")

if os.path.exists(pca_path):
    pca_df = load_csv(pca_path)

    if {"PC1", "PC2"}.issubset(pca_df.columns):
        label_col = "KMeans_Cluster"
        if label_col not in pca_df.columns:
            label_col = "Hierarchical_Cluster"

        if label_col in pca_df.columns:
            # Align PCA rows to the employee table by original row order.
            # This works when both exports retain the same row order.
            if len(pca_df) == len(df):
                selected_positions = df.index[
                    df.index.isin(filtered.index)
                ]
                pca_filtered = pca_df.iloc[
                    selected_positions.tolist()
                ].copy()
            else:
                pca_filtered = pca_df.copy()

            fig = px.scatter(
                pca_filtered,
                x="PC1",
                y="PC2",
                color=label_col,
                title="Employees Projected onto Two Principal Components",
                opacity=0.75
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "PCA is a two-dimensional projection of the processed "
                "features. Overlapping points do not necessarily indicate "
                "identical employees."
            )
        else:
            st.info("Cluster labels were not found in the PCA file.")
    else:
        st.info("The PCA file must contain PC1 and PC2 columns.")
else:
    st.info("PCA results file is not available.")

# Performance and satisfaction
profile_cols = [
    "PerformanceRating",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "YearsAtCompany",
    "MonthlyIncome"
]
available_profile_cols = [
    col for col in profile_cols if col in filtered.columns
]

if available_profile_cols and "KMeans_Cluster" in filtered.columns:
    st.header("Performance and Employee Characteristics")
    profile = (
        filtered.groupby("KMeans_Cluster")[available_profile_cols]
        .mean()
        .round(2)
    )
    st.dataframe(profile, use_container_width=True)

# Historical attrition
if "Attrition" in filtered.columns:
    st.header("Historical Attrition Analysis")

    attrition = (
        filtered["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1, "no": 0,
            "true": 1, "false": 0,
            "1": 1, "0": 0
        })
    )

    known = filtered.assign(Attrition_Flag=attrition).dropna(
        subset=["Attrition_Flag"]
    )

    if not known.empty:
        overall_rate = known["Attrition_Flag"].mean() * 100
        st.metric("Historical Attrition Rate", f"{overall_rate:.2f}%")

        if "KMeans_Cluster" in known.columns:
            rates = (
                known.groupby("KMeans_Cluster")["Attrition_Flag"]
                .mean()
                .mul(100)
                .reset_index(name="Historical Attrition (%)")
            )
            fig = px.bar(
                rates,
                x="KMeans_Cluster",
                y="Historical Attrition (%)",
                title="Historical Attrition by Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "These are historical group-level statistics, not predictions "
            "of individual employees' future behavior."
        )
    else:
        st.info("No recognizable attrition labels were found.")

# Model comparison
if os.path.exists(comparison_path):
    st.header("Clustering Model Comparison")
    comparison = load_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True)

# Employee data and download
st.header("Employee Segmentation Results")
st.dataframe(filtered, use_container_width=True)

st.download_button(
    "Download Filtered Employee Data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="employee_segmentation_results.csv",
    mime="text/csv"
)

st.success("Dashboard data loaded successfully.")
