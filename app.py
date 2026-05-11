import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import DataPreprocessor
from eda import show_summary
from query_engine import (
    parse_query,
    execute_query,
    generate_query_insight
)
from ml_engine import train_prediction_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Data Analysis Automation Tool",
    layout="wide"
)

st.title("Data Analysis Automation Tool")

st.markdown(
    "Upload data, clean it, visualize it, query it, and predict insights using AI-powered analytics."
)

# =========================================================
# FILE UPLOAD
# =========================================================

file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

dataset_url = st.text_input(
    "Or Paste Dataset URL"
)

# =========================================================
# LOAD DATA
# =========================================================

raw_df = None

# ================= FILE =================

if file is not None:

    if file.name.endswith(".csv"):

        raw_df = pd.read_csv(
            file,
            encoding="latin1"
        )

    elif file.name.endswith(".xlsx"):

        raw_df = pd.read_excel(file)

# ================= URL =================

elif dataset_url:

    try:

        # CSV URL
        raw_df = pd.read_csv(dataset_url)

    except:

        try:

            # WEBSITE TABLE SCRAPING
            tables = pd.read_html(dataset_url)

            raw_df = tables[0]

        except:

            st.error(
                "Could not load dataset from URL"
            )

# =========================================================
# MAIN APP
# =========================================================

if raw_df is not None:

    # =====================================================
    # COPY FOR CLEANING
    # =====================================================

    df = raw_df.copy()

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title("Controls")

    # ================= CLEANING OPTIONS =================

    st.sidebar.subheader("Data Cleaning")

    outlier_option = st.sidebar.selectbox(
        "Handle Outliers",
        [
            "No Action",
            "Remove Outliers",
            "Cap Outliers"
        ]
    )

    missing_option = st.sidebar.selectbox(
        "Handle Missing Values",
        [
            "Mean",
            "Median",
            "Mode",
            "Drop Rows"
        ]
    )

    # =====================================================
    # PREPROCESSING
    # =====================================================

    processor = DataPreprocessor(
        df,
        outlier_option,
        missing_option
    )

    clean_df, report = processor.process()

    # =====================================================
    # SIDEBAR FILTERS
    # =====================================================

    st.sidebar.subheader("Filters")

    filtered_df = clean_df.copy()

    for col in clean_df.select_dtypes(include='object').columns:

        unique_vals = clean_df[col].dropna().unique().tolist()

        if len(unique_vals) <= 50:

            selected = st.sidebar.selectbox(
                f"Filter by {col}",
                ["All"] + unique_vals
            )

            if selected != "All":

                filtered_df = filtered_df[
                    filtered_df[col] == selected
                ]

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Dashboard",
            "Visualization Studio",
            "Query Engine",
            "ML Prediction"
        ]
    )

    # =====================================================
    # DASHBOARD TAB
    # =====================================================

    with tab1:

        st.subheader("Dashboard Overview")

        numeric_cols = filtered_df.select_dtypes(
            include='number'
        ).columns.tolist()

        # ================= KPI METRICS =================

        if len(numeric_cols) > 0:

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Rows",
                len(filtered_df)
            )

            col2.metric(
                "Total Columns",
                filtered_df.shape[1]
            )

            col3.metric(
                "Missing Values",
                int(filtered_df.isnull().sum().sum())
            )

            col4.metric(
                "Avg Value",
                round(filtered_df[numeric_cols[0]].mean(), 2)
            )

        st.divider()

        # ================= RAW DATA =================

        st.subheader("Raw Data Preview")

        rows_to_show = st.slider(
            "Select number of rows",
            5,
            50,
            10
        )

        st.dataframe(
            raw_df.head(rows_to_show),
            use_container_width=True
        )

        st.divider()

        # ================= CLEANING REPORT =================

        st.subheader("Data Cleaning Report")

        for r in report:

            st.success(r)

        st.divider()

        # ================= CLEANED DATA =================

        st.subheader("Cleaned Data Preview")

        st.dataframe(
            clean_df.head(rows_to_show),
            use_container_width=True
        )

        # ================= DOWNLOAD =================

        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=clean_df.to_csv(index=False).encode('utf-8'),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # =====================================================
    # VISUALIZATION STUDIO
    # =====================================================

    with tab2:

        st.subheader("Visualization Studio")

        all_columns = filtered_df.columns.tolist()

        numeric_cols = filtered_df.select_dtypes(
            include='number'
        ).columns.tolist()

        categorical_cols = filtered_df.select_dtypes(
            exclude='number'
        ).columns.tolist()

        chart_type = st.selectbox(

            "Choose Chart Type",

            [
                "Bar Chart",
                "Line Chart",
                "Scatter Plot",
                "Histogram",
                "Box Plot",
                "Pie Chart",
                "Correlation Heatmap"
            ]
        )

        # =================================================
        # BAR CHART
        # =================================================

        if chart_type == "Bar Chart":

            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox(
                    "Select X-axis",
                    categorical_cols
                )

            with col2:
                y_col = st.selectbox(
                    "Select Y-axis",
                    numeric_cols
                )

            fig = px.bar(
                filtered_df,
                x=x_col,
                y=y_col,
                color=x_col,
                title=f"{y_col} by {x_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # LINE CHART
        # =================================================

        elif chart_type == "Line Chart":

            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox(
                    "Select X-axis",
                    all_columns
                )

            with col2:
                y_col = st.selectbox(
                    "Select Y-axis",
                    numeric_cols
                )

            fig = px.line(
                filtered_df,
                x=x_col,
                y=y_col,
                title=f"{y_col} Trend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # SCATTER PLOT
        # =================================================

        elif chart_type == "Scatter Plot":

            col1, col2, col3 = st.columns(3)

            with col1:
                x_col = st.selectbox(
                    "Select X-axis",
                    numeric_cols
                )

            with col2:
                y_col = st.selectbox(
                    "Select Y-axis",
                    numeric_cols,
                    index=1 if len(numeric_cols) > 1 else 0
                )

            with col3:
                color_col = st.selectbox(
                    "Color By",
                    categorical_cols
                )

            fig = px.scatter(
                filtered_df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{x_col} vs {y_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )