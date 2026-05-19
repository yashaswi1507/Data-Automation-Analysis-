# =========================================================
# IMPORTS
# =========================================================

import os
import io
import glob
import zipfile
import requests
import kagglehub
import streamlit as st
import pandas as pd
import plotly.express as px

from bs4 import BeautifulSoup

from eda import show_summary

from preprocessing import DataPreprocessor

from query_engine import (
    parse_query,
    execute_query,
    generate_query_insight
)

from ml_engine import train_prediction_model

from dashboard_generator import (
    generate_kpis,
    generate_auto_charts,
    generate_insights
)

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
    type=[
        "csv",
        "xlsx",
        "json",
        "zip"
    ]
)

dataset_url = st.text_input(
    "Or Paste Dataset URL / API / Kaggle Dataset"
)

# =========================================================
# LOAD DATA
# =========================================================

raw_df = None

# =========================================================
# SAFE CSV LOADER
# =========================================================

def load_csv_safely(file):

    encodings = [
        "utf-8",
        "latin1",
        "ISO-8859-1",
        "cp1252"
    ]

    separators = [
        ",",
        ";",
        "\t"
    ]

    for enc in encodings:

        for sep in separators:

            try:

                file.seek(0)

                df = pd.read_csv(
                    file,
                    encoding=enc,
                    sep=sep,
                    on_bad_lines="skip"
                )

                if len(df.columns) > 1:

                    return df

            except:
                continue

    return None

# =========================================================
# UNIVERSAL DATA LOADER
# =========================================================

def universal_data_loader(source):

    # =====================================================
    # KAGGLE DATASET
    # =====================================================

    try:

        if "kaggle.com/datasets/" in source:

            parts = source.split("/datasets/")[1]

            dataset_path = parts.split("?")[0]

            path = kagglehub.dataset_download(
                dataset_path
            )

            csv_files = glob.glob(
                os.path.join(path, "*.csv")
            )

            if len(csv_files) > 0:

                return pd.read_csv(
                    csv_files[0]
                )

            xlsx_files = glob.glob(
                os.path.join(path, "*.xlsx")
            )

            if len(xlsx_files) > 0:

                return pd.read_excel(
                    xlsx_files[0]
                )

    except:
        pass

    # =====================================================
    # CSV URL
    # =====================================================

    try:

        if ".csv" in source:

            return pd.read_csv(source)

    except:
        pass

    # =====================================================
    # EXCEL URL
    # =====================================================

    try:

        if ".xlsx" in source:

            return pd.read_excel(source)

    except:
        pass

    # =====================================================
    # JSON API
    # =====================================================

    try:

        response = requests.get(source)

        data = response.json()

        return pd.DataFrame(data)

    except:
        pass

    # =====================================================
    # HTML TABLES
    # =====================================================

    try:

        tables = pd.read_html(source)

        if len(tables) > 0:

            return tables[0]

    except:
        pass

    # =====================================================
    # ZIP URL
    # =====================================================

    try:

        if ".zip" in source:

            response = requests.get(source)

            zip_file = zipfile.ZipFile(
                io.BytesIO(response.content)
            )

            for file_name in zip_file.namelist():

                if file_name.endswith(".csv"):

                    with zip_file.open(file_name) as f:

                        return pd.read_csv(f)

    except:
        pass

    # =====================================================
    # WEB SCRAPING
    # =====================================================

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0"
        }

        response = requests.get(
            source,
            headers=headers,
            timeout=15
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        scraped_data = []

        for p in paragraphs[:100]:

            text = p.get_text().strip()

            if text:

                scraped_data.append(
                    {"Text": text}
                )

        if len(scraped_data) > 0:

            return pd.DataFrame(
                scraped_data
            )

    except:
        pass

    return None

# =========================================================
# CLEAN MISSING VALUES
# =========================================================

def clean_missing_values(df):

    df.replace(
        [
            "?",
            "NA",
            "N/A",
            "null",
            "NULL",
            "",
            "none",
            "None",
            "NaN",
            "nan"
        ],
        pd.NA,
        inplace=True
    )

    return df

# =========================================================
# FILE INPUT
# =========================================================

if file is not None:

    # =====================================================
    # CSV
    # =====================================================

    if file.name.endswith(".csv"):

        raw_df = load_csv_safely(file)

    # =====================================================
    # EXCEL
    # =====================================================

    elif file.name.endswith(".xlsx"):

        try:

            raw_df = pd.read_excel(file)

        except:

            st.error(
                "Unable to read Excel file"
            )

    # =====================================================
    # JSON
    # =====================================================

    elif file.name.endswith(".json"):

        try:

            raw_df = pd.read_json(file)

        except:

            st.error(
                "Unable to read JSON file"
            )

    # =====================================================
    # ZIP
    # =====================================================

    elif file.name.endswith(".zip"):

        try:

            zip_file = zipfile.ZipFile(file)

            for file_name in zip_file.namelist():

                if file_name.endswith(".csv"):

                    with zip_file.open(file_name) as f:

                        raw_df = pd.read_csv(f)

        except:

            st.error(
                "Unable to read ZIP file"
            )

    # =====================================================
    # CLEAN MISSING VALUES
    # =====================================================

    if raw_df is not None:

        raw_df = clean_missing_values(
            raw_df
        )

# =========================================================
# URL INPUT
# =========================================================

if raw_df is None and dataset_url:

    raw_df = universal_data_loader(
        dataset_url
    )

    if raw_df is not None:

        raw_df = clean_missing_values(
            raw_df
        )

    else:

        st.error(
            "Could not load data from source"
        )

# =========================================================
# MAIN APP
# =========================================================

if raw_df is not None:

    df = raw_df.copy()

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title("Controls")

    st.sidebar.subheader(
        "Data Cleaning"
    )

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
    # FILTERS
    # =====================================================

    st.sidebar.subheader("Filters")

    filtered_df = clean_df.copy()

    for col in clean_df.select_dtypes(
        include='object'
    ).columns:

        unique_vals = (
            clean_df[col]
            .dropna()
            .unique()
            .tolist()
        )

        if len(unique_vals) <= 50:

            selected = st.sidebar.selectbox(
                f"Filter by {col}",
                ["All"] + unique_vals
            )

            if selected != "All":

                filtered_df = filtered_df[
                    filtered_df[col]
                    == selected
                ]

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Dashboard",
            "Summary",
            "Visualization Studio",
            "Query Engine",
            "ML Prediction",
            "Auto Dashboard"
        ]
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    with tab1:

        st.subheader(
            "Dashboard Overview"
        )

        numeric_cols = (
            filtered_df
            .select_dtypes(
                include='number'
            )
            .columns
            .tolist()
        )

        if len(numeric_cols) > 0:

            c1, c2, c3, c4, c5 = st.columns(5)

            raw_missing = (
                raw_df
                .isnull()
                .sum()
                .sum()
            )

            clean_missing = (
                clean_df
                .isnull()
                .sum()
                .sum()
            )

            c1.metric(
                "Rows",
                len(filtered_df)
            )

            c2.metric(
                "Columns",
                filtered_df.shape[1]
            )

            c3.metric(
                "Raw Missing",
                f"{raw_missing:,}"
            )

            c4.metric(
                "Cleaned Missing",
                f"{clean_missing:,}"
            )

            useful_numeric_cols = [

                col for col in numeric_cols

                if "id" not in col.lower()
            ]

            if len(useful_numeric_cols) > 0:

                avg_col = useful_numeric_cols[0]

                avg_value = filtered_df[
                    avg_col
                ].mean()

                c5.metric(
                    f"Avg {avg_col}",
                    f"{avg_value:,.2f}"
                )

            else:

                c5.metric(
                    "Numeric Columns",
                    len(numeric_cols)
                )

        st.divider()

        st.subheader("Raw Data")

        rows_to_show = st.slider(
            "Rows",
            5,
            100,
            10
        )

        st.dataframe(
            raw_df.head(rows_to_show),
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "Cleaning Report"
        )

        for r in report:

            st.success(r)

        st.divider()

        st.subheader(
            "Cleaned Data"
        )

        st.dataframe(
            clean_df.head(rows_to_show),
            use_container_width=True
        )

        st.download_button(
            label="Download Cleaned CSV",
            data=clean_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    with tab2:

        st.subheader(
            "Statistical Summary"
        )

        st.dataframe(
            filtered_df.describe(
                include="all"
            ),
            use_container_width=True
        )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    with tab3:

        st.subheader(
            "Visualization Studio"
        )

        all_columns = (
            filtered_df.columns.tolist()
        )

        numeric_cols = (
            filtered_df
            .select_dtypes(
                include='number'
            )
            .columns
            .tolist()
        )

        categorical_cols = (
            filtered_df
            .select_dtypes(
                exclude='number'
            )
            .columns
            .tolist()
        )

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
        # GROUP BY OPTION
        # =================================================

        use_groupby = st.checkbox(
            "Use Group By"
        )

        group_col = None
        agg_func = None

        if use_groupby:

            group_col = st.selectbox(
                "Group By Column",
                categorical_cols
            )

            agg_func = st.selectbox(
                "Aggregation Function",
                [
                    "sum",
                    "mean",
                    "max",
                    "min",
                    "count"
                ]
            )

        # =================================================
        # BAR CHART
        # =================================================

        if chart_type == "Bar Chart":

            if (
                len(categorical_cols) > 0
                and
                len(numeric_cols) > 0
            ):

                x_col = st.selectbox(
                    "X-axis",
                    categorical_cols,
                    key="bar_x"
                )

                y_col = st.selectbox(
                    "Y-axis",
                    numeric_cols,
                    key="bar_y"
                )

                chart_df = filtered_df.copy()

                if use_groupby:

                    chart_df = (
                        chart_df
                        .groupby(group_col)[y_col]
                        .agg(agg_func)
                        .reset_index()
                    )

                    fig = px.bar(
                        chart_df,
                        x=group_col,
                        y=y_col,
                        color=group_col
                    )

                else:

                    fig = px.bar(
                        chart_df,
                        x=x_col,
                        y=y_col,
                        color=x_col
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # =================================================
        # LINE CHART
        # =================================================

        elif chart_type == "Line Chart":

            x_col = st.selectbox(
                "X-axis",
                all_columns,
                key="line_x"
            )

            y_col = st.selectbox(
                "Y-axis",
                numeric_cols,
                key="line_y"
            )

            chart_df = filtered_df.copy()

            if use_groupby:

                chart_df = (
                    chart_df
                    .groupby(group_col)[y_col]
                    .agg(agg_func)
                    .reset_index()
                )

                fig = px.line(
                    chart_df,
                    x=group_col,
                    y=y_col
                )

            else:

                fig = px.line(
                    chart_df,
                    x=x_col,
                    y=y_col
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # SCATTER PLOT
        # =================================================

        elif chart_type == "Scatter Plot":

            x_col = st.selectbox(
                "X-axis",
                numeric_cols,
                key="scatter_x"
            )

            y_col = st.selectbox(
                "Y-axis",
                numeric_cols,
                key="scatter_y"
            )

            chart_df = filtered_df.copy()

            if use_groupby:

                chart_df = (
                    chart_df
                    .groupby(group_col)[[x_col, y_col]]
                    .mean()
                    .reset_index()
                )

                fig = px.scatter(
                    chart_df,
                    x=x_col,
                    y=y_col,
                    color=group_col
                )

            else:

                fig = px.scatter(
                    chart_df,
                    x=x_col,
                    y=y_col
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # HISTOGRAM
        # =================================================

        elif chart_type == "Histogram":

            col = st.selectbox(
                "Column",
                numeric_cols,
                key="hist_col"
            )

            chart_df = filtered_df.copy()

            fig = px.histogram(
                chart_df,
                x=col,
                color=group_col if use_groupby else None
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # BOX PLOT
        # =================================================

        elif chart_type == "Box Plot":

            col = st.selectbox(
                "Column",
                numeric_cols,
                key="box_col"
            )

            chart_df = filtered_df.copy()

            fig = px.box(
                chart_df,
                y=col,
                color=group_col if use_groupby else None
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # PIE CHART
        # =================================================

        elif chart_type == "Pie Chart":

            if len(categorical_cols) > 0:

                col = st.selectbox(
                    "Category",
                    categorical_cols,
                    key="pie_col"
                )

                chart_df = filtered_df.copy()

                if use_groupby:

                    pie_data = (
                        chart_df[group_col]
                        .value_counts()
                        .reset_index()
                    )

                    pie_data.columns = [
                        group_col,
                        "Count"
                    ]

                    fig = px.pie(
                        pie_data,
                        names=group_col,
                        values="Count"
                    )

                else:

                    pie_data = (
                        chart_df[col]
                        .value_counts()
                        .reset_index()
                    )

                    pie_data.columns = [
                        col,
                        "Count"
                    ]

                    fig = px.pie(
                        pie_data,
                        names=col,
                        values="Count"
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # =================================================
        # HEATMAP
        # =================================================

        elif (
            chart_type
            ==
            "Correlation Heatmap"
        ):

            if len(numeric_cols) > 0:

                corr = (
                    filtered_df[
                        numeric_cols
                    ].corr()
                )

                fig = px.imshow(
                    corr,
                    text_auto=True
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    # =====================================================
    # QUERY ENGINE
    # =====================================================

    with tab4:

        st.subheader(
            "Ask Questions About Data"
        )

        query = st.text_input(
            "Enter Query"
        )

        if query:

            op, target, group = (
                parse_query(
                    query,
                    filtered_df
                )
            )

            result = execute_query(
                filtered_df,
                op,
                target,
                group
            )

            st.write(result)

            insight = (
                generate_query_insight(
                    result,
                    target,
                    group
                )
            )

            if insight:

                st.info(insight)

    # =====================================================
    # ML PREDICTION
    # =====================================================

    with tab5:

        st.subheader(
            "Machine Learning Prediction"
        )

        numeric_targets = (
            filtered_df
            .select_dtypes(
                include='number'
            )
            .columns
            .tolist()
        )

        if len(numeric_targets) > 0:

            target = st.selectbox(
                "Select Target",
                numeric_targets
            )

            if st.button(
                "Train Model"
            ):

                model, score = (
                    train_prediction_model(
                        filtered_df,
                        target
                    )
                )

                if model:

                    st.success(
                        f"Model Trained | R² Score: {score:.2f}"
                    )

                    input_data = {}

                    for col in numeric_targets:

                        if col != target:

                            input_data[col] = (
                                st.number_input(
                                    f"Enter {col}",
                                    value=0.0
                                )
                            )

                    if st.button(
                        "Predict"
                    ):

                        input_df = pd.DataFrame(
                            [input_data]
                        )

                        prediction = (
                            model.predict(
                                input_df
                            )
                        )

                        st.success(
                            f"Predicted {target}: {prediction[0]:.2f}"
                        )
                        
    # =====================================================
    # AUTO DASHBOARD
    # =====================================================

    with tab6:

        st.subheader(
            "AI Auto Dashboard"
        )

        # ================================================
        # KPIs
        # ================================================

        metrics = generate_kpis(
            filtered_df
        )

        c1, c2, c3, c4 = st.columns(4)

        metric_keys = list(
            metrics.keys()
        )

        c1.metric(
            metric_keys[0],
            metrics[metric_keys[0]]
        )

        c2.metric(
            metric_keys[1],
            metrics[metric_keys[1]]
        )

        c3.metric(
            metric_keys[2],
            metrics[metric_keys[2]]
        )

        c4.metric(
            metric_keys[3],
            metrics[metric_keys[3]]
        )

        st.divider()

        # ================================================
        # AUTO CHARTS
        # ================================================

        st.subheader(
            "Auto Generated Charts"
        )

        charts = generate_auto_charts(
            filtered_df
        )

        for fig in charts:

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ================================================
        # AI INSIGHTS
        # ================================================

        st.subheader(
            "AI Insights"
        )

        insights = generate_insights(
            filtered_df
        )

        for insight in insights:

            st.success(insight)