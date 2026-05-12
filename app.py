import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import DataPreprocessor
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

    separators = [",", ";", "\t"]

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
# FILE INPUT
# =========================================================

if file is not None:

    if file.name.endswith(".csv"):

        raw_df = load_csv_safely(file)

        if raw_df is None:

            st.error(
                "Unable to read CSV file"
            )

    elif file.name.endswith(".xlsx"):

        try:

            raw_df = pd.read_excel(file)

        except:

            st.error(
                "Unable to read Excel file"
            )

# =========================================================
# URL INPUT
# =========================================================

elif dataset_url:

    try:

        raw_df = pd.read_csv(dataset_url)

    except:

        try:

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

    df = raw_df.copy()

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title("Controls")

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
    # FILTERS
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
    # DASHBOARD
    # =====================================================

    with tab1:

        st.subheader("Dashboard Overview")

        numeric_cols = filtered_df.select_dtypes(
            include='number'
        ).columns.tolist()

        if len(numeric_cols) > 0:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Rows",
                len(filtered_df)
            )

            c2.metric(
                "Columns",
                filtered_df.shape[1]
            )

            c3.metric(
                "Missing Values",
                int(raw_df.isnull().sum().sum())
            )

            c4.metric(
                "Average",
                round(filtered_df[numeric_cols[0]].mean(), 2)
            )

        st.divider()

        st.subheader("Raw Data")

        rows_to_show = st.slider(
            "Rows",
            5,
            50,
            10
        )

        st.dataframe(
            raw_df.head(rows_to_show),
            use_container_width=True
        )

        st.divider()

        st.subheader("Cleaning Report")

        for r in report:

            st.success(r)

        st.divider()

        st.subheader("Cleaned Data")

        st.dataframe(
            clean_df.head(rows_to_show),
            use_container_width=True
        )

        st.download_button(
            label="Download Cleaned CSV",
            data=clean_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # =====================================================
    # VISUALIZATION
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
        # BAR
        # =================================================

        if chart_type == "Bar Chart":

            if len(categorical_cols) > 0 and len(numeric_cols) > 0:

                x_col = st.selectbox(
                    "X-axis",
                    categorical_cols
                )

                y_col = st.selectbox(
                    "Y-axis",
                    numeric_cols
                )

                fig = px.bar(
                    filtered_df,
                    x=x_col,
                    y=y_col,
                    color=x_col
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # =================================================
        # LINE
        # =================================================

        elif chart_type == "Line Chart":

            x_col = st.selectbox(
                "X-axis",
                all_columns
            )

            y_col = st.selectbox(
                "Y-axis",
                numeric_cols
            )

            fig = px.line(
                filtered_df,
                x=x_col,
                y=y_col
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # SCATTER
        # =================================================

        elif chart_type == "Scatter Plot":

            x_col = st.selectbox(
                "X-axis",
                numeric_cols
            )

            y_col = st.selectbox(
                "Y-axis",
                numeric_cols
            )

            fig = px.scatter(
                filtered_df,
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
                numeric_cols
            )

            fig = px.histogram(
                filtered_df,
                x=col
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
                numeric_cols
            )

            fig = px.box(
                filtered_df,
                y=col
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =================================================
        # PIE CHART
        # =================================================

        elif chart_type == "Pie Chart":

            col = st.selectbox(
                "Category",
                categorical_cols
            )

            pie_data = (
                filtered_df[col]
                .value_counts()
                .reset_index()
            )

            pie_data.columns = [col, "Count"]

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

        elif chart_type == "Correlation Heatmap":

            corr = filtered_df[numeric_cols].corr()

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

    with tab3:

        st.subheader("Ask Questions About Data")

        query = st.text_input(
            "Enter Query"
        )

        if query:

            op, target, group = parse_query(
                query,
                filtered_df
            )

            result = execute_query(
                filtered_df,
                op,
                target,
                group
            )

            st.write(result)

            insight = generate_query_insight(
                result,
                target,
                group
            )

            if insight:

                st.info(insight)

    # =====================================================
    # ML PREDICTION
    # =====================================================

    with tab4:

        st.subheader("Machine Learning Prediction")

        numeric_targets = filtered_df.select_dtypes(
            include='number'
        ).columns.tolist()

        if len(numeric_targets) > 0:

            target = st.selectbox(
                "Select Target",
                numeric_targets
            )

            if st.button("Train Model"):

                model, score = train_prediction_model(
                    filtered_df,
                    target
                )

                if model:

                    st.success(
                        f"Model Trained | R² Score: {score:.2f}"
                    )

                    input_data = {}

                    for col in numeric_targets:

                        if col != target:

                            input_data[col] = st.number_input(
                                f"Enter {col}",
                                value=0.0
                            )

                    if st.button("Predict"):

                        input_df = pd.DataFrame(
                            [input_data]
                        )

                        prediction = model.predict(
                            input_df
                        )

                        st.success(
                            f"Predicted {target}: {prediction[0]:.2f}"
                        )