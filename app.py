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
        "xls",
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
# SAFE EXCEL LOADER
# =========================================================

def load_excel_safely(file):

    engines = [
        "openpyxl",
        "xlrd"
    ]

    for engine in engines:

        try:

            file.seek(0)

            excel_file = pd.ExcelFile(
                file,
                engine=engine
            )

            sheet_name = excel_file.sheet_names[0]

            df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                engine=engine
            )

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

            # CSV FILES

            csv_files = glob.glob(
                os.path.join(path, "**/*.csv"),
                recursive=True
            )

            if len(csv_files) > 0:

                return pd.read_csv(
                    csv_files[0]
                )

            # EXCEL FILES

            excel_files = (
                glob.glob(
                    os.path.join(path, "**/*.xlsx"),
                    recursive=True
                )
                +
                glob.glob(
                    os.path.join(path, "**/*.xls"),
                    recursive=True
                )
            )

            if len(excel_files) > 0:

                return pd.read_excel(
                    excel_files[0]
                )

    except Exception as e:

        st.error(f"Kaggle Loading Error: {e}")

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

        if ".xlsx" in source or ".xls" in source:

            return pd.read_excel(source)

    except:
        pass

    # =====================================================
    # JSON API
    # =====================================================

    try:

        response = requests.get(
            source,
            timeout=20
        )

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

                # CSV

                if file_name.endswith(".csv"):

                    with zip_file.open(file_name) as f:

                        return pd.read_csv(f)

                # EXCEL

                elif (
                    file_name.endswith(".xlsx")
                    or
                    file_name.endswith(".xls")
                ):

                    with zip_file.open(file_name) as f:

                        return pd.read_excel(f)

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

    df = df.copy()

    df.replace(
        [
            "?",
            "NA",
            "N/A",
            "null",
            "NULL",
            "",
            " ",
            "none",
            "None",
            "NaN",
            "nan",
            "-",
            "--"
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

        if raw_df is None:

            st.error(
                "Unable to read CSV file"
            )

    # =====================================================
    # EXCEL
    # =====================================================

    elif (
        file.name.endswith(".xlsx")
        or
        file.name.endswith(".xls")
    ):

        raw_df = load_excel_safely(file)

        if raw_df is None:

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

                # CSV

                if file_name.endswith(".csv"):

                    with zip_file.open(file_name) as f:

                        raw_df = pd.read_csv(f)

                        break

                # EXCEL

                elif (
                    file_name.endswith(".xlsx")
                    or
                    file_name.endswith(".xls")
                ):

                    with zip_file.open(file_name) as f:

                        raw_df = pd.read_excel(f)

                        break

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

    st.success("Dataset Loaded Successfully")

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
    # DASHBOARD
    # =====================================================

    st.subheader("Raw Data")

    st.dataframe(
        raw_df.head(),
        use_container_width=True
    )

    st.subheader("Cleaned Data")

    st.dataframe(
        clean_df.head(),
        use_container_width=True
    )

    st.subheader("Cleaning Report")

    for r in report:

        st.success(r)

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(
        label="Download Cleaned CSV",
        data=clean_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )