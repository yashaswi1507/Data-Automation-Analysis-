# IMPORTS
from dataset_profiler import DatasetProfiler
import os, io, glob, zipfile, requests, kagglehub
import streamlit as st
import pandas as pd
import plotly.express as px
import openpyxl
from bs4 import BeautifulSoup

from eda import show_summary
from preprocessing import DataPreprocessor
from query_engine import parse_query, execute_query, generate_query_insight
from ml_engine import train_prediction_model
from dashboard_generator import generate_kpis, generate_auto_charts, generate_insights

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Data Analysis Automation Tool", layout="wide")

st.title("Data Analysis Automation Tool")
st.markdown("Upload data, clean it, visualize it, query it, and predict insights using AI-powered analytics.")

# =========================================================
# FILE UPLOAD
# =========================================================

file        = st.file_uploader("Upload Dataset", type=["csv","xlsx","xls","json","zip"])
dataset_url = st.text_input("Or Paste Dataset URL / API / Kaggle Dataset")

raw_df = None

# =========================================================
# SAFE CSV LOADER
# =========================================================

def load_csv_safely(file):
    for enc in ["utf-8","latin1","ISO-8859-1","cp1252"]:
        for sep in [",",";","\t"]:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=sep, on_bad_lines="skip")
                if len(df.columns) > 1:
                    return df
            except:
                continue
    return None

# =========================================================
# UNIVERSAL DATA LOADER
# =========================================================

def universal_data_loader(source):

    try:
        if "kaggle.com/datasets/" in source:
            parts        = source.split("/datasets/")[1]
            dataset_path = parts.split("?")[0]
            path         = kagglehub.dataset_download(dataset_path)
            csv_files    = glob.glob(os.path.join(path, "*.csv"))
            if csv_files:  return pd.read_csv(csv_files[0])
            xlsx_files   = glob.glob(os.path.join(path, "*.xlsx"))
            if xlsx_files: return pd.read_excel(xlsx_files[0])
    except: pass

    try:
        if ".csv" in source:  return pd.read_csv(source)
    except: pass

    try:
        if ".xlsx" in source: return pd.read_excel(source, engine="openpyxl")
    except: pass

    try:
        response = requests.get(source)
        return pd.DataFrame(response.json())
    except: pass

    try:
        tables = pd.read_html(source)
        if tables: return tables[0]
    except: pass

    try:
        if ".zip" in source:
            zip_file = zipfile.ZipFile(io.BytesIO(requests.get(source).content))
            for name in zip_file.namelist():
                if name.endswith(".csv"):
                    with zip_file.open(name) as f: return pd.read_csv(f)
                if name.endswith(".xlsx"):
                    with zip_file.open(name) as f: return pd.read_excel(f, engine="openpyxl")
    except: pass

    try:
        resp = requests.get(source, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        data = [{"Text": p.get_text().strip()} for p in soup.find_all("p")[:100] if p.get_text().strip()]
        if data: return pd.DataFrame(data)
    except: pass

    return None

# =========================================================
# FILE INPUT
# =========================================================

if file is not None:
    if file.name.endswith(".csv"):
        raw_df = load_csv_safely(file)

    elif file.name.endswith((".xlsx",".xls")):
        try:
            file.seek(0)
            engine = "openpyxl" if file.name.endswith(".xlsx") else "xlrd"
            raw_df = pd.read_excel(file, engine=engine)
            raw_df.columns = raw_df.columns.astype(str).str.strip()
        except Exception as e:
            st.error(f"Excel Error: {e}")

    elif file.name.endswith(".json"):
        try:
            raw_df = pd.read_json(file)
        except:
            st.error("Unable to read JSON file")

    elif file.name.endswith(".zip"):
        try:
            zf = zipfile.ZipFile(file)
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as f: raw_df = pd.read_csv(f); break
        except:
            st.error("Unable to read ZIP file")

if raw_df is None and dataset_url:
    raw_df = universal_data_loader(dataset_url)
    if raw_df is None:
        st.error("Could not load data from source")

# =========================================================
# MAIN APP
# =========================================================

if raw_df is not None:

    df = raw_df.copy()

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title("Controls")

    # ── Outliers ────────────────────────────────────────
    st.sidebar.subheader("Outlier Handling")
    outlier_option = st.sidebar.selectbox(
        "Handle Outliers",
        ["No Action", "Remove Outliers", "Cap Outliers"]
    )

    st.sidebar.divider()

    # ── Missing Values — Auto by default ─────────────────
    st.sidebar.subheader("Missing Value Handling")

    manual_override = st.sidebar.toggle(
        "Manual Override",
        value=False,
        help=(
            "OFF → each column is filled automatically based on its own data pattern "
            "(skewed=median, discrete int=mode, binary=Unknown, etc.).\n\n"
            "ON → you choose the method; applies to numeric columns."
        )
    )

    if manual_override:
        missing_option = st.sidebar.selectbox(
            "Fill Method",
            ["Median", "Mean", "Mode", "Drop Rows"],
        )
        st.sidebar.info("⚙️ Manual override active — numeric columns will use your chosen method.")
    else:
        missing_option = "Auto"
        st.sidebar.success("✅ Auto mode — each column filled by its data personality.")

    st.sidebar.divider()

    # =====================================================
    # DATASET PROFILING
    # =====================================================

    dataset_profile = "general"
    column_profiles = {}

    try:
        profiler        = DatasetProfiler(df)
        dataset_profile = profiler.detect_dataset_type()
        column_profiles = profiler.profile_columns()
    except Exception as e:
        st.warning(f"Profiler Warning: {e}")

    # =====================================================
    # PREPROCESSING
    # =====================================================

    processor        = DataPreprocessor(df, outlier_option, missing_option, dataset_profile, column_profiles)
    clean_df, report = processor.process()

    # =====================================================
    # SIDEBAR FILTERS
    # =====================================================

    st.sidebar.subheader("Filters")
    filtered_df = clean_df.copy()

    for col in clean_df.select_dtypes(include="object").columns:
        unique_vals = clean_df[col].dropna().unique().tolist()
        if len(unique_vals) <= 50:
            selected = st.sidebar.selectbox(f"Filter: {col}", ["All"] + sorted([str(v) for v in unique_vals]))
            if selected != "All":
                filtered_df = filtered_df[filtered_df[col].astype(str) == selected]

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", "📋 Summary", "📈 Visualization Studio",
        "🔍 Query Engine", "🤖 ML Prediction", "⚡ Auto Dashboard"
    ])

    # =====================================================
    # TAB 1 — DASHBOARD
    # =====================================================

    with tab1:
        st.subheader("Dashboard Overview")

        # Dataset type badge
        type_colors = {
            "transactional":  "🟦",
            "time-series":    "🟩",
            "numeric-heavy":  "🟨",
            "people-records": "🟧",
            "categorical":    "🟪",
            "mixed":          "⬜",
        }
        badge = type_colors.get(dataset_profile, "⬜")
        st.info(f"{badge} Detected Dataset Type: **{dataset_profile.upper()}**")

        # KPI metrics
        numeric_cols  = filtered_df.select_dtypes(include="number").columns.tolist()
        raw_missing   = int(raw_df.isnull().sum().sum())
        clean_missing = int(clean_df.isnull().sum().sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows",            len(filtered_df))
        c2.metric("Columns",         filtered_df.shape[1])
        c3.metric("Missing (Raw)",   f"{raw_missing:,}")
        c4.metric("Missing (Clean)", f"{clean_missing:,}", delta=f"-{raw_missing - clean_missing:,}", delta_color="inverse")
        c5.metric("Numeric Cols",    len(numeric_cols))

        st.divider()

        # Raw data preview
        st.subheader("Raw Data Preview")
        rows_to_show = st.slider("Rows to show", 5, 100, 10)
        st.dataframe(raw_df.head(rows_to_show), use_container_width=True)

        st.divider()

        # ── Cleaning Report — improved UI ────────────────
        st.subheader("🧹 Cleaning Report")

        # Categorise report lines
        info_lines   = [r for r in report if r.startswith("📋")]
        split_lines  = [r for r in report if "Split" in r]
        fill_lines   = [r for r in report if ("missing)" in r or "↳" in r) and "📋" not in r]
        dup_lines    = [r for r in report if "duplicate" in r.lower()]
        outlier_lines= [r for r in report if "outlier" in r.lower() or "capped" in r.lower()]
        done_lines   = [r for r in report if r.startswith("✅")]
        other_lines  = [r for r in report if r not in info_lines + split_lines + fill_lines
                        + dup_lines + outlier_lines + done_lines
                        and not r.startswith("─")]

        # Summary bar
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Columns Split",    len(split_lines))
        col_b.metric("Columns Filled",   len([l for l in fill_lines if "↳" not in l]))
        col_c.metric("Duplicates",       next((l.split(": ")[-1] for l in dup_lines if "Removed" in l), "0"))
        col_d.metric("Outlier Actions",  len(outlier_lines))

        st.divider()

        with st.expander("📝 Dataset Info", expanded=False):
            for r in info_lines:
                st.write(r)

        if split_lines:
            with st.expander(f"✂️ Structured Columns Split ({len(split_lines)})", expanded=True):
                for r in split_lines + [l for l in fill_lines if "↳" in l]:
                    if "↳" in r:
                        st.caption(r)
                    else:
                        st.success(r)

        non_arrow_fills = [l for l in fill_lines if "↳" not in l]
        if non_arrow_fills:
            with st.expander(f"🔧 Missing Values Filled ({len(non_arrow_fills)})", expanded=True):
                for r in non_arrow_fills:
                    # Colour by method used
                    if "Unknown" in r:
                        st.warning(r)
                    elif "median" in r or "mean" in r:
                        st.info(r)
                    elif "mode" in r or "forward" in r:
                        st.success(r)
                    else:
                        st.success(r)

        if dup_lines:
            with st.expander("🗑️ Duplicates", expanded=False):
                for r in dup_lines:
                    st.success(r)

        if outlier_lines:
            with st.expander(f"📉 Outlier Handling ({len(outlier_lines)})", expanded=False):
                for r in outlier_lines:
                    st.info(r)

        if done_lines:
            st.divider()
            for r in done_lines:
                st.success(r)

        st.divider()

        # Cleaned data preview
        st.subheader("Cleaned Data")
        st.dataframe(clean_df.head(rows_to_show), use_container_width=True)

        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=clean_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # =====================================================
    # TAB 2 — SUMMARY
    # =====================================================

    with tab2:
        st.subheader("Statistical Summary")
        summary_df = filtered_df.describe(include=[float, int]).fillna("-")
        st.dataframe(summary_df, use_container_width=True)

        # Column type breakdown
        st.subheader("Column Profiles")
        if column_profiles:
            profile_rows = []
            for col, p in column_profiles.items():
                profile_rows.append({
                    "Column":        col,
                    "Detected Type": p.get("detected_type",""),
                    "Fill Strategy": p.get("cleaning_strategy",""),
                    "Missing %":     f"{p.get('missing_percent',0):.1f}%",
                    "Unique Count":  p.get("unique_count",""),
                })
            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # =====================================================
    # TAB 3 — VISUALIZATION STUDIO
    # =====================================================

    with tab3:
        st.subheader("Visualization Studio")

        all_columns     = filtered_df.columns.tolist()
        numeric_cols    = filtered_df.select_dtypes(include="number").columns.tolist()
        categorical_cols= filtered_df.select_dtypes(exclude="number").columns.tolist()

        chart_type = st.selectbox("Choose Chart Type", [
            "Bar Chart","Line Chart","Scatter Plot",
            "Histogram","Box Plot","Pie Chart","Correlation Heatmap"
        ])

        use_groupby = st.checkbox("Use Group By")
        group_col   = None
        agg_func    = None

        if use_groupby and categorical_cols:
            group_col = st.selectbox("Group By Column", categorical_cols)
            agg_func  = st.selectbox("Aggregation", ["sum","mean","max","min","count"])

        if chart_type == "Bar Chart" and categorical_cols and numeric_cols:
            x_col    = st.selectbox("X-axis", categorical_cols, key="bar_x")
            y_col    = st.selectbox("Y-axis", numeric_cols,    key="bar_y")
            chart_df = filtered_df.copy()
            if use_groupby and group_col:
                chart_df = chart_df.groupby(group_col)[y_col].agg(agg_func).reset_index()
                fig = px.bar(chart_df, x=group_col, y=y_col, color=group_col)
            else:
                fig = px.bar(chart_df, x=x_col, y=y_col, color=x_col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart" and numeric_cols:
            x_col    = st.selectbox("X-axis", all_columns, key="line_x")
            y_col    = st.selectbox("Y-axis", numeric_cols, key="line_y")
            chart_df = filtered_df.copy()
            if use_groupby and group_col:
                chart_df = chart_df.groupby(group_col)[y_col].agg(agg_func).reset_index()
                fig = px.line(chart_df, x=group_col, y=y_col)
            else:
                fig = px.line(chart_df, x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            x_col    = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col    = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            chart_df = filtered_df.copy()
            if use_groupby and group_col:
                chart_df = chart_df.groupby(group_col)[[x_col, y_col]].mean().reset_index()
                fig = px.scatter(chart_df, x=x_col, y=y_col, color=group_col)
            else:
                fig = px.scatter(chart_df, x=x_col, y=y_col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram" and numeric_cols:
            col      = st.selectbox("Column", numeric_cols, key="hist_col")
            chart_df = filtered_df.copy()
            fig      = px.histogram(chart_df, x=col, color=group_col if use_groupby else None)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot" and numeric_cols:
            col      = st.selectbox("Column", numeric_cols, key="box_col")
            chart_df = filtered_df.copy()
            fig      = px.box(chart_df, y=col, color=group_col if use_groupby else None)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie Chart" and categorical_cols:
            col      = st.selectbox("Category", categorical_cols, key="pie_col")
            chart_df = filtered_df.copy()
            pie_col  = group_col if (use_groupby and group_col) else col
            pie_data = chart_df[pie_col].value_counts().reset_index()
            pie_data.columns = [pie_col, "Count"]
            fig = px.pie(pie_data, names=pie_col, values="Count")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap" and numeric_cols:
            corr = filtered_df[numeric_cols].corr()
            fig  = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Not enough columns of the required type for this chart.")

    # =====================================================
    # TAB 4 — QUERY ENGINE
    # =====================================================

    with tab4:
        st.subheader("Ask Questions About Your Data")
        query = st.text_input("Enter Query (e.g. 'average salary by department')")
        if query:
            op, target, group = parse_query(query, filtered_df)
            result = execute_query(filtered_df, op, target, group)
            st.write(result)
            insight = generate_query_insight(result, target, group)
            if insight:
                st.info(insight)

    # =====================================================
    # TAB 5 — ML PREDICTION
    # =====================================================

    with tab5:
        st.subheader("Machine Learning Prediction")

        numeric_targets = filtered_df.select_dtypes(include="number").columns.tolist()

        if numeric_targets:
            target = st.selectbox("Select Target Column", numeric_targets)

            if st.button("Train Model"):
                model, score = train_prediction_model(filtered_df, target)
                if model:
                    st.success(f"✅ Model Trained | R² Score: {score:.2f}")

                    st.subheader("Make a Prediction")
                    input_data = {}
                    for col in numeric_targets:
                        if col != target:
                            input_data[col] = st.number_input(f"Enter {col}", value=0.0)

                    if st.button("Predict"):
                        input_df   = pd.DataFrame([input_data])
                        prediction = model.predict(input_df)
                        st.success(f"🎯 Predicted {target}: {prediction[0]:.2f}")
                else:
                    st.error("Model training failed. Check your data.")
        else:
            st.warning("No numeric columns found for prediction.")

    # =====================================================
    # TAB 6 — AUTO DASHBOARD
    # =====================================================

    with tab6:
        st.subheader("⚡ AI Auto Dashboard")

        # KPIs
        metrics     = generate_kpis(filtered_df)
        metric_keys = list(metrics.keys())

        if len(metric_keys) >= 4:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(metric_keys[0], metrics[metric_keys[0]])
            c2.metric(metric_keys[1], metrics[metric_keys[1]])
            c3.metric(metric_keys[2], metrics[metric_keys[2]])
            c4.metric(metric_keys[3], metrics[metric_keys[3]])

        st.divider()

        # Auto Charts
        st.subheader("Auto Generated Charts")
        charts = generate_auto_charts(filtered_df)
        for fig in charts:
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # AI Insights
        st.subheader("AI Insights")
        insights = generate_insights(filtered_df)
        for insight in insights:
            st.success(insight)