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
from ml_engine import train_prediction_model, detect_task_type, predict_single
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

        all_columns      = filtered_df.columns.tolist()
        numeric_cols     = filtered_df.select_dtypes(include="number").columns.tolist()
        categorical_cols = filtered_df.select_dtypes(exclude="number").columns.tolist()

        # ── Smart data limiter ────────────────────────────────────────
        # For categorical axes: too many unique values = messy chart.
        # Default: show Top N aggregated. User can expand if needed.

        DEFAULT_TOP_N = 15   # clean default
        MAX_SCATTER   = 500  # scatter points before sampling

        def smart_limit_categorical(df, col, y_col, agg, top_n, show_all):
            """
            Aggregate col→y_col and return top_n rows (or all if show_all).
            Works for bar, pie, line-by-category.
            """
            grouped = (
                df.groupby(col)[y_col]
                .agg(agg)
                .reset_index()
                .sort_values(y_col, ascending=False)
            )
            if not show_all:
                grouped = grouped.head(top_n)
            return grouped

        def smart_limit_scatter(df, show_all):
            """Sample large scatter datasets for readability."""
            if show_all or len(df) <= MAX_SCATTER:
                return df, len(df)
            sampled = df.sample(MAX_SCATTER, random_state=42)
            return sampled, len(df)

        # ── Chart type selector ───────────────────────────────────────
        chart_type = st.selectbox("Choose Chart Type", [
            "Bar Chart", "Line Chart", "Scatter Plot",
            "Histogram", "Box Plot", "Pie Chart", "Correlation Heatmap"
        ])

        st.divider()

        # ── Controls row ──────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

        with ctrl1:
            use_groupby = st.checkbox("Use Group By")
            group_col   = None
            agg_func    = "sum"
            if use_groupby and categorical_cols:
                group_col = st.selectbox("Group By Column", categorical_cols, key="grp_col")
                agg_func  = st.selectbox("Aggregation", ["sum","mean","max","min","count"], key="grp_agg")

        with ctrl2:
            # Charts that aggregate categories benefit from Top N
            if chart_type in ("Bar Chart","Pie Chart","Line Chart"):
                show_all = st.toggle("Show All Values", value=False,
                    help=f"OFF = Top {DEFAULT_TOP_N} values only (cleaner). ON = all values.")
                if show_all:
                    top_n = st.slider("Max values to show", 10, 200, 50, key="topn_slider")
                else:
                    top_n = DEFAULT_TOP_N
            elif chart_type == "Scatter Plot":
                show_all = st.toggle("Show All Points", value=False,
                    help=f"OFF = max {MAX_SCATTER} points sampled. ON = all rows.")
                top_n = DEFAULT_TOP_N
            else:
                show_all = True
                top_n    = DEFAULT_TOP_N

        with ctrl3:
            sort_order = "desc"
            if chart_type in ("Bar Chart","Line Chart"):
                sort_order = st.radio("Sort", ["Top → Bottom","Bottom → Top","Original Order"],
                                      horizontal=True, key="sort_radio")

        st.divider()

        # ── Render charts ──────────────────────────────────────────────

        # ── BAR CHART ────────────────────────────────────────────────
        if chart_type == "Bar Chart":
            if not categorical_cols or not numeric_cols:
                st.warning("Need at least one categorical and one numeric column.")
            else:
                x_col = st.selectbox("X-axis (Category)", categorical_cols, key="bar_x")
                y_col = st.selectbox("Y-axis (Value)",    numeric_cols,     key="bar_y")
                agg   = agg_func if use_groupby else "sum"
                grp   = group_col if use_groupby else x_col

                chart_df = smart_limit_categorical(filtered_df, grp, y_col, agg, top_n, show_all)

                ascending = sort_order == "Bottom → Top"
                if sort_order != "Original Order":
                    chart_df = chart_df.sort_values(y_col, ascending=ascending)

                total_cats = filtered_df[grp].nunique()
                showing    = len(chart_df)

                if not show_all and total_cats > top_n:
                    st.caption(f"📊 Showing top {showing} of {total_cats} categories by {y_col}. Toggle 'Show All Values' to see more.")

                fig = px.bar(
                    chart_df, x=grp, y=y_col, color=grp,
                    text_auto=".2s",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(
                    xaxis_tickangle=-35,
                    showlegend=False,
                    bargap=0.25,
                    height=480,
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)

        # ── LINE CHART ───────────────────────────────────────────────
        elif chart_type == "Line Chart":
            if not numeric_cols:
                st.warning("Need at least one numeric column.")
            else:
                x_col = st.selectbox("X-axis", all_columns,   key="line_x")
                y_col = st.selectbox("Y-axis", numeric_cols,  key="line_y")

                if use_groupby and group_col:
                    chart_df = smart_limit_categorical(filtered_df, group_col, y_col, agg_func, top_n, show_all)
                    x_use    = group_col
                else:
                    chart_df = filtered_df[[x_col, y_col]].copy().dropna()
                    # For high-cardinality x, sort and optionally limit
                    if not show_all and len(chart_df) > top_n * 10:
                        chart_df = chart_df.sort_values(x_col).iloc[:: max(1, len(chart_df)//(top_n*10))]
                    else:
                        chart_df = chart_df.sort_values(x_col)
                    x_use = x_col

                total_pts = len(filtered_df)
                showing   = len(chart_df)
                if not show_all and showing < total_pts:
                    st.caption(f"📈 Showing {showing} of {total_pts} points for clarity.")

                fig = px.line(
                    chart_df, x=x_use, y=y_col,
                    markers=len(chart_df) <= 60,
                    template="plotly_white",
                    color_discrete_sequence=["#636EFA"],
                )
                fig.update_layout(height=460, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

        # ── SCATTER PLOT ─────────────────────────────────────────────
        elif chart_type == "Scatter Plot":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns.")
            else:
                x_col    = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_col    = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                color_col= st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_color")

                chart_df, total_pts = smart_limit_scatter(filtered_df, show_all)
                showing = len(chart_df)

                if not show_all and showing < total_pts:
                    st.caption(f"🔵 Showing {showing} sampled points of {total_pts} total for performance. Toggle 'Show All Points' to see everything.")

                fig = px.scatter(
                    chart_df,
                    x=x_col, y=y_col,
                    color=color_col if color_col != "None" else None,
                    opacity=0.65,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    trendline="ols" if len(chart_df) >= 10 else None,
                )
                fig.update_layout(height=480)
                fig.update_traces(marker=dict(size=6))
                st.plotly_chart(fig, use_container_width=True)

        # ── HISTOGRAM ────────────────────────────────────────────────
        elif chart_type == "Histogram":
            if not numeric_cols:
                st.warning("Need at least one numeric column.")
            else:
                col      = st.selectbox("Column", numeric_cols, key="hist_col")
                nbins    = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
                chart_df = filtered_df[[col]].dropna()

                fig = px.histogram(
                    chart_df, x=col,
                    nbins=nbins,
                    template="plotly_white",
                    color_discrete_sequence=["#636EFA"],
                    marginal="box",       # mini box plot on top for distribution shape
                )
                fig.update_layout(
                    height=460,
                    bargap=0.05,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── BOX PLOT ─────────────────────────────────────────────────
        elif chart_type == "Box Plot":
            if not numeric_cols:
                st.warning("Need at least one numeric column.")
            else:
                col      = st.selectbox("Column", numeric_cols, key="box_col")
                grp_box  = group_col if use_groupby and group_col else None

                if grp_box:
                    # Limit categories for box plot too
                    top_cats = (
                        filtered_df[grp_box].value_counts()
                        .head(top_n).index.tolist()
                    )
                    chart_df = filtered_df[filtered_df[grp_box].isin(top_cats)]
                    if not show_all and filtered_df[grp_box].nunique() > top_n:
                        st.caption(f"📦 Showing top {top_n} categories by frequency.")
                else:
                    chart_df = filtered_df

                fig = px.box(
                    chart_df, y=col,
                    x=grp_box,
                    color=grp_box,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    points="outliers",
                )
                fig.update_layout(height=460, showlegend=False, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

        # ── PIE CHART ────────────────────────────────────────────────
        elif chart_type == "Pie Chart":
            if not categorical_cols:
                st.warning("Need at least one categorical column.")
            else:
                col      = st.selectbox("Category Column", categorical_cols, key="pie_col")
                pie_col  = group_col if (use_groupby and group_col) else col

                pie_data = (
                    filtered_df[pie_col]
                    .value_counts()
                    .reset_index()
                )
                pie_data.columns = [pie_col, "Count"]

                total_cats = len(pie_data)
                if not show_all and total_cats > top_n:
                    top_data   = pie_data.head(top_n)
                    other_sum  = pie_data.iloc[top_n:]["Count"].sum()
                    if other_sum > 0:
                        other_row  = pd.DataFrame([{pie_col: f"Others ({total_cats - top_n})", "Count": other_sum}])
                        top_data   = pd.concat([top_data, other_row], ignore_index=True)
                    pie_data = top_data
                    st.caption(f"🥧 Top {top_n} shown. Remaining {total_cats - top_n} categories grouped as 'Others'.")

                fig = px.pie(
                    pie_data, names=pie_col, values="Count",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.35,            # donut style — easier to read labels
                )
                fig.update_traces(
                    textposition="outside",
                    textinfo="percent+label",
                    pull=[0.03]*len(pie_data),
                )
                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

        # ── CORRELATION HEATMAP ───────────────────────────────────────
        elif chart_type == "Correlation Heatmap":
            if not numeric_cols:
                st.warning("Need at least two numeric columns.")
            else:
                if len(numeric_cols) > 20:
                    st.caption(f"📐 {len(numeric_cols)} numeric columns — showing all. Deselect columns from filters if needed.")

                corr = filtered_df[numeric_cols].corr().round(2)
                fig  = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    aspect="auto",
                    template="plotly_white",
                )
                fig.update_layout(
                    height=max(400, len(numeric_cols) * 40),
                    coloraxis_colorbar=dict(title="r"),
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Not enough columns of the required type for this chart.")

    # =====================================================
    # TAB 4 — QUERY ENGINE
    # =====================================================

    with tab4:
        st.subheader("🔍 Query Engine")
        st.caption("Ask questions in plain English — e.g. 'average salary by department', 'total sales per region', 'count of students by gender'")

        query = st.text_input("Ask a question about your data", placeholder="e.g. average math score by gender")

        # Quick example buttons
        example_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        cat_cols     = filtered_df.select_dtypes(exclude="number").columns.tolist()
        if example_cols and cat_cols:
            ex1 = f"average {example_cols[0]} by {cat_cols[0]}"
            ex2 = f"maximum {example_cols[0]}"
            ex3 = f"count by {cat_cols[0]}"
            e1, e2, e3 = st.columns(3)
            if e1.button(f"📊 {ex1}", use_container_width=True): query = ex1
            if e2.button(f"📈 {ex2}", use_container_width=True): query = ex2
            if e3.button(f"🔢 {ex3}", use_container_width=True): query = ex3

        if query:
            op, target, group = parse_query(query, filtered_df)
            raw_result        = execute_query(filtered_df, op, target, group)

            # Support both old (returns value directly) and new (returns dict) query_engine
            if isinstance(raw_result, dict):
                result_dict = raw_result
            else:
                result_dict = {
                    "result":     raw_result,
                    "error":      None if raw_result != "Invalid query" else "Could not understand query.",
                    "query_desc": f"{op}({target})" if op and target else "",
                }

            error      = result_dict.get("error")
            result     = result_dict.get("result")
            query_desc = result_dict.get("query_desc", "")

            st.divider()

            if error:
                st.error(f"❌ {error}")
                st.info("💡 Try queries like: 'average [column] by [column]', 'total [column]', 'count by [column]'")
            else:
                st.success(f"✅ {query_desc}")

                # Show result
                if isinstance(result, pd.Series):
                    result_df = result.reset_index()
                    result_df.columns = [group, f"{op}({target})"]
                    result_df = result_df.sort_values(f"{op}({target})", ascending=False)

                    # Table + chart side by side
                    col_tbl, col_chart = st.columns([1, 2])
                    with col_tbl:
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                    with col_chart:
                        fig = px.bar(
                            result_df,
                            x=group,
                            y=f"{op}({target})",
                            color=group,
                            template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            text_auto=".2s",
                        )
                        fig.update_layout(showlegend=False, height=350, xaxis_tickangle=-30)
                        fig.update_traces(textposition="outside")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    try:
                        display_val = f"{float(result):,.2f}" if not isinstance(result, str) else result
                    except Exception:
                        display_val = str(result)
                    st.metric(label=query_desc, value=display_val)

                # Insights
                insights = generate_query_insight(result_dict, target, group)
                if insights:
                    st.divider()
                    st.subheader("💡 Insights")
                    for ins in insights:
                        st.info(ins)

    # =====================================================
    # TAB 5 — ML PREDICTION
    # =====================================================

    with tab5:
        st.subheader("🤖 ML Prediction")

        all_targets = filtered_df.columns.tolist()
        target      = st.selectbox("Select Target Column to Predict", all_targets)

        if target:
            task_hint = detect_task_type(filtered_df[target])
            st.caption(f"Detected task: **{task_hint}** — {'predicting a number' if task_hint == 'regression' else 'predicting a category'}")

        if st.button("🚀 Train & Compare Models", type="primary"):
            with st.spinner("Training multiple models and selecting the best one..."):
                ml_result = train_prediction_model(filtered_df, target)

            if ml_result.get("error"):
                st.error(f"❌ {ml_result['error']}")
            else:
                st.session_state["ml_result"] = ml_result
                st.session_state["ml_target"] = target

        if "ml_result" in st.session_state and st.session_state.get("ml_target") == target:
            ml_result = st.session_state["ml_result"]

            st.divider()

            # ── Best model banner ────────────────────────────────
            st.success(f"🏆 Best Model: **{ml_result['best_model_name']}** | Task: {ml_result['task_type'].title()}")

            # ── Metrics ──────────────────────────────────────────
            metrics = ml_result["metrics"]
            metric_cols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                metric_cols[i].metric(k, f"{v:.4f}")

            st.caption(f"Trained on {ml_result['n_train']} rows, tested on {ml_result['n_test']} rows | {ml_result['cv_folds']}-fold cross-validation")

            st.divider()

            # ── Model comparison table + chart ───────────────────
            col_cmp, col_imp = st.columns(2)

            with col_cmp:
                st.subheader("📊 Model Comparison")
                cmp_df = pd.DataFrame(ml_result["model_comparison"])
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            with col_imp:
                st.subheader("🎯 Feature Importance")
                if ml_result["feature_importance"]:
                    imp_df = pd.DataFrame(ml_result["feature_importance"]).head(10)
                    fig_imp = px.bar(
                        imp_df.sort_values("Importance"),
                        x="Importance", y="Feature",
                        orientation="h",
                        template="plotly_white",
                        color="Importance",
                        color_continuous_scale="Blues",
                        text_auto=".1f",
                    )
                    fig_imp.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
                    fig_imp.update_traces(textposition="outside")
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")

            st.divider()

            # ── Predict on new input ─────────────────────────────
            st.subheader("🔮 Make a Prediction")
            st.caption("Fill in the values below to predict the target.")

            feature_names  = ml_result["feature_names"]
            label_encoders = ml_result["label_encoders"]
            input_values   = {}

            # Dynamic input form — 3 columns for clean layout
            n_feats  = len(feature_names)
            n_cols   = min(3, n_feats)
            form_cols = st.columns(n_cols)

            for i, feat in enumerate(feature_names):
                col_ui = form_cols[i % n_cols]
                if feat in label_encoders:
                    # Categorical — show selectbox with known classes
                    classes = list(label_encoders[feat].classes_)
                    input_values[feat] = col_ui.selectbox(feat, classes, key=f"ml_input_{feat}")
                else:
                    # Numeric — show number input with sensible default
                    col_data   = filtered_df[feat].dropna()
                    default_val = float(col_data.median()) if len(col_data) > 0 else 0.0
                    min_val    = float(col_data.min())    if len(col_data) > 0 else 0.0
                    max_val    = float(col_data.max())    if len(col_data) > 0 else 100.0
                    input_values[feat] = col_ui.number_input(
                        feat,
                        value=default_val,
                        min_value=min_val,
                        max_value=max_val,
                        key=f"ml_input_{feat}"
                    )

            if st.button("🎯 Predict", type="primary"):
                try:
                    pred = predict_single(ml_result, input_values)
                    if ml_result["task_type"] == "regression":
                        try:
                            st.success(f"### Predicted **{target}**: `{float(pred):,.2f}`")
                        except Exception:
                            st.success(f"### Predicted **{target}**: `{pred}`")
                    else:
                        st.success(f"### Predicted **{target}**: `{pred}`")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

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