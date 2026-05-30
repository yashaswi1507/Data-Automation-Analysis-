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
from export_engine import export_html, export_pdf, export_ppt


# ─────────────────────────────────────────────────────────────
# HELPER — Save chart from Visualization Studio to Dashboard
# ─────────────────────────────────────────────────────────────

def _fig_to_json(fig):
    """Serialize plotly fig to JSON string — much lighter than keeping fig object."""
    return fig.to_json()

def _json_to_fig(json_str):
    """Deserialize plotly fig from JSON string."""
    import plotly.io as pio
    return pio.from_json(json_str)

def _save_to_dashboard_btn(fig, title, chart_type, all_charts_count=0):
    """Renders a small 'Save to Dashboard' button below a chart."""
    if "dashboard_charts" not in st.session_state:
        st.session_state["dashboard_charts"] = []

    btn_key = f"save_dash_{title}_{chart_type}_{all_charts_count}"
    if st.button(f"📌 Save to Dashboard", key=btn_key, help="Add this chart to Auto Dashboard"):
        new_id  = int(pd.Timestamp.now().timestamp() * 1000)
        already = any(c["title"] == title for c in st.session_state["dashboard_charts"])
        if already:
            st.toast(f"'{title}' is already in the dashboard.", icon="ℹ️")
        else:
            st.session_state["dashboard_charts"].append({
                "id":         new_id,
                "title":      title,
                "fig_json":   _fig_to_json(fig),   # store as JSON not object
                "chart_type": chart_type,
                "pinned":     True,
                "source":     "studio",
            })
            st.toast(f"✅ '{title}' saved to Auto Dashboard!", icon="📌")


def _suggest_target_column(df):
    """
    Suggest best target column for ML using predictability scoring.
    Trains a quick model on each candidate column and picks the most predictable one.
    Returns (column_name, reason, scores_dict) or (None, "", {}).
    """
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings("ignore")

    if df.empty or df.shape[1] < 2:
        return None, "", {}

    n_rows = len(df)

    # Use sample for speed — max 300 rows for quick scoring
    sample_df = df.sample(min(300, n_rows), random_state=42) if n_rows > 300 else df.copy()

    candidates = []

    for col in df.columns:
        s        = df[col].dropna()
        miss_pct = df[col].isnull().mean()
        n_unique = s.nunique()

        # Hard filters — skip bad targets
        if miss_pct > 0.4:       continue  # too many missing
        if n_unique <= 1:        continue  # constant
        if n_unique == n_rows:   continue  # unique ID column

        # Categorical: skip high-cardinality (names, addresses)
        if not pd.api.types.is_numeric_dtype(s):
            if n_unique > 20:    continue
            if n_unique < 2:     continue

        reasons = []

        # ── Quality signals ───────────────────────────────────
        quality_score = 0

        # Missing % penalty
        if miss_pct < 0.05:
            quality_score += 20
            reasons.append("very few missing values")
        elif miss_pct < 0.20:
            quality_score += 10

        # Class balance check (classification)
        if not pd.api.types.is_numeric_dtype(s) and n_unique >= 2:
            vc      = s.value_counts(normalize=True)
            min_cls = vc.min()
            if min_cls >= 0.15:
                quality_score += 15
                reasons.append(f"balanced classes ({n_unique} classes)")
            elif min_cls >= 0.05:
                quality_score += 5

        # Variance check (regression)
        if pd.api.types.is_numeric_dtype(s):
            cv = abs(s.std() / s.mean()) if s.mean() != 0 else 0
            if 0.05 < cv < 3.0:
                quality_score += 15
                reasons.append("good variance")

        # ── Predictability score — quick Random Forest ────────
        pred_score = 0.0
        try:
            # Build feature matrix — all cols except current target
            feat_cols = [c for c in sample_df.columns if c != col]
            X = sample_df[feat_cols].copy()
            y = sample_df[col].copy()

            # Drop rows where target is missing
            mask = y.notna()
            X, y = X[mask], y[mask]

            if len(y) < 20:
                raise ValueError("too few rows")

            # Encode categorical features
            for fc in X.select_dtypes(include=["object","category"]).columns:
                le = LabelEncoder()
                X[fc] = le.fit_transform(X[fc].astype(str))

            # Impute
            imp = SimpleImputer(strategy="median")
            X_arr = imp.fit_transform(X)

            # Encode target if categorical
            is_clf = not pd.api.types.is_numeric_dtype(y)
            if is_clf:
                le_y = LabelEncoder()
                y    = le_y.fit_transform(y.astype(str))

            # Quick cross-val (2-fold for speed)
            cv_folds = min(3, len(y) // 10)
            if cv_folds < 2:
                raise ValueError("not enough data for CV")

            if is_clf:
                model   = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
                scoring = "accuracy"
            else:
                model   = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
                scoring = "r2"

            scores    = cross_val_score(model, X_arr, y, cv=cv_folds, scoring=scoring)
            pred_score = max(float(scores.mean()), 0.0)

            # Label the predictability
            if pred_score >= 0.7:
                reasons.append(f"highly predictable (score={pred_score:.2f})")
            elif pred_score >= 0.4:
                reasons.append(f"moderately predictable (score={pred_score:.2f})")
            else:
                reasons.append(f"low predictability (score={pred_score:.2f})")

        except Exception:
            pred_score = 0.0

        total_score = quality_score + (pred_score * 100)
        candidates.append((col, total_score, pred_score, reasons))

    if not candidates:
        return None, "", {}

    # Sort by total score
    candidates.sort(key=lambda x: x[1], reverse=True)

    best_col, best_total, best_pred, best_reasons = candidates[0]

    # Build scores dict for display
    scores_dict = {
        c[0]: round(c[2], 3)
        for c in candidates
    }

    reason_str = " | ".join(best_reasons[:2]) if best_reasons else "best candidate in dataset"
    return best_col, reason_str, scores_dict


def _filter_useful_charts(charts, df):
    """
    Remove auto-generated charts that make no sense:
    - ID vs anything (high cardinality unique cols)
    - Name vs anything
    - Two identical columns
    """
    id_like_cols = set()
    for col in df.columns:
        s = df[col].dropna()
        if s.nunique() / max(len(s), 1) > 0.9:
            id_like_cols.add(col.lower())

    name_keywords = {"name","id","index","key","code","no","num","serial"}

    useful = []
    for chart in charts:
        title = chart.get("title","").lower()
        skip  = False
        for kw in name_keywords:
            if kw in title.split():
                skip = True
                break
        for col in id_like_cols:
            if col in title:
                skip = True
                break
        if not skip:
            useful.append(chart)

    # If all filtered out, return original (better than empty dashboard)
    return useful if useful else charts

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Data Analysis Automation Tool", layout="wide")

st.title("Data Analysis Automation Tool")
st.markdown("Upload data, clean it, visualize it, query it, and predict insights using AI-powered analytics.")

# =========================================================
# FILE UPLOAD
# =========================================================

# Session state for stable file handling
if "raw_df" not in st.session_state:
    st.session_state["raw_df"]      = None
if "loaded_file_name" not in st.session_state:
    st.session_state["loaded_file_name"] = None
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

col_upload, col_url = st.columns([1, 1])

with col_upload:
    file = st.file_uploader(
        "📂 Upload Dataset",
        type=["csv","xlsx","xls","json","zip"],
        help="Supported formats: CSV, Excel (.xlsx/.xls), JSON, ZIP",
    )

with col_url:
    dataset_url = st.text_input(
        "🔗 Or paste a URL",
        placeholder="https://example.com/data.csv  |  kaggle.com/datasets/...",
        help="Paste a direct CSV/Excel URL, Kaggle dataset URL, or JSON API endpoint",
    )
    if dataset_url:
        # Basic URL validation
        if not (dataset_url.startswith("http://") or dataset_url.startswith("https://")):
            st.warning("⚠️ URL must start with http:// or https://")
            dataset_url = ""

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
    # Only reload if new file uploaded (prevent reset on same file)
    if file.name != st.session_state.get("loaded_file_name"):
        with st.spinner(f"📂 Loading {file.name}..."):
            loaded = None

            if file.name.endswith(".csv"):
                loaded = load_csv_safely(file)
                if loaded is None:
                    st.error("❌ Could not read CSV. Check if file is valid.")

            elif file.name.endswith((".xlsx",".xls")):
                try:
                    file.seek(0)
                    engine = "openpyxl" if file.name.endswith(".xlsx") else "xlrd"
                    loaded = pd.read_excel(file, engine=engine)
                    loaded.columns = loaded.columns.astype(str).str.strip()
                except Exception as e:
                    st.error(f"❌ Excel Error: {e}")

            elif file.name.endswith(".json"):
                try:
                    loaded = pd.read_json(file)
                except:
                    st.error("❌ Invalid JSON file.")

            elif file.name.endswith(".zip"):
                try:
                    zf = zipfile.ZipFile(file)
                    for name in zf.namelist():
                        if name.endswith(".csv"):
                            with zf.open(name) as f: loaded = pd.read_csv(f); break
                except:
                    st.error("❌ Could not read ZIP file.")

            if loaded is not None:
                st.session_state["raw_df"]           = loaded
                st.session_state["loaded_file_name"] = file.name
                st.session_state["data_loaded"]      = True
                # Clear previous session on new file
                for key in ["dashboard_generated","ml_result","ml_target","last_prediction",
                            "dashboard_charts","dashboard_ml_insights","dashboard_query_insights"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.toast(f"✅ {file.name} loaded successfully!", icon="📂")

    raw_df = st.session_state.get("raw_df")

elif dataset_url:
    if st.button("🔍 Load Data from URL", type="primary"):
        with st.spinner("Fetching data..."):
            loaded = universal_data_loader(dataset_url)
        if loaded is not None:
            st.session_state["raw_df"]           = loaded
            st.session_state["loaded_file_name"] = dataset_url[:50]
            st.session_state["data_loaded"]      = True
            st.toast("✅ Data loaded from URL!", icon="🔗")
        else:
            st.error("❌ Could not load data. Check the URL format.")
            st.info("💡 Expected formats: direct .csv/.xlsx URL, Kaggle dataset URL, or JSON API")
    raw_df = st.session_state.get("raw_df")

else:
    raw_df = st.session_state.get("raw_df")

# =========================================================
# MAIN APP
# =========================================================

if raw_df is not None:

    # ── Edge case: empty dataset ──────────────────────────────
    if raw_df.empty:
        st.error("⚠️ The uploaded file is empty. Please upload a file with data.")
        st.stop()

    if raw_df.shape[1] == 0:
        st.error("⚠️ No columns found in the file. Please check the file format.")
        st.stop()

    if raw_df.shape[0] < 2:
        st.warning("⚠️ Dataset has very few rows — some features may not work correctly.")

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
    # DATASET PROFILING + PREPROCESSING  (cached)
    # =====================================================

    @st.cache_data(show_spinner=False)
    def run_profiler(df_hash, _df):
        profiler        = DatasetProfiler(_df)
        dataset_profile = profiler.detect_dataset_type()
        column_profiles = profiler.profile_columns()
        return dataset_profile, column_profiles

    @st.cache_data(show_spinner=False)
    def run_preprocessing(df_hash, _df, outlier_option, missing_option, _dataset_profile, _column_profiles):
        processor = DataPreprocessor(_df, outlier_option, missing_option, _dataset_profile, _column_profiles)
        return processor.process()

    # Use df shape+columns as cache key (avoids hashing large df)
    df_hash = f"{df.shape}_{list(df.columns)}_{df.dtypes.to_dict()}"

    dataset_profile = "general"
    column_profiles = {}

    with st.spinner("🔍 Analysing dataset..."):
        try:
            dataset_profile, column_profiles = run_profiler(df_hash, df)
        except Exception as e:
            st.warning(f"Profiler Warning: {e}")

    with st.spinner("🧹 Cleaning data..."):
        try:
            clean_df, report = run_preprocessing(df_hash, df, outlier_option, missing_option, dataset_profile, column_profiles)
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            clean_df, report = df.copy(), []

    # =====================================================
    # SIDEBAR FILTERS
    # =====================================================

    st.sidebar.subheader("Filters")
    filtered_df = clean_df.copy()

    try:
        cat_filter_cols = [
            c for c in clean_df.select_dtypes(include="object").columns
            if 2 <= clean_df[c].nunique() <= 30
        ][:6]  # max 6 filters

        for col in cat_filter_cols:
            try:
                unique_vals = clean_df[col].dropna().unique().tolist()
                selected    = st.sidebar.selectbox(
                    f"Filter: {col}",
                    ["All"] + sorted([str(v) for v in unique_vals]),
                    key=f"filter_{col}",
                )
                if selected != "All":
                    temp_df = filtered_df[filtered_df[col].astype(str) == selected]
                    if len(temp_df) == 0:
                        st.sidebar.warning(f"No data for {col} = {selected}")
                    else:
                        filtered_df = temp_df
            except Exception:
                continue
    except Exception:
        pass

    # =====================================================
    # FEEDBACK — Sidebar
    # =====================================================

    st.sidebar.divider()
    st.sidebar.subheader("💬 Feedback")

    with st.sidebar.form("feedback_form", clear_on_submit=True):
        feedback_type = st.selectbox(
            "Type",
            ["🐛 Bug Report", "💡 Feature Request", "👍 Compliment", "🤔 Other"],
        )
        feedback_text = st.text_area("Your feedback", placeholder="Tell us what you think...", height=100)
        submitted     = st.form_submit_button("Send", use_container_width=True)

        if submitted:
            if feedback_text.strip():
                # Save to session state log
                if "feedback_log" not in st.session_state:
                    st.session_state["feedback_log"] = []

                import datetime
                st.session_state["feedback_log"].append({
                    "time":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "type":    feedback_type,
                    "message": feedback_text.strip(),
                })

                # Send email notification
                try:
                    import smtplib
                    from email.mime.text import MIMEText
                    from email.mime.multipart import MIMEMultipart
                    import os

                    sender_email   = os.environ.get("FEEDBACK_EMAIL")
                    sender_pass    = os.environ.get("FEEDBACK_PASSWORD")
                    receiver_email = os.environ.get("FEEDBACK_EMAIL")

                    if sender_email and sender_pass:
                        msg = MIMEMultipart()
                        msg["From"]    = sender_email
                        msg["To"]      = receiver_email
                        msg["Subject"] = f"[DataTool Feedback] {feedback_type}"

                        body = f"""
New feedback received!

Type: {feedback_type}
Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

Message:
{feedback_text.strip()}
"""
                        msg.attach(MIMEText(body, "plain"))

                        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                            server.login(sender_email, sender_pass)
                            server.sendmail(sender_email, receiver_email, msg.as_string())

                except Exception:
                    pass  # Email fail hone pe app crash nahi hoga

                # Also save to CSV as backup
                try:
                    import csv, os
                    fb_file     = "feedback.csv"
                    file_exists = os.path.exists(fb_file)
                    with open(fb_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["time","type","message"])
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow({
                            "time":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "type":    feedback_type,
                            "message": feedback_text.strip(),
                        })
                except Exception:
                    pass

                st.sidebar.success("✅ Thanks for your feedback!")
            else:
                st.sidebar.warning("Please write something before sending.")

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard", "📋 Summary", "📈 Visualization Studio",
        "🔍 Query Engine", "🤖 ML Prediction", "⚡ Auto Dashboard", "📁 My Reports"
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
        try:
            numeric_cols  = filtered_df.select_dtypes(include="number").columns.tolist()
            raw_missing   = int(raw_df.isnull().sum().sum())
            clean_missing = int(clean_df.isnull().sum().sum())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Rows",            len(filtered_df))
            c2.metric("Columns",         filtered_df.shape[1])
            c3.metric("Missing (Raw)",   f"{raw_missing:,}")
            c4.metric("Missing (Clean)", f"{clean_missing:,}", delta=f"-{raw_missing - clean_missing:,}", delta_color="inverse")
            c5.metric("Numeric Cols",    len(numeric_cols))
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")

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
        show_summary(filtered_df, st)

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
            use_groupby = st.checkbox(
                "📊 Compare by Category",
                disabled=not bool(categorical_cols),
                help="Split your chart by a category — e.g. show sales by region, scores by gender" if categorical_cols else "No categorical columns available",
            )
            group_col   = None
            agg_func    = "sum"
            if use_groupby and categorical_cols:
                group_col = st.selectbox(
                    "Compare by",
                    categorical_cols,
                    key="grp_col",
                    help="Your chart will be split by this category"
                )
                agg_func = st.selectbox(
                    "Calculate",
                    ["mean","sum","max","min","count"],
                    key="grp_agg",
                    format_func=lambda x: {
                        "mean":  "Average",
                        "sum":   "Total",
                        "max":   "Highest",
                        "min":   "Lowest",
                        "count": "Count (rows)",
                    }[x],
                )

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
                try:
                    st.plotly_chart(fig, use_container_width=True)
                    _save_to_dashboard_btn(fig, f"Bar: {y_col} by {grp}", "bar", all_charts_count=len(st.session_state.get("dashboard_charts",[])))
                except Exception as e:
                    st.error(f"Chart error: {e}")

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
                try:
                    st.plotly_chart(fig, use_container_width=True)
                    _save_to_dashboard_btn(fig, f"Line: {y_col} over {x_use}", "line", all_charts_count=len(st.session_state.get("dashboard_charts",[])))
                except Exception as e:
                    st.error(f"Chart error: {e}")

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

                )
                fig.update_layout(height=480)
                fig.update_traces(marker=dict(size=6))
                try:
                    st.plotly_chart(fig, use_container_width=True)
                    _save_to_dashboard_btn(fig, f"Scatter: {x_col} vs {y_col}", "scatter", all_charts_count=len(st.session_state.get("dashboard_charts",[])))
                except Exception as e:
                    st.error(f"Chart error: {e}")

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
                _save_to_dashboard_btn(fig, f"Histogram: {col}", "histogram", all_charts_count=len(st.session_state.get("dashboard_charts",[])))

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
                _save_to_dashboard_btn(fig, f"Box: {col}", "box", all_charts_count=len(st.session_state.get("dashboard_charts",[])))

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
                _save_to_dashboard_btn(fig, f"Pie: {pie_col}", "pie", all_charts_count=len(st.session_state.get("dashboard_charts",[])))

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
                _save_to_dashboard_btn(fig, "Correlation Heatmap", "heatmap", all_charts_count=len(st.session_state.get("dashboard_charts",[])))

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
                num_cols_q = filtered_df.select_dtypes(include="number").columns.tolist()
                cat_cols_q = filtered_df.select_dtypes(exclude="number").columns.tolist()
                nc = num_cols_q[0] if num_cols_q else "value"
                cc = cat_cols_q[0] if cat_cols_q else "category"
                st.info(f"💡 Try: 'average {nc} by {cc}'  |  'total {nc}'  |  'count by {cc}'")
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

                # Send to Dashboard
                if st.button("📤 Send to Auto Dashboard", key=f"query_to_dash_{query}"):
                    if "dashboard_charts" not in st.session_state:
                        st.session_state["dashboard_charts"] = []

                    # Build chart from result
                    if isinstance(result, pd.Series) and len(result) > 0:
                        q_df   = result.reset_index()
                        q_df.columns = [group, f"{op}({target})"]
                        fig_q  = px.bar(
                            q_df.sort_values(f"{op}({target})", ascending=False),
                            x=group, y=f"{op}({target})",
                            color=group,
                            template="plotly_white",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            text_auto=".2s",
                            title=query_desc,
                        )
                        fig_q.update_layout(showlegend=False, height=380, xaxis_tickangle=-30)
                        fig_q.update_traces(textposition="outside")

                        chart_entry = {
                            "id":         int(pd.Timestamp.now().timestamp() * 1000),
                            "title":      f"🔍 {query_desc}",
                            "fig_json":   fig_q.to_json(),
                            "chart_type": "bar",
                            "pinned":     True,
                            "source":     "query",
                        }
                        existing = {c["title"] for c in st.session_state["dashboard_charts"]}
                        if chart_entry["title"] not in existing:
                            st.session_state["dashboard_charts"].append(chart_entry)

                    # Store query insight
                    if "dashboard_query_insights" not in st.session_state:
                        st.session_state["dashboard_query_insights"] = []
                    q_ins = f"🔍 Query: '{query}' → {query_desc}"
                    if insights:
                        q_ins += " | " + " | ".join([i.replace("**","").replace("*","") for i in insights])
                    st.session_state["dashboard_query_insights"].append(q_ins)

                    st.success("✅ Sent to **⚡ Auto Dashboard**! Go there to review and save to report.")

    # =====================================================
    # TAB 5 — ML PREDICTION
    # =====================================================

    with tab5:
        st.subheader("🤖 ML Prediction")

        all_targets = filtered_df.columns.tolist()

        if not all_targets:
            st.warning("No columns found. Please upload a valid dataset.")
            target = None
        else:
            # ── Smart column suggestion ───────────────────────
            with st.spinner("🔍 Finding best target column..."):
                suggested, suggest_reason, scores_dict = _suggest_target_column(filtered_df)

            if suggested:
                st.info(f"💡 **Suggested target:** `{suggested}`  |  {suggest_reason}")
                default_idx = all_targets.index(suggested) if suggested in all_targets else 0

                # Show predictability scores for all candidates
                if scores_dict:
                    with st.expander("📊 Predictability scores for all columns", expanded=False):
                        import pandas as pd
                        sc_df = (
                            pd.DataFrame(list(scores_dict.items()), columns=["Column","Score"])
                            .sort_values("Score", ascending=False)
                            .reset_index(drop=True)
                        )
                        sc_df["Score"] = sc_df["Score"].apply(
                            lambda x: f"{x:.2f} {'🟢' if x>=0.7 else '🟡' if x>=0.4 else '🔴'}"
                        )
                        st.dataframe(sc_df, use_container_width=True, hide_index=True)
                        st.caption("Score = how well other columns can predict this one (0=unpredictable, 1=perfect)")
            else:
                default_idx = 0

            target = st.selectbox(
                "Select Target Column to Predict",
                all_targets,
                index=default_idx,
                help="Pre-selected based on predictability scoring. You can change this."
            )

        if target:
            task_hint = detect_task_type(filtered_df[target])
            st.caption(f"Detected task: **{task_hint}** — {'predicting a number' if task_hint == 'regression' else 'predicting a category'}")

        if target and st.button("🚀 Train & Compare Models", type="primary",
                               help="Trains multiple models and picks the best one automatically"):
            try:
                with st.spinner("⏳ Training models... please wait, do not change inputs."):
                    ml_result = train_prediction_model(filtered_df, target)
            except Exception as e:
                ml_result = {"error": f"Unexpected error: {e}"}

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

                    # Format prediction
                    try:
                        pred_display = f"{float(pred):,.2f}" if ml_result["task_type"] == "regression" else str(pred)
                    except:
                        pred_display = str(pred)

                    st.success(f"### Predicted **{target}**: `{pred_display}`")

                    # ── Interpret the prediction ──────────────────────
                    try:
                        col_data = filtered_df[target].dropna()

                        if ml_result["task_type"] == "regression":
                            pred_val   = float(pred)
                            col_min    = float(col_data.min())
                            col_max    = float(col_data.max())
                            col_mean   = float(col_data.mean())
                            col_std    = float(col_data.std())
                            col_range  = col_max - col_min
                            percentile = float((col_data < pred_val).mean() * 100)

                            st.divider()
                            st.subheader("🧠 Result Interpretation")

                            # User defines what HIGH means for this column
                            high_means = st.radio(
                                f"For **{target}**, a HIGH value means:",
                                ["⚠️ Bad (e.g. Risk, Error, Loss, Debt)",
                                 "✅ Good (e.g. Sales, Score, Profit, Performance)"],
                                horizontal=True,
                                key="high_means_radio",
                            )
                            bad_is_high = "Bad" in high_means

                            # Compute zone
                            if col_range > 0:
                                low_thresh  = col_min + col_range * 0.33
                                high_thresh = col_min + col_range * 0.67

                                if pred_val <= low_thresh:
                                    zone = "Low"
                                elif pred_val <= high_thresh:
                                    zone = "Medium"
                                else:
                                    zone = "High"
                            else:
                                zone = "Medium"

                            # Verdict depends on user's context
                            if bad_is_high:
                                verdict_map = {
                                    "Low":    ("🟢 Low Risk",    "success", "This is a GOOD result — low concern."),
                                    "Medium": ("🟡 Medium Risk", "warning", "This needs attention — moderate concern."),
                                    "High":   ("🔴 High Risk",   "error",   "This is a BAD result — high concern, action needed."),
                                }
                            else:
                                verdict_map = {
                                    "Low":    ("🔴 Low Performance", "error",   "This is a POOR result — below expectations."),
                                    "Medium": ("🟡 Average",         "warning", "This is an AVERAGE result — room for improvement."),
                                    "High":   ("🟢 High Performance","success", "This is a GREAT result — above expectations."),
                                }

                            verdict_label, verdict_type, verdict_msg = verdict_map[zone]

                            # Show metrics
                            i1, i2, i3 = st.columns(3)
                            i1.metric("Min in Dataset",  f"{col_min:,.2f}")
                            i2.metric("Max in Dataset",  f"{col_max:,.2f}")
                            i3.metric("Average",         f"{col_mean:,.2f}")

                            # Verdict box
                            msg = f"**{verdict_label}** — {verdict_msg}"
                            if verdict_type == "success":
                                st.success(msg)
                            elif verdict_type == "warning":
                                st.warning(msg)
                            else:
                                st.error(msg)

                            # Progress bar
                            if col_range > 0:
                                progress_val = min(max((pred_val - col_min) / col_range, 0.0), 1.0)
                                st.markdown(f"**Where {pred_display} falls in the data range:**")
                                st.progress(progress_val)
                                st.caption(
                                    f"📍 **{pred_display}** is higher than **{percentile:.0f}%** of all values in your dataset "
                                    f"(range: {col_min:,.1f} → {col_max:,.1f})"
                                )

                        else:
                            # Classification
                            st.divider()
                            st.subheader("🧠 Result Interpretation")

                            vc             = col_data.value_counts(normalize=True) * 100
                            total_classes  = len(vc)
                            vc_counts      = col_data.value_counts()

                            st.markdown(f"**Predicted:** `{pred_display}`")

                            if str(pred_display) in vc.index.astype(str).tolist():
                                pred_freq  = vc[vc.index.astype(str) == str(pred_display)].iloc[0]
                                pred_count = vc_counts[vc_counts.index.astype(str) == str(pred_display)].iloc[0]
                                st.info(
                                    f"In your dataset, **{pred_freq:.1f}%** of people "
                                    f"({int(pred_count)} out of {len(col_data)}) "
                                    f"belong to the **'{pred_display}'** category."
                                )

                            # Class distribution bar chart
                            dist_df = vc.reset_index()
                            dist_df.columns = [target, "% in Dataset"]
                            dist_df["% in Dataset"] = dist_df["% in Dataset"].round(1)
                            dist_df["Is Predicted"]  = dist_df[target].astype(str) == str(pred_display)

                            fig_dist = px.bar(
                                dist_df,
                                x=target, y="% in Dataset",
                                color="Is Predicted",
                                color_discrete_map={True: "#2ecc71", False: "#bdc3c7"},
                                template="plotly_white",
                                text_auto=".1f",
                                title=f"Where '{pred_display}' sits among all {target} categories",
                            )
                            fig_dist.update_layout(
                                height=320, showlegend=False, xaxis_tickangle=-30
                            )
                            fig_dist.update_traces(textposition="outside")
                            st.plotly_chart(fig_dist, use_container_width=True)

                    except Exception:
                        pass

                    # Store prediction in session state
                    st.session_state["last_prediction"] = {
                        "target":        target,
                        "prediction":    pred_display,
                        "input_values":  dict(input_values),
                        "model_name":    ml_result["best_model_name"],
                        "task_type":     ml_result["task_type"],
                        "metrics":       ml_result["metrics"],
                        "feature_importance": ml_result["feature_importance"][:8],
                        "model_comparison":   ml_result["model_comparison"],
                    }

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            # ── Send Prediction to Auto Dashboard ────────────────
            if "last_prediction" in st.session_state and st.session_state.get("ml_target") == target:
                lp = st.session_state["last_prediction"]

                st.divider()
                if st.button("📤 Send to Auto Dashboard", key="ml_to_dashboard", type="primary"):
                    if "dashboard_charts" not in st.session_state:
                        st.session_state["dashboard_charts"] = []

                    new_charts = []

                    # Feature importance chart
                    if lp["feature_importance"]:
                        imp_df  = pd.DataFrame(lp["feature_importance"])
                        fig_imp = px.bar(
                            imp_df.sort_values("Importance"),
                            x="Importance", y="Feature",
                            orientation="h",
                            template="plotly_white",
                            color="Importance",
                            color_continuous_scale="Blues",
                            text_auto=".1f",
                            title=f"Feature Importance — {target}",
                        )
                        fig_imp.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
                        new_charts.append({
                            "id":         int(pd.Timestamp.now().timestamp() * 1000),
                            "title":      f"🤖 Feature Importance — {target}",
                            "fig_json":   fig_imp.to_json(),
                            "chart_type": "bar",
                            "pinned":     True,
                            "source":     "ml",
                        })

                    # Model comparison chart
                    cmp_df  = pd.DataFrame(lp["model_comparison"])
                    fig_cmp = px.bar(
                        cmp_df.sort_values("CV Score (mean)", ascending=True),
                        x="CV Score (mean)", y="Model",
                        orientation="h",
                        template="plotly_white",
                        color="CV Score (mean)",
                        color_continuous_scale="Greens",
                        text_auto=".3f",
                        title="Model Comparison (CV Score)",
                    )
                    fig_cmp.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
                    new_charts.append({
                        "id":         int(pd.Timestamp.now().timestamp() * 1000) + 1,
                        "title":      "🤖 Model Comparison",
                        "fig_json":   fig_cmp.to_json(),
                        "chart_type": "bar",
                        "pinned":     True,
                        "source":     "ml",
                    })

                    # Store ML metadata as dashboard insight
                    input_str = ", ".join([f"{k}={v}" for k,v in lp["input_values"].items()])
                    ml_insight = (
                        f"🎯 ML Prediction — {lp['target']} = {lp['prediction']} "
                        f"| Model: {lp['model_name']} "
                        f"| {', '.join([f'{k}: {v:.4f}' for k,v in lp['metrics'].items()])} "
                        f"| Inputs: {input_str}"
                    )
                    if "dashboard_ml_insights" not in st.session_state:
                        st.session_state["dashboard_ml_insights"] = []
                    st.session_state["dashboard_ml_insights"].append(ml_insight)

                    # Add to dashboard (avoid duplicates by title)
                    existing_titles = {c["title"] for c in st.session_state["dashboard_charts"]}
                    for c in new_charts:
                        if c["title"] not in existing_titles:
                            st.session_state["dashboard_charts"].append(c)

                    st.success("✅ Sent to **⚡ Auto Dashboard**! Go there to review and save to report.")

    # =====================================================
    # TAB 6 — AUTO DASHBOARD
    # =====================================================

    with tab6:
        st.subheader("⚡ Hybrid Auto Dashboard")

        # ── Session state init ────────────────────────────────────
        if "dashboard_charts" not in st.session_state:
            st.session_state["dashboard_charts"] = []
        if "saved_reports" not in st.session_state:
            st.session_state["saved_reports"] = {}
        if "dashboard_generated" not in st.session_state:
            st.session_state["dashboard_generated"] = False
        if "dashboard_ml_insights" not in st.session_state:
            st.session_state["dashboard_ml_insights"] = []
        if "dashboard_query_insights" not in st.session_state:
            st.session_state["dashboard_query_insights"] = []

        # ── KPIs ─────────────────────────────────────────────────
        metrics     = generate_kpis(filtered_df)
        metric_keys = list(metrics.keys())
        n_kpi       = min(len(metric_keys), 5)
        kpi_cols    = st.columns(n_kpi)
        for i in range(n_kpi):
            kpi_cols[i].metric(metric_keys[i], metrics[metric_keys[i]])

        st.divider()

        # ── Generate / Regenerate auto charts ────────────────────
        col_gen, col_clr = st.columns([2, 1])
        with col_gen:
            if st.button("🔄 Generate Auto Charts", type="primary"):
                if filtered_df.empty:
                    st.warning("No data available to generate charts.")
                    auto_charts = []
                else:
                    auto_charts = generate_auto_charts(filtered_df)
                # Merge: keep studio charts, replace auto charts
                studio_charts = [
                    c for c in st.session_state["dashboard_charts"]
                    if c.get("source") == "studio"
                ]
                # Convert fig objects to JSON for memory efficiency
                for c in auto_charts:
                    if "fig" in c and "fig_json" not in c:
                        c["fig_json"] = _fig_to_json(c["fig"])
                        del c["fig"]
                st.session_state["dashboard_charts"] = auto_charts + studio_charts
                st.session_state["dashboard_generated"] = True
        with col_clr:
            if st.button("🗑️ Clear All Charts"):
                st.session_state["dashboard_charts"] = []
                st.session_state["dashboard_generated"] = False

        # Auto-generate on first load
        if not st.session_state["dashboard_generated"] and len(filtered_df) > 0:
            with st.spinner("Generating charts..."):
                auto_charts = generate_auto_charts(filtered_df)
            studio_charts = [
                c for c in st.session_state["dashboard_charts"]
                if c.get("source") == "studio"
            ]
            for c in auto_charts:
                if "fig" in c and "fig_json" not in c:
                    c["fig_json"] = _fig_to_json(c["fig"])
                    del c["fig"]
            st.session_state["dashboard_charts"] = auto_charts + studio_charts
            st.session_state["dashboard_generated"] = True

        # ── Chart grid with pin / remove controls ─────────────────
        all_charts = st.session_state.get("dashboard_charts", [])

        if all_charts:
            st.subheader(f"📊 Dashboard Charts ({len(all_charts)} total)")

            # Two charts per row
            pinned_ids = set()
            for idx in range(0, len(all_charts), 2):
                row_charts = all_charts[idx: idx + 2]
                cols       = st.columns(len(row_charts))

                for col_ui, chart in zip(cols, row_charts):
                    with col_ui:
                        # Source badge
                        badge = "🤖 Auto" if chart.get("source") == "auto" else "🎨 Studio"
                        pin_label = "📌 Pinned" if chart.get("pinned", True) else "📍 Unpinned"

                        # Title bar
                        t_col, p_col, d_col = st.columns([3, 1, 1])
                        with t_col:
                            st.caption(f"{badge}  |  {chart['title']}")
                        with p_col:
                            is_pinned = chart.get("pinned", True)
                            if st.button(
                                "📌" if is_pinned else "📍",
                                key=f"pin_{chart['id']}",
                                help="Click to unpin / pin from report",
                            ):
                                chart["pinned"] = not is_pinned
                        with d_col:
                            if st.button("✕", key=f"del_{chart['id']}", help="Remove chart"):
                                st.session_state["dashboard_charts"] = [
                                    c for c in all_charts if c["id"] != chart["id"]
                                ]
                                st.rerun()

                        _fig = _json_to_fig(chart["fig_json"]) if "fig_json" in chart else chart.get("fig")
                        if _fig:
                            st.plotly_chart(_fig, use_container_width=True, key=f"dchart_{chart['id']}")

                        if chart.get("pinned", True):
                            pinned_ids.add(chart["id"])

                        if st.button("💾 Save to Report", key=f"quick_save_{chart['id']}", use_container_width=True):
                            if "quick_report_charts" not in st.session_state:
                                st.session_state["quick_report_charts"] = []
                            already = any(c["id"] == chart["id"] for c in st.session_state.get("quick_report_charts",[]))
                            if not already:
                                st.session_state["quick_report_charts"].append(chart)
                            st.toast(f"✅ '{chart['title']}' added to report!", icon="💾")

            st.divider()

            # ── Save Report ───────────────────────────────────────
            st.subheader("💾 Save Report")
            pinned_charts = [c for c in all_charts if c.get("pinned", True)]
            st.caption(f"{len(pinned_charts)} pinned chart(s) will be saved in the report.")

            r_col1, r_col2 = st.columns([3, 1])
            with r_col1:
                report_name = st.text_input(
                    "Report name",
                    placeholder="e.g. Sales Q1 Analysis",
                    key="report_name_input",
                )
            with r_col2:
                st.write("")
                st.write("")
                if st.button("💾 Save Report", type="primary"):
                    if not report_name.strip():
                        st.warning("Please enter a report name.")
                    elif not pinned_charts:
                        st.warning("⚠️ No charts pinned! Click 📌 on charts you want, then save.")
                    else:
                        # Combine all insights
                        all_insights = (
                            st.session_state.get("dashboard_ml_insights", []) +
                            st.session_state.get("dashboard_query_insights", []) +
                            generate_insights(filtered_df)
                        )
                        st.session_state["saved_reports"][report_name.strip()] = {
                            "charts":   pinned_charts,
                            "insights": all_insights,
                            "kpis":     metrics,
                        }
                        st.success(f"✅ Report **'{report_name}'** saved with {len(pinned_charts)} chart(s)!")

        else:
            st.info("Click **Generate Auto Charts** to build your dashboard.")

        st.divider()

        # ── AI Insights (live + ML + Query) ──────────────────────
        st.subheader("💡 Insights")

        # ML prediction insights
        ml_ins_list = st.session_state.get("dashboard_ml_insights", [])
        if ml_ins_list:
            st.markdown("**🤖 ML Predictions**")
            for ins in ml_ins_list:
                st.success(ins)

        # Query insights
        q_ins_list = st.session_state.get("dashboard_query_insights", [])
        if q_ins_list:
            st.markdown("**🔍 Query Results**")
            for ins in q_ins_list:
                st.info(ins)

        # Auto insights
        st.markdown("**📊 Data Insights**")
        auto_insights = generate_insights(filtered_df)
        for ins in auto_insights:
            st.info(ins)

        # Clear insights button
        if ml_ins_list or q_ins_list:
            if st.button("🗑️ Clear ML & Query Insights"):
                st.session_state["dashboard_ml_insights"]    = []
                st.session_state["dashboard_query_insights"] = []
                st.rerun()

    # =====================================================
    # TAB 7 — MY REPORTS  (Power BI style)
    # =====================================================

    with tab7:
        saved = st.session_state.get("saved_reports", {})
        st.subheader("📁 My Reports")

        if not saved:
            st.info("""**No reports saved yet.** How to create one:
1. Go to **⚡ Auto Dashboard** tab
2. Charts auto-generate — review them
3. Click **📌** on charts you want to keep
4. Enter a name → click **💾 Save Report**
5. Come back here to view and export""")
        else:
            left_col, right_col = st.columns([1, 3])

            with left_col:
                st.markdown("### 📂 Reports")
                st.caption(f"{len(saved)} report(s) saved")
                st.divider()

                for rname in list(saved.keys()):
                    n_charts = len(saved[rname].get("charts", []))
                    if st.button(f"📄 {rname}  ·  {n_charts} chart(s)", key=f"rpt_btn_{rname}", use_container_width=True):
                        st.session_state["active_report"] = rname

                if not st.session_state.get("active_report") or st.session_state["active_report"] not in saved:
                    st.session_state["active_report"] = list(saved.keys())[0]

            with right_col:
                report_choice = st.session_state.get("active_report")
                rpt = saved[report_choice]

                h_col, del_col = st.columns([4, 1])
                with h_col:
                    st.markdown(f"## 📊 {report_choice}")
                with del_col:
                    if st.button("🗑️ Delete", key=f"del_rpt_{report_choice}"):
                        del st.session_state["saved_reports"][report_choice]
                        st.session_state.pop("active_report", None)
                        st.rerun()

                st.divider()

                # KPIs
                rpt_kpis = rpt.get("kpis", {})
                rpt_keys = list(rpt_kpis.keys())[:5]
                if rpt_keys:
                    kpi_cols = st.columns(len(rpt_keys))
                    for i, k in enumerate(rpt_keys):
                        kpi_cols[i].metric(k, rpt_kpis[k])
                    st.divider()

                # Charts 2 per row
                rpt_charts = rpt.get("charts", [])
                st.markdown(f"**{len(rpt_charts)} Chart(s)**")
                for i in range(0, len(rpt_charts), 2):
                    row  = rpt_charts[i: i + 2]
                    cols = st.columns(len(row))
                    for col_ui, chart in zip(cols, row):
                        with col_ui:
                            badge = "🤖 Auto" if chart.get("source") == "auto" else "🎨 Studio"
                            st.caption(f"{badge}  |  {chart['title']}")
                            _fig = _json_to_fig(chart["fig_json"]) if "fig_json" in chart else chart.get("fig")
                            if _fig:
                                st.plotly_chart(_fig, use_container_width=True, key=f"myreport_{report_choice}_{chart['id']}_{i}")

                st.divider()

                # Insights
                st.markdown("### 💡 Insights")
                for ins in rpt.get("insights", []):
                    st.info(ins)

                st.divider()

                # Export Report
                st.markdown("### ⬇️ Export Report")
                fmt_col, btn_col = st.columns([2,1])
                with fmt_col:
                    export_fmt = st.radio(
                        "Export format:",
                        ["HTML (interactive)", "PDF (printable)", "PPT (slides)"],
                        horizontal=True,
                        key=f"fmt_{report_choice}",
                        help="Select format, then click Generate",
                    )
                    export_fmt = export_fmt.split(" ")[0]
                with btn_col:
                    st.write("")
                    st.write("")
                    gen_export = st.button(f"📥 Generate {export_fmt}", key=f"gen_{report_choice}", type="primary")

                if gen_export:
                    try:
                        with st.spinner(f"Generating {export_fmt}..."):
                            exp_kpis   = rpt.get("kpis", {})
                            exp_charts = rpt.get("charts", [])
                            exp_ins    = rpt.get("insights", [])
                            if export_fmt == "HTML":
                                data  = export_html(report_choice, exp_kpis, exp_charts, exp_ins)
                                fname = f"{report_choice.replace(' ','_')}.html"
                                mime  = "text/html"
                            elif export_fmt == "PDF":
                                data  = export_pdf(report_choice, exp_kpis, exp_charts, exp_ins)
                                fname = f"{report_choice.replace(' ','_')}.pdf"
                                mime  = "application/pdf"
                            else:
                                data  = export_ppt(report_choice, exp_kpis, exp_charts, exp_ins)
                                fname = f"{report_choice.replace(' ','_')}.pptx"
                                mime  = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    except Exception as e:
                        st.error(f"Export failed: {e}")
                        data = None

                    if data:
                        pass

                    st.download_button(
                        label=f"⬇️ Download {export_fmt}",
                        data=data,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{report_choice}_{export_fmt}",
                    )