import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────

def generate_kpis(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    metrics = {
        "Rows":            len(df),
        "Columns":         df.shape[1],
        "Missing Values":  int(df.isnull().sum().sum()),
        "Numeric Columns": len(numeric_cols),
    }

    # Add domain-specific KPIs if columns exist
    for col in numeric_cols[:4]:
        metrics[f"Avg {col}"] = round(df[col].mean(), 2)

    return metrics


# ─────────────────────────────────────────────────────────────
# SMART AUTO CHART GENERATOR
# Generates meaningful charts based on actual data composition
# Returns list of dicts: {id, title, fig, chart_type, pinned}
# ─────────────────────────────────────────────────────────────

def generate_auto_charts(df, max_cats=15):
    charts     = []
    chart_id   = 0

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        c for c in df.select_dtypes(exclude="number").columns.tolist()
        if df[c].nunique() <= 30   # skip high-cardinality text cols
    ]

    TEMPLATE = "plotly_white"
    COLORS   = px.colors.qualitative.Set2

    def _add(fig, title, chart_type):
        nonlocal chart_id
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            template=TEMPLATE,
            height=380,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        charts.append({
            "id":         chart_id,
            "title":      title,
            "fig":        fig,
            "chart_type": chart_type,
            "pinned":     True,    # all auto-charts pinned by default
            "source":     "auto",  # 'auto' or 'studio'
        })
        chart_id += 1

    # ── 1. Distribution histograms for top numeric cols ──────
    for col in numeric_cols[:3]:
        fig = px.histogram(
            df, x=col,
            nbins=30,
            marginal="box",
            color_discrete_sequence=[COLORS[0]],
        )
        fig.update_layout(showlegend=False, bargap=0.05)
        _add(fig, f"Distribution of {col}", "histogram")

    # ── 2. Bar charts: each categorical vs best numeric ──────
    if numeric_cols:
        best_num = numeric_cols[0]
        for cat_col in categorical_cols[:3]:
            n_unique = df[cat_col].nunique()
            top_n    = min(n_unique, max_cats)
            grouped  = (
                df.groupby(cat_col)[best_num]
                .mean()
                .reset_index()
                .sort_values(best_num, ascending=False)
                .head(top_n)
            )
            fig = px.bar(
                grouped,
                x=cat_col, y=best_num,
                color=cat_col,
                text_auto=".2s",
                color_discrete_sequence=COLORS,
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            fig.update_traces(textposition="outside")
            _add(fig, f"Avg {best_num} by {cat_col}", "bar")

    # ── 3. Scatter plots: numeric pairs ──────────────────────
    if len(numeric_cols) >= 2:
        for i in range(min(2, len(numeric_cols) - 1)):
            x_col = numeric_cols[i]
            y_col = numeric_cols[i + 1]
            color_col = categorical_cols[0] if categorical_cols else None

            sample_df = df.sample(min(500, len(df)), random_state=42)
            fig = px.scatter(
                sample_df,
                x=x_col, y=y_col,
                color=color_col,
                opacity=0.65,

                color_discrete_sequence=COLORS,
                trendline="ols" if len(sample_df) >= 10 else None,
            )
            fig.update_traces(marker=dict(size=5))
            title = f"{x_col} vs {y_col}"
            if color_col:
                title += f" (by {color_col})"
            _add(fig, title, "scatter")

    # ── 4. Pie / donut for categorical cols ──────────────────
    for cat_col in categorical_cols[:2]:
        pie_data = (
            df[cat_col].value_counts()
            .head(max_cats)
            .reset_index()
        )
        pie_data.columns = [cat_col, "Count"]

        # Group remainder as "Others"
        total = df[cat_col].value_counts().sum()
        shown = pie_data["Count"].sum()
        if shown < total:
            others = pd.DataFrame([{cat_col: "Others", "Count": total - shown}])
            pie_data = pd.concat([pie_data, others], ignore_index=True)

        fig = px.pie(
            pie_data,
            names=cat_col, values="Count",
            hole=0.35,
            color_discrete_sequence=COLORS,
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            pull=[0.02] * len(pie_data),
        )
        fig.update_layout(showlegend=False)
        _add(fig, f"{cat_col} Distribution", "pie")

    # ── 5. Correlation heatmap ────────────────────────────────
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        fig  = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
        )
        fig.update_layout(
            height=max(380, len(numeric_cols) * 40),
            coloraxis_colorbar=dict(title="r"),
        )
        _add(fig, "Correlation Heatmap", "heatmap")

    # ── 6. Box plots for numeric cols by best categorical ─────
    if numeric_cols and categorical_cols:
        cat_col = categorical_cols[0]
        top_cats = df[cat_col].value_counts().head(max_cats).index.tolist()
        box_df   = df[df[cat_col].isin(top_cats)]

        for num_col in numeric_cols[:2]:
            fig = px.box(
                box_df, x=cat_col, y=num_col,
                color=cat_col,
                color_discrete_sequence=COLORS,
                points="outliers",
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            _add(fig, f"{num_col} by {cat_col} (Box)", "box")

    return charts


# ─────────────────────────────────────────────────────────────
# AUTO INSIGHTS  — data-driven, not hardcoded strings
# ─────────────────────────────────────────────────────────────

def generate_insights(df):
    insights = []
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Missing value insight
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        pct = round(total_missing / df.size * 100, 1)
        insights.append(f"⚠️ Dataset has {total_missing:,} missing values ({pct}% of all cells)")
    else:
        insights.append("✅ No missing values found in the dataset")

    for col in numeric_cols[:4]:
        s        = df[col].dropna()
        skewness = s.skew()
        mean_val = round(s.mean(), 2)
        max_val  = s.max()
        min_val  = s.min()

        insights.append(f"📊 **{col}**: mean={mean_val:,}, range=[{min_val:,} – {max_val:,}]")

        if abs(skewness) > 1.5:
            direction = "right (positive)" if skewness > 0 else "left (negative)"
            insights.append(f"↗️ **{col}** is heavily skewed {direction} (skew={skewness:.2f}) — outliers likely present")

        # Outlier count via IQR
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR     = Q3 - Q1
        outliers = ((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum()
        if outliers > 0:
            insights.append(f"🔴 **{col}** has {outliers} outlier(s) ({round(outliers/len(s)*100,1)}% of rows)")

    # Correlation insight
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                pairs.append((
                    numeric_cols[i],
                    numeric_cols[j],
                    corr.iloc[i, j]
                ))
        if pairs:
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top = pairs[0]
            direction = "positively" if top[2] > 0 else "negatively"
            insights.append(
                f"🔗 Strongest correlation: **{top[0]}** & **{top[1]}** are {direction} correlated (r={top[2]:.2f})"
            )

    # Categorical dominance
    for col in categorical_cols[:2]:
        vc  = df[col].value_counts()
        if len(vc) > 0:
            top_val = vc.index[0]
            top_pct = round(vc.iloc[0] / len(df) * 100, 1)
            if top_pct > 50:
                insights.append(f"📌 **{col}**: '{top_val}' dominates with {top_pct}% of rows")

    return insights
