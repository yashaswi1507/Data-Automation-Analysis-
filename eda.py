import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def show_summary(df, st):
    """
    Full EDA summary — works on any dataset.
    No column name assumptions.
    """

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    total_cells      = df.shape[0] * df.shape[1]
    missing_total    = int(df.isnull().sum().sum())
    dup_count        = int(df.duplicated().sum())

    # ── Overview KPIs ─────────────────────────────────────────
    st.subheader("📋 Dataset Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows",            f"{df.shape[0]:,}")
    c2.metric("Columns",         df.shape[1])
    c3.metric("Numeric Cols",    len(numeric_cols))
    c4.metric("Categorical Cols",len(categorical_cols))
    c5.metric("Missing Values",  f"{missing_total:,}",
              delta=f"{round(missing_total/total_cells*100,1)}%" if total_cells else None,
              delta_color="inverse")
    c6.metric("Duplicates",      f"{dup_count:,}",
              delta_color="inverse")

    st.divider()

    # ── Column-level summary table ────────────────────────────
    st.subheader("🗂️ Column Summary")

    rows = []
    for col in df.columns:
        s         = df[col]
        missing   = int(s.isnull().sum())
        miss_pct  = round(missing / len(s) * 100, 1) if len(s) else 0
        n_unique  = int(s.nunique())
        dtype     = str(s.dtype)

        row = {
            "Column":       col,
            "Type":         dtype,
            "Non-Null":     int(s.notnull().sum()),
            "Missing":      missing,
            "Missing %":    f"{miss_pct}%",
            "Unique":       n_unique,
        }

        if pd.api.types.is_numeric_dtype(s):
            row["Min"]    = round(float(s.min()),  2) if missing < len(s) else "-"
            row["Max"]    = round(float(s.max()),  2) if missing < len(s) else "-"
            row["Mean"]   = round(float(s.mean()), 2) if missing < len(s) else "-"
            row["Median"] = round(float(s.median()),2) if missing < len(s) else "-"
            row["Std"]    = round(float(s.std()),  2) if missing < len(s) else "-"
            row["Sample Values"] = "-"
        else:
            row["Min"]    = "-"
            row["Max"]    = "-"
            row["Mean"]   = "-"
            row["Median"] = "-"
            row["Std"]    = "-"
            top_vals = s.dropna().value_counts().head(3).index.tolist()
            row["Sample Values"] = ", ".join([str(v) for v in top_vals])

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Statistical Describe ──────────────────────────────────
    if numeric_cols:
        st.subheader("📐 Statistical Summary")
        desc = df[numeric_cols].describe().round(2)
        st.dataframe(desc, use_container_width=True)

    st.divider()

    # ── Missing value heatmap ─────────────────────────────────
    if missing_total > 0:
        st.subheader("🔍 Missing Value Pattern")

        miss_series = df.isnull().sum()
        miss_series = miss_series[miss_series > 0].sort_values(ascending=False)

        if len(miss_series) > 0:
            miss_df = miss_series.reset_index()
            miss_df.columns = ["Column", "Missing Count"]
            miss_df["Missing %"] = (miss_df["Missing Count"] / len(df) * 100).round(1)

            fig = px.bar(
                miss_df,
                x="Column", y="Missing %",
                color="Missing %",
                color_continuous_scale="Reds",
                text_auto=".1f",
                template="plotly_white",
                title="Missing Values by Column (%)",
            )
            fig.update_layout(
                height=350, showlegend=False,
                coloraxis_showscale=False,
                xaxis_tickangle=-30,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values in this dataset!")

    st.divider()

    # ── Numeric distributions ─────────────────────────────────
    if numeric_cols:
        st.subheader("📊 Numeric Column Distributions")

        # Show 3 per row
        for i in range(0, min(len(numeric_cols), 9), 3):
            cols_batch = numeric_cols[i:i+3]
            ui_cols    = st.columns(len(cols_batch))

            for col_ui, col in zip(ui_cols, cols_batch):
                s = df[col].dropna()
                if len(s) == 0:
                    continue

                skew  = round(float(s.skew()), 2)
                q1    = s.quantile(0.25)
                q3    = s.quantile(0.75)
                iqr   = q3 - q1
                n_out = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())

                fig = px.histogram(
                    s, nbins=25,
                    template="plotly_white",
                    color_discrete_sequence=["#636EFA"],
                )
                fig.update_layout(
                    title=dict(text=col, font=dict(size=12)),
                    height=220,
                    margin=dict(t=35, b=20, l=20, r=10),
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="",
                )
                fig.update_traces(marker_line_width=0.5, marker_line_color="white")
                col_ui.plotly_chart(fig, use_container_width=True)

                # Stats below chart
                col_ui.caption(
                    f"skew={skew} | outliers={n_out} | "
                    f"mean={round(float(s.mean()),1)} | std={round(float(s.std()),1)}"
                )

    st.divider()

    # ── Categorical value counts ──────────────────────────────
    if categorical_cols:
        st.subheader("🏷️ Categorical Columns")

        for i in range(0, min(len(categorical_cols), 6), 2):
            cols_batch = categorical_cols[i:i+2]
            ui_cols    = st.columns(len(cols_batch))

            for col_ui, col in zip(ui_cols, cols_batch):
                vc = df[col].value_counts().head(10).reset_index()
                vc.columns = [col, "Count"]

                fig = px.bar(
                    vc, x=col, y="Count",
                    color=col,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    text_auto=True,
                )
                fig.update_layout(
                    title=dict(text=col, font=dict(size=12)),
                    height=250,
                    margin=dict(t=35, b=20, l=20, r=10),
                    showlegend=False,
                    xaxis_tickangle=-30,
                    xaxis_title="",
                )
                fig.update_traces(textposition="outside", textfont_size=9)
                col_ui.plotly_chart(fig, use_container_width=True)
                col_ui.caption(f"{df[col].nunique()} unique values")

    st.divider()

    # ── Correlation matrix ────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Correlation Matrix")

        corr = df[numeric_cols].corr().round(2)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            template="plotly_white",
        )
        fig.update_layout(
            height=max(350, len(numeric_cols) * 45),
            coloraxis_colorbar=dict(title="r"),
            margin=dict(t=30, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations text
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                pairs.append((
                    numeric_cols[i],
                    numeric_cols[j],
                    corr.iloc[i, j]
                ))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        if pairs:
            st.markdown("**Top correlations:**")
            for a, b, r in pairs[:3]:
                direction = "📈 positive" if r > 0 else "📉 negative"
                strength  = "strong" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.4 else "weak")
                st.caption(f"**{a}** ↔ **{b}**: r={r} — {strength} {direction} correlation")