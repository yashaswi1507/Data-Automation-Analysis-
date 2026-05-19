import pandas as pd
import plotly.express as px

# =====================================================
# GENERATE KPI METRICS
# =====================================================

def generate_kpis(df):

    numeric_cols = df.select_dtypes(
        include='number'
    ).columns.tolist()

    metrics = {

        "Rows": len(df),

        "Columns": df.shape[1],

        "Missing Values":
        int(
            df.isnull().sum().sum()
        ),

        "Numeric Columns":
        len(numeric_cols)
    }

    return metrics

# =====================================================
# AUTO CHART GENERATOR
# =====================================================

def generate_auto_charts(df):

    charts = []

    numeric_cols = df.select_dtypes(
        include='number'
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        exclude='number'
    ).columns.tolist()

    # ================================================
    # HISTOGRAM
    # ================================================

    if len(numeric_cols) > 0:

        fig = px.histogram(
            df,
            x=numeric_cols[0],
            title=f"Distribution of {numeric_cols[0]}"
        )

        charts.append(fig)

    # ================================================
    # BAR CHART
    # ================================================

    if (
        len(categorical_cols) > 0
        and
        len(numeric_cols) > 0
    ):

        grouped = (
            df.groupby(
                categorical_cols[0]
            )[numeric_cols[0]]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            grouped,
            x=categorical_cols[0],
            y=numeric_cols[0],
            color=categorical_cols[0],
            title=f"{numeric_cols[0]} by {categorical_cols[0]}"
        )

        charts.append(fig)

    # ================================================
    # SCATTER
    # ================================================

    if len(numeric_cols) >= 2:

        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
        )

        charts.append(fig)

    # ================================================
    # PIE CHART
    # ================================================

    if len(categorical_cols) > 0:

        pie_data = (
            df[categorical_cols[0]]
            .value_counts()
            .reset_index()
        )

        pie_data.columns = [
            categorical_cols[0],
            "Count"
        ]

        fig = px.pie(
            pie_data,
            names=categorical_cols[0],
            values="Count",
            title=f"{categorical_cols[0]} Distribution"
        )

        charts.append(fig)

    # ================================================
    # HEATMAP
    # ================================================

    if len(numeric_cols) > 1:

        corr = (
            df[numeric_cols]
            .corr()
        )

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap"
        )

        charts.append(fig)

    return charts

# =====================================================
# AUTO INSIGHTS
# =====================================================

def generate_insights(df):

    insights = []

    numeric_cols = df.select_dtypes(
        include='number'
    ).columns.tolist()

    if len(numeric_cols) > 0:

        highest_col = numeric_cols[0]

        max_value = df[
            highest_col
        ].max()

        insights.append(
            f"Highest value in {highest_col} is {max_value}"
        )

        avg_value = round(
            df[highest_col].mean(),
            2
        )

        insights.append(
            f"Average {highest_col} is {avg_value}"
        )

    return insights