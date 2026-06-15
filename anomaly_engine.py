"""
Anomaly Detection Engine
Detects unusual spikes/drops in numeric columns.
No API, pure Python/statistics.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def detect_anomalies(df, method="iqr", threshold=2.5):
    """
    Detect anomalies in all numeric columns.
    Returns dict: {col: [row_indices of anomalies]}
    Methods: 'iqr', 'zscore', 'both'
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    results = {}

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue

        anomaly_idx = set()

        # IQR method
        if method in ("iqr", "both"):
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR    = Q3 - Q1
            if IQR > 0:
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                idx   = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
                anomaly_idx.update(idx)

        # Z-score method
        if method in ("zscore", "both"):
            mean, std = s.mean(), s.std()
            if std > 0:
                z_scores = ((df[col] - mean) / std).abs()
                idx      = df[col][z_scores > threshold].index.tolist()
                anomaly_idx.update(idx)

        if anomaly_idx:
            results[col] = sorted(list(anomaly_idx))

    return results


def explain_anomaly(df, col, idx):
    """
    Returns plain English explanation for an anomaly.
    """
    val  = df.loc[idx, col]
    s    = df[col].dropna()
    mean = s.mean()
    std  = s.std()
    pct  = (s < val).mean() * 100

    direction = "above" if val > mean else "below"
    z         = abs(val - mean) / std if std > 0 else 0

    return (
        f"Value **{val:,.2f}** is {direction} average ({mean:,.2f}) "
        f"by {z:.1f}x standard deviation — "
        f"higher than {pct:.0f}% of all values."
    )
