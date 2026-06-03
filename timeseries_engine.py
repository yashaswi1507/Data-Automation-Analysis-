"""
Time Series Forecasting Engine
Auto-detects date column, forecasts using multiple methods,
picks the best one. No API, pure Python.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# DETECT DATE + VALUE COLUMNS
# ─────────────────────────────────────────────────────────────

def detect_timeseries_cols(df):
    """
    Returns (date_col, value_cols) — purely from data.
    date_col: best date column
    value_cols: numeric columns suitable for forecasting
    """
    date_col   = None
    value_cols = []

    # Find date column
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.8:
                date_col = col
                break
        except Exception:
            pass

    # Find numeric value columns (exclude IDs)
    for col in df.select_dtypes(include="number").columns:
        s             = df[col].dropna()
        unique_ratio  = s.nunique() / max(len(s), 1)
        if unique_ratio > 0.05:   # not a constant or near-constant
            value_cols.append(col)

    return date_col, value_cols


# ─────────────────────────────────────────────────────────────
# PREPARE TIME SERIES
# ─────────────────────────────────────────────────────────────

def prepare_series(df, date_col, value_col, freq="auto"):
    """
    Returns a clean, sorted, resampled Series ready for forecasting.
    """
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna()
    ts = ts.sort_values(date_col)
    ts = ts.set_index(date_col)[value_col]

    # Auto-detect frequency
    if freq == "auto":
        freq = _detect_freq(ts)

    # Resample
    try:
        ts = ts.resample(freq).mean().interpolate(method="time")
    except Exception:
        ts = ts.resample("D").mean().interpolate(method="linear")

    return ts, freq


def _detect_freq(ts):
    """Detect most likely time frequency from index."""
    if len(ts) < 2:
        return "D"
    diffs = pd.Series(ts.index).diff().dropna()
    median_diff = diffs.median()
    days = median_diff.days if hasattr(median_diff, 'days') else 1

    if days <= 1:    return "D"
    elif days <= 7:  return "W"
    elif days <= 31: return "ME"
    else:            return "QE"


# ─────────────────────────────────────────────────────────────
# FORECASTING MODELS
# ─────────────────────────────────────────────────────────────

def forecast(ts, periods=30, freq="D"):
    """
    Tries multiple models, returns best forecast.
    Returns dict with forecast Series and model info.
    """
    results = []

    # ── 1. Exponential Smoothing ─────────────────────────────
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(
            ts,
            trend="add" if len(ts) >= 4 else None,
            seasonal=None,
            damped_trend=True,
        ).fit(optimized=True, disp=False)
        pred = model.forecast(periods)
        mse  = float(model.sse / len(ts))
        results.append({"name": "Exponential Smoothing", "forecast": pred, "score": mse})
    except Exception:
        pass

    # ── 2. ARIMA ─────────────────────────────────────────────
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(ts, order=(1,1,1)).fit()
        pred  = model.forecast(steps=periods)
        pred.index = _future_index(ts, periods, freq)
        mse   = float(model.mae)
        results.append({"name": "ARIMA", "forecast": pred, "score": mse})
    except Exception:
        pass

    # ── 3. Linear Trend (always works as fallback) ───────────
    try:
        x     = np.arange(len(ts))
        coefs = np.polyfit(x, ts.values, deg=1)
        future_x = np.arange(len(ts), len(ts) + periods)
        pred_vals = np.polyval(coefs, future_x)
        pred_idx  = _future_index(ts, periods, freq)
        pred      = pd.Series(pred_vals, index=pred_idx)
        residuals = ts.values - np.polyval(coefs, x)
        mse       = float(np.mean(residuals**2))
        results.append({"name": "Linear Trend", "forecast": pred, "score": mse})
    except Exception:
        pass

    # ── 4. Moving Average ────────────────────────────────────
    try:
        window    = min(7, len(ts))
        last_avg  = ts.rolling(window).mean().iloc[-1]
        pred_vals = np.full(periods, last_avg)
        pred_idx  = _future_index(ts, periods, freq)
        pred      = pd.Series(pred_vals, index=pred_idx)
        mse       = float(ts.rolling(window).mean().dropna().std() ** 2)
        results.append({"name": "Moving Average", "forecast": pred, "score": mse})
    except Exception:
        pass

    if not results:
        return None, "No model could be fit."

    # Pick best (lowest score)
    best = min(results, key=lambda r: r["score"])
    return best, None


def _future_index(ts, periods, freq):
    """Generate future datetime index."""
    try:
        return pd.date_range(
            start=ts.index[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=periods,
            freq=freq,
        )
    except Exception:
        return pd.date_range(start=ts.index[-1], periods=periods+1, freq="D")[1:]


# ─────────────────────────────────────────────────────────────
# CONFIDENCE INTERVAL
# ─────────────────────────────────────────────────────────────

def confidence_interval(ts, forecast_series, confidence=0.95):
    """Simple CI based on historical std dev."""
    std    = float(ts.std())
    z      = 1.96 if confidence == 0.95 else 1.645
    margin = std * z
    upper  = forecast_series + margin
    lower  = forecast_series - margin
    return upper, lower