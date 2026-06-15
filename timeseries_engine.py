"""
Time Series Forecasting Engine — Improved
Handles ALL date formats, better models, no deprecation warnings.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# All date formats — Indian, US, EU, ISO, custom
DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%m/%d/%Y", "%m-%d-%Y",
    "%d/%m/%y", "%m/%d/%y", "%d-%m-%y",
    "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%B %d, %Y",
    "%d-%B-%Y", "%d %B %Y", "%b-%d-%Y",
    "%d-%b-%y", "%b %d %Y",
    "%b-%Y", "%B-%Y", "%b %Y", "%B %Y",
    "%Y-%m", "%m-%Y", "%m/%Y",
]


def _parse_dates_robust(series):
    """Try every possible date format. Returns parsed Series or None."""
    s = series.dropna()
    if len(s) == 0:
        return None

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # Numeric epoch / Excel serial
    if pd.api.types.is_numeric_dtype(series):
        try:
            parsed = pd.to_datetime(series, unit='s', errors='coerce')
            if parsed.notna().mean() >= 0.7:
                return parsed
        except Exception:
            pass
        try:
            parsed = pd.to_datetime('1899-12-30') + pd.to_timedelta(series, unit='D')
            if parsed.notna().mean() >= 0.7:
                return parsed
        except Exception:
            pass
        return None

    str_s = s.astype(str).str.strip()

    # Pandas auto-inference
    try:
        parsed = pd.to_datetime(str_s, infer_datetime_format=True, errors='coerce')
        if parsed.notna().mean() >= 0.7:
            return pd.to_datetime(series.astype(str), infer_datetime_format=True, errors='coerce')
    except Exception:
        pass

    # Try each format
    best_parsed = None
    best_ratio  = 0.0
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(str_s, format=fmt, errors='coerce')
            ratio  = parsed.notna().mean()
            if ratio > best_ratio:
                best_ratio  = ratio
                best_parsed = pd.to_datetime(
                    series.astype(str).str.strip(), format=fmt, errors='coerce'
                )
        except Exception:
            continue

    if best_ratio >= 0.6:
        return best_parsed

    # dateutil fallback — handles almost anything
    try:
        from dateutil import parser as du
        def _safe(v):
            try:
                return du.parse(str(v), dayfirst=True)
            except Exception:
                return pd.NaT
        parsed = pd.to_datetime(series.apply(_safe), errors='coerce')
        if parsed.notna().mean() >= 0.6:
            return parsed
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────
# DETECT DATE + VALUE COLUMNS
# ─────────────────────────────────────────────────────────────

def detect_timeseries_cols(df):
    """Returns (date_col, value_cols) — purely from data, no name assumptions."""
    date_col   = None
    value_cols = []

    # 1. Already datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break

    # 2. Try each column
    if not date_col:
        for col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            # Year column
            if pd.api.types.is_numeric_dtype(s):
                if s.between(1900, 2100).mean() >= 0.9 and s.nunique() <= 200:
                    date_col = col
                    break
                continue
            parsed = _parse_dates_robust(df[col])
            if parsed is not None and parsed.notna().mean() >= 0.6:
                date_col = col
                break

    # 3. Value columns
    for col in df.select_dtypes(include="number").columns:
        if col == date_col:
            continue
        s = df[col].dropna()
        if len(s) < 3:
            continue
        if s.nunique() / max(len(s), 1) > 0.02 and s.std() > 0:
            value_cols.append(col)

    return date_col, value_cols


# ─────────────────────────────────────────────────────────────
# PREPARE TIME SERIES
# ─────────────────────────────────────────────────────────────

def prepare_series(df, date_col, value_col, freq="auto"):
    """Returns clean sorted resampled Series ready for forecasting."""
    ts_df = df[[date_col, value_col]].copy()

    if not pd.api.types.is_datetime64_any_dtype(ts_df[date_col]):
        parsed = _parse_dates_robust(ts_df[date_col])
        ts_df[date_col] = parsed if parsed is not None else pd.to_datetime(
            ts_df[date_col], errors='coerce'
        )

    ts_df = ts_df.dropna()
    ts_df = ts_df.sort_values(date_col)
    ts    = ts_df.set_index(date_col)[value_col]

    if freq == "auto":
        freq = _detect_freq(ts)

    try:
        ts = ts.resample(freq).mean().interpolate(method="time")
    except Exception:
        try:
            ts = ts.resample("D").mean().interpolate(method="linear")
        except Exception:
            pass

    return ts, freq


def _detect_freq(ts):
    """Detect frequency — modern pandas aliases only (no ME/QE deprecation)."""
    if len(ts) < 2:
        return "D"
    try:
        diffs       = pd.Series(ts.index).diff().dropna()
        median_diff = diffs.median()
        days        = getattr(median_diff, 'days', 1)
    except Exception:
        return "D"

    if days <= 1:    return "D"
    elif days <= 7:  return "W"
    elif days <= 15: return "2W"
    elif days <= 31: return "MS"
    elif days <= 92: return "QS"
    else:            return "YS"


# ─────────────────────────────────────────────────────────────
# FORECASTING
# ─────────────────────────────────────────────────────────────

def forecast(ts, periods=30, freq="D"):
    """Tries 5 models, returns best by lowest error."""
    if len(ts) < 3:
        return None, "Need at least 3 data points for forecasting."

    results = []

    # 1. Exponential Smoothing
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        use_trend = "add" if len(ts) >= 6 else None
        model = ExponentialSmoothing(
            ts, trend=use_trend, seasonal=None,
            damped_trend=bool(use_trend),
        ).fit(optimized=True, disp=False)
        pred = model.forecast(periods)
        pred.index = _future_index(ts, periods, freq)
        results.append({"name": "Exponential Smoothing", "forecast": pred,
                         "score": float(model.sse / max(len(ts), 1))})
    except Exception:
        pass

    # 2. ARIMA
    try:
        from statsmodels.tsa.arima.model import ARIMA
        p     = 1 if len(ts) >= 10 else 0
        model = ARIMA(ts, order=(p, 1, 1)).fit()
        pred  = model.forecast(steps=periods)
        pred.index = _future_index(ts, periods, freq)
        results.append({"name": "ARIMA", "forecast": pred,
                         "score": float(np.mean(model.resid ** 2))})
    except Exception:
        pass

    # 3. Linear Trend
    try:
        x         = np.arange(len(ts))
        coefs     = np.polyfit(x, ts.values, deg=1)
        pred_vals = np.polyval(coefs, np.arange(len(ts), len(ts) + periods))
        pred      = pd.Series(pred_vals, index=_future_index(ts, periods, freq))
        mse       = float(np.mean((ts.values - np.polyval(coefs, x)) ** 2))
        results.append({"name": "Linear Trend", "forecast": pred, "score": mse})
    except Exception:
        pass

    # 4. Moving Average
    try:
        window   = min(7, max(2, len(ts) // 3))
        last_avg = float(ts.rolling(window).mean().dropna().iloc[-1])
        pred     = pd.Series(np.full(periods, last_avg),
                             index=_future_index(ts, periods, freq))
        roll_std = ts.rolling(window).mean().dropna().std()
        results.append({"name": "Moving Average", "forecast": pred,
                         "score": float(roll_std ** 2) if roll_std > 0 else 9999})
    except Exception:
        pass

    # 5. Polynomial Trend
    try:
        if len(ts) >= 12:
            x         = np.arange(len(ts))
            coefs     = np.polyfit(x, ts.values, deg=2)
            pred_vals = np.polyval(coefs, np.arange(len(ts), len(ts) + periods))
            pred      = pd.Series(pred_vals, index=_future_index(ts, periods, freq))
            mse       = float(np.mean((ts.values - np.polyval(coefs, x)) ** 2))
            results.append({"name": "Polynomial Trend", "forecast": pred, "score": mse})
    except Exception:
        pass

    if not results:
        return None, "No forecasting model could be fit on this data."

    return min(results, key=lambda r: r["score"]), None


def _future_index(ts, periods, freq):
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        return pd.date_range(start=ts.index[-1] + offset, periods=periods, freq=freq)
    except Exception:
        return pd.date_range(start=ts.index[-1], periods=periods + 1, freq="D")[1:]


def confidence_interval(ts, forecast_series, confidence=0.95):
    std    = float(ts.std())
    z      = 1.96 if confidence == 0.95 else 1.645
    margin = std * z
    return forecast_series + margin, forecast_series - margin
