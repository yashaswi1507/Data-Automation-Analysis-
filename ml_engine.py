"""
ML Engine — Improved
Better model selection, XGBoost support, smarter feature prep,
faster training, better metrics.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier,
                              ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# DETECT TASK TYPE — purely from data
# ─────────────────────────────────────────────────────────────

def detect_task_type(series):
    """
    Returns 'classification' or 'regression'.
    Never uses column name — only data properties.
    """
    s = series.dropna()
    if len(s) == 0:
        return "regression"

    # Non-numeric → always classification
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"

    n_unique    = s.nunique()
    n_total     = len(s)

    # Binary → classification
    if n_unique == 2:
        return "classification"

    # Few unique values → classification
    if n_unique <= 10:
        return "classification"

    # Integer-like with bounded range (e.g. grade 1-5, rating 1-10)
    whole_frac  = (s == s.round()).mean()
    value_range = float(s.max() - s.min())
    if whole_frac >= 0.95 and value_range <= 20:
        return "classification"

    # High unique ratio → regression
    return "regression"


# ─────────────────────────────────────────────────────────────
# PREPARE FEATURES
# ─────────────────────────────────────────────────────────────

def prepare_features(df, target):
    """
    Returns X, y, feature_names, label_encoders, target_encoder, scaler, imputer.
    Robust feature engineering.
    """
    df = df.copy()
    df = df.dropna(subset=[target])

    y = df[target].copy()
    X = df.drop(columns=[target])

    n = len(X)

    # Remove >60% missing columns
    X = X.loc[:, X.isnull().mean() <= 0.60]

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    # Remove ID-like columns (>95% unique)
    id_cols = [c for c in X.columns if X[c].nunique() / max(n, 1) > 0.95]
    if id_cols:
        X = X.drop(columns=id_cols)

    # Remove pure text/name columns (object >50% unique)
    text_cols = [
        c for c in X.select_dtypes(include="object").columns
        if X[c].nunique() / max(n, 1) > 0.5
    ]
    if text_cols:
        X = X.drop(columns=text_cols)

    # Remove datetime columns
    date_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    if date_cols:
        X = X.drop(columns=date_cols)

    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le      = LabelEncoder()
        X[col]  = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target if string labels
    target_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)))

    feature_names = X.columns.tolist()

    # Impute + Scale
    imputer = SimpleImputer(strategy="median")
    X_arr   = imputer.fit_transform(X)
    scaler  = StandardScaler()
    X_arr   = scaler.fit_transform(X_arr)

    return X_arr, y, feature_names, label_encoders, target_encoder, scaler, imputer


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

def _get_models(task_type, n_samples):
    """Get appropriate models based on task and dataset size."""
    # For small datasets — simpler models
    n_estimators = 50 if n_samples < 500 else 100

    if task_type == "regression":
        models = {
            "Linear Regression":   LinearRegression(),
            "Ridge Regression":    Ridge(alpha=1.0),
            "Random Forest":       RandomForestRegressor(n_estimators=n_estimators, max_depth=8, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingRegressor(n_estimators=n_estimators, max_depth=4, random_state=42),
            "Extra Trees":         ExtraTreesRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1),
            "Decision Tree":       DecisionTreeRegressor(max_depth=6, random_state=42),
            "KNN Regressor":       KNeighborsRegressor(n_neighbors=min(5, n_samples//10)),
        }
        # Try XGBoost if available
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(
                n_estimators=n_estimators, max_depth=4, random_state=42,
                n_jobs=-1, verbosity=0
            )
        except ImportError:
            pass
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=n_estimators, max_depth=8, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=n_estimators, max_depth=4, random_state=42),
            "Extra Trees":         ExtraTreesClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1),
            "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
            "KNN Classifier":      KNeighborsClassifier(n_neighbors=min(5, n_samples//10)),
        }
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(
                n_estimators=n_estimators, max_depth=4, random_state=42,
                n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
        except ImportError:
            pass

    return models


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def train_prediction_model(df, target):
    """
    Main entry point. Returns result dict with everything UI needs.
    """
    if target not in df.columns:
        return {"error": f"Target column '{target}' not found."}

    if df[target].isnull().mean() > 0.5:
        return {"error": f"Target column '{target}' has >50% missing values."}

    if len(df) < 20:
        return {"error": "Need at least 20 rows to train a model."}

    task_type = detect_task_type(df[target])

    try:
        X, y, feature_names, label_encoders, target_encoder, scaler, imputer = prepare_features(df, target)
    except Exception as e:
        return {"error": f"Feature preparation failed: {str(e)}"}

    if X.shape[1] == 0:
        return {"error": "No usable feature columns found. Check your dataset."}

    if X.shape[0] < 10:
        return {"error": "Not enough rows after cleaning."}

    # Train/test split
    try:
        stratify = y if (task_type == "classification" and len(np.unique(y)) <= 20
                         and min(np.bincount(y.astype(int))) >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models   = _get_models(task_type, len(X_train))
    cv_folds = min(5, max(2, X_train.shape[0] // 10))
    scoring  = "r2" if task_type == "regression" else "accuracy"

    # Cross-validate all models
    results = []
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv_folds, scoring=scoring, n_jobs=-1)
            results.append({
                "name":    name,
                "model":   model,
                "cv_mean": float(cv_scores.mean()),
                "cv_std":  float(cv_scores.std()),
            })
        except Exception:
            continue

    if not results:
        return {"error": "All models failed. Check your data quality."}

    results.sort(key=lambda r: r["cv_mean"], reverse=True)
    best = results[0]

    # Final fit on full training set
    best["model"].fit(X_train, y_train)
    y_pred = best["model"].predict(X_test)

    # Metrics
    metrics = {}
    if task_type == "regression":
        metrics["R² Score"] = round(float(r2_score(y_test, y_pred)), 4)
        metrics["MAE"]      = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics["RMSE"]     = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        # Adjusted R²
        n, p = len(y_test), X_test.shape[1]
        r2   = metrics["R² Score"]
        if p < n - 1:
            metrics["Adj R²"] = round(float(1 - (1-r2)*(n-1)/(n-p-1)), 4)
    else:
        y_pred_cls      = np.round(y_pred).astype(int)
        metrics["Accuracy"] = round(float(accuracy_score(y_test, y_pred_cls)), 4)
        try:
            metrics["F1 Score"] = round(float(f1_score(
                y_test, y_pred_cls, average="weighted", zero_division=0
            )), 4)
        except Exception:
            pass

    # Feature importance
    feature_importance = _get_feature_importance(best["model"], feature_names)

    # Model comparison table
    model_comparison = [
        {
            "Model":           r["name"],
            "CV Score (mean)": round(r["cv_mean"], 4),
            "CV Std":          round(r["cv_std"], 4),
        }
        for r in results
    ]

    return {
        "error":              None,
        "task_type":          task_type,
        "best_model_name":    best["name"],
        "model":              best["model"],
        "metrics":            metrics,
        "feature_importance": feature_importance,
        "model_comparison":   model_comparison,
        "feature_names":      feature_names,
        "label_encoders":     label_encoders,
        "target_encoder":     target_encoder,
        "scaler":             scaler,
        "imputer":            imputer,
        "n_train":            len(X_train),
        "n_test":             len(X_test),
        "cv_folds":           cv_folds,
    }


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def _get_feature_importance(model, feature_names):
    importance = None

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        importance = np.abs(coef)

    if importance is None or len(importance) != len(feature_names):
        return []

    total = importance.sum()
    if total == 0:
        return []

    importance_pct = (importance / total) * 100

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importance_pct})
        .sort_values("Importance", ascending=False)
        .to_dict(orient="records")
    )


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

def predict_single(result_dict, input_values):
    """
    input_values: dict {feature_name: value}
    Returns predicted value (decoded if classification).
    """
    feature_names  = result_dict["feature_names"]
    label_encoders = result_dict["label_encoders"]
    scaler         = result_dict["scaler"]
    imputer        = result_dict["imputer"]
    model          = result_dict["model"]
    target_encoder = result_dict["target_encoder"]

    row = []
    for feat in feature_names:
        val = input_values.get(feat, np.nan)
        if feat in label_encoders:
            le = label_encoders[feat]
            try:
                val = le.transform([str(val)])[0]
            except Exception:
                # Unknown label — use most frequent class (0)
                val = 0
        try:
            val = float(val)
        except Exception:
            val = np.nan
        row.append(val)

    X_input = np.array(row, dtype=float).reshape(1, -1)
    X_input = imputer.transform(X_input)
    X_input = scaler.transform(X_input)

    pred = model.predict(X_input)[0]

    if target_encoder is not None:
        try:
            pred = target_encoder.inverse_transform([int(round(float(pred)))])[0]
        except Exception:
            pass

    return pred
