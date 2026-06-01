import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# DETECT TASK TYPE
# Regression vs Classification — purely from target column data
# ─────────────────────────────────────────────────────────────

def detect_task_type(series):
    """
    Returns 'classification' or 'regression' based on target column.
    Never uses column name — only data properties.
    """
    s = series.dropna()

    # Non-numeric → always classification
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"

    n_unique    = s.nunique()
    n_total     = len(s)
    unique_ratio = n_unique / n_total

    # Few unique values → classification (binary or multiclass)
    if n_unique <= 10:
        return "classification"

    # Integer-like with bounded range → likely classification (e.g. grade 1-5)
    whole_frac = (s == s.round()).mean()
    value_range = float(s.max() - s.min())
    if whole_frac >= 0.95 and value_range <= 20:
        return "classification"

    # Continuous → regression
    return "regression"


# ─────────────────────────────────────────────────────────────
# PREPARE FEATURES
# Handle categoricals, missing values, scaling
# ─────────────────────────────────────────────────────────────

def prepare_features(df, target):
    """
    Returns X (features), y (target), feature_names, label_encoders, scaler.
    Works on any dataset — encodes categoricals, imputes missing.
    """
    df = df.copy()

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    y = df[target].copy()
    X = df.drop(columns=[target])

    # Remove columns with >60% missing
    missing_ratio = X.isnull().mean()
    X = X.loc[:, missing_ratio <= 0.60]

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    # Remove ID-like columns (high cardinality unique — useless as features)
    n = len(X)
    id_cols = [
        c for c in X.columns
        if X[c].nunique() / max(n, 1) > 0.95  # nearly all unique
    ]
    if id_cols:
        X = X.drop(columns=id_cols)

    # Remove pure text/name columns (object with high cardinality)
    text_cols = [
        c for c in X.select_dtypes(include="object").columns
        if X[c].nunique() / max(n, 1) > 0.5
    ]
    if text_cols:
        X = X.drop(columns=text_cols)

    # Encode categorical columns
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target if classification with string labels
    target_encoder = None
    if not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
        y = pd.Series(y)

    feature_names = X.columns.tolist()

    # Impute remaining missing values
    imputer = SimpleImputer(strategy="median")
    X_arr   = imputer.fit_transform(X)

    # Scale features
    scaler  = StandardScaler()
    X_arr   = scaler.fit_transform(X_arr)

    return X_arr, y, feature_names, label_encoders, target_encoder, scaler, imputer


# ─────────────────────────────────────────────────────────────
# TRAIN — AUTO MODEL SELECTION
# Tries multiple models, picks best by cross-val score
# ─────────────────────────────────────────────────────────────

REGRESSION_MODELS = {
    "Linear Regression":        LinearRegression(),
    "Ridge Regression":         Ridge(alpha=1.0),
    "Lasso Regression":         Lasso(alpha=0.1),
    "Decision Tree":            DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest":            RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    "KNN Regressor":            KNeighborsRegressor(n_neighbors=5),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":            DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    "KNN Classifier":           KNeighborsClassifier(n_neighbors=5),
}


def train_prediction_model(df, target):
    """
    Main entry point.
    Returns a result dict with everything the UI needs.
    """
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
        return {"error": "No usable feature columns found after preprocessing."}

    if X.shape[0] < 20:
        return {"error": "Not enough rows after removing missing values."}

    # Train/test split
    test_size = 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if task_type == "classification" and len(np.unique(y)) <= 20 else None
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    models = REGRESSION_MODELS if task_type == "regression" else CLASSIFICATION_MODELS

    # ── Try all models, rank by cross-val score ──────────────────
    results  = []
    cv_folds = min(5, max(2, X_train.shape[0] // 10))

    for name, model in models.items():
        try:
            scoring = "r2" if task_type == "regression" else "accuracy"
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv_folds, scoring=scoring, n_jobs=-1)
            results.append({
                "name":     name,
                "model":    model,
                "cv_mean":  float(cv_scores.mean()),
                "cv_std":   float(cv_scores.std()),
            })
        except Exception:
            continue

    if not results:
        return {"error": "All models failed to train. Check your data."}

    # Best model by cross-val mean
    results.sort(key=lambda r: r["cv_mean"], reverse=True)
    best = results[0]

    # Final fit on full train set
    best["model"].fit(X_train, y_train)
    y_pred = best["model"].predict(X_test)

    # ── Metrics ──────────────────────────────────────────────────
    metrics = {}
    if task_type == "regression":
        metrics["R² Score"]  = round(r2_score(y_test, y_pred), 4)
        metrics["MAE"]       = round(mean_absolute_error(y_test, y_pred), 4)
        metrics["RMSE"]      = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    else:
        y_pred_cls = y_pred.round().astype(int)
        metrics["Accuracy"]  = round(accuracy_score(y_test, y_pred_cls), 4)
        try:
            from sklearn.metrics import f1_score
            metrics["F1 Score"] = round(f1_score(y_test, y_pred_cls, average="weighted", zero_division=0), 4)
        except Exception:
            pass

    # ── Feature importance ────────────────────────────────────────
    feature_importance = _get_feature_importance(best["model"], feature_names)

    # ── All model comparison ──────────────────────────────────────
    model_comparison = [
        {
            "Model":          r["name"],
            "CV Score (mean)":round(r["cv_mean"], 4),
            "CV Std":         round(r["cv_std"], 4),
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
# Works for tree models (feature_importances_) and
# linear models (coef_) — falls back to permutation if needed
# ─────────────────────────────────────────────────────────────

def _get_feature_importance(model, feature_names):
    importance = None

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    # Linear models
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        importance = np.abs(coef)

    if importance is None or len(importance) != len(feature_names):
        return []

    # Normalize to 0-100
    total = importance.sum()
    if total == 0:
        return []

    importance_pct = (importance / total) * 100

    df_imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importance_pct,
    }).sort_values("Importance", ascending=False)

    return df_imp.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────
# PREDICT — for new input
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

    # Build input array
    row = []
    for feat in feature_names:
        val = input_values.get(feat, np.nan)
        if feat in label_encoders:
            le = label_encoders[feat]
            try:
                val = le.transform([str(val)])[0]
            except Exception:
                val = 0
        row.append(val)

    X_input = np.array(row).reshape(1, -1)
    X_input = imputer.transform(X_input)
    X_input = scaler.transform(X_input)

    pred = model.predict(X_input)[0]

    # Decode classification label
    if target_encoder is not None:
        try:
            pred = target_encoder.inverse_transform([int(round(pred))])[0]
        except Exception:
            pass

    return pred