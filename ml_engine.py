"""
ml_engine.py — ML Pipeline for Predictive Denial Management
=============================================================
Trains an XGBoost classifier to predict claim denials.
Provides SHAP-based feature importance and per-claim explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve,
)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
import shap

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "denial_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.joblib")

# Features to use
CAT_FEATURES = ["insurance", "visit_type", "icd_code", "gender"]
NUM_FEATURES = ["claim_amount", "age", "fraud_score", "num_cpt_codes", "total_cpt_amount"]
BOOL_FEATURES = ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]


def _build_feature_df(master_df, cpt_summary_df):
    """Build the modeling DataFrame with all engineered features."""
    df = master_df.copy()

    # Merge CPT summary
    if cpt_summary_df is not None:
        df = df.merge(cpt_summary_df[["claim_id", "num_cpt_codes", "total_cpt_amount"]],
                      on="claim_id", how="left")
        df["num_cpt_codes"] = df["num_cpt_codes"].fillna(1)
        df["total_cpt_amount"] = df["total_cpt_amount"].fillna(df["claim_amount"])
    else:
        df["num_cpt_codes"] = 1
        df["total_cpt_amount"] = df["claim_amount"]

    # Boolean → int
    for col in BOOL_FEATURES:
        df[col] = df[col].astype(int)

    # One-hot encode categoricals
    df_ohe = pd.get_dummies(df[CAT_FEATURES], drop_first=False)
    ohe_cols = list(df_ohe.columns)

    # Combine all features
    df = pd.concat([df.drop(columns=CAT_FEATURES), df_ohe], axis=1)
    feature_cols = NUM_FEATURES + BOOL_FEATURES + ohe_cols

    return df, feature_cols


def train_model(master_df, cpt_summary_df):
    """
    Train an XGBoost denial prediction model.

    Returns dict with: model, metrics, feature_importance, roc_curve_data,
    confusion_matrix, feature_cols, shap_values, X_test
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    df, feature_cols = _build_feature_df(master_df, cpt_summary_df)

    X = df[feature_cols].fillna(0)
    y = df["is_denied"].astype(int)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)

    # Feature importance (mean |SHAP|)
    feat_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False)

    # Score all claims for portfolio risk
    all_probs = model.predict_proba(X.fillna(0))[:, 1]

    # Save model + metadata
    joblib.dump(model, MODEL_PATH)
    meta = {
        "feature_cols": feature_cols,
        "metrics": metrics,
        "roc_data": roc_data,
        "confusion_matrix": cm,
        "feature_importance": feat_importance,
        "all_probabilities": all_probs,
    }
    joblib.dump(meta, META_PATH)

    return {
        "model": model,
        "metrics": metrics,
        "roc_data": roc_data,
        "confusion_matrix": cm,
        "feature_importance": feat_importance,
        "feature_cols": feature_cols,
        "shap_values": shap_values,
        "shap_sample": shap_sample,
        "explainer": explainer,
        "all_probabilities": all_probs,
    }


def load_model():
    """Load a previously trained model. Returns (model, meta) or (None, None)."""
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        model = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        return model, meta
    return None, None


def predict_single_claim(claim_dict, model, feature_cols):
    """
    Predict denial probability for a single claim.

    Args:
        claim_dict: dict with raw features (insurance, visit_type, icd_code,
                    gender, claim_amount, age, fraud_score, num_cpt_codes,
                    total_cpt_amount, cpt_icd_mismatch, high_amount_flag,
                    strict_insurance_flag)
        model: trained XGBClassifier
        feature_cols: list of feature column names from training

    Returns:
        dict with denial_probability, risk_level, shap_explanation
    """
    # Build a single-row DataFrame with all feature columns set to 0
    row = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Set numeric features
    for col in NUM_FEATURES:
        if col in claim_dict:
            row[col] = claim_dict[col]

    # Set boolean features
    for col in BOOL_FEATURES:
        if col in claim_dict:
            row[col] = int(claim_dict[col])

    # Set one-hot encoded features
    for cat in CAT_FEATURES:
        if cat in claim_dict:
            ohe_col = f"{cat}_{claim_dict[cat]}"
            if ohe_col in row.columns:
                row[ohe_col] = 1

    # Predict
    prob = model.predict_proba(row)[:, 1][0]

    # Risk level
    if prob >= 0.7:
        risk = "HIGH"
    elif prob >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row)[0]
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": shap_vals,
    })
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    top_drivers = shap_df.sort_values("abs_shap", ascending=False).head(8)

    return {
        "denial_probability": float(prob),
        "risk_level": risk,
        "shap_explanation": top_drivers,
    }


# ============================================================
# Phase 3 — CPT-ICD Mismatch Probability Model
# ============================================================

MISMATCH_MODEL_PATH = os.path.join(MODEL_DIR, "cpt_icd_mismatch_model.joblib")
MISMATCH_META_PATH = os.path.join(MODEL_DIR, "cpt_icd_mismatch_model_meta.joblib")

APPEALS_SUCCESS_MODEL_PATH = os.path.join(MODEL_DIR, "appeals_success_model.joblib")
APPEALS_SUCCESS_META_PATH = os.path.join(MODEL_DIR, "appeals_success_model_meta.joblib")

APPEALS_RECOVERY_MODEL_PATH = os.path.join(MODEL_DIR, "appeals_recovery_model.joblib")
APPEALS_RECOVERY_META_PATH = os.path.join(MODEL_DIR, "appeals_recovery_model_meta.joblib")

FRAUD_PROB_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_probability_model.joblib")
FRAUD_PROB_META_PATH = os.path.join(MODEL_DIR, "fraud_probability_model_meta.joblib")

FRAUD_ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "fraud_anomaly_model.joblib")
FRAUD_ANOMALY_META_PATH = os.path.join(MODEL_DIR, "fraud_anomaly_model_meta.joblib")
ELIGIBILITY_MODEL_PATH = os.path.join(MODEL_DIR, "eligibility_risk_model.joblib")
ELIGIBILITY_META_PATH = os.path.join(MODEL_DIR, "eligibility_risk_meta.joblib")
RECON_MODEL_PATH = os.path.join(MODEL_DIR, "reconciliation_risk_model.joblib")
RECON_META_PATH = os.path.join(MODEL_DIR, "reconciliation_risk_meta.joblib")


CAT_FEATURES_BASE = ["insurance", "visit_type", "icd_code", "gender"]


class ConstantProbabilityModel:
    """
    Simple fallback classifier that always returns a fixed positive probability.
    Used when training labels have only one class.
    """

    def __init__(self, positive_probability: float):
        self.positive_probability = float(positive_probability)

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.positive_probability, dtype=float)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ConstantRegressorModel:
    """Simple fallback regressor returning a fixed value."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class ConstantForecastModel:
    """Simple fallback forecaster with fixed value."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)

def _get_cpt_code_feature_cols(df: pd.DataFrame):
    # data_loader.get_cpt_summary() creates columns like `cpt_<code>_count`.
    return [c for c in df.columns if c.startswith("cpt_") and c.endswith("_count")]


def _build_mismatch_feature_df(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame):
    df = master_df.merge(cpt_summary_df, on="claim_id", how="left")
    df["num_cpt_codes"] = df["num_cpt_codes"].fillna(1)
    df["total_cpt_amount"] = df["total_cpt_amount"].fillna(df["claim_amount"])

    # Prevent casting errors if any scrubbing columns are missing.
    for col in ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]:
        if col not in df.columns:
            df[col] = False
    df["cpt_icd_mismatch"] = df["cpt_icd_mismatch"].fillna(False)
    for col in ["high_amount_flag", "strict_insurance_flag"]:
        df[col] = df[col].fillna(False).astype(int)

    cpt_code_cols = _get_cpt_code_feature_cols(df)

    ohe = pd.get_dummies(df[CAT_FEATURES_BASE], drop_first=False)
    ohe_cols = list(ohe.columns)
    df_feat = pd.concat([df.drop(columns=CAT_FEATURES_BASE), ohe], axis=1)

    num_features = ["claim_amount", "age", "fraud_score", "num_cpt_codes", "total_cpt_amount"]
    bool_features = ["high_amount_flag", "strict_insurance_flag"]
    feature_cols = num_features + bool_features + cpt_code_cols + ohe_cols

    X = df_feat[feature_cols].fillna(0)
    y = df_feat["cpt_icd_mismatch"].astype(int)
    return X, y, feature_cols


def train_mismatch_model(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame):
    """
    Train a model to predict `cpt_icd_mismatch` using CPT composition + ICD/context.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, feature_cols = _build_mismatch_feature_df(master_df, cpt_summary_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        max_depth=6,
        n_estimators=220,
        learning_rate=0.08,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    # Full portfolio probabilities (aligned to master_df row order)
    all_probs = model.predict_proba(X)[:, 1]

    joblib.dump(model, MISMATCH_MODEL_PATH)
    joblib.dump(
        {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs},
        MISMATCH_META_PATH,
    )
    return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}}


def load_mismatch_model():
    if os.path.exists(MISMATCH_MODEL_PATH) and os.path.exists(MISMATCH_META_PATH):
        return joblib.load(MISMATCH_MODEL_PATH), joblib.load(MISMATCH_META_PATH)
    return None, None


def score_all_mismatch(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame, model, feature_cols):
    X, _, _ = _build_mismatch_feature_df(master_df, cpt_summary_df)
    X = X[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    return probs


# ============================================================
# Phase 4 — Appeals Success & Expected Recovery
# ============================================================

APPEALS_CAT_FEATURES = CAT_FEATURES_BASE + ["denial_reason"]
APPEALS_NUM_FEATURES = ["claim_amount", "age", "fraud_score", "num_cpt_codes", "total_cpt_amount"]
APPEALS_BOOL_FEATURES = ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]

def _build_appeals_feature_df(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame):
    df = master_df.merge(cpt_summary_df, on="claim_id", how="left")
    df["num_cpt_codes"] = df["num_cpt_codes"].fillna(1)
    df["total_cpt_amount"] = df["total_cpt_amount"].fillna(df["claim_amount"])

    for col in ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]:
        if col not in df.columns:
            df[col] = False
    for col in APPEALS_BOOL_FEATURES:
        df[col] = df[col].fillna(False).astype(int)

    if "denial_reason" not in df.columns:
        df["denial_reason"] = "None"
    df["denial_reason"] = df["denial_reason"].fillna("None").astype(str)

    cpt_code_cols = _get_cpt_code_feature_cols(df)

    ohe = pd.get_dummies(df[APPEALS_CAT_FEATURES], drop_first=False)
    ohe_cols = list(ohe.columns)
    df_feat = pd.concat([df.drop(columns=APPEALS_CAT_FEATURES), ohe], axis=1)

    feature_cols = APPEALS_NUM_FEATURES + APPEALS_BOOL_FEATURES + cpt_code_cols + ohe_cols
    X = df_feat[feature_cols].fillna(0)
    return X, feature_cols


def train_appeals_success_model(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame):
    """
    Predict P(appeal_success=True) on appealed claims.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, feature_cols = _build_appeals_feature_df(master_df, cpt_summary_df)
    X = X.reset_index(drop=True)
    df = master_df.reset_index(drop=True).copy()

    mask = df["is_appealed"].fillna(False).astype(bool).values
    y = df["appeal_success"].fillna(False).astype(bool).values.astype(int)
    y_masked = y[mask]

    # Degenerate case: only one class present among appealed claims.
    if len(y_masked) == 0 or len(np.unique(y_masked)) < 2:
        pos_prob = float(y_masked.mean()) if len(y_masked) > 0 else 0.0
        model = ConstantProbabilityModel(pos_prob)
        all_probs = model.predict_proba(X)[:, 1]
        y_pred_masked = model.predict(X.iloc[mask]) if len(y_masked) > 0 else np.array([], dtype=int)

        metrics = {
            "auc": None,  # Not defined for a single-class label set
            "precision": precision_score(y_masked, y_pred_masked, zero_division=0) if len(y_masked) > 0 else 0.0,
            "recall": recall_score(y_masked, y_pred_masked, zero_division=0) if len(y_masked) > 0 else 0.0,
            "f1": f1_score(y_masked, y_pred_masked, zero_division=0) if len(y_masked) > 0 else 0.0,
            "accuracy": accuracy_score(y_masked, y_pred_masked) if len(y_masked) > 0 else 0.0,
            "note": "Single-class appeals labels; using constant probability fallback.",
        }

        joblib.dump(model, APPEALS_SUCCESS_MODEL_PATH)
        joblib.dump(
            {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs},
            APPEALS_SUCCESS_META_PATH,
        )
        return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}}

    X_train, X_test, y_train, y_test = train_test_split(
        X.iloc[mask],
        y_masked,
        test_size=0.2,
        random_state=42,
        stratify=y_masked if len(np.unique(y_masked)) > 1 else None,
    )

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        max_depth=6,
        n_estimators=240,
        learning_rate=0.08,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    # All probability predictions aligned to master_df rows
    all_probs = model.predict_proba(X)[:, 1]

    joblib.dump(model, APPEALS_SUCCESS_MODEL_PATH)
    joblib.dump(
        {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs},
        APPEALS_SUCCESS_META_PATH,
    )
    return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}}


def load_appeals_success_model():
    if os.path.exists(APPEALS_SUCCESS_MODEL_PATH) and os.path.exists(APPEALS_SUCCESS_META_PATH):
        return joblib.load(APPEALS_SUCCESS_MODEL_PATH), joblib.load(APPEALS_SUCCESS_META_PATH)
    return None, None


def train_appeals_recovery_model(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame):
    """
    Predict expected paid_amount conditioned on successful appeals.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, feature_cols = _build_appeals_feature_df(master_df, cpt_summary_df)
    X = X.reset_index(drop=True)
    df = master_df.reset_index(drop=True).copy()

    mask_success = (df["is_appealed"].fillna(False).astype(bool).values) & (df["appeal_success"].fillna(False).astype(bool).values)
    y = df["paid_amount"].fillna(0).values.astype(float)
    y_success = y[mask_success]

    # Degenerate case: no successful appeals available for supervised regression.
    if len(y_success) == 0:
        fallback_value = 0.0
        model = ConstantRegressorModel(fallback_value)
        all_paid_pred = model.predict(X)
        metrics = {
            "mae": None,
            "rmse": None,
            "note": "No successful appeals found; using constant recovery fallback.",
        }

        joblib.dump(model, APPEALS_RECOVERY_MODEL_PATH)
        joblib.dump(
            {"feature_cols": feature_cols, "metrics": metrics, "all_paid_predictions": all_paid_pred},
            APPEALS_RECOVERY_META_PATH,
        )
        return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_paid_predictions": all_paid_pred}}

    # If only one sample exists, avoid train/test split and use constant fallback.
    if len(y_success) < 2:
        fallback_value = float(np.mean(y_success))
        model = ConstantRegressorModel(fallback_value)
        all_paid_pred = model.predict(X)
        metrics = {
            "mae": 0.0,
            "rmse": 0.0,
            "note": "Insufficient successful appeals samples; using constant recovery fallback.",
        }

        joblib.dump(model, APPEALS_RECOVERY_MODEL_PATH)
        joblib.dump(
            {"feature_cols": feature_cols, "metrics": metrics, "all_paid_predictions": all_paid_pred},
            APPEALS_RECOVERY_META_PATH,
        )
        return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_paid_predictions": all_paid_pred}}

    X_train, X_test, y_train, y_test = train_test_split(
        X.iloc[mask_success], y[mask_success], test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        max_depth=6,
        n_estimators=280,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Basic regression diagnostics
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }

    # All predicted paid amounts (still only learned from success subset)
    all_paid_pred = model.predict(X)

    joblib.dump(model, APPEALS_RECOVERY_MODEL_PATH)
    joblib.dump(
        {"feature_cols": feature_cols, "metrics": metrics, "all_paid_predictions": all_paid_pred},
        APPEALS_RECOVERY_META_PATH,
    )
    return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_paid_predictions": all_paid_pred}}


def load_appeals_recovery_model():
    if os.path.exists(APPEALS_RECOVERY_MODEL_PATH) and os.path.exists(APPEALS_RECOVERY_META_PATH):
        return joblib.load(APPEALS_RECOVERY_MODEL_PATH), joblib.load(APPEALS_RECOVERY_META_PATH)
    return None, None


def predict_appeals_ranked_candidates(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame,
                                       success_model, success_feature_cols,
                                       recovery_model, recovery_feature_cols):
    X, feature_cols = _build_appeals_feature_df(master_df, cpt_summary_df)

    # Align scoring matrix to training-time feature sets.
    # This avoids KeyError when some one-hot columns are absent in the current batch.
    X_s = X.reindex(columns=success_feature_cols, fill_value=0)
    X_r = X.reindex(columns=recovery_feature_cols, fill_value=0)

    p_success = success_model.predict_proba(X_s)[:, 1]
    paid_given_success = recovery_model.predict(X_r)
    expected_recovery = p_success * paid_given_success

    out = master_df.copy()
    out = out.reset_index(drop=True)
    out["p_appeal_success"] = p_success
    out["expected_recovery"] = expected_recovery
    return out


# ============================================================
# Phase 5 — Fraud Probability Enhancement (Supervised + Anomaly)
# ============================================================

FRAUD_CAT_FEATURES = CAT_FEATURES_BASE + ["denial_reason"]
FRAUD_NUM_FEATURES = ["claim_amount", "age", "num_cpt_codes", "total_cpt_amount", "revenue_leakage", "collection_rate"]
FRAUD_BOOL_FEATURES = ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag"]

FRAUD_NUMERIC_ANOMALY_COLS = ["claim_amount", "age", "revenue_leakage", "collection_rate"]


def _build_fraud_feature_df(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame, include_fraud_score: bool = False):
    df = master_df.merge(cpt_summary_df, on="claim_id", how="left")
    df["num_cpt_codes"] = df["num_cpt_codes"].fillna(1)
    df["total_cpt_amount"] = df["total_cpt_amount"].fillna(df["claim_amount"])

    if "denial_reason" not in df.columns:
        df["denial_reason"] = "None"
    df["denial_reason"] = df["denial_reason"].fillna("None").astype(str)

    for col in FRAUD_BOOL_FEATURES:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(int)

    cpt_code_cols = _get_cpt_code_feature_cols(df)

    ohe = pd.get_dummies(df[FRAUD_CAT_FEATURES], drop_first=False)
    ohe_cols = list(ohe.columns)
    df_feat = pd.concat([df.drop(columns=FRAUD_CAT_FEATURES), ohe], axis=1)

    num_features = list(FRAUD_NUM_FEATURES)
    if include_fraud_score and "fraud_score" in df_feat.columns:
        num_features.append("fraud_score")

    feature_cols = num_features + FRAUD_BOOL_FEATURES + cpt_code_cols + ohe_cols
    X = df_feat[feature_cols].fillna(0)
    return X, feature_cols


def train_fraud_probability_model(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame):
    """
    Predict probability of fraud_flag using supervised learning.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, feature_cols = _build_fraud_feature_df(master_df, cpt_summary_df, include_fraud_score=False)
    y = master_df["fraud_flag"].fillna(False).astype(bool).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        max_depth=6,
        n_estimators=260,
        learning_rate=0.08,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    all_probs = model.predict_proba(X)[:, 1]

    joblib.dump(model, FRAUD_PROB_MODEL_PATH)
    joblib.dump({"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}, FRAUD_PROB_META_PATH)
    return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}}


def load_fraud_probability_model():
    if os.path.exists(FRAUD_PROB_MODEL_PATH) and os.path.exists(FRAUD_PROB_META_PATH):
        return joblib.load(FRAUD_PROB_MODEL_PATH), joblib.load(FRAUD_PROB_META_PATH)
    return None, None


def train_fraud_anomaly_model(master_df: pd.DataFrame):
    """
    Fit an IsolationForest over numeric "weirdness" features (unsupervised anomaly).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = master_df.copy()
    for col in FRAUD_NUMERIC_ANOMALY_COLS:
        if col not in df.columns:
            df[col] = 0
    X_num = df[FRAUD_NUMERIC_ANOMALY_COLS].fillna(0).astype(float).values

    # IsolationForest: smaller decision_function => more anomalous. We'll invert later.
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_num)

    joblib.dump(model, FRAUD_ANOMALY_MODEL_PATH)
    joblib.dump({"numeric_cols": FRAUD_NUMERIC_ANOMALY_COLS}, FRAUD_ANOMALY_META_PATH)
    return {"model": model, "meta": {"numeric_cols": FRAUD_NUMERIC_ANOMALY_COLS}}


def load_fraud_anomaly_model():
    if os.path.exists(FRAUD_ANOMALY_MODEL_PATH) and os.path.exists(FRAUD_ANOMALY_META_PATH):
        return joblib.load(FRAUD_ANOMALY_MODEL_PATH), joblib.load(FRAUD_ANOMALY_META_PATH)
    return None, None


def score_fraud_enhanced(master_df: pd.DataFrame, cpt_summary_df: pd.DataFrame,
                         fraud_prob_model, fraud_prob_feature_cols,
                         fraud_anomaly_model, fraud_anomaly_meta,
                         alpha: float = 0.6):
    X, _ = _build_fraud_feature_df(master_df, cpt_summary_df, include_fraud_score=False)
    X = X[fraud_prob_feature_cols]
    p_fraud = fraud_prob_model.predict_proba(X)[:, 1]

    # Anomaly score -> probability-like scaling (min-max over scored set).
    cols = fraud_anomaly_meta.get("numeric_cols", FRAUD_NUMERIC_ANOMALY_COLS)
    df = master_df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    X_num = df[cols].fillna(0).astype(float).values
    # More anomalous => larger `anomaly_strength`
    anomaly_strength = -fraud_anomaly_model.decision_function(X_num)
    a_min, a_max = float(np.min(anomaly_strength)), float(np.max(anomaly_strength))
    if a_max - a_min < 1e-12:
        p_anomaly = np.zeros_like(anomaly_strength, dtype=float)
    else:
        p_anomaly = (anomaly_strength - a_min) / (a_max - a_min)

    improved = alpha * p_fraud + (1 - alpha) * p_anomaly
    out = master_df.copy().reset_index(drop=True)
    out["fraud_probability"] = p_fraud
    out["fraud_anomaly_probability"] = p_anomaly
    out["fraud_probability_improved"] = improved
    return out


# ============================================================
# Front-End Automation: Eligibility Risk Model
# ============================================================

ELIGIBILITY_FEATURES = [
    "insurance",
    "visit_type",
    "claim_amount",
    "age",
    "high_amount_flag",
    "strict_insurance_flag",
    "insurance_match_flag",
]


def _build_eligibility_feature_df(master_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = master_df.copy()
    if "insurance_pat" not in df.columns:
        df["insurance_pat"] = df["insurance"]
    df["insurance_pat"] = df["insurance_pat"].fillna("Unknown")
    df["insurance_match_flag"] = (df["insurance"].astype(str) == df["insurance_pat"].astype(str)).astype(int)
    df["high_amount_flag"] = df["high_amount_flag"].fillna(False).astype(int)
    df["strict_insurance_flag"] = df["strict_insurance_flag"].fillna(False).astype(int)

    cat_cols = ["insurance", "visit_type"]
    num_cols = ["claim_amount", "age", "high_amount_flag", "strict_insurance_flag", "insurance_match_flag"]
    ohe = pd.get_dummies(df[cat_cols], drop_first=False)
    X = pd.concat([df[num_cols], ohe], axis=1).fillna(0)
    feature_cols = list(X.columns)
    return X, feature_cols


def train_eligibility_risk_model(master_df: pd.DataFrame):
    """
    Predict eligibility/registration risk using proxy target:
      denial_reason in {'Coverage', 'Auth required'}.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, feature_cols = _build_eligibility_feature_df(master_df)
    y = master_df["denial_reason"].fillna("None").isin(["Coverage", "Auth required"]).astype(int).values

    # Degenerate labels fallback
    if len(np.unique(y)) < 2:
        model = ConstantProbabilityModel(float(np.mean(y)))
        probs = model.predict_proba(X)[:, 1]
        metrics = {"auc": None, "note": "Single-class labels; constant fallback."}
        joblib.dump(model, ELIGIBILITY_MODEL_PATH)
        joblib.dump({"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": probs}, ELIGIBILITY_META_PATH)
        return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": probs}}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        max_depth=5,
        n_estimators=180,
        learning_rate=0.08,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }
    all_probs = model.predict_proba(X)[:, 1]
    joblib.dump(model, ELIGIBILITY_MODEL_PATH)
    joblib.dump({"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}, ELIGIBILITY_META_PATH)
    return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}}


def load_eligibility_risk_model():
    if os.path.exists(ELIGIBILITY_MODEL_PATH) and os.path.exists(ELIGIBILITY_META_PATH):
        return joblib.load(ELIGIBILITY_MODEL_PATH), joblib.load(ELIGIBILITY_META_PATH)
    return None, None


def score_eligibility_risk(master_df: pd.DataFrame, model, feature_cols: list[str]) -> np.ndarray:
    X, _ = _build_eligibility_feature_df(master_df)
    X = X.reindex(columns=feature_cols, fill_value=0)
    return model.predict_proba(X)[:, 1]


# ============================================================
# Revenue Forecasting Model (monthly collected revenue)
# ============================================================

def fit_revenue_forecast_model(monthly_df: pd.DataFrame, value_col: str = "collected"):
    """
    Fit a simple linear trend model for monthly revenue forecasting.
    Returns fitted values + next period forecast.
    """
    if monthly_df is None or len(monthly_df) == 0:
        model = ConstantForecastModel(0.0)
        return {"model": model, "fitted": np.array([]), "next_forecast": 0.0, "metrics": {"note": "No data"}}

    y = monthly_df[value_col].fillna(0).values.astype(float)
    X = np.arange(len(y)).reshape(-1, 1)

    if len(y) < 2:
        model = ConstantForecastModel(float(y.mean()))
        fitted = model.predict(X)
        next_forecast = float(model.predict(np.array([[len(y)]], dtype=float))[0])
        return {"model": model, "fitted": fitted, "next_forecast": next_forecast, "metrics": {"note": "Insufficient history; constant fallback."}}

    model = LinearRegression()
    model.fit(X, y)
    fitted = model.predict(X)
    next_forecast = float(model.predict(np.array([[len(y)]], dtype=float))[0])
    # R^2 as rough fit quality
    r2 = float(model.score(X, y))
    return {"model": model, "fitted": fitted, "next_forecast": next_forecast, "metrics": {"r2": r2}}


# ============================================================
# Payment Reconciliation Risk Model
# ============================================================

RECON_CAT_FEATURES = ["insurance", "visit_type", "icd_code"]
RECON_NUM_FEATURES = ["claim_amount", "paid_amount", "revenue_leakage", "collection_rate", "age", "fraud_score"]
RECON_BOOL_FEATURES = ["cpt_icd_mismatch", "high_amount_flag", "strict_insurance_flag", "is_denied"]


def _build_recon_feature_df(master_df: pd.DataFrame):
    df = master_df.copy()
    for col in RECON_BOOL_FEATURES:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(int)

    ohe = pd.get_dummies(df[RECON_CAT_FEATURES], drop_first=False)
    ohe_cols = list(ohe.columns)
    X = pd.concat([df[RECON_NUM_FEATURES + RECON_BOOL_FEATURES], ohe], axis=1).fillna(0)
    feature_cols = list(X.columns)
    return X, feature_cols


def train_reconciliation_risk_model(master_df: pd.DataFrame):
    """
    Predict whether a claim likely needs reconciliation review.
    Proxy target: posting_gap > dynamic threshold.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = master_df.copy()
    df["posting_gap"] = (df["claim_amount"] - df["paid_amount"]).clip(lower=0)
    # Dynamic threshold at 60th percentile of non-zero gap (or 50 fallback).
    non_zero = df[df["posting_gap"] > 0]["posting_gap"]
    thresh = float(non_zero.quantile(0.6)) if len(non_zero) > 0 else 50.0
    thresh = max(thresh, 50.0)
    y = (df["posting_gap"] > thresh).astype(int).values

    X, feature_cols = _build_recon_feature_df(df)

    if len(np.unique(y)) < 2:
        model = ConstantProbabilityModel(float(np.mean(y)))
        all_probs = model.predict_proba(X)[:, 1]
        metrics = {"auc": None, "note": "Single-class labels; constant fallback.", "gap_threshold": thresh}
        joblib.dump(model, RECON_MODEL_PATH)
        joblib.dump({"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}, RECON_META_PATH)
        return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
    model = XGBClassifier(
        max_depth=5,
        n_estimators=200,
        learning_rate=0.08,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else None,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "gap_threshold": thresh,
    }
    all_probs = model.predict_proba(X)[:, 1]
    joblib.dump(model, RECON_MODEL_PATH)
    joblib.dump({"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}, RECON_META_PATH)
    return {"model": model, "meta": {"feature_cols": feature_cols, "metrics": metrics, "all_probabilities": all_probs}}


def load_reconciliation_risk_model():
    if os.path.exists(RECON_MODEL_PATH) and os.path.exists(RECON_META_PATH):
        return joblib.load(RECON_MODEL_PATH), joblib.load(RECON_META_PATH)
    return None, None


def score_reconciliation_risk(master_df: pd.DataFrame, model, feature_cols: list[str]) -> np.ndarray:
    X, _ = _build_recon_feature_df(master_df)
    X = X.reindex(columns=feature_cols, fill_value=0)
    return model.predict_proba(X)[:, 1]
