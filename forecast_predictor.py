# forecast_predictor.py
import os
import numpy as np
import joblib


MODELS_DIR = "models"
MODEL_7  = os.path.join(MODELS_DIR, "stress_y_7.joblib")
MODEL_14 = os.path.join(MODELS_DIR, "stress_y_14.joblib")


def _load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)


def predict_stress(row_df):
    """
    row_df: single-row DataFrame with numeric features.
    Returns dict: {"y_7": float, "y_14": float}
    """
    m7 = _load_model(MODEL_7)
    m14 = _load_model(MODEL_14)

    # Align columns to model features (very important!)
    feat7 = getattr(m7, "feature_name_", None) or getattr(m7, "feature_names_in_", None)
    feat14 = getattr(m14, "feature_name_", None) or getattr(m14, "feature_names_in_", None)

    X7 = row_df.reindex(columns=list(feat7), fill_value=np.nan) if feat7 is not None else row_df
    X14 = row_df.reindex(columns=list(feat14), fill_value=np.nan) if feat14 is not None else row_df

    y7 = float(m7.predict(X7)[0])
    y14 = float(m14.predict(X14)[0])

    return {"y_7": y7, "y_14": y14}


def cultivation_recommendation(current_stress, y7, y14):
    """
    Simple advisory text.
    """
    worst = max(current_stress, y7, y14)

    if worst < 0.30:
        return "✅ LOW STRESS: Maintain current irrigation and monitoring."
    if worst < 0.60:
        return "⚠️ MODERATE STRESS: Check irrigation timing, soil moisture, and heat stress mitigation."
    return "🛑 HIGH STRESS: Increase irrigation priority, inspect crop health, and consider protective actions."
