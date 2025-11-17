import numpy as np
import joblib
from build_402_features import build_402_features


# -----------------------------
# Load trained model (XGBoost)
# -----------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    raise RuntimeError(f"ERROR: Could not load model.pkl → {e}")


# -----------------------------
# Fraud Prediction Function
# -----------------------------
def predict_fraud(raw_input_dict):
    """
    raw_input_dict = dictionary from Streamlit UI fields
                     Example: {"TransactionAmt": 120.5, "DeviceInfo": "Windows", ...}

    Steps:
      1. Convert raw UI inputs to a DataFrame with 402 engineered features
      2. Ensure model-compatible column order
      3. Run XGBoost predict_proba
      4. Return prediction + probability
    """

    # STEP 1 — Convert raw UI input → full 402-feature vector
    try:
        X = build_402_features(raw_input_dict)   # Returns pandas DataFrame (1 row)
    except Exception as e:
        raise RuntimeError(f"[build_402_features] FAILED → {e}")

    if X.shape[1] != 402:
        raise ValueError(f"Expected 402 features, but got {X.shape[1]} features.")

    # STEP 2 — Predict probabilities
    try:
        proba = float(model.predict_proba(X)[0][1])   # probability of fraud = class 1
    except Exception as e:
        raise RuntimeError(f"Model prediction failed → {e}")

    # STEP 3 — Convert probability → class label
    pred = int(proba >= 0.50)   # threshold 0.50

    return pred, proba
