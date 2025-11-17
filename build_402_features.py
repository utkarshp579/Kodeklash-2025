import joblib
import numpy as np
import pandas as pd

# Loads the scaler that contains feature_names_in_
SCALER_PKL = "scaler.pkl"  # adjust name if different, e.g., all_scalers.pkl

def build_402_features(cleaned):
    """
    Build a 1x402 vector (DataFrame) in the exact order expected by the model.
    Uses scaler.feature_names_in_ to order and fill missing with sensible defaults.
    """
    scaler = joblib.load(SCALER_PKL)
    feature_names = list(scaler.feature_names_in_)

    # start with zeros
    features = {fn: 0.0 for fn in feature_names}

    # Basic transaction fields
    features["TransactionID"] = cleaned.get("TransactionID", 0)
    features["TransactionAmt"] = cleaned.get("TransactionAmt", 0.0)
    features["TransactionDT"] = cleaned.get("TransactionDT", 0.0)

    # Card, address, dist
    for c in ["card1", "card2", "card3", "card5", "addr1", "addr2", "dist1"]:
        if c in features and c in cleaned:
            features[c] = float(cleaned.get(c, 0.0))

    # Email domains map: keep as label encoded integers if your saved encoders expect ints.
    # Here we use domain names raw; if model expects encoded ints, you must use label encoder saved.
    features["P_emaildomain"] = cleaned.get("P_emaildomain", "unknown")
    features["R_emaildomain"] = cleaned.get("R_emaildomain", "unknown")

    # Device info & type
    features["DeviceInfo"] = cleaned.get("DeviceInfo", 0)
    features["DeviceType"] = cleaned.get("DeviceType", -999)

    # Add V, C, D features from the small UI subset. If key exists in feature list set it.
    V = cleaned.get("V_data", {})
    for k, v in V.items():
        if k in features:
            try:
                features[k] = float(v)
            except:
                features[k] = 0.0

    C = cleaned.get("C_data", {})
    for k, v in C.items():
        if k in features:
            try:
                features[k] = float(v)
            except:
                features[k] = 0.0

    D = cleaned.get("D_data", {})
    for k, v in D.items():
        if k in features:
            try:
                features[k] = float(v)
            except:
                features[k] = 0.0

    # If PCA outputs exist in cleaned (rare), set them
    if "PCA_V_1" in feature_names and cleaned.get("PCA_V_data") is not None:
        pca_v = cleaned.get("PCA_V_data")
        # try to map first few
        try:
            features["PCA_V_1"] = float(pca_v[0][0])
        except:
            pass

    # Final: build DataFrame with correct column ordering
    X = pd.DataFrame([features], columns=feature_names)
    return X
