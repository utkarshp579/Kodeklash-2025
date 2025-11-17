import numpy as np

def preprocess_raw_input(data):
    """
    Convert UI data dict (from streamlitapp.py) into a cleaned dict
    expected by build_402_features.build_402_features.
    Keep keys that your build_402_features will use.
    """
    cleaned = {
        "TransactionID": int(data.get("Transaction ID", 0)),
        "TransactionDT": float(data.get("Transaction Date", 0)),
        "TransactionAmt": float(np.exp(data.get("Transaction Amount", 0))) if data.get("Transaction Amount") is not None else 0.0,
        "card1": float(data.get("Card Data", {}).get("Card1", 0)) if data.get("Card Data", {}) else 0.0,
        "card2": float(data.get("Card Data", {}).get("Card2", 0)) if data.get("Card Data", {}) else 0.0,
        "card3": float(data.get("Card Data", {}).get("Card3", 0)) if data.get("Card Data", {}) else 0.0,
        "card5": float(data.get("Card Data", {}).get("Card5", 0)) if data.get("Card Data", {}) else 0.0,
        "addr1": float(data.get("Address 1", 0)),
        "addr2": float(data.get("Address 2", 0)),
        "dist1": float(data.get("Distance", 0)),
        "P_emaildomain": data.get("Purchaser Email Domain", "unknown"),
        "R_emaildomain": data.get("Recipient Email Domain", "unknown"),
        "DeviceInfo": data.get("Device Data", {}).get("DeviceInfo", "unknown"),
        "DeviceType": data.get("Device Data", {}).get("DeviceType", -999),
        "Hours": data.get("Hours", 0),
        "Days": data.get("Days", 0),
        "Weekdays": data.get("Weekdays", 0),
        # Keep behavioral blocks to be expanded by build_402_features
        "V_data": data.get("Behavioral Data", {}),
        "C_data": data.get("Transactional Usage Data", {}),
        "D_data": data.get("Transaction Time Behavioral Data", {}),
        # other billing fields
        "M_data": data.get("Billing Data", {})
    }
    return cleaned
