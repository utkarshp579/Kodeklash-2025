# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

print("🔄 Loading datasets...")
df_tr = pd.read_csv("train_transaction.csv")
df_id = pd.read_csv("train_identity.csv")

print("🔄 Merging...")
df = df_tr.merge(df_id, on="TransactionID", how="left")

print("🔄 Cleaning...")
df = df.fillna(0)

# Target variable
y = df["isFraud"]
df = df.drop(columns=["isFraud"])

# Encode categorical columns
print("🔄 Encoding categorical features...")
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ label_encoders.pkl saved.")

# Scale all numeric features
print("🔄 Scaling 402 features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

joblib.dump(scaler, "scaler.pkl")
print("✅ scaler.pkl saved.")

# Train XGBoost
print("🚀 Training XGBoost on CPU...")
model = XGBClassifier(
    n_estimators=250,
    max_depth=10,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist"
)

model.fit(X_scaled, y)

joblib.dump(model, "model.pkl")
print("✅ model.pkl saved.")

print("🎉 Training completed successfully!")
