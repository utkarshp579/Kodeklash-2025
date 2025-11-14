import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

# ----------------- 1. Load Dataset -----------------
print("🚀 Loading dataset...")
train_tr = pd.read_csv("train_transaction.csv")
train_id = pd.read_csv("train_identity.csv")

print("🔗 Merging datasets...")
df = train_tr.merge(train_id, on="TransactionID", how="left")

# ----------------- 2. Sample & Clean -----------------
print("📉 Using 10% of data for lightweight training...")
df = df.sample(frac=0.10, random_state=42)

print("🧹 Filling missing values...")
df.fillna(-999, inplace=True)

# ----------------- 3. Separate Features -----------------
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('isFraud')  # Remove target

# Example splitting: V, D, C features
V_features = [col for col in numeric_features if col.startswith("V")]
D_features = [col for col in numeric_features if col.startswith("D")]
C_features = [col for col in numeric_features if col.startswith("C")]

# ----------------- 4. Prepare Target -----------------
y = df['isFraud']

# ----------------- 5. PCA for each feature group -----------------
print("🧪 Applying PCA...")

def train_pca(features, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(df[features])
    return pca

pca_V = train_pca(V_features)
pca_D = train_pca(D_features)
pca_C = train_pca(C_features)

# ----------------- 6. KMeans for each feature group -----------------
print("🔍 Applying KMeans...")

def train_kmeans(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df[features])
    return kmeans

kmeans_V = train_kmeans(V_features)
kmeans_D = train_kmeans(D_features)
kmeans_C = train_kmeans(C_features)

# ----------------- 7. XGBoost Model -----------------
print("🤖 Training XGBoost model...")
X = df[numeric_features]  # Use all numeric features together
model = XGBClassifier(
    max_depth=5,
    n_estimators=50,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric="logloss",
    use_label_encoder=False
)
model.fit(X, y)

# ----------------- 8. Save Models -----------------
print("💾 Saving models...")

pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(pca_V, open("PCA_V_features.pkl", "wb"))
pickle.dump(pca_D, open("PCA_D_features.pkl", "wb"))
pickle.dump(pca_C, open("PCA_C_features.pkl", "wb"))

pickle.dump(kmeans_V, open("km_V_features.pkl", "wb"))
pickle.dump(kmeans_D, open("km_D_features.pkl", "wb"))
pickle.dump(kmeans_C, open("km_C_features.pkl", "wb"))

# ----------------- 9. Save Label Encoder -----------------
le = LabelEncoder()
le.fit(y)
pickle.dump(le, open("labels.pkl", "wb"))

print("\n🎉 All models saved successfully!")
