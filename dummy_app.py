import streamlit as st
import numpy as np
import pickle
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# ------------------------------
# Step 1: Create/load dummy models
# ------------------------------

# You can skip this step if you've already saved dummy models
# For testing, we create dummy PCA and model
def create_dummy_models():
    # Dummy data: 100 samples, 6 features
    X_dummy = np.random.rand(100, 6)
    y_dummy = np.random.randint(0, 2, 100)
    
    # PCA
    pca_dummy = PCA(n_components=3)
    pca_dummy.fit(X_dummy)
    
    # XGBoost
    model_dummy = XGBClassifier(
        max_depth=3,
        n_estimators=10,
        learning_rate=0.1,
        eval_metric="logloss"
    )
    model_dummy.fit(X_dummy, y_dummy)
    
    # Save models
    pickle.dump(pca_dummy, open("PCA_dummy.pkl", "wb"))
    pickle.dump(model_dummy, open("model_dummy.pkl", "wb"))

# Uncomment to create dummy models (run once)
#create_dummy_models()

# Load dummy models
pca_dummy = pickle.load(open("PCA_dummy.pkl", "rb"))
model_dummy = pickle.load(open("model_dummy.pkl", "rb"))

# ------------------------------
# Step 2: Streamlit UI
# ------------------------------
st.title("💳 Dummy Fraud Detection")

st.markdown("Enter transaction details to test the fraud prediction:")

# User inputs
TransactionAmt = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
card4 = st.selectbox("Card Brand", ["Visa", "Mastercard", "Amex", "Discover"])
card6 = st.selectbox("Card Type", ["Debit", "Credit"])
addr1 = st.number_input("Address1", min_value=0, value=100)
addr2 = st.number_input("Address2", min_value=0, value=50)
dist1 = st.number_input("Distance from merchant", min_value=0.0, value=10.0)

# ------------------------------
# Step 3: Encode categorical features
# ------------------------------
card4_map = {"Visa":0, "Mastercard":1, "Amex":2, "Discover":3}
card6_map = {"Debit":0, "Credit":1}

user_input = np.array([[TransactionAmt, card4_map[card4], card6_map[card6], addr1, addr2, dist1]])

# ------------------------------
# Step 4: Predict button
# ------------------------------
if st.button("Predict Fraud"):
    # Transform using PCA
    user_input_pca = pca_dummy.transform(user_input)
    
    # Predict
    prediction = model_dummy.predict(user_input)
    
    if prediction[0] == 1:
        st.error("⚠️ This transaction is predicted as FRAUD.")
    else:
        st.success("✅ This transaction is predicted as NOT fraud.")
