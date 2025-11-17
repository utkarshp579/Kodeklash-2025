# 🛡️ FraudLense  
### **AI-Powered Online Payment Fraud Detection System**

FraudLense is an intelligent machine learning–based system built to identify fraudulent online payment transactions in real-time. Using the IEEE-CIS Fraud Detection dataset, the system applies **advanced feature engineering**, **PCA-based dimensionality reduction**, and **XGBoost classification** to deliver highly accurate, low-latency predictions.  
The project is designed for scalable fintech environments and aligns strongly with the **Fintech for Bharat** mission to make digital payments safe and accessible for all.

---

## 🌐 Live Application  
🔗 **Streamlit App:**  
[https://online-payment-fraud-detector.streamlit.app/](https://hari659tri-kodeklash-2025-codebuilt-fraudl-streamlit-app-avlwk2.streamlit.app/)

🔗 **Youtube Video:**


---

## 🚀 Overview  
As India rapidly transitions into a digital-first economy, online payment fraud has become a critical challenge. Fraudsters exploit device spoofing, behavioral manipulation, and high-velocity attacks to bypass traditional rule-based systems.

FraudLense solves these challenges by combining:

- ⚡ **Real-time ML inference**  
- 📉 **Dimensionality reduction (394 → 53 features)**  
- 🤖 **XGBoost with tuned hyperparameters**  
- 🧩 **KMeans clustering for behavioral pattern detection**  
- 🖥️ **A smooth, interactive Streamlit UI**

The result is a fast, reliable, scalable solution ready for real-world fintech deployment.

---

## 🎯 Key Features  
- **High-Accuracy Fraud Detection** using XGBoost (AUC ROC: **0.9556**)  
- **PCA-Optimized Feature Space** for improved model efficiency  
- **Streamlit Web App** for instant predictions  
- **KMeans clustering** to detect hidden transactional patterns  
- **Smart Preprocessing Pipeline** (scaling, label encoding, missing value handling)  
- **Pickle-based deployment** for portability  
- **Designed for large-scale, real-time systems**

---

## 🧠 System Architecture

### 🔹 1. Data Exploration & Analysis  
- IEEE-CIS dataset (anonymized transaction, device, and behavioural data)  
- Outlier detection and fraud distribution mapping  

### 🔹 2. Preprocessing  
- Numerical scaling  
- Missing value imputation  
- Label encoding  
- Noise filtering  

### 🔹 3. Feature Engineering  
- PCA applied to V-, C-, and D-type feature groups  
- Reduced 394 engineered features → **53 most important features**

### 🔹 4. Model Training (XGBoost)  
- Hyperparameter tuning  
- Class imbalance handling  
- Cross-validation  
### 📊 Final Performance Metrics
🔹 **CV Score 1:** 0.9462   
🔹 **CV Score 2:** 0.9620   
🔹 **CV Score 3:** 0.9386   
✨ **Mean AUC ROC:** **0.9756**

### 🔹 5. Model Packaging  
- `model.pkl` — trained XGBoost model  
- `pca.pkl` — dimensionality-reduction model  
- `preprocessing.py` — full preprocessing pipeline  

### 🔹 6. Deployment (Streamlit Cloud)  
- Real-time inference  
- User-friendly experience  
- API-ready system architecture  

---

## 📲 How to Use  
1. Open the Streamlit app  
2. Enter details such as:  
   - Transaction amount  
   - Transaction type (credit/debit)  
   - Card brand  
   - Device type  
   - Behavioural cues  
3. Click **Predict**  
4. View result:  
   - 🟢 *Genuine*  
   - 🔴 *Fraudulent* (High Risk)

---

## 🧩 Practical Use Cases  
- **Banks & Fintechs:** Instant risk scoring for payments  
- **UPI Payment Platforms:** High-volume fraud screening  
- **E-commerce:** Reduce “card-not-present” fraud  
- **Payment Gateways:** Detect behavioural anomalies  
- **Credit Card Networks:** Identify suspicious patterns across devices & geographies  

---

## 🇮🇳 Why FraudLense Fits the *Fintech for Bharat* Vision  
FraudLense empowers digital security for millions of new internet users across Bharat—especially Tier-2, Tier-3, and rural regions adopting UPI and online payments for the first time.  
By offering fast, accurate fraud detection with a simple interface, the project strengthens trust in digital finance and supports the vision of a secure, inclusive, and scalable fintech ecosystem for India.

---

## 🔧 Challenges & Learnings  
### 🟠 Memory Overload During Preprocessing  
Large feature sets caused RAM issues → optimized using chunk processing and dtype reduction.

### 🟠 Label Encoding Failures  
Unexpected categories triggered errors → added validation and fallback mappings.

### 🟠 Streamlit Secrets Configuration Errors  
Missing keys caused UI breaks → implemented safe loaders and key checks.

### 🟠 Google Drive Pickle File Failures  
Corrupted downloads due to invalid IDs → restructured downloader with verified file IDs.

### 🟠 UI Stability Issues  
Broken components when users skipped fields → implemented validation wrappers and default values.

These challenges improved the robustness, reliability, and production-readiness of the entire pipeline.

---

## 🛠️ Tech Stack  
- **Python 3.8+**  
- **XGBoost**  
- **Scikit-learn**  
- **Pandas, NumPy**  
- **Matplotlib/Seaborn**  
- **Streamlit**  
- **Pickle**

---

## 🗂️ Dataset  
This project uses the **IEEE-CIS Fraud Detection Dataset**, one of the largest and most complex datasets for transaction fraud detection.  
It includes anonymized details such as:  
- Device characteristics  
- Transaction timing  
- Card information  
- Behavioural variables  
- Transaction amounts  

---

## ⚠️ Disclaimer  
This project is intended for **educational and research purposes only**.  
Do not use these predictions for actual financial or commercial decision-making.  
Accuracy depends on input quality, preprocessing, and dataset limitations.

---

## 👥 Team  
- **Harikesh Tripathi**  
- **Sandhya Pandey**  
- **Utkarsh Pandey**

---

## ⭐ Support  
If you found this project useful, please consider giving it a **⭐ on GitHub**.  
Your support motivates future improvements and open-source contributions!

