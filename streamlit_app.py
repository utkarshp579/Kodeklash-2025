import streamlit as st
import numpy as np
import datetime
import pickle
import gdown
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from preprocessing import preprocess_raw_input
from build_402_features import build_402_features
from predict_fraud import predict_fraud
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(
    page_title="FraudLens – Fraud Detection",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .big-title {
        font-size: 38px !important;
        font-weight: 700 !important;
        color: #3E64FF;
    }
    .sub-text {
        font-size: 16px !important;
        color: #555;
    }
    .box {
        padding: 20px;
        background: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e1e1e1;
    }
    .section-title {
        font-size: 24px !important;
        font-weight: 600 !important;
        padding-bottom: 10px;
        color: #3E64FF;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🛡️ FraudLens – Smart Fraud Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">A clean, modern and simplified interface for fraud prediction</p>', unsafe_allow_html=True)
add_vertical_space(1)

# st.divider()



abstract_content = '''
<p>This project focuses on developing an online payment fraud detection system using advanced machine learning and the IEEE-CIS Fraud Detection dataset. It accurately identifies potentially fraudulent transactions while keeping the process simple and user-friendly. Principal Component Analysis (PCA) is used to reduce features, particularly V1–V339, D-type, and C-type features, improving efficiency. The system employs XGBoost for prediction and KMeans clustering to detect patterns in transaction data. With hyperparameter tuning, the models are made more accurate and reliable, offering a fast, practical, and trustworthy solution for online payment fraud detection.</p>

<p>The application features a user-friendly Streamlit interface, where users answer simple yes/no and multiple-choice questions. These responses are transformed using pre-trained PCA models and analyzed with an XGBoost classifier to predict potential fraud. The system efficiently handles missing values, prioritizes the most important features to minimize user input, and incorporates clustering techniques to examine behavioral and transaction patterns. It also leverages device and behavioral data through Vesta’s proprietary features, providing thorough and reliable fraud detection.</p>

<p>The project has greatly improved the handling of both categorical and numerical data, optimized model scaling, and reduced the amount of user input needed without compromising prediction accuracy. The PCA and clustering models are saved in pickle format for easy reuse and portability. Overall, the system is designed to be highly accurate and computationally efficient, making it practical for real-world deployment in online payment environments where reliable fraud detection is critical.</p>
'''

disclaimer_content = '''
<p>This application is designed for educational and informational purposes only. The predictions made by the machine learning models should not be relied upon as the sole basis for financial decisions or risk assessment. Users are advised to exercise caution and consult with financial professionals before taking any action. The developer is not responsible for any financial loss or damage resulting from the use of this tool. Prediction accuracy depends on the quality of input data and the assumptions used during model training.</p>
'''

@st.cache_resource
def downloadFiles():
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['LABELS']}", f'labels.pkl', quiet=False)
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['KM_C_FEATURES']}", f'km_C_features.pkl', quiet=False)
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['KM_D_FEATURES']}", f'km_D_features.pkl', quiet=False)
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['KM_V_FEATURES']}", f'km_V_features.pkl', quiet=False)
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['PCA_C_FEATURES']}", f'PCA_C_features.pkl', quiet=False)
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['PCA_D_FEATURES']}", f'PCA_D_features.pkl', quiet=False)
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['PCA_V_FEATURES']}", f'PCA_V_features.pkl', quiet=False)
  gdown.download(f"https://drive.google.com/uc?id={st.secrets['MODEL']}", f'model.pkl', quiet=False)

# User Interface System
def userDetails():
  st.markdown('<p class="section-title">👤 User Information</p>', unsafe_allow_html=True)
  


  col1, col2 = st.columns(2)
  with col1:
    name = st.text_input("Enter your name", key="name", placeholder="John Doe")
  with col2:
    age = st.number_input("Enter your age", min_value=1, key="age")
  
  col3, col4 = st.columns(2)
  with col3:
    phone_extension = st.text_input("Enter your phone extension", key="phone_extension", placeholder="+91")
  with col4:
    phone = st.text_input("Enter your phone number", key="phone", placeholder="1234567890")
  phone_no = f"{phone_extension} {phone}"

  st.markdown('<p class="section-title">🏠 Your Home Address</p>', unsafe_allow_html=True)

  col5, col6 = st.columns(2)
  with col5:
    street = st.text_input("Street", key="street", placeholder="123, Example Street")
  with col6:
    city = st.text_input("City", key="city", placeholder="Example City")
  col7, col8, col9 = st.columns(3)
  with col7:
    state = st.text_input("State", key="state", placeholder="Example State")
  with col8:
    country = st.text_input("Country", key="country", placeholder="Example Country")
  with col9:
    postal_code = st.number_input("Postal Code", min_value=10000, max_value=999999, step=1, value=123456, key="postal_code")
  my_address_data = {"Street": street, "City": city, "State": state, "Country": country, "Postal Code": postal_code}
  my_address = f"{street}, {city}, {state}, {country}, {postal_code}"

  if st.checkbox("Correspondence Address same as Home Address", key="corr_address"):
    corr_address_data = my_address_data
    corr_address = my_address
  else:
    st.markdown('<p class="section-title">📮Your Correspondence Address</p>', unsafe_allow_html=True)
    col10, col11 = st.columns(2)
    with col10:
      corr_street = st.text_input("Street", key="corr_street", placeholder="123, Example Street")
    with col11:
      corr_city = st.text_input("City", key="corr_city", placeholder="Example City")
    col12, col13, col14 = st.columns(3)
    with col12:
      corr_state = st.text_input("State", key="corr_state", placeholder="Example State")
    with col13:
      corr_country = st.text_input("Country", key="corr_country", placeholder="Example Country")
    with col14:
      corr_postal_code = st.number_input("Postal Code", min_value=10000, max_value=999999, step=1, value=123456, key="corr_postal_code")
    corr_address_data = {"Street": corr_street, "City": corr_city, "State": corr_state, "Country": corr_country, "Postal Code": corr_postal_code}
    corr_address = f"{corr_street}, {corr_city}, {corr_state}, {corr_country}, {corr_postal_code}"

  st.divider() ############################################################################################################
  return name, age, phone_no, my_address_data, my_address, corr_address_data, corr_address

def transactionDetails(labels, my_address_data, my_address):
    st.markdown('<p class="section-title">💳 Transaction Details</p>', unsafe_allow_html=True)
    

    # ---------------- TransactionDT ----------------
    START_DATE = '2017-12-01'
    StartDate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    col1, col2 = st.columns(2)
    with col1:
        TransactionDT_date = st.date_input("Transaction Date", key="TransactionDT_date")
    with col2:
        TransactionDT_time = st.time_input("Transaction Time", step=60, key="TransactionDT_time")
    TransactionDateTime = datetime.datetime.combine(TransactionDT_date, TransactionDT_time)
    TransactionDT = int((TransactionDateTime - StartDate).total_seconds())

    # ---------------- TransactionID & Amount ----------------
    cols = st.columns(2)
    with cols[0]:
        TransactionID = st.number_input("Transaction ID", key="TransactionID",
                                        min_value=1000000, max_value=999999999999,
                                        value=123456789012, placeholder="7-12 Digits")
    with cols[1]:
        TransactionAmt = np.log(st.number_input("Transaction Amount", min_value=0.1,
                                               value=1.0, step=1.0, key="TransactionAmt"))

    # ---------------- Transaction range ----------------
    min_last, max_last = st.slider("Your usual transaction amount range", min_value=1,
                                   max_value=100000, value=(10500, 70000), step=10, key="min_max_last")
    mean_last = np.log(np.mean([min_last, max_last]))
    std_last = mean_last / np.log(np.std([min_last, max_last]))

    # ---------------- Distance ----------------
    dist1 = st.slider("Distance from the home location (in km)", min_value=0, max_value=10000,
                      value=100, step=1, key="dist1")

    # ---------------- Weekday, Hour, Day ----------------
    _Weekdays, _Hours, _Days = TransactionDateTime.weekday(), TransactionDateTime.hour, TransactionDateTime.day

    # ---------------- ProductCD ----------------
    product_type = st.selectbox("What type of item is being purchased?",
                                ["Widgets", "Clothing", "Retail", "Healthcare", "Subscription","Others"],
                                key="ProductCD").lower()
    # Safe transform for ProductCD
    if product_type not in labels.classes_:
       ProductCD = -1  # unseen category
    else:
          ProductCD = labels.transform([product_type])[0]

    # ---------------- Purchaser & Recipient Email ----------------
    col3, col4 = st.columns(2)
    with col3:
        purchaser_email = st.text_input("Purchaser Email Address", key="puchaser_email",
                                        placeholder="user@example.example")
        P_emaildomain = purchaser_email.split('@')[-1] if '@' in purchaser_email else 'unknown'
        if P_emaildomain not in labels.classes_:
            P_emaildomain_encoded = -1
        else:
            P_emaildomain_encoded = labels.transform([P_emaildomain])[0]

    with col4:
        recipient_email = st.text_input("Recipient Email Address", key="recipient_email",
                                        placeholder="user@example.example")
        R_emaildomain = recipient_email.split('@')[-1] if '@' in recipient_email else 'unknown'
        if R_emaildomain not in labels.classes_:
            R_emaildomain_encoded = -1
        else:
            R_emaildomain_encoded = labels.transform([R_emaildomain])[0]

    # ---------------- Phone ----------------
    col5, col6 = st.columns(2)
    with col5:
        tx_phone_extension = st.text_input("Enter the phone extension", key="tx_phone_extension", placeholder="+91")
    with col6:
        tx_phone = st.text_input("Enter transaction phone number", key="tx_phone", placeholder="1234567890")
    tx_phone_no = f"{tx_phone_extension} {tx_phone}"

    # ---------------- Transaction Address ----------------
    if st.checkbox("Transaction Address same as Home Address", key="tx_address", value=True):
        tx_address_data = my_address_data
        tx_address = my_address
    else:
        st.markdown("##### From where did you conduct your transaction?")
        col7, col8 = st.columns(2)
        with col7:
            tx_street = st.text_input("Street", key="tx_street", placeholder="123, Example Street")
        with col8:
            tx_city = st.text_input("City", key="tx_city", placeholder="Example City")
        col9, col10, col11 = st.columns(3)
        with col9:
            tx_state = st.text_input("State", key="tx_state", placeholder="Example State")
        with col10:
            tx_country = st.text_input("Country", key="tx_country", placeholder="Example Country")
        with col11:
            tx_postal_code = st.number_input("Postal Code", min_value=10000, max_value=999999,
                                             step=1, value=123456, key="tx_postal_code")
        tx_address_data = {"Street": tx_street, "City": tx_city, "State": tx_state,
                           "Country": tx_country, "Postal Code": tx_postal_code}
        tx_address = f"{tx_street}, {tx_city}, {tx_state}, {tx_country}, {tx_postal_code}"

    # ---------------- Address values ----------------
    addr1 = float(str(tx_address_data['Postal Code'])[3:])
    addr2 = float(str(tx_address_data['Postal Code'])[:3])
    first_value_addr1 = float(str(tx_address_data['Postal Code'])[0])

    st.divider()

    return (TransactionDT, TransactionID, TransactionAmt, mean_last, min_last, max_last, std_last,
            dist1, _Weekdays, _Days, _Hours, ProductCD, purchaser_email, P_emaildomain_encoded,
            recipient_email, R_emaildomain_encoded, tx_phone_no, tx_address_data, tx_address,
            addr1, addr2, first_value_addr1)


def cardDetails(labels, TransactionAmt):
    st.markdown('<p class="section-title">💳 Provide the Card Details</p>', unsafe_allow_html=True)
   

    # --------------------------
    # Card ID (card1, card2, card3, card5)
    # --------------------------
    cols = st.columns(5)
    with cols[0]:
        st.text_input("Card ID", key="Card_ID", label_visibility="hidden", value="Card ID", disabled=True)
    with cols[1]:
        card1 = st.text_input("Card ID 1", key="card1", label_visibility="hidden", value="1000", placeholder="1000")
        first_value_card1 = float(card1[0]) if card1 else 0
    with cols[2]:
        card2 = st.text_input("Card ID 2", key="card2", label_visibility="hidden", value="5552", placeholder="5552")
    with cols[3]:
        card3 = st.text_input("Card ID 3", key="card3", label_visibility="hidden", value="1835", placeholder="1835")
    with cols[4]:
        card5 = st.text_input("Card ID 5", key="card5", label_visibility="hidden", value="2246", placeholder="2246")

    Card_ID = f"{card1} {card2} {card3} {card5}"

    # --------------------------
    # Card holder name
    # --------------------------
    card_holder_name = st.text_input("Card Holder Name", key="card_holder_name", placeholder="John Doe")

    # --------------------------
    # Transaction amount: mean ratios
    # --------------------------
    max_limit = st.number_input(
        "Maximum transaction limit of this card",
        min_value=1, value=100000, step=10, key="max_limit"
    )

    min_last, max_last = st.slider(
        "Your usual transaction amount range through this card",
        min_value=1,
        max_value=max_limit,
        value=(1, max_limit),
        step=1,
        key="min_max_last_card"
    )

    mean_val = np.mean([min_last, max_last])
    std_last = np.log(mean_val) / np.log(np.std([min_last, max_last])) if np.std([min_last, max_last]) != 0 else 1

    TransactionAmt_to_mean_card_id = np.exp(TransactionAmt) - mean_val
    TransactionAmt_to_mean_card1 = np.exp(TransactionAmt) / mean_val
    TransactionAmt_to_mean_card4 = abs((np.exp(TransactionAmt) / mean_val) - std_last)

    # --------------------------
    # card4 & card6 Label Encoding (Brand, Usage)
    # --------------------------
    payment_method = st.selectbox(
    "Have you made the payment using a card?",
    ['Select One', 'Yes', 'No'],
    key="payment_method"
)
    card4, card6 = -999, -999
    
    brand_map = {
    "visa": 0,
    "mastercard": 1,
    "american express": 2,
    "discover": 3
}
    usage_map = {
    "credit": 0,
    "debit": 1,
    "debit or credit": 2,
    "charge card": 3
}

    if payment_method == 'Yes':
    
      card_brand = st.selectbox(
        "Brand of the card",
        ['Visa', 'Mastercard', 'American Express', 'Discover'],
        key="card4"
    ).lower()

      card_usage = st.selectbox(
        "Usage of the card",
        ['Credit', 'Debit', 'Debit or Credit', 'Charge Card'],
        key="card6"
    ).lower()

        # ✔ Correct use of LabelEncoder
      try:
           card4 = brand_map.get(card_brand, -999)
           card6 = usage_map.get(card_usage, -999)
      except ValueError:
            st.error("Selected card brand or usage not found in LabelEncoder classes.")
            st.write("Available classes:", labels.classes_)

    card_data = {
        "Card1": card1,
        "Card2": card2,
        "Card3": card3,
        "Card4": card4,
        "Card5": card5,
        "Card6": card6
    }

    st.divider()

    # Return everything
    return (
        card_data,
        Card_ID,
        first_value_card1,
        card_holder_name,
        TransactionAmt_to_mean_card_id,
        TransactionAmt_to_mean_card1,
        TransactionAmt_to_mean_card4
    )

def billingDetails(purchaser_email, card_holder_name, name, phone_no, tx_phone_no, my_address, tx_address, country, tx_country):
  st.markdown('<p class="section-title">🧾 Billing Details</p>', unsafe_allow_html=True)
  


  # For M1, M4, M5, M6, M7 
  # M1 -> Billing Address == Shipping Address
  M1 = st.selectbox("Is the billing address the same as the shipping address?", ['Select One', 'Yes', 'No'], key="M1")
  M1 = 1 if M1 == 'Yes' else 2 if M1 == 'No' else -999

  # M2 -> Email Address (at T/X) == Email Address (Owner)
  email_tx = st.text_input("Email Address at the Transaction Receipt", key="email_tx", placeholder="user@example.example")
  M2 = -999 if email_tx == '' or purchaser_email == '' else 1 if email_tx == purchaser_email else 2

  # M3 -> Card Holder Name == Name of the Owner
  M3 = -999 if card_holder_name == '' or name == '' else 1 if card_holder_name == name else 2

  # M4 -> T/X from same device as last transaction
  M4 = st.selectbox("Have you done the transaction from the same device as last time?", ['Select One', 'Yes', 'No'], key="M4_device")
  if M4 == 'Yes':
    M4 = st.selectbox("Have you done the transaction from a nearby home location?", ['Select One', 'Yes', 'No'], key="M4_home")
    M4 = 4 if M4 == 'Yes' else 5 if M4 == 'No' else -999
  elif M4 == 'No':
    M4 = 3
  else:
    M4 = -999

  # M5 -> Phone Number (at T/X) == Phone Number of the Owner
  M5 = -999 if phone_no == ' ' or len(phone_no) < 14 or tx_phone_no == ' ' or len(tx_phone_no) < 14 else 1 if phone_no == tx_phone_no else 2

  # M6 -> Address (at T/X) == Address of the Owner
  M6 = -999 if my_address == ", , , , 123456" or tx_address == ", , , , 123456" else 1 if my_address == tx_address else 2

  # M7 -> Country (at T/X) == Country of the Owner
  M7 = -999 if country == '' or tx_country == '' else 1 if country == tx_country else 2

  M_data = {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5, "M6": M6, "M7": M7}
  st.divider() ############################################################################################################
  return M_data

def behavioralDetails(TransactionAmt, min_last, max_last, _Hours, my_address_data, tx_address_data):
  # For V1, V12, V14, V35, V41, V65, V69, V75, V88, V94, V241
  st.markdown('<p class="section-title">📈 Transactional Usage Details</p>', unsafe_allow_html=True)
  


  cols = st.columns(3)
  with cols[0]:
    V1 = st.selectbox("Have you recently changed your account information?", ['Select One', 'Yes', 'No'], key="V1")
    V1 = 1 if V1 == 'No' else 0
  with cols[1]:
    V14 = st.selectbox("Have you encountered any other fraud scenarios?", ['Select One', 'Yes', 'No'], key="V14")
    V14 = 1 if V14 == 'No' else 0
  with cols[2]:
    V88 = st.selectbox("Is this transaction done from your own device?", ['Select One', 'Yes', 'No'], key="V88")
    V88 = 1 if V88 == 'Yes' else 0

  # Is the transaction amount significantly higher than the user's maximum?
  V41 = 1 if (np.exp(TransactionAmt) > max_last) else 0

  # Was this transaction made within normal business hours?
  V65 = 1 if (8 <= _Hours <= 18) else 0

  # Is the user's transaction history within the expected range for their account?
  V241 = 1 if (min_last <= np.exp(TransactionAmt) <= max_last) else 0

  V94 = st.selectbox("What is the nature of your purchase history?", ['Select One', 'Mostly small transactions', 'A mix of small and large transactions', 'Mostly large transactions'], key="V94")
  V94 = (0 if V94 == 'Mostly small transactions' else 1 if V94 == 'A mix of small and large transactions' else 2) / 2

  # 0 Exactly the same address, 1 Same city but different street, 2 Same state but different city, 3 Same country but different state
  V12 = abs(0 if my_address_data == tx_address_data else 1 if my_address_data['City'] == tx_address_data['City'] else 2 if my_address_data['State'] == tx_address_data['State'] else 3 if my_address_data['Country'] == tx_address_data['Country'] else -1) / 3

  V35 = st.selectbox("How long has your account been active?", ['Select One', 'Less than 1 month', '1-6 months', '6-12 months', 'More than a year'], key="V35")
  V35 = (0 if V35 == 'More than a year' else 1 if V35 == '6-12 months' else 2 if V35 == '1-6 months' else 3) / 3

  V75 = st.selectbox("How often do you change your password?", ['Select One', 'Less than once a year', 'Once a year', 'Every 6 months', 'Every 3 months', 'Monthly'], key="V75")
  V75 = (0 if V75 == 'Less than once a year' else 1 if V75 == 'Once a year' else 2 if V75 == 'Every 6 months' else 3 if V75 == 'Every 3 months' else 4) / 4

  V69 = st.selectbox("How frequently do you make transactions online?", ['Select One', 'Daily', 'Weekly', 'Bi-weekly', 'Monthly', 'Every few months', 'Rarely'], key="V69")
  V69 = (0 if V69 == 'Daily' else 1 if V69 == 'Weekly' else 2 if V69 == 'Bi-weekly' else 3 if V69 == 'Monthly' else 4 if V69 == 'Every few months' else 5) / 5

  V_data = {"V1": V1, "V12": V12, "V14": V14, "V35": V35, "V41": V41, "V65": V65, "V69": V69, "V75": V75, "V88": V88, "V94": V94, "V241": V241}

  # For C5, C6, C7, C9, C12, C14
  st.divider() ############################################################################################################
  time_tx = st.expander("Transaction Details", expanded=True)
  with time_tx:
    st.markdown("<div style='text-align: justify; margin: 1rem;'>\
                <p style='margin-bottom: 5px;'>Ranges of the transaction amount:</p>\
                <b>Small Transactions:</b> 1 - 1000<br />\
                <b>Medium Transactions:</b> 1000 - 20000<br />\
                <b>Large Transactions:</b> 20000 - 100000<br />\
                <b>Very Large Transactions:</b> 100000 - more than 100000<br />\
                </div>", unsafe_allow_html=True)

  st.markdown('<p class="section-title">🧠 Transaction Behavior Details</p>', unsafe_allow_html=True)
  C7 = st.slider("How frequently do small transactions occur per day?", min_value=0, max_value=100, value=5, step=1, key="C7") / 2256
  C12 = st.slider("How frequently do large transactions occur per day?", min_value=0, max_value=10, value=2, step=1, key="C12") / 3188

  C6 = st.slider("How frequently do small transactions occur per week?", min_value=0, max_value=1000, value=30, step=1, key="C6") / 2252
  C14 = st.slider("How frequently do large transactions occur per week?", min_value=0, max_value=100, value=13, step=1, key="C14") / 1429

  C5 = st.slider("How frequently do small transactions occur per month?", min_value=0, max_value=10000, value=200, step=1, key="C5") / 376
  C9 = st.slider("How frequently do large transactions occur per month?", min_value=0, max_value=1000, value=30, step=1, key="C9") / 572

  C_data = {"C5": C5, "C6": C6, "C7": C7, "C9": C9, "C12": C12, "C14": C14}

  # For D2, D3, D4, D5, D11
  st.divider() ############################################################################################################
  st.markdown('<p class="section-title">⏱️Transaction Time behavioral details</p>', unsafe_allow_html=True)

  D2 = st.slider("How many days between small transactions?", min_value=0, max_value=100, value=2, step=1, key="D2") / 641
  D11 = st.slider("How many days between medium transactions?", min_value=0, max_value=365, value=15, step=1, key="D11") / 936
  D3 = st.slider("How many days between large transactions?", min_value=0, max_value=730, value=40, step=1, key="D3") / 1076
  D5 = st.slider("How many days between very large transactions?", min_value=0, max_value=730, value=80, step=1, key="D5") / 1088
  D4 = st.slider("How many days between exceptional transactions?", min_value=0, max_value=1095, value=200, step=1, key="D4") / 1213

  D_data = {"D2": D2, "D3": D3, "D4": D4, "D5": D5, "D11": D11}
  st.divider() ############################################################################################################
  return V_data, C_data, D_data

def deviceInfo(labels):
    st.markdown('<p class="section-title">📱 Device Details</p>', unsafe_allow_html=True)
   
    # For DeviceType
    DeviceType = st.selectbox("Device Type", ['Select One', 'Desktop', 'Mobile'], key="DeviceType")
    DeviceType = 1 if DeviceType == 'Desktop' else 2 if DeviceType == 'Mobile' else -999

    # For DeviceInfo
    device_choice = st.selectbox("Device Info", list(labels.classes_), key="DeviceInfo")

    # Safe transform: handle unseen label
    if device_choice not in labels.classes_:
        DeviceInfo = -1
    else:
        DeviceInfo = labels.transform([device_choice])[0]

    device_data = {"DeviceType": DeviceType, "DeviceInfo": DeviceInfo}

    st.divider()
    
    return device_data


# Model Prediction System
def preprocessing(data):
  # Preprocessing of C features 
  with open('PCA_C_features.pkl', 'rb') as f:
    PCA_C_features = pickle.load(f)
  C_data = data['Transactional Usage Data']
  C_data = np.array([C_data['C5'], C_data['C9'], C_data['C14'], C_data['C7'], C_data['C12'], C_data['C6']]).reshape(1, -1)
  PCA_C_data = PCA_C_features.transform(C_data)
  
  with open('km_C_features.pkl', 'rb') as f:
    km_C_features = pickle.load(f)
  clusters_C = km_C_features.predict(PCA_C_data)
  clusters_C = clusters_C[0]

  # Preprocessing of D features 
  with open('PCA_D_features.pkl', 'rb') as f:
    PCA_D_features = pickle.load(f)
  D_data = data['Transaction Time Behavioral Data']
  D_data = np.array([D_data['D5'], D_data['D4'], D_data['D3'], D_data['D11'], D_data['D2']]).reshape(1, -1)
  PCA_D_data = PCA_D_features.transform(D_data)

  with open('km_D_features.pkl', 'rb') as f:
    km_D_features = pickle.load(f)
  clusters_D = km_D_features.predict(PCA_D_data)
  clusters_D = clusters_D[0]

  # Preprocessing of V features 
  with open('PCA_V_features.pkl', 'rb') as f:
    PCA_V_features = pickle.load(f)
  V_data = data['Behavioral Data']
  V_data = np.array([V_data['V88'], V_data['V14'], V_data['V1'], V_data['V65'], V_data['V41'], V_data['V94'], V_data['V35'], V_data['V12'], V_data['V241'], V_data['V69'], V_data['V75']]).reshape(1, -1)
  PCA_V_data = PCA_V_features.transform(V_data)

  with open('km_V_features.pkl', 'rb') as f:
    km_V_features = pickle.load(f)
  clusters_V = km_V_features.predict(PCA_V_data)
  clusters_V = clusters_V[0]

  # For count_cluster
  count_cluster = clusters_C + clusters_D + clusters_V

  data['PCA_C_data'] = PCA_C_data
  data['PCA_D_data'] = PCA_D_data
  data['PCA_V_data'] = PCA_V_data

  data['clusters_C'] = clusters_C
  data['clusters_D'] = clusters_D
  data['clusters_V'] = clusters_V

  data['count_cluster'] = count_cluster

  return data

def predict(data):
  prepared_data = np.array([
    int(data['Transaction ID']),
    float(data['Transaction Amount']),
    int(data['ProductCD']) - 66,
    int(data['Card Data']['Card1']),
    float(data['Card Data']['Card2']),
    data['Card Data']['Card4'],
    data['Card Data']['Card6'],
    float(data['Address 1']),
    float(data['Address 2']),
    float(data['Distance']),
    data['Purchaser Email Domain'],
    data['Recipient Email Domain'],
    data['Billing Data']['M1'],
    data['Billing Data']['M4'],
    data['Billing Data']['M5'],
    data['Billing Data']['M6'],
    data['Billing Data']['M7'],
    float(data['PCA_V_data'][0][0]),
    float(data['PCA_V_data'][0][1]),
    float(data['PCA_V_data'][0][2]),
    float(data['PCA_V_data'][0][3]),
    float(data['PCA_V_data'][0][4]),
    float(data['PCA_V_data'][0][5]),
    float(data['PCA_V_data'][0][6]),
    float(data['PCA_V_data'][0][7]),
    float(data['PCA_V_data'][0][8]),
    float(data['PCA_V_data'][0][9]),
    int(data['clusters_V']),
    float(data['PCA_C_data'][0][0]),
    float(data['PCA_C_data'][0][1]),
    float(data['PCA_C_data'][0][2]),
    int(data['clusters_C']),
    float(data['Device Data']['DeviceType']),
    float(data['Device Data']['DeviceInfo']),
    data['Weekdays'],
    data['Hours'],
    data['Days'],
    float(data['Mean Transaction Amount']),
    float(data['Minimum Transaction Amount']),
    float(data['Maximum Transaction Amount']),
    float(data['Standard Deviation Transaction Amount']),
    float(data['PCA_D_data'][0][0]),
    float(data['PCA_D_data'][0][1]),
    float(data['PCA_D_data'][0][2]),
    float(data['PCA_D_data'][0][3]),
    float(data['PCA_D_data'][0][4]),
    int(data['clusters_D']),
    int(data['count_cluster']),
    float(data['First Value Address 1']),
    float(data['Transaction Amount with Card ID']),
    float(data['Transaction Amount with Card1']),
    float(data['Transaction Amount with Card4']),
    float(data['First Value Card1'])
  ]).reshape(1, -1)

  with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
  prediction = model.predict_proba(prepared_data)
  return prediction

def app():
    st.write("This is a simple web app to predict whether a transaction is fraudulent or not.")  
    st.write("Please provide the necessary details to classify the transaction.")
  
    # Load labels from file
    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    # ----- User Inputs -----
    name, age, phone_no, my_address_data, my_address, corr_address_data, corr_address = userDetails()

    # ----- Transaction Inputs -----
    (TransactionDT, TransactionID, TransactionAmt, mean_last, min_last, max_last, std_last, dist1, 
     _Weekdays, _Days, _Hours, ProductCD, purchaser_email, P_emaildomain, recipient_email, 
     R_emaildomain, tx_phone_no, tx_address_data, tx_address, addr1, addr2, first_value_addr1) = \
        transactionDetails(labels, my_address_data, my_address)

    # ----- Card Details -----
    (card_data, Card_ID, first_value_card1, card_holder_name, TransactionAmt_to_mean_card_id,
     TransactionAmt_to_mean_card1, TransactionAmt_to_mean_card4) = cardDetails(labels, TransactionAmt)

    # ----- Billing -----
    M_data = billingDetails(
        purchaser_email, card_holder_name, name, phone_no, tx_phone_no, 
        my_address, tx_address, my_address_data['Country'], tx_address_data['Country']
    )

    # ----- Behavior -----
    V_data, C_data, D_data = behavioralDetails(
        TransactionAmt, min_last, max_last, _Hours, my_address_data, tx_address_data
    )

    # ----- Device Info -----
    device_data = deviceInfo(labels)

    # ---------------------------
    #        PREDICT BUTTON
    # ---------------------------
    if st.button("Predict"):

        # Everything stored in dictionary
        data = {
            "Name": name, "Age": age, "Phone Number": phone_no,
            "My Address Data": my_address_data, "Address": my_address, 
            "Correspondence Address Data": corr_address_data, "Correspondence Address": corr_address, 
            "Transaction Date": TransactionDT, "Transaction ID": TransactionID, 
            "Transaction Amount": TransactionAmt, "Mean Transaction Amount": mean_last, 
            "Minimum Transaction Amount": min_last, "Maximum Transaction Amount": max_last, 
            "Standard Deviation Transaction Amount": std_last, "Distance": dist1, "Weekdays": _Weekdays, 
            "Days": _Days, "Hours": _Hours, "ProductCD": ProductCD, "Purchaser Email": purchaser_email, 
            "Purchaser Email Domain": P_emaildomain, "Recipient Email": recipient_email, 
            "Recipient Email Domain": R_emaildomain, "Transaction Phone Number": tx_phone_no, 
            "Transaction Address Data": tx_address_data, "Transaction Address": tx_address, 
            "Address 1": addr1, "Address 2": addr2, "First Value Address 1": first_value_addr1,
            "Card Data": card_data, "Card ID": Card_ID, "First Value Card1": first_value_card1, 
            "Card Holder Name": card_holder_name, "Transaction Amount with Card ID": TransactionAmt_to_mean_card_id, 
            "Transaction Amount with Card1": TransactionAmt_to_mean_card1, 
            "Transaction Amount with Card4": TransactionAmt_to_mean_card4, "Billing Data": M_data, 
            "Behavioral Data": V_data, "Transactional Usage Data": C_data, 
            "Transaction Time Behavioral Data": D_data, "Device Data": device_data
        }

        # Progress UI
        st.toast("Pre-processing has started...", icon="⏳")
        with st.spinner('Model is processing...'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)

        # -----------------------------
        #    🔥 NEW PREPROCESSING
        # -----------------------------
        try:
            data = preprocess_raw_input(data)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            return

        st.toast("Pre-processing is complete...", icon="✅")
        time.sleep(1)

        # UI container for results
        results = st.container()
        with results:
            st.subheader("Results")
            st.toast("Model is working now...", icon="⏳")

            with st.spinner('Model is processing...'):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)

            st.toast("Prediction is complete...", icon="✅")

            try:
                # -----------------------------
                #    🔥 NEW PREDICT FUNCTION
                # -----------------------------
                pred, proba = predict_fraud(data)

                st.write(f"Hello {name}!")
                st.write("Based on the machine learning model, the risk of this transaction being fraudulent is:")

                if pred == 1:
                    st.error("**HIGH**", icon="🚫")
                    st.toast("The transaction is classified as fraudulent.", icon="🚫")
                else:
                    st.success("**LOW**", icon="✅")
                    st.toast("The transaction is classified as non-fraudulent.", icon="🎉")

                # Probabilities details
                details = st.expander("More Details", expanded=False)
                with details:
                    st.write("Fraud Probability:")
                    st.error(f"**{proba:.2%}**")

                    st.write("Non-Fraud Probability:")
                    st.success(f"**{1 - proba:.2%}**")

            except Exception as e:
                st.error(f"Prediction Error: {e}", icon="🚨")
                st.toast("Oops... Something went wrong. Please try again.", icon="🚨")

    # Back button
    if st.button("Go Back"):
        st.toast("Thanks for using the application!", icon="👋")
        time.sleep(2)
        st.session_state.button_clicked = False
        st.rerun()

if __name__ == '__main__':
  # st.set_page_config(page_title="FraudLense App", page_icon="https://github.com/user-attachments/assets/662c2283-a6c9-471a-a2cd-b4173c64cd54")
  
  if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

  title = st.container()
  with title:
    col1, col2 = st.columns([1, 7])
    with col1:
      st.image('https://github.com/user-attachments/assets/662c2283-a6c9-471a-a2cd-b4173c64cd54', width=100)
    with col2:
      st.title("FraudLense")
  
  if not st.session_state.button_clicked:
    st.toast("Welcome to the FraudLense App!", icon="🎉")
    time.sleep(2)
    st.toast("Please wait while we load the application...", icon="⏳")
    downloadFiles()
    st.toast("Files downloaded...", icon="✅")
    time.sleep(2)
    st.toast("Application is ready to use...", icon="🚀")

  if st.session_state.button_clicked:
    st.success("You have agreed to the terms. You can now proceed with using the application.", icon="✅")
    app()
  else:
    abstract = st.expander(label="Abstract", expanded=False)
    with abstract:
      st.write('**Read the full article** [ **here!** ](https://docs.google.com/document/d/1B4aMbTaAT1tb9Ls02syBxZToGUz_BFcRp6xdmEHHiUk/edit?usp=sharing)')
      st.markdown(f'<div style="text-align: justify; margin: 1rem;">\
                  <h3 style="text-align: center; font-weight: 900;">Detecting Fraud with Machine Learning</h3>\
                  {abstract_content}\
                  </div>', unsafe_allow_html=True)
    
    youtube_video = st.expander("Watch the Video", expanded=False)
    with youtube_video:
      st.video(f"https://youtu.be/{st.secrets['YOUTUBE_VIDEO_ID']}")

    disclaimer = st.expander("Disclaimer", expanded=True)
    with disclaimer:
      st.markdown(f'<div style="text-align: justify; margin: 1rem;">{disclaimer_content}</div>', unsafe_allow_html=True)

    cols = st.columns(7)
    with cols[0]:
      if st.button('Agree'):
        st.session_state.button_clicked = True
        st.rerun()
    with cols[6]:
      st.button('Disagree')
      st.toast("You must agree to the terms to proceed with the application.", icon="🚨")
    st.markdown("""
<div style="
    padding: 20px;
    background-color:#ffe5e5;
    color: #212529;
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    font-size: 16px;
    font-family: Arial, sans-serif;
">
💡 <b>Need Help?</b> For questions or issues, reach out to support. 
<a href='https://www.linkedin.com/in/harikesh-tripathi-7841a0181/' target='_blank' style='color: #0d6efd; font-weight: bold; text-decoration: none;'>Click here!</a>
</div>
""", unsafe_allow_html=True)


