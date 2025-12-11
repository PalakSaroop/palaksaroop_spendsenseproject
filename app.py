import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

spending_model = joblib.load("spendsense_spending_predictor.pkl")
overspend_model = joblib.load("spendsense_overspend_classifier.pkl")

label_encoders = joblib.load("label_encoders.pkl")
scaler_reg = joblib.load("scaler_reg.pkl")
scaler_cls = joblib.load("scaler_cls.pkl")

# PAGE CONFIG

st.set_page_config(
    page_title="SpendSense Dashboard",
    page_icon="💰",
    layout="wide"
)

# DARK MODE PREMIUM CSS

st.markdown("""
<style>

body {
    background-color: #0D0D0D !important;
    color: #EAECEE !important;
    font-family: 'Inter', sans-serif;
}

/* Title */
.big-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00E6FF, #00FFA3);
    -webkit-background-clip: text;
    color: transparent;
    margin-top: 15px;
}

.sub-text {
    text-align: center;
    font-size: 18px;
    color: #A6ACAF;
    margin-bottom: 35px;
}

/* Neon Card */
.card {
    background-color: #111418;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #1F2A30;
    box-shadow: 0px 0px 18px rgba(0, 255, 180, 0.12);
    margin-bottom: 30px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00E6FF, #00FFA3) !important;
    color: black !important;
    padding: 12px 25px;
    border-radius: 10px !important;
    font-size: 17px !important;
    font-weight: 700;
    border: none !important;
    transition: 0.2s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 12px #00FFA3;
}

/* Success Box */
.success-box {
    background-color: #062E1D;
    padding: 15px;
    border-radius: 10px;
    color: #2ECC71;
    font-weight: 700;
    border: 1px solid #27AE60;
    box-shadow: 0px 0px 12px rgba(0, 255, 153, 0.15);
}

/* Error Box */
.error-box {
    background-color: #3A0E0E;
    padding: 15px;
    border-radius: 10px;
    color: #FF5C5C;
    font-weight: 700;
    border: 1px solid #D98880;
    box-shadow: 0px 0px 12px rgba(255, 0, 0, 0.15);
}

/* Input Fields */
input, select, textarea {
    background-color: #1B1F23 !important;
    color: #EAECEE !important;
    border-radius: 8px !important;
    padding: 12px !important;
    border: 1px solid #2C3338 !important;
}

</style>
""", unsafe_allow_html=True)

# TITLE

st.markdown("<div class='big-title'>SpendSense </div>", unsafe_allow_html=True)

# INPUT SECTION

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    categories = label_encoders["Category"].classes_
    category = st.selectbox("Expense Category", categories)

    amount = st.number_input("Amount Spent ($)", min_value=0.0, step=1.0)

with col2:
    payment_methods = label_encoders["Payment Method"].classes_
    payment_method = st.selectbox("Payment Method", payment_methods)

    date = st.date_input("Transaction Date", datetime.today())

month = date.month
year = date.year

category_encoded = label_encoders["Category"].transform([category])[0]
payment_encoded = label_encoders["Payment Method"].transform([payment_method])[0]

st.markdown("</div>", unsafe_allow_html=True)

# PREDICTIONS

colA, colB = st.columns(2)

# FUTURE SPENDING

with colA:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔮 Predict Future Spending")

    if st.button("Predict Spending"):
        if amount <= 0:
            st.warning("⚠ Please enter a valid amount.")
        else:
            reg_input = pd.DataFrame([{
                "Category": category_encoded,
                "Payment Method": payment_encoded,
                "Month": month,
                "Year": year
            }])

            reg_scaled = scaler_reg.transform(reg_input)
            predicted_amount = spending_model.predict(reg_scaled)[0]

            st.markdown(
                f"<div class='success-box'>💵 Estimated Future Spend: <b>${predicted_amount:.2f}</b></div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# OVERSPENDING RISK

with colB:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ⚠ Overspending Risk Detector")

    if st.button("Check Risk"):
        if amount <= 0:
            st.warning("⚠ Please enter a valid amount.")
        else:
            cls_input = pd.DataFrame([{
                "Month": month,
                "Total Spent": amount
            }])

            cls_scaled = scaler_cls.transform(cls_input)
            risk = overspend_model.predict(cls_scaled)[0]

            if risk == 1:
                st.markdown(
                    "<div class='error-box'>🚨 High Risk of Overspending Detected!</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='success-box'>✔ Your Spending Is Safe</div>",
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)


# FOOTER

st.markdown("---")
st.info("""
### 💡 SpendSense – Premium AI Spending Prediction  
Built with Machine Learning, Clean Architecture, and a modern Dark Mode FinTech UI.
""")
