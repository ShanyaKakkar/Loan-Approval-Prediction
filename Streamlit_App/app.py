import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    h1 {
        color: #1f3a5f;
        text-align: center;
        padding-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: #5a6b7d;
        font-size: 16px;
        margin-bottom: 30px;
    }
    .section-header {
        color: #1f3a5f;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and columns
model = joblib.load('../Model/loan_approval_model.pkl')
model_columns = joblib.load('../Model/model_columns.pkl')

# Header
st.markdown("<h1>🏦 Loan Approval Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered loan eligibility checker — fill in the details below</p>", unsafe_allow_html=True)

# Personal Details Section
st.markdown("<div class='section-header'>👤 Personal Details</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
with col2:
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# Financial Details Section
st.markdown("<div class='section-header'>💰 Financial Details</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    applicant_income = st.number_input("Applicant Income (₹/month)", min_value=0, value=5000, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=0, value=150, step=10)
with col4:
    coapplicant_income = st.number_input("Coapplicant Income (₹/month)", min_value=0, value=0, step=500)
    loan_amount_term = st.selectbox("Loan Term (days)", [360, 180, 240, 120, 84, 60, 36, 12], index=0)

# Property & Credit Section
st.markdown("<div class='section-header'>🏠 Property & Credit</div>", unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    credit_history = st.selectbox("Credit History", ["1 (Good)", "0 (Poor)"])
with col6:
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Loan Status"):
    dependents_val = 3 if dependents == "3+" else int(dependents)
    total_income = applicant_income + coapplicant_income
    total_income_log = np.log(total_income) if total_income > 0 else 0
    loan_amount_log = np.log(loan_amount) if loan_amount > 0 else 0
    credit_val = 1 if "1" in credit_history else 0

    input_dict = {
        'Dependents': dependents_val,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_val,
        'Total_Income': total_income,
        'Total_Income_log': total_income_log,
        'LoanAmount_log': loan_amount_log,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Married_Yes': 1 if married == "Yes" else 0,
        'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        st.success(f"✅ Loan Approved! Confidence: {probability*100:.1f}%")
        st.progress(float(probability))
    else:
        st.error(f"❌ Loan Rejected. Confidence: {(1-probability)*100:.1f}%")
        st.progress(float(1-probability))

st.markdown("<br><hr><p style='text-align:center; color:grey; font-size:12px;'>AIML Summer Internship 2026 Capstone Project | IIHMF, MNNIT Allahabad</p>", unsafe_allow_html=True)