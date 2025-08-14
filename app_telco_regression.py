import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Telco CLV (Total Charges) Regression Agent", layout="wide")

ARTIFACT_DIR = Path("artifacts_telco_regression")
MODEL_PATH = ARTIFACT_DIR / "final_model.pkl"
META_PATH = ARTIFACT_DIR / "meta.json"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()
q_hat = meta["q_hat"]

st.title("Telco CLV Prediction (Total Charges)")
st.markdown("Estimate Customer Lifetime Value using a regression model trained on a public Telco dataset. Includes 95% prediction intervals via split conformal.")

with st.sidebar:
    st.header("Inputs")
    tenure = st.number_input("Tenure in Months", min_value=0.0, max_value=120.0, value=24.0, step=1.0)
    monthly = st.number_input("Monthly Charge", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)

    def pick(label, options, default_idx=0):
        return st.selectbox(label, options, index=default_idx) if options else None

    gender = pick("Gender", ["Male", "Female"])
    senior = pick("Senior Citizen", ["No", "Yes"])
    partner = pick("Partner", ["No", "Yes"])
    dependents = pick("Dependents", ["No", "Yes"])
    phone = pick("Phone Service", ["Yes", "No"])
    multiline = pick("Multiple Lines", ["No", "Yes"])
    internet = pick("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"])
    online_sec = pick("Online Security", ["No", "Yes"])
    online_bkp = pick("Online Backup", ["No", "Yes"])
    device_plan = pick("Device Protection Plan", ["No", "Yes"])
    tech_support = pick("Premium Tech Support", ["No", "Yes"])
    stream_tv = pick("Streaming TV", ["No", "Yes"])
    stream_movies = pick("Streaming Movies", ["No", "Yes"])
    contract = pick("Contract", ["Month-to-Month", "One Year", "Two Year"])
    paperless = pick("Paperless Billing", ["Yes", "No"])
    pay_method = pick("Payment Method", ["Credit Card", "Bank Transfer", "Mailed Check", "Electronic Check"])

if st.button("Predict CLV"):
    rec = {
        "Tenure in Months": tenure,
        "Monthly Charge": monthly,
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Phone Service": phone,
        "Multiple Lines": multiline,
        "Internet Type": internet,
        "Online Security": online_sec,
        "Online Backup": online_bkp,
        "Device Protection Plan": device_plan,
        "Premium Tech Support": tech_support,
        "Streaming TV": stream_tv,
        "Streaming Movies": stream_movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": pay_method
    }
    X_new = pd.DataFrame([rec])

    y_pred = model.predict(X_new)[0]
    lower = y_pred - q_hat
    upper = y_pred + q_hat

    st.subheader("Result")
    st.metric("Predicted Total Charges (USD)", f"${y_pred:,.2f}")
    st.write(f"95% Prediction Interval: ${lower:,.2f} — ${upper:,.2f}")
    st.caption("Note: Interval uses split conformal prediction.")

st.divider()
st.subheader("Model Performance (Test Summary)")
st.write(pd.DataFrame([meta["metrics_test"]]))

st.divider()
st.subheader("About")
st.markdown(f"""
- Final Model: {meta['best_model_name']}
- Target: Total Charges (USD) - CLV proxy
- Use Case: Retention budgeting, discount caps, upsell targeting
- Disclaimer: Educational demonstration, not financial advice.
""")
