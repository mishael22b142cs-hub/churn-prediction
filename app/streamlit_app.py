"""
Streamlit app for the Churn Prediction project.

Loads the best-performing model (saved by notebooks/05_evaluate.ipynb) along
with the fitted StandardScaler and the training feature column list (both
saved by notebooks/03_preprocessing.ipynb), collects a single customer's
attributes from the sidebar, encodes them exactly the way training data was
encoded, and displays a churn prediction with probability.
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

MULTI_CAT_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

MULTI_CAT_OPTIONS = {
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, scaler, feature_columns


def preprocess_input(raw: dict, scaler, feature_columns) -> pd.DataFrame:
    """Encode a single raw customer record the same way training data was encoded."""
    df = pd.DataFrame([raw])

    # Binary label encoding
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping)

    # SeniorCitizen is already 0/1 from the sidebar (Yes/No selectbox mapped below)

    # One-hot encode multi-category columns, matching pd.get_dummies(drop_first=True)
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)

    # Align to the exact training feature columns; any dummy category not
    # produced for this single row (because it's the dropped reference level,
    # or simply not among this customer's choices) is filled with 0.
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric columns with the scaler fitted on the training set
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    return df


def main():
    st.title("📉 Customer Churn Prediction")
    st.write(
        "Enter a customer's attributes in the sidebar, then click **Predict** "
        "to estimate their probability of churning."
    )

    try:
        model, scaler, feature_columns = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run notebooks 01 through 05 in order "
            "first — they generate `models/best_model.pkl`, `models/scaler.pkl`, "
            "and `models/feature_columns.pkl`."
        )
        st.stop()

    st.sidebar.header("Customer Attributes")

    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen_label = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

    st.sidebar.subheader("Account")
    tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.sidebar.selectbox("Contract", MULTI_CAT_OPTIONS["Contract"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox("Payment Method", MULTI_CAT_OPTIONS["PaymentMethod"])
    monthly_charges = st.sidebar.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
    )
    total_charges = st.sidebar.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(round(tenure * monthly_charges, 2)),
        step=1.0,
        help="Defaults to tenure x monthly charges; adjust if you know the exact value.",
    )

    st.sidebar.subheader("Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", MULTI_CAT_OPTIONS["MultipleLines"])
    internet_service = st.sidebar.selectbox("Internet Service", MULTI_CAT_OPTIONS["InternetService"])
    online_security = st.sidebar.selectbox("Online Security", MULTI_CAT_OPTIONS["OnlineSecurity"])
    online_backup = st.sidebar.selectbox("Online Backup", MULTI_CAT_OPTIONS["OnlineBackup"])
    device_protection = st.sidebar.selectbox("Device Protection", MULTI_CAT_OPTIONS["DeviceProtection"])
    tech_support = st.sidebar.selectbox("Tech Support", MULTI_CAT_OPTIONS["TechSupport"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", MULTI_CAT_OPTIONS["StreamingTV"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", MULTI_CAT_OPTIONS["StreamingMovies"])

    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen_label == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    st.subheader("Input Summary")
    st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)

    if st.button("Predict", type="primary"):
        X = preprocess_input(raw_input, scaler, feature_columns)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ This customer is likely to **CHURN** (probability: {probability:.1%})")
        else:
            st.success(f"✅ This customer is likely to **STAY** (churn probability: {probability:.1%})")

        st.progress(min(max(probability, 0.0), 1.0))


if __name__ == "__main__":
    main()
