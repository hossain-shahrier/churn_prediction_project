import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import FeatureEngineer

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
TARGET_COLUMN = "Churn Label"

METRICS = {
    "roc_auc": 0.85,
    "pr_auc": 0.66,
    "accuracy": 0.79,
    "churn_precision": 0.59,
    "churn_recall": 0.69,
}


@st.cache_resource
def load_artifact():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_sample_options():
    df = pd.read_excel(RAW_DATA_PATH)
    return {
        "defaults": df.iloc[0].to_dict(),
        "states": sorted(df["State"].dropna().unique().tolist()),
    }


def build_customer_row(
    tenure_months,
    monthly_charges,
    total_charges,
    cltv,
    gender,
    senior_citizen,
    partner,
    dependents,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    defaults,
):
    return pd.DataFrame(
        [
            {
                "CustomerID": "DEMO-0001",
                "Count": 1,
                "Country": defaults["Country"],
                "State": defaults["State"],
                "City": defaults["City"],
                "Zip Code": defaults["Zip Code"],
                "Lat Long": defaults["Lat Long"],
                "Latitude": defaults["Latitude"],
                "Longitude": defaults["Longitude"],
                "Gender": gender,
                "Senior Citizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": tenure_months,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Contract": contract,
                "Paperless Billing": paperless_billing,
                "Payment Method": payment_method,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
                "CLTV": cltv,
                "Churn Label": "No",
                "Churn Value": 0,
                "Churn Score": 0,
                "Churn Reason": "Don't know",
            }
        ]
    )


def predict_churn(raw_df, artifact):
    processed = FeatureEngineer().run_pipeline(raw_df)
    if TARGET_COLUMN in processed.columns:
        processed = processed.drop(columns=[TARGET_COLUMN])

    feature_columns = artifact["feature_columns"]
    features = processed.reindex(columns=feature_columns, fill_value=0)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    probability = float(artifact["model"].predict_proba(features)[0, 1])
    threshold = float(artifact["threshold"])
    churn_prediction = probability >= threshold

    return probability, threshold, churn_prediction


def main():
    st.set_page_config(
        page_title="Churn Prediction Demo",
        page_icon="📉",
        layout="wide",
    )

    artifact = load_artifact()
    options = load_sample_options()
    defaults = options["defaults"]

    st.title("Customer Churn Prediction")
    st.caption("XGBoost model trained on 7,043 telecom customers | MongoDB feature store pipeline")

    overview_col, metrics_col = st.columns([1.2, 1])

    with overview_col:
        st.markdown(
            """
            Enter customer details to estimate churn risk.
            The app runs the same feature engineering pipeline used in training,
            then scores the customer with the tuned XGBoost model.
            """
        )

    with metrics_col:
        st.metric("ROC-AUC", f"{METRICS['roc_auc']:.2f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{METRICS['accuracy']:.0%}")
        m2.metric("Precision", f"{METRICS['churn_precision']:.0%}")
        m3.metric("Recall", f"{METRICS['churn_recall']:.0%}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Account Profile")
        tenure_months = st.slider("Tenure (months)", 0, 72, int(defaults["Tenure Months"]))
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=float(defaults["Monthly Charges"]),
            step=1.0,
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(defaults["Total Charges"]),
            step=10.0,
        )
        cltv = st.number_input(
            "CLTV",
            min_value=0,
            max_value=10000,
            value=int(defaults["CLTV"]),
            step=50,
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]),
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=0,
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)

    with right:
        st.subheader("Services")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
        partner = st.selectbox("Partner", ["No", "Yes"], index=1)
        dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
            index=0,
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=1,
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
            index=0,
        )

    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        customer = build_customer_row(
            tenure_months=tenure_months,
            monthly_charges=monthly_charges,
            total_charges=total_charges,
            cltv=cltv,
            gender=gender,
            senior_citizen=senior_citizen,
            partner=partner,
            dependents=dependents,
            phone_service=phone_service,
            multiple_lines=multiple_lines,
            internet_service=internet_service,
            online_security=online_security,
            online_backup=online_backup,
            device_protection=device_protection,
            tech_support=tech_support,
            streaming_tv=streaming_tv,
            streaming_movies=streaming_movies,
            contract=contract,
            paperless_billing=paperless_billing,
            payment_method=payment_method,
            defaults=defaults,
        )

        with st.spinner("Running feature engineering and model scoring..."):
            probability, threshold, churn_prediction = predict_churn(customer, artifact)

        st.divider()
        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if churn_prediction:
                st.error(f"Likely to Churn ({probability:.1%} probability)")
            else:
                st.success(f"Likely to Stay ({probability:.1%} churn probability)")

            st.write(f"Tuned decision threshold: **{threshold:.3f}**")

        with gauge_col:
            st.progress(min(max(probability, 0.0), 1.0), text="Churn probability")

    with st.expander("Project architecture"):
        st.markdown(
            """
            1. **Data ingestion** loads Telco customer data from Excel
            2. **Feature engineering** cleans, encodes, and enriches customer records
            3. **MongoDB Atlas** stores engineered features as a feature store
            4. **XGBoost training** tunes hyperparameters and optimizes the decision threshold
            5. **Streamlit app** serves live predictions from the saved model artifact
            """
        )


if __name__ == "__main__":
    main()
