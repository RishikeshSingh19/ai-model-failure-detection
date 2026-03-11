# =============================
# 1️⃣ IMPORTS
# =============================
import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np
import matplotlib.pyplot as plt


# =============================
# 2️⃣ PAGE CONFIG (FIRST STREAMLIT COMMAND)
# =============================
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# =============================
# 3️⃣ MANUAL ENCODING MAPS
# =============================
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

yes_no_map = {
    "No": 0,
    "Yes": 1
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}


# =============================
# 4️⃣ LOAD TRAINED MODEL
# =============================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "baseline_pipeline.pkl")
    model = joblib.load(model_path)
    return model


model = load_model()


# =============================
# 5️⃣ LOAD SHAP EXPLAINER
# =============================
@st.cache_resource
def load_explainer(_model):

    # Background dataset required by SHAP
    background = pd.DataFrame(
        np.zeros((50, len(_model.feature_names_in_))),
        columns=_model.feature_names_in_
    )

    # Prediction probability function
    def predict_fn(X):
        return _model.predict_proba(X)[:, 1]

    masker = shap.maskers.Independent(background)

    explainer = shap.Explainer(
        predict_fn,
        masker=masker,
        feature_names=_model.feature_names_in_
    )

    return explainer


explainer = load_explainer(model)

st.sidebar.success("✅ Model & SHAP Explainer loaded")


# =============================
# 6️⃣ PAGE HEADER
# =============================
st.title("📊 Telecom Customer Churn Prediction")

st.write("""
This application predicts whether a telecom customer is likely to churn
using a trained Machine Learning model.
""")


# =============================
# 7️⃣ CUSTOMER INPUT SECTION
# =============================
st.markdown("## 🧾 Customer Input")

tenure = st.number_input("Tenure (months)", 0, 72, 12)

monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.success("✅ Customer inputs captured")
st.info("ℹ️ Enter realistic customer values for best prediction accuracy.")


# =============================
# 8️⃣ PREDICTION SECTION
# =============================
if st.button("Predict Churn Risk"):

    # -----------------------------
    # Input Validation
    # -----------------------------
    if tenure <= 0:
        st.error("❌ Tenure must be greater than 0 months.")
        st.stop()

    if monthly_charges <= 0:
        st.error("❌ Monthly charges must be greater than 0.")
        st.stop()

    # -----------------------------
    # Create Feature Template
    # -----------------------------
    input_data = pd.DataFrame(columns=model.feature_names_in_)
    input_data.loc[0] = 0

    input_data.loc[0, "tenure"] = tenure
    input_data.loc[0, "MonthlyCharges"] = monthly_charges
    input_data.loc[0, "Contract"] = contract_map[contract]
    input_data.loc[0, "InternetService"] = internet_map[internet_service]
    input_data.loc[0, "TechSupport"] = yes_no_map[tech_support]
    input_data.loc[0, "PaymentMethod"] = payment_map[payment_method]

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -----------------------------
    # Dashboard
    # -----------------------------
    st.markdown("## 📊 Prediction Dashboard")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Prediction Outcome
    with col1:
        st.markdown("### Prediction Outcome")

        if prediction == 1:
            st.error("⚠️ Customer Likely to Churn")
        else:
            st.success("✅ Customer Likely to Stay")

    # Probability Display
    with col2:
        st.markdown("### Churn Probability")
        st.metric(
            label="Model Confidence",
            value=f"{probability:.2f}"
        )

    st.markdown("---")

    # -----------------------------
    # Risk Category
    # -----------------------------
    st.markdown("### Risk Assessment")

    if probability < 0.3:
        st.success("🟢 LOW RISK CUSTOMER")
    elif probability < 0.6:
        st.warning("🟡 MEDIUM RISK CUSTOMER")
    else:
        st.error("🔴 HIGH RISK CUSTOMER")

    st.markdown("---")

    # =============================
    # SHAP EXPLANATION
    # =============================
    st.subheader("🧠 Why this prediction?")

    try:
        shap_values = explainer(input_data)

        # Waterfall Plot
        st.markdown("### 📈 Prediction Explanation (SHAP)")
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)

        # Feature Importance
        st.markdown("### 🌍 Model Feature Importance")
        fig2 = plt.figure()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig2)
        plt.close(fig2)

    except Exception:
        st.warning("SHAP visualization could not be generated.")

    # -----------------------------
    # Feature Driver Summary
    # -----------------------------
    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact": shap_values.values[0]
    })

    shap_df["abs_impact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values("abs_impact", ascending=False)

    top_features = shap_df.head(5)

    increase_risk = top_features[top_features["Impact"] > 0]
    decrease_risk = top_features[top_features["Impact"] < 0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔺 Increased Churn Risk")
        if len(increase_risk) > 0:
            for f in increase_risk["Feature"]:
                st.write(f"• {f}")
        else:
            st.write("No strong risk drivers")

    with col2:
        st.markdown("### 🔻 Reduced Churn Risk")
        if len(decrease_risk) > 0:
            for f in decrease_risk["Feature"]:
                st.write(f"• {f}")
        else:
            st.write("No strong protective factors")