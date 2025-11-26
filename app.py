# app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load model and scaler
# -----------------------------
logreg = joblib.load("customer_segmentor.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Customer Cluster Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Inject custom CSS for dark neon theme
# -----------------------------
st.markdown("""
<style>
/* Body and text */
body {background-color:#0f0f1a; color:white;}
h1, h2, h3 {color:#00FFFF;}

/* Inputs */
.stNumberInput>div>input, .stSelectbox>div>div>div>span {
    background-color:#2A2A3F; 
    color:white;
    border-radius:8px;
}

/* Buttons */
.stButton>button {
    background-color:#00FFFF !important; 
    color:black !important; 
    font-weight:bold; 
    font-size:16px;
    border-radius:10px;
    padding:10px 20px;
}

/* Heatmap container */
.heatmap-container {
    background-color:#1E1E2F;
    padding:15px;
    border-radius:15px;
    box-shadow: 0 0 10px #00FFFF;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Page title
# -----------------------------
st.markdown("<h1 style='text-align:center;'>Customer Cluster Predictor</h1>", unsafe_allow_html=True)
st.write("Enter customer details below:")

# -----------------------------
# Inputs in 3 columns with emoji labels inside the input
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("🧑 Age (0-90)", min_value=0, max_value=90, value=30, step=1, key="age")
    education_level = st.selectbox("🎓 Education Level", ["Basic", "BSc", "MSc", "PhD"], key="edu")
    family_size = st.number_input("👪 Family Size", min_value=1, value=3, step=1, key="family")

with col2:
    annual_income = st.number_input("💰 Annual Income", min_value=0, value=50000, step=1000, key="income")
    accepted_campaign_count = st.number_input("📊 Accepted Campaign Count (0-5)", min_value=0, max_value=5, value=0, step=1, key="campaigns")
    num_purchases = st.number_input("🛍️ Number of Purchases", min_value=0, value=5, step=1, key="purchases")

with col3:
    total_spending = st.number_input("💵 Total Spending", min_value=0, value=1000, step=100, key="spending")
    customer_days_active = st.number_input("📅 Customer Days Active", min_value=0, value=100, step=1, key="days")

# -----------------------------
# Predict button
# -----------------------------
st.markdown("<br>")
if st.button("Predict Cluster"):
    # Validation
    inputs = [age, annual_income, family_size, accepted_campaign_count,
              num_purchases, total_spending, customer_days_active]
    names = ["Age","Annual Income","Family Size","Accepted Campaign Count",
             "Number of Purchases","Total Spending","Customer Days Active"]
    error_msg = next((f"{n} cannot be negative!" for n, v in zip(names, inputs) if v < 0), None)

    if error_msg:
        st.error(error_msg)
    else:
        # Map education level
        edu_map = {"Basic":0, "BSc":1, "MSc":2, "PhD":3}
        edu_num = edu_map[education_level]

        # Prepare input
        X_input = np.array([[age, edu_num, annual_income, family_size,
                             accepted_campaign_count, num_purchases, total_spending, customer_days_active]])
        X_scaled = scaler.transform(X_input)

        # Predict
        cluster = logreg.predict(X_scaled)[0]
        probabilities = logreg.predict_proba(X_scaled)[0]

        # Display results
        st.markdown(f"<h2 style='color:#00FFFF;'>Predicted Cluster: {cluster}</h2>", unsafe_allow_html=True)
        prob_str = ", ".join([f"Cluster {i}: {p:.2f}" for i, p in enumerate(probabilities)])
        st.write(f"**Probabilities:** {prob_str}")

       