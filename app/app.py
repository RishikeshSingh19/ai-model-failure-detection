import streamlit as st
import pandas as pd
import joblib

# App title
st.title("Telecom Customer Churn Prediction")

# App description
st.write(
    "Enter telecom customer information to predict the probability of churn."
)