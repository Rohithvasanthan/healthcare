import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('disease_predictor.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# App title
st.title("AI-Powered Disease Prediction")
st.markdown("Enter patient details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
bp = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
sugar = st.number_input("Blood Sugar", min_value=50, max_value=300, value=120)
hr = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

# Convert gender to numerical
gender_num = 1 if gender == "Male" else 0

# Prepare input
input_data = np.array([[age, gender_num, bp, cholesterol, sugar, hr, bmi]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Disease"):
    prediction = model.predict(input_scaled)
    disease = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Disease: {disease}")
