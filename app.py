import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sample_healthcare_dataset.csv")
    return df

df = load_data()

# Encode categorical columns
le_gender = LabelEncoder()
le_disease = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Disease'] = le_disease.fit_transform(df['Disease'])

# Feature and target split
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("AI-Powered Disease Prediction")
st.markdown("### Enter patient details:")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
bp = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
sugar = st.number_input("Blood Sugar", min_value=50, max_value=300, value=120)
hr = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

gender_num = 1 if gender == "Male" else 0
input_data = np.array([[age, gender_num, bp, chol, sugar, hr, bmi]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Disease"):
    prediction = model.predict(input_scaled)
    disease = le_disease.inverse_transform(prediction)[0]
    st.success(f"Predicted Disease: {disease}")
