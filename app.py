import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load assets
model = joblib.load('lg_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("❤️ Heart Disease Prediction System")
st.markdown("### Provide The Following Details")

# ----------------------------
# User Inputs
# ----------------------------

age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.slider(
    "Resting Blood Pressure (mm Hg)",
    80, 200, 120
)

cholesterol = st.slider(
    "Cholesterol (mg/dL)",
    100, 600, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["NORMAL", "ST", "LVH"]
)

max_hr = st.slider(
    "Max Heart Rate",
    60, 220, 150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Show Result
    st.markdown("## Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease")
        st.write(f"Risk Probability: {round(probability * 100, 2)}%")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write(f"Risk Probability: {round(probability * 100, 2)}%")
