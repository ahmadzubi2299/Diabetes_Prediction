import streamlit as st
import pandas as pd
import joblib


# Load the trained Decision Tree model from the models folder
model = joblib.load("models/diabetes_decision_tree.pkl")

# Load the saved median values used during preprocessing
medians = joblib.load("models/diabetes_medians.pkl")


# Page title
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)


# App heading
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's information below to get a prediction.")


# Patient inputs
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=300,
    value=117
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=72
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=29
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=125
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=32.3
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)


# Prediction button
if st.button("Predict Diabetes"):

    # Put the patient's data into a dictionary
    patient_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    # Convert dictionary into a DataFrame
    patient_df = pd.DataFrame([patient_data])

    # Replace zero values with the medians
    for feature, median in medians.items():
        if patient_df.loc[0, feature] == 0:
            patient_df.loc[0, feature] = median

    # Make prediction
    prediction = model.predict(patient_df)[0]

    # Display result
    if prediction == 1:
        st.error("Diabetes predicted")
    else:
        st.success("No diabetes predicted")


# Disclaimer
st.warning(
    "This is an educational machine-learning project and "
    "is not a medical diagnosis. Please consult a qualified "
    "healthcare professional for medical advice."
)