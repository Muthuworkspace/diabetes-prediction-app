import streamlit as st
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('disease_model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))


st.title("Early Disease Prediction")

# Clear, descriptive names
pregnancies = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose', min_value=0.0)
blood_pressure = st.number_input('Blood Pressure', min_value=0.0)
skin_thickness = st.number_input('Skin Thickness', min_value=0.0)
insulin = st.number_input('Insulin', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)

if st.button('Predict'):
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                            columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    features_scaled = scaler.transform(input_df)
    result = model.predict(features_scaled)

    output = 'Diabetes Detected' if result[0] == 1 else 'No Diabetes Detected'
    st.success(output)
