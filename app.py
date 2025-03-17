import streamlit as st
import numpy as np
import pickle

# Load Model dan Scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Konfigurasi UI
st.set_page_config(page_title="Deteksi Diabetes", page_icon="🩺", layout="centered")
st.markdown("""
    <style>
        .main {background-color: #f4f4f4;}
        h1 {color: #2E8B57; text-align: center;}
        .stButton>button {background-color: #2E8B57; color: white; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Deteksi Diabetes dengan Naïve Bayes")

# Input Data
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

st.markdown("---")

# Prediksi
if st.button("🔍 Prediksi", key="predict_button"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    result = "✅ Tidak Diabetes" if prediction[0] == 0 else "⚠️ Diabetes"
    
    st.success(f"**Hasil Prediksi: {result}**")
