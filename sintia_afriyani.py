import streamlit as st
import numpy as np
import pickle

# Load the trained model
filename_dt = 'model_dt.sav'
model = pickle.load(open(filename_dt, 'rb'))

# Load the scaler (pastikan scaler juga disimpan sebelumnya)
scaler_filename = 'scaler.sav'
scaler = pickle.load(open(scaler_filename, 'rb'))

# Streamlit UI dengan styling lebih menarik
st.set_page_config(page_title="Prediksi Diabetes", page_icon="💎", layout="centered")
st.title("Prediksi Diabetes")
st.markdown("""
    Silakan masukkan data berikut untuk memprediksi apakah pasien terkena diabetes atau tidak.
    Semua kolom wajib diisi untuk prediksi yang akurat.
    ----Sintia Afriyani----
""")

# Input dari pengguna dengan nilai default 0
col1, col2 = st.columns(2)

with col1:
    Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=0)
    Age = st.number_input("Age", min_value=0, max_value=120, value=0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=60.0, value=0.0, format="%.1f")
    
with col2:
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    Insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=0)

# Styling for the prediction button
st.markdown("""<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 15px 32px;
        border: none;
        cursor: pointer;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>""", unsafe_allow_html=True)

# Tombol Prediksi
if st.button("Prediksi"):
    # Mengonversi input ke numpy array
    input_data = np.array([Glucose, Age, BMI, BloodPressure, Pregnancies, Insulin]).reshape(1, -1)
    
    # Standarisasi input
    std_data = scaler.transform(input_data)
    
    # Melakukan prediksi
    prediction = model.predict(std_data)
    
    # Menampilkan hasil dengan styling yang lebih baik
    if prediction[0] == 0:
        st.success("✅ Pasien **TIDAK** terkena diabetes.")
    else:
        st.error("❌ Pasien **TERKENA** diabetes.")
