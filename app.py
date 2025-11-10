import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# ==============================
# Load saved model and encoders
# ==============================
model = joblib.load("best_disaster_model.pkl")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Load feature names if available
try:
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
except:
    feature_names = [
        "rainfall", "temperature", "humidity", "wind_speed", "soil_moisture",
        "river_level", "seismic_activity", "year", "month", "day", "hour"
    ]

# ==============================
# Streamlit App UI
# ==============================
st.set_page_config(page_title="AI Disaster Prediction System", layout="wide")
st.title("🌍 AI-Powered Multi-Disaster Prediction System")
st.write("Predict potential natural disasters based on environmental conditions using a trained AI model.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, 50.0)
    temperature = st.number_input("🌡️ Temperature (°C)", -10.0, 60.0, 28.0)
    humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 70.0)
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", 0.0, 200.0, 30.0)
    soil_moisture = st.number_input("🌱 Soil Moisture (%)", 0.0, 100.0, 40.0)

with col2:
    river_level = st.number_input("🌊 River Level (m)", 0.0, 15.0, 3.0)
    seismic_activity = st.number_input("🌋 Seismic Activity (0–10)", 0.0, 10.0, 2.0)
    year = st.number_input("📅 Year", 2000, 2100, 2025)
    month = st.number_input("📆 Month", 1, 12, 6)
    day = st.number_input("🗓️ Day", 1, 31, 15)
    hour = st.number_input("⏰ Hour", 0, 23, 12)

# Prediction button
if st.button("🔮 Predict Disaster Type"):
    input_data = np.array([[rainfall, temperature, humidity, wind_speed, soil_moisture,
                            river_level, seismic_activity, year, month, day, hour]])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    pred_idx = model.predict(input_scaled)
    pred_label = label_encoder.inverse_transform(pred_idx.astype(int))[0]

    st.success(f"🌍 **Predicted Disaster Type:** {pred_label}")

    # Add a simple safety interpretation
    st.info("⚠️ This prediction is based on environmental data. Always verify with official disaster monitoring systems.")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, Scikit-learn, and XGBoost.")
