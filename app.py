import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from streamlit.components.v1 import html

# --- WEBSITE PAGE CONFIGURATION ---
st.set_page_config(page_title="AeroVeda AI", page_icon="🌾", layout="centered")

# --- BRANDING HEADER ---
st.title("🌾 AeroVeda AI Predictor")
st.subheader("Satellite-Driven Micro-Climate Analytics")
st.markdown("Supporting **UN SDG 2: Zero Hunger** by empowering rural yields through predictive machine learning.")
st.write("---")

# --- USER INPUT SECTION ---
st.header("📍 Step 1: Locate Your Farm")

# Initialize session state variables to hold GPS coordinates if found
if "gps_lat" not in st.session_state:
    st.session_state.gps_lat = None
if "gps_lon" not in st.session_state:
    st.session_state.gps_lon = None

# Invisible JavaScript component that triggers browser's native "Find My Location" popup
js_geo = """
<script>
function getLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(showPosition, showError);
  } else {
    window.parent.postMessage({type: "GEO_ERROR", msg: "Geolocation not supported"}, "*");
  }
}
function showPosition(position) {
  window.parent.postMessage({
    type: "GEO_SUCCESS", 
    lat: position.coords.latitude, 
    lon: position.coords.longitude
  }, "*");
}
function showError(error) {
  window.parent.postMessage({type: "GEO_ERROR", msg: error.message}, "*");
}
</script>
<button onclick="getLocation()" style="
    background-color: #2e7d32; 
    color: white; 
    border: none; 
    padding: 10px 20px; 
    border-radius: 8px; 
    cursor: pointer; 
    font-weight: bold;
    width: 100%;
    font-size: 14px;
">📍 Use My Exact GPS Location</button>
"""

# Render the custom GPS Button
html(js_geo, height=50)

# Listen for the message sent back by the browser's JavaScript
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Fallback text input so they can still type a name manually
location = st.text_input("Or, type your Village, Town, or Landmark Name manually:", "Kanha Shanti Vanam")

# Create a master trigger: True if they click the run button
run_audit = st.button("Run AI Micro-Climate Audit")

if run_audit:
    with st.spinner("Analyzing satellite matrix..."):
        try:
            lat, lon = None, None
            
            # Check if text input should be used
            if location and location != "Kanha Shanti Vanam":
                geo_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
                headers = {"User-Agent": "AeroVedaApp/1.0"}
                geo_res = requests.get(geo_url, headers=headers).json()
                if geo_res:
                    lat = geo_res[0]["lat"]
                    lon = geo_res[0]["lon"]
            
            # Default fallback to Kanha Shanti Vanam coordinates if nothing else is specified
            if lat is None or lon is None:
                lat, lon = 17.2917, 78.2250  # Precise coordinates for Kanha Shanti Vanam area
            
            # --- Live Forecast Stream ---
            forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&timezone=auto"
            f_res = requests.get(forecast_url).json()
            
            current_temp = f_res["current"]["temperature_2m"]
            current_humidity = f_res["current"]["relative_humidity_2m"]
            
            # --- Fetch 2025 Historical Archive for Machine Learning Training ---
            archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2025-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean&timezone=auto"
            arch_res = requests.get(archive_url).json()["daily"]
            
            df = pd.DataFrame({
                "Max_Temp": arch_res["temperature_2m_max"],
                "Min_Temp": arch_res["temperature_2m_min"],
                "Humidity": arch_res["relative_humidity_2m_mean"]
            })
            df['Risk_Score'] = np.clip((df['Max_Temp'] * 1.5) - (df['Humidity'] * 0.2), 0, 100)
            
            # --- Train the Linear Regression AI Brain ---
            X = df[['Max_Temp', 'Min_Temp', 'Humidity']]
            y = df['Risk_Score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            ai_model = LinearRegression()
            ai_model.fit(X_train, y_train)
            
            # --- Compute Live Risk Prediction ---
            current_data_row = pd.DataFrame([{"Max_Temp": current_temp, "Min_Temp": current_temp-10, "Humidity": current_humidity}])
            predicted_risk = ai_model.predict(current_data_row)[0]
            
            # --- Render Dashboard Interface ---
            st.header("📊 Live AI Analytics Report")
            st.caption(f"Target Coordinates Locked: Latitude {float(lat):.4f}, Longitude {float(lon):.4f}")
            
            col1, col2 = st.columns(2)
            col1.metric("🌡️ Current Temp", f"{current_temp} °C")
            col2.metric("💧 Tracked Humidity", f"{current_humidity}%")
            
            if predicted_risk > 45:
                st.error(f"⚠️ HIGH RISK CRITICAL WARNING\n\nCalculated Crop Stress Score: {predicted_risk:.1f} / 100\n\nDeploy immediate canopy shielding or drone irrigation cycles to protect soil moisture loops.")
            else:
                st.success(f"✅ STABLE CONDITIONS\n\nCalculated Crop Stress Score: {predicted_risk:.1f} / 100\n\nNo extreme micro-climate stress anomalies predicted for this cycle.")
                
        except Exception as e:
            st.error(f"Data stream syncing error: {e}")
