import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from streamlit_js_eval import get_geolocation

# --- WEBSITE PAGE CONFIGURATION ---
st.set_page_config(page_title="AeroVeda AI", page_icon="🌾", layout="centered")

# --- BRANDING HEADER ---
st.title("🌾 AeroVeda AI Predictor")
st.subheader("Satellite-Driven Micro-Climate Analytics")
st.markdown("Supporting **UN SDG 2: Zero Hunger** by empowering rural yields through predictive machine learning.")
st.write("---")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Location Settings")
mode = st.sidebar.radio("Choose Mode:", ["Type Location Manually", "Use Device GPS"])

# Deep accurate default target: Kanha Shanti Vanam coordinates (Telangana)
lat, lon = 17.2917, 78.2250  
location_display = "Kanha Shanti Vanam (Telangana Base)"

# --- PATHWAY 1: MANUAL TYPING ---
if mode == "Type Location Manually":
    st.header("📍 Step 1: Type Your Farm Location")
    location_name = st.text_input("Enter your Village, Town, or Landmark Name:", "Kanha Shanti Vanam")
    
    # If the user types something other than the default, search for it
    if location_name and location_name != "Kanha Shanti Vanam":
        try:
            geo_url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
            headers = {"User-Agent": "AeroVedaApp/1.0"}
            geo_res = requests.get(geo_url, headers=headers).json()
            if geo_res:
                lat = float(geo_res[0]["lat"])
                lon = float(geo_res[0]["lon"])
                location_display = location_name
                st.success(f"📍 Map Grid Found! Lat: {lat:.4f}, Lon: {lon:.4f}")
            else:
                st.error("Location name not recognized. Using default baseline grid.")
        except Exception:
            pass
    else:
        st.success("📍 Locked to exact Kanha Shanti Vanam coordinates.")

# --- PATHWAY 2: AUTOMATIC GPS ---
else:
    st.header("📍 Step 1: Device GPS Active")
    st.info("Please accept the browser location pop-up if prompted.")
    gps_location = get_geolocation()
    
    if gps_location and 'coords' in gps_location:
        lat = gps_location['coords']['latitude']
        lon = gps_location['coords']['longitude']
        location_display = "Your Current Device Location"
        st.success(f"✅ GPS Signal Locked! Lat: {lat:.4f}, Lon: {lon:.4f}")
    else:
        st.warning("Warming up GPS module... Awaiting authorization from your browser.")

# --- TRIGGER AUDIT BUTTON ---
st.write("---")
if st.button("Run AI Micro-Climate Audit"):
    with st.spinner("Connecting to live satellite telemetry..."):
        try:
            # 1. Fetch 100% Real-Time Live Current Weather Data with explicit current variables
            forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&timezone=auto"
            f_res = requests.get(forecast_url).json()
            
            # Explicit dictionary parsing for Open-Meteo's standard response format
            current_temp = float(f_res["current"]["temperature_2m"])
            current_humidity = float(f_res["current"]["relative_humidity_2m"])
            
            # 2. Fetch 2025 Historical Archive for Machine Learning Training
            archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2025-01-01&end_date=2025-12-31&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean&timezone=auto"
            arch_res = requests.get(archive_url).json()["daily"]
            
            df = pd.DataFrame({
                "Max_Temp": arch_res["temperature_2m_max"],
                "Min_Temp": arch_res["temperature_2m_min"],
                "Humidity": arch_res["relative_humidity_2m_mean"]
            })
            df['Risk_Score'] = np.clip((df['Max_Temp'] * 1.5) - (df['Humidity'] * 0.2), 0, 100)
            
            # 3. Train the Linear Regression AI Brain
            X = df[['Max_Temp', 'Min_Temp', 'Humidity']]
            y = df['Risk_Score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            ai_model = LinearRegression()
            ai_model.fit(X_train, y_train)
            
            # 4. Compute Live Risk Prediction
            current_data_row = pd.DataFrame([{"Max_Temp": current_temp, "Min_Temp": current_temp-10, "Humidity": current_humidity}])
            predicted_risk = ai_model.predict(current_data_row)[0]
            
            # 5. Render Dashboard Interface
            st.header("📊 Live AI Analytics Report")
            st.caption(f"Target: {location_display} | Coordinates: {lat:.4f}, {lon:.4f}")
            
            col1, col2 = st.columns(2)
            col1.metric("🌡️ Current Temp", f"{current_temp} °C")
            col2.metric("💧 Tracked Humidity", f"{current_humidity}%")
            
            if predicted_risk > 45:
                st.error(f"⚠️ HIGH RISK CRITICAL WARNING\n\nCalculated Crop Stress Score: {predicted_risk:.1f} / 100\n\nDeploy immediate canopy shielding or drone irrigation cycles to protect soil moisture loops.")
            else:
                st.success(f"✅ STABLE CONDITIONS\n\nCalculated Crop Stress Score: {predicted_risk:.1f} / 100\n\nNo extreme micro-climate stress anomalies predicted for this cycle.")
                
        except Exception as e:
            st.error(f"Data stream syncing error: {e}")
