import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.svm import SVC

st.set_page_config(page_title="🌦️ Rain Prediction (SVM)", layout="centered")
st.title("🌦️ Rain Prediction using SVM")
st.write("Predict whether it will rain tomorrow based on today’s weather conditions.")
st.markdown("---")

# -------------------------------
# Load and prepare dataset
# -------------------------------
df = pd.read_csv("weather.csv")

# Drop columns with too many missing values
df = df.dropna(thresh=len(df) * 0.7, axis=1)
df = df.fillna(df.median(numeric_only=True))

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
y = df["RainTomorrow_Yes"]
X = df.drop("RainTomorrow_Yes", axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance data
data_bal = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), y.reset_index(drop=True)], axis=1)
majority = data_bal[data_bal["RainTomorrow_Yes"] == 0]
minority = data_bal[data_bal["RainTomorrow_Yes"] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced_data = pd.concat([majority, minority_upsampled])

X_bal = balanced_data.drop("RainTomorrow_Yes", axis=1)
y_bal = balanced_data["RainTomorrow_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
accuracy = svm_model.score(X_test, y_test)
st.success(f"✅ SVM Model trained successfully! Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Streamlit Inputs
# -------------------------------
st.markdown("### Enter Weather Conditions:")

col1, col2 = st.columns(2)

with col1:
    MinTemp = st.number_input("MinTemp (°C)", value=15.0)
    MaxTemp = st.number_input("MaxTemp (°C)", value=20.0)
    Rainfall = st.number_input("Rainfall (mm)", value=10.0)
    Evaporation = st.number_input("Evaporation (mm)", value=3.5)
    Sunshine = st.number_input("Sunshine (hours)", value=2.0)
    WindGustSpeed = st.number_input("WindGustSpeed (km/h)", value=45.0)
    WindSpeed9am = st.number_input("WindSpeed9am (km/h)", value=20.0)
    WindSpeed3pm = st.number_input("WindSpeed3pm (km/h)", value=30.0)
    Humidity9am = st.number_input("Humidity9am (%)", value=90.0)
    Humidity3pm = st.number_input("Humidity3pm (%)", value=85.0)

with col2:
    Pressure9am = st.number_input("Pressure9am (hPa)", value=1005.0)
    Pressure3pm = st.number_input("Pressure3pm (hPa)", value=1003.0)
    Cloud9am = st.number_input("Cloud9am (oktas)", value=7.0)
    Cloud3pm = st.number_input("Cloud3pm (oktas)", value=7.0)
    Temp9am = st.number_input("Temp9am (°C)", value=17.0)
    Temp3pm = st.number_input("Temp3pm (°C)", value=19.0)
    RISK_MM = st.number_input("RISK_MM", value=10.0)
    WindGustDir = st.selectbox("WindGustDir", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    WindDir9am = st.selectbox("WindDir9am", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    WindDir3pm = st.selectbox("WindDir3pm", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    RainToday = st.selectbox("RainToday", ["No", "Yes"])

# -------------------------------
# One-hot encode categorical inputs
# -------------------------------
input_data = {
    "MinTemp": MinTemp,
    "MaxTemp": MaxTemp,
    "Rainfall": Rainfall,
    "Evaporation": Evaporation,
    "Sunshine": Sunshine,
    "WindGustSpeed": WindGustSpeed,
    "WindSpeed9am": WindSpeed9am,
    "WindSpeed3pm": WindSpeed3pm,
    "Humidity9am": Humidity9am,
    "Humidity3pm": Humidity3pm,
    "Pressure9am": Pressure9am,
    "Pressure3pm": Pressure3pm,
    "Cloud9am": Cloud9am,
    "Cloud3pm": Cloud3pm,
    "Temp9am": Temp9am,
    "Temp3pm": Temp3pm,
    "RISK_MM": RISK_MM
}

# Create a DataFrame
input_df = pd.DataFrame([input_data])

# Add all dummy columns as 0 initially
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Set correct one-hot columns to 1 based on user selection
if f"WindGustDir_{WindGustDir}" in input_df.columns:
    input_df[f"WindGustDir_{WindGustDir}"] = 1
if f"WindDir9am_{WindDir9am}" in input_df.columns:
    input_df[f"WindDir9am_{WindDir9am}"] = 1
if f"WindDir3pm_{WindDir3pm}" in input_df.columns:
    input_df[f"WindDir3pm_{WindDir3pm}"] = 1
if RainToday == "Yes" and "RainToday_Yes" in input_df.columns:
    input_df["RainToday_Yes"] = 1

# Reorder columns exactly like training
input_df = input_df[X.columns]

# Scale input
scaled_input = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Rain Tomorrow"):
    pred = svm_model.predict(scaled_input)[0]
    prob = svm_model.predict_proba(scaled_input)[0][1] * 100

    if pred == 1:
        st.success(f"🌧️ Yes, it will rain tomorrow. (Probability: {prob:.2f}%)")
    else:
        st.info(f"☀️ No, it will not rain tomorrow. (Probability: {100 - prob:.2f}%)")
