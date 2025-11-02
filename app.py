import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Load Saved Pickle Files
# --------------------------------------------------
with open("svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(page_title="🌦️ Rain Prediction App", layout="centered")
st.title("🌦️ Weather Prediction Using SVM")
st.write("This app predicts whether it will rain tomorrow based on weather data.")

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.header("🔹 Enter Weather Details")

# Example features (change if your feature_columns are different)
# Use st.number_input for numeric fields
input_data = {}

for col in feature_columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Convert user inputs to DataFrame
input_df = pd.DataFrame([input_data])

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🔍 Predict"):
    try:
        # Reorder columns to match training order
        input_df = input_df.reindex(columns=feature_columns)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = svm_model.predict(input_scaled)[0]
        prediction_proba = svm_model.predict_proba(input_scaled)[0]

        # Display result
        st.success(f"✅ Prediction: {'🌧️ Rain Tomorrow' if prediction == 1 else '☀️ No Rain Tomorrow'}")
        st.write(f"**Probability of Rain:** {prediction_proba[1]*100:.2f}%")
    except Exception as e:
        st.error(f"Error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.write("---")
st.caption("Developed by Giriraj Pande | Mini ML Project 🌦️")
