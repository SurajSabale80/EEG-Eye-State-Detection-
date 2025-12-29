import streamlit as st
import numpy as np
import joblib

# ===============================
# Load trained Random Forest model
# ===============================
model = joblib.load("random_forest_model_smaller.joblib")

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="EEG Eye State Detection",
    layout="centered"
)

st.title("🧠 EEG Eye State Detection")
st.write(
    "This application predicts whether the **eye is Open or Closed** "
    "using EEG brain signal values."
)

# =====================================================
# INTERNAL EEG CHANNEL ORDER (DO NOT CHANGE / DO NOT SHOW)
# =====================================================
eeg_channels = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# =================================
# USER-FRIENDLY SENSOR LABELS (UI)
# =================================
sensor_labels = [
    "Front Brain Sensor (Left)",
    "Front Brain Sensor (Left Side)",
    "Front Brain Sensor (Center Left)",
    "Motor Area Sensor (Left)",
    "Side Brain Sensor (Left)",
    "Upper Brain Sensor (Left)",
    "Vision Area Sensor (Left)",
    "Vision Area Sensor (Right)",
    "Upper Brain Sensor (Right)",
    "Side Brain Sensor (Right)",
    "Motor Area Sensor (Right)",
    "Front Brain Sensor (Right)",
    "Front Brain Sensor (Right Side)",
    "Front Brain Sensor (Right)"
]

st.subheader("Enter EEG Sensor Values")

# ===============================
# Collect User Input
# ===============================
user_input = []

for label in sensor_labels:
    value = st.number_input(
        label,
        value=0.0,
        help="Enter EEG signal value"
    )
    user_input.append(value)

# ===============================
# Prediction Button
# ===============================
if st.button("🔍 Predict Eye State"):
    input_array = np.array(user_input).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("👁️ Eye Closed")
    else:
        st.success("👁️ Eye Open")




