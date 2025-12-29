import streamlit as st
import numpy as np
import joblib

# Load trained Random Forest model
model = joblib.load("random_forest_model_smaller.joblib")

st.set_page_config(page_title="EEG Eye State Detection", layout="centered")

st.title("🧠 EEG Eye State Detection")
st.write("Predict whether the eye is **Open** or **Closed** using EEG signals")

# IMPORTANT: Internal EEG channel order (DO NOT CHANGE)
eeg_channels = [
    "AF3","F7","F3","FC5","T7","P7","O1",
    "O2","P8","T8","FC6","F4","F8","AF4"
]

# User-friendly labels
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

user_input = []

for label in sensor_labels:
    value = st.number_input(label, value=0.0)
    user_input.append(value)

if st.button("🔍 Predict Eye State"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("👁️ Eye Closed")
    else:
        st.success("👁️ Eye Open")



