import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="EEG Eye State Detection", layout="centered")

st.title("🧠 EEG Eye State Detection")
st.write("Predict whether the **eye is open or closed** using EEG signals")

st.subheader("Enter EEG Channel Values")

# 14 EEG channels
channels = [
    "AF3","F7","F3","FC5","T7","P7","O1",
    "O2","P8","T8","FC6","F4","F8","AF4"
]

user_input = []
for ch in channels:
    value = st.number_input(f"{ch}", value=0.0)
    user_input.append(value)

if st.button("🔍 Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.success("👁️ Eye Closed")
    else:
        st.success("👁️ Eye Open")
