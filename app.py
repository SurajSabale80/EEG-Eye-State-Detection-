import streamlit as st
import numpy as np
import joblib

# Load Random Forest model
model = joblib.load("random_forest_model_smaller.joblib")

st.set_page_config(page_title="EEG Eye State Detection", layout="centered")

st.title("🧠 EEG Eye State Detection")
st.write("Predict **Eye Open / Eye Closed** using EEG signals")

st.subheader("Enter EEG Channel Values")

channels = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

user_input = []
for ch in channels:
    val = st.number_input(f"{ch}", value=0.0)
    user_input.append(val)

if st.button("🔍 Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("👁️ Eye Closed")
    else:
        st.success("👁️ Eye Open")

