import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="ECG Heartbeat Classification",
    layout="centered"
)

st.title("❤️ ECG Heartbeat Classification (LSTM)")
st.write("Upload a CSV file containing ECG signals (187 values per row)")

# ------------------------------
# Load model & scaler
# ------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = load_model("ecg_lstm_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()
st.success("Model and scaler loaded successfully")

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_ecg(ecg_values):
    ecg_values = np.array(ecg_values).reshape(1, -1)
    ecg_values = scaler.transform(ecg_values)
    ecg_values = ecg_values.reshape(1, 187, 1)
    return ecg_values

def predict_ecg(ecg_input):
    ecg_processed = preprocess_ecg(ecg_input)
    pred = model.predict(ecg_processed)
    confidence = float(pred[0][0])
    label = "Abnormal" if confidence > 0.5 else "Normal"
    return label, confidence

# ------------------------------
# File upload
# ------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload ECG CSV file",
    type=["csv"]
)

# ------------------------------
# Run after upload
# ------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)

    st.write(f"📊 File loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

    if df.shape[1] < 187:
        st.error("❌ CSV must contain at least 187 ECG values per row")
    else:
        # Take one random ECG sample
        sample = df.sample(1).iloc[:, :187].values.flatten()

        with st.spinner("🔍 Analyzing ECG signal..."):
            result, confidence = predict_ecg(sample)

        # ------------------------------
        # Results
        # ------------------------------
        st.subheader("🧠 Prediction Result")

        if result == "Normal":
            st.success(f"✅ **Prediction:** {result}")
        else:
            st.error(f"⚠️ **Prediction:** {result}")

        st.write(f"**Confidence:** {confidence:.2f}")

        # ------------------------------
        # Plot ECG
        # ------------------------------
        st.subheader("📈 ECG Signal Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Amplitude")
        ax.set_title("ECG Signal")
        st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to start ECG classification")
