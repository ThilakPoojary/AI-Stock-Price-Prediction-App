# ===== PLAN =====
# Correct feature order: [open, high, low, adjclose, volume]

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Stock Predictor", layout="wide")

# ===== CSS =====
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #00f5a0;
}
.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 15px;
}
.result {
    font-size: 30px;
    text-align: center;
    color: white;
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, #22c55e, #0ea5e9);
}
</style>
""", unsafe_allow_html=True)

# ===== MODEL =====
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# ===== LOAD =====
model = ANN()
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load("scaler_x.pkl")

# ===== TITLE =====
st.markdown('<div class="title">📈 Stock Price Prediction</div>', unsafe_allow_html=True)

left, right = st.columns(2)

# ===== INPUT =====
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter Stock Features")

    with st.form("form"):
        f1 = st.number_input("Open Price", value=25.0)
        f2 = st.number_input("High Price", value=26.0)
        f3 = st.number_input("Low Price", value=24.5)
        f4 = st.number_input("Adj Close", value=23.5)   # ✅ FIXED
        f5 = st.number_input("Volume", value=25000000.0) # ✅ FIXED SCALE

        submit = st.form_submit_button("Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ===== OUTPUT =====
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if submit:
        input_data = np.array([[f1, f2, f3, f4, f5]])

        # scale input
        input_scaled = scaler.transform(input_data)

        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor)

        st.markdown(f"""
        <div class="result">
            Predicted Price<br>
            ₹ {prediction.item():.2f}
        </div>
        """, unsafe_allow_html=True)

        # sanity check
        if prediction.item() < f3 or prediction.item() > f2:
            st.warning("⚠️ Prediction outside High-Low range (check input)")
        else:
            st.success("✅ Prediction looks realistic")

    else:
        st.info("Enter values and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("---")
st.caption("Built using PyTorch + Streamlit")