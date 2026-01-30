import streamlit as st
import requests

# -------------------------
# CONFIG
# -------------------------
API_URL = "https://hybrid-emotion-api.onrender.com/predict"

st.set_page_config(
    page_title="Hybrid Emotion Detection",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Hybrid Emotion Detection")
st.write("Emotion classification using **Semantic + BERT Hybrid Model**")

# -------------------------
# INPUTS (ADD UNIQUE KEYS)
# -------------------------
text = st.text_area(
    "Enter text",
    placeholder="I feel very happy today...",
    height=120,
    key="input_text_area"   # 🔥 FIX
)

alpha = st.slider(
    "Hybrid weight (alpha)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    key="alpha_slider"      # 🔥 FIX
)

# -------------------------
# PREDICT
# -------------------------
if st.button("🔍 Predict Emotion", key="predict_button"):  # 🔥 FIX
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("⏳ Warming up model & predicting..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text, "alpha": alpha},
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Prediction successful")

                    st.markdown(
                        f"""
                        ### 🎯 Result
                        **Emotion:** `{result['predicted_emotion']}`  
                        **Confidence:** `{result['confidence']}`
                        """
                    )
                else:
                    st.error(
                        f"API Error {response.status_code}: {response.text}"
                    )

            except requests.exceptions.Timeout:
                st.error(
                    "⏰ Request timed out. The model may be waking up."
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("Hybrid Emotion Detection • FastAPI + BERT + XGBoost")
