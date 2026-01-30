# =========================
# FastAPI App
# =========================

from fastapi import FastAPI
from pydantic import BaseModel
from inference import hybrid_predict

# -------------------------
# Initialize app
# -------------------------
app = FastAPI(
    title="Hybrid Emotion Detection API",
    description="Emotion classification using Semantic + BERT Hybrid Model",
    version="1.0"
)

# -------------------------
# Request schema
# -------------------------
class TextInput(BaseModel):
    text: str
    alpha: float | None = 0.5  # best performing value

# -------------------------
# Health check
# -------------------------
@app.get("/")
def root():
    return {
        "message": "Emotion Detection API is running",
        "status": "OK"
    }

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict_emotion(data: TextInput):
    emotion, confidence = hybrid_predict(
        text=data.text,
        alpha=data.alpha
    )

    return {
        "input_text": data.text,
        "predicted_emotion": emotion,
        "confidence": round(float(confidence), 3),
        "alpha": data.alpha
    }

