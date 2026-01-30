# =========================
# Hybrid Inference Module
# =========================

import numpy as np
import torch
import joblib
import pickle
import re

from transformers import BertTokenizer, BertForSequenceClassification
from scipy.sparse import hstack

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load paths
# -------------------------
MODEL_DIR = "models"

# -------------------------
# Load semantic models
# -------------------------
xgb = joblib.load(f"{MODEL_DIR}/xgb.pkl")
tfidf = joblib.load(f"{MODEL_DIR}/tfidf.pkl")
scaler = joblib.load(f"{MODEL_DIR}/nrc_scaler.pkl")

# -------------------------
# Load label encoder
# -------------------------
with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------
# Load BERT
# -------------------------
tokenizer = BertTokenizer.from_pretrained(f"{MODEL_DIR}/bert")
bert_model = BertForSequenceClassification.from_pretrained(
    f"{MODEL_DIR}/bert"
).to(device)
bert_model.eval()

# -------------------------
# Text cleaning (semantic only)
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -------------------------
# Semantic prediction
# -------------------------
def semantic_predict(text):
    clean = clean_text(text)

    X_tfidf = tfidf.transform([clean])
    X_nrc = scaler.transform(
        np.zeros((1, scaler.n_features_in_))
    )

    X_sem = hstack([X_tfidf, X_nrc])
    return xgb.predict_proba(X_sem)

# -------------------------
# BERT prediction
# -------------------------
def bert_predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs

# -------------------------
# Hybrid prediction
# -------------------------
def hybrid_predict(text, alpha=0.5):
    ctx_probs = bert_predict(text)
    sem_probs = semantic_predict(text)

    hybrid_probs = alpha * ctx_probs + (1 - alpha) * sem_probs
    pred_idx = np.argmax(hybrid_probs)

    emotion = le.classes_[pred_idx]
    confidence = hybrid_probs[0][pred_idx]

    return emotion, confidence

