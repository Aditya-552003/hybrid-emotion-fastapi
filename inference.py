<<<<<<< HEAD
import numpy as np
import torch
import joblib
import pickle
import re

from transformers import BertTokenizer, BertForSequenceClassification
from scipy.sparse import hstack

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load models
# --------------------
xgb = joblib.load("models/xgb.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/nrc_scaler.pkl")

tokenizer = BertTokenizer.from_pretrained("models/bert")
bert_model = BertForSequenceClassification.from_pretrained("models/bert").to(device)
bert_model.eval()

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# --------------------
# Utils
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# --------------------
# Semantic predict
# --------------------
def semantic_predict(text):
    clean = clean_text(text)
    X_tfidf = tfidf.transform([clean])
    X_nrc = scaler.transform(np.zeros((1, scaler.n_features_in_)))
    X_sem = hstack([X_tfidf, X_nrc])
    return xgb.predict_proba(X_sem)

# --------------------
# BERT predict
# --------------------
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
        return torch.softmax(logits, dim=1).cpu().numpy()

# --------------------
# HYBRID predict
# --------------------
def hybrid_predict(text, alpha=0.5):
    ctx_probs = bert_predict(text)
    sem_probs = semantic_predict(text)

    hybrid_probs = alpha * ctx_probs + (1 - alpha) * sem_probs
    idx = np.argmax(hybrid_probs)

    return le.classes_[idx], float(hybrid_probs[0][idx])
=======
import numpy as np
import torch
import joblib
import pickle
import re

from transformers import BertTokenizer, BertForSequenceClassification
from scipy.sparse import hstack

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load models
# --------------------
xgb = joblib.load("models/xgb.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/nrc_scaler.pkl")

tokenizer = BertTokenizer.from_pretrained("models/bert")
bert_model = BertForSequenceClassification.from_pretrained("models/bert").to(device)
bert_model.eval()

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# --------------------
# Utils
# --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# --------------------
# Semantic predict
# --------------------
def semantic_predict(text):
    clean = clean_text(text)
    X_tfidf = tfidf.transform([clean])
    X_nrc = scaler.transform(np.zeros((1, scaler.n_features_in_)))
    X_sem = hstack([X_tfidf, X_nrc])
    return xgb.predict_proba(X_sem)

# --------------------
# BERT predict
# --------------------
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
        return torch.softmax(logits, dim=1).cpu().numpy()

# --------------------
# HYBRID predict
# --------------------
def hybrid_predict(text, alpha=0.5):
    ctx_probs = bert_predict(text)
    sem_probs = semantic_predict(text)

    hybrid_probs = alpha * ctx_probs + (1 - alpha) * sem_probs
    idx = np.argmax(hybrid_probs)

    return le.classes_[idx], float(hybrid_probs[0][idx])
>>>>>>> 4708182 (Initial commit: Hybrid Emotion Detection API)
