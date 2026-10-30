"""Inference helpers used by the Streamlit app and tests."""
from pathlib import Path
import joblib

def load_model(path="models/cosmetic_risk_model.joblib"):
    return joblib.load(Path(path))

def predict_text(model, ingredients: str, category: str = "") -> dict:
    text = f"{ingredients or ''} {category or ''}".strip()
    probabilities = model.predict_proba([text])
    probability = float(probabilities[0][1])
    return {
        "has_risk": int(probability >= 0.5),
        "risk_probability": round(probability, 4),
        "message": "Reference-based alert: inspect the ingredients and consult an expert if needed." if probability >= 0.5 else "No reference-based alert predicted by this model.",
    }
