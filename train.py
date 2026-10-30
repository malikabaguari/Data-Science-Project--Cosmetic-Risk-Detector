"""Train, evaluate and persist the Cosmetic Risk Detector model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .data_pipeline import load_labeled_data, prepare_features

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "average_precision": round(average_precision_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

def train(data_path, model_dir):
    df = load_labeled_data(data_path)
    X, y, _ = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]).fit(X_train, y_train)
    out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "cosmetic_risk_model.joblib")
    results = {
        "dataset_rows": int(len(X)),
        "positive_rate": round(float(y.mean()), 6),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "baseline_majority": evaluate(baseline, X_test, y_test),
        "logistic_regression_tfidf": evaluate(model, X_test, y_test),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/products_labeled.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model_dir), indent=2))
