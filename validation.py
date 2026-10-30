"""Validation experiments for the Cosmetic Risk Detector."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

COSMETIC_TAGS = [
    "cosmetic-products", "shampoos", "hair-conditioners", "hair-care",
    "shower-gels", "body", "face", "deodorants", "soaps", "sunscreen",
    "makeup", "skin-care", "hand-creams", "body-creams", "facial-creams", "bath"
]

def filter_strict_cosmetics(df: pd.DataFrame) -> pd.DataFrame:
    tags = (df.get("categories_tags", pd.Series("", index=df.index)).fillna("").astype(str)
            + " " + df.get("categories_fr", pd.Series("", index=df.index)).fillna("").astype(str)).str.lower()
    pattern = "|".join(COSMETIC_TAGS)
    return df.loc[tags.str.contains(pattern, regex=True, na=False)].copy()

def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

def run_cross_validation(X, y, folds: int = 5) -> dict:
    scores = cross_validate(
        build_model(), X, y,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=-1,
    )
    return {metric: {"mean": round(float(scores[f"test_{metric}"].mean()), 4),
                     "std": round(float(scores[f"test_{metric}"].std()), 4)}
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]}

def create_validation_report(data_path: str, output_path: str) -> dict:
    df = pd.read_csv(data_path, low_memory=False)
    strict = filter_strict_cosmetics(df)
    ingredients = strict["ingredients_list"].fillna("").astype(str)
    categories = strict.get("categories_fr", pd.Series("", index=strict.index)).fillna("").astype(str)
    X = (ingredients + " " + categories).str.replace(r"\s+", " ", regex=True).str.strip()
    valid = X.ne("") & strict["has_risk"].notna()
    X, y = X.loc[valid], strict.loc[valid, "has_risk"].astype(int)
    report = {
        "strict_cosmetic_rows": int(len(X)),
        "strict_cosmetic_positive_rate": round(float(y.mean()), 4),
        "strict_cosmetic_positive_rows": int(y.sum()),
        "cross_validation": run_cross_validation(X, y),
        "definition": "Category-based scope filter; not a regulatory cosmetic classification.",
    }
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
