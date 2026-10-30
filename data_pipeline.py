"""Data loading, validation and feature engineering."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["ingredients_list", "has_risk"]

def load_labeled_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def prepare_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    data = df.copy()
    ingredients = data["ingredients_list"].fillna("").astype(str)
    if "categories_fr" in data.columns:
        category = data["categories_fr"].fillna("").astype(str)
    elif "categories" in data.columns:
        category = data["categories"].fillna("").astype(str)
    else:
        category = pd.Series("", index=data.index)
    text = (ingredients + " " + category).str.replace(r"\s+", " ", regex=True).str.strip()
    valid = text.ne("") & data["has_risk"].notna()
    X = text.loc[valid]
    y = data.loc[valid, "has_risk"].astype(int)
    return X, y, data.loc[valid].copy()
