"""Inference for supplied-label scores, never a calibrated personal health risk."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .features import normalize_ingredient_input


def load_model(path="models/cosmetic_risk_model.joblib"):
    """Load only trusted local artifacts: joblib/pickle can execute code."""
    return joblib.load(Path(path))


def load_config(path="models/model_config.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_text(model, ingredients, category: str = "", threshold: float = 0.5) -> dict:
    """Score an explicit local ingredient list with the persisted pipeline.

    ``category`` remains part of the public signature for backward
    compatibility. It is not used by the ingredient-only final model.
    """
    if not normalize_ingredient_input(ingredients):
        raise ValueError("Ingredient text must contain a list, a literal list, or comma/semicolon-separated ingredients.")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must lie strictly between zero and one.")
    score = float(model.predict_proba([ingredients])[0][1])
    return {
        "has_risk": int(score >= threshold),
        "model_score": score,
        "risk_probability": round(score, 4),
        "decision_threshold": threshold,
        "message": (
            "Predicted supplied-label alert; not a safety verdict."
            if score >= threshold
            else "No supplied-label alert predicted; safety is not established."
        ),
    }


def explain_text(model, text, top_n: int = 8) -> list[dict]:
    """Return drop-one present-feature probability changes, not SHAP values.

    Each listed feature is present in local input and ranked by fitted unsigned
    RF impurity importance. ``contribution`` is the base score minus the score
    after only that ingredient is removed. Values are context-dependent,
    associative diagnostics and not causal effects, safety claims, or a
    substance registry.
    """
    if top_n < 1 or not hasattr(model, "named_steps"):
        return []
    encoder = model.named_steps.get("ingredient_presence")
    classifier = model.named_steps.get("classifier")
    if encoder is None or classifier is None or not hasattr(classifier, "feature_importances_"):
        return []
    ingredients = normalize_ingredient_input(text)
    if not ingredients:
        return []
    vocabulary = getattr(encoder, "vocabulary_", {})
    importances = np.asarray(classifier.feature_importances_)
    ranked: list[tuple[str, float]] = []
    for ingredient in sorted(set(ingredients)):
        index = vocabulary.get(ingredient)
        if index is not None:
            ranked.append((ingredient, float(importances[index])))
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    base_score = float(model.predict_proba([ingredients])[0][1])
    explanations: list[dict] = []
    for ingredient, impurity_importance in ranked[:top_n]:
        dropped = [value for value in ingredients if value != ingredient]
        dropped_score = float(model.predict_proba([dropped])[0][1])
        change = base_score - dropped_score
        explanations.append(
            {
                "term": f"ingredient={ingredient}",
                "contribution": float(change),
                "drop_one_probability_change": float(change),
                "base_model_score": base_score,
                "score_without_term": dropped_score,
                "unsigned_impurity_importance": impurity_importance,
                "method": "drop-one-present-feature (context-dependent; not SHAP)",
            }
        )
    return explanations
