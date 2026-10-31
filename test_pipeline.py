"""Small unit fixtures validate the RF code contract, not project performance."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.features import IngredientPresenceEncoder
from src.predict import explain_text, predict_text


def fitted_unit_pipeline():
    # These tiny fixtures only exercise software behavior. Reported project
    # metrics are computed separately on the real supplied product cohort.
    ingredients = [
        ["aqua", "glycerin"],
        ["aqua", "glycerin", "panthenol"],
        ["aqua", "phenoxyethanol"],
        ["glycerin", "phenoxyethanol"],
    ]
    model = Pipeline([
        ("ingredient_presence", IngredientPresenceEncoder(max_ingredients=10)),
        ("classifier", RandomForestClassifier(n_estimators=12, random_state=42)),
    ])
    return model.fit(ingredients, [0, 0, 1, 1])


def test_predict_contract():
    class Dummy:
        def predict_proba(self, rows):
            return [[0.2, 0.8]]
    result = predict_text(Dummy(), "Phenoxyethanol")
    assert result["has_risk"] == 1
    assert 0 <= result["risk_probability"] <= 1
    assert result["model_score"] == 0.8
    assert "not a safety verdict" in result["message"]


def test_fitted_pipeline_predicts_list_and_literal_consistently():
    model = fitted_unit_pipeline()
    items = ["aqua", "phenoxyethanol"]
    direct = predict_text(model, items)
    literal = predict_text(model, repr(items))
    assert direct["model_score"] == pytest.approx(literal["model_score"])
    assert direct["has_risk"] in (0, 1)


def test_predict_rejects_empty_input_and_invalid_threshold():
    model = fitted_unit_pipeline()
    with pytest.raises(ValueError):
        predict_text(model, "")
    with pytest.raises(ValueError):
        predict_text(model, ["aqua"], threshold=1.0)


def test_local_explanation_matches_actual_drop_ingredient_probability():
    model = fitted_unit_pipeline()
    items = ["aqua", "phenoxyethanol"]
    rows = explain_text(model, items)
    assert rows
    baseline = float(model.predict_proba([items])[0, 1])
    for row in rows:
        term = row["term"].removeprefix("ingredient=")
        dropped = [item for item in items if item != term]
        changed = float(model.predict_proba([dropped])[0, 1])
        assert row["contribution"] == pytest.approx(baseline - changed)
        assert row["unsigned_impurity_importance"] >= 0
        assert "not SHAP" in row["method"]
    assert np.isfinite([row["contribution"] for row in rows]).all()
