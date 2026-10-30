import pandas as pd
from src.data_pipeline import prepare_features
from src.predict import predict_text

def test_prepare_features():
    df = pd.DataFrame({"ingredients_list": ["Aqua, Glycerin", "Phenoxyethanol"], "categories_fr": ["cream", "cream"], "has_risk": [0, 1]})
    X, y, clean = prepare_features(df)
    assert len(X) == 2
    assert y.tolist() == [0, 1]

def test_predict_contract():
    class Dummy:
        def predict_proba(self, rows):
            return [[0.2, 0.8]]
    result = predict_text(Dummy(), "Phenoxyethanol")
    assert result["has_risk"] == 1
    assert 0 <= result["risk_probability"] <= 1
