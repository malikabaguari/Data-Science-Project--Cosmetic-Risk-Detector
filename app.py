from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import explain_text, load_config, load_model, predict_text

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Cosmetic Risk Detector", layout="centered")
st.title("Cosmetic Risk Detector")
st.write("An educational classifier of a supplied ingredient-based label. Not a medical, safety or regulatory assessment.")
model_path = ROOT / "models/cosmetic_risk_model.joblib"
config_path = ROOT / "models/model_config.json"
if not model_path.exists() or not config_path.exists():
    st.warning("Train the model first: python -m src.train --root .")
    st.stop()


@st.cache_resource
def artifacts():
    return load_model(model_path), load_config(config_path)


model, config = artifacts()
ingredients = st.text_area(
    "Paste an ingredient list",
    height=180,
    placeholder="Aqua, Glycerin, Phenoxyethanol",
    help="Use a comma- or semicolon-separated list, or a safe literal Python-style list. This convenience parser does not validate INCI or regulatory terminology.",
)
if st.button("Analyze", type="primary"):
    if not ingredients.strip():
        st.warning("Enter an ingredient list.")
    else:
        try:
            result = predict_text(model, ingredients, threshold=config["decision_threshold"])
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.subheader("Supplied-label alert predicted" if result["has_risk"] else "No supplied-label alert predicted")
            st.metric("Model score for supplied label", f"{result['model_score']:.1%}")
            st.caption(f"Operating threshold selected on grouped training OOF data: {config['decision_threshold']:.3f}")
            st.info(result["message"])
            explanations = explain_text(model, ingredients)
            if explanations:
                st.write("Present-feature drop-one checks")
                st.dataframe(pd.DataFrame(explanations), hide_index=True)
                st.caption("Each probability change removes one recognized present ingredient from this formula. These context-dependent checks are not SHAP values, causal effects, or a safety registry.")
            else:
                st.warning("No recognized vocabulary features. Treat this as unsupported input.")
st.divider()
st.write("The source metadata mentions 16 reference substances, but the exact registry and labeling code are unavailable. Concentration, exposure and personal sensitivity are not modeled. No toxicity or regulatory validation is claimed.")
