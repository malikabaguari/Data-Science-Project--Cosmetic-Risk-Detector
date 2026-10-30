import streamlit as st
from pathlib import Path
from src.predict import load_model, predict_text

st.set_page_config(page_title="Cosmetic Risk Detector", page_icon="🧴", layout="centered")
st.title("Cosmetic Risk Detector")
st.caption("A reference-based ingredient alert — not a medical or toxicological diagnosis.")
model_path = Path("models/cosmetic_risk_model.joblib")
if not model_path.exists():
    st.error("Model not found. Run: python src/train.py")
    st.stop()
model = load_model(model_path)
ingredients = st.text_area("Paste the INCI ingredient list", height=180, placeholder="Aqua, Glycerin, Phenoxyethanol, ...")
category = st.text_input("Optional product category", placeholder="Shampoo, face cream, deodorant...")
if st.button("Analyze", type="primary"):
    if not ingredients.strip():
        st.warning("Please enter an ingredient list.")
    else:
        result = predict_text(model, ingredients, category)
        if result["has_risk"]:
            st.error(f"Reference-based alert — probability: {result['risk_probability']:.1%}")
        else:
            st.success(f"No reference-based alert predicted — probability: {result['risk_probability']:.1%}")
        st.info(result["message"])
st.divider()
st.markdown("**Scope:** the label reflects the supplied reference list. Concentration, exposure, formulation context and personal sensitivity are not modeled.")
