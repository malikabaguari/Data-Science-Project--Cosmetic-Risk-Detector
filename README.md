# Cosmetic Risk Detector

![Python](https://img.shields.io/badge/Python-3.11-blue) ![ML](https://img.shields.io/badge/Model-TF--IDF%20%2B%20Logistic%20Regression-red) ![Status](https://img.shields.io/badge/Status-Ironhack%20final%20project-informational)

## Project summary

Cosmetic Risk Detector is an end-to-end NLP project that predicts whether a product contains at least one ingredient included in a predefined reference list. It is designed as an explainable screening prototype for cosmetic ingredient lists. It is **not** a medical, toxicological or regulatory diagnosis.

## Business question

Can we use ingredient text and product category to identify products labeled positive under a reference-based ingredient rule?

## Dataset

The data comes from Open Beauty Facts France.

- Raw export: 64,237 products
- Intermediate products with ingredients: 17,354
- Final labeled dataset: 15,621 products
- Positive label: 4,059 products (25.98%)
- Target: `has_risk`

See [`docs/data_dictionary.md`](docs/data_dictionary.md) and [`data/collection_metadata.json`](data/collection_metadata.json).

## Model and results

The production baseline is a scikit-learn pipeline with TF-IDF unigrams/bigrams and balanced logistic regression. On a stratified held-out test set of 3,125 rows:

| Metric | Majority baseline | NLP model |
|---|---:|---:|
| Accuracy | 74.02% | 94.88% |
| Precision | 0.00% | 86.63% |
| Recall | 0.00% | 94.95% |
| F1-score | 0.00% | 90.60% |
| ROC-AUC | 0.500 | 0.982 |
| Average precision | 0.260 | 0.943 |

The result is reproducible with `python -m src.train`.

## Methodological robustness checks

The repository also includes a stricter category-based cosmetic subset and a 5-fold stratified cross-validation experiment:

```bash
python run_validation.py
```

The current category filter selects 5,671 rows, with a positive rate of approximately 29.85%. This is a scope filter based on product categories, not an official regulatory definition of cosmetics.

The exact names and provenance of the 16 reference substances are not present in the supplied collection metadata. This is documented in [`docs/reference_substances.md`](docs/reference_substances.md). A deterministic matching baseline should be added only after that versioned registry is supplied; reconstructing it manually would create undocumented assumptions.

## Repository structure

```text
.
├── app.py                         # Streamlit demo
├── data/processed/                # Labeled input dataset
├── docs/                          # Data dictionary, methodology, limitations
├── models/                        # Generated model artifacts (not committed by default)
├── reports/                       # Notebook and final presentation PDF
├── sql/analysis.sql               # Business/analytics queries
├── src/data_pipeline.py           # Loading and feature preparation
├── src/train.py                   # Training and evaluation
├── src/predict.py                 # Inference contract
├── src/validation.py              # Strict scope and cross-validation experiments
├── run_validation.py              # Reproducible robustness checks
└── tests/                         # Unit tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --data data/processed/products_labeled.csv --model-dir models
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

## GitHub upload recommendation

The processed CSV is about 63 MB. GitHub can accept it, but Git LFS is recommended for a clean repository:

```bash
git lfs install
git add .gitattributes data/processed/products_labeled.csv
```

If dataset redistribution is not permitted, keep only the data dictionary and collection metadata in the public repository and document the local data path.

## Deliverables

- Reproducible notebook: [`reports/cosmetic_risk_detector.ipynb`](reports/cosmetic_risk_detector.ipynb)
- English final presentation: [`reports/Cosmetic_Risk_Detector_Final_Presentation.pdf`](reports/Cosmetic_Risk_Detector_Final_Presentation.pdf)
- Training pipeline, inference code, tests and Streamlit application
- Strict cosmetic-scope experiment and cross-validation report

## Responsible-use statement

`has_risk` is a proxy label based on a supplied reference list of 16 substances. Presence does not equal toxicity or personal danger. Concentration, exposure, formulation context and individual sensitivity are not modeled.
