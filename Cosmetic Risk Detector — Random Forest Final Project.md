# **Cosmetic Risk Detector — Random Forest Final Project**

**Ironhack Data Science · ingredient-list alert-label classification · educational prototype**

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Model](https://img.shields.io/badge/Model-Random%20Forest-228B22) ![Tests](https://img.shields.io/badge/Tests-23%20passed-brightgreen) ![Status](https://img.shields.io/badge/Status-Ironhack%20final%20project-informational)

| **Navigate** | **Core evidence** | **Project documentation** |
|---|---|---|
| [Start here](START_HERE.md) | [Final metrics](reports/final_metrics.json) | [Methodology](docs/methodology.md) |
| [Executed notebook](notebooks/01_random_forest_project.ipynb) | [Final results](reports/FINAL_RESULTS.md) | [Model card](docs/model_card.md) |
| [Run tests](reports/test_results.txt) | [Execution summary](reports/execution_summary.json) | [Limitations](docs/limitations.md) |

> **Scope boundary:** This project predicts the supplied `has_risk` label from ingredient-list features. It is **not** a cosmetic-safety certificate, toxicity assessment, medical tool, or regulatory determination.

---

## **Project question**

Can a grouped, ingredient-based Random Forest reproduce the supplied binary alert label while ensuring exact normalized formulas do not appear on both sides of the internal train/test split?

The supplied metadata records a reference count of **16 substances**, but not their exact roster, sources, matching rules, or original labeling code. Therefore this project evaluates prediction of the **supplied label** only. It does not reconstruct an undocumented registry, report a deterministic matching score, or claim substance-level verification.

## **Data, cleaning, and split integrity**

| **Stage** | **Rows** | **What it means** |
|---|---:|---|
| Supplied raw export | 64,237 | Historical source snapshot |
| Supplied interim export | 17,354 | Intermediate supplied snapshot |
| Supplied labeled input | 15,621 | Input before reproducible cleaning |
| Cleaning quarantine | 47 | Low raw/list alignment; not silently rewritten |
| Cleaned modeling cohort | **15,574** | **4,059 positive / 11,515 negative labels** |

Cleaning safely parses ingredient-list literals, normalizes Unicode/HTML/casing/whitespace, validates the binary label and ingredient counts, and quarantines low-alignment rows. It retains source evidence and does not claim a full synonym map. The final cohort has **14,832** order-insensitive normalized formula groups and **742** extra rows belonging to repeated formulas.

The grouped internal split has **12,459** training rows and **3,115** held-out test rows, with **zero** normalized formula-group overlap. The primary test is an internal holdout, **not external validation**. A category-recognized operational scope contains 6,785 rows; it is not a legal or regulatory population definition.

## **Model and feature design**

The final pipeline is **Random Forest**. Its family was chosen for the ensemble-project focus; its **hyperparameters** were selected using training-only grouped CV with **average precision** as the tuning criterion. The initial candidate benchmark CV leader is reported transparently as **Balanced Logistic Regression**. Initial model configurations and the later RF tuning are separate experiments, not a fully nested cross-family winner-selection study.

Features are intentionally narrow and auditable:

- The **top 1,500 ingredients learned on each training fold**, represented as binary ingredient-presence columns.
- **`log1p(ingredient_count)`**.
- **No categories, brands, product names, or external registry matches** as model inputs.

Three-fold grouped training CV keeps formula groups intact. Random Forest out-of-fold probabilities select the decision threshold that **maximizes F2 subject to recall ≥ 0.95** on training OOF predictions. No masking is applied (`masking_applied: false`).

### **Recorded training selection evidence**

- Training grouped-CV average precision: **0.9990**.
- Training grouped-CV F1: **99.86%**.

### **Tuned parameters**

| **Parameter** | **Recorded value** |
|---|---|
| `class_weight` | `balanced_subsample` |
| `max_features` | `0.2` |
| `min_samples_leaf` | `2` |
| `n_estimators` | `200` |

## **Internal held-out results**

Both operating points are reported. The default 0.5 threshold is included for context; the primary operating point is the training-selected threshold **0.438665**. The held-out test did not choose the model, its parameters, or the threshold.

| **Metric** | **Default threshold (0.5)** | **Training-selected threshold** |
|---|---:|---:|
| Accuracy | 99.90% | **99.90%** |
| Precision | 100.00% | **100.00%** |
| Recall | 99.63% | **99.63%** |
| F1 | 99.81% | **99.81%** |
| F2 | 99.70% | **99.70%** |
| ROC-AUC | 0.9990 | **0.9990** |
| Average precision | 0.9985 | **0.9985** |

Rows are supplied labels; columns are model predictions.

| | **Predicted 0** | **Predicted 1** |
|---|---:|---:|
| **Observed 0** | 2,303 | 0 |
| **Observed 1** | 3 | 809 |

The metrics measure agreement with supplied labels, not health outcomes or independent regulatory truth. Full dynamic results: [JSON](reports/final_metrics.json) and [readable summary](reports/FINAL_RESULTS.md).

## **Scope-sensitivity check**

The category-recognized scope has **6,785 rows** in this run. If reported, it uses five-fold grouped CV and is only an overlapping-source sensitivity analysis. **Recognized-category sensitivity reuses/overlaps the primary source cohort and is not independent external validation; metrics use default threshold 0.5.**

| **Metric** | **Mean** | **Fold standard deviation** |
|---|---:|---:|
| Accuracy | **99.79%** | 0.23% |
| Precision | **100.00%** | 0.00% |
| Recall | **99.21%** | 0.90% |
| F1 | **99.60%** | 0.46% |
| F2 | **99.36%** | 0.72% |
| ROC-AUC | **0.9991** | 0.0007 |
| Average precision | **0.9986** | 0.0013 |

## **Interpretation and responsible use**

Random Forest mean-decrease-in-impurity (MDI) feature importance is non-negative. It ranks split-usage importance; it does **not** provide direction, ingredient toxicity, causation, or a safety endorsement. Local explanations use drop-one ingredient probability perturbations, **not SHAP**. The Streamlit application is local only; no hosted deployment, OCR, or full regulatory workflow is implemented.

See [the model card](docs/model_card.md) and [limitations](docs/limitations.md) before interpreting a prediction.

## **Quick start**

Use Python 3.12 and run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import gzip; from pathlib import Path; p=Path('data/processed/products_labeled.csv'); p.write_bytes(gzip.decompress(p.with_suffix('.csv.gz').read_bytes())) if not p.exists() else None"
python -m src.train
python -m pytest -q
python -m streamlit run app.py
```

The supplied labeled input belongs at `data/processed/products_labeled.csv`. A smaller **single-file gzip** copy, `data/processed/products_labeled.csv.gz`, contains exactly the same bytes after decompression and can be used for distribution. If you received only that copy, run the command below **before training or opening the notebook**:

```bash
python -c "import gzip; from pathlib import Path; p=Path('data/processed/products_labeled.csv'); p.write_bytes(gzip.decompress(p.with_suffix('.csv.gz').read_bytes())) if not p.exists() else None"
```

Raw and interim snapshots support optional source auditing; the labeled input is required to reproduce training. Do not publish both large uncompressed data and its compressed duplicate unnecessarily. To execute the notebook and optionally export readable HTML:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=900 notebooks/01_random_forest_project.ipynb
python -m jupyter nbconvert --to html --output-dir reports notebooks/01_random_forest_project.ipynb
```

Automated-test evidence records **23 passed**; see the saved test output. The run's [notebook execution summary](reports/execution_summary.json) records: 43 notebook cells, 24 executed code cells, 0 error outputs, 15 embedded PNG figures.

## **Repository essentials**

```text
notebooks/01_random_forest_project.ipynb  end-to-end executed analysis
src/train.py                               reproducible pipeline entry point
src/experiments.py                         grouped CV, benchmarks, RF tuning, thresholding
data/processed/products_labeled.csv        supplied labeled input
reports/final_metrics.json                 machine-readable final metrics
reports/test_results.txt                   saved automated-test evidence
docs/                                      methodology, model card, limits, requirements map
```

For the minimal submission path, open [START_HERE.md](START_HERE.md). The project board is a current evidence map, not a fabricated history: [docs/project_board.md](docs/project_board.md).

## **Submission boundary**

The Ironhack Case Study 1 brief calls for a new GitHub repository, working code, a notebook and required files, a README, a slides URL, and the repository URL in Student Portal. Slides/PPTX/PDF are delivered separately; this repository intentionally contains **no broken presentation placeholder link**, speaker notes, or jury Q&A.

Before submission, review and understand the assisted code, reproduce the key run, follow the cohort's authorship/AI-use policy, verify the intended default branch yourself, and enter the real repository and slides URLs in Student Portal. File delivery does **not** claim that GitHub has already been updated.
