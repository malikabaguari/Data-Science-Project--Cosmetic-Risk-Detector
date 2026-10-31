Cosmetic Risk Detector

An end-to-end NLP project for the Ironhack Data Science final project. It includes data acquisition, API evidence, source auditing, cleaning, exploratory data analysis, model comparison, tuning, formula-grouped evaluation, error analysis, an explainable local demo and automated tests.

Project summary

Cosmetic Risk Detector predicts the supplied binary label has_risk from normalized ingredient text. It is an educational screening prototype, not a medical, toxicological or regulatory assessment.

The supplied metadata mentions a reference count of 16 substances, but their exact names, versioned sources and original labeling code are unavailable. Consequently, the project evaluates reproduction of the supplied labels, not independently verified substance presence or cosmetic safety. It does not reconstruct an undocumented reference list.

Business question

Can NLP reproduce the supplied ingredient-based alert labels from ingredient text alone, while keeping repeated formulas out of both sides of the evaluation split?

Categories and brands support data auditing and scope analysis; they are not inputs to the final classifier.

Start with the analysis

•
Complete executed notebook: collection/API, source checks, cleaning, EDA, feature engineering, model comparison, tuning, evaluation and interpretation.

•
Readable notebook HTML: code, tables, charts and executed outputs.

•
Final model metrics.

•
Cleaning report and EDA summary.

•
Saved API sample.

Dataset and cleaning

The project uses supplied Open Beauty Facts snapshots. Some source fields use French-language taxonomy; this does not establish a France-only population. Source country/category metadata is incomplete or noisy, so the full cohort is not presented as an independently verified cosmetic-only dataset.

Stage
Product rows
Supplied raw export
64,237
Supplied intermediate snapshot
17,354
Supplied labeled input, before cleaning
15,621
Low-alignment rows quarantined from modeling
47
Final cleaned modeling cohort
15,574




The input snapshots are historical artifacts. Their upstream filtering and exact original labeling procedure cannot be fully reconstructed from the supplied metadata. The project audits those snapshots and provides a reproducible cleaning/modeling workflow from the labeled input.

Cleaning safely parses ingredient-list literals, normalizes Unicode/HTML/casing/whitespace, validates binary labels and ingredient counts, and checks raw-text/list alignment. Long ingredient lists and prose-like entries are flagged for review rather than automatically deleted. Source snapshots and original ingredient names remain unchanged; no complete synonym mapping is claimed.

Exploratory findings

•
Positive supplied labels: 4,059 products (26.06% of the cleaned cohort).

•
Negative supplied labels: 11,515 products (73.94%).

•
Mean ingredient-list length: 19.53 entries; median: 18.

•
Distinct normalized formula groups: 14,832.

•
Repeated-formula rows beyond the first row in each group: 742.

Formula fingerprints are order-insensitive. Matching formula groups stay together in train/test and cross-validation partitions. Descriptive label rates are not toxicological or category-safety rankings.

See the data dictionary, source metadata, data delivery notes and methodology.

Model selection and preprocessing

Four candidate families are compared using three-fold grouped cross-validation on training data:

1.
Majority-class baseline.

2.
Multinomial Naive Bayes.

3.
Word TF-IDF + logistic regression.

4.
Character TF-IDF + logistic regression.

Selection uses training-CV F1, with a simplicity preference for word logistic regression if it is within 0.01 F1 of the best candidate. The character model won and was tuned over six parameter combinations on training folds.

Final model

A scikit-learn pipeline with:

•
Normalized ingredient text only.

•
TF-IDF character word-boundary features (analyzer="char_wb").

•
Character 3–5-grams, maximum 60,000 features, min_df=3, and sublinear term-frequency scaling.

•
Logistic regression with C=2 and balanced class weights.

The held-out split is the first fold of a shuffled, stratified five-fold StratifiedGroupKFold split, with random_state=42:

•
Training: 12,459 rows.

•
Test: 3,115 rows.

•
Shared normalized formula groups between train and test: 0.

An out-of-fold procedure on training data selected an operating threshold of 0.5612599785, approximately 0.5613, targeting at least 95% training out-of-fold recall. The held-out test did not select the model, parameters or threshold.

Final held-out results

These results concern the tuned final model at the training-selected operating threshold, not an earlier random-split model or an untuned candidate.

Metric
Majority baseline on the same test
Final NLP model
Accuracy
73.93%
96.21%
Precision
0.00%
89.52%
Recall
0.00%
96.80%
F1-score
0.00%
93.02%
ROC-AUC
0.5000
0.9862
Average precision
0.2607
0.9483




The majority baseline has approximately 74% accuracy but detects none of the supplied positive labels. The 73.94% figure in the EDA describes the full cleaned cohort; the 73.93% baseline above describes the held-out test.

Confusion matrix: rows are observed supplied labels; columns are predicted labels.



Predicted 0
Predicted 1
Observed 0
2,211
92
Observed 1
26
786




The model detects 786 of 812 supplied positive test labels, with 26 missed labels and 92 false positives. These are classification outcomes against supplied labels, not clinical safety outcomes.

Evidence: final metrics JSON, classification report and results summary.

Scope-sensitivity analysis

A category-based operational filter identifies 6,785 taxonomy-recognized cosmetic/personal-care rows, including oral-care categories. This is not an official regulatory classification or an independently verified cosmetic-only benchmark.

Five-fold stratified grouped CV of the selected, tuned pipeline on this subset produces:

Metric
Mean
Standard deviation across folds
Recall
96.38%
0.72 percentage points
F1-score
92.72%
0.51 percentage points
ROC-AUC
0.9882
0.0019




This scope check uses the default 0.5 threshold, not the primary operating threshold of 0.5613. It overlaps the original source population and uses a previously selected model family. It is not independent external validation or fully nested model selection.

Historical experiment artifacts may be retained as legacy_*.json; they are not the final performance figures.

Reference registry and deterministic matching

The reference-registry manifest explicitly records the missing exact roster. The exact-match implementation in src/deterministic.py requires documented canonical names and source URLs, and rejects an empty registry.

No deterministic-baseline score is reported because the original registry is unavailable. Recovering and versioning that registry, then comparing deterministic matching with NLP, is a priority next step. A negative model prediction is not proof that a product lacks all relevant substances.

Repository structure

Plain Text


.
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── notebooks/
│   └── 01_complete_project.ipynb    # Complete executed analysis
├── data/
│   ├── processed/                  # Required labeled input and generated cleaned data
│   ├── api/                        # Saved acquisition evidence
│   ├── collection_metadata.json
│   └── reference_registry.json
├── src/
│   ├── collection.py               # CSV/API acquisition
│   ├── cleaning.py                 # Safe parsing, audits and formula fingerprints
│   ├── analysis.py                 # EDA, tables, figures and SQLite
│   ├── experiments.py              # Grouped CV, tuning, threshold selection and evaluation
│   ├── deterministic.py            # Documented-registry matching contract
│   ├── train.py                    # Complete pipeline entry point
│   └── predict.py                  # Inference and local feature contributions
├── app.py                          # Local Streamlit demo
├── models/                         # Generated model and configuration
├── reports/                        # HTML, metrics, audits, tables and figures
├── docs/                           # Methodology, data dictionary, model card and limitations
├── sql/analysis.sql                # Supplementary analytics queries
├── tests/                          # Automated unit/mock tests
├── Makefile
└── Dockerfile



Legacy src/data_pipeline.py and src/validation.py may remain for backward compatibility. The authoritative workflow is the complete notebook and python -m src.train. The current pipeline does not use run_validation.py.

Quick start

The reported results were generated with Python 3.12.3. Run the following from the project root:

Bash


python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt



requirements-lock.txt records the package versions used for the reported execution. Use compatible versions when reproducing the results or regenerating the model.

Place the required supplied labeled input at:

Plain Text


data/processed/products_labeled.csv



The raw and intermediate exports are optional for the source audit. They are not required to run the training entry point from the labeled input.

Run the complete pipeline, tests and local demo

Bash


python -m src.train
python -m pytest -q
python -m streamlit run app.py



The training entry point performs cleaning, EDA and model experiments and regenerates the model artifacts. It accepts --root if an alternative project root is needed; it does not accept --data or --model-dir.

Execute the notebook and export HTML

Bash


python -m ipykernel install --user --name cosmetic-python --display-name "Cosmetic Python"
python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=900 notebooks/01_complete_project.ipynb
python -m jupyter nbconvert --to html --output-dir reports notebooks/01_complete_project.ipynb



Only load trusted joblib model files. Regenerate the model with compatible scikit-learn versions rather than relying on an incompatible saved binary.

API acquisition

Three public product API requests were executed and saved as a JSON snapshot. All returned HTTP 200 and API product status 1. This sample demonstrates real acquisition; it is not the origin of the historical training labels.

To collect a new sample:

Bash


python -m src.collection --barcodes 1785 1959 2337 --output data/api/new_sample.json



Public API content can change. A new API sample or bulk export is not guaranteed to reproduce the historical snapshots or their unavailable upstream labels. Bulk downloading is explicit opt-in.

See the collection documentation.

Verification and limitations

•
17 automated unit/mock tests passed.

•
The complete notebook was executed with embedded results.

•
The local Streamlit application passed an AppTest smoke check with a real dataset row.

•
Docker packaging is provided, but Docker execution was not verified.

•
No hosted production application, OCR implementation, complete synonym resolution, independent clinical labels or calibrated personal-health probabilities are claimed.

Character coefficients and local TF-IDF contributions explain the classifier's lexical behavior. They are not ingredient toxicity scores. The project does not model concentrations, individual sensitivity or exposure conditions.

See the model card and limitations.

Data delivery and submission

The supplied labeled CSV is approximately 62.81 MiB. If sharing it, preserve the required attribution and verify redistribution conditions. Raw/intermediate exports and generated model binaries are excluded from ordinary Git commits by .gitignore. If a required dataset is not distributed with the repository, provide an authorized download route and exact placement instructions; do not describe a data-less repository as immediately runnable.

Keep the supplied relative paths when uploading files to GitHub so notebook imports, commands and documentation links continue to work. Verify that the default branch contains the complete current version before submitting its URL.

The English presentation is delivered separately: ten slides using the final 15,574-row cohort and 96.80% held-out recall. Speaker notes and private preparation material are not part of the technical public repository.

Responsible-use statement

has_risk is a supplied proxy label with incomplete upstream provenance. Prediction of that label does not establish substance presence, toxicity, regulatory compliance or personal safety. Use this project as an auditable educational NLP workflow, with its limitations visible.

