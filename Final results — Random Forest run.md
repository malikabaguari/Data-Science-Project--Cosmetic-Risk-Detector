# **Final results — Random Forest run**

This document is generated from [`reports/final_metrics.json`](final_metrics.json). It is the readable companion to the machine-readable final metrics and must be regenerated after any new training run.

## **Run identity and dataset boundary**

| **Item** | **Recorded value** |
|---|---|
| Selected model | **Random Forest** |
| Cleaned modeling cohort | 15,574 rows (4,059 positive / 11,515 negative supplied labels) |
| Split | 12,459 train / 3,115 test; zero normalized formula-group overlap |
| Tuning criterion | Training grouped-CV average precision |
| Final threshold rule | Maximize F2 subject to training OOF recall ≥ 0.95 |
| Decision threshold | **0.438665** |
| Masking | Not applied (`false`) |
| Benchmark CV leader | Balanced Logistic Regression |

The exact 16-substance roster and original label-generation code are unavailable. Results are agreement metrics for the supplied target; they are not independent safety, toxicity, or regulatory results.

## **Training selection evidence**

- Training grouped-CV average precision: **0.9990**.
- Training grouped-CV F1: **99.86%**.

| **Parameter** | **Recorded value** |
|---|---|
| `class_weight` | `balanced_subsample` |
| `max_features` | `0.2` |
| `min_samples_leaf` | `2` |
| `n_estimators` | `200` |

## **Grouped internal holdout**

### **Default threshold (0.5)**

| **Metric** | **Value** |
|---|---:|
| Accuracy | **99.90%** |
| Precision | **100.00%** |
| Recall | **99.63%** |
| F1 | **99.81%** |
| F2 | **99.70%** |
| ROC-AUC | **0.9990** |
| Average precision | **0.9985** |

### **Training-selected threshold (0.438665)**

| **Metric** | **Value** |
|---|---:|
| Accuracy | **99.90%** |
| Precision | **100.00%** |
| Recall | **99.63%** |
| F1 | **99.81%** |
| F2 | **99.70%** |
| ROC-AUC | **0.9990** |
| Average precision | **0.9985** |

Rows are supplied labels; columns are model predictions.

| | **Predicted 0** | **Predicted 1** |
|---|---:|---:|
| **Observed 0** | 2,303 | 0 |
| **Observed 1** | 3 | 809 |

The primary holdout is internal. It was not used to choose the model family, its parameters, or the operating threshold, and it is not external validation.

## **Recognized-scope sensitivity**

Operational recognized scope: **6,785 rows**. **Recognized-category sensitivity reuses/overlaps the primary source cohort and is not independent external validation; metrics use default threshold 0.5.**

| **Metric** | **Mean** | **Fold standard deviation** |
|---|---:|---:|
| Accuracy | **99.79%** | 0.23% |
| Precision | **100.00%** | 0.00% |
| Recall | **99.21%** | 0.90% |
| F1 | **99.60%** | 0.46% |
| F2 | **99.36%** | 0.72% |
| ROC-AUC | **0.9991** | 0.0007 |
| Average precision | **0.9986** | 0.0013 |

## **Interpretation and evidence**

Random Forest MDI importance is non-negative and does not encode direction, causality, toxicity, or safety. Local app explanations are drop-one ingredient probability perturbations, not SHAP. No masking was applied, no deterministic registry matching score is reported, and the application is local only (no hosted service/OCR/full regulatory workflow).

- [Notebook execution summary](execution_summary.json): 43 notebook cells, 24 executed code cells, 0 error outputs, 15 embedded PNG figures.
- [Automated test output](test_results.txt): 23 passed
- [Model card](../docs/model_card.md) and [limitations](../docs/limitations.md)
