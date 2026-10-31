"""Grouped, leakage-aware Random Forest experiments for supplied labels.

All feature fitting happens inside sklearn pipelines and therefore occurs only
on each training fold. The held-out test partition is never used to tune a
model or select the operating threshold.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from .features import IngredientPresenceEncoder

RANDOM_STATE = 42
MAX_INGREDIENT_FEATURES = 1500
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _progress(root: Path, message: str) -> None:
    """Emit and persist concise checkpoints for a potentially minutes-long run."""
    text = f"[{_timestamp()}] {message}"
    print(text, flush=True)
    log = root / "reports" / "training_progress.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("ingredient_presence", IngredientPresenceEncoder(max_ingredients=MAX_INGREDIENT_FEATURES)),
            ("classifier", classifier),
        ]
    )


def model_candidates() -> dict[str, Pipeline]:
    """Candidate classifiers evaluated on the identical fitted feature scheme."""
    return {
        "Majority baseline": _pipeline(DummyClassifier(strategy="most_frequent")),
        "Balanced Logistic Regression": _pipeline(
            LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": _pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=2,
            )
        ),
    }


def score_predictions(y, scores, threshold: float = 0.5) -> dict[str, Any]:
    """Return the shared report-schema metric dictionary from scores and a cutoff."""
    y_true = np.asarray(y, dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def _metric_row(name: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    return {"model": name, **{key: value for key, value in evaluation.items() if key != "confusion_matrix"}}


def _choose_f2_threshold(y, oof_scores, minimum_recall: float = 0.95) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Use training OOF scores only to maximize F2 subject to a recall floor."""
    precision, recall, thresholds = precision_recall_curve(y, oof_scores)
    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]
    f2_at_threshold = (5 * precision_at_threshold * recall_at_threshold) / (
        4 * precision_at_threshold + recall_at_threshold + 1e-15
    )
    eligible = np.flatnonzero(recall_at_threshold >= minimum_recall)
    if eligible.size == 0:
        selected_index = int(np.argmax(f2_at_threshold))
        selection_status = "No OOF threshold met the recall target; selected maximum OOF F2."
    else:
        eligible_f2 = f2_at_threshold[eligible]
        best_f2 = eligible_f2.max()
        tied = eligible[np.isclose(eligible_f2, best_f2)]
        selected_index = int(tied[np.argmax(thresholds[tied])])
        selection_status = "Maximized OOF F2 among thresholds with OOF recall at least 0.95."
    chosen_threshold = float(thresholds[selected_index])
    details = {
        "training_oof_precision": float(precision_at_threshold[selected_index]),
        "training_oof_recall": float(recall_at_threshold[selected_index]),
        "training_oof_f2": float(f2_at_threshold[selected_index]),
        "status": selection_status,
    }
    return chosen_threshold, details, thresholds, precision_at_threshold, recall_at_threshold


def _save_figures(
    figures: Path,
    y_test,
    test_scores: np.ndarray,
    tuned_metrics: dict[str, Any],
    threshold: float,
    thresholds: np.ndarray,
    oof_precision: np.ndarray,
    oof_recall: np.ndarray,
    feature_importances: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        tuned_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Reds", cbar=False,
        xticklabels=["0", "1"], yticklabels=["0", "1"],
    )
    plt.xlabel("Predicted supplied label")
    plt.ylabel("Observed supplied label")
    plt.title("Grouped held-out confusion matrix")
    plt.tight_layout()
    plt.savefig(figures / "10_confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, test_scores)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="#d9281c", label=f"ROC-AUC = {tuned_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Grouped held-out ROC curve")
    plt.tight_layout()
    plt.savefig(figures / "11_roc_curve.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color="#d9281c", label=f"AP = {tuned_metrics['average_precision']:.3f}")
    plt.axhline(float(np.mean(y_test)), linestyle="--", color="grey", label="Class prevalence")
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Grouped held-out precision-recall curve")
    plt.tight_layout()
    plt.savefig(figures / "12_precision_recall.png", dpi=150)
    plt.close()

    f2 = (5 * oof_precision * oof_recall) / (4 * oof_precision + oof_recall + 1e-15)
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, oof_precision, label="OOF precision")
    plt.plot(thresholds, oof_recall, label="OOF recall")
    plt.plot(thresholds, f2, label="OOF F2")
    plt.axvline(threshold, color="black", linestyle="--", label=f"Selected {threshold:.3f}")
    plt.axhline(0.95, color="grey", linestyle=":", label="Recall target 0.95")
    plt.xlabel("Threshold")
    plt.ylabel("OOF metric")
    plt.title("Threshold selected using training OOF probabilities only")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figures / "13_threshold_selection.png", dpi=150)
    plt.close()

    # MDI values are unsigned impurity reductions and not coefficient signs,
    # causal effects, or a toxicology/regulatory ranking.
    selected = feature_importances.head(20).sort_values("importance")
    plt.figure(figsize=(10, 7))
    plt.barh(selected["term"], selected["importance"], color="#d9281c")
    plt.title("Random Forest impurity importance (unsigned; associative only)")
    plt.xlabel("Mean decrease in impurity importance")
    plt.tight_layout()
    plt.savefig(figures / "14_feature_importance.png", dpi=150)
    plt.close()


def _cv_summary(result: dict[str, np.ndarray], name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"model": name}
    for metric in SCORING:
        values = result[f"test_{metric}"]
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def run_experiments(data: pd.DataFrame, root, folds: int = 3):
    """Run grouped evaluation and export final RF artifacts.

    A fresh deterministic run can take several minutes because the final family
    is tuned and re-fit with grouped OOF prediction. ``data`` must contain the
    cleaned ``parsed_ingredients``, ``has_risk``, and ``formula_group`` fields.
    """
    root = Path(root)
    tables = root / "reports" / "tables"
    figures = root / "reports" / "figures"
    models = root / "models"
    for directory in (tables, figures, models):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "reports" / "training_progress.log").write_text("", encoding="utf-8")

    required = {"parsed_ingredients", "has_risk", "formula_group", "code"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"run_experiments requires cleaned columns: {missing}")
    if folds < 2:
        raise ValueError("folds must be at least 2")

    X = data["parsed_ingredients"]
    y = data["has_risk"].astype(int)
    groups = data["formula_group"].astype(str)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(outer.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
    group_overlap = len(set(groups_train) & set(groups_test))
    assert group_overlap == 0, "Formula leakage detected between grouped train and test partitions"
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    _progress(root, f"Prepared deterministic grouped split: train={len(train_idx)}, test={len(test_idx)}, overlap={group_overlap}.")
    candidates = model_candidates()
    cv_rows: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        _progress(root, f"Starting {folds}-fold training-only grouped CV for {name}.")
        result = cross_validate(
            estimator, X_train, y_train, groups=groups_train, cv=cv, scoring=SCORING,
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        cv_rows.append(_cv_summary(result, name))
        _progress(root, f"Completed training-only grouped CV for {name}.")
    cv_table = pd.DataFrame(cv_rows).sort_values("average_precision_mean", ascending=False, kind="stable").reset_index(drop=True)
    cv_table.to_csv(tables / "model_comparison_cv.csv", index=False)
    benchmark_leader = str(cv_table.iloc[0]["model"])

    # The project final is deliberately the requested Random Forest family. The
    # comparison table records whether another candidate led its CV benchmark.
    rf_base = model_candidates()["Random Forest"]
    compact_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_features": ["sqrt", 0.2],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__class_weight": ["balanced_subsample"],
    }
    _progress(root, "Starting four-setting Random Forest AP tuning on training folds only.")
    search = GridSearchCV(
        rf_base,
        param_grid=compact_grid,
        cv=cv,
        scoring=SCORING,
        refit="average_precision",
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train, groups=groups_train)
    search_frame = pd.DataFrame(search.cv_results_)
    useful_search_columns = [
        "params", "mean_test_average_precision", "std_test_average_precision",
        "mean_test_f1", "std_test_f1", "mean_test_f2", "std_test_f2", "rank_test_average_precision",
    ]
    search_frame[useful_search_columns].sort_values("rank_test_average_precision").to_csv(
        tables / "hyperparameter_search.csv", index=False
    )
    _progress(root, "Completed Random Forest AP tuning.")

    best_index = int(search.best_index_)
    best_params = {key.replace("classifier__", ""): value for key, value in search.best_params_.items()}
    best_training_cv_ap = float(search.cv_results_["mean_test_average_precision"][best_index])
    best_training_cv_f1 = float(search.cv_results_["mean_test_f1"][best_index])
    tuned_rf = search.best_estimator_

    _progress(root, "Generating 3-fold grouped out-of-fold Random Forest probabilities for threshold selection.")
    oof_scores = cross_val_predict(
        clone(tuned_rf), X_train, y_train, groups=groups_train, cv=cv,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    threshold, threshold_details, threshold_values, oof_precision, oof_recall = _choose_f2_threshold(y_train, oof_scores, 0.95)
    _progress(root, f"Selected OOF-only threshold={threshold:.6f}; OOF recall={threshold_details['training_oof_recall']:.4f}.")

    _progress(root, "Fitting final tuned Random Forest on the complete training partition.")
    final_model = clone(tuned_rf).fit(X_train, y_train)
    test_scores = final_model.predict_proba(X_test)[:, 1]
    default_metrics = score_predictions(y_test, test_scores, threshold=0.5)
    selected_threshold_metrics = score_predictions(y_test, test_scores, threshold=threshold)

    test_rows: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        _progress(root, f"Fitting {name} once on training partition for held-out comparison.")
        fitted = clone(estimator).fit(X_train, y_train)
        test_rows.append(_metric_row(name, score_predictions(y_test, fitted.predict_proba(X_test)[:, 1], threshold=0.5)))
    pd.DataFrame(test_rows).sort_values("average_precision", ascending=False, kind="stable").to_csv(
        tables / "model_comparison_test.csv", index=False
    )

    manifest = data[["code", "formula_group", "has_risk"]].copy()
    manifest["split"] = "train"
    manifest.loc[manifest.index[test_idx], "split"] = "test"
    manifest.to_csv(tables / "split_manifest.csv", index=False)

    prediction_columns = [column for column in ["code", "product_name", "category_label", "parsed_ingredients", "has_risk"] if column in data.columns]
    test_predictions = data.iloc[test_idx][prediction_columns].copy()
    test_predictions["model_score"] = test_scores
    test_predictions["prediction_default_0_5"] = (test_scores >= 0.5).astype(int)
    test_predictions["prediction_training_selected_threshold"] = (test_scores >= threshold).astype(int)
    test_predictions["decision_threshold"] = threshold
    test_predictions.to_csv(tables / "test_predictions.csv", index=False)
    errors = test_predictions.loc[
        test_predictions["has_risk"].ne(test_predictions["prediction_training_selected_threshold"])
    ].copy()
    errors["error_type"] = np.where(
        errors["has_risk"].eq(1), "false_negative", "false_positive"
    )
    errors.to_csv(tables / "prediction_errors.csv", index=False)

    feature_names = final_model.named_steps["ingredient_presence"].get_feature_names_out()
    importances = final_model.named_steps["classifier"].feature_importances_
    feature_importances = pd.DataFrame({"term": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False, kind="stable"
    ).reset_index(drop=True)
    feature_importances.to_csv(tables / "feature_importance.csv", index=False)
    _save_figures(
        figures, y_test, test_scores, selected_threshold_metrics, threshold,
        threshold_values, oof_precision, oof_recall, feature_importances,
    )

    strict = data.loc[data.get("scope_status", pd.Series("", index=data.index)).eq("recognized_cosmetic_or_personal_care")].copy()
    sensitivity: dict[str, Any] | None = None
    if len(strict) and strict["has_risk"].nunique() == 2:
        _progress(root, f"Starting 5-fold grouped overlapping-scope sensitivity analysis ({len(strict)} recognized rows).")
        sensitivity_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        sensitivity_result = cross_validate(
            clone(final_model), strict["parsed_ingredients"], strict["has_risk"].astype(int),
            groups=strict["formula_group"].astype(str), cv=sensitivity_cv, scoring=SCORING,
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        sensitivity = {
            "threshold": 0.5,
            "metrics": {
                metric: {"mean": float(np.mean(sensitivity_result[f"test_{metric}"])), "std": float(np.std(sensitivity_result[f"test_{metric}"]))}
                for metric in SCORING
            },
        }
        _progress(root, "Completed scope sensitivity analysis.")

    classifier_path = models / "cosmetic_risk_model.joblib"
    joblib.dump(final_model, classifier_path)
    config = {
        "model_name": "Random Forest",
        "classifier_path": str(classifier_path.name),
        "feature_scheme": "top training-document-frequency ingredient-presence features plus log1p ingredient-list length",
        "max_ingredient_features": MAX_INGREDIENT_FEATURES,
        "decision_threshold": threshold,
        "threshold_selection": "Training 3-fold grouped OOF Random Forest probabilities only; maximize F2 subject to recall >= 0.95.",
        "best_params": best_params,
        "interpretation": "Model score for a supplied label, not a calibrated health-risk probability or safety verdict.",
        "raw_input_caveat": "Local comma/semicolon parsing is a convenience input format, not validated INCI or regulatory parsing.",
    }
    (models / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = {
        "primary_clean_rows": int(len(data)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "formula_group_overlap_train_test": int(group_overlap),
        "split_type": "First split from StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "selected_model": "Random Forest",
        "best_params": best_params,
        "best_training_cv_average_precision": best_training_cv_ap,
        "best_training_cv_f1": best_training_cv_f1,
        "decision_threshold": threshold,
        "threshold_selection": "Training 3-fold grouped OOF Random Forest probabilities only; maximize F2 subject to recall >= 0.95.",
        "threshold_target_training_oof_recall": 0.95,
        "threshold_selection_details": threshold_details,
        "test_default_threshold": default_metrics,
        "test_training_selected_threshold": selected_threshold_metrics,
        "scope_sensitivity_rows": int(len(strict)),
        "scope_sensitivity_5fold_grouped_cv": sensitivity,
        "scope_note": "Recognized-category sensitivity reuses/overlaps the primary source cohort and is not independent external validation; metrics use default threshold 0.5.",
        "label_provenance_status": "Ingredient-derived supplied labels were preserved. Source metadata mentions 16 reference substances, but the exact registry and label-generation code were unavailable; no toxicity or regulatory validation is claimed.",
        "masking_applied": False,
        "feature_scheme": "Binary presence for up to 1,500 top ingredients by training document frequency plus log1p parsed ingredient-list length.",
        "max_ingredient_features": MAX_INGREDIENT_FEATURES,
        "benchmark_cv_leader": benchmark_leader,
        "benchmark_note": "Random Forest was selected as the final family for the ensemble-project focus. The comparison table, rather than this selection, identifies the training-CV leader.",
        "artifacts": {
            "classifier": "models/cosmetic_risk_model.joblib",
            "model_config": "models/model_config.json",
            "cv_comparison": "reports/tables/model_comparison_cv.csv",
            "hyperparameter_search": "reports/tables/hyperparameter_search.csv",
            "test_predictions": "reports/tables/test_predictions.csv",
            "prediction_errors": "reports/tables/prediction_errors.csv",
            "feature_importance": "reports/tables/feature_importance.csv",
        },
    }
    (root / "reports" / "final_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "reports" / "classification_report.txt").write_text(
        "Held-out test metrics are stored in reports/final_metrics.json. Threshold selection used training grouped OOF probabilities only.\n"
        + json.dumps({"default_0.5": default_metrics, "training_selected": selected_threshold_metrics}, indent=2),
        encoding="utf-8",
    )
    _progress(root, "Saved final Random Forest model, metrics, tables, and figures.")
    return report, cv_table, test_predictions, feature_importances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    project_root = Path(args.root).resolve()
    frame = pd.read_csv(project_root / "data/processed/products_clean.csv", dtype={"code": "string"}, low_memory=False)
    run_experiments(frame, project_root, folds=args.folds)
