"""
comparator.py
Compares candidate model against production model using diagnost.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
import diagnost
from diagnost.drift import check_drift
import contextlib
import io

@dataclass
class ComparisonResult:
    candidate_metrics: dict
    production_metrics: dict
    metric_deltas: dict
    calibration_candidate: dict
    calibration_production: dict
    drift_results: dict
    candidate_run_id: str
    production_run_id: str | None


def compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    from sklearn.metrics import roc_auc_score
    report = diagnost.evaluate(model, X_test, y_test, task="classification")
    d = report.to_dict()

    try:
        proba = model.predict_proba(X_test)
        classes = np.unique(y_test)
        if len(classes) == 2:
            roc_auc = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
        roc_auc = round(roc_auc, 4)
    except Exception:
        roc_auc = None

    return {
        "accuracy": round(float(d.get("accuracy", 0.0)), 4),
        "f1": round(float(d.get("f1", 0.0)), 4),
        "roc_auc": roc_auc,
        "precision": round(float(d.get("precision", 0.0)), 4),
        "recall": round(float(d.get("recall", 0.0)), 4),
    }


def compute_calibration(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute ECE directly without relying on diagnost's print-only calibration."""
    try:
        proba = model.predict_proba(X_test)
        n_classes = proba.shape[1]
        n_bins = 10
        ece_per_class = []

        y_arr = np.array(y_test)
        classes = np.unique(y_arr)

        for i, cls in enumerate(classes):
            prob = proba[:, i]
            binary_y = (y_arr == cls).astype(int)
            bins = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for j in range(n_bins):
                mask = (prob >= bins[j]) & (prob < bins[j + 1])
                if mask.sum() == 0:
                    continue
                bin_conf = prob[mask].mean()
                bin_acc = binary_y[mask].mean()
                ece += (mask.sum() / len(prob)) * abs(bin_acc - bin_conf)
            ece_per_class.append(ece)

        mean_ece = round(float(np.mean(ece_per_class)), 4)
        if mean_ece < 0.05:
            verdict = "Well calibrated"
        elif mean_ece < 0.10:
            verdict = "Moderately calibrated"
        else:
            verdict = "Poorly calibrated"

        return {"ece": mean_ece, "verdict": verdict}
    except Exception as e:
        return {"ece": None, "verdict": f"Could not compute: {e}"}


def compute_drift(X_reference: pd.DataFrame, X_candidate: pd.DataFrame) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        result = check_drift(X_reference, X_candidate)
    drifted = [
        f for f, v in result.items()
        if isinstance(v, dict) and bool(v.get("drifted", False))
    ]
    return {
        "drifted_features": drifted,
        "drift_count": len(drifted),
        "total_features": len(X_reference.columns),
    }


def compute_deltas(candidate: dict, production: dict) -> dict:
    deltas = {}
    for metric in candidate:
        if metric in production and candidate[metric] is not None and production[metric] is not None:
            deltas[metric] = round(candidate[metric] - production[metric], 4)
    return deltas


def compare(
    candidate_model,
    production_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidate_run_id: str,
    production_run_id: str | None,
) -> ComparisonResult:

    candidate_metrics = compute_metrics(candidate_model, X_test, y_test)
    calibration_candidate = compute_calibration(candidate_model, X_test, y_test)

    if production_model is not None:
        production_metrics = compute_metrics(production_model, X_test, y_test)
        calibration_production = compute_calibration(production_model, X_test, y_test)
        metric_deltas = compute_deltas(candidate_metrics, production_metrics)
        drift_results = compute_drift(X_test, X_test)
    else:
        production_metrics = {}
        calibration_production = {}
        metric_deltas = {}
        drift_results = {
            "drifted_features": [],
            "drift_count": 0,
            "total_features": len(X_test.columns),
        }

    return ComparisonResult(
        candidate_metrics=candidate_metrics,
        production_metrics=production_metrics,
        metric_deltas=metric_deltas,
        calibration_candidate=calibration_candidate,
        calibration_production=calibration_production,
        drift_results=drift_results,
        candidate_run_id=candidate_run_id,
        production_run_id=production_run_id,
    )