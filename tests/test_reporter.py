import pytest
from src.reporter import generate_report
from src.comparator import ComparisonResult
from src.gates import GateResult


def make_comparison():
    return ComparisonResult(
        candidate_metrics={"accuracy": 0.85, "f1": 0.80, "roc_auc": 0.88, "precision": 0.82, "recall": 0.79},
        production_metrics={"accuracy": 0.83, "f1": 0.78, "roc_auc": 0.86, "precision": 0.80, "recall": 0.77},
        metric_deltas={"accuracy": 0.02, "f1": 0.02, "roc_auc": 0.02, "precision": 0.02, "recall": 0.02},
        calibration_candidate={"ece": 0.05, "verdict": "Well calibrated"},
        calibration_production={"ece": 0.07, "verdict": "Moderately calibrated"},
        drift_results={"drifted_features": [], "drift_count": 0, "total_features": 19},
        candidate_run_id="abc123",
        production_run_id="def456",
    )


def test_report_contains_model_name():
    report = generate_report(make_comparison(), GateResult(passed=True), "telco_churn")
    assert "telco_churn" in report


def test_report_passed_status():
    report = generate_report(make_comparison(), GateResult(passed=True), "telco_churn")
    assert "PASSED" in report


def test_report_failed_status():
    gate = GateResult(passed=False, failures=["Accuracy 0.50 below minimum 0.70"])
    report = generate_report(make_comparison(), gate, "telco_churn")
    assert "FAILED" in report
    assert "Accuracy 0.50 below minimum 0.70" in report


def test_report_contains_metrics():
    report = generate_report(make_comparison(), GateResult(passed=True), "telco_churn")
    for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
        assert metric in report


def test_report_contains_run_ids():
    report = generate_report(make_comparison(), GateResult(passed=True), "telco_churn")
    assert "abc123" in report
    assert "def456" in report


def test_report_warnings_shown():
    gate = GateResult(passed=True, warnings=["Drift detected in 2 features"])
    report = generate_report(make_comparison(), gate, "telco_churn")
    assert "Drift detected in 2 features" in report