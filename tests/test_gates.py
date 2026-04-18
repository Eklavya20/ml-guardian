import pytest
from src.gates import GateConfig, GateResult, evaluate_gates
from src.comparator import ComparisonResult


def make_comparison(candidate_metrics, production_metrics=None, deltas=None,
                    calib_candidate=None, calib_production=None, drift=None):
    return ComparisonResult(
        candidate_metrics=candidate_metrics,
        production_metrics=production_metrics or {},
        metric_deltas=deltas or {},
        calibration_candidate=calib_candidate or {"ece": 0.05, "verdict": "Well calibrated"},
        calibration_production=calib_production or {},
        drift_results=drift or {"drifted_features": [], "drift_count": 0, "total_features": 10},
        candidate_run_id="abc123",
        production_run_id=None,
    )


def test_gates_pass_clean_model():
    comparison = make_comparison(
        candidate_metrics={"accuracy": 0.85, "f1": 0.80, "roc_auc": 0.88, "precision": 0.82, "recall": 0.79}
    )
    result = evaluate_gates(comparison, GateConfig())
    assert result.passed is True
    assert result.failures == []


def test_gates_fail_low_accuracy():
    comparison = make_comparison(
        candidate_metrics={"accuracy": 0.50, "f1": 0.80, "roc_auc": 0.88, "precision": 0.82, "recall": 0.79}
    )
    result = evaluate_gates(comparison, GateConfig())
    assert result.passed is False
    assert any("Accuracy" in f for f in result.failures)


def test_gates_fail_low_f1():
    comparison = make_comparison(
        candidate_metrics={"accuracy": 0.85, "f1": 0.50, "roc_auc": 0.88, "precision": 0.82, "recall": 0.79}
    )
    result = evaluate_gates(comparison, GateConfig())
    assert result.passed is False
    assert any("F1" in f for f in result.failures)


def test_gates_fail_accuracy_regression():
    comparison = make_comparison(
        candidate_metrics={"accuracy": 0.80, "f1": 0.75, "roc_auc": 0.85, "precision": 0.78, "recall": 0.74},
        production_metrics={"accuracy": 0.85, "f1": 0.75, "roc_auc": 0.85, "precision": 0.78, "recall": 0.74},
        deltas={"accuracy": -0.05, "f1": 0.0, "roc_auc": 0.0},
    )
    result = evaluate_gates(comparison, GateConfig())
    assert result.passed is False
    assert any("Accuracy regressed" in f for f in result.failures)


def test_gates_fail_high_ece():
    comparison = make_comparison(
        candidate_metrics={"accuracy": 0.85, "f1": 0.80, "roc_auc": 0.88, "precision": 0.82, "recall": 0.79},
        calib_candidate={"ece": 0.25, "verdict": "Poorly calibrated"},
    )
    result = evaluate_gates(comparison, GateConfig())
    assert result.passed is False
    assert any("ECE" in f for f in result.failures)


def test_gates_warn_drift():
    comparison = make_comparison(
        candidate_metrics={"accuracy": 0.85, "f1": 0.80, "roc_auc": 0.88, "precision": 0.82, "recall": 0.79},
        drift={"drifted_features": ["f1", "f2"], "drift_count": 2, "total_features": 10},
    )
    result = evaluate_gates(comparison, GateConfig())
    assert result.passed is True
    assert any("Drift" in w for w in result.warnings)


def test_gates_fail_excessive_drift():
    comparison = make_comparison(
        candidate_metrics={"accuracy": 0.85, "f1": 0.80, "roc_auc": 0.88, "precision": 0.82, "recall": 0.79},
        drift={"drifted_features": ["f1","f2","f3","f4"], "drift_count": 4, "total_features": 10},
    )
    result = evaluate_gates(comparison, GateConfig())
    assert result.passed is False
    assert any("Drift" in f for f in result.failures)