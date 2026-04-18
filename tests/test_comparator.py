import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.comparator import compute_metrics, compute_calibration, compute_drift, compute_deltas


@pytest.fixture
def binary_model_and_data():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
    return model, X_test, y_test


def test_compute_metrics_keys(binary_model_and_data):
    model, X_test, y_test = binary_model_and_data
    metrics = compute_metrics(model, X_test, y_test)
    assert set(metrics.keys()) == {"accuracy", "f1", "roc_auc", "precision", "recall"}


def test_compute_metrics_ranges(binary_model_and_data):
    model, X_test, y_test = binary_model_and_data
    metrics = compute_metrics(model, X_test, y_test)
    for k, v in metrics.items():
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{k} out of range: {v}"


def test_compute_calibration_keys(binary_model_and_data):
    model, X_test, y_test = binary_model_and_data
    result = compute_calibration(model, X_test, y_test)
    assert "ece" in result
    assert "verdict" in result


def test_compute_calibration_ece_range(binary_model_and_data):
    model, X_test, y_test = binary_model_and_data
    result = compute_calibration(model, X_test, y_test)
    assert 0.0 <= result["ece"] <= 1.0


def test_compute_drift_no_drift(binary_model_and_data):
    _, X_test, _ = binary_model_and_data
    result = compute_drift(X_test, X_test)
    assert result["drift_count"] == 0
    assert result["total_features"] == X_test.shape[1]


def test_compute_drift_with_drift(binary_model_and_data):
    _, X_test, _ = binary_model_and_data
    X_shifted = X_test.copy() + 10.0
    result = compute_drift(X_test, X_shifted)
    assert result["drift_count"] > 0


def test_compute_deltas():
    candidate = {"accuracy": 0.85, "f1": 0.80, "roc_auc": 0.90}
    production = {"accuracy": 0.80, "f1": 0.82, "roc_auc": 0.88}
    deltas = compute_deltas(candidate, production)
    assert deltas["accuracy"] == pytest.approx(0.05, abs=1e-4)
    assert deltas["f1"] == pytest.approx(-0.02, abs=1e-4)
    assert deltas["roc_auc"] == pytest.approx(0.02, abs=1e-4)