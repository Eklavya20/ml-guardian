"""
gates.py
Configurable quality gates that determine pass/fail for promotion.
"""

from dataclasses import dataclass, field
from src.comparator import ComparisonResult


@dataclass
class GateConfig:
    # Minimum absolute metric thresholds for candidate
    min_accuracy: float = 0.70
    min_f1: float = 0.65
    min_roc_auc: float = 0.70

    # Maximum allowed regression vs production (negative = regression)
    max_accuracy_drop: float = 0.02
    max_f1_drop: float = 0.03
    max_roc_auc_drop: float = 0.02

    # Calibration
    max_ece: float = 0.15

    # Drift
    max_drifted_features: int = 3


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def evaluate_gates(result: ComparisonResult, config: GateConfig) -> GateResult:
    failures = []
    warnings = []

    c = result.candidate_metrics
    deltas = result.metric_deltas
    calib = result.calibration_candidate
    drift = result.drift_results

    # --- Absolute thresholds ---
    if c.get("accuracy") is not None and c["accuracy"] < config.min_accuracy:
        failures.append(f"Accuracy {c['accuracy']} below minimum {config.min_accuracy}")

    if c.get("f1") is not None and c["f1"] < config.min_f1:
        failures.append(f"F1 {c['f1']} below minimum {config.min_f1}")

    if c.get("roc_auc") is not None and c["roc_auc"] < config.min_roc_auc:
        failures.append(f"ROC-AUC {c['roc_auc']} below minimum {config.min_roc_auc}")

    # --- Regression vs production ---
    if deltas:
        if deltas.get("accuracy") is not None and deltas["accuracy"] < -config.max_accuracy_drop:
            failures.append(
                f"Accuracy regressed by {abs(deltas['accuracy']):.4f} "
                f"(max allowed drop: {config.max_accuracy_drop})"
            )
        if deltas.get("f1") is not None and deltas["f1"] < -config.max_f1_drop:
            failures.append(
                f"F1 regressed by {abs(deltas['f1']):.4f} "
                f"(max allowed drop: {config.max_f1_drop})"
            )
        if deltas.get("roc_auc") is not None and deltas["roc_auc"] < -config.max_roc_auc_drop:
            failures.append(
                f"ROC-AUC regressed by {abs(deltas['roc_auc']):.4f} "
                f"(max allowed drop: {config.max_roc_auc_drop})"
            )

    # --- Calibration ---
    if calib.get("ece") is not None and calib["ece"] > config.max_ece:
        failures.append(
            f"ECE {calib['ece']} exceeds maximum {config.max_ece} — model poorly calibrated"
        )

    if calib.get("verdict") == "Poorly calibrated":
        warnings.append("Calibration verdict: Poorly calibrated")

    # --- Drift ---
    if drift.get("drift_count", 0) > config.max_drifted_features:
        failures.append(
            f"Drift detected in {drift['drift_count']} features "
            f"(max allowed: {config.max_drifted_features})"
        )
    elif drift.get("drift_count", 0) > 0:
        warnings.append(
            f"Drift detected in {drift['drift_count']} features: "
            f"{', '.join(drift.get('drifted_features', []))}"
        )

    passed = len(failures) == 0
    return GateResult(passed=passed, failures=failures, warnings=warnings)