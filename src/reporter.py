"""
reporter.py
Generates a markdown report for GitHub Actions PR comments and local output.
"""

from src.comparator import ComparisonResult
from src.gates import GateResult
from datetime import datetime, timezone


def _metric_row(metric: str, candidate: dict, production: dict, deltas: dict) -> str:
    c_val = candidate.get(metric, "N/A")
    p_val = production.get(metric, "N/A") if production else "N/A"
    delta = deltas.get(metric)

    if delta is not None:
        if delta > 0:
            delta_str = f"🟢 +{delta:.4f}"
        elif delta < 0:
            delta_str = f"🔴 {delta:.4f}"
        else:
            delta_str = f"⚪ 0.0000"
    else:
        delta_str = "N/A"

    return f"| {metric} | {c_val} | {p_val} | {delta_str} |"


def generate_report(
    comparison: ComparisonResult,
    gate_result: GateResult,
    model_name: str,
) -> str:

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "✅ PASSED — Auto-promotion approved" if gate_result.passed else "❌ FAILED — Promotion blocked"

    lines = [
        f"# 🛡️ ML Guardian Report",
        f"**Model:** `{model_name}`  ",
        f"**Candidate run:** `{comparison.candidate_run_id}`  ",
        f"**Production run:** `{comparison.production_run_id or 'None (first deployment)'}` ",
        f"**Generated:** {now}",
        f"",
        f"## Status: {status}",
        f"",
    ]

    # Failures
    if gate_result.failures:
        lines.append("### ❌ Gate Failures")
        for f in gate_result.failures:
            lines.append(f"- {f}")
        lines.append("")

    # Warnings
    if gate_result.warnings:
        lines.append("### ⚠️ Warnings")
        for w in gate_result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Metrics table
    lines += [
        "### 📊 Metric Comparison",
        "| Metric | Candidate | Production | Delta |",
        "|--------|-----------|------------|-------|",
    ]

    for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
        lines.append(_metric_row(
            metric,
            comparison.candidate_metrics,
            comparison.production_metrics,
            comparison.metric_deltas,
        ))

    lines.append("")

    # Calibration
    lines += [
        "### 🎯 Calibration",
        "| | Candidate | Production |",
        "|---|-----------|------------|",
        f"| ECE | {comparison.calibration_candidate.get('ece', 'N/A')} | "
        f"{comparison.calibration_production.get('ece', 'N/A')} |",
        f"| Verdict | {comparison.calibration_candidate.get('verdict', 'N/A')} | "
        f"{comparison.calibration_production.get('verdict', 'N/A')} |",
        "",
    ]

    # Drift
    drift = comparison.drift_results
    lines += [
        "### 🌊 Drift Detection",
        f"**Drifted features:** {drift.get('drift_count', 0)} / {drift.get('total_features', 0)}",
    ]

    if drift.get("drifted_features"):
        lines.append(f"**Features:** {', '.join(drift['drifted_features'])}")

    lines.append("")

    return "\n".join(lines)


def save_report(report: str, path: str = "reports/report.md") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {path}")


def print_report(report: str) -> None:
    print(report)