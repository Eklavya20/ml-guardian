"""
rollback.py

Emergency rollback for ml-guardian.
Finds the last known good model version and re-promotes it to Production.

Usage:
    python scripts/rollback.py

Environment variables:
    MLFLOW_TRACKING_URI  — MLflow tracking server (default: http://localhost:5000)
    MODEL_NAME           — Registered model name (default: telco_churn)
    MIN_ROC_AUC          — Minimum ROC-AUC to consider a version "good" (default: 0.80)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loader import get_mlflow_client
from src.promoter import promote_candidate
from src.reporter import save_report


def find_last_good_version(client, model_name: str, min_roc_auc: float, current_production_version):
    """Find the most recent archived version that meets the quality bar."""
    all_versions = client.search_model_versions(f"name='{model_name}'")

    candidates = []
    for v in all_versions:
        # Skip current production
        if current_production_version and v.version == current_production_version.version:
            continue

        # Skip the degraded demo tag
        if v.tags.get("model_type") == "degraded_demo":
            continue

        # Get metrics from the run
        try:
            run = client.get_run(v.run_id)
            roc_auc = run.data.metrics.get("roc_auc")
            if roc_auc and roc_auc >= min_roc_auc:
                candidates.append((v, roc_auc))
        except Exception:
            continue

    if not candidates:
        return None, None

    # Pick the one with highest ROC-AUC
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


def generate_rollback_report(
    model_name: str,
    rolled_back_from,
    rolled_back_to,
    roc_auc: float,
) -> str:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# 🔄 ML Guardian Rollback Report",
        f"**Model:** `{model_name}`",
        f"**Generated:** {now}",
        f"",
        f"## Status: ✅ Rollback Complete",
        f"",
        f"### What happened",
        f"Production model was rolled back due to quality degradation.",
        f"",
        f"| | Version | Run ID |",
        f"|---|---|---|",
        f"| Rolled back from | {rolled_back_from.version} | `{rolled_back_from.run_id[:8]}...` |",
        f"| Restored to | {rolled_back_to.version} | `{rolled_back_to.run_id[:8]}...` |",
        f"",
        f"### Restored model quality",
        f"| Metric | Value |",
        f"|---|---|",
        f"| ROC-AUC | {roc_auc:.4f} |",
        f"",
        f"### Next steps",
        f"1. Investigate why the degraded model was promoted",
        f"2. Fix the root cause in the training pipeline",
        f"3. Retrain and re-evaluate before next promotion attempt",
    ]

    return "\n".join(lines)


def rollback(
    tracking_uri: str = None,
    model_name: str = None,
    min_roc_auc: float = None,
):
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = model_name or os.environ.get("MODEL_NAME", "telco_churn")
    min_roc_auc = min_roc_auc or float(os.environ.get("MIN_ROC_AUC", "0.80"))

    print(f"Connecting to MLflow at {tracking_uri}")
    client = get_mlflow_client(tracking_uri)

    # Find current production
    from src.loader import load_production_model
    _, current_production = load_production_model(client, model_name)

    if not current_production:
        print("No production model found. Nothing to roll back.")
        sys.exit(1)

    print(f"Current production: version {current_production.version}")

    # Find last good version
    print(f"Searching for last good version with ROC-AUC >= {min_roc_auc}...")
    good_version, roc_auc = find_last_good_version(
        client, model_name, min_roc_auc, current_production
    )

    if not good_version:
        print(f"No suitable rollback version found with ROC-AUC >= {min_roc_auc}.")
        sys.exit(1)

    print(f"Found version {good_version.version} with ROC-AUC {roc_auc:.4f}")

    # Re-promote
    print(f"Rolling back to version {good_version.version}...")
    promote_candidate(
        client=client,
        model_name=model_name,
        candidate_version=good_version,
        production_version=current_production,
    )

    # Generate report
    report_md = generate_rollback_report(
        model_name=model_name,
        rolled_back_from=current_production,
        rolled_back_to=good_version,
        roc_auc=roc_auc,
    )

    os.makedirs("reports", exist_ok=True)
    with open("reports/rollback_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + report_md)
    print("Rollback report saved to reports/rollback_report.md")


if __name__ == "__main__":
    rollback()