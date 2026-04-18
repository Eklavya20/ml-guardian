"""
main.py
Entry point for ml-guardian. Wired together for CLI and GitHub Actions.
"""

import os
import sys
from src.loader import get_mlflow_client, load_production_model, load_candidate_model, load_test_data
from src.comparator import compare
from src.gates import GateConfig, evaluate_gates
from src.reporter import generate_report, save_report, print_report
from src.promoter import promote_candidate


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.environ.get("MODEL_NAME", "telco_churn")

    min_accuracy = float(os.environ.get("MIN_ACCURACY", 0.70))
    min_f1 = float(os.environ.get("MIN_F1", 0.65))
    min_roc_auc = float(os.environ.get("MIN_ROC_AUC", 0.70))
    max_accuracy_drop = float(os.environ.get("MAX_ACCURACY_DROP", 0.02))
    max_f1_drop = float(os.environ.get("MAX_F1_DROP", 0.03))
    max_roc_auc_drop = float(os.environ.get("MAX_ROC_AUC_DROP", 0.02))
    max_ece = float(os.environ.get("MAX_ECE", 0.15))
    max_drifted_features = int(os.environ.get("MAX_DRIFTED_FEATURES", 3))
    auto_promote = os.environ.get("AUTO_PROMOTE", "true").lower() == "true"

    print(f"Connecting to MLflow at {tracking_uri}")
    client = get_mlflow_client(tracking_uri)

    print("Loading production model...")
    production_model, production_version = load_production_model(client, model_name)

    print("Loading candidate model...")
    candidate_model, candidate_version = load_candidate_model(client, model_name)

    if candidate_model is None:
        print("No candidate model found. Exiting.")
        sys.exit(1)

    print(f"Loading test data from run {candidate_version.run_id}...")
    X_test, y_test = load_test_data(candidate_version.run_id, client)

    print("Running comparison...")
    comparison = compare(
        candidate_model=candidate_model,
        production_model=production_model,
        X_test=X_test,
        y_test=y_test,
        candidate_run_id=candidate_version.run_id,
        production_run_id=production_version.run_id if production_version else None,
    )

    config = GateConfig(
        min_accuracy=min_accuracy,
        min_f1=min_f1,
        min_roc_auc=min_roc_auc,
        max_accuracy_drop=max_accuracy_drop,
        max_f1_drop=max_f1_drop,
        max_roc_auc_drop=max_roc_auc_drop,
        max_ece=max_ece,
        max_drifted_features=max_drifted_features,
    )

    print("Evaluating gates...")
    gate_result = evaluate_gates(comparison, config)

    report = generate_report(comparison, gate_result, model_name)
    print_report(report)
    save_report(report)

    if gate_result.passed:
        if auto_promote:
            print("Gates passed. Promoting candidate to Production...")
            promote_candidate(client, model_name, candidate_version, production_version)
        else:
            print("Gates passed. AUTO_PROMOTE=false — skipping promotion.")
    else:
        print("Gates failed. Promotion blocked.")
        sys.exit(1)


if __name__ == "__main__":
    main()