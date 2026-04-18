"""
promoter.py
Handles automatic promotion of candidate model to Production in MLflow.
Uses tags instead of deprecated MLflow stages.
"""

from mlflow.tracking import MlflowClient


def promote_candidate(
    client: MlflowClient,
    model_name: str,
    candidate_version,
    production_version,
) -> None:
    # Untag current production
    if production_version:
        client.delete_model_version_tag(
            name=model_name,
            version=production_version.version,
            key="ml_guardian_stage",
        )
        # Also handle native stage
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=production_version.version,
                stage="Archived",
            )
        except Exception:
            pass
        print(f"Archived production version {production_version.version}")

    # Tag candidate as production
    client.set_model_version_tag(
        name=model_name,
        version=candidate_version.version,
        key="ml_guardian_stage",
        value="production",
    )

    # Also set native stage for backwards compat
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=candidate_version.version,
            stage="Production",
        )
    except Exception:
        pass

    print(f"Promoted candidate version {candidate_version.version} to Production")