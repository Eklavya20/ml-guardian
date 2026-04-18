"""
loader.py
Fetches candidate and production model info + test data from MLflow.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


def get_mlflow_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _get_versions_by_alias_or_tag(client: MlflowClient, model_name: str, stage: str):
    """Get model versions filtered by tag ml_guardian_stage."""
    all_versions = client.search_model_versions(f"name='{model_name}'")
    return [v for v in all_versions if v.tags.get("ml_guardian_stage") == stage]


def load_production_model(client: MlflowClient, model_name: str):
    """Load current Production model by tag. Returns None if none exists."""
    try:
        versions = _get_versions_by_alias_or_tag(client, model_name, "production")
        if not versions:
            # Fallback: try MLflow native stage for backwards compat
            all_versions = client.search_model_versions(f"name='{model_name}'")
            versions = [v for v in all_versions if v.current_stage == "Production"]
        if not versions:
            return None, None
        version = sorted(versions, key=lambda v: int(v.version))[-1]
        model = mlflow.sklearn.load_model(f"runs:/{version.run_id}/model")
        return model, version
    except Exception as e:
        raise RuntimeError(f"Failed to load production model: {e}")


def load_candidate_model(client: MlflowClient, model_name: str):
    """Load latest candidate — tagged staging or latest None-stage version."""
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            return None, None

        # Try tagged staging first
        staging = [v for v in all_versions if v.tags.get("ml_guardian_stage") == "staging"]
        if staging:
            version = sorted(staging, key=lambda v: int(v.version))[-1]
            model = mlflow.sklearn.load_model(f"runs:/{version.run_id}/model")
            return model, version

        # Fallback: MLflow native Staging stage
        native_staging = [v for v in all_versions if v.current_stage == "Staging"]
        if native_staging:
            version = sorted(native_staging, key=lambda v: int(v.version))[-1]
            model = mlflow.sklearn.load_model(f"runs:/{version.run_id}/model")
            return model, version

        # Fallback: latest None-stage version that isn't Production
        none_stage = [v for v in all_versions if v.current_stage == "None"]
        if none_stage:
            version = sorted(none_stage, key=lambda v: int(v.version))[-1]
            model = mlflow.sklearn.load_model(f"runs:/{version.run_id}/model")
            return model, version

        return None, None
    except Exception as e:
        raise RuntimeError(f"Failed to load candidate model: {e}")


def load_test_data(run_id: str, client: MlflowClient) -> tuple[pd.DataFrame, pd.Series]:
    """Load X_test.csv and y_test.csv logged as artifacts in the training run."""
    try:
        import mlflow.artifacts
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="",
        )
        x_path = os.path.join(local_dir, "X_test.csv")
        y_path = os.path.join(local_dir, "y_test.csv")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(
                f"X_test.csv or y_test.csv not found in artifacts for run {run_id}. "
                f"Looking in: {local_dir}"
            )

        X_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path).squeeze()
        return X_test, y_test
    except Exception as e:
        raise RuntimeError(
            f"Failed to load test data from run {run_id}. "
            f"Ensure X_test.csv and y_test.csv are logged as artifacts during training. Error: {e}"
        )