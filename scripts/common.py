"""Shared helpers: reading params.yaml and connecting to MLflow."""

import os
from pathlib import Path

import mlflow
import yaml

PARAMS_FILE = Path("params.yaml")
RUN_ID_FILE = Path("models/mlflow_run_id.txt")


def load_params() -> dict:
    return yaml.safe_load(PARAMS_FILE.read_text())


def start_mlflow_run(params: dict, run_id: str | None = None) -> mlflow.ActiveRun:
    """Start (or resume, if `run_id` is given) an MLflow run.

    Where runs are logged (see README, "MLflow"), first match wins:
        1. the MLFLOW_TRACKING_URI environment variable (to point at another server for a one-off)
        2. mlflow.tracking_uri in params.yaml: the SageMaker MLflow app ARN (arn:aws:sagemaker:...),
           resolved by the sagemaker-mlflow plugin and authenticated with your AWS credentials
        3. a local ./mlflow.db (view with `uv run mlflow ui`)
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or params["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("No MLflow tracking URI set - logging to local ./mlflow.db")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    return mlflow.start_run(run_id=run_id)
