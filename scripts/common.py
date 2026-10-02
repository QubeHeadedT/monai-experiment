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

    Where runs are logged is controlled by environment variables (see README, "MLflow"):
        MLFLOW_TRACKING_URI                                 e.g. https://mlflow.example.com
        MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD  if the server uses basic auth
    Without MLFLOW_TRACKING_URI, runs go to a local ./mlflow.db (view with `uv run mlflow ui`).
    """
    if "MLFLOW_TRACKING_URI" not in os.environ:
        print("MLFLOW_TRACKING_URI not set - logging to local ./mlflow.db")
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    return mlflow.start_run(run_id=run_id)
