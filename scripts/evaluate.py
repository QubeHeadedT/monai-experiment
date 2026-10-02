"""Stage 5 - score test predictions against the ground truth.

Inputs:
    data/predictions/test/<roi>.npy, data/prepared/test/<roi>.npz
    models/mlflow_run_id.txt
    params.yaml: evaluate.*, prepare.class_names

Outputs:
    metrics/test.json   {"bPQ": ..., "mPQ": ..., "PQ_tumour": ..., ...}

TODO:
    1. Compute panoptic quality per class with monai.metrics.PanopticQualityMetric
       (input: (B, 2, H, W) = [instance map, class map] for prediction and ground truth).
       Report binary PQ (all nuclei as one class), PQ for each class, and mPQ (their mean).
    2. Write metrics/test.json and log the same numbers to the training run in MLflow (wiring below).
"""

import json
from pathlib import Path

import mlflow

from common import RUN_ID_FILE, load_params, start_mlflow_run


def main() -> None:
    params = load_params()

    # ---------------------------------------------------------------------------------------------
    # PLACEHOLDER - delete this block when you implement the stage.
    # Nothing is scored: every metric is set to 0.0 so the output has the real keys and format.
    # ---------------------------------------------------------------------------------------------
    print("WARNING: evaluate.py is a placeholder - all scores are 0.0")
    class_names = params["prepare"]["class_names"][1:]  # skip background
    scores = {"bPQ": 0.0, "mPQ": 0.0, **{f"PQ_{name}": 0.0 for name in class_names}}
    # ------------------------------------- END PLACEHOLDER ---------------------------------------

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/test.json").write_text(json.dumps(scores, indent=2))
    with start_mlflow_run(params, run_id=RUN_ID_FILE.read_text().strip()):
        mlflow.log_metrics({f"test_{k}": v for k, v in scores.items()})


if __name__ == "__main__":
    main()
