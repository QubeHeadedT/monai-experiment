"""Stage 3 - fine-tune MONAI's HoVerNet on the prepared training data.

Inputs:
    data/prepared/train/*.npz, data/prepared/val/*.npz
    params.yaml: train.*, mlflow.*

Outputs:
    models/hovernet.pt          best checkpoint (state_dict)
    models/mlflow_run_id.txt    MLflow run ID, so the evaluate stage logs to the same run
    metrics/train.json          best validation scores
    metrics/train_log.csv       one row per epoch (columns: epoch, train_loss, val_loss, ...)

TODO:
    1. Dataset + transforms: random 256x256 crops, flips/rotations, colour augmentation, then
       targets with ComputeHoVerMapsd (see README "MONAI resources").
    2. Model: monai.networks.nets.HoVerNet with out_classes=len(prepare.class_names). Start from
       the MONAI model-zoo bundle train.pretrained_bundle (trained on CoNSeP, 5 classes): load
       all weights except the final type-prediction layer.
    3. Loss: monai.apps.pathology.losses.HoVerNetLoss. Optimiser, schedule, encoder freezing.
    4. Validation each epoch, best checkpoint saved to models/hovernet.pt.
    5. Log params, per-epoch metrics and artefacts to MLflow (wiring below).
"""

import csv
import json
from pathlib import Path

import mlflow
import torch

from common import RUN_ID_FILE, load_params, start_mlflow_run


def main() -> None:
    params = load_params()
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    with start_mlflow_run(params) as run:
        RUN_ID_FILE.write_text(run.info.run_id)
        mlflow.log_params({f"train.{k}": v for k, v in params["train"].items()})

        # -----------------------------------------------------------------------------------------
        # PLACEHOLDER - delete this block when you implement the stage.
        # No training happens: it writes made-up losses and an empty checkpoint in the real output
        # formats so `dvc repro` and MLflow logging can be tried end to end.
        # -----------------------------------------------------------------------------------------
        print("WARNING: train.py is a placeholder - no model is trained")
        rows = []
        for epoch in range(3):
            row = {"epoch": epoch, "train_loss": 1.0 / (epoch + 1), "val_loss": 1.2 / (epoch + 1)}
            rows.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        with open("metrics/train_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        best = min(rows, key=lambda r: r["val_loss"])
        Path("metrics/train.json").write_text(json.dumps(best, indent=2))
        # A real checkpoint would be model.state_dict(); this one is an empty stand-in.
        torch.save({"placeholder": True}, "models/hovernet.pt")
        # ----------------------------------- END PLACEHOLDER -------------------------------------

        mlflow.log_artifact("metrics/train_log.csv")


if __name__ == "__main__":
    main()
