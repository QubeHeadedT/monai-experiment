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

import json
from pathlib import Path

import mlflow

from common import RUN_ID_FILE, load_params, start_mlflow_run


def main() -> None:
    params = load_params()
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)

    with start_mlflow_run(params) as run:
        RUN_ID_FILE.write_text(run.info.run_id)
        mlflow.log_params({f"train.{k}": v for k, v in params["train"].items()})

        # TODO: build data loaders, model, loss and optimiser, then train. Inside your epoch loop:
        #   mlflow.log_metrics({"train_loss": ..., "val_loss": ..., "val_bPQ": ...}, step=epoch)
        #   append a row to metrics/train_log.csv
        # After training:
        #   torch.save(model.state_dict(), "models/hovernet.pt")
        #   Path("metrics/train.json").write_text(json.dumps(best_scores))
        #   mlflow.log_artifact("models/hovernet.pt")
        raise NotImplementedError("scripts/train.py: see the TODO list in the docstring")


if __name__ == "__main__":
    main()
