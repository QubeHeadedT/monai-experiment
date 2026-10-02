# Fine-tuning HoVerNet on PanopTILs with MONAI

## The task

Train a model that finds every nucleus in a breast-cancer H&E image, outlines it, and says what kind of
cell it is. Start from MONAI's **HoVerNet** and fine-tune it on **PanopTILs**.

- **Input:** a 1024×1024 RGB region of interest (ROI) at 0.25 µm/pixel (40×).
- **Output:** an instance map (one ID per nucleus) and a class map with 6 classes: background, tumour,
  stromal, TIL (lymphocytes + plasma cells), normal epithelial, and other. The mapping from the raw
  PanopTILs codes is in [params.yaml](params.yaml).
- **Score:** panoptic quality (PQ) on the held-out test hospitals. Report binary PQ (bPQ), PQ for each
  class, and mPQ (the mean over classes).

The pipeline has five stages, defined in [dvc.yaml](dvc.yaml). Each stage is a script in
[scripts/](scripts/), and each script's docstring lists its inputs, outputs and TODOs:

```text
download ─► prepare ─► train ─► predict ─► evaluate
               └──────────────────┴───────────┘   (prepare's test data also feeds predict and evaluate)
```

| Stage | Script | Produces |
| --- | --- | --- |
| download | [download_data.py](scripts/download_data.py) (done) | `data/raw/` – RGB PNGs, 3-channel mask PNGs, official splits |
| prepare | [prepare.py](scripts/prepare.py) | `data/prepared/{train,val,test}/<roi>.npz` (image, inst_map, type_map), `splits.json` |
| train | [train.py](scripts/train.py) | `models/hovernet.pt`, `metrics/train.json`, `metrics/train_log.csv`, MLflow run |
| predict | [predict.py](scripts/predict.py) | `data/predictions/test/<roi>.npy` – (2, H, W) instance + class maps |
| evaluate | [evaluate.py](scripts/evaluate.py) | `metrics/test.json`, logged to the same MLflow run |

**Deliverables:** a working `uv run dvc repro`, your test metrics, the MLflow run, and a short write-up
covering your choices, learning curves, failure cases, and what you would try next.

## Setup

```bash
uv sync                                                            # Python 3.13, MONAI, PyTorch, DVC, MLflow
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Data

PanopTILs: <https://sites.google.com/view/panoptils>. Use the **"manual regions & bootstrapped
nuclei"** release ([Drive folder](https://drive.google.com/drive/folders/1QOaUSz3zIoVwVVuyj3uAyfVbIdOTNMmZ)).
It has 1,709 ROIs from 151 TCGA patients, plus official hospital-wise 5-fold splits.

There are two ways to get it:

1. **Script:** `uv run dvc repro download` (or `uv run scripts/download_data.py`). Google Drive throttles
   bulk downloads and may start refusing files after a few hundred. Re-run later to resume; files
   already downloaded are skipped.
2. **Browser:** open the Drive folder and download `tcga/` (only `rgbs/` and `masks/` are needed) and
   `train_test_splits/`. Drive zips them. Unzip so you end up with `data/raw/tcga/rgbs/*.png`,
   `data/raw/tcga/masks/*.png` and `data/raw/train_test_splits/*.csv`. Then run
   `uv run dvc commit download` so DVC treats the stage as done.

Each mask is a 3-channel PNG:

| Channel | Content |
| --- | --- |
| 0 | region class: 0 exclude, 1 cancerous epithelium, 2 stroma, 3 TILs, 4 normal epithelium, 5 junk/debris, 6 blood, 7 other, 8 whitespace |
| 1 | nucleus class: 0 exclude, 1 cancer, 2 stromal, 3 large stromal, 4 lymphocyte, 5 plasma/large TIL, 6 normal epithelial, 7 other, 8 unknown, 9 background |
| 2 | nucleus boundary edges (binary) |

There are no instance IDs. Deriving them from channels 1 and 2 is part of the `prepare` stage.

## Running the pipeline with DVC

[DVC](https://dvc.org) (Data Version Control) does for data and models roughly what Git does for code.
Here it does three jobs:

1. **Pipeline runner.** [dvc.yaml](dvc.yaml) declares each stage's command, inputs (`deps`), parameters
   (`params`) and outputs (`outs`). `dvc repro` re-runs only the stages whose inputs changed, like `make`.
2. **Data versioning.** Large outputs (`data/prepared`, `models/hovernet.pt`, ...) are kept out of Git.
   DVC stores them in its cache (`.dvc/cache`) and records their hashes in `dvc.lock`, which *is*
   committed. Checking out an old commit plus `dvc checkout` gives you that commit's data and model.
3. **Experiment tracking.** Metrics (`metrics/*.json`), plots (`metrics/train_log.csv`) and
   [params.yaml](params.yaml) are tracked, so you can compare runs and Git commits.

### How the files in this repo fit together

| File | What it is | In Git? |
| --- | --- | --- |
| `dvc.yaml` | Pipeline definition: stages, deps, params, outs, metrics, plots | yes |
| `params.yaml` | Hyperparameters. Each stage lists the sections it depends on | yes |
| `dvc.lock` | Created by `dvc repro`: exact hashes of every dep/out from the last run | yes, commit it after each run |
| `.dvc/` | DVC config. `.dvc/cache/` holds the data itself | config yes, cache no |
| `data/`, `models/` | Stage outputs. DVC adds them to `.gitignore` automatically | no (tracked by DVC) |

Always read hyperparameters from `params.yaml` (see `load_params()` in
[scripts/common.py](scripts/common.py)) and never hard-code them. That way DVC knows when a stage is
stale. If a script reads a file that isn't listed in its stage's `deps`, DVC won't re-run the stage
when that file changes.

### Everyday commands

```bash
uv run dvc dag                     # draw the pipeline
uv run dvc status                  # which stages are out of date, and why
uv run dvc repro                   # run all stages that are out of date
uv run dvc repro train             # run up to and including one stage
uv run dvc repro -s train          # run only that stage (its inputs must already exist)
uv run dvc repro -f evaluate       # force a re-run even if nothing changed

uv run dvc metrics show            # metrics/train.json and metrics/test.json
uv run dvc metrics diff            # compare with the last commit (or: dvc metrics diff main)
uv run dvc params diff             # which hyperparameters changed
uv run dvc plots show              # HTML plot of metrics/train_log.csv

git add dvc.lock params.yaml && git commit -m "..."   # record a result
```

### Experiments

`dvc exp` runs the pipeline with modified params without making a commit for each try, then lets you
compare the runs in a table:

```bash
uv run dvc exp run -S train.lr=3e-4 -S train.batch_size=2   # one experiment
uv run dvc exp run --queue -S train.lr=1e-4,3e-4,1e-3       # queue a small sweep...
uv run dvc queue start                                      # ...and run it
uv run dvc exp show                                         # table of params + metrics
uv run dvc exp apply <exp-name>                             # bring the best one into your workspace
```

DVC experiments and MLflow overlap. In this activity, MLflow is where learning curves and runs are
shared with the team, and DVC is what makes each result reproducible from a Git commit.

### Sharing data and models

DVC's cache is local. To share outputs (or move to a bigger GPU machine), add a remote and push:

```bash
uv run dvc remote add -d storage s3://my-bucket/panoptils   # or gs://, azure://, ssh://, /shared/path
uv run dvc push                    # upload cached outputs
uv run dvc pull                    # on another machine: fetch them instead of re-running
```

`data/raw` is marked `cache: false` in `dvc.yaml`, so DVC never copies the ~1,700 raw images into
its cache or a remote. Anyone who needs them downloads them from Drive.

### Learning DVC

- [Get Started](https://dvc.org/doc/start): the official tutorial. Do the
  [data pipelines](https://dvc.org/doc/start/data-pipelines/data-pipelines) and
  [metrics, parameters and plots](https://dvc.org/doc/start/data-pipelines/metrics-parameters-plots)
  sections first. They match this repo most closely.
- [Defining pipelines](https://dvc.org/doc/user-guide/pipelines/defining-pipelines) and the
  [dvc.yaml reference](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files): every field
  used in our `dvc.yaml` (`deps`, `params`, `outs`, `persist`, `cache`, `metrics`, `plots`).
- [Experiments: Get Started](https://dvc.org/doc/start/experiments) and
  [experiment management](https://dvc.org/doc/user-guide/experiment-management).
- [Remote storage](https://dvc.org/doc/user-guide/data-management/remote-storage): setting up S3,
  GCS, SSH or a shared drive.
- [Versioning data and models](https://dvc.org/doc/use-cases/versioning-data-and-models): the "why",
  in more depth.
- Command reference: [`repro`](https://dvc.org/doc/command-reference/repro),
  [`status`](https://dvc.org/doc/command-reference/status),
  [`dag`](https://dvc.org/doc/command-reference/dag),
  [`commit`](https://dvc.org/doc/command-reference/commit),
  [`exp show`](https://dvc.org/doc/command-reference/exp/show).
- [DVC Basics](https://www.youtube.com/playlist?list=PL7WG7YrwYcnDb0qdPl9-KEStsL-3oaEjg): a YouTube
  video playlist, good to watch before the Get Started tutorial.
- [Free DVC course](https://learn.dvc.org) and the
  [VS Code extension](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc), which shows
  experiments, plots and the DAG inside the editor.

## MLflow

[scripts/common.py](scripts/common.py) sets this up: `train.py` creates a run, logs `params.yaml`, and
stores the run ID in `models/mlflow_run_id.txt`, and `evaluate.py` adds the test metrics to the same run.
Your job is to log per-epoch metrics, and optionally the checkpoint and example predictions, inside the
training loop.

To log to the shared MLflow server, set these environment variables before running the pipeline
(for example in a `.env` file you `source`, which should not be committed):

```bash
export MLFLOW_TRACKING_URI=https://<your-mlflow-server>       # TODO: fill in when the server is up
export MLFLOW_TRACKING_USERNAME=<username>                     # only if the server uses basic auth
export MLFLOW_TRACKING_PASSWORD=<password>
```

Runs are grouped under the experiment `mlflow.experiment_name` in `params.yaml` (`panoptils-hovernet`).
Give each run a name (`mlflow.start_run(run_name=...)` or `mlflow.set_tag("mlflow.runName", ...)`) so
runs can be compared. Without `MLFLOW_TRACKING_URI`, runs go to a local `./mlflow.db`; view them with `uv run mlflow ui`.

## MONAI resources for HoVerNet

The library code, with paths in [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI):

| What | Where |
| --- | --- |
| Network (`HoVerNet`, fast/original modes, pretrained encoder, `freeze_encoder`) | [monai/networks/nets/hovernet.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/hovernet.py) |
| Loss (`HoVerNetLoss`) | [monai/apps/pathology/losses/hovernet_loss.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/pathology/losses/hovernet_loss.py) |
| Horizontal/vertical target maps (`ComputeHoVerMaps[d]`) | [monai/transforms/intensity/array.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/intensity/array.py) |
| Sliding-window inference (`SlidingWindowHoVerNetInferer`) | [monai/apps/pathology/inferers/inferer.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/pathology/inferers/inferer.py) |
| Post-processing (`HoVerNetInstanceMapPostProcessing`, `HoVerNetNuclearTypePostProcessing`) | [monai/apps/pathology/transforms/post/array.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/pathology/transforms/post/array.py) |
| Metric (`PanopticQualityMetric`) | [monai/metrics/panoptic_quality.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/panoptic_quality.py) |

Worked examples, all on the CoNSeP dataset:

- Tutorial: [Project-MONAI/tutorials – pathology/hovernet](https://github.com/Project-MONAI/tutorials/tree/main/pathology/hovernet)
  (`prepare_patches.py`, `training.py`, `evaluation.py`, `inference.py`, `hovernet_torch.ipynb`).
- Pretrained model-zoo bundle to fine-tune from:
  [pathology_nuclei_segmentation_classification](https://github.com/Project-MONAI/model-zoo/tree/dev/models/pathology_nuclei_segmentation_classification).
  Download it with `monai.bundle.download(name="pathology_nuclei_segmentation_classification", bundle_dir="data/pretrained")`.
  Its `configs/train.json` shows a complete transform pipeline. It predicts 5 classes and we have 6,
  so the final type-prediction layer has to be re-initialised.

## Papers

- **PanopTILs:** Liu, Amgad et al., *A panoptic segmentation dataset and deep-learning approach for
  explainable scoring of tumor-infiltrating lymphocytes*, npj Breast Cancer, 2024.
  [PMC11213912](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11213912/) ·
  [medRxiv](https://www.medrxiv.org/content/10.1101/2022.01.08.22268814)
- **HoVerNet:** Graham et al., *Hover-Net: Simultaneous segmentation and classification of nuclei in
  multi-tissue histology images*, Medical Image Analysis, 2019.
  [doi:10.1016/j.media.2019.101563](https://doi.org/10.1016/j.media.2019.101563) ·
  [arXiv:1812.06499](https://arxiv.org/abs/1812.06499) · [original code](https://github.com/vqdang/hover_net)
- **Panoptic quality:** Kirillov et al., *Panoptic Segmentation*, CVPR 2019.
  [arXiv:1801.00868](https://arxiv.org/abs/1801.00868). For PQ as used on nuclei, see the
  [CoNIC challenge](https://github.com/TissueImageAnalytics/CoNIC).

## Tips

- HoVerNet is memory-hungry. On a 4 GB GPU, use batch sizes of 2–4 with the encoder frozen, and
  smaller ones once it is unfrozen.
- On GPUs without tensor cores (GTX 10xx/16xx), mixed precision (AMP) is much *slower* than fp32,
  so leave `train.amp: false`. On RTX or data-centre GPUs, turn it on.
- Never split by ROI. ROIs from the same slide or hospital look alike, so let them leak across splits
  and your validation score will be too optimistic.
- The nucleus labels in this release are partly generated by an algorithm ("bootstrapped"). Look at
  some of them before trusting them.
