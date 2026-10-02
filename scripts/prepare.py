"""Stage 2 - turn raw PanopTILs masks into HoVerNet targets and split the data.

Inputs:
    data/raw/tcga/rgbs/<roi>.png, data/raw/tcga/masks/<roi>.png
    data/raw/train_test_splits/fold_<k>_{train,test}.csv
    params.yaml: prepare.*

Outputs:
    data/prepared/<split>/<roi>.npz   image (H, W, 3) uint8, inst_map (H, W) int32, type_map (H, W) uint8
    data/prepared/splits.json         {"train": [roi, ...], "val": [...], "test": [...]}

TODO:
    1. Split by slide/hospital, never by ROI: ROIs from one slide must stay in one split.
       Use the official fold's test slides as the test set and hold out whole hospitals from its
       training slides for validation (slide name = ROI filename before "_xmin").
    2. Build an instance map: the masks have no instance IDs, only nucleus classes (channel 1) and
       nucleus boundary edges (channel 2). Separate touching nuclei using the edges
       (hint: skimage.measure.label, skimage.segmentation.expand_labels).
    3. Build a class map with prepare.class_map, one class per nucleus.
    4. Decide what to do with "exclude" pixels (nucleus code 0, about 1% of pixels).
"""

import json
from pathlib import Path

import numpy as np

from common import load_params

OUT_DIR = Path("data/prepared")


def main() -> None:
    params = load_params()["prepare"]

    # ---------------------------------------------------------------------------------------------
    # PLACEHOLDER - delete this block when you implement the stage.
    # Writes a few synthetic ROIs in the real output format so `dvc repro` can run end to end.
    # It does not read data/raw at all.
    # ---------------------------------------------------------------------------------------------
    print("WARNING: prepare.py is a placeholder - writing dummy data")
    rng = np.random.default_rng(params["seed"])
    n_classes = len(params["class_names"])
    yy, xx = np.mgrid[:1024, :1024]
    splits = {}
    for split in ("train", "val", "test"):
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)
        splits[split] = []
        for i in range(3):
            roi = f"DUMMY-{split}-{i}"
            image = np.full((1024, 1024, 3), 230, dtype=np.uint8)  # pale pink-ish background
            inst_map = np.zeros((1024, 1024), dtype=np.int32)
            type_map = np.zeros((1024, 1024), dtype=np.uint8)
            for nucleus_id in range(1, 51):  # 50 round "nuclei"
                cy, cx = rng.integers(20, 1004, size=2)
                disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= rng.integers(6, 12) ** 2
                image[disc] = (90, 40, 140)  # haematoxylin purple
                inst_map[disc] = nucleus_id
                type_map[disc] = rng.integers(1, n_classes)
            np.savez_compressed(OUT_DIR / split / f"{roi}.npz", image=image, inst_map=inst_map, type_map=type_map)
            splits[split].append(roi)
    (OUT_DIR / "splits.json").write_text(json.dumps(splits, indent=2))
    # ------------------------------------- END PLACEHOLDER ---------------------------------------


if __name__ == "__main__":
    main()
