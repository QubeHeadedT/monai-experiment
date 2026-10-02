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

from common import load_params


def main() -> None:
    params = load_params()["prepare"]
    raise NotImplementedError("scripts/prepare.py: see the TODO list in the docstring")


if __name__ == "__main__":
    main()
