"""Stage 4 - run the trained model over whole test ROIs.

Inputs:
    models/hovernet.pt
    data/prepared/test/*.npz (only the image is used)
    params.yaml: predict.*

Outputs:
    data/predictions/test/<roi>.npy   int32 array (2, H, W): [0] instance map (0 = background),
                                      [1] class map (codes as in prepare.class_names)

TODO:
    1. Sliding-window inference over each 1024x1024 ROI:
       monai.apps.pathology.inferers.SlidingWindowHoVerNetInferer.
    2. Post-processing: HoVerNetInstanceMapPostProcessing, then HoVerNetNuclearTypePostProcessing
       (monai.apps.pathology.transforms.post.array).
"""

from pathlib import Path

import numpy as np

from common import load_params

IN_DIR = Path("data/prepared/test")
OUT_DIR = Path("data/predictions/test")


def main() -> None:
    params = load_params()["predict"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------------------------
    # PLACEHOLDER - delete this block when you implement the stage.
    # No model is used: it writes an empty prediction (no nuclei found) for every test ROI, in the
    # real output format.
    # ---------------------------------------------------------------------------------------------
    print("WARNING: predict.py is a placeholder - writing empty predictions")
    for path in sorted(IN_DIR.glob("*.npz")):
        with np.load(path) as roi:
            height, width = roi["image"].shape[:2]
        prediction = np.zeros((2, height, width), dtype=np.int32)  # [instance map, class map]
        np.save(OUT_DIR / f"{path.stem}.npy", prediction)
    # ------------------------------------- END PLACEHOLDER ---------------------------------------


if __name__ == "__main__":
    main()
