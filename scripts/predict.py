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

from common import load_params


def main() -> None:
    params = load_params()["predict"]
    raise NotImplementedError("scripts/predict.py: see the TODO list in the docstring")


if __name__ == "__main__":
    main()
