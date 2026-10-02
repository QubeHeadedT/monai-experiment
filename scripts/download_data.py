"""Stage 1 - download PanopTILs (refined) from Hugging Face and unpack it into PNGs.

Source: https://huggingface.co/datasets/histolytics-hub/panoptils_refined, a curated version of
the PanopTILs manual regions + bootstrapped nuclei release (1,349 ROIs, CC0-1.0). It ships as a
single ~2.8 GB Parquet file, cached by huggingface_hub (~/.cache/huggingface) so interrupted
downloads resume.

Outputs (one PNG per ROI and per array, all 1024x1024 at 0.25 MPP):
    data/raw/images/<roi>.png    RGB image
    data/raw/inst/<roi>.png      nucleus instance IDs (0 = background)
    data/raw/type/<roi>.png      nucleus class per pixel (codes in params.yaml prepare.class_map)
    data/raw/sem/<roi>.png       tissue region class per pixel
    data/raw/samples.csv         roi, slide_name, hospital

Usage:
    uv run scripts/download_data.py
"""

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "histolytics-hub/panoptils_refined"
REVISION = "d40293bc1ac0d0da8f271959fdb990aeea2d884f"  # pinned so re-runs get the same data
ARRAYS = ("image", "inst", "type", "sem")
DIRS = {"image": "images", "inst": "inst", "type": "type", "sem": "sem"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    print(f"Downloading {REPO_ID} (~2.8 GB, resumes if interrupted)...")
    parquet = hf_hub_download(REPO_ID, "panoptils_refined.parquet", repo_type="dataset", revision=REVISION)

    for name in DIRS.values():
        (args.out / name).mkdir(parents=True, exist_ok=True)
    table = pq.ParquetFile(parquet)
    rows = []
    with tqdm(total=table.metadata.num_rows, desc="Unpacking") as bar:
        # The PNG bytes are written as-is: no decoding, so pixel values are exactly as published.
        for batch in table.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                for array in ARRAYS:
                    (args.out / DIRS[array] / f"{row['sample']}.png").write_bytes(row[array])
                rows.append((row["sample"], row["slide_name"], row["hospital"]))
                bar.update()

    with open(args.out / "samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["roi", "slide_name", "hospital"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ROIs to {args.out}")


if __name__ == "__main__":
    main()
