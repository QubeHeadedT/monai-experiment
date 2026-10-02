"""Stage 1 - download PanopTILs (manual regions + bootstrapped nuclei) from Google Drive.

Outputs:
    data/raw/tcga/rgbs/<roi>.png      1024x1024 RGB ROIs at 0.25 MPP (40x)
    data/raw/tcga/masks/<roi>.png     3-channel masks: region class, nucleus class, nucleus edges
    data/raw/train_test_splits/       official hospital-wise 5-fold splits (fold_<k>_{train,test}.csv)
    data/raw/region_summary.csv

Google Drive throttles bulk downloads (it can start refusing files after a few hundred).
Files already on disk are skipped, so re-run later to resume, or download the folder from
the browser instead (see README).

Usage:
    uv run scripts/download_data.py
    uv run scripts/download_data.py --extras csv vis   # also per-nucleus CSVs and visualisations
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gdown
from tqdm import tqdm

FOLDER_URL = "https://drive.google.com/drive/folders/1QOaUSz3zIoVwVVuyj3uAyfVbIdOTNMmZ"


def wanted(path: str, extras: list[str]) -> bool:
    parts = path.split("/")
    if parts[0] == "tcga":
        return parts[1] in {"rgbs", "masks", *extras}
    return True  # train_test_splits/ and region_summary.csv


def fetch(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    gdown.download(id=file_id, output=str(tmp), quiet=True, retries=3)
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--extras", nargs="*", default=[], choices=["csv", "vis"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("Listing the Drive folder (takes a minute)...")
    listing = gdown.download_folder(url=FOLDER_URL, skip_download=True, quiet=True)
    todo = [(f.id, args.out / f.path) for f in listing if wanted(f.path, args.extras)]
    todo = [(i, dest) for i, dest in todo if not dest.exists()]
    print(f"{len(listing)} files in folder, {len(todo)} to download into {args.out}")

    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, i, dest) for i, dest in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            failed += future.exception() is not None
    if failed:
        raise SystemExit(f"{failed} downloads failed (Drive throttling?). Re-run later to resume.")


if __name__ == "__main__":
    main()
