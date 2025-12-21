"""Download ATIS dataset (kaggle) or accept local files.
Usage:
    python -m data.download --dataset siddhadev/atis-dataset-clean
"""
import os
import argparse
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None


def download_kaggle_dataset(dataset: str, out_dir: str = "data/raw") -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if KaggleApi is None:
        msg = (
            "kaggle package not available. Either install kaggle and set KAGGLE_USERNAME/KAGGLE_KEY, "
            "or download the dataset manually and place files in data/raw/"
        )
        raise RuntimeError(msg)

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {dataset} to {out_dir} (this may take a while)")
    api.dataset_download_files(dataset, path=str(out_dir), unzip=True)
    return str(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="siddhadev/atis-dataset-clean")
    parser.add_argument("--out", type=str, default="data/raw")
    args = parser.parse_args()
    path = download_kaggle_dataset(args.dataset, args.out)
    print("Downloaded to:", path)
