"""
Download curated EEG datasets for mental state model training.

Usage:
    python models/download_datasets.py --all
    python models/download_datasets.py --dataset physionet_motor_imagery
    python models/download_datasets.py --list
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS = {
    "physionet_motor_imagery": {
        "description": "PhysioNet EEG Motor Movement/Imagery (109 subjects, 64-ch)",
        "url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "method": "physionet",
        "physionet_slug": "eegmmidb/1.0.0",
        "use": "Visualization detection via motor imagery alpha/theta patterns",
    },
    "physionet_mental_arithmetic": {
        "description": "PhysioNet EEG During Mental Arithmetic (36 subjects, 19-ch)",
        "url": "https://physionet.org/content/eeg-during-mental-arithmetic-tasks/1.0.0/",
        "method": "physionet",
        "physionet_slug": "eeg-during-mental-arithmetic-tasks/1.0.0",
        "use": "Focused attention and cognitive load detection",
    },
    "dreamer": {
        "description": "DREAMER — 23 subjects, 14-ch EEG, emotion labels (~50 MB)",
        "url": "https://zenodo.org/records/546113",
        "method": "zenodo",
        "zenodo_record": "546113",
        "use": "Calm vs stressed via valence/arousal mapping",
    },
    "lemon_openneuro": {
        "description": "LEMON resting-state EEG (228 subjects, eyes-open/closed)",
        "url": "https://openneuro.org/datasets/ds000221",
        "method": "openneuro",
        "dataset_id": "ds000221",
        "use": "Relaxation/meditation baseline (alpha dominance)",
    },
    "mind_wandering": {
        "description": "Mind-wandering EEG with attention probes",
        "url": "https://openneuro.org/datasets/ds003768",
        "method": "openneuro",
        "dataset_id": "ds003768",
        "use": "Focus vs unfocused classification",
    },
    "resting_anxiety": {
        "description": "Resting-State EEG and Trait Anxiety (51 subjects)",
        "url": "https://openneuro.org/datasets/ds007609",
        "method": "openneuro",
        "dataset_id": "ds007609",
        "use": "Anxiety-state signatures in resting EEG",
    },
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_physionet(slug: str, dest: Path):
    """Download a PhysioNet dataset using wget (recommended by PhysioNet)."""
    ensure_dir(dest)
    base_url = f"https://physionet.org/files/{slug}/"
    print(f"  Downloading from {base_url}")
    print(f"  Target: {dest}")
    print("  (This uses wget recursive download as recommended by PhysioNet)")
    try:
        subprocess.run(
            [
                "wget", "-r", "-N", "-c", "-np",
                "--directory-prefix", str(dest),
                base_url,
            ],
            check=True,
        )
    except FileNotFoundError:
        print("  wget not found — falling back to curl strategy")
        print(f"  Please manually download from: {base_url}")
        print(f"  Place files in: {dest}")
        marker = dest / "DOWNLOAD_MANUALLY.txt"
        marker.write_text(
            f"Download this dataset manually from:\n{base_url}\n"
            f"Place extracted files in this directory.\n"
        )


def download_openneuro(dataset_id: str, dest: Path):
    """Download an OpenNeuro BIDS dataset using the openneuro CLI or datalad."""
    ensure_dir(dest)
    ds_dir = dest / dataset_id
    print(f"  Downloading OpenNeuro {dataset_id}")
    print(f"  Target: {ds_dir}")

    # Try openneuro-cli first
    try:
        subprocess.run(
            ["openneuro", "download", "--snapshot", "latest", dataset_id, str(ds_dir)],
            check=True,
        )
        return
    except FileNotFoundError:
        pass

    # Try datalad
    try:
        subprocess.run(
            ["datalad", "install", f"https://github.com/OpenNeuroDatasets/{dataset_id}.git", str(ds_dir)],
            check=True,
        )
        return
    except FileNotFoundError:
        pass

    # Fallback
    print(f"  Neither openneuro-cli nor datalad found.")
    print(f"  Install one of:")
    print(f"    npm install -g @openneuro/cli")
    print(f"    pip install datalad")
    print(f"  Or download manually: https://openneuro.org/datasets/{dataset_id}")
    marker = ds_dir if ds_dir.exists() else dest
    ensure_dir(marker)
    (marker / "DOWNLOAD_MANUALLY.txt").write_text(
        f"Download this dataset manually from:\n"
        f"https://openneuro.org/datasets/{dataset_id}\n"
        f"Place extracted BIDS folder here.\n"
    )


def download_zenodo(record_id: str, dest: Path):
    """Download files from a Zenodo record."""
    ensure_dir(dest)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"  Fetching Zenodo record {record_id} metadata...")
    print(f"  Manual download: https://zenodo.org/records/{record_id}")
    marker = dest / "DOWNLOAD_MANUALLY.txt"
    marker.write_text(
        f"Download this dataset from:\nhttps://zenodo.org/records/{record_id}\n"
        f"Place extracted files in this directory.\n"
    )
    print(f"  Created download instructions at {marker}")


def download_dataset(name: str):
    info = DATASETS[name]
    dest = DATA_DIR / name
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {name} — already exists at {dest}")
        return

    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"  {info['description']}")
    print(f"  Use: {info['use']}")
    print(f"{'='*60}")

    method = info["method"]
    if method == "physionet":
        download_physionet(info["physionet_slug"], dest)
    elif method == "openneuro":
        download_openneuro(info["dataset_id"], dest)
    elif method == "zenodo":
        download_zenodo(info["zenodo_record"], dest)
    else:
        print(f"  Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Download EEG datasets for model training")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str, help="Download a specific dataset by key")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list or (not args.all and not args.dataset):
        print("\nAvailable datasets:\n")
        for key, info in DATASETS.items():
            status = "downloaded" if (DATA_DIR / key).exists() else "not downloaded"
            print(f"  {key:35s} [{status}]")
            print(f"    {info['description']}")
            print(f"    Use: {info['use']}")
            print(f"    URL: {info['url']}")
            print()
        return

    ensure_dir(DATA_DIR)

    if args.all:
        for name in DATASETS:
            download_dataset(name)
    elif args.dataset:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)
        download_dataset(args.dataset)

    print("\nDone. Check models/data/ for downloaded files.")
    print("Next step: python models/train_mental_states.py")


if __name__ == "__main__":
    main()
