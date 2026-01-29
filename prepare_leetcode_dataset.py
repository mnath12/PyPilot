"""
Download newfacade/LeetCodeDataset and materialize train/test splits locally.

Usage (from repo root):
  python scripts/prepare_leetcode_dataset.py --out data/leetcode

Optional:
  python scripts/prepare_leetcode_dataset.py --out data/leetcode --train_frac 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset


DATASET_ID = "newfacade/LeetCodeDataset"


def load_leetcode_dataset(train_frac: float = 1.0, seed: int = 42) -> DatasetDict:
    """
    Loads LeetCodeDataset from Hugging Face.

    Returns a DatasetDict with at least:
      - "train"
      - "test"

    If the hosted dataset already provides train/test splits (it should), we use them.
    Otherwise, we fall back to loading the explicit JSONL data files.
    """
    ds = load_dataset(DATASET_ID)

    # Most likely path: the dataset repo exposes train/test already.
    if isinstance(ds, DatasetDict) and "train" in ds and "test" in ds:
        out = ds
    else:
        # Fallback: load explicit files from the dataset repo
        # (These filenames exist in the dataset repo.)
        out = load_dataset(
            DATASET_ID,
            data_files={
                "train": "LeetCodeDataset-train.jsonl",
                "test": "LeetCodeDataset-test.jsonl",
            },
        )

    # Optional: downsample train for faster iteration (keeps test intact).
    if train_frac < 1.0:
        if not (0.0 < train_frac <= 1.0):
            raise ValueError("--train_frac must be in (0, 1].")
        out["train"] = out["train"].shuffle(seed=seed).select(
            range(int(len(out["train"]) * train_frac))
        )

    return out


def main(out_dir: str, train_frac: float, seed: int) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds = load_leetcode_dataset(train_frac=train_frac, seed=seed)

    print("Loaded splits:", list(ds.keys()))
    print("Train rows:", len(ds["train"]))
    print("Test rows:", len(ds["test"]))
    print("Columns:", ds["train"].column_names)

    # Save to disk so you don't re-download every time
    ds.save_to_disk(str(out_path))
    print(f"Saved DatasetDict to: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory for save_to_disk")
    parser.add_argument("--train_frac", type=float, default=1.0, help="Downsample train split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(out_dir=args.out, train_frac=args.train_frac, seed=args.seed)
