#!/usr/bin/env python
"""Upload the mate-augmented dataset to HuggingFace Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_from_disk

DEFAULT_DATASET_PATH = Path("/fs/scratch/PAS2836/lees_stuff/searchless_mates_hf")
REPO_ID = "Ramora0/chess-av-mates"


def main():
    parser = argparse.ArgumentParser(description="Upload mate dataset to HuggingFace Hub")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the HuggingFace dataset saved by generate_mates.py (the _hf directory)",
    )
    parser.add_argument("--repo-id", type=str, default=REPO_ID)
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: No HuggingFace token. Set HF_TOKEN env var or pass --token")
        return 1

    if not args.data_dir.exists():
        print(f"ERROR: Dataset directory not found: {args.data_dir}")
        return 1

    print(f"Loading dataset from {args.data_dir}")
    dataset = load_from_disk(str(args.data_dir))
    print(f"Loaded {len(dataset):,} rows")

    print(f"Pushing dataset to {args.repo_id}...")
    dataset.push_to_hub(args.repo_id, token=token)

    print(f"Done. Dataset available at https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    exit(main())
