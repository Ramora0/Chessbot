#!/usr/bin/env python
"""
Phase 1a: Partition dataset by ply.

1. Add ply column to dataset (fast batched map)
2. Sort by ply (contiguous grouping)
3. Split into separate datasets per ply

Output: plys_data/ply_XXXX/data/
Run build_hashes.py after this to generate hash maps.
"""
import os
import json
from pathlib import Path
from typing import Dict

from datasets import load_from_disk
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

SCRATCH_BASE = Path("/fs/scratch/PAS2836/lees_stuff")

os.environ["HF_HOME"] = str(SCRATCH_BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "datasets")

DATASET_PATH = SCRATCH_BASE / "action_value"
OUTPUT_DIR = SCRATCH_BASE / "plys_data"

BATCH_SIZE = 10000  # For map operations

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------


def extract_ply(fen: str) -> int:
    """Extract ply from FEN: (fullmove-1)*2 + (1 if black else 0)."""
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return (fullmove - 1) * 2 + (1 if side == 'b' else 0)


def add_ply_batch(batch):
    """Add ply column to a batch of examples."""
    batch["ply"] = [extract_ply(fen) for fen in batch["fen"]]
    return batch


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(str(DATASET_PATH))
    print(f"Dataset has {len(dataset):,} positions")

    # Step 1: Add ply column
    print("\nStep 1: Adding ply column...")
    dataset = dataset.map(
        add_ply_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Computing plies",
    )

    # Step 2: Sort by ply
    print("\nStep 2: Sorting by ply...")
    dataset = dataset.sort("ply")
    print("Sort complete.")

    # Step 3: Find ply boundaries using binary search (data is sorted)
    print("\nStep 3: Finding ply boundaries...")
    n = len(dataset)
    min_ply = dataset[0]["ply"]
    max_ply = dataset[n - 1]["ply"]
    print(f"Ply range: {min_ply} to {max_ply}")

    def find_first(target_ply: int) -> int:
        """Binary search for first occurrence of target_ply."""
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if dataset[mid]["ply"] < target_ply:
                lo = mid + 1
            else:
                hi = mid
        return lo

    boundaries: Dict[int, tuple] = {}
    start_idx = 0

    for ply in tqdm(range(min_ply, max_ply + 1), desc="Finding boundaries"):
        # Find where next ply starts
        end_idx = find_first(ply + 1)
        if end_idx > start_idx:
            boundaries[ply] = (start_idx, end_idx)
        start_idx = end_idx

    plies = sorted(boundaries.keys())
    print(f"Found {len(plies)} distinct plies (0 to {max(plies)})")

    # Step 4: Split and save each ply
    print("\nStep 4: Splitting and saving ply datasets...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ply_counts = {}

    for ply in tqdm(plies, desc="Saving plies"):
        start, end = boundaries[ply]
        ply_dataset = dataset.select(range(start, end))

        ply_dir = OUTPUT_DIR / f"ply_{ply:04d}"
        ply_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset (without ply column - no longer needed)
        ply_dataset = ply_dataset.remove_columns(["ply"])
        ply_dataset.save_to_disk(str(ply_dir / "data"))

        ply_counts[ply] = end - start

    # Save metadata
    metadata = {
        "plies": plies,
        "counts": ply_counts,
        "total": len(dataset),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Ply datasets saved to {OUTPUT_DIR}/")
    print(f"Plies: {min(plies)} to {max(plies)}")
    print(f"Total positions: {len(dataset):,}")
    print("\nRun build_hashes.py next to generate hash maps.")


if __name__ == "__main__":
    main()
