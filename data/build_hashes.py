#!/usr/bin/env python
"""
Phase 1b: Generate hash maps for each ply dataset.

Loads each ply dataset (from build_plys.py) and creates hash_to_idx.json.
Each ply is processed independently with sequential reads.

Output: plys_data/ply_XXXX/hash_to_idx.json
"""
import os
import json
import hashlib
from pathlib import Path

from datasets import load_from_disk
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

SCRATCH_BASE = Path("/fs/scratch/PAS2836/lees_stuff")

os.environ["HF_HOME"] = str(SCRATCH_BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "datasets")

PLYS_DIR = SCRATCH_BASE / "plys_data"

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------


def board_hash(fen: str) -> int:
    """Hash FEN to 64-bit integer."""
    digest = hashlib.md5(fen.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def main():
    # Load metadata
    metadata_path = PLYS_DIR / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Run build_plys.py first.")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    plies = metadata["plies"]
    print(f"Found {len(plies)} plies (0 to {max(plies)})")
    print(f"Total positions: {metadata['total']:,}")

    # Process each ply
    for ply in tqdm(plies, desc="Building hash maps"):
        ply_dir = PLYS_DIR / f"ply_{ply:04d}"
        hash_path = ply_dir / "hash_to_idx.json"

        # Skip if already done
        if hash_path.exists():
            continue

        # Load ply dataset
        dataset = load_from_disk(str(ply_dir / "data"))

        # Build hash map (sequential columnar access)
        fens = dataset["fen"]
        hash_to_idx = {}
        for i, fen in enumerate(fens):
            h = board_hash(fen)
            hash_to_idx[h] = i

        # Save hash map
        with open(hash_path, "w") as f:
            json.dump({str(k): v for k, v in hash_to_idx.items()}, f)

    print(f"\nDone! Hash maps saved to {PLYS_DIR}/ply_XXXX/hash_to_idx.json")


if __name__ == "__main__":
    main()
