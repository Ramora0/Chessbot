#!/usr/bin/env python
"""
Phase 1: Partition dataset by ply.

Reads the original action_value dataset and creates:
- One dataset per ply value in plys_data/ply_XXXX/
- One hash map per ply in plys_data/ply_XXXX/hash_to_idx.json

This enables sequential access in phase 2 (build_paths.py).
"""
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from multiprocessing import Pool

from datasets import load_from_disk, Dataset
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

SCRATCH_BASE = Path("/fs/scratch/PAS2836/lees_stuff")

os.environ["HF_HOME"] = str(SCRATCH_BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "datasets")

DATASET_PATH = SCRATCH_BASE / "action_value"
OUTPUT_DIR = SCRATCH_BASE / "plys_data"

NUM_WORKERS = 10
CHUNK_SIZE = 50000

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------


def board_hash(fen: str) -> int:
    """Hash FEN to 64-bit integer."""
    digest = hashlib.md5(fen.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def extract_ply(fen: str) -> int:
    """Extract ply from FEN: (fullmove-1)*2 + (1 if black else 0)."""
    parts = fen.split()
    side = parts[1]
    fullmove = int(parts[5])
    return (fullmove - 1) * 2 + (1 if side == 'b' else 0)


def process_fen_batch(args: Tuple[int, List[str]]) -> List[Tuple[int, int, int]]:
    """Process a batch of FENs. Returns list of (index, ply, hash)."""
    start_idx, fens = args
    results = []
    for i, fen in enumerate(fens):
        ply = extract_ply(fen)
        h = board_hash(fen)
        results.append((start_idx + i, ply, h))
    return results


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def build_ply_index(dataset) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build index: ply -> list of (dataset_index, hash).

    Uses parallel processing for CPU-bound ply/hash computation.
    """
    n = len(dataset)
    num_batches = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Processing {n:,} positions ({NUM_WORKERS} workers, {CHUNK_SIZE} batch size)...")

    indices_by_ply: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    with Pool(NUM_WORKERS) as pool:
        pending = []

        for start in tqdm(range(0, n, CHUNK_SIZE), total=num_batches, desc="Indexing"):
            end = min(start + CHUNK_SIZE, n)
            batch_fens = list(dataset.select(range(start, end))["fen"])
            pending.append(pool.apply_async(process_fen_batch, ((start, batch_fens),)))

            # Collect results periodically
            if len(pending) >= NUM_WORKERS * 2:
                result = pending.pop(0).get()
                for idx, ply, h in result:
                    indices_by_ply[ply].append((idx, h))

        # Collect remaining
        for future in pending:
            result = future.get()
            for idx, ply, h in result:
                indices_by_ply[ply].append((idx, h))

    return dict(indices_by_ply)


def save_ply_dataset(dataset, ply: int, entries: List[Tuple[int, int]]):
    """
    Save dataset subset and hash map for a single ply.

    entries: list of (dataset_index, hash)
    """
    ply_dir = OUTPUT_DIR / f"ply_{ply:04d}"
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Sort by index for sequential access when reading
    entries.sort(key=lambda x: x[0])
    indices = [idx for idx, _ in entries]
    hashes = [h for _, h in entries]

    # Save hash -> local_index map (index within this ply's dataset)
    hash_to_idx = {h: i for i, h in enumerate(hashes)}
    with open(ply_dir / "hash_to_idx.json", "w") as f:
        json.dump({str(k): v for k, v in hash_to_idx.items()}, f)

    # Select and save dataset subset
    ply_dataset = dataset.select(indices)
    ply_dataset.save_to_disk(str(ply_dir / "data"))


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(str(DATASET_PATH))
    print(f"Dataset has {len(dataset):,} positions")

    # Build index
    indices_by_ply = build_ply_index(dataset)
    plies = sorted(indices_by_ply.keys())
    print(f"\nFound {len(plies)} distinct plies (0 to {max(plies)})")

    # Save each ply
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ply in tqdm(plies, desc="Saving ply datasets"):
        entries = indices_by_ply[ply]
        save_ply_dataset(dataset, ply, entries)

    # Save metadata
    metadata = {
        "plies": plies,
        "counts": {ply: len(indices_by_ply[ply]) for ply in plies},
        "total": len(dataset),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Ply datasets saved to {OUTPUT_DIR}/")
    print(f"Plies: {min(plies)} to {max(plies)}")
    print(f"Total positions: {len(dataset):,}")


if __name__ == "__main__":
    main()
