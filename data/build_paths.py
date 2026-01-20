#!/usr/bin/env python
"""
Phase 2: Build move history paths using ply-partitioned datasets.

Reads adjacent ply datasets (from build_plys.py) and builds paths:
- For each position, find up to 5 move sequences from game start
- All intermediate positions must exist in the dataset (in-network)
- Paths filtered by average regret (prefer 1-8% range)
- Paths sampled for diversity (different sources, different regret levels)

Output: One file per ply in paths_by_ply/ directory
"""
import os
import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import chess
from datasets import load_from_disk
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

SCRATCH_BASE = Path("/fs/scratch/PAS2836/lees_stuff")

os.environ["HF_HOME"] = str(SCRATCH_BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "datasets")

PLYS_DIR = SCRATCH_BASE / "plys_data"
OUTPUT_DIR = SCRATCH_BASE / "paths_by_ply"

MAX_PATHS_PER_POSITION = 5
MIN_AVG_REGRET = 0.01  # 1%
MAX_AVG_REGRET = 0.08  # 8%

# Show progress bar for plies with more than this many positions
VERBOSE_PLY_THRESHOLD = 1000

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------


def board_hash(fen: str) -> int:
    """Hash FEN to 64-bit integer."""
    digest = hashlib.md5(fen.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


# -------------------------------------------------------------------
# PATH FILTERING & SAMPLING
# -------------------------------------------------------------------


def filter_and_sample_paths(
    candidate_paths: List[Tuple[List[str], float, int]],
    max_paths: int = MAX_PATHS_PER_POSITION,
) -> List[Tuple[List[str], float, int]]:
    """
    Filter and sample paths down to max_paths.

    Each path is (moves, total_regret, source_hash).

    Strategy:
    1. Compute avg_regret for each path
    2. Prefer paths with 1% <= avg_regret <= 8%
    3. If still > max_paths, sample for diversity:
       - Different source positions
       - Different avg_regret values
    """
    if len(candidate_paths) <= max_paths:
        return candidate_paths

    def avg_regret(path):
        moves, total_regret, _ = path
        return total_regret / len(moves) if moves else 0.0

    # Split into preferred (1-8% avg regret) and other
    preferred = []
    other = []
    for path in candidate_paths:
        ar = avg_regret(path)
        if MIN_AVG_REGRET <= ar <= MAX_AVG_REGRET:
            preferred.append(path)
        else:
            other.append(path)

    pool = preferred if preferred else candidate_paths

    if len(pool) <= max_paths:
        return pool

    # Sample for diversity: different sources and different regret levels
    by_source: Dict[int, List] = defaultdict(list)
    for path in pool:
        _, _, source_hash = path
        by_source[source_hash].append(path)

    selected = []
    sources = list(by_source.keys())
    random.shuffle(sources)

    while len(selected) < max_paths and any(by_source.values()):
        for src in sources:
            if by_source[src] and len(selected) < max_paths:
                by_source[src].sort(key=avg_regret)
                mid = len(by_source[src]) // 2
                selected.append(by_source[src].pop(mid))

    return selected


# -------------------------------------------------------------------
# PLY DATA LOADING
# -------------------------------------------------------------------


def load_ply_data(ply: int) -> Tuple[any, Dict[int, int], Set[int]]:
    """
    Load data for a single ply.

    Returns:
    - dataset: HuggingFace dataset for this ply
    - hash_to_idx: hash -> index within this ply's dataset
    - hashes: set of all hashes at this ply
    """
    ply_dir = PLYS_DIR / f"ply_{ply:04d}"

    if not ply_dir.exists():
        return None, {}, set()

    # Load hash map
    with open(ply_dir / "hash_to_idx.json", "r") as f:
        raw = json.load(f)
    hash_to_idx = {int(k): v for k, v in raw.items()}

    # Load dataset
    dataset = load_from_disk(str(ply_dir / "data"))

    return dataset, hash_to_idx, set(hash_to_idx.keys())


def load_positions_from_ply(
    dataset, hash_to_idx: Dict[int, int], hashes: Set[int], chunk_size: int = 10000
) -> Dict[int, dict]:
    """
    Load position data for specific hashes from a ply dataset.

    Returns hash -> {fen, moves, p_win, best_p_win}

    Uses chunked batch loading to balance speed and memory usage.
    """
    # Filter to valid hashes and get their indices
    valid_hashes = [h for h in hashes if h in hash_to_idx]
    if not valid_hashes:
        return {}

    indices = [hash_to_idx[h] for h in valid_hashes]

    data = {}

    # Process in chunks to avoid OOM
    for chunk_start in range(0, len(valid_hashes), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(valid_hashes))
        chunk_hashes = valid_hashes[chunk_start:chunk_end]
        chunk_indices = indices[chunk_start:chunk_end]

        # Batch fetch this chunk
        rows = dataset.select(chunk_indices)

        for h, row in zip(chunk_hashes, rows):
            p_wins = row["p_win"]
            data[h] = {
                "fen": row["fen"],
                "moves": row["moves"],
                "p_win": p_wins,
                "best_p_win": max(p_wins) if p_wins else 0.5,
            }

    return data


# -------------------------------------------------------------------
# PATH PERSISTENCE
# -------------------------------------------------------------------


def save_ply_paths(ply: int, paths_by_hash: Dict[int, List[Tuple[List[str], float, int]]]):
    """Save paths for a ply to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"ply_{ply:04d}.json"

    data = {
        str(h): [
            {"moves": moves, "total_regret": round(regret, 6), "source_hash": src}
            for moves, regret, src in path_list
        ]
        for h, path_list in paths_by_hash.items()
    }

    with open(path, "w") as f:
        json.dump(data, f)


def load_ply_paths(ply: int) -> Dict[int, List[Tuple[List[str], float, int]]]:
    """Load paths for a ply from disk."""
    path = OUTPUT_DIR / f"ply_{ply:04d}.json"
    if not path.exists():
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    return {
        int(h): [
            (p["moves"], p["total_regret"], p["source_hash"])
            for p in path_list
        ]
        for h, path_list in data.items()
    }


def find_last_completed_ply() -> int:
    """Find the highest ply that has been saved to disk."""
    if not OUTPUT_DIR.exists():
        return -1

    completed = []
    for f in OUTPUT_DIR.glob("ply_*.json"):
        try:
            ply = int(f.stem.split("_")[1])
            completed.append(ply)
        except (ValueError, IndexError):
            continue

    return max(completed) if completed else -1


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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint
    last_completed = find_last_completed_ply()
    current_paths: Dict[int, List[Tuple[List[str], float, int]]] = {}
    prev_dataset = None
    prev_hash_to_idx = {}
    prev_hashes = set()

    if last_completed >= 0:
        print(f"Resuming from checkpoint: ply {last_completed}")
        current_paths = load_ply_paths(last_completed)
        prev_dataset, prev_hash_to_idx, prev_hashes = load_ply_data(last_completed)
        # Filter plies to only process those after the checkpoint
        plies = [p for p in plies if p > last_completed]
        if not plies:
            print("All plies already processed!")
            return
        print(f"Remaining plies to process: {len(plies)} ({min(plies)} to {max(plies)})")
    else:
        # Initialize ply 0
        if 0 in plies:
            _, hash_to_idx_0, hashes_0 = load_ply_data(0)
            for h in hashes_0:
                current_paths[h] = [([], 0.0, None)]
            save_ply_paths(0, current_paths)
            print(f"Ply 0: {len(current_paths)} positions initialized")

    # Forward pass
    for ply in tqdm(plies, desc="Processing plies"):
        if ply == 0:
            # Load ply 0 data for next iteration
            prev_dataset, prev_hash_to_idx, prev_hashes = load_ply_data(0)
            continue

        # Load current ply data (targets)
        curr_dataset, curr_hash_to_idx, curr_hashes = load_ply_data(ply)

        if not curr_hashes:
            current_paths = {}
            prev_dataset, prev_hash_to_idx, prev_hashes = curr_dataset, curr_hash_to_idx, curr_hashes
            continue

        # Load previous ply paths if not in memory (resuming)
        if not current_paths and (ply - 1) in plies:
            current_paths = load_ply_paths(ply - 1)

        if not current_paths:
            print(f"Ply {ply}: No paths from previous ply, skipping")
            prev_dataset, prev_hash_to_idx, prev_hashes = curr_dataset, curr_hash_to_idx, curr_hashes
            continue

        # Load position data for source positions (those with paths)
        source_hashes = set(current_paths.keys())
        prev_data = load_positions_from_ply(prev_dataset, prev_hash_to_idx, source_hashes)

        # Compute candidate paths for each position at current ply
        next_paths: Dict[int, List[Tuple[List[str], float, int]]] = defaultdict(list)

        # Show progress for large plies
        source_items = current_paths.items()
        if len(current_paths) >= VERBOSE_PLY_THRESHOLD:
            source_items = tqdm(
                source_items,
                desc=f"  Ply {ply} positions",
                total=len(current_paths),
                leave=False,
            )

        for source_hash, source_paths in source_items:
            if source_hash not in prev_data:
                continue

            pos = prev_data[source_hash]
            board = chess.Board(pos["fen"])

            for move_uci, p_win in zip(pos["moves"], pos["p_win"]):
                regret = pos["best_p_win"] - p_win

                try:
                    move = chess.Move.from_uci(move_uci)
                    board.push(move)
                    target_hash = board_hash(board.fen())
                    board.pop()
                except:
                    continue

                # Only add if target exists at this ply
                if target_hash not in curr_hashes:
                    continue

                # Extend each source path with this move
                for moves, total_regret, _ in source_paths:
                    new_path = (
                        moves + [move_uci],
                        total_regret + regret,
                        source_hash,
                    )
                    next_paths[target_hash].append(new_path)

        # Filter and sample paths
        filter_items = next_paths.items()
        if len(next_paths) >= VERBOSE_PLY_THRESHOLD:
            filter_items = tqdm(
                filter_items,
                desc=f"  Ply {ply} filtering",
                total=len(next_paths),
                leave=False,
            )
        current_paths = {
            h: filter_and_sample_paths(candidates)
            for h, candidates in filter_items
        }

        # Save this ply
        save_ply_paths(ply, current_paths)

        # Shift: current becomes previous
        prev_dataset, prev_hash_to_idx, prev_hashes = curr_dataset, curr_hash_to_idx, curr_hashes

        if ply % 10 == 0:
            positions_with_paths = len(current_paths)
            total_at_ply = len(curr_hashes)
            pct = 100 * positions_with_paths / total_at_ply if total_at_ply else 0
            print(f"Ply {ply}: {positions_with_paths:,}/{total_at_ply:,} positions have paths ({pct:.1f}%)")

    print(f"\nDone! Paths saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
