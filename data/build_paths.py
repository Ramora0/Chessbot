#!/usr/bin/env python
"""
Build move history paths for each position in the dataset.

Single forward pass from ply 0 to max ply:
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

DATASET_PATH = SCRATCH_BASE / "action_value"
OUTPUT_DIR = SCRATCH_BASE / "paths_by_ply"

MAX_PATHS_PER_POSITION = 5
MIN_AVG_REGRET = 0.01  # 1%
MAX_AVG_REGRET = 0.08  # 8%

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

    # Compute avg regret for each path
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

    # Use preferred if available, otherwise fall back to all
    pool = preferred if preferred else candidate_paths

    if len(pool) <= max_paths:
        return pool

    # Sample for diversity: different sources and different regret levels
    # Group by source_hash
    by_source: Dict[int, List] = defaultdict(list)
    for path in pool:
        _, _, source_hash = path
        by_source[source_hash].append(path)

    selected = []
    sources = list(by_source.keys())
    random.shuffle(sources)

    # Round-robin from different sources
    while len(selected) < max_paths and any(by_source.values()):
        for src in sources:
            if by_source[src] and len(selected) < max_paths:
                # Pick path with median-ish regret from this source
                by_source[src].sort(key=avg_regret)
                mid = len(by_source[src]) // 2
                selected.append(by_source[src].pop(mid))

    return selected


# -------------------------------------------------------------------
# MAIN ALGORITHM
# -------------------------------------------------------------------


def build_index(dataset) -> Tuple[Dict[int, Set[int]], Dict[int, int]]:
    """
    Build:
    1. hashes_at_ply: ply -> set of position hashes
    2. hash_to_idx: hash -> dataset index (for loading position data)
    """
    print("Building index...")

    hashes_at_ply = defaultdict(set)
    hash_to_idx = {}

    for i in tqdm(range(len(dataset)), desc="Indexing"):
        fen = dataset[i]["fen"]
        ply = extract_ply(fen)
        h = board_hash(fen)

        hashes_at_ply[ply].add(h)
        hash_to_idx[h] = i

    max_ply = max(hashes_at_ply.keys()) if hashes_at_ply else 0
    print(f"Found {len(hashes_at_ply)} distinct plies (0 to {max_ply})")
    print(f"Indexed {len(hash_to_idx):,} positions")
    return dict(hashes_at_ply), hash_to_idx


def load_positions_by_hash(dataset, hashes: Set[int], hash_to_idx: Dict[int, int]) -> Dict[int, dict]:
    """Load position data for specific hashes. Returns hash -> {fen, moves, p_win, best_p_win}."""
    data = {}
    for h in hashes:
        if h not in hash_to_idx:
            continue
        i = hash_to_idx[h]
        row = dataset[i]
        fen = row["fen"]
        p_wins = list(row["p_win"])
        data[h] = {
            "fen": fen,
            "moves": list(row["moves"]),
            "p_win": p_wins,
            "best_p_win": max(p_wins) if p_wins else 0.5,
        }
    return data


def save_ply_paths(ply: int, paths_by_hash: Dict[int, List[Tuple[List[str], float, int]]]):
    """Save paths for a ply to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"ply_{ply:04d}.json"

    # Convert to JSON-serializable format
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


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(str(DATASET_PATH))
    print(f"Dataset has {len(dataset):,} positions")

    # Build index
    hashes_at_ply, hash_to_idx = build_index(dataset)
    plies = sorted(hashes_at_ply.keys())

    print(f"\nProcessing {len(plies)} plies...")

    # Initialize ply 0 with empty paths
    current_paths: Dict[int, List[Tuple[List[str], float, int]]] = {}

    if 0 in hashes_at_ply:
        for h in hashes_at_ply[0]:
            current_paths[h] = [([], 0.0, None)]  # empty moves, 0 regret, no source
        save_ply_paths(0, current_paths)
        print(f"Ply 0: {len(current_paths)} positions initialized")

    # Forward pass
    for ply in tqdm(plies, desc="Processing plies"):
        if ply == 0:
            continue

        prev_ply = ply - 1

        # Load previous ply paths if not in memory (resuming)
        if not current_paths and prev_ply in hashes_at_ply:
            current_paths = load_ply_paths(prev_ply)

        if not current_paths:
            print(f"Ply {ply}: No paths from previous ply, skipping")
            continue

        # Get hashes that exist at current ply (targets)
        target_hashes = hashes_at_ply.get(ply, set())
        if not target_hashes:
            current_paths = {}
            continue

        # Load position data for positions that have paths (previous ply)
        source_hashes = set(current_paths.keys())
        prev_data = load_positions_by_hash(dataset, source_hashes, hash_to_idx)

        # Compute candidate paths for each position at current ply
        next_paths: Dict[int, List[Tuple[List[str], float, int]]] = defaultdict(list)

        for source_hash, source_paths in current_paths.items():
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

                # Only add if target exists in dataset at this ply
                if target_hash not in target_hashes:
                    continue

                # Extend each source path with this move
                for moves, total_regret, _ in source_paths:
                    new_path = (
                        moves + [move_uci],
                        total_regret + regret,
                        source_hash,
                    )
                    next_paths[target_hash].append(new_path)

        # Filter and sample paths for each position
        current_paths = {
            h: filter_and_sample_paths(candidates)
            for h, candidates in next_paths.items()
        }

        # Save this ply
        save_ply_paths(ply, current_paths)

        if ply % 10 == 0:
            positions_with_paths = len(current_paths)
            total_at_ply = len(target_hashes)
            pct = 100 * positions_with_paths / total_at_ply if total_at_ply else 0
            print(f"Ply {ply}: {positions_with_paths:,}/{total_at_ply:,} positions have paths ({pct:.1f}%)")

    print(f"\nDone! Paths saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
