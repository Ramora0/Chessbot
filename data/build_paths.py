#!/usr/bin/env python
"""
Build 8-move predecessor paths for each position in the dataset.

For each position, finds all sequences of up to 8 moves that could lead to it
(where all intermediate positions exist in the dataset), tracking cumulative
regret along each path.

Output: New HuggingFace dataset with paths embedded as a column.

Memory-efficient implementation: processes positions in fullmove buckets
using a sliding window to avoid loading the entire 500M+ position dataset.
"""
import os
import json
import heapq
import shutil
import hashlib
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import chess
import numpy as np
from datasets import Dataset, Features, Value, Sequence, load_from_disk
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Ensure ALL caches and temp files go to scratch, not home directory
SCRATCH_BASE = Path("/fs/scratch/PAS2836/lees_stuff")

# HF cache location
os.environ["HF_HOME"] = str(SCRATCH_BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "hub")

# Temp directories
temp_dir = SCRATCH_BASE / "tmp"
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(temp_dir)
os.environ["TEMP"] = str(temp_dir)
os.environ["TMP"] = str(temp_dir)

# Input/output paths
DATASET_PATH = SCRATCH_BASE / "action_value"
OUTPUT_PATH = SCRATCH_BASE / "action_value_with_paths"
PARTITIONED_DIR = SCRATCH_BASE / "fullmove_partitions"
PATHS_CACHE_DIR = SCRATCH_BASE / "paths_cache"

# Algorithm parameters
MAX_PATHS_PER_POSITION = 10  # Reduced from 100 to limit dataset size (~500GB total)
MAX_PATH_DEPTH = 8

# -------------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------------


@dataclass
class PositionData:
    """Data for a single position in the dataset."""
    fen: str
    moves: List[str]
    p_wins: List[float]
    best_p_win: float
    side: str
    halfmove: int
    fullmove: int


# Type aliases
PositionIndex = Dict[int, PositionData]
PredecessorGraph = Dict[int, List[Tuple[int, str, float]]]


# -------------------------------------------------------------------
# OUTPUT SCHEMA
# -------------------------------------------------------------------

# Schema for the new dataset with paths
# Paths are stored as parallel arrays for efficiency
OUTPUT_FEATURES = Features({
    "fen": Value("string"),
    "moves": Sequence(Value("string")),
    "p_win": Sequence(Value("float32")),
    # New fields for paths
    "path_moves": Sequence(Sequence(Value("string"))),  # [num_paths, path_len]
    "path_hashes": Sequence(Sequence(Value("int64"))),  # [num_paths, path_len]
    "path_regrets": Sequence(Value("float32")),  # [num_paths]
})


# -------------------------------------------------------------------
# FEN UTILITIES
# -------------------------------------------------------------------


def board_hash(fen: str) -> int:
    """
    Hash full FEN including halfmove and fullmove.
    Positions at different game stages are distinct.

    Uses MD5 truncated to 64 bits for deterministic hashing with low
    collision probability (~1 in 10^19 for 500M positions).
    """
    digest = hashlib.md5(fen.encode("utf-8")).digest()
    # Take first 8 bytes as unsigned 64-bit integer
    return int.from_bytes(digest[:8], "little")


def parse_fen_metadata(fen: str) -> Tuple[str, int, int]:
    """Extract (side, halfmove, fullmove) from FEN."""
    parts = fen.split()
    return parts[1], int(parts[4]), int(parts[5])


def get_max_path_depth(side: str, fullmove: int) -> int:
    """
    Get maximum path depth based on game progress.
    Early game positions can only have shorter paths.

    Returns the minimum of MAX_PATH_DEPTH and the number of ply
    that have been played to reach this position.
    """
    # Total ply (half-moves) that have been played to reach this position
    ply = (fullmove - 1) * 2 + (1 if side == 'b' else 0)
    return min(MAX_PATH_DEPTH, ply)


# -------------------------------------------------------------------
# PHASE 1: PARTITION BY FULLMOVE
# -------------------------------------------------------------------


def partition_by_fullmove(dataset) -> Dict[int, int]:
    """
    Partition positions by fullmove number into separate JSONL files.
    Returns a dict mapping fullmove -> count of positions.
    """
    print("Phase 1: Partitioning positions by fullmove number...")

    if PARTITIONED_DIR.exists():
        shutil.rmtree(PARTITIONED_DIR)
    PARTITIONED_DIR.mkdir(parents=True, exist_ok=True)

    # Track counts per fullmove
    fullmove_counts: Dict[int, int] = defaultdict(int)

    # Open file handles for each fullmove bucket
    file_handles: Dict[int, any] = {}

    try:
        for example in tqdm(dataset, desc="Partitioning"):
            fen = example["fen"]
            side, halfmove, fullmove = parse_fen_metadata(fen)

            # Get or create file handle for this fullmove
            if fullmove not in file_handles:
                path = PARTITIONED_DIR / f"fullmove_{fullmove:04d}.jsonl"
                file_handles[fullmove] = open(path, "w")

            # Write position data including original fields
            record = {
                "fen": fen,
                "moves": list(example["moves"]),
                "p_win": list(example["p_win"]),
            }
            file_handles[fullmove].write(json.dumps(record) + "\n")
            fullmove_counts[fullmove] += 1

    finally:
        for f in file_handles.values():
            f.close()

    print(f"Partitioned into {len(fullmove_counts)} fullmove buckets")
    print(f"Fullmove range: {min(fullmove_counts.keys())} to {max(fullmove_counts.keys())}")

    return dict(fullmove_counts)


def load_fullmove_bucket(fullmove: int) -> PositionIndex:
    """Load all positions from a specific fullmove bucket."""
    path = PARTITIONED_DIR / f"fullmove_{fullmove:04d}.jsonl"

    if not path.exists():
        return {}

    index: PositionIndex = {}

    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            fen = record["fen"]
            moves = record["moves"]
            p_wins = record["p_win"]

            h = board_hash(fen)
            side, halfmove, fm = parse_fen_metadata(fen)

            index[h] = PositionData(
                fen=fen,
                moves=moves,
                p_wins=p_wins,
                best_p_win=max(p_wins) if p_wins else 0.5,
                side=side,
                halfmove=halfmove,
                fullmove=fm,
            )

    return index


def load_fullmove_range(start_fm: int, end_fm: int) -> PositionIndex:
    """Load positions from a range of fullmove buckets."""
    combined: PositionIndex = {}

    for fm in range(start_fm, end_fm + 1):
        bucket = load_fullmove_bucket(fm)
        combined.update(bucket)

    return combined


# -------------------------------------------------------------------
# PHASE 2: BUILD TRANSITION GRAPH
# -------------------------------------------------------------------


def compute_transitions(pos: PositionData) -> List[Tuple[str, int, float]]:
    """
    Compute all transitions from this position.
    Returns: [(move_uci, target_hash, regret), ...]
    """
    board = chess.Board(pos.fen)
    transitions = []

    for move_uci, p_win in zip(pos.moves, pos.p_wins):
        try:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            result_fen = board.fen()
            target_hash = board_hash(result_fen)
            board.pop()

            regret = pos.best_p_win - p_win
            transitions.append((move_uci, target_hash, regret))
        except Exception:
            continue  # Skip invalid moves

    return transitions


def build_predecessor_graph(position_index: PositionIndex) -> PredecessorGraph:
    """
    Build inverted graph: target_hash -> [(source_hash, move, regret), ...]

    Only includes transitions where target exists in position_index (in-network).
    """
    predecessors: PredecessorGraph = defaultdict(list)

    for source_hash, pos in position_index.items():
        transitions = compute_transitions(pos)

        for move_uci, target_hash, regret in transitions:
            # Only include if target position is in our index
            if target_hash in position_index:
                predecessors[target_hash].append((source_hash, move_uci, regret))

    return dict(predecessors)


# -------------------------------------------------------------------
# PHASE 3: BACKWARD PATH ENUMERATION
# -------------------------------------------------------------------


def find_paths(
    target_hash: int,
    target_pos: PositionData,
    predecessors: PredecessorGraph,
    position_index: PositionIndex,
    max_paths: int = MAX_PATHS_PER_POSITION,
) -> List[Tuple[List[str], List[int], float]]:
    """
    Find up to max_paths paths leading to target.
    Path depth is variable based on game progress: min(8, ply_count).
    Uses min-heap to prioritize lowest cumulative regret.

    Returns list of (moves, position_hashes, cumulative_regret) tuples where:
    - position_hashes[i] = position BEFORE moves[i] is played
    - After playing all moves in order, you arrive at target
    """
    max_depth = get_max_path_depth(target_pos.side, target_pos.fullmove)

    if max_depth == 0:
        return []  # Starting position, no paths possible

    # Heap: (cumulative_regret, current_hash, path_moves, path_hashes, depth)
    heap = [(0.0, target_hash, [], [], 0)]
    paths = []
    visited_states: Set[Tuple[int, int]] = set()

    while heap and len(paths) < max_paths:
        regret, current_hash, path_moves, path_hashes, d = heapq.heappop(heap)

        # Skip if we've visited this (hash, depth) before with lower regret
        state = (current_hash, d)
        if state in visited_states:
            continue
        visited_states.add(state)

        if d == max_depth:
            # Found complete path - reverse to chronological order
            paths.append((
                list(reversed(path_moves)),
                list(reversed(path_hashes)),
                round(regret, 6),
            ))
            continue

        # Expand predecessors
        for pred_hash, move_uci, move_regret in predecessors.get(current_hash, []):
            if pred_hash not in position_index:
                continue  # Not in-network

            new_path_moves = path_moves + [move_uci]
            new_path_hashes = path_hashes + [pred_hash]
            new_regret = regret + move_regret

            heapq.heappush(heap, (
                new_regret,
                pred_hash,
                new_path_moves,
                new_path_hashes,
                d + 1,
            ))

    return paths


# -------------------------------------------------------------------
# PHASE 4: COMPUTE AND CACHE PATHS
# -------------------------------------------------------------------


def process_fullmove_bucket(current_fm: int) -> Dict[int, List[Tuple[List[str], List[int], float]]]:
    """
    Process a single fullmove bucket and return paths for each position.

    Returns: dict mapping target_hash -> list of (moves, hashes, regret) tuples
    """
    # Determine window bounds - need to look back MAX_PATH_DEPTH fullmoves
    window_start = max(1, current_fm - MAX_PATH_DEPTH)

    # Load positions in the window
    window_positions = load_fullmove_range(window_start, current_fm)

    if not window_positions:
        return {}

    # Get hashes of target positions (positions at current_fm)
    target_hashes = {
        h for h, pos in window_positions.items()
        if pos.fullmove == current_fm
    }

    if not target_hashes:
        return {}

    # Build predecessor graph for this window
    predecessors = build_predecessor_graph(window_positions)

    # Find paths for each target position
    results = {}
    for target_hash in target_hashes:
        target_pos = window_positions[target_hash]
        paths = find_paths(target_hash, target_pos, predecessors, window_positions)
        if paths:
            results[target_hash] = paths

    return results


def compute_all_paths(fullmove_counts: Dict[int, int]) -> None:
    """
    Compute paths for all positions and cache to disk.
    Writes one JSON file per fullmove bucket.
    """
    print("\nPhase 2: Computing predecessor paths...")

    if PATHS_CACHE_DIR.exists():
        shutil.rmtree(PATHS_CACHE_DIR)
    PATHS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fullmoves = sorted(fullmove_counts.keys())
    total_positions_with_paths = 0

    for current_fm in tqdm(fullmoves, desc="Computing paths"):
        paths_by_hash = process_fullmove_bucket(current_fm)

        if paths_by_hash:
            # Cache to disk
            cache_path = PATHS_CACHE_DIR / f"paths_{current_fm:04d}.json"
            # Convert to JSON-serializable format (hashes as strings for large ints)
            serializable = {
                str(h): [
                    {"moves": m, "hashes": [str(x) for x in hs], "regret": r}
                    for m, hs, r in path_list
                ]
                for h, path_list in paths_by_hash.items()
            }
            with open(cache_path, "w") as f:
                json.dump(serializable, f)

            total_positions_with_paths += len(paths_by_hash)

        # Force garbage collection to manage memory
        gc.collect()

    print(f"Found paths for {total_positions_with_paths:,} positions")


def load_paths_cache(fullmove: int) -> Dict[int, List[Tuple[List[str], List[int], float]]]:
    """Load cached paths for a fullmove bucket."""
    cache_path = PATHS_CACHE_DIR / f"paths_{fullmove:04d}.json"

    if not cache_path.exists():
        return {}

    with open(cache_path, "r") as f:
        data = json.load(f)

    # Convert back from JSON format
    return {
        int(h): [
            (p["moves"], [int(x) for x in p["hashes"]], p["regret"])
            for p in path_list
        ]
        for h, path_list in data.items()
    }


# -------------------------------------------------------------------
# PHASE 5: CREATE FINAL DATASET
# -------------------------------------------------------------------


def create_final_dataset(fullmove_counts: Dict[int, int]) -> None:
    """
    Create the final HuggingFace dataset with paths embedded.
    Processes fullmove buckets sequentially to manage memory.
    """
    print("\nPhase 3: Creating final dataset with paths...")

    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)

    fullmoves = sorted(fullmove_counts.keys())

    def generate_examples():
        """Generator that yields examples with paths."""
        for fm in tqdm(fullmoves, desc="Building dataset"):
            # Load original positions for this fullmove
            partition_path = PARTITIONED_DIR / f"fullmove_{fm:04d}.jsonl"
            if not partition_path.exists():
                continue

            # Load paths for this fullmove
            paths_by_hash = load_paths_cache(fm)

            with open(partition_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    fen = record["fen"]
                    h = board_hash(fen)

                    # Get paths for this position (or empty lists)
                    position_paths = paths_by_hash.get(h, [])

                    # Convert to parallel arrays
                    if position_paths:
                        path_moves = [p[0] for p in position_paths]
                        path_hashes = [p[1] for p in position_paths]
                        path_regrets = [p[2] for p in position_paths]
                    else:
                        path_moves = []
                        path_hashes = []
                        path_regrets = []

                    yield {
                        "fen": fen,
                        "moves": record["moves"],
                        "p_win": record["p_win"],
                        "path_moves": path_moves,
                        "path_hashes": path_hashes,
                        "path_regrets": path_regrets,
                    }

    # Create dataset from generator
    dataset = Dataset.from_generator(generate_examples, features=OUTPUT_FEATURES)

    print(f"Final dataset has {len(dataset):,} examples")
    print(f"Saving to {OUTPUT_PATH}...")
    dataset.save_to_disk(str(OUTPUT_PATH))
    print("Done!")


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------


def main():
    """Main entry point."""
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(str(DATASET_PATH))
    print(f"Dataset has {len(dataset):,} examples")
    print(f"Max paths per position: {MAX_PATHS_PER_POSITION}")
    print(f"Max path depth: {MAX_PATH_DEPTH}")

    # Phase 1: Partition by fullmove
    fullmove_counts = partition_by_fullmove(dataset)

    # Free dataset memory
    del dataset
    gc.collect()

    # Phase 2: Compute and cache paths
    compute_all_paths(fullmove_counts)

    # Phase 3: Create final dataset
    create_final_dataset(fullmove_counts)

    # Cleanup intermediate files
    print("\nCleaning up intermediate files...")
    shutil.rmtree(PARTITIONED_DIR)
    shutil.rmtree(PATHS_CACHE_DIR)
    print("Cleanup complete.")

    print(f"\nFinal dataset saved to: {OUTPUT_PATH}")


def verify_paths(sample_size: int = 100):
    """
    Verify a sample of paths for correctness.
    Run this after main() completes to validate output.
    """
    import random

    print(f"Loading dataset from {OUTPUT_PATH}...")
    dataset = load_from_disk(str(OUTPUT_PATH))
    print(f"Dataset has {len(dataset):,} examples")

    # Get indices of examples with paths
    indices_with_paths = [
        i for i in range(len(dataset))
        if len(dataset[i]["path_moves"]) > 0
    ]

    print(f"Found {len(indices_with_paths):,} examples with paths")

    if len(indices_with_paths) < sample_size:
        sample_indices = indices_with_paths
    else:
        sample_indices = random.sample(indices_with_paths, sample_size)

    print(f"Verifying {len(sample_indices)} random paths...")

    # Build hash -> FEN lookup from dataset
    print("Building hash lookup...")
    hash_to_fen = {}
    for i in tqdm(range(len(dataset)), desc="Indexing"):
        fen = dataset[i]["fen"]
        h = board_hash(fen)
        hash_to_fen[h] = fen

    errors = 0
    for idx in tqdm(sample_indices, desc="Verifying"):
        example = dataset[idx]
        target_fen = example["fen"]
        target_hash = board_hash(target_fen)

        for path_idx, (moves, hashes, regret) in enumerate(zip(
            example["path_moves"],
            example["path_hashes"],
            example["path_regrets"],
        )):
            if len(moves) != len(hashes):
                print(f"ERROR: moves/hashes length mismatch at {idx}, path {path_idx}")
                errors += 1
                continue

            if not hashes:
                continue

            # Get starting FEN from first hash
            starting_hash = hashes[0]
            if starting_hash not in hash_to_fen:
                print(f"ERROR: starting hash {starting_hash} not found")
                errors += 1
                continue

            starting_fen = hash_to_fen[starting_hash]
            board = chess.Board(starting_fen)

            # Verify each move and hash
            try:
                for i, (move_uci, expected_hash) in enumerate(zip(moves, hashes)):
                    actual_hash = board_hash(board.fen())
                    if actual_hash != expected_hash:
                        print(f"ERROR: Hash mismatch at step {i}")
                        errors += 1
                        break

                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        print(f"ERROR: Illegal move {move_uci} at step {i}")
                        errors += 1
                        break

                    board.push(move)

                # Verify we reached target
                final_hash = board_hash(board.fen())
                if final_hash != target_hash:
                    print(f"ERROR: Did not reach target")
                    errors += 1

            except Exception as e:
                print(f"ERROR: Exception during verification: {e}")
                errors += 1

    if errors == 0:
        print(f"All {len(sample_indices)} paths verified successfully!")
    else:
        print(f"Found {errors} errors in {len(sample_indices)} paths")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_paths()
    else:
        main()
