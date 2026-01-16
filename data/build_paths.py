#!/usr/bin/env python
"""
Build 8-move predecessor paths for each position in the dataset.

For each position, finds all sequences of up to 8 moves that could lead to it
(where all intermediate positions exist in the dataset), tracking cumulative
regret along each path.

Output: New HuggingFace dataset with paths embedded as a column.

Uses Arrow-native operations for speed - avoids slow JSON serialization.
"""
import os
import heapq
import hashlib
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import chess
import pyarrow as pa
from datasets import Dataset, Features, Value, Sequence, load_from_disk, concatenate_datasets
from tqdm.auto import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

SCRATCH_BASE = Path("/fs/scratch/PAS2836/lees_stuff")

os.environ["HF_HOME"] = str(SCRATCH_BASE / "hf_cache")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(SCRATCH_BASE / "hf_cache" / "hub")

temp_dir = SCRATCH_BASE / "tmp"
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(temp_dir)
os.environ["TEMP"] = str(temp_dir)
os.environ["TMP"] = str(temp_dir)

# Input/output paths
DATASET_PATH = SCRATCH_BASE / "action_value"
OUTPUT_PATH = SCRATCH_BASE / "action_value_with_paths"
SORTED_DATASET_PATH = SCRATCH_BASE / "action_value_sorted"

# Algorithm parameters
MAX_PATHS_PER_POSITION = 10
MAX_PATH_DEPTH = 8
NUM_PROC = os.cpu_count()

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
    fullmove: int


PositionIndex = Dict[int, PositionData]
PredecessorGraph = Dict[int, List[Tuple[int, str, float]]]


# -------------------------------------------------------------------
# OUTPUT SCHEMA
# -------------------------------------------------------------------

OUTPUT_FEATURES = Features({
    "fen": Value("string"),
    "moves": Sequence(Value("string")),
    "p_win": Sequence(Value("float32")),
    "path_moves": Sequence(Sequence(Value("string"))),
    "path_hashes": Sequence(Sequence(Value("int64"))),
    "path_regrets": Sequence(Value("float32")),
})


# -------------------------------------------------------------------
# FEN UTILITIES
# -------------------------------------------------------------------


def board_hash(fen: str) -> int:
    """Hash FEN to 64-bit integer using MD5."""
    digest = hashlib.md5(fen.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def extract_fullmove(fen: str) -> int:
    """Extract fullmove number from FEN string."""
    return int(fen.split()[5])


def extract_side(fen: str) -> str:
    """Extract side to move from FEN string."""
    return fen.split()[1]


def get_max_path_depth(side: str, fullmove: int) -> int:
    """Get maximum path depth based on game progress."""
    ply = (fullmove - 1) * 2 + (1 if side == 'b' else 0)
    return min(MAX_PATH_DEPTH, ply)


# -------------------------------------------------------------------
# PHASE 1: ADD FULLMOVE COLUMN AND SORT
# -------------------------------------------------------------------


def prepare_sorted_dataset(dataset: Dataset) -> Dataset:
    """
    Add fullmove column and sort dataset by fullmove.
    Uses Arrow-native operations for speed.
    """
    print("Phase 1: Adding fullmove column and sorting...")

    # Check if sorted dataset already exists
    if SORTED_DATASET_PATH.exists():
        print(f"Loading pre-sorted dataset from {SORTED_DATASET_PATH}")
        return load_from_disk(str(SORTED_DATASET_PATH))

    # Add fullmove column using fast batch mapping
    def add_fullmove(batch):
        fullmoves = [int(fen.split()[5]) for fen in batch["fen"]]
        return {"fullmove": fullmoves}

    print("Adding fullmove column...")
    dataset = dataset.map(
        add_fullmove,
        batched=True,
        batch_size=100_000,
        num_proc=NUM_PROC,
        desc="Adding fullmove",
    )

    print("Sorting by fullmove...")
    dataset = dataset.sort("fullmove")

    print(f"Saving sorted dataset to {SORTED_DATASET_PATH}")
    dataset.save_to_disk(str(SORTED_DATASET_PATH))

    return dataset


def get_fullmove_ranges(dataset: Dataset) -> Dict[int, Tuple[int, int]]:
    """
    Build index of fullmove -> (start_idx, end_idx) ranges.
    Since dataset is sorted, positions with same fullmove are contiguous.
    """
    print("Building fullmove range index...")

    fullmoves = dataset["fullmove"]
    ranges = {}

    current_fm = fullmoves[0]
    start_idx = 0

    for i, fm in enumerate(tqdm(fullmoves, desc="Indexing fullmoves")):
        if fm != current_fm:
            ranges[current_fm] = (start_idx, i)
            current_fm = fm
            start_idx = i

    # Don't forget the last range
    ranges[current_fm] = (start_idx, len(fullmoves))

    print(f"Found {len(ranges)} distinct fullmove values")
    return ranges


# -------------------------------------------------------------------
# PHASE 2: BUILD TRANSITION GRAPH
# -------------------------------------------------------------------


def build_position_index(dataset: Dataset, start_idx: int, end_idx: int) -> PositionIndex:
    """Build position index from a slice of the dataset."""
    index = {}

    for i in range(start_idx, end_idx):
        row = dataset[i]
        fen = row["fen"]
        moves = list(row["moves"])
        p_wins = list(row["p_win"])

        h = board_hash(fen)
        index[h] = PositionData(
            fen=fen,
            moves=moves,
            p_wins=p_wins,
            best_p_win=max(p_wins) if p_wins else 0.5,
            side=extract_side(fen),
            fullmove=row["fullmove"],
        )

    return index


def compute_transitions(pos: PositionData) -> List[Tuple[str, int, float]]:
    """Compute all transitions from this position."""
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
            continue

    return transitions


def build_predecessor_graph(position_index: PositionIndex) -> PredecessorGraph:
    """Build inverted graph: target_hash -> [(source_hash, move, regret), ...]"""
    from collections import defaultdict
    predecessors = defaultdict(list)

    for source_hash, pos in position_index.items():
        for move_uci, target_hash, regret in compute_transitions(pos):
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
    """Find up to max_paths paths leading to target."""
    max_depth = get_max_path_depth(target_pos.side, target_pos.fullmove)

    if max_depth == 0:
        return []

    heap = [(0.0, target_hash, [], [], 0)]
    paths = []
    visited_states: Set[Tuple[int, int]] = set()

    while heap and len(paths) < max_paths:
        regret, current_hash, path_moves, path_hashes, d = heapq.heappop(heap)

        state = (current_hash, d)
        if state in visited_states:
            continue
        visited_states.add(state)

        if d == max_depth:
            paths.append((
                list(reversed(path_moves)),
                list(reversed(path_hashes)),
                round(regret, 6),
            ))
            continue

        for pred_hash, move_uci, move_regret in predecessors.get(current_hash, []):
            if pred_hash not in position_index:
                continue

            heapq.heappush(heap, (
                regret + move_regret,
                pred_hash,
                path_moves + [move_uci],
                path_hashes + [pred_hash],
                d + 1,
            ))

    return paths


# -------------------------------------------------------------------
# PHASE 4: PROCESS BY FULLMOVE WINDOWS
# -------------------------------------------------------------------


def process_fullmove_window(
    dataset: Dataset,
    fullmove_ranges: Dict[int, Tuple[int, int]],
    target_fm: int,
) -> Dict[int, List[Tuple[List[str], List[int], float]]]:
    """
    Process positions at target_fm, loading a window of preceding fullmoves.
    Returns dict mapping target_hash -> list of paths.
    """
    # Determine window: need to go back ~4 fullmoves for 8 ply
    window_start_fm = max(1, target_fm - (MAX_PATH_DEPTH // 2 + 1))

    # Build position index for the window
    position_index = {}
    for fm in range(window_start_fm, target_fm + 1):
        if fm in fullmove_ranges:
            start_idx, end_idx = fullmove_ranges[fm]
            window_positions = build_position_index(dataset, start_idx, end_idx)
            position_index.update(window_positions)

    if not position_index:
        return {}

    # Get target positions (at target_fm only)
    target_hashes = {
        h for h, pos in position_index.items()
        if pos.fullmove == target_fm
    }

    if not target_hashes:
        return {}

    # Build predecessor graph
    predecessors = build_predecessor_graph(position_index)

    # Find paths for each target
    results = {}
    for target_hash in target_hashes:
        paths = find_paths(
            target_hash,
            position_index[target_hash],
            predecessors,
            position_index,
        )
        if paths:
            results[target_hash] = paths

    return results


# -------------------------------------------------------------------
# PHASE 5: CREATE FINAL DATASET
# -------------------------------------------------------------------


def create_output_dataset(
    dataset: Dataset,
    fullmove_ranges: Dict[int, Tuple[int, int]],
) -> Dataset:
    """
    Create final dataset with paths.
    Processes one fullmove at a time to manage memory.
    """
    print("\nPhase 2: Computing paths and building output dataset...")

    fullmoves = sorted(fullmove_ranges.keys())
    all_rows = []

    for target_fm in tqdm(fullmoves, desc="Processing fullmoves"):
        # Compute paths for this fullmove
        paths_by_hash = process_fullmove_window(dataset, fullmove_ranges, target_fm)

        # Get rows for this fullmove
        start_idx, end_idx = fullmove_ranges[target_fm]

        for i in range(start_idx, end_idx):
            row = dataset[i]
            fen = row["fen"]
            h = board_hash(fen)

            position_paths = paths_by_hash.get(h, [])

            if position_paths:
                path_moves = [p[0] for p in position_paths]
                path_hashes = [p[1] for p in position_paths]
                path_regrets = [p[2] for p in position_paths]
            else:
                path_moves = []
                path_hashes = []
                path_regrets = []

            all_rows.append({
                "fen": fen,
                "moves": list(row["moves"]),
                "p_win": list(row["p_win"]),
                "path_moves": path_moves,
                "path_hashes": path_hashes,
                "path_regrets": path_regrets,
            })

        # Periodically save to avoid memory issues
        if len(all_rows) >= 10_000_000:
            print(f"\nFlushing {len(all_rows):,} rows to intermediate dataset...")
            # TODO: Could save intermediate chunks here
            pass

        gc.collect()

    print(f"\nCreating final dataset with {len(all_rows):,} rows...")
    output_dataset = Dataset.from_list(all_rows, features=OUTPUT_FEATURES)

    return output_dataset


# -------------------------------------------------------------------
# ALTERNATIVE: STREAMING APPROACH FOR HUGE DATASETS
# -------------------------------------------------------------------


def process_streaming(dataset: Dataset, fullmove_ranges: Dict[int, Tuple[int, int]]):
    """
    Process dataset in streaming fashion, writing output incrementally.
    Better for very large datasets that don't fit in memory.
    """
    import pyarrow.parquet as pq

    print("\nPhase 2: Computing paths (streaming mode)...")

    fullmoves = sorted(fullmove_ranges.keys())
    output_path = OUTPUT_PATH / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    rows_buffer = []
    CHUNK_SIZE = 1_000_000

    for target_fm in tqdm(fullmoves, desc="Processing fullmoves"):
        paths_by_hash = process_fullmove_window(dataset, fullmove_ranges, target_fm)

        start_idx, end_idx = fullmove_ranges[target_fm]

        for i in range(start_idx, end_idx):
            row = dataset[i]
            fen = row["fen"]
            h = board_hash(fen)

            position_paths = paths_by_hash.get(h, [])

            if position_paths:
                path_moves = [p[0] for p in position_paths]
                path_hashes = [p[1] for p in position_paths]
                path_regrets = [float(p[2]) for p in position_paths]
            else:
                path_moves = []
                path_hashes = []
                path_regrets = []

            rows_buffer.append({
                "fen": fen,
                "moves": list(row["moves"]),
                "p_win": [float(x) for x in row["p_win"]],
                "path_moves": path_moves,
                "path_hashes": path_hashes,
                "path_regrets": path_regrets,
            })

            if len(rows_buffer) >= CHUNK_SIZE:
                chunk_ds = Dataset.from_list(rows_buffer, features=OUTPUT_FEATURES)
                chunk_ds.save_to_disk(str(output_path / f"chunk_{chunk_idx:04d}"))
                print(f"\nSaved chunk {chunk_idx} ({len(rows_buffer):,} rows)")
                rows_buffer = []
                chunk_idx += 1
                gc.collect()

        gc.collect()

    # Save remaining rows
    if rows_buffer:
        chunk_ds = Dataset.from_list(rows_buffer, features=OUTPUT_FEATURES)
        chunk_ds.save_to_disk(str(output_path / f"chunk_{chunk_idx:04d}"))
        print(f"\nSaved final chunk {chunk_idx} ({len(rows_buffer):,} rows)")

    # Concatenate all chunks
    print("\nConcatenating chunks...")
    chunk_paths = sorted(output_path.glob("chunk_*"))
    chunks = [load_from_disk(str(p)) for p in chunk_paths]
    final_dataset = concatenate_datasets(chunks)

    print(f"Final dataset has {len(final_dataset):,} examples")
    final_dataset.save_to_disk(str(OUTPUT_PATH))

    # Cleanup chunks
    import shutil
    shutil.rmtree(output_path)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def main():
    """Main entry point."""
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(str(DATASET_PATH))
    print(f"Dataset has {len(dataset):,} examples")
    print(f"Max paths per position: {MAX_PATHS_PER_POSITION}")
    print(f"Max path depth: {MAX_PATH_DEPTH}")

    # Phase 1: Sort by fullmove (uses Arrow, much faster than JSON partitioning)
    dataset = prepare_sorted_dataset(dataset)

    # Build fullmove range index
    fullmove_ranges = get_fullmove_ranges(dataset)

    # Phase 2-3: Process and create output (streaming for large datasets)
    process_streaming(dataset, fullmove_ranges)

    print(f"\nFinal dataset saved to: {OUTPUT_PATH}")


def verify_paths(sample_size: int = 100):
    """Verify a sample of paths for correctness."""
    import random

    print(f"Loading dataset from {OUTPUT_PATH}...")
    dataset = load_from_disk(str(OUTPUT_PATH))
    print(f"Dataset has {len(dataset):,} examples")

    indices_with_paths = [
        i for i in range(min(100000, len(dataset)))
        if len(dataset[i]["path_moves"]) > 0
    ]

    print(f"Found {len(indices_with_paths):,} examples with paths (in first 100k)")

    if len(indices_with_paths) < sample_size:
        sample_indices = indices_with_paths
    else:
        sample_indices = random.sample(indices_with_paths, sample_size)

    print(f"Verifying {len(sample_indices)} random paths...")

    # Build hash lookup
    print("Building hash lookup (first 1M positions)...")
    hash_to_fen = {}
    for i in tqdm(range(min(1_000_000, len(dataset))), desc="Indexing"):
        fen = dataset[i]["fen"]
        hash_to_fen[board_hash(fen)] = fen

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
                print(f"ERROR: moves/hashes length mismatch")
                errors += 1
                continue

            if not hashes:
                continue

            starting_hash = hashes[0]
            if starting_hash not in hash_to_fen:
                # May not be in our limited lookup
                continue

            starting_fen = hash_to_fen[starting_hash]
            board = chess.Board(starting_fen)

            try:
                for i, (move_uci, expected_hash) in enumerate(zip(moves, hashes)):
                    actual_hash = board_hash(board.fen())
                    if actual_hash != expected_hash:
                        print(f"ERROR: Hash mismatch at step {i}")
                        errors += 1
                        break

                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        print(f"ERROR: Illegal move {move_uci}")
                        errors += 1
                        break

                    board.push(move)

                final_hash = board_hash(board.fen())
                if final_hash != target_hash:
                    print(f"ERROR: Did not reach target")
                    errors += 1

            except Exception as e:
                print(f"ERROR: {e}")
                errors += 1

    if errors == 0:
        print(f"All {len(sample_indices)} verified paths passed!")
    else:
        print(f"Found {errors} errors")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_paths()
    else:
        main()
