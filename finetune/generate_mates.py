#!/usr/bin/env python
"""
Generate dataset with mate-in-n information for extreme win probability moves.

Processes the dataset shard-by-shard, maintaining 1-1 correspondence with input shards.
Each shard is saved independently, allowing resume after interruption.

Keeps original win percentages unchanged and adds mate depth information:
- mate[i] > 0: Move leads to mate in N moves (winning)
- mate[i] < 0: Move leads to getting mated in N moves (losing)
- mate[i] = 0: Not a forced mate (or not analyzed)

Only moves with p_win >= 0.9999 or p_win <= 0.0001 are analyzed for mate depth.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from glob import glob
from pathlib import Path

import numpy as np
import chess
import chess.engine
from datasets import Dataset, load_from_disk
from tqdm import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DEFAULT_OUTPUT_PATH = Path("/fs/scratch/PAS2836/lees_stuff/searchless_mates")
DEFAULT_STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"
DEFAULT_NUM_ENGINES = 22

# Depth limit for mate search
MATE_SEARCH_DEPTH = 16

WIN_THRESHOLD = 0.9999
LOSS_THRESHOLD = 0.0001


# Global engine for worker processes (initialized once per process)
_worker_engine = None


def _init_worker(stockfish_path: str):
    """Initialize Stockfish engine in worker process."""
    global _worker_engine
    _worker_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _worker_engine.configure({"Threads": 1})


def _analyze_move(args: tuple) -> tuple:
    """
    Analyze a move to detect mate depth. Runs in worker process.

    Returns: (key, mate_depth)
    """
    key, fen, move_uci, is_winning_move = args
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return (key, 0)

        board.push(move)

        # Immediate checkmate
        if board.is_checkmate():
            return (key, 1 if is_winning_move else -1)

        info = _worker_engine.analyse(
            board,
            chess.engine.Limit(depth=MATE_SEARCH_DEPTH)
        )

        score = info.get("score")
        if score is None:
            return (key, 0)

        # Score is from perspective of side to move (after our move)
        pov_score = score.relative
        if pov_score.is_mate():
            mate_in = pov_score.mate()
            if mate_in is not None:
                if mate_in < 0:
                    return (key, abs(mate_in) + 1)
                else:
                    return (key, -mate_in)
        return (key, 0)
    except Exception:
        return (key, 0)


def get_shard_files(dataset_path: Path) -> list[Path]:
    """Get list of Arrow shard files from a HuggingFace dataset directory."""
    # HF datasets store data as data-00000-of-00500.arrow files
    pattern = str(dataset_path / "data-*.arrow")
    files = sorted(glob(pattern))
    if not files:
        # Fallback: might be named differently
        pattern = str(dataset_path / "*.arrow")
        files = sorted(glob(pattern))
    return [Path(f) for f in files]


def process_shard(
    shard_path: Path,
    shard_idx: int,
    output_dir: Path,
    stockfish_path: str,
    num_engines: int,
) -> dict:
    """Process a single shard of the dataset."""
    output_path = output_dir / shard_path.with_suffix(".parquet").name
    lock_path = output_path.with_suffix(".lock")

    # Check if shard already complete
    if output_path.exists():
        return {"skipped": True}

    # Atomically claim this shard so parallel instances don't duplicate work
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return {"skipped": True}

    try:
        # Load just this shard
        shard_data = Dataset.from_file(str(shard_path))
        shard_size = len(shard_data)

        # Phase 1: Vectorized scan for mate candidates
        table = shard_data.data.combine_chunks()
        p_win_arr = table.column("p_win").chunk(0)
        offsets = p_win_arr.offsets.to_numpy()
        flat_p_win = p_win_arr.values.to_numpy()

        win_mask = flat_p_win >= WIN_THRESHOLD
        loss_mask = flat_p_win <= LOSS_THRESHOLD
        candidate_indices = np.nonzero(win_mask | loss_mask)[0]

        row_indices = np.searchsorted(offsets, candidate_indices, side='right') - 1
        move_indices = candidate_indices - offsets[row_indices]
        is_winning = win_mask[candidate_indices]

        # Convert columns to Python for work item construction & output
        fens = table.column("fen").to_pylist()
        moves_col = table.column("moves").to_pylist()
        p_wins_col = table.column("p_win").to_pylist()

        work_items = [
            ((int(r), int(m)), fens[int(r)], moves_col[int(r)][int(m)], bool(w))
            for r, m, w in zip(row_indices, move_indices, is_winning)
        ]

        # Phase 2: Stockfish analysis
        analysis_results: dict[tuple[int, int], int] = {}
        confirmed_mates = 0
        total_mate_depth = 0

        if work_items:
            with mp.Pool(num_engines, initializer=_init_worker, initargs=(stockfish_path,)) as pool:
                pbar = tqdm(
                    pool.imap_unordered(_analyze_move, work_items, chunksize=100),
                    total=len(work_items),
                    desc=f"  Shard {shard_idx}",
                    leave=False
                )
                for key, mate_depth in pbar:
                    analysis_results[key] = mate_depth
                    if mate_depth != 0:
                        confirmed_mates += 1
                        total_mate_depth += abs(mate_depth)
                        mate_pct = 100.0 * confirmed_mates / len(analysis_results)
                        avg_depth = total_mate_depth / confirmed_mates
                        pbar.set_postfix(mate_pct=f"{mate_pct:.1f}%", avg_depth=f"{avg_depth:.1f}")

        # Phase 3: Build output
        output_data = [
            {
                "fen": fens[i],
                "moves": moves_col[i],
                "p_win": p_wins_col[i],
                "mate": [analysis_results.get((i, m), 0) for m in range(len(moves_col[i]))],
            }
            for i in tqdm(range(shard_size), desc=f"  Building output", leave=False)
        ]

        # Save shard as parquet (atomic: write temp, then rename)
        tmp_path = output_path.with_suffix(".parquet.tmp")
        output_dataset = Dataset.from_list(output_data)
        output_dataset.to_parquet(str(tmp_path))
        os.rename(str(tmp_path), str(output_path))

        return {
            "skipped": False,
            "positions": shard_size,
            "mate_candidates": len(work_items),
            "confirmed_mates": confirmed_mates,
        }
    finally:
        # Release lock so other instances can retry on failure
        try:
            lock_path.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset with mate-in-n information for extreme win% moves"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stockfish", type=str, default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--num-engines", type=int, default=DEFAULT_NUM_ENGINES)
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Max shards to process (for testing)")

    args = parser.parse_args()

    input_path = os.environ.get("DATASET_PATH")
    if not input_path:
        print("ERROR: DATASET_PATH environment variable not set")
        return 1
    input_path = Path(input_path)

    print("=" * 60)
    print("MATE DEPTH DATASET GENERATION")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    print(f"Stockfish: {args.stockfish}")
    print(f"Engines: {args.num_engines}")
    print("=" * 60 + "\n")

    if not Path(args.stockfish).exists():
        print(f"ERROR: Stockfish not found at {args.stockfish}")
        return 1

    # Discover input shards
    shard_files = get_shard_files(input_path)
    num_shards = len(shard_files)

    if num_shards == 0:
        print(f"ERROR: No Arrow shard files found in {input_path}")
        return 1

    print(f"Found {num_shards} input shards\n")

    if args.max_shards:
        shard_files = shard_files[:args.max_shards]
        num_shards = len(shard_files)
        print(f"Processing first {num_shards} shards (--max-shards)\n")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "input_path": str(input_path),
        "num_shards": num_shards,
        "win_threshold": WIN_THRESHOLD,
        "loss_threshold": LOSS_THRESHOLD,
        "mate_search_depth": MATE_SEARCH_DEPTH,
    }
    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Process each shard
    total_stats = {
        "positions_processed": 0,
        "positions_skipped": 0,
        "shards_skipped": 0,
        "mate_candidates": 0,
        "confirmed_mates": 0,
    }

    overall_start = time.time()

    shard_bar = tqdm(shard_files, desc="Shards", unit="shard")
    for shard_idx, shard_path in enumerate(shard_bar):
        stats = process_shard(
            shard_path=shard_path,
            shard_idx=shard_idx,
            output_dir=args.output,
            stockfish_path=args.stockfish,
            num_engines=args.num_engines,
        )

        if stats["skipped"]:
            total_stats["shards_skipped"] += 1
        else:
            total_stats["positions_processed"] += stats["positions"]
            total_stats["mate_candidates"] += stats["mate_candidates"]
            total_stats["confirmed_mates"] += stats["confirmed_mates"]


    # Final summary
    total_time = time.time() - overall_start
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Positions processed: {total_stats['positions_processed']:,}")
    print(f"Shards skipped: {total_stats['shards_skipped']}")
    print(f"Mate candidates analyzed: {total_stats['mate_candidates']:,}")
    print(f"Confirmed mates: {total_stats['confirmed_mates']:,}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Combine parquet shards into a proper HuggingFace dataset
    hf_path = args.output.parent / (args.output.name + "_hf")
    if hf_path.exists():
        print(f"\nHuggingFace dataset already exists at {hf_path}, skipping.")
    else:
        parquet_files = sorted(glob(str(args.output / "*.parquet")))
        if not parquet_files:
            print("\nERROR: No parquet files found to combine")
            return 1

        print(f"\nCombining {len(parquet_files)} parquet shards into HuggingFace dataset...")
        hf_cache = str(args.output.parent / "hf_cache")
        os.environ["HF_DATASETS_CACHE"] = hf_cache
        combined = Dataset.from_parquet(parquet_files, cache_dir=hf_cache)
        combined.save_to_disk(str(hf_path))
        print(f"Saved HuggingFace dataset ({len(combined):,} rows) to {hf_path}")

    return 0


if __name__ == "__main__":
    exit(main())
