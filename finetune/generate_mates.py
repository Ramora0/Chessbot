#!/usr/bin/env python
"""
Generate dataset with mate-in-n information for extreme win probability moves.

Keeps original win percentages unchanged and adds mate depth information:
- mate[i] > 0: Move leads to mate in N moves (winning)
- mate[i] < 0: Move leads to getting mated in N moves (losing)
- mate[i] = 0: Not a forced mate (or not analyzed)

Only moves with p_win >= 0.9999 or p_win <= 0.0001 are analyzed for mate depth.

The model classifies into 5 classes: no_mate, mate_in_1, mate_in_2, mate_in_3, mate_in_4_plus.
Depth 6 search is sufficient since mate_in_4+ is our coarsest bucket.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import chess
import chess.engine
from datasets import Dataset, load_from_disk
from tqdm import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DEFAULT_OUTPUT_PATH = Path("/fs/scratch/PAS2836/lees_stuff/searchless_mates")
DEFAULT_STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"

DEFAULT_NUM_POSITIONS = 2_000_000
DEFAULT_NUM_ENGINES = 40

# Time limit per position in seconds
MATE_SEARCH_TIME = 0.005

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
            chess.engine.Limit(time=MATE_SEARCH_TIME)
        )

        score = info.get("score")
        if score is None:
            return (key, 0)

        # Score is from perspective of side to move (after our move)
        pov_score = score.relative
        if pov_score.is_mate():
            mate_in = pov_score.mate()
            if mate_in is not None:
                # mate_in < 0 means opponent getting mated = we win
                # mate_in > 0 means opponent will mate us = we lose
                if mate_in < 0:
                    return (key, abs(mate_in))
                else:
                    return (key, -mate_in)
        return (key, 0)
    except Exception:
        return (key, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset with mate-in-n information for extreme win% moves"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stockfish", type=str, default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--num-positions", type=int, default=DEFAULT_NUM_POSITIONS)
    parser.add_argument("--num-engines", type=int, default=DEFAULT_NUM_ENGINES)

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
    print(f"Positions: {args.num_positions:,}")
    print(f"Engines: {args.num_engines}")
    print("=" * 60 + "\n")

    if not Path(args.stockfish).exists():
        print(f"ERROR: Stockfish not found at {args.stockfish}")
        return 1

    # =========================================================
    # Load dataset (memory-mapped, not loaded into RAM)
    # =========================================================
    t0 = time.time()
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(str(input_path))
    print(f"Dataset has {len(dataset):,} positions ({time.time() - t0:.1f}s)")

    if len(dataset) > args.num_positions:
        print(f"Selecting first {args.num_positions:,} positions (dataset pre-shuffled)...")
        dataset = dataset.select(range(args.num_positions))

    num_positions = len(dataset)
    print(f"Processing {num_positions:,} positions\n")

    # =========================================================
    # Phase 1: Scan for mate candidates AND build work items
    # =========================================================
    print("Phase 1: Scanning for mate candidate moves...")

    # Build work items directly while scanning (avoids second pass over dataset)
    work_items: list[tuple[tuple[int, int], str, str, bool]] = []

    # Batched iteration amortizes Arrow deserialization overhead
    batch_size = 10000
    global_idx = 0
    time_iter = 0.0
    time_inner = 0.0
    iter_start = time.time()
    pbar = tqdm(dataset.iter(batch_size=batch_size),
                total=(num_positions + batch_size - 1) // batch_size,
                desc="Scanning positions", leave=True)
    for batch in pbar:
        time_iter += time.time() - iter_start
        inner_start = time.time()
        for fen, moves, p_wins in zip(batch["fen"], batch["moves"], batch["p_win"]):
            for move_idx, p_win in enumerate(p_wins):
                if p_win >= WIN_THRESHOLD:
                    work_items.append(((global_idx, move_idx), fen, moves[move_idx], True))
                elif p_win <= LOSS_THRESHOLD:
                    work_items.append(((global_idx, move_idx), fen, moves[move_idx], False))
            global_idx += 1
        time_inner += time.time() - inner_start
        pbar.set_postfix({"iter": f"{time_iter:.1f}s", "proc": f"{time_inner:.1f}s"})
        iter_start = time.time()

    print(f"  Total moves to analyze: {len(work_items):,}")
    print(f"  Avg per position: {len(work_items) / num_positions:.2f}\n")

    # =========================================================
    # Phase 2: Stockfish analysis for mate depths
    # =========================================================
    # Results stored sparsely: analysis_results[(pos_idx, move_idx)] = mate_depth
    analysis_results: dict[tuple[int, int], int] = {}

    if work_items:
        print("Phase 2: Analyzing mate depths with Stockfish...")
        total_work = len(work_items)
        print(f"  Analyzing {total_work:,} moves with {args.num_engines} processes...")

        # Use multiprocessing Pool - each process has its own Stockfish engine (no GIL contention)
        confirmed_mates = 0
        with mp.Pool(args.num_engines, initializer=_init_worker, initargs=(args.stockfish,)) as pool:
            for key, mate_depth in tqdm(
                pool.imap_unordered(_analyze_move, work_items, chunksize=100),
                total=total_work,
                desc="Analyzing moves",
                leave=True
            ):
                analysis_results[key] = mate_depth
                if mate_depth != 0:
                    confirmed_mates += 1

        print(f"  Confirmed mates: {confirmed_mates:,} / {len(work_items):,}")
        print(f"  Confirmation rate: {confirmed_mates / len(work_items) * 100:.1f}%\n")

    # =========================================================
    # Phase 3: Save output
    # =========================================================
    print("Phase 3: Building output (streaming from dataset)...")

    # Build final dataset - read from dataset, lookup mate depths from sparse results
    output_data = []
    time_iter = 0.0
    time_build = 0.0
    iter_start = time.time()
    pbar = tqdm(dataset.iter(batch_size=batch_size),
                total=(num_positions + batch_size - 1) // batch_size,
                desc="Building dataset", leave=True)
    global_idx = 0
    for batch in pbar:
        time_iter += time.time() - iter_start
        build_start = time.time()
        for fen, moves, p_wins in zip(batch["fen"], batch["moves"], batch["p_win"]):
            num_moves = len(moves)
            mate = [analysis_results.get((global_idx, m), 0) for m in range(num_moves)]
            output_data.append({
                "fen": fen,
                "moves": list(moves),
                "p_win": list(p_wins),
                "mate": mate,
            })
            global_idx += 1
        time_build += time.time() - build_start
        pbar.set_postfix({"iter": f"{time_iter:.1f}s", "build": f"{time_build:.1f}s"})
        iter_start = time.time()

    if args.output.exists():
        print(f"Removing existing output at {args.output}")
        shutil.rmtree(args.output)

    t0 = time.time()
    print(f"Creating Dataset from list...")
    output_dataset = Dataset.from_list(output_data)
    print(f"  from_list: {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"Saving {len(output_data):,} positions to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    output_dataset.save_to_disk(str(args.output))
    print(f"  save_to_disk: {time.time() - t0:.1f}s")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
