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
DEFAULT_SEED = 42
DEFAULT_NUM_ENGINES = 40

# Depth 6 is sufficient to detect mate-in-4 (8 plies after our move = 6 plies remaining)
# Since we classify mate_in_4_plus as a single bucket, no need for deeper search
MATE_SEARCH_DEPTH = 6

WIN_THRESHOLD = 0.9999
LOSS_THRESHOLD = 0.0001


# Global engine for worker processes (initialized once per process)
_worker_engine = None
_worker_stockfish_path = None


def _init_worker(stockfish_path: str):
    """Initialize Stockfish engine in worker process."""
    global _worker_engine, _worker_stockfish_path
    _worker_stockfish_path = stockfish_path
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
                # mate_in < 0 means the side to move is getting mated
                # mate_in > 0 means the side to move will deliver mate
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    # Load dataset into memory
    # =========================================================
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(str(input_path))
    print(f"Loaded {len(dataset):,} positions")

    if len(dataset) > args.num_positions:
        print(f"Sampling {args.num_positions:,} positions (seed={args.seed})...")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_positions))

    # Convert to plain Python lists
    print("Loading into memory...")
    num_positions = len(dataset)
    fens = []
    moves_list = []
    p_wins_list = []

    batch_size = 50_000
    for start in tqdm(range(0, num_positions, batch_size), desc="Loading to RAM"):
        end = min(start + batch_size, num_positions)
        batch = dataset[start:end]
        fens.extend(batch["fen"])
        moves_list.extend(batch["moves"])
        p_wins_list.extend(batch["p_win"])

    print(f"Loaded {num_positions:,} positions into memory\n")

    # =========================================================
    # Phase 1: Identify positions with extreme win probability moves
    # =========================================================
    print("Phase 1: Scanning for mate candidate moves...")

    # Track positions needing Stockfish analysis
    # mate_candidates[pos_idx] = [(move_idx, move_uci, is_winning), ...]
    mate_candidates: dict[int, list[tuple[int, str, bool]]] = {}

    total_mate_moves = 0
    for i in tqdm(range(num_positions), desc="Scanning positions"):
        p_wins = p_wins_list[i]
        moves = moves_list[i]

        position_mates = []
        for move_idx, p_win in enumerate(p_wins):
            if p_win >= WIN_THRESHOLD:
                position_mates.append((move_idx, moves[move_idx], True))
            elif p_win <= LOSS_THRESHOLD:
                position_mates.append((move_idx, moves[move_idx], False))

        if position_mates:
            mate_candidates[i] = position_mates
            total_mate_moves += len(position_mates)

    print(f"  Positions with mate candidates: {len(mate_candidates):,}")
    print(f"  Total moves to analyze: {total_mate_moves:,}")
    print(f"  Mate position ratio: {len(mate_candidates) / num_positions * 100:.1f}%\n")

    # =========================================================
    # Phase 2: Stockfish analysis for mate depths
    # =========================================================
    # Initialize mate arrays (0 = not a forced mate)
    mate_depths: list[list[int]] = [[0] * len(moves_list[i]) for i in range(num_positions)]

    if mate_candidates:
        print("Phase 2: Analyzing mate depths with Stockfish...")

        # Collect all work items
        work_items: list[tuple[tuple[int, int], str, str, bool]] = []
        for pos_idx, candidates in mate_candidates.items():
            fen = fens[pos_idx]
            for move_idx, move_uci, is_winning in candidates:
                work_items.append(((pos_idx, move_idx), fen, move_uci, is_winning))

        total_work = len(work_items)
        print(f"  Analyzing {total_work:,} moves with {args.num_engines} processes...")

        # Use multiprocessing Pool - each process has its own Stockfish engine (no GIL contention)
        confirmed_mates = 0
        with mp.Pool(args.num_engines, initializer=_init_worker, initargs=(args.stockfish,)) as pool:
            for key, mate_depth in tqdm(
                pool.imap_unordered(_analyze_move, work_items, chunksize=100),
                total=total_work,
                desc="Analyzing moves"
            ):
                pos_idx, move_idx = key
                mate_depths[pos_idx][move_idx] = mate_depth
                if mate_depth != 0:
                    confirmed_mates += 1

        print(f"  Confirmed mates: {confirmed_mates:,} / {total_mate_moves:,}")
        print(f"  Confirmation rate: {confirmed_mates / total_mate_moves * 100:.1f}%\n")

    # =========================================================
    # Phase 3: Save output
    # =========================================================
    print("Phase 3: Saving output...")

    # Build final dataset - keep p_win unchanged, add mate depths
    output_data = [
        {
            "fen": fens[i],
            "moves": list(moves_list[i]),
            "p_win": list(p_wins_list[i]),  # Unchanged from source
            "mate": mate_depths[i],  # New field: mate depth for each move
        }
        for i in tqdm(range(num_positions), desc="Building dataset")
    ]

    if args.output.exists():
        print(f"Removing existing output at {args.output}")
        shutil.rmtree(args.output)

    print(f"Saving {len(output_data):,} positions to {args.output}...")
    output_dataset = Dataset.from_list(output_data)
    args.output.mkdir(parents=True, exist_ok=True)
    output_dataset.save_to_disk(str(args.output))

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
