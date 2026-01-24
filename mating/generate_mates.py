#!/usr/bin/env python
"""
Generate a smaller dataset (2M positions) with refined mate-in-n win percentages.

The original dataset assigns 100% win rate to any mating move and 0% to getting mated.
This script re-runs Stockfish to find the actual mate-in-n depth and assigns:
- Mating moves: 98% + 2%/n (so mate-in-1 = 100%, mate-in-2 = 99%, etc.)
- Getting mated: 2% - 2%/n (so mate-in-1 = 0%, mate-in-2 = 1%, etc.)

This helps the model learn to distinguish between immediate mates and longer forced mates.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import chess
import chess.engine
from datasets import Dataset, load_from_disk
from tqdm import tqdm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DEFAULT_OUTPUT_PATH = Path("/fs/scratch/PAS2836/lees_stuff/action_value_mates")
DEFAULT_STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"

DEFAULT_NUM_POSITIONS = 2_000_000
DEFAULT_SEED = 42
DEFAULT_NUM_ENGINES = 40
DEFAULT_BATCH_SIZE = 10_000

MATE_SEARCH_TIME = 0.05
MATE_SEARCH_DEPTH = 20

WIN_THRESHOLD = 0.9999
LOSS_THRESHOLD = 0.0001


def compute_mate_winrate(mate_in_n: int, is_winning: bool) -> float:
    """Compute refined win rate based on mate depth."""
    if mate_in_n <= 0:
        mate_in_n = 1
    if is_winning:
        return 0.98 + 0.02 / mate_in_n
    else:
        return 0.02 - 0.02 / mate_in_n


def remap_non_mate_winrate(p_win: float) -> float:
    """Remap non-mate win percentages from (0, 1) to (0.02, 0.98)."""
    return 0.02 + p_win * 0.96


def analyze_move(
    engine: chess.engine.SimpleEngine,
    fen: str,
    move_uci: str,
) -> Optional[Tuple[bool, int]]:
    """Analyze a move to detect if it leads to forced mate."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None

        board.push(move)
        info = engine.analyse(
            board,
            chess.engine.Limit(time=MATE_SEARCH_TIME, depth=MATE_SEARCH_DEPTH)
        )

        score = info.get("score")
        if score is None:
            return None

        pov_score = score.relative
        if pov_score.is_mate():
            mate_in = pov_score.mate()
            if mate_in is not None:
                if mate_in < 0:
                    return (True, abs(mate_in))
                else:
                    return (False, mate_in)
        return None
    except Exception:
        return None


class EnginePool:
    """Pool of Stockfish engines for parallel analysis."""

    def __init__(self, stockfish_path: str, num_engines: int):
        self.engines: list[chess.engine.SimpleEngine] = []
        print(f"Starting {num_engines} Stockfish engines...")
        for _ in tqdm(range(num_engines), desc="Initializing engines"):
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            engine.configure({"Threads": 1})
            self.engines.append(engine)
        print(f"All {num_engines} engines ready.\n")

    def close(self):
        for engine in self.engines:
            engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def process_batch(
    pool: EnginePool,
    batch: dict,
    executor: ThreadPoolExecutor,
) -> list[dict]:
    """Process a batch of positions, analyzing mate moves in parallel."""
    results = []
    batch_size = len(batch["fen"])

    # Collect all mate-candidate moves that need analysis
    work_items = []  # (pos_idx, move_idx, fen, move, is_winning_move)
    for pos_idx in range(batch_size):
        for move_idx, (move, p_win) in enumerate(
            zip(batch["moves"][pos_idx], batch["p_win"][pos_idx])
        ):
            if p_win >= WIN_THRESHOLD or p_win <= LOSS_THRESHOLD:
                work_items.append((
                    pos_idx,
                    move_idx,
                    batch["fen"][pos_idx],
                    move,
                    p_win >= WIN_THRESHOLD,
                ))

    # Analyze all mate candidates in parallel using thread pool
    analysis_results = {}  # (pos_idx, move_idx) -> (is_winning, mate_in_n) or None

    if work_items:
        # Round-robin assign work to engines
        futures = {}
        for i, (pos_idx, move_idx, fen, move, is_winning) in enumerate(work_items):
            engine = pool.engines[i % len(pool.engines)]
            future = executor.submit(analyze_move, engine, fen, move)
            futures[future] = (pos_idx, move_idx, is_winning)

        for future in as_completed(futures):
            pos_idx, move_idx, is_winning_move = futures[future]
            result = future.result()
            analysis_results[(pos_idx, move_idx)] = (result, is_winning_move)

    # Build output for each position
    for pos_idx in range(batch_size):
        new_p_wins = []
        for move_idx, p_win in enumerate(batch["p_win"][pos_idx]):
            key = (pos_idx, move_idx)
            if key in analysis_results:
                result, is_winning_move = analysis_results[key]
                if result is not None:
                    is_winning, mate_in_n = result
                    new_p_wins.append(compute_mate_winrate(mate_in_n, is_winning))
                else:
                    new_p_wins.append(0.98 if is_winning_move else 0.02)
            else:
                new_p_wins.append(remap_non_mate_winrate(p_win))

        results.append({
            "fen": batch["fen"][pos_idx],
            "moves": list(batch["moves"][pos_idx]),
            "p_win": new_p_wins,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset with refined mate-in-n win percentages"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Output dataset path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--stockfish", type=str, default=DEFAULT_STOCKFISH_PATH,
        help=f"Stockfish executable path (default: {DEFAULT_STOCKFISH_PATH})",
    )
    parser.add_argument(
        "--num-positions", type=int, default=DEFAULT_NUM_POSITIONS,
        help=f"Number of positions to sample (default: {DEFAULT_NUM_POSITIONS:,})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--num-engines", type=int, default=DEFAULT_NUM_ENGINES,
        help=f"Number of Stockfish engines (default: {DEFAULT_NUM_ENGINES})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE:,})",
    )

    args = parser.parse_args()

    input_path = os.environ.get("DATASET_PATH")
    if not input_path:
        print("ERROR: DATASET_PATH environment variable not set")
        return 1
    input_path = Path(input_path)

    print("=" * 60)
    print("MATE-REFINED DATASET GENERATION")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    print(f"Stockfish: {args.stockfish}")
    print(f"Positions: {args.num_positions:,}")
    print(f"Engines: {args.num_engines}")
    print(f"Batch size: {args.batch_size:,}")
    print("=" * 60 + "\n")

    if not Path(args.stockfish).exists():
        print(f"ERROR: Stockfish not found at {args.stockfish}")
        return 1

    # Load dataset
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(str(input_path))
    print(f"Loaded {len(dataset):,} positions")

    if len(dataset) > args.num_positions:
        print(f"Sampling {args.num_positions:,} positions (seed={args.seed})...")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_positions))
    print()

    # Process in batches
    all_results = []
    with EnginePool(args.stockfish, args.num_engines) as pool:
        with ThreadPoolExecutor(max_workers=args.num_engines) as executor:
            for start in tqdm(
                range(0, len(dataset), args.batch_size),
                desc="Processing batches",
            ):
                end = min(start + args.batch_size, len(dataset))
                batch = dataset[start:end]
                batch_results = process_batch(pool, batch, executor)
                all_results.extend(batch_results)

    # Save output
    if args.output.exists():
        print(f"Removing existing output at {args.output}")
        shutil.rmtree(args.output)

    print(f"Saving {len(all_results):,} positions to {args.output}...")
    output_dataset = Dataset.from_list(all_results)
    args.output.mkdir(parents=True, exist_ok=True)
    output_dataset.save_to_disk(str(args.output))

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
