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
from pathlib import Path
from typing import Optional, Tuple

import chess
import chess.engine
from datasets import load_from_disk

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Default paths (can be overridden via CLI)
DEFAULT_INPUT_PATH = Path("/fs/scratch/PAS2836/lees_stuff/action_value")
DEFAULT_OUTPUT_PATH = Path("/fs/scratch/PAS2836/lees_stuff/action_value_mates")
DEFAULT_STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"

# Dataset configuration
DEFAULT_NUM_POSITIONS = 2_000_000
DEFAULT_SEED = 42

# Stockfish configuration for mate detection
MATE_SEARCH_TIME = 0.05  # 50ms per move - fast but usually finds mates
MATE_SEARCH_DEPTH = 20   # Max depth to search (mates beyond this are rare)
NUM_PROC = os.cpu_count() or 8  # Number of parallel worker processes

# Winrate thresholds to detect mate positions
WIN_THRESHOLD = 0.9999  # Consider 1.0 as winning/mating
LOSS_THRESHOLD = 0.0001  # Consider 0.0 as losing/getting mated

# Global engine instance (one per worker process, lazily initialized)
_engine: Optional[chess.engine.SimpleEngine] = None

# Environment variable used to pass stockfish path to worker processes
STOCKFISH_ENV_VAR = "CHESSBOT_STOCKFISH_PATH"


def get_engine() -> chess.engine.SimpleEngine:
    """Get or initialize the Stockfish engine for this worker process."""
    global _engine
    if _engine is None:
        stockfish_path = os.environ.get(STOCKFISH_ENV_VAR, DEFAULT_STOCKFISH_PATH)
        _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _engine.configure({"Threads": 1})
    return _engine


def compute_mate_winrate(mate_in_n: int, is_winning: bool) -> float:
    """
    Compute refined win rate based on mate depth.

    For winning (mating):  98% + 2%/n  -> mate-in-1 = 100%, mate-in-2 = 99%
    For losing (mated):    2% - 2%/n   -> mate-in-1 = 0%, mate-in-2 = 1%
    """
    if mate_in_n <= 0:
        mate_in_n = 1

    if is_winning:
        return 0.98 + 0.02 / mate_in_n
    else:
        return 0.02 - 0.02 / mate_in_n


def remap_non_mate_winrate(p_win: float) -> float:
    """
    Remap non-mate win percentages from (0, 1) to (0.02, 0.98).

    Reserves 0-2% for getting mated, 98-100% for delivering mate.
    """
    return 0.02 + p_win * 0.96


def analyze_move_for_mate(
    board: chess.Board,
    move_uci: str,
) -> Optional[Tuple[bool, int]]:
    """
    Analyze a position after a move to detect if it's a forced mate.

    Returns:
        None if no mate found
        (is_winning, mate_in_n) if mate detected
    """
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None

        board.push(move)
        engine = get_engine()
        info = engine.analyse(
            board,
            chess.engine.Limit(time=MATE_SEARCH_TIME, depth=MATE_SEARCH_DEPTH)
        )
        board.pop()

        score = info.get("score")
        if score is None:
            return None

        pov_score = score.relative
        if pov_score.is_mate():
            mate_in = pov_score.mate()
            if mate_in is not None:
                # mate_in < 0: opponent is getting mated -> we are winning
                # mate_in > 0: opponent can mate us -> we are losing
                if mate_in < 0:
                    return (True, abs(mate_in))
                else:
                    return (False, mate_in)

        return None

    except Exception:
        try:
            if len(board.move_stack) > 0:
                board.pop()
        except Exception:
            pass
        return None


def process_position(example: dict) -> dict:
    """
    Process a single position: analyze mate moves and remap win percentages.

    This function is called by Dataset.map() in parallel worker processes.
    """
    board = chess.Board(example["fen"])
    new_p_wins = []

    for move, p_win in zip(example["moves"], example["p_win"]):
        is_mate_candidate = p_win >= WIN_THRESHOLD or p_win <= LOSS_THRESHOLD

        if is_mate_candidate:
            is_winning_move = p_win >= WIN_THRESHOLD
            result = analyze_move_for_mate(board, move)

            if result is not None:
                is_winning, mate_in_n = result
                new_p_wins.append(compute_mate_winrate(mate_in_n, is_winning))
            else:
                # Couldn't confirm mate, assign default deep mate value
                new_p_wins.append(0.98 if is_winning_move else 0.02)
        else:
            # Non-mate move: remap from (0,1) to (0.02, 0.98)
            new_p_wins.append(remap_non_mate_winrate(p_win))

    return {
        "fen": example["fen"],
        "moves": list(example["moves"]),
        "p_win": new_p_wins,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset with refined mate-in-n win percentages"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to input action_value dataset (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for output dataset (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=DEFAULT_STOCKFISH_PATH,
        help=f"Path to Stockfish executable (default: {DEFAULT_STOCKFISH_PATH})",
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=DEFAULT_NUM_POSITIONS,
        help=f"Number of positions to sample (default: {DEFAULT_NUM_POSITIONS:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=NUM_PROC,
        help=f"Number of parallel worker processes (default: {NUM_PROC})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MATE-REFINED DATASET GENERATION")
    print("=" * 60)
    print(f"Input dataset: {args.input}")
    print(f"Output dataset: {args.output}")
    print(f"Stockfish path: {args.stockfish}")
    print(f"Target positions: {args.num_positions:,}")
    print(f"Random seed: {args.seed}")
    print(f"Worker processes: {args.num_proc}")
    print("=" * 60)
    print()

    # Verify stockfish exists
    if not Path(args.stockfish).exists():
        print(f"ERROR: Stockfish not found at {args.stockfish}")
        return 1

    # Set stockfish path in environment for worker processes to inherit
    os.environ[STOCKFISH_ENV_VAR] = args.stockfish

    # Load and sample dataset
    print(f"Loading dataset from {args.input}...")
    dataset = load_from_disk(str(args.input))
    print(f"Loaded {len(dataset):,} positions")

    # Sample if dataset is larger than target
    if len(dataset) > args.num_positions:
        print(f"Randomly sampling {args.num_positions:,} positions (seed={args.seed})...")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_positions))
        print(f"Sampled {len(dataset):,} positions")
    else:
        print(f"Dataset size ({len(dataset):,}) <= target, using all positions")
    print()

    # Process dataset using HuggingFace's parallel map
    print("Processing positions with mate refinement...")
    processed_dataset = dataset.map(
        process_position,
        num_proc=args.num_proc,
        desc="Refining mate positions",
    )

    # Remove existing output if it exists
    if args.output.exists():
        print(f"Removing existing output at {args.output}")
        shutil.rmtree(args.output)

    # Save dataset
    print(f"Saving to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(args.output))

    print("\nDone!")
    print(f"Output dataset saved to: {args.output}")
    print(f"Total positions: {len(processed_dataset):,}")

    return 0


if __name__ == "__main__":
    exit(main())
