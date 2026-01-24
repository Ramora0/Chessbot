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
import asyncio
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import numpy as np
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from tqdm.auto import tqdm

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
# Short time limit but enough to find reasonable mates
MATE_SEARCH_TIME = 0.05  # 50ms per move - fast but usually finds mates
MATE_SEARCH_DEPTH = 20   # Max depth to search (mates beyond this are rare)
MAX_CONCURRENT_ENGINES = 32  # Number of parallel Stockfish processes

# Winrate thresholds to detect mate positions
WIN_THRESHOLD = 0.9999  # Consider 1.0 as winning/mating
LOSS_THRESHOLD = 0.0001  # Consider 0.0 as losing/getting mated

# Schema for output dataset
FEATURES = Features({
    "fen": Value("string"),
    "moves": Sequence(Value("string")),
    "p_win": Sequence(Value("float32")),
})


def compute_mate_winrate(mate_in_n: int, is_winning: bool) -> float:
    """
    Compute refined win rate based on mate depth.

    For winning (mating):  98% + 2%/n  -> mate-in-1 = 100%, mate-in-2 = 99%
    For losing (mated):    2% - 2%/n   -> mate-in-1 = 0%, mate-in-2 = 1%

    Args:
        mate_in_n: Number of moves until mate (always positive)
        is_winning: True if this side is delivering mate, False if receiving

    Returns:
        Refined win percentage [0.0, 1.0]
    """
    if mate_in_n <= 0:
        mate_in_n = 1  # Shouldn't happen, but safety

    if is_winning:
        # Mating: 0.98 + 0.02/n
        # n=1 -> 1.0, n=2 -> 0.99, n=10 -> 0.982, etc.
        return 0.98 + 0.02 / mate_in_n
    else:
        # Getting mated: 0.02 - 0.02/n
        # n=1 -> 0.0, n=2 -> 0.01, n=10 -> 0.018, etc.
        return 0.02 - 0.02 / mate_in_n


async def analyze_position_for_mate(
    engine: chess.engine.UciProtocol,
    board: chess.Board,
    move_uci: str,
    time_limit: float,
    depth_limit: int = MATE_SEARCH_DEPTH,
) -> Optional[Tuple[bool, int]]:
    """
    Analyze a position after a move to detect if it's a forced mate.

    Returns:
        None if no mate found in search time/depth
        (is_winning, mate_in_n) tuple if mate detected
            is_winning: True if the side that just moved is winning
            mate_in_n: Positive number of moves until mate
    """
    try:
        # Make the move to analyze the resulting position
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None

        board.push(move)

        # Analyze the position
        info = await engine.analyse(
            board,
            chess.engine.Limit(time=time_limit, depth=depth_limit)
        )

        board.pop()  # Restore board state

        score = info.get("score")
        if score is None:
            return None

        # Get score from the perspective of the side to move (after our move)
        # This is the opponent's perspective
        pov_score = score.relative

        if pov_score.is_mate():
            mate_in = pov_score.mate()
            if mate_in is not None:
                # mate_in > 0: opponent (side to move) can mate us -> we are losing
                # mate_in < 0: opponent is getting mated -> we are winning
                if mate_in < 0:
                    return (True, abs(mate_in))  # We are winning
                else:
                    return (False, mate_in)  # We are losing

        return None

    except Exception:
        # If anything goes wrong, try to restore board and return None
        try:
            if len(board.move_stack) > 0:
                board.pop()
        except Exception:
            pass
        return None


async def process_batch_async(
    positions: List[Dict],
    engines: List[Tuple[object, chess.engine.UciProtocol]],
    stats: Dict[str, int],
    time_limit: float,
) -> List[Dict]:
    """
    Process a batch of positions, refining mate win percentages.

    Args:
        positions: List of position dicts with fen, moves, p_win
        engines: List of (transport, engine) tuples
        stats: Statistics dict to update

    Returns:
        List of processed position dicts with updated p_win values
    """
    results = []

    for pos in positions:
        fen = pos["fen"]
        moves = list(pos["moves"])
        p_wins = list(pos["p_win"])

        # Find moves that need mate analysis
        moves_to_check = []
        for i, (move, p_win) in enumerate(zip(moves, p_wins)):
            if p_win >= WIN_THRESHOLD or p_win <= LOSS_THRESHOLD:
                moves_to_check.append((i, move, p_win >= WIN_THRESHOLD))

        if not moves_to_check:
            # No mate-ish moves, keep as-is
            results.append(pos)
            continue

        stats["positions_with_mates"] += 1
        stats["total_mate_moves"] += len(moves_to_check)

        # Create tasks for all moves needing analysis
        board = chess.Board(fen)
        tasks = []
        for idx, move, is_winning_move in moves_to_check:
            # Cycle through available engines
            engine = engines[len(tasks) % len(engines)][1]
            tasks.append(analyze_position_for_mate(engine, board.copy(), move, time_limit))

        # Run all analyses in parallel
        mate_results = await asyncio.gather(*tasks)

        # Update p_win values based on results
        new_p_wins = list(p_wins)
        for (idx, move, is_winning_move), mate_result in zip(moves_to_check, mate_results):
            if mate_result is not None:
                is_winning, mate_in_n = mate_result
                new_p_wins[idx] = compute_mate_winrate(mate_in_n, is_winning)
                stats["mates_found"] += 1
                stats[f"mate_depth_{min(mate_in_n, 10)}"] += 1
            else:
                # Couldn't find mate, assign default deep mate value
                stats["mates_not_found"] += 1
                if is_winning_move:
                    new_p_wins[idx] = 0.98  # Assume very long mate
                else:
                    new_p_wins[idx] = 0.02  # Assume very long getting mated

        results.append({
            "fen": fen,
            "moves": moves,
            "p_win": np.asarray(new_p_wins, dtype=np.float32).tolist(),
        })

    return results


async def process_dataset_async(
    dataset: Dataset,
    stockfish_path: str,
    batch_size: int = 100,
    num_engines: int = MAX_CONCURRENT_ENGINES,
    time_limit: float = MATE_SEARCH_TIME,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Process the entire dataset asynchronously with multiple Stockfish engines.

    Returns:
        Tuple of (processed_positions, statistics)
    """
    # Initialize statistics
    stats = defaultdict(int)
    stats["total_positions"] = len(dataset)

    # Initialize engines
    print(f"Initializing {num_engines} Stockfish engines...")
    engines = []
    for _ in tqdm(range(num_engines), desc="Starting engines"):
        transport, engine = await chess.engine.popen_uci(stockfish_path)
        engines.append((transport, engine))
    print("All engines ready.\n")

    all_results = []

    try:
        # Process in batches
        with tqdm(total=len(dataset), desc="Processing positions") as pbar:
            for batch_start in range(0, len(dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(dataset))
                batch = [dataset[i] for i in range(batch_start, batch_end)]

                batch_results = await process_batch_async(batch, engines, stats, time_limit)
                all_results.extend(batch_results)
                pbar.update(len(batch))

                # Update progress description with stats
                if stats["total_mate_moves"] > 0:
                    found_rate = stats["mates_found"] / stats["total_mate_moves"] * 100
                    not_found_rate = stats["mates_not_found"] / stats["total_mate_moves"] * 100
                    pbar.set_postfix({
                        "found": f"{found_rate:.1f}%",
                        "not_found": f"{not_found_rate:.1f}%"
                    })

    finally:
        # Clean up engines
        print("\nShutting down engines...")
        for _, engine in engines:
            await engine.quit()

    return all_results, dict(stats)


def print_statistics(stats: Dict[str, int]) -> None:
    """Print detailed statistics about the mate analysis."""
    print("\n" + "=" * 60)
    print("MATE ANALYSIS STATISTICS")
    print("=" * 60)

    print(f"Total positions processed: {stats['total_positions']:,}")
    print(f"Positions with mate moves: {stats['positions_with_mates']:,}")
    print(f"Total mate moves analyzed: {stats['total_mate_moves']:,}")
    print()

    mates_found = stats["mates_found"]
    mates_not_found = stats["mates_not_found"]
    total_mates = mates_found + mates_not_found

    if total_mates > 0:
        print(f"Mates successfully found: {mates_found:,} ({mates_found/total_mates*100:.2f}%)")
        print(f"Mates not found (timeout): {mates_not_found:,} ({mates_not_found/total_mates*100:.2f}%)")
        print()

        # Print mate depth distribution
        print("Mate depth distribution:")
        print("-" * 40)
        for depth in range(1, 11):
            key = f"mate_depth_{depth}"
            count = stats.get(key, 0)
            if count > 0:
                label = f"Mate in {depth}" if depth < 10 else "Mate in 10+"
                pct = count / mates_found * 100
                bar = "█" * int(pct / 2)
                print(f"  {label:<12}: {count:>8,} ({pct:>5.1f}%) {bar}")

    print("=" * 60)


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
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--num-engines",
        type=int,
        default=MAX_CONCURRENT_ENGINES,
        help=f"Number of parallel Stockfish engines (default: {MAX_CONCURRENT_ENGINES})",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=MATE_SEARCH_TIME,
        help=f"Time limit per move for mate search in seconds (default: {MATE_SEARCH_TIME})",
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
    print(f"Parallel engines: {args.num_engines}")
    print(f"Search time/move: {args.time_limit}s")
    print("=" * 60)
    print()

    # Verify stockfish exists
    if not Path(args.stockfish).exists():
        print(f"ERROR: Stockfish not found at {args.stockfish}")
        return 1

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

    # Process dataset
    print("Starting mate analysis...")
    processed_positions, stats = asyncio.run(
        process_dataset_async(
            dataset,
            args.stockfish,
            batch_size=args.batch_size,
            num_engines=args.num_engines,
            time_limit=args.time_limit,
        )
    )

    # Print statistics
    print_statistics(stats)

    # Create output dataset
    print(f"\nCreating output dataset with {len(processed_positions):,} positions...")

    # Remove existing output if it exists
    if args.output.exists():
        print(f"Removing existing output at {args.output}")
        shutil.rmtree(args.output)

    # Create dataset from processed positions
    output_dataset = Dataset.from_list(processed_positions, features=FEATURES)

    # Save dataset
    print(f"Saving to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    output_dataset.save_to_disk(str(args.output))

    print("\nDone!")
    print(f"Output dataset saved to: {args.output}")
    print(f"Total positions: {len(output_dataset):,}")

    return 0


if __name__ == "__main__":
    exit(main())
