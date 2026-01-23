"""
Analyze the distribution of extreme winrate moves (0% or 100%) in the dataset.

Randomly samples positions and estimates the average number of legal moves
per position that have either 0% or 100% predicted winrate.
"""

import os
import argparse
import random

from datasets import load_from_disk
from tqdm import tqdm


def count_extreme_winrate_moves(p_win: list[float], epsilon: float = 1e-6) -> tuple[int, int, int]:
    """
    Count moves with extreme winrates (0% or 100%).

    Args:
        p_win: List of win probabilities for each legal move
        epsilon: Tolerance for floating point comparison

    Returns:
        (num_zero_winrate, num_hundred_winrate, total_legal_moves)
    """
    num_zero = sum(1 for p in p_win if p <= epsilon)
    num_hundred = sum(1 for p in p_win if p >= 1.0 - epsilon)
    return num_zero, num_hundred, len(p_win)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze extreme winrate moves (0% or 100%) in the dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.getenv("DATASET_PATH"),
        help="Path to the HuggingFace dataset (default: $DATASET_PATH env var)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of positions to sample (default: 100,000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    if not args.dataset_path:
        raise ValueError(
            "Dataset path required. Set DATASET_PATH env var or use --dataset-path"
        )

    random.seed(args.seed)

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    total_size = len(dataset)
    print(f"Dataset contains {total_size:,} positions")

    sample_size = min(args.sample_size, total_size)
    print(f"Randomly sampling {sample_size:,} positions (seed={args.seed})...")

    # Random sampling
    indices = random.sample(range(total_size), sample_size)

    total_zero_moves = 0
    total_hundred_moves = 0
    total_legal_moves = 0

    positions_with_zero = 0
    positions_with_hundred = 0
    positions_with_either = 0

    for idx in tqdm(indices, desc="Analyzing positions"):
        example = dataset[idx]
        p_win = example["p_win"]

        num_zero, num_hundred, num_legal = count_extreme_winrate_moves(p_win)

        total_zero_moves += num_zero
        total_hundred_moves += num_hundred
        total_legal_moves += num_legal

        if num_zero > 0:
            positions_with_zero += 1
        if num_hundred > 0:
            positions_with_hundred += 1
        if num_zero > 0 or num_hundred > 0:
            positions_with_either += 1

    # Calculate statistics
    avg_legal_moves = total_legal_moves / sample_size
    avg_zero_moves = total_zero_moves / sample_size
    avg_hundred_moves = total_hundred_moves / sample_size
    avg_extreme_moves = (total_zero_moves + total_hundred_moves) / sample_size

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print(f"\nSample size: {sample_size:,} positions")
    print(f"Average legal moves per position: {avg_legal_moves:.2f}")

    print(f"\n--- Extreme Winrate Moves ---")
    print(f"Average moves with 0% winrate per position:   {avg_zero_moves:.4f}")
    print(f"Average moves with 100% winrate per position: {avg_hundred_moves:.4f}")
    print(f"Average moves with 0% OR 100% winrate:        {avg_extreme_moves:.4f}")

    print(f"\n--- Position Counts ---")
    print(f"Positions with at least one 0% winrate move:   {positions_with_zero:,} ({100*positions_with_zero/sample_size:.2f}%)")
    print(f"Positions with at least one 100% winrate move: {positions_with_hundred:,} ({100*positions_with_hundred/sample_size:.2f}%)")
    print(f"Positions with at least one extreme move:      {positions_with_either:,} ({100*positions_with_either/sample_size:.2f}%)")

    print(f"\n--- Total Counts ---")
    print(f"Total legal moves analyzed:     {total_legal_moves:,}")
    print(f"Total 0% winrate moves:         {total_zero_moves:,} ({100*total_zero_moves/total_legal_moves:.4f}%)")
    print(f"Total 100% winrate moves:       {total_hundred_moves:,} ({100*total_hundred_moves/total_legal_moves:.4f}%)")
    print(f"Total extreme winrate moves:    {total_zero_moves + total_hundred_moves:,} ({100*(total_zero_moves + total_hundred_moves)/total_legal_moves:.4f}%)")


if __name__ == "__main__":
    main()
