#!/usr/bin/env python
"""
Fix off-by-one bug in mate depths for winning mates.

The original generate_mates.py analyzed positions by pushing the move first,
then running Stockfish. For winning mates, the Stockfish result didn't account
for the already-pushed move, so all positive mate values from Stockfish are
off by one (e.g., mate-in-2 was stored as 1).

Fix logic for each mate value > 0:
- If mate > 1: increment by 1 (definitely from Stockfish, always off by one)
- If mate == 1: check if the move is immediate checkmate; if not, increment by 1

Negative mate values (losing) are unaffected.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chess
from datasets import Dataset, load_from_disk
from tqdm import tqdm


DEFAULT_DATASET_PATH = Path("/fs/scratch/PAS2836/lees_stuff/searchless_mates_hf")


def fix_row(fen: str, moves: list[str], mates: list[int]) -> list[int]:
    """Fix mate depths for a single row."""
    fixed = list(mates)
    board = chess.Board(fen)

    for i, mate_val in enumerate(fixed):
        if mate_val > 1:
            fixed[i] = mate_val + 1
        elif mate_val == 1:
            move = chess.Move.from_uci(moves[i])
            board.push(move)
            is_checkmate = board.is_checkmate()
            board.pop()
            if not is_checkmate:
                fixed[i] = 2

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Fix off-by-one in winning mate depths")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: <dataset>_fixed)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.dataset.parent / (args.dataset.name + "_fixed")

    if args.output.exists():
        print(f"Output already exists at {args.output}")
        return 1

    print(f"Loading dataset from {args.dataset}...")
    ds = load_from_disk(str(args.dataset))
    print(f"Loaded {len(ds):,} rows")

    fens = ds["fen"]
    moves_col = ds["moves"]
    mates_col = ds["mate"]

    # Count affected rows before fixing
    affected = sum(1 for mates in mates_col if any(m > 0 for m in mates))
    print(f"Rows with positive mate values to check: {affected:,}")

    print("Fixing mate depths...")
    fixed_mates = [
        fix_row(fen, moves, mates)
        for fen, moves, mates in tqdm(
            zip(fens, moves_col, mates_col), total=len(ds), desc="Fixing"
        )
    ]

    # Count actual changes
    changed = sum(
        1 for old, new in zip(mates_col, fixed_mates) if old != new
    )
    print(f"Rows changed: {changed:,}")

    print(f"Saving to {args.output}...")
    fixed_ds = Dataset.from_dict({
        "fen": fens,
        "moves": moves_col,
        "p_win": ds["p_win"],
        "mate": fixed_mates,
    })
    fixed_ds.save_to_disk(str(args.output))
    print(f"Saved {len(fixed_ds):,} rows to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
