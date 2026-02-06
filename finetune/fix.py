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

Uses ds.map(batched=True) to avoid loading 500M rows into memory at once.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chess
from datasets import load_from_disk


DEFAULT_DATASET_PATH = Path("/fs/scratch/PAS2836/lees_stuff/searchless_mates_hf")
SCRATCH_CACHE = "/fs/scratch/PAS2836/lees_stuff/hf_cache"


def fix_batch(batch):
    """Fix mate depths for a batch of rows."""
    all_fixed = []
    for fen, moves, mates in zip(batch["fen"], batch["moves"], batch["mate"]):
        fixed = list(mates)
        has_positive = False
        for m in fixed:
            if m > 0:
                has_positive = True
                break

        if has_positive:
            board = chess.Board(fen)
            for i, mate_val in enumerate(fixed):
                if mate_val > 1:
                    fixed[i] = mate_val + 1
                elif mate_val == 1:
                    move = chess.Move.from_uci(moves[i])
                    board.push(move)
                    if not board.is_checkmate():
                        fixed[i] = 2
                    board.pop()

        all_fixed.append(fixed)

    return {"mate": all_fixed}


def main():
    parser = argparse.ArgumentParser(description="Fix off-by-one in winning mate depths")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: <dataset>_fixed)")
    parser.add_argument("--num-proc", type=int, default=1,
                        help="Number of parallel processes for map()")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.dataset.parent / (args.dataset.name + "_fixed")

    if args.output.exists():
        print(f"Output already exists at {args.output}")
        return 1

    # Redirect HF cache to scratch so intermediate Arrow files
    # don't fill up the home directory quota
    os.makedirs(SCRATCH_CACHE, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = SCRATCH_CACHE
    os.environ["HF_HOME"] = SCRATCH_CACHE

    print(f"Loading dataset from {args.dataset}...")
    ds = load_from_disk(str(args.dataset))
    print(f"Loaded {len(ds):,} rows")

    print(f"Fixing mate depths with {args.num_proc} processes...")
    fixed_ds = ds.map(
        fix_batch,
        batched=True,
        batch_size=10000,
        num_proc=args.num_proc if args.num_proc > 1 else None,
        desc="Fixing",
    )

    print(f"Saving to {args.output}...")
    fixed_ds.save_to_disk(str(args.output), num_proc=args.num_proc)
    print(f"Saved {len(fixed_ds):,} rows to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
