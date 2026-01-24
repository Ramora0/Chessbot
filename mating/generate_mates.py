#!/usr/bin/env python
"""
Generate a smaller dataset (2M positions) with refined mate-in-n win percentages.

Two-phase approach for speed:
1. Fast pass: Identify positions with mate candidates, remap non-mate positions (no Stockfish)
2. Slow pass: Run Stockfish only on positions with mate-candidate moves

This is much faster than running Stockfish analysis on every position.
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple

import chess
import chess.engine
from datasets import Dataset, concatenate_datasets, load_from_disk
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


# -------------------------------------------------------------------
# PHASE 1: Fast filtering and remapping (no Stockfish)
# -------------------------------------------------------------------

def has_mate_candidate(example: dict) -> bool:
    """Check if position has any mate-candidate moves."""
    for p_win in example["p_win"]:
        if p_win >= WIN_THRESHOLD or p_win <= LOSS_THRESHOLD:
            return True
    return False


def remap_non_mate_batch(batch: dict) -> dict:
    """Remap all win percentages for positions without mate candidates."""
    new_p_wins = []
    for p_win_list in batch["p_win"]:
        # Linear map: 0 -> 0.02, 1 -> 0.98
        new_p_wins.append([0.02 + p * 0.96 for p in p_win_list])
    return {
        "fen": batch["fen"],
        "moves": batch["moves"],
        "p_win": new_p_wins,
    }


# -------------------------------------------------------------------
# PHASE 2: Stockfish analysis for mate positions
# -------------------------------------------------------------------

def compute_mate_winrate(mate_in_n: int, is_winning: bool) -> float:
    """Compute refined win rate based on mate depth."""
    if mate_in_n <= 0:
        mate_in_n = 1
    if is_winning:
        return 0.98 + 0.02 / mate_in_n
    else:
        return 0.02 - 0.02 / mate_in_n


class EngineWorker(threading.Thread):
    """Worker thread that owns a single Stockfish engine."""

    def __init__(self, stockfish_path: str, work_queue: queue.Queue, results: dict, lock: threading.Lock):
        super().__init__(daemon=True)
        self.stockfish_path = stockfish_path
        self.work_queue = work_queue
        self.results = results
        self.lock = lock
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def run(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Threads": 1})

        while True:
            item = self.work_queue.get()
            if item is None:
                self.work_queue.task_done()
                break

            key, fen, move_uci = item
            result = self._analyze_move(fen, move_uci)

            with self.lock:
                self.results[key] = result

            self.work_queue.task_done()

        self.engine.quit()

    def _analyze_move(self, fen: str, move_uci: str) -> Optional[Tuple[bool, int]]:
        """Analyze a move to detect if it leads to forced mate."""
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return None

            board.push(move)
            info = self.engine.analyse(
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
    """Pool of Stockfish engine workers."""

    def __init__(self, stockfish_path: str, num_engines: int):
        self.work_queue: queue.Queue = queue.Queue()
        self.results: dict = {}
        self.lock = threading.Lock()
        self.workers: list[EngineWorker] = []

        print(f"Starting {num_engines} Stockfish engines...")
        for _ in tqdm(range(num_engines), desc="Initializing engines"):
            worker = EngineWorker(stockfish_path, self.work_queue, self.results, self.lock)
            worker.start()
            self.workers.append(worker)
        print(f"All {num_engines} engines ready.\n")

    def submit(self, key: tuple, fen: str, move_uci: str):
        self.work_queue.put((key, fen, move_uci))

    def wait_and_get_results(self) -> dict:
        self.work_queue.join()
        with self.lock:
            results = dict(self.results)
            self.results.clear()
        return results

    def shutdown(self):
        for _ in self.workers:
            self.work_queue.put(None)
        for worker in self.workers:
            worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


def process_mate_batch(pool: EnginePool, batch: dict) -> list[dict]:
    """Process a batch of positions that have mate candidates."""
    batch_size = len(batch["fen"])
    mate_info: dict[tuple, bool] = {}

    for pos_idx in range(batch_size):
        for move_idx, (move, p_win) in enumerate(
            zip(batch["moves"][pos_idx], batch["p_win"][pos_idx])
        ):
            if p_win >= WIN_THRESHOLD or p_win <= LOSS_THRESHOLD:
                key = (pos_idx, move_idx)
                mate_info[key] = p_win >= WIN_THRESHOLD
                pool.submit(key, batch["fen"][pos_idx], move)

    analysis_results = pool.wait_and_get_results()

    results = []
    for pos_idx in range(batch_size):
        new_p_wins = []
        for move_idx, p_win in enumerate(batch["p_win"][pos_idx]):
            key = (pos_idx, move_idx)
            if key in mate_info:
                is_winning_move = mate_info[key]
                result = analysis_results.get(key)
                if result is not None:
                    is_winning, mate_in_n = result
                    new_p_wins.append(compute_mate_winrate(mate_in_n, is_winning))
                else:
                    new_p_wins.append(0.98 if is_winning_move else 0.02)
            else:
                # Non-mate move in a position that has some mate moves
                new_p_wins.append(0.02 + p_win * 0.96)

        results.append({
            "fen": batch["fen"][pos_idx],
            "moves": list(batch["moves"][pos_idx]),
            "p_win": new_p_wins,
        })

    return results


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset with refined mate-in-n win percentages"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stockfish", type=str, default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--num-positions", type=int, default=DEFAULT_NUM_POSITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-engines", type=int, default=DEFAULT_NUM_ENGINES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()

    input_path = os.environ.get("DATASET_PATH")
    if not input_path:
        print("ERROR: DATASET_PATH environment variable not set")
        return 1
    input_path = Path(input_path)

    print("=" * 60)
    print("MATE-REFINED DATASET GENERATION (Two-Phase)")
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

    # Load dataset
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(str(input_path))
    print(f"Loaded {len(dataset):,} positions")

    if len(dataset) > args.num_positions:
        print(f"Sampling {args.num_positions:,} positions (seed={args.seed})...")
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_positions))
    print()

    # =========================================================
    # PHASE 1: Split dataset and process non-mate positions fast
    # =========================================================
    print("Phase 1: Filtering positions...")

    # Split into mate vs non-mate positions
    mate_positions = dataset.filter(has_mate_candidate, desc="Finding mate positions")
    non_mate_positions = dataset.filter(
        lambda x: not has_mate_candidate(x),
        desc="Finding non-mate positions"
    )

    print(f"  Positions with mate candidates: {len(mate_positions):,}")
    print(f"  Positions without mates: {len(non_mate_positions):,}")
    print(f"  Mate ratio: {len(mate_positions) / len(dataset) * 100:.1f}%")
    print()

    # Fast remap for non-mate positions (no Stockfish needed)
    print("Phase 1: Remapping non-mate positions (fast)...")
    non_mate_processed = non_mate_positions.map(
        remap_non_mate_batch,
        batched=True,
        batch_size=10_000,
        desc="Remapping non-mate",
    )
    print()

    # =========================================================
    # PHASE 2: Stockfish analysis for mate positions only
    # =========================================================
    print("Phase 2: Analyzing mate positions with Stockfish...")

    mate_results = []
    with EnginePool(args.stockfish, args.num_engines) as pool:
        for start in tqdm(
            range(0, len(mate_positions), args.batch_size),
            desc="Processing mate batches",
        ):
            end = min(start + args.batch_size, len(mate_positions))
            batch = mate_positions[start:end]
            batch_results = process_mate_batch(pool, batch)
            mate_results.extend(batch_results)

    mate_processed = Dataset.from_list(mate_results)
    print()

    # =========================================================
    # PHASE 3: Combine and save
    # =========================================================
    print("Phase 3: Combining results...")

    # Concatenate both datasets
    final_dataset = concatenate_datasets([non_mate_processed, mate_processed])
    print(f"Final dataset: {len(final_dataset):,} positions")

    # Save
    if args.output.exists():
        print(f"Removing existing output at {args.output}")
        shutil.rmtree(args.output)

    print(f"Saving to {args.output}...")
    args.output.mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(str(args.output))

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
