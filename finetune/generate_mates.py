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
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

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

# Time limit per position in seconds
MATE_SEARCH_TIME = 0.005

WIN_THRESHOLD = 0.9999
LOSS_THRESHOLD = 0.0001


class EngineWorker(threading.Thread):
    """Worker thread that owns a single Stockfish engine."""

    def __init__(self, stockfish_path: str, pool: "EnginePool", results: dict, lock: threading.Lock, pool_ref: "EnginePool"):
        super().__init__(daemon=True)
        self.stockfish_path = stockfish_path
        self.pool = pool
        self.results = results
        self.lock = lock
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def run(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Threads": 1})

        while True:
            item = self.pool.get_next_work()
            if item is None:
                break

            key, fen, move_uci, is_winning_move = item
            result = self._analyze_move(fen, move_uci, is_winning_move)

            with self.lock:
                self.results[key] = result
            with self.pool.count_lock:
                self.pool.completed_count += 1

        self.engine.quit()

    def _analyze_move(self, fen: str, move_uci: str, is_winning_move: bool) -> int:
        """
        Analyze a move to detect mate depth.

        Returns:
            Positive int: Mate in N moves (we are winning)
            Negative int: Getting mated in N moves (we are losing)
            0: Not a forced mate or analysis failed
        """
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return 0

            board.push(move)

            # Immediate checkmate
            if board.is_checkmate():
                return 1 if is_winning_move else -1

            info = self.engine.analyse(
                board,
                chess.engine.Limit(time=MATE_SEARCH_TIME)
            )

            score = info.get("score")
            if score is None:
                return 0

            # Score is from perspective of side to move (after our move)
            pov_score = score.relative
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                if mate_in is not None:
                    # mate_in < 0 means the side to move is getting mated
                    # mate_in > 0 means the side to move will deliver mate
                    # We made the move, so:
                    # - If mate_in < 0: opponent (who just moved) is getting mated = we win
                    # - If mate_in > 0: opponent will deliver mate = we lose
                    if mate_in < 0:
                        # We are winning (opponent getting mated)
                        return abs(mate_in)
                    else:
                        # We are losing (opponent will mate us)
                        return -mate_in
            return 0
        except Exception:
            return 0


class EnginePool:
    """Pool of Stockfish engine workers."""

    def __init__(self, stockfish_path: str, num_engines: int, work_items: list = None):
        self.results: dict = {}
        self.lock = threading.Lock()
        self.completed_count = 0
        self.count_lock = threading.Lock()
        self.workers: list[EngineWorker] = []

        # Use shared list + atomic index instead of queue (no per-item locking)
        self.work_items = work_items or []
        self.work_index = 0
        self.index_lock = threading.Lock()
        self.num_engines = num_engines
        self.stockfish_path = stockfish_path

    def start_workers(self):
        """Start worker threads after work items are populated."""
        print(f"Starting {self.num_engines} Stockfish engines...")
        for _ in tqdm(range(self.num_engines), desc="Initializing engines"):
            worker = EngineWorker(self.stockfish_path, self, self.results, self.lock, self)
            worker.start()
            self.workers.append(worker)
        print(f"All {self.num_engines} engines ready.\n")

    def get_next_work(self):
        """Get next work item (lock-free for reads, minimal lock for index update)."""
        with self.index_lock:
            idx = self.work_index
            if idx >= len(self.work_items):
                return None
            self.work_index += 1
        return self.work_items[idx]

    def get_completed_count(self) -> int:
        with self.count_lock:
            return self.completed_count

    def wait_and_get_results(self) -> dict:
        for worker in self.workers:
            worker.join()
        with self.lock:
            results = dict(self.results)
            self.results.clear()
        return results

    def shutdown(self):
        # Workers exit when get_next_work returns None
        for worker in self.workers:
            worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


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

        print(f"  Queued {len(work_items):,} moves for analysis...")

        with EnginePool(args.stockfish, args.num_engines, work_items) as pool:
            total_work = len(work_items)
            pool.start_workers()

            start_time = time.time()
            with tqdm(total=total_work, desc="Analyzing moves") as pbar:
                while True:
                    completed = pool.get_completed_count()
                    pbar.n = completed

                    elapsed = time.time() - start_time
                    if elapsed > 0 and completed > 0:
                        rate = completed / elapsed
                        pbar.set_postfix({"rate": f"{rate:.0f}/s"})

                    pbar.refresh()
                    if completed >= total_work:
                        break
                    time.sleep(0.1)

            analysis_results = pool.wait_and_get_results()

        print("  Populating mate depths...")

        # Fill in mate depths from analysis
        confirmed_mates = 0
        for (pos_idx, move_idx), mate_depth in analysis_results.items():
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
