#!/usr/bin/env python
"""
Generate a smaller dataset (2M positions) with refined mate-in-n win percentages.

Loads everything into memory for speed, then:
1. Fast pass: Remap non-mate positions (pure Python, no Stockfish)
2. Slow pass: Run Stockfish only on positions with mate-candidate moves
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset with refined mate-in-n win percentages"
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
    print("MATE-REFINED DATASET GENERATION")
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

    # Convert to plain Python lists (actually in memory, not lazy)
    print("Loading into memory...")
    num_positions = len(dataset)
    fens = []
    moves_list = []
    p_wins_list = []

    for row in tqdm(dataset, total=num_positions, desc="Loading to RAM"):
        fens.append(row["fen"])
        moves_list.append(row["moves"])
        p_wins_list.append(row["p_win"])

    print(f"Loaded {num_positions:,} positions into memory\n")

    # =========================================================
    # Phase 1: Identify mate positions and remap non-mate positions
    # =========================================================
    print("Phase 1: Scanning for mate candidates and remapping...")

    # Output arrays
    output_p_wins: list[list[float]] = []

    # Track positions needing Stockfish analysis
    mate_positions: list[int] = []  # indices of positions with mate candidates
    mate_moves: list[list[tuple[int, str, bool]]] = []  # [(move_idx, move_uci, is_winning), ...]

    for i in tqdm(range(num_positions), desc="Scanning positions"):
        p_wins = p_wins_list[i]
        moves = moves_list[i]

        # Check for mate candidates
        position_mate_moves = []
        for move_idx, p_win in enumerate(p_wins):
            if p_win >= WIN_THRESHOLD or p_win <= LOSS_THRESHOLD:
                position_mate_moves.append((move_idx, moves[move_idx], p_win >= WIN_THRESHOLD))

        if position_mate_moves:
            # Has mate candidates - will process with Stockfish later
            mate_positions.append(i)
            mate_moves.append(position_mate_moves)
            output_p_wins.append(None)  # Placeholder, will fill in later
        else:
            # No mate candidates - just remap
            output_p_wins.append([0.02 + p * 0.96 for p in p_wins])

    print(f"  Positions with mate candidates: {len(mate_positions):,}")
    print(f"  Positions without mates: {num_positions - len(mate_positions):,}")
    print(f"  Mate ratio: {len(mate_positions) / num_positions * 100:.1f}%\n")

    # =========================================================
    # Phase 2: Stockfish analysis for mate positions
    # =========================================================
    if mate_positions:
        print("Phase 2: Analyzing mate positions with Stockfish...")

        # Collect all work items
        # Key: (mate_pos_idx, move_idx) -> result
        work_items: list[tuple[tuple[int, int], str, str]] = []
        for mate_pos_idx, pos_idx in enumerate(mate_positions):
            fen = fens[pos_idx]
            for move_idx, move_uci, _ in mate_moves[mate_pos_idx]:
                work_items.append(((mate_pos_idx, move_idx), fen, move_uci))

        print(f"  Total moves to analyze: {len(work_items):,}")

        # Submit all work
        with EnginePool(args.stockfish, args.num_engines) as pool:
            for key, fen, move_uci in tqdm(work_items, desc="Submitting work"):
                pool.submit(key, fen, move_uci)

            print("  Waiting for analysis to complete...")
            analysis_results = pool.wait_and_get_results()

        print("  Building results...")

        # Fill in mate positions
        for mate_pos_idx, pos_idx in enumerate(tqdm(mate_positions, desc="Assembling")):
            p_wins = p_wins_list[pos_idx]
            mate_move_info = {m[0]: m[2] for m in mate_moves[mate_pos_idx]}  # move_idx -> is_winning

            new_p_wins = []
            for move_idx, p_win in enumerate(p_wins):
                if move_idx in mate_move_info:
                    is_winning_move = mate_move_info[move_idx]
                    result = analysis_results.get((mate_pos_idx, move_idx))
                    if result is not None:
                        is_winning, mate_in_n = result
                        new_p_wins.append(compute_mate_winrate(mate_in_n, is_winning))
                    else:
                        new_p_wins.append(0.98 if is_winning_move else 0.02)
                else:
                    new_p_wins.append(0.02 + p_win * 0.96)

            output_p_wins[pos_idx] = new_p_wins

    print()

    # =========================================================
    # Phase 3: Save output
    # =========================================================
    print("Phase 3: Saving output...")

    # Build final dataset
    output_data = [
        {"fen": fens[i], "moves": list(moves_list[i]), "p_win": output_p_wins[i]}
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
