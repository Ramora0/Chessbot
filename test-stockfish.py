from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import chess
import chess.engine

# Engine configuration
STOCKFISH_PATH = Path("/usr/games/stockfish")
ENGINE_TIME_LIMIT = 0.002  # seconds
ENGINE_DEPTH = None
ENGINE_NODES = None
ENGINE_MULTIPV = 300
ENGINE_THREADS = 1
ENGINE_HASH_MB = 16

# Dataset configuration
PUZZLES_CSV_PATH = Path("data/puzzles.csv")
SAMPLE_RATIO = 0.01  # Evaluate 1% of the puzzles for timing
RANDOM_SEED = 2024
FEN_COLUMN_INDEX = 4  # 0-based index for the FEN column in puzzles.csv


def build_limit() -> chess.engine.Limit:
    if ENGINE_NODES is not None:
        return chess.engine.Limit(nodes=ENGINE_NODES)
    if ENGINE_DEPTH is not None:
        return chess.engine.Limit(depth=ENGINE_DEPTH)
    return chess.engine.Limit(time=max(ENGINE_TIME_LIMIT, 1e-4))


def score_to_cp(score: chess.engine.PovScore, color: chess.Color) -> float:
    pov_score = score.pov(color)
    cp = pov_score.score(mate_score=100000)
    if cp is not None:
        return float(cp)
    mate_value = pov_score.mate()
    if mate_value is None:
        return 0.0
    return 100000.0 if mate_value > 0 else -100000.0


def extract_scores(
    info: Iterable[chess.engine.InfoDict],
    board: chess.Board,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for entry in info:
        pv = entry.get("pv")
        if not pv:
            continue
        move = pv[0]
        if not isinstance(move, chess.Move):
            continue
        score = entry.get("score")
        if score is None:
            continue
        scores[move.uci()] = score_to_cp(score, board.turn)
    return scores


def evaluate_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    requested_multipv: int,
) -> List[Tuple[str, float]]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    multipv = min(requested_multipv, len(legal_moves))
    info = engine.analyse(board, limit, multipv=multipv)
    move_scores = extract_scores(info, board)
    ordered_scores: List[Tuple[str, float]] = []
    for move in legal_moves:
        move_uci = move.uci()
        ordered_scores.append((move_uci, move_scores.get(move_uci, 0.0)))
    return ordered_scores


def load_fens(csv_path: Path, sample_ratio: float) -> Tuple[Sequence[str], int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Puzzle CSV not found at {csv_path}")

    fens: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) <= FEN_COLUMN_INDEX:
                continue
            fen = row[FEN_COLUMN_INDEX].strip()
            if not fen or fen.lower() == "fen":
                continue
            fens.append(fen)

    total = len(fens)
    if total == 0:
        raise ValueError("No FEN positions found in the CSV")

    random_generator = random.Random(RANDOM_SEED)
    sample_size = max(1, int(total * sample_ratio))
    if sample_size >= total:
        return fens, total
    sample = random_generator.sample(fens, sample_size)
    return sample, total


def analyse_sample(fens: Sequence[str]) -> Tuple[float, List[float]]:
    engine_path = STOCKFISH_PATH.expanduser()
    if not engine_path.exists():
        raise FileNotFoundError(f"Stockfish binary not found at {engine_path}")

    limit = build_limit()
    per_position_times: List[float] = []

    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({
            "Threads": ENGINE_THREADS,
            "Hash": ENGINE_HASH_MB,
        })

        for fen in fens:
            board = chess.Board(fen)
            start = time.perf_counter()
            _ = evaluate_position(engine, board, limit, ENGINE_MULTIPV)
            per_position_times.append(time.perf_counter() - start)

    total_time = sum(per_position_times)
    return total_time, per_position_times


def print_metrics(total_puzzles: int, sample_count: int, total_sample_time: float) -> None:
    average_per_position = total_sample_time / max(sample_count, 1)
    estimated_full_time = average_per_position * total_puzzles

    print(f"Total puzzles in dataset: {total_puzzles}")
    print(f"Sampled puzzles analysed: {sample_count}")
    print(f"Total sample time: {total_sample_time:.3f} s")
    print(f"Average per puzzle: {average_per_position * 1000:.2f} ms")
    print(f"Estimated full evaluation time: {estimated_full_time:.3f} s")


def main() -> None:
    sample_fens, total_puzzles = load_fens(PUZZLES_CSV_PATH, SAMPLE_RATIO)

    overall_start = time.perf_counter()
    total_sample_time, _ = analyse_sample(sample_fens)
    overall_elapsed = time.perf_counter() - overall_start

    print_metrics(total_puzzles, len(sample_fens), total_sample_time)
    print(f"Wall-clock including overhead: {overall_elapsed:.3f} s")


if __name__ == "__main__":
    main()
