"""Test ROOK-CLF-9m model from HuggingFace on puzzle evaluation."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import ROOK utilities
from rook_tokenizer import BCChessPolicy


# Evaluation constants (same as evaluation_puzzle.py)
LOG10 = float(np.log(10.0))
ELO_K = LOG10 / 400.0
CLIP_EPS = 1e-6
PUZZLES_CSV_PATH = Path("data/puzzles.csv")
BATCH_SIZE = 256


def process_fen_for_rook(fen: str) -> str:
    """Convert standard FEN to ROOK's 77-character format."""
    position, turn, castling, en_passant, halfmove, fullmove = fen.split(" ")

    # Expand empty squares: "8" → "........"
    position = re.sub(r'\d', lambda m: "." * int(m.group()), position)
    position = position.replace("/", "")  # Remove row separators

    # Pad castling to 4 chars
    castling = castling.ljust(4, ".")

    # Pad en passant to 2 chars
    en_passant = en_passant.ljust(2, ".")

    # Pad halfmove to 3 chars
    halfmove = halfmove.ljust(2, ".") + "."

    # Pad fullmove to 3 chars
    fullmove = fullmove.ljust(3, ".")

    return "".join([position, turn, castling, en_passant, halfmove, fullmove])


def _clip01(arr: np.ndarray, eps: float = CLIP_EPS) -> np.ndarray:
    """Clamp probabilities into (0, 1) to keep the optimizer stable."""
    return np.clip(arr, eps, 1.0 - eps)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def estimate_elo_mle(
    puzzle_ratings: np.ndarray,
    solve_probabilities: np.ndarray,
    init: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> Tuple[float, float]:
    """Maximum-likelihood Elo estimate under the logistic solve model."""

    ratings = np.asarray(puzzle_ratings, dtype=np.float64)
    q = _clip01(np.asarray(solve_probabilities, dtype=np.float64))

    if init is None:
        init = float(np.median(ratings))
    rating_estimate = float(init)

    for _ in range(max_iter):
        logits = ELO_K * (rating_estimate - ratings)
        solve_probs = _sigmoid(logits)
        gradient = np.sum((q - solve_probs) * ELO_K)
        fisher_info = np.sum(solve_probs * (1.0 - solve_probs) * (ELO_K ** 2))
        if fisher_info <= 0.0:
            break
        step = gradient / fisher_info
        new_rating = rating_estimate + step
        if abs(new_rating - rating_estimate) < tol:
            rating_estimate = new_rating
            break
        rating_estimate = new_rating

    logits = ELO_K * (rating_estimate - ratings)
    solve_probs = _sigmoid(logits)
    fisher_info = np.sum(solve_probs * (1.0 - solve_probs) * (ELO_K ** 2))
    se = 1.0 / np.sqrt(fisher_info) if fisher_info > 0.0 else float("nan")

    return rating_estimate, se


class PuzzleDataset(Dataset):
    """Dataset for chess puzzles from CSV."""

    def __init__(self, csv_path: Path):
        self.puzzles = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse the FEN and the correct move (first move in Moves column)
                fen = row['FEN']
                moves = row['Moves'].split()
                if not moves:
                    continue
                # First move is what the model should predict
                correct_move = moves[0]
                rating = int(row['Rating'])
                puzzle_id = row['PuzzleId']

                self.puzzles.append({
                    'puzzle_id': puzzle_id,
                    'fen': fen,
                    'correct_move': correct_move,
                    'rating': rating,
                })

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return self.puzzles[idx]


def collate_fn(batch: List[dict]) -> Dict:
    """Collate function for DataLoader."""
    puzzle_ids = [item['puzzle_id'] for item in batch]
    fens = [item['fen'] for item in batch]
    correct_moves = [item['correct_move'] for item in batch]
    ratings = [item['rating'] for item in batch]

    return {
        'puzzle_ids': puzzle_ids,
        'fens': fens,
        'correct_moves': correct_moves,
        'ratings': ratings,
    }


def evaluate_model_elo(
    policy: BCChessPolicy,
    dataset: PuzzleDataset,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Run evaluation and return an Elo estimate, its standard error, and the
    expected puzzle solve percentage across the evaluation set.
    """

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Track per-puzzle solve probabilities
    puzzle_probabilities: Dict[str, float] = defaultdict(lambda: 1.0)
    puzzle_ratings: Dict[str, int] = {}
    puzzle_order: List[str] = []
    encountered_puzzles: set[str] = set()

    dataloader_iter = tqdm(
        dataloader, desc="Evaluating puzzles", unit="batch") if verbose else dataloader

    for batch in dataloader_iter:
        puzzle_ids = batch['puzzle_ids']
        fens = batch['fens']
        correct_moves = batch['correct_moves']
        ratings = batch['ratings']

        # Convert FENs to ROOK format (77 chars + [CLS])
        inputs = [process_fen_for_rook(fen) + "[CLS]" for fen in fens]

        # Use the pipeline to get all predictions with scores (top_k=None returns all)
        predictions = policy._pipeline(inputs, top_k=None)

        # For each puzzle, get the probability of the correct move
        for i, (puzzle_id, correct_move, rating) in enumerate(
            zip(puzzle_ids, correct_moves, ratings)
        ):
            if puzzle_id not in encountered_puzzles:
                puzzle_order.append(puzzle_id)
                encountered_puzzles.add(puzzle_id)

            # Find the correct move in the predictions
            pred_list = predictions[i]  # List of {label: move, score: prob}
            prob = 0.0
            for pred in pred_list:
                if pred['label'] == correct_move:
                    prob = float(pred['score'])
                    break

            puzzle_probabilities[puzzle_id] *= prob
            puzzle_ratings[puzzle_id] = rating

    # Prepare results for ELO estimation
    per_puzzle_results: List[Tuple[int, float]] = []
    for puzzle_id in puzzle_order:
        combined_prob = puzzle_probabilities.get(puzzle_id, 0.0)
        per_puzzle_results.append(
            (puzzle_ratings.get(puzzle_id, 0), combined_prob))

    if not per_puzzle_results:
        raise ValueError(
            "Evaluation produced no puzzle results; cannot estimate Elo.")

    ratings_arr = np.array(
        [rating for rating, _ in per_puzzle_results], dtype=np.float64
    )
    solve_probabilities_arr = np.array(
        [probability for _, probability in per_puzzle_results], dtype=np.float64
    )

    total_puzzles = solve_probabilities_arr.size
    expected_solved_puzzles = float(np.sum(solve_probabilities_arr))
    if total_puzzles > 0:
        solve_percentage = float(
            (expected_solved_puzzles / float(total_puzzles)) * 100.0
        )
    else:
        solve_percentage = float("nan")

    init_rating = float(np.median(ratings_arr)) if len(
        ratings_arr) > 0 else None

    elo, elo_se = estimate_elo_mle(
        puzzle_ratings=ratings_arr,
        solve_probabilities=solve_probabilities_arr,
        init=init_rating,
    )

    if total_puzzles > 0:
        print(
            f"\nExpected puzzle solve rate: "
            f"{solve_percentage:.2f}% "
            f"({expected_solved_puzzles:.2f}/{total_puzzles})"
        )
    else:
        print("\nExpected puzzle solve rate: nan (no puzzles evaluated)")

    return elo, elo_se, solve_percentage


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("ROOK-CLF-9m Model Evaluation")
    print("=" * 80)

    # Load model and tokenizer from HuggingFace using BCChessPolicy
    print("\nLoading ROOK-CLF-9m model from HuggingFace...")
    model_name = "jrahn/ROOK-CLF-9m"
    tokenizer_name = model_name  # Same location

    try:
        # Create BCChessPolicy which handles model and tokenizer loading
        policy = BCChessPolicy(
            model=model_name,
            tokenizer=tokenizer_name,
            batch_size=BATCH_SIZE,
            train_task="clf",
            filter_illegal=False,  # We don't need to filter, we just need all probs
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load puzzle dataset
    print(f"\nLoading puzzles from {PUZZLES_CSV_PATH}...")
    if not PUZZLES_CSV_PATH.exists():
        print(f"Error: Puzzle file not found at {PUZZLES_CSV_PATH}")
        return

    dataset = PuzzleDataset(PUZZLES_CSV_PATH)
    print(f"Loaded {len(dataset)} puzzles")

    # Run evaluation
    print("\nStarting puzzle evaluation...")
    print()

    elo, elo_se, solve_percentage = evaluate_model_elo(
        policy=policy,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        verbose=True,
    )

    print()
    print("=" * 80)
    print("PUZZLE EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Estimated ELO: {elo:.0f} ± {elo_se:.0f}")
    print(f"Solve percentage: {solve_percentage:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
