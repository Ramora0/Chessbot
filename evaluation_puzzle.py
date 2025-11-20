"""Utility to evaluate a policy model on the processed puzzle dataset."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chess

import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from policy_index import policy_index
from tokenizer import process_fen


class RuntimeTokenizationCollate:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.act_token_id = tokenizer.token_to_id("<ACT>") if tokenizer else None

    def __call__(self, batch: Iterable[dict]) -> Dict[str, object]:
        batch_list = list(batch)
        
        if self.tokenizer:
            # Runtime tokenization
            input_ids_list = []
            for example in batch_list:
                fen = example["fen"]
                processed = process_fen(fen)
                encoding = self.tokenizer.encode(processed)
                input_ids_list.append(torch.tensor(encoding.ids, dtype=torch.long))
            
            input_ids = torch.stack(input_ids_list)
        else:
            # Pre-tokenized
            input_ids = torch.tensor([
                example["input_ids"] for example in batch_list
            ], dtype=torch.long)

        target_indices = torch.tensor([
            example["target_policy_index"] for example in batch_list
        ], dtype=torch.long)
        legal_masks = torch.tensor([
            example["legal_moves_mask"] for example in batch_list
        ], dtype=torch.bool)

        puzzle_ids = [example["puzzle_id"] for example in batch_list]
        ratings = [int(example.get("rating", 0)) for example in batch_list]

        return {
            "input_ids": input_ids,
            "target_indices": target_indices,
            "legal_masks": legal_masks,
            "puzzle_ids": puzzle_ids,
            "ratings": ratings,
        }


def _collate_fn(batch: Iterable[dict]) -> Dict[str, object]:
    # Legacy support for when no tokenizer is provided
    return RuntimeTokenizationCollate(tokenizer=None)(batch)


LOG10 = float(np.log(10.0))
ELO_K = LOG10 / 400.0
CLIP_EPS = 1e-6
DEFAULT_EVAL_DATASET_DIR = Path(
    "/fs/scratch/PAS3150/lees_stuff/processed_puzzles_eval")
DEFAULT_EVAL_CSV_PATH = Path("data/puzzles.csv")
MOVE_PROB_THRESHOLD = 0.01
MOVE_DISTRIBUTION_OUTPUT_DIR = Path("eval")
POLICY_MOVES: List[str] = policy_index


def generate_legal_moves_mask(fen: str) -> List[bool]:
    """Generate a boolean mask of legal moves for a given FEN position."""
    board = chess.Board(fen)
    legal_moves_set = {move.uci() for move in board.legal_moves}
    
    # Create mask: True if move is legal, False otherwise
    mask = [move in legal_moves_set for move in POLICY_MOVES]
    return mask


def get_target_policy_index(moves_str: str) -> int:
    """Extract the first move from the solution and return its policy index."""
    moves = moves_str.strip().split()
    if not moves:
        raise ValueError(f"No moves found in solution: {moves_str}")
    
    target_move = moves[0]
    try:
        return POLICY_MOVES.index(target_move)
    except ValueError:
        raise ValueError(f"Move {target_move} not found in policy_index")


def load_puzzles_from_csv(csv_path: str | Path) -> List[Dict[str, object]]:
    """Load puzzles from CSV file and prepare them for evaluation.
    
    Each puzzle with multiple moves in the solution will be expanded into
    multiple evaluation states (one per move). All states share the same
    puzzle_id so their probabilities can be multiplied together.
    
    Returns a list of dictionaries with keys:
    - puzzle_id: str (same for all moves in a puzzle)
    - rating: int
    - fen: str (position after opponent's move)
    - target_policy_index: int (the correct move to make)
    - legal_moves_mask: List[bool]
    """
    puzzles = []
    csv_path = Path(csv_path)
    
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = row['PuzzleId']
            rating = int(row['Rating'])
            initial_fen = row['FEN']
            moves_str = row['Moves']
            
            try:
                # Parse all moves in the solution
                moves = moves_str.strip().split()
                if not moves:
                    print(f"Warning: Skipping puzzle {puzzle_id}: no moves in solution")
                    continue
                
                # Create a board from the initial position
                board = chess.Board(initial_fen)
                
                # Process each move in the solution
                # Moves alternate: opponent (0), player (1), opponent (2), player (3), ...
                # We only evaluate the model on player moves (odd indices)
                for move_idx, move_uci in enumerate(moves):
                    # Apply opponent moves without evaluation
                    if move_idx % 2 == 0:
                        # This is an opponent move - just apply it to the board
                        try:
                            move = chess.Move.from_uci(move_uci)
                            board.push(move)
                        except Exception as e:
                            print(f"Warning: Invalid opponent move {move_uci} in puzzle {puzzle_id}: {e}")
                            break
                        continue
                    
                    # This is a player move - evaluate the model on this
                    current_fen = board.fen()
                    
                    # Generate legal moves for this position
                    legal_moves_mask = generate_legal_moves_mask(current_fen)
                    
                    # Get the target move index
                    try:
                        target_policy_index = POLICY_MOVES.index(move_uci)
                    except ValueError:
                        print(f"Warning: Move {move_uci} not in policy_index for puzzle {puzzle_id}")
                        break
                    
                    # Add this state to the puzzle list
                    puzzles.append({
                        'puzzle_id': puzzle_id,
                        'rating': rating,
                        'fen': current_fen,
                        'target_policy_index': target_policy_index,
                        'legal_moves_mask': legal_moves_mask,
                    })
                    
                    # Make the player move on the board to get to the next position
                    try:
                        move = chess.Move.from_uci(move_uci)
                        board.push(move)
                    except Exception as e:
                        print(f"Warning: Invalid player move {move_uci} in puzzle {puzzle_id}: {e}")
                        break

                    
            except Exception as e:
                print(f"Warning: Skipping puzzle {puzzle_id}: {e}")
                continue
    
    return puzzles



@dataclass
class EvaluationDetails:
    puzzle_order: List[str]
    puzzle_probabilities: Dict[str, float]
    puzzle_ratings: Dict[str, int]
    puzzle_states: Dict[str, List[Dict[str, object]]]


def _clip01(arr: np.ndarray, eps: float = CLIP_EPS) -> np.ndarray:
    """Clamp probabilities into (0, 1) to keep the optimizer stable."""
    return np.clip(arr, eps, 1.0 - eps)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def estimate_elo_mle(
    puzzle_ratings: np.ndarray,
    solve_probabilities: np.ndarray,
    init: Optional[float] = None,
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


def evaluate_puzzles(
    model: torch.nn.Module,
    dataset_dir: str | Path = DEFAULT_EVAL_DATASET_DIR,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    dataset: Optional[Dataset] = None,
    verbose: bool = False,
    tokenizer=None,
    csv_path: Optional[str | Path] = None,
) -> Tuple[List[Tuple[int, float]], EvaluationDetails]:
    """Run the model on the evaluation dataset and collect per-puzzle statistics.

    For every puzzle we compute the probability the model selects the correct move
    at each step (after masking illegal moves), then multiply those probabilities to
    obtain the overall chance of solving the full puzzle. Returned values are
    `(rating, percent_chance)` tuples for downstream analysis along with detailed
    state-level move distributions for downstream logging.
    """

    if csv_path is not None:
        # Load from CSV
        puzzles = load_puzzles_from_csv(csv_path)
        dataloader = DataLoader(
            puzzles,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=RuntimeTokenizationCollate(tokenizer=tokenizer),
        )
    elif dataset is not None:
        hf_dataset = dataset
        dataloader = DataLoader(
            hf_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=RuntimeTokenizationCollate(tokenizer=tokenizer),
        )
    else:
        hf_dataset = load_from_disk(str(dataset_dir))
        dataloader = DataLoader(
            hf_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=RuntimeTokenizationCollate(tokenizer=tokenizer),
        )

    model_device = device or next(model.parameters()).device
    model.eval()

    puzzle_probabilities: Dict[str, float] = defaultdict(lambda: 1.0)
    puzzle_ratings: Dict[str, int] = {}
    puzzle_states: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    puzzle_state_counters: Dict[str, int] = defaultdict(int)
    puzzle_order: List[str] = []
    encountered_puzzles: set[str] = set()

    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        dataloader_iter = tqdm(
            dataloader, desc="Evaluating puzzles", unit="batch") if verbose else dataloader
        for batch in dataloader_iter:
            input_ids = batch["input_ids"].to(model_device)
            legal_masks = batch["legal_masks"].to(model_device)
            target_indices = batch["target_indices"].to(model_device)

            outputs = model(
                input_ids=input_ids,
                return_dict=True,
            )
            logits = outputs.policy_logits

            masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
            probs = softmax(masked_logits)
            selected_probs = probs.gather(
                1, target_indices.unsqueeze(-1)).squeeze(-1)

            # Filter legal moves on GPU before CPU transfer
            prob_mask = probs >= MOVE_PROB_THRESHOLD
            filtered_legal_masks = legal_masks & prob_mask

            # Batch CPU transfer - single operation
            batch_cpu = {
                'input_ids': input_ids.cpu(),
                'logits': logits.cpu(),
                'probs': probs.cpu(),
                'selected_probs': selected_probs.cpu(),
                'target_indices': target_indices.cpu(),
                'filtered_legal_masks': filtered_legal_masks.cpu(),
            }

            puzzle_ids = batch["puzzle_ids"]
            ratings = batch["ratings"]

            for batch_index, (puzzle_id, rating) in enumerate(
                zip(puzzle_ids, ratings)
            ):
                if puzzle_id not in encountered_puzzles:
                    puzzle_order.append(puzzle_id)
                    encountered_puzzles.add(puzzle_id)

                prob = float(batch_cpu['selected_probs'][batch_index])
                puzzle_probabilities[puzzle_id] *= prob
                puzzle_ratings[puzzle_id] = rating

                target_idx = int(batch_cpu['target_indices'][batch_index])
                correct_move = POLICY_MOVES[target_idx]
                correct_move_probability = float(
                    batch_cpu['probs'][batch_index][target_idx])
                correct_move_logit = float(
                    batch_cpu['logits'][batch_index][target_idx])

                # Get filtered legal moves
                legal_indices = torch.nonzero(
                    batch_cpu['filtered_legal_masks'][batch_index], as_tuple=False
                ).flatten()

                moves: List[Dict[str, object]] = []
                if legal_indices.numel() > 0:
                    legal_probs = batch_cpu['probs'][batch_index][legal_indices]
                    legal_logits = batch_cpu['logits'][batch_index][legal_indices]
                    for move_idx, move_prob, move_logit in zip(
                        legal_indices.tolist(), legal_probs.tolist(), legal_logits.tolist()
                    ):
                        moves.append(
                            {
                                "move": POLICY_MOVES[move_idx],
                                "move_id": move_idx,
                                "probability": float(move_prob),
                                "logit": float(move_logit),
                            }
                        )
                    moves.sort(
                        key=lambda item: item["probability"], reverse=True)

                puzzle_states[puzzle_id].append(
                    {
                        "state_index": puzzle_state_counters[puzzle_id],
                        "input_ids": batch_cpu['input_ids'][batch_index].tolist(),
                        "correct_move": correct_move,
                        "correct_move_id": target_idx,
                        "correct_move_probability": correct_move_probability,
                        "correct_move_logit": correct_move_logit,
                        "moves": moves,
                    }
                )
                puzzle_state_counters[puzzle_id] += 1

    results: List[Tuple[int, float]] = []
    for puzzle_id in puzzle_order:
        combined_prob = puzzle_probabilities.get(puzzle_id, 0.0)
        results.append((puzzle_ratings.get(puzzle_id, 0), combined_prob))

    details = EvaluationDetails(
        puzzle_order=puzzle_order,
        puzzle_probabilities=dict(puzzle_probabilities),
        puzzle_ratings=dict(puzzle_ratings),
        puzzle_states=dict(puzzle_states),
    )

    return results, details


def _write_move_distribution_log(
    details: EvaluationDetails,
    per_puzzle_results: List[Tuple[int, float]],
    elo: float,
    elo_se: float,
    expected_solved_puzzles: float,
    solve_percentage: float,
    output_dir: Path = MOVE_DISTRIBUTION_OUTPUT_DIR,
) -> Path:
    """Persist move probability samples and aggregate solve stats to disk."""

    total_puzzles = len(details.puzzle_order)
    if total_puzzles == 0:
        raise ValueError(
            "No puzzles were evaluated; cannot write move distributions.")

    selection_stride = max(1, 1_000 // 5)
    selected_indices = list(range(0, total_puzzles, selection_stride))
    last_index = total_puzzles - 1
    if last_index not in selected_indices:
        selected_indices.append(last_index)
    selected_indices = sorted(
        set(idx for idx in selected_indices if 0 <= idx < total_puzzles))

    # Map puzzle index to solve probability using the ordered results list to avoid
    # any mismatch between the dictionary order and the evaluation order.
    index_to_solve_prob = {
        idx: solve_prob for idx, (_, solve_prob) in enumerate(per_puzzle_results)
    }

    selected_puzzles: List[Dict[str, object]] = []
    for selected_index in selected_indices:
        puzzle_id = details.puzzle_order[selected_index]
        states = details.puzzle_states.get(puzzle_id, [])
        states_sorted = sorted(states, key=lambda state: state["state_index"])

        selected_puzzles.append(
            {
                "puzzle_id": puzzle_id,
                "index": selected_index,
                "rating": details.puzzle_ratings.get(puzzle_id, 0),
                "solve_probability": float(
                    index_to_solve_prob.get(selected_index, 0.0)
                ),
                "states": states_sorted,
            }
        )

    selected_puzzles.sort(key=lambda item: item["rating"], reverse=True)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    elo_label = f"{elo:.2f}"
    safe_elo_label = elo_label.replace(".", "-")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / \
        f"move-distributions_elo-{safe_elo_label}_{timestamp_str}.json"

    solve_percentage_value = None
    if not np.isnan(solve_percentage):
        solve_percentage_value = float(solve_percentage)

    payload = {
        "generated_at": timestamp_str,
        "elo_estimate": float(elo),
        "elo_standard_error": float(elo_se),
        "puzzle_count": total_puzzles,
        "expected_solved_puzzles": float(expected_solved_puzzles),
        "expected_solve_percentage": solve_percentage_value,
        "probability_threshold": MOVE_PROB_THRESHOLD,
        "selected_indices": selected_indices,
        "puzzles": selected_puzzles,
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    return output_path


def evaluate_model_elo(
    model: torch.nn.Module,
    dataset_dir: str | Path = DEFAULT_EVAL_DATASET_DIR,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    init_rating: Optional[float] = None,
    dataset: Optional[Dataset] = None,
    verbose: bool = False,
    tokenizer=None,
    csv_path: Optional[str | Path] = None,
) -> Tuple[float, float, float]:
    """
    Run evaluation and return an Elo estimate, its standard error, and the
    expected puzzle solve percentage across the evaluation set.

    The estimator treats each puzzle solve probability as an independent Bernoulli
    trial under the Elo logistic model and fits the single Elo parameter R.
    """

    per_puzzle_results, eval_details = evaluate_puzzles(
        model=model,
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        device=device,
        dataset=dataset,
        verbose=verbose,
        tokenizer=tokenizer,
        csv_path=csv_path,
    )

    if not per_puzzle_results:
        raise ValueError(
            "Evaluation produced no puzzle results; cannot estimate Elo.")

    ratings = np.array(
        [rating for rating, _ in per_puzzle_results], dtype=np.float64)
    solve_probabilities = np.array([
        probability for _, probability in per_puzzle_results
    ], dtype=np.float64)

    total_puzzles = solve_probabilities.size
    expected_solved_puzzles = float(np.sum(solve_probabilities))
    if total_puzzles > 0:
        solve_percentage = float(
            (expected_solved_puzzles / float(total_puzzles)) * 100.0
        )
    else:
        solve_percentage = float("nan")

    if init_rating is None and len(ratings) > 0:
        init_rating = float(np.median(ratings))

    elo, elo_se = estimate_elo_mle(
        puzzle_ratings=ratings,
        solve_probabilities=solve_probabilities,
        init=init_rating,
    )

    output_path = _write_move_distribution_log(
        details=eval_details,
        per_puzzle_results=per_puzzle_results,
        elo=elo,
        elo_se=elo_se,
        expected_solved_puzzles=expected_solved_puzzles,
        solve_percentage=solve_percentage,
    )
    print(f"Wrote move distribution log to '{output_path}'.")

    if total_puzzles > 0:
        print(
            "Expected puzzle solve rate: "
            f"{solve_percentage:.2f}% "
            f"({expected_solved_puzzles:.2f}/{total_puzzles})"
        )
    else:
        print("Expected puzzle solve rate: nan (no puzzles evaluated)")

    return elo, elo_se, solve_percentage


if __name__ == "__main__":
    import torch
    from model import ChessPolicyValueModel

    # Configuration - same as test.py
    CHECKPOINT_PATH = "./long"

    print("Loading model from checkpoint...")
    model = ChessPolicyValueModel.from_pretrained_compiled(CHECKPOINT_PATH)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print("Starting puzzle evaluation...")
    print()

    # Run evaluation
    elo, elo_se, solve_percentage = evaluate_model_elo(
        model=model,
        device=device,
        batch_size=256,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("PUZZLE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Estimated ELO: {elo:.0f} ± {elo_se:.0f}")
    print(f"Solve percentage: {solve_percentage:.2f}%")
    print("=" * 60)
