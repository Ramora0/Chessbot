"""Utility to evaluate a policy model on the processed puzzle dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk

from policy_index import policy_index


def _collate_fn(batch: Iterable[dict]) -> Dict[str, object]:
    batch_list = list(batch)
    input_ids = torch.tensor([
        example["input_ids"] for example in batch_list
    ], dtype=torch.long)
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        raise ValueError("input_ids sequence too short to apply causal masking")
    causal_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    causal_mask[:, seq_len - 2 :] = True
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
        "causal_mask": causal_mask,
        "target_indices": target_indices,
        "legal_masks": legal_masks,
        "puzzle_ids": puzzle_ids,
        "ratings": ratings,
    }


LOG10 = float(np.log(10.0))
ELO_K = LOG10 / 400.0
CLIP_EPS = 1e-6
DEFAULT_EVAL_DATASET_DIR = Path(
    "/fs/scratch/PAS3150/lees_stuff/processed_puzzles_eval")
MOVE_PROB_THRESHOLD = 0.01
MOVE_DISTRIBUTION_OUTPUT_DIR = Path("eval")
POLICY_MOVES: List[str] = policy_index


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
) -> Tuple[List[Tuple[int, float]], EvaluationDetails]:
    """Run the model on the evaluation dataset and collect per-puzzle statistics.

    For every puzzle we compute the probability the model selects the correct move
    at each step (after masking illegal moves), then multiply those probabilities to
    obtain the overall chance of solving the full puzzle. Returned values are
    `(rating, percent_chance)` tuples for downstream analysis along with detailed
    state-level move distributions for downstream logging.
    """

    if dataset is not None:
        hf_dataset = dataset
    else:
        hf_dataset = load_from_disk(str(dataset_dir))
    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
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
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model_device)
            causal_mask = batch["causal_mask"].to(model_device)
            legal_masks = batch["legal_masks"].to(model_device)
            target_indices = batch["target_indices"].to(model_device)

            outputs = model(
                input_ids=input_ids,
                causal_mask=causal_mask,
                return_dict=True,
            )
            logits = outputs.policy_logits

            masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
            probs = softmax(masked_logits)
            selected_probs = probs.gather(
                1, target_indices.unsqueeze(-1)).squeeze(-1)

            legal_masks_cpu = legal_masks.cpu()
            probs_cpu = probs.cpu()
            selected_probs = selected_probs.cpu().tolist()
            target_indices_cpu = target_indices.cpu().tolist()
            puzzle_ids = batch["puzzle_ids"]
            ratings = batch["ratings"]

            for batch_index, (puzzle_id, rating, prob, target_idx) in enumerate(
                zip(puzzle_ids, ratings, selected_probs, target_indices_cpu)
            ):
                if puzzle_id not in encountered_puzzles:
                    puzzle_order.append(puzzle_id)
                    encountered_puzzles.add(puzzle_id)

                puzzle_probabilities[puzzle_id] *= float(prob)
                puzzle_ratings[puzzle_id] = rating

                correct_move_index = int(target_idx)
                correct_move = POLICY_MOVES[correct_move_index]
                correct_move_probability = float(probs_cpu[batch_index][correct_move_index])

                legal_indices = torch.nonzero(
                    legal_masks_cpu[batch_index], as_tuple=False
                ).flatten()
                moves: List[Dict[str, object]] = []
                if legal_indices.numel() > 0:
                    legal_probs = probs_cpu[batch_index][legal_indices]
                    for move_idx, move_prob in zip(
                        legal_indices.tolist(), legal_probs.tolist()
                    ):
                        if move_prob < MOVE_PROB_THRESHOLD:
                            continue
                        moves.append(
                            {
                                "move": POLICY_MOVES[move_idx],
                                "move_id": move_idx,
                                "probability": float(move_prob),
                            }
                        )
                    moves.sort(
                        key=lambda item: item["probability"], reverse=True)

                puzzle_states[puzzle_id].append(
                    {
                        "state_index": puzzle_state_counters[puzzle_id],
                        "correct_move": correct_move,
                        "correct_move_id": correct_move_index,
                        "correct_move_probability": correct_move_probability,
                        "moves": moves,
                    }
                )
                puzzle_state_counters[puzzle_id] += 1

    puzzle_probabilities_dict = dict(puzzle_probabilities)
    puzzle_states_dict = {key: value for key, value in puzzle_states.items()}

    results: List[Tuple[int, float]] = []
    for puzzle_id in puzzle_order:
        combined_prob = puzzle_probabilities_dict.get(puzzle_id, 0.0)
        results.append((puzzle_ratings.get(puzzle_id, 0), combined_prob))

    details = EvaluationDetails(
        puzzle_order=puzzle_order,
        puzzle_probabilities=puzzle_probabilities_dict,
        puzzle_ratings=dict(puzzle_ratings),
        puzzle_states=puzzle_states_dict,
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
