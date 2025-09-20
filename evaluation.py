"""Utility to evaluate a policy model on the processed puzzle dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk


def _collate_fn(batch: Iterable[dict]) -> Dict[str, object]:
    batch_list = list(batch)
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


LOG10 = float(np.log(10.0))
ELO_K = LOG10 / 400.0
CLIP_EPS = 1e-6
DEFAULT_EVAL_DATASET_DIR = Path("/fs/scratch/PAS3150/lees_stuff/processed_puzzles_eval")


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
) -> List[Tuple[int, float]]:
    """Run the model on the evaluation dataset and return per-puzzle accuracy odds.

    For every puzzle we compute the probability the model selects the correct move
    at each step (after masking illegal moves), then multiply those probabilities to
    obtain the overall chance of solving the full puzzle. Returned values are
    `(rating, percent_chance)` tuples for downstream analysis.
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

    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        for batch in dataloader:
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

            selected_probs = selected_probs.cpu().tolist()
            puzzle_ids = batch["puzzle_ids"]
            ratings = batch["ratings"]

            for puzzle_id, rating, prob in zip(puzzle_ids, ratings, selected_probs):
                puzzle_probabilities[puzzle_id] *= float(prob)
                puzzle_ratings[puzzle_id] = rating

    results: List[Tuple[int, float]] = []
    for puzzle_id, combined_prob in puzzle_probabilities.items():
        results.append((puzzle_ratings.get(puzzle_id, 0), combined_prob))

    return results


def evaluate_model_elo(
    model: torch.nn.Module,
    dataset_dir: str | Path = DEFAULT_EVAL_DATASET_DIR,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    init_rating: Optional[float] = None,
    dataset: Optional[Dataset] = None,
) -> Tuple[float, float]:
    """
    Run evaluation and return an Elo estimate with its standard error.

    The estimator treats each puzzle solve probability as an independent Bernoulli
    trial under the Elo logistic model and fits the single Elo parameter R.
    """

    per_puzzle_results = evaluate_puzzles(
        model=model,
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        device=device,
        dataset=dataset,
    )

    if not per_puzzle_results:
        raise ValueError("Evaluation produced no puzzle results; cannot estimate Elo.")

    ratings = np.array([rating for rating, _ in per_puzzle_results], dtype=np.float64)
    solve_probabilities = np.array([
        probability for _, probability in per_puzzle_results
    ], dtype=np.float64)

    if init_rating is None and len(ratings) > 0:
        init_rating = float(np.median(ratings))

    elo, elo_se = estimate_elo_mle(
        puzzle_ratings=ratings,
        solve_probabilities=solve_probabilities,
        init=init_rating,
    )

    return elo, elo_se
