"""Utility to evaluate a policy model on the processed puzzle dataset."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk


def _collate_fn(batch: Iterable[dict]) -> Dict[str, object]:
    batch_list = list(batch)
    input_ids = torch.tensor([
        example["input_ids"] for example in batch_list
    ], dtype=torch.long)
    attention_mask = torch.tensor([
        example["attention_mask"] for example in batch_list
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
        "attention_mask": attention_mask,
        "target_indices": target_indices,
        "legal_masks": legal_masks,
        "puzzle_ids": puzzle_ids,
        "ratings": ratings,
    }


def evaluate_puzzles(
    model: torch.nn.Module,
    dataset_dir: str = "processed_puzzles_eval",
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> List[Tuple[int, float]]:
    """Run the model on the evaluation dataset and return per-puzzle accuracy odds.

    For every puzzle we compute the probability the model selects the correct move
    at each step (after masking illegal moves), then multiply those probabilities to
    obtain the overall chance of solving the full puzzle. Returned values are
    `(rating, percent_chance)` tuples for downstream analysis.
    """

    dataset = load_from_disk(dataset_dir)
    dataloader = DataLoader(
        dataset,
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
            attention_mask = batch["attention_mask"].to(model_device)
            legal_masks = batch["legal_masks"].to(model_device)
            target_indices = batch["target_indices"].to(model_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
