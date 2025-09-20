from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from policy_index import policy_index


EXPECTED_SEQ_LEN = 70


class ChessPolicyDataset(Dataset):
    """Dataset wrapper that defers expensive preprocessing to the collator."""

    def __init__(self, hf_dataset: HFDataset) -> None:
        self.dataset = hf_dataset
        self.policy_size = len(policy_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.dataset[int(index)]
        policy_values = example["policy"]
        if len(policy_values) != self.policy_size:
            raise ValueError(
                f"policy field expected length {self.policy_size}, "
                f"received {len(policy_values)}"
            )

        return {
            "policy": policy_values,
            "input_ids": example["input_ids"],
        }


class ChessPolicyCollator:
    """Collator that batches tensor creation for already-tokenized data."""

    def __init__(self) -> None:
        self.policy_size = len(policy_index)

    def __call__(self, batch: Iterable[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        items: List[Dict[str, object]] = list(batch)
        if not items:
            raise ValueError("Empty batch provided to ChessPolicyCollator")

        input_ids = torch.stack([item["input_ids"] for item in items]).long()

        if input_ids.ndim != 2 or input_ids.shape[1] != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Batch tokenized length {input_ids.shape[1]} does not match expected {EXPECTED_SEQ_LEN}"
            )

        policy_values = torch.stack([item["policy"] for item in items]).float()
        if policy_values.shape[1] != self.policy_size:
            raise ValueError(
                f"policy tensor expected width {self.policy_size}, "
                f"received {policy_values.shape[1]}"
            )

        policy_mask = policy_values >= 0
        policy = torch.where(policy_mask, policy_values,
                             torch.zeros_like(policy_values))

        return {
            "input_ids": input_ids,
            "policy": policy,
            "policy_mask": policy_mask.to(torch.bool),
        }


def create_dataloader(
    hf_dataset: HFDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = ChessPolicyDataset(hf_dataset)
    collator = ChessPolicyCollator()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
