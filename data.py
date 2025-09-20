from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from policy_index import policy_index
from tokenizer import create_tokenizer, process_fen_batch


EXPECTED_SEQ_LEN = 71


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

        item: Dict[str, object] = {"policy": policy_values}
        if "input_ids" in example:
            item["input_ids"] = example["input_ids"]
            if "attention_mask" in example:
                item["attention_mask"] = example["attention_mask"]
        else:
            item["fen"] = example["fen"]

        return item


class ChessPolicyCollator:
    """Collator that batches tokenization and tensor creation."""

    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer or create_tokenizer()
        self.policy_size = len(policy_index)

    def __call__(self, batch: Iterable[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        items: List[Dict[str, object]] = list(batch)
        if not items:
            raise ValueError("Empty batch provided to ChessPolicyCollator")

        if "input_ids" in items[0]:
            input_ids = torch.tensor(
                [item["input_ids"] for item in items], dtype=torch.long
            )
            if "attention_mask" in items[0]:
                attention_mask = torch.tensor(
                    [item["attention_mask"] for item in items], dtype=torch.long
                )
            else:
                attention_mask = torch.ones_like(input_ids)
        else:
            processed_fens = process_fen_batch(item["fen"] for item in items)
            encodings = self.tokenizer.encode_batch(processed_fens)
            input_ids = torch.tensor(
                [encoding.ids for encoding in encodings], dtype=torch.long
            )
            attention_mask = torch.tensor(
                [encoding.attention_mask for encoding in encodings], dtype=torch.long
            )

        if input_ids.ndim != 2 or input_ids.shape[1] != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Batch tokenized length {input_ids.shape[1]} does not match expected {EXPECTED_SEQ_LEN}"
            )

        policy_values = torch.tensor(
            [item["policy"] for item in items], dtype=torch.float32
        )
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
            "attention_mask": attention_mask,
            "policy": policy,
            "policy_mask": policy_mask.to(torch.bool),
        }

def create_dataloader(
    hf_dataset: HFDataset,
    tokenizer=None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = ChessPolicyDataset(hf_dataset)
    collator = ChessPolicyCollator(tokenizer=tokenizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
