from __future__ import annotations

from typing import Dict, Iterable, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import IterableDataset as HFIterableDataset

from policy_index import policy_index
from tokenizer import process_fen


EXPECTED_SEQ_LEN = 71


class ChessPolicyDataset(IterableDataset):
    """Streaming-only dataset wrapper that validates incoming examples."""

    def __init__(self, hf_dataset: HFIterableDataset, act_token_id: int = -1) -> None:
        if act_token_id is None:
            raise ValueError("act_token_id must be provided")
        if not isinstance(hf_dataset, HFIterableDataset):
            raise TypeError(
                "ChessPolicyDataset requires a streaming Hugging Face dataset")

        self.dataset = hf_dataset
        self.act_token_id = int(act_token_id)
        self.policy_size = len(policy_index)

    @property
    def is_streaming(self) -> bool:
        return True

    def __len__(self) -> int:
        raise TypeError(
            "ChessPolicyDataset wraps streaming data and has no length")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is not None:
            shard = self.dataset.shard(worker_info.num_workers, worker_info.id)
            iterator: Iterable[Dict[str, object]] = iter(shard)
        else:
            iterator = iter(self.dataset)

        for example in iterator:
            yield self._process_example(example)

    def _process_example(self, example: Dict[str, object]) -> Dict[str, torch.Tensor]:
        policy = torch.as_tensor(example["policy"])
        if policy.dtype != torch.float32:
            policy = policy.to(dtype=torch.float32)

        if policy.ndim != 1 or policy.shape[0] != self.policy_size:
            raise ValueError(
                f"policy field expected shape ({self.policy_size},), received {tuple(policy.shape)}"
            )

        input_ids = torch.as_tensor(example["input_ids"])
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(dtype=torch.long)

        if input_ids.ndim != 1:
            raise ValueError(
                f"input_ids expected 1D tensor, received shape {tuple(input_ids.shape)}"
            )

        seq_len = input_ids.shape[0]
        if seq_len == EXPECTED_SEQ_LEN:
            if input_ids[-1].item() != self.act_token_id:
                raise ValueError(
                    f"input_ids final token id {input_ids[-1].item()} does not match expected act token id {self.act_token_id}"
                )
        elif seq_len == EXPECTED_SEQ_LEN - 1:
            act_token = input_ids.new_tensor([self.act_token_id])
            input_ids = torch.cat((input_ids, act_token))
        else:
            raise ValueError(
                f"input_ids length {seq_len} does not match expected {EXPECTED_SEQ_LEN}"
            )

        wdl = torch.as_tensor(example["wdl"])
        if wdl.dtype != torch.float32:
            wdl = wdl.to(dtype=torch.float32)

        if wdl.ndim != 1 or wdl.shape[0] != 3:
            raise ValueError(
                f"wdl field expected shape (3,), received {tuple(wdl.shape)}"
            )

        return {
            "policy": policy,
            "input_ids": input_ids,
            "wdl": wdl,
        }


class ChessPolicyDatasetRuntimeTokenization(IterableDataset):
    """Streaming dataset that tokenizes FENs at runtime.
    
    This ensures the tokenizer used during training matches the one used during inference.
    """

    def __init__(self, hf_dataset: HFIterableDataset, tokenizer) -> None:
        if not isinstance(hf_dataset, HFIterableDataset):
            raise TypeError(
                "ChessPolicyDatasetRuntimeTokenization requires a streaming Hugging Face dataset")

        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.act_token_id = tokenizer.token_to_id("<ACT>")
        if self.act_token_id is None:
            raise ValueError("Tokenizer must have <ACT> token")
        
        self.policy_size = len(policy_index)

    @property
    def is_streaming(self) -> bool:
        return True

    def __len__(self) -> int:
        raise TypeError(
            "ChessPolicyDatasetRuntimeTokenization wraps streaming data and has no length")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is not None:
            shard = self.dataset.shard(worker_info.num_workers, worker_info.id)
            iterator: Iterable[Dict[str, object]] = iter(shard)
        else:
            iterator = iter(self.dataset)

        for example in iterator:
            yield self._process_example(example)

    def _process_example(self, example: Dict[str, object]) -> Dict[str, torch.Tensor]:
        # Extract policy
        policy = torch.as_tensor(example["policy"])
        if policy.dtype != torch.float32:
            policy = policy.to(dtype=torch.float32)

        if policy.ndim != 1 or policy.shape[0] != self.policy_size:
            raise ValueError(
                f"policy field expected shape ({self.policy_size},), received {tuple(policy.shape)}"
            )

        # Tokenize FEN at runtime
        fen = example["fen"]
        processed = process_fen(fen) + " <ACT>"
        encoding = self.tokenizer.encode(processed)
        input_ids = torch.tensor(encoding.ids, dtype=torch.long)

        if input_ids.shape[0] != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Tokenized FEN has length {input_ids.shape[0]}, expected {EXPECTED_SEQ_LEN}. FEN: {fen}"
            )

        # Extract WDL
        wdl = torch.as_tensor(example["wdl"])
        if wdl.dtype != torch.float32:
            wdl = wdl.to(dtype=torch.float32)

        if wdl.ndim != 1 or wdl.shape[0] != 3:
            raise ValueError(
                f"wdl field expected shape (3,), received {tuple(wdl.shape)}"
            )

        return {
            "policy": policy,
            "input_ids": input_ids,
            "wdl": wdl,
        }


class ChessPolicyCollator:
    """Collator that batches tensor creation for already-tokenized data.

    Optionally applies masked token prediction by randomly masking 15% of tokens
    in 50% of examples.
    """

    def __init__(self, mask_token_id: int | None = None, mask_prob: float = 0.15) -> None:
        self.policy_size = len(policy_index)
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

        # Maskable positions: board (0-63), castling (65-68), en passant (69)
        # Never mask: turn (64), <ACT> (70)
        self.maskable_positions = list(range(64)) + list(range(65, 70))

    def __call__(self, batch: Iterable[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        policy_list = []
        wdl_list = []
        for item in batch:
            input_ids_list.append(item["input_ids"])
            policy_list.append(item["policy"])
            wdl_list.append(item["wdl"])

        if not input_ids_list:
            raise ValueError("Empty batch provided to ChessPolicyCollator")

        input_ids = torch.stack(input_ids_list)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(dtype=torch.long)

        if input_ids.ndim != 2 or input_ids.shape[1] != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Batch tokenized length {input_ids.shape[1]} does not match expected {EXPECTED_SEQ_LEN}"
            )

        policy_values = torch.stack(policy_list)
        if policy_values.dtype != torch.float32:
            policy_values = policy_values.to(dtype=torch.float32)
        if policy_values.shape[1] != self.policy_size:
            raise ValueError(
                f"policy tensor expected width {self.policy_size}, "
                f"received {policy_values.shape[1]}"
            )

        wdl_values = torch.stack(wdl_list)
        if wdl_values.dtype != torch.float32:
            wdl_values = wdl_values.to(dtype=torch.float32)
        if wdl_values.shape[1] != 3:
            raise ValueError(
                f"wdl tensor expected width 3, received {wdl_values.shape[1]}"
            )

        # Apply masked token prediction if mask_token_id is provided
        original_input_ids = None
        masked_positions = None

        if self.mask_token_id is not None:
            # Store original input_ids before masking
            original_input_ids = input_ids.clone()

            # Initialize masked_positions (all False initially)
            batch_size, seq_len = input_ids.shape
            masked_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)

            # For each example, randomly decide whether to apply masking (50% chance)
            for i in range(batch_size):
                if torch.rand(1).item() < 0.5:
                    # Apply masking to this example
                    # Calculate number of tokens to mask (15% of maskable positions)
                    num_maskable = len(self.maskable_positions)
                    num_to_mask = max(1, int(num_maskable * self.mask_prob))

                    # Randomly select positions to mask
                    maskable_indices = torch.tensor(self.maskable_positions, dtype=torch.long)
                    perm = torch.randperm(num_maskable)[:num_to_mask]
                    positions_to_mask = maskable_indices[perm]

                    # Mark these positions as masked
                    masked_positions[i, positions_to_mask] = True

                    # For each masked position, decide whether to replace with [MASK] (90%) or keep (10%)
                    for pos in positions_to_mask:
                        if torch.rand(1).item() < 0.9:
                            # Replace with [MASK] token
                            input_ids[i, pos] = self.mask_token_id
                        # else: keep original token (10% of the time)

        result = {
            "input_ids": input_ids,
            "policy": policy_values,
            "wdl": wdl_values,
        }

        # Add masking-related fields if masking was applied
        if original_input_ids is not None:
            result["original_input_ids"] = original_input_ids
            result["masked_positions"] = masked_positions

        return result


def create_dataloader(
    hf_dataset: HFIterableDataset,
    act_token_id: int,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = ChessPolicyDataset(hf_dataset, act_token_id=act_token_id)
    collator = ChessPolicyCollator()

    if shuffle:
        raise ValueError(
            "Shuffling must be applied to the source dataset before wrapping in ChessPolicyDataset"
        )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
    )
