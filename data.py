from __future__ import annotations

from typing import Dict, Iterable, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import IterableDataset as HFIterableDataset

from policy_index import policy_index
from tokenizer import process_fen


EXPECTED_SEQ_LEN = 70


class ChessPolicyDataset(IterableDataset):
    """Streaming-only dataset wrapper that validates incoming examples."""

    def __init__(self, hf_dataset: HFIterableDataset) -> None:
        if not isinstance(hf_dataset, HFIterableDataset):
            raise TypeError(
                "ChessPolicyDataset requires a streaming Hugging Face dataset")

        self.dataset = hf_dataset
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

        # Accept any reasonable sequence length (board state is typically 69-70 tokens)
        seq_len = input_ids.shape[0]
        if seq_len < 60 or seq_len > 80:
            raise ValueError(
                f"input_ids length {seq_len} is outside expected range [60, 80]"
            )

        wdl = torch.as_tensor(example["wdl"])
        if wdl.dtype != torch.float32:
            wdl = wdl.to(dtype=torch.float32)

        # Accept either 3-bin (W/D/L) or 128-bin value distribution
        if wdl.ndim != 1 or (wdl.shape[0] != 3 and wdl.shape[0] != 128):
            raise ValueError(
                f"wdl field expected shape (3,) or (128,), received {tuple(wdl.shape)}"
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
        processed = process_fen(fen)
        encoding = self.tokenizer.encode(processed)
        input_ids = torch.tensor(encoding.ids, dtype=torch.long)

        # Accept any reasonable sequence length (board state is typically 69-70 tokens)
        seq_len = input_ids.shape[0]
        if seq_len < 60 or seq_len > 80:
            raise ValueError(
                f"Tokenized FEN has length {seq_len}, outside expected range [60, 80]. FEN: {fen}"
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

    Optionally applies masked token prediction by randomly masking 5% of tokens
    in 50% of examples.
    """

    def __init__(self, mask_token_id: int | None = None, mask_prob: float = 0.05) -> None:
        self.policy_size = len(policy_index)
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

        # Maskable positions: board (0-63), castling (65-68)
        # Never mask: turn (64), en passant (69) - both are time-dependent
        self.maskable_positions = list(range(64)) + list(range(65, 69))


    def _convert_3bin_to_128bin(self, wdl_3bin: torch.Tensor) -> torch.Tensor:
        """Convert 3-bin WDL [W, D, L] to 128-bin value distribution.

        Args:
            wdl_3bin: [batch_size, 3] tensor with [win, draw, loss] probabilities

        Returns:
            [batch_size, 128] tensor with smooth distribution over win probability bins
        """
        device = wdl_3bin.device

        # Compute expected win probability: W + 0.5*D
        # (win counts as 1.0, draw as 0.5, loss as 0.0)
        win_prob = wdl_3bin[:, 0] + 0.5 * wdl_3bin[:, 1]  # [batch_size]

        # Create 128-bin smooth distribution centered on win_prob
        bin_centers = torch.linspace(0, 1, 128, device=device)  # [128]

        # Gaussian distribution with small std to create smooth target
        sigma = 0.05  # Smoothing factor (5% std)

        # Expand dimensions for broadcasting: [batch_size, 1] and [1, 128]
        win_prob_expanded = win_prob.unsqueeze(1)  # [batch_size, 1]
        bin_centers_expanded = bin_centers.unsqueeze(0)  # [1, 128]

        # Compute Gaussian: exp(-0.5 * ((x - mean) / sigma)^2)
        value_dist = torch.exp(-0.5 * ((bin_centers_expanded -
                               win_prob_expanded) / sigma) ** 2)

        # Normalize each row to sum to 1
        value_dist = value_dist / \
            value_dist.sum(dim=1, keepdim=True)  # [batch_size, 128]

        return value_dist

    def __call__(self, batch: Iterable[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        policy_list = []
        wdl_list = []
        legal_move_mask_list = []
        fen_list = []

        for item in batch:
            input_ids_list.append(item["input_ids"])
            policy_list.append(item["policy"])
            wdl_list.append(item["wdl"])
            if "legal_move_mask" in item:
                legal_move_mask_list.append(item["legal_move_mask"])

            # Collect FENs to pass to trainer for cross-worker duplicate detection
            if "fen" in item:
                fen_list.append(item["fen"])

        if not input_ids_list:
            raise ValueError("Empty batch provided to ChessPolicyCollator")

        input_ids = torch.stack(input_ids_list)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(dtype=torch.long)

        # Validate sequence length is reasonable (board state is typically 69-70 tokens)
        if input_ids.ndim != 2:
            raise ValueError(
                f"Batch input_ids expected 2D tensor, received shape {tuple(input_ids.shape)}"
            )
        seq_len = input_ids.shape[1]
        if seq_len < 60 or seq_len > 80:
            raise ValueError(
                f"Batch tokenized length {seq_len} is outside expected range [60, 80]"
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

        # Accept either 3-bin (W/D/L) or 128-bin value distribution
        wdl_width = wdl_values.shape[1]
        if wdl_width != 3 and wdl_width != 128:
            raise ValueError(
                f"wdl tensor expected width 3 or 128, received {wdl_width}"
            )

        # Convert 3-bin WDL to 128-bin format if needed
        if wdl_width == 3:
            wdl_values = self._convert_3bin_to_128bin(wdl_values)

        # Apply masked token prediction if mask_token_id is provided
        original_input_ids = None
        masked_positions = None

        if self.mask_token_id is not None:
            # Store original input_ids before masking
            original_input_ids = input_ids.clone()

            # Initialize masked_positions (all False initially)
            batch_size, seq_len = input_ids.shape
            masked_positions = torch.zeros(
                batch_size, seq_len, dtype=torch.bool)

            # For each example, randomly decide whether to apply masking (50% chance)
            for i in range(batch_size):
                if torch.rand(1).item() < 0.5:
                    # Apply masking to this example
                    # Calculate number of tokens to mask (15% of maskable positions)
                    num_maskable = len(self.maskable_positions)
                    num_to_mask = max(1, int(num_maskable * self.mask_prob))

                    # Randomly select positions to mask
                    maskable_indices = torch.tensor(
                        self.maskable_positions, dtype=torch.long)
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

        # Handle true_value if present (for value head metrics)
        true_value_list = []
        for item in batch:
            if "true_value" in item:
                true_value_list.append(item["true_value"])

        result = {
            "input_ids": input_ids,
            "policy": policy_values,
            "wdl": wdl_values,
        }

        if true_value_list:
            true_values = torch.stack(true_value_list)
            result["true_value"] = true_values

        if legal_move_mask_list:
            legal_move_masks = torch.stack(legal_move_mask_list)
            result["legal_move_mask"] = legal_move_masks

        # Add masking-related fields if masking was applied
        if original_input_ids is not None:
            result["original_input_ids"] = original_input_ids
            result["masked_positions"] = masked_positions

        # Add FENs for duplicate detection in trainer (list of strings)
        if fen_list:
            result["fens"] = fen_list

        return result


def create_dataloader(
    hf_dataset: HFIterableDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    dataset = ChessPolicyDataset(hf_dataset)
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
