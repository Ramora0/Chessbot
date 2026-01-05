from __future__ import annotations

from typing import Dict, Iterable

import torch

from policy_index import policy_index


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

    def __call__(self, batch: Iterable[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        policy_list = []
        wdl_list = []
        legal_move_mask_list = []

        for item in batch:
            # Convert lists to tensors (HF datasets return lists)
            input_ids = item["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids_list.append(input_ids)

            policy = item["policy"]
            if not isinstance(policy, torch.Tensor):
                policy = torch.tensor(policy, dtype=torch.float32)
            policy_list.append(policy)

            wdl = item["wdl"]
            if not isinstance(wdl, torch.Tensor):
                wdl = torch.tensor(wdl, dtype=torch.float32)
            wdl_list.append(wdl)

            if "legal_move_mask" in item:
                legal_move_mask = item["legal_move_mask"]
                if not isinstance(legal_move_mask, torch.Tensor):
                    legal_move_mask = torch.tensor(legal_move_mask, dtype=torch.float32)
                legal_move_mask_list.append(legal_move_mask)

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

        # Expect 128-bin value distribution
        wdl_width = wdl_values.shape[1]
        if wdl_width != 128:
            raise ValueError(
                f"wdl tensor expected width 128, received {wdl_width}"
            )

        # Apply token prediction loss on all examples (no actual masking)
        # This helps the model retain token information for downstream LLM use
        original_input_ids = None
        masked_positions = None

        if self.mask_token_id is not None:
            batch_size, seq_len = input_ids.shape
            original_input_ids = input_ids.clone()
            masked_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)

            # Enable token prediction loss on maskable positions for all examples
            maskable_indices = torch.tensor(self.maskable_positions, dtype=torch.long)
            masked_positions[:, maskable_indices] = True

        # Handle true_value if present (for value head metrics)
        true_value_list = []
        for item in batch:
            if "true_value" in item:
                true_val = item["true_value"]
                if not isinstance(true_val, torch.Tensor):
                    true_val = torch.tensor(true_val, dtype=torch.float32)
                true_value_list.append(true_val)

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

        return result
