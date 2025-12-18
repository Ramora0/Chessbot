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
