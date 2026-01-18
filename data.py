from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from policy_index import policy_index
from loss_weights import ENDGAME_WEIGHT, ENDGAME_MATERIAL_THRESHOLD

# Material values for chess pieces (kings excluded - always present)
PIECE_MATERIAL_VALUES = {
    'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,  # White pieces
    'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1,  # Black pieces
}


def build_material_lookup(vocab_size: int, token_to_id: dict) -> torch.Tensor:
    """Build a lookup tensor mapping token IDs to (white_material, black_material).

    Args:
        vocab_size: Total vocabulary size
        token_to_id: Mapping from token string to token ID

    Returns:
        Tensor of shape [vocab_size, 2] where [:, 0] is white material and [:, 1] is black
    """
    material_lookup = torch.zeros(vocab_size, 2, dtype=torch.float32)

    for piece, value in PIECE_MATERIAL_VALUES.items():
        token = f"{piece}-p"
        if token in token_to_id:
            token_id = token_to_id[token]
            # White pieces are uppercase, black pieces are lowercase
            if piece.isupper():
                material_lookup[token_id, 0] = value  # White material
            else:
                material_lookup[token_id, 1] = value  # Black material

    return material_lookup


class ChessPolicyCollator:
    """Collator that batches tensor creation for already-tokenized data.

    Optionally applies masked token prediction by randomly masking 5% of tokens
    in 50% of examples. Also computes endgame weights for loss upweighting.
    """

    def __init__(
        self,
        mask_token_id: int | None = None,
        mask_prob: float = 0.05,
        material_lookup: Optional[torch.Tensor] = None,
    ) -> None:
        self.policy_size = len(policy_index)
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.material_lookup = material_lookup
        self.endgame_weight = ENDGAME_WEIGHT
        self.endgame_threshold = ENDGAME_MATERIAL_THRESHOLD

        # Maskable positions: board (0-63), castling (65-68)
        # Never mask: turn (64), en passant (69) - both are time-dependent
        self.maskable_positions = list(range(64)) + list(range(65, 69))

    def __call__(self, batch: Iterable[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        policy_list = []
        wdl_list = []
        legal_move_mask_list = []
        control_map_list = []

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

            if "control_map" in item:
                control_map = item["control_map"]
                if not isinstance(control_map, torch.Tensor):
                    control_map = torch.tensor(control_map, dtype=torch.float32)
                control_map_list.append(control_map)

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

        if control_map_list:
            control_maps = torch.stack(control_map_list)
            result["control_map"] = control_maps

        # Add masking-related fields if masking was applied
        if original_input_ids is not None:
            result["original_input_ids"] = original_input_ids
            result["masked_positions"] = masked_positions

        # Compute endgame weights for loss upweighting (done on CPU in dataloader workers)
        if self.material_lookup is not None:
            batch_size = input_ids.size(0)
            # Only look at board tokens (first 64 positions)
            board_tokens = input_ids[:, :64]  # [batch, 64]

            # Gather material values for each token
            material_per_square = self.material_lookup[board_tokens]  # [batch, 64, 2]

            # Sum material for white (index 0) and black (index 1)
            white_material = material_per_square[:, :, 0].sum(dim=1)  # [batch]
            black_material = material_per_square[:, :, 1].sum(dim=1)  # [batch]

            # Endgame: both sides have <= threshold material
            is_endgame = (white_material <= self.endgame_threshold) & \
                         (black_material <= self.endgame_threshold)

            # Apply endgame weight (3x by default), else 1.0
            endgame_weights = torch.where(
                is_endgame,
                torch.full((batch_size,), self.endgame_weight, dtype=torch.float32),
                torch.ones(batch_size, dtype=torch.float32)
            )
            result["endgame_weights"] = endgame_weights

        return result
