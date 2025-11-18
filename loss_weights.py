"""Centralized configuration for loss weighting."""

POLICY_LOSS_WEIGHT: float = 1
WDL_LOSS_WEIGHT: float = 0.3
# Loss weight for illegality prediction head
ILLEGALITY_HEAD_LOSS_WEIGHT: float = 0.2
MASKED_TOKEN_LOSS_WEIGHT: float = 0  # Loss weight for masked token prediction
