"""Centralized configuration for loss weighting."""

POLICY_LOSS_WEIGHT: float = 1
WDL_LOSS_WEIGHT: float = 0.3
ILLEGALITY_HEAD_LOSS_WEIGHT: float = 0.5  # Loss weight for illegality prediction head
MASKED_TOKEN_LOSS_WEIGHT: float = 0.5  # Loss weight for masked token prediction
