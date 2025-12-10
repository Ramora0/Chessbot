"""Centralized configuration for loss weighting."""

POLICY_LOSS_WEIGHT: float = 1
WDL_LOSS_WEIGHT: float = 0.5
# Loss weight for illegality prediction head
ILLEGALITY_HEAD_LOSS_WEIGHT: float = 0.5
# Masked token prediction - helps model learn board state representation
MASKED_TOKEN_LOSS_WEIGHT: float = 1.0
