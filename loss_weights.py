"""Centralized configuration for loss weighting."""

# OLD: Global pooling policy head (disabled by default, use attention head instead)
POLICY_LOSS_WEIGHT: float = 0.0  # (0.5 / 3)
MOVE_WINRATE_LOSS_WEIGHT: float = 0.0  # (0.5 / 0.5)

# NEW: Attention-based policy head (64x64 square attention) - DEFAULT
# Uses direct from_square->to_square attention mechanism
ATTENTION_POLICY_LOSS_WEIGHT: float = 0.5 / 3      # Softmax expected regret
ATTENTION_MOVE_WINRATE_LOSS_WEIGHT: float = 0.5 / 0.015  # Sigmoid win% prediction

# WDL head predicts position value distribution
WDL_LOSS_WEIGHT: float = 0 / 0.005

# Masked token prediction - helps model learn board state representation
MASKED_TOKEN_LOSS_WEIGHT: float = 1.0

# Temperature for softmax loss
TEMPERATURE: float = 0.5
