"""Centralized configuration for loss weighting."""

# Policy head uses two complementary losses on the same logits:
# 1. Softmax-based expected regret loss (original policy loss)
POLICY_LOSS_WEIGHT: float = 0.5 / 0.05

# 2. Sigmoid-based win% prediction loss
# This unified loss handles BOTH:
#   - Ranking legal moves by their actual win% (e.g., 52%, 48%, etc.)
#   - Learning that illegal moves are bad (target 0% win rate)
MOVE_WINRATE_LOSS_WEIGHT: float = 0.35 / 0.5

# WDL head predicts position value distribution
WDL_LOSS_WEIGHT: float = 0.15 / 0.005

ILLEGALITY_HEAD_LOSS_WEIGHT: float = 0.1 / 0.002

# Masked token prediction - helps model learn board state representation
MASKED_TOKEN_LOSS_WEIGHT: float = 1.0
