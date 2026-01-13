"""Centralized configuration for loss weighting."""

# Policy head uses two complementary losses on the same logits:
# 1. Softmax-based expected regret loss (original policy loss)
POLICY_LOSS_WEIGHT: float = 0.5 / 3

# 2. Sigmoid-based win% prediction loss
# This unified loss handles BOTH:
#   - Ranking legal moves by their actual win% (e.g., 52%, 48%, etc.)
#   - Illegality detection (illegal moves target 0% win rate, encouraging very negative logits)
MOVE_WINRATE_LOSS_WEIGHT: float = 0.4 / 0.5

# WDL head predicts position value distribution
WDL_LOSS_WEIGHT: float = 0 / 0.005

# Masked token prediction - helps model learn board state representation
MASKED_TOKEN_LOSS_WEIGHT: float = 1.0

# Temperature for softmax loss
TEMPERATURE: float = 0.5
