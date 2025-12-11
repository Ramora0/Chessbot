"""Centralized configuration for loss weighting."""

# Policy head uses two complementary losses on the same logits:
# 1. Softmax-based expected regret loss (original policy loss)
POLICY_LOSS_WEIGHT: float = 1.0
# 2. Sigmoid-based win% prediction loss (encourages correct ranking of all moves)
MOVE_WINRATE_LOSS_WEIGHT: float = 0.5

# WDL head predicts position value distribution
WDL_LOSS_WEIGHT: float = 0.5
# Illegality head - completely separate, predicts legal vs illegal
ILLEGALITY_HEAD_LOSS_WEIGHT: float = 0.5
# Masked token prediction - helps model learn board state representation
MASKED_TOKEN_LOSS_WEIGHT: float = 1.0
