"""Centralized configuration for loss weighting."""

# Global pooling policy head
POLICY_LOSS_WEIGHT: float = 0.5 / 3  # Softmax expected regret
MOVE_WINRATE_LOSS_WEIGHT: float = 0.5 / 0.5  # Sigmoid win% prediction (legal moves only)
ILLEGALITY_LOSS_WEIGHT: float = 0.05  # Hinge loss to push illegal logits down (~10x less than policy)

# Winrate head predicts position value distribution
WINRATE_LOSS_WEIGHT: float = 0.1 / 3

# Masked token prediction - helps model learn board state representation
MASKED_TOKEN_LOSS_WEIGHT: float = 1.0

# Control map prediction - count attackers per square for each side
# Helps model learn board control and piece activity
CONTROL_MAP_LOSS_WEIGHT: float = 1.0 / 5.0

# Temperature for softmax loss
TEMPERATURE: float = 0.5

# Endgame upweighting: multiply loss by this factor for endgame positions
# Endgame defined as both players having <= 13 points of material
ENDGAME_WEIGHT: float = 1.0
ENDGAME_MATERIAL_THRESHOLD: int = 13
