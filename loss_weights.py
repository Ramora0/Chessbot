"""Centralized configuration for loss weighting."""

# Global pooling policy head
POLICY_LOSS_WEIGHT: float = 0.5 / 3  # Softmax expected regret

# Sigmoid win% prediction - separate weights for legal and illegal moves
# Legal moves: mean BCE per position, expected ~0.5 after training
# Illegal moves: mean BCE per position, expected ~0.015 after training
# Weights chosen so both contribute ~1.0 initially (when both have ~0.69 raw loss)
LEGAL_MOVE_WINRATE_LOSS_WEIGHT: float = 1.0 / 0.5  # ~2.0, gives ~1.0 contribution at convergence
ILLEGAL_MOVE_WINRATE_LOSS_WEIGHT: float = 0.7 / 0.69  # ~1.0, conservative to not dominate initially

# WDL head predicts position value distribution
WDL_LOSS_WEIGHT: float = 0 / 0.005

# Masked token prediction - helps model learn board state representation
MASKED_TOKEN_LOSS_WEIGHT: float = 1.0

# Temperature for softmax loss
TEMPERATURE: float = 0.5

# Endgame upweighting: multiply loss by this factor for endgame positions
# Endgame defined as both players having <= 13 points of material
ENDGAME_WEIGHT: float = 1.0
ENDGAME_MATERIAL_THRESHOLD: int = 13
