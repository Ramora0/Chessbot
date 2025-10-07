"""Test script to evaluate model from checkpoint against Stockfish."""

import torch

from model import ChessPolicyValueModel
from evaluation_game import evaluate_model_against_stockfish

# Configuration
CHECKPOINT_PATH = "./outputs/checkpoint-45000"
# Adjust path as needed
STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"
NUM_GAMES = 400
BATCH_SIZE = 128
# TRUE_PERCENT = 1  # 100% random moves for testing
TRUE_EVAL = 1350


def main():
    print("Loading model from checkpoint...")

    # Load model (handles _orig_mod. prefix from torch.compile)
    model = ChessPolicyValueModel.from_pretrained_compiled(CHECKPOINT_PATH)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(
        f"Starting evaluation with {NUM_GAMES} games (batch size: {BATCH_SIZE})...")
    print()

    # Run evaluation
    estimated_elo, std_error = evaluate_model_against_stockfish(
        model=model,
        stockfish_path=STOCKFISH_PATH,
        num_games=NUM_GAMES,
        batch_size=BATCH_SIZE,
        opponent_elo=TRUE_EVAL,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Estimated ELO: {estimated_elo:.0f} ± {std_error:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
