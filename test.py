"""Test script for generating self-play games with Stockfish supervision."""

import asyncio
from pathlib import Path

import torch

from generate_games import generate_games
from model import ChessPolicyValueModel
from tokenizer import create_tokenizer


def main():
    """Test generating a small number of games."""

    # Configuration
    CHECKPOINT_PATH = "./outputs/checkpoint-90000"
    STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"
    NUM_GAMES = 1
    BATCH_SIZE = 2
    TEMPERATURE = 1.0
    TOP_P = 0.9

    print("=" * 60)
    print("TESTING GAME GENERATION WITH STOCKFISH SUPERVISION")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = ChessPolicyValueModel.from_pretrained_compiled(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print()

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer.get_vocab())}")
    print()

    # Generate games
    games = asyncio.run(
        generate_games(
            model=model,
            stockfish_path=STOCKFISH_PATH,
            num_games=NUM_GAMES,
            device=device,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            verbose=True,
        )
    )

    print()
    print("=" * 60)
    print("GENERATION COMPLETE - INSPECTING DATA")
    print("=" * 60)
    print()

    # Inspect the generated data
    print(f"Generated {len(games)} games")
    print()

    for i, game in enumerate(games[:3]):  # Show first 3 games
        print(f"Game {i+1}:")
        print(f"  Result: {game.result}")
        print(f"  Num moves: {game.num_moves}")
        print(f"  Num positions: {len(game.positions)}")

        if game.positions:
            pos = game.positions[0]
            print(f"  First position:")
            print(f"    Input IDs shape: {pos.input_ids.shape}")
            print(f"    Stockfish policy shape: {pos.stockfish_policy.shape}")
            print(f"    Move played: {pos.move_played}")

            # Check how many legal moves had evaluations
            legal_moves = (pos.stockfish_policy >= 0).sum().item()
            print(f"    Legal moves evaluated: {legal_moves}")

            # Show top 3 moves by Stockfish policy
            top_probs, top_indices = torch.topk(
                pos.stockfish_policy, k=min(3, legal_moves))
            print(f"    Top 3 moves by Stockfish:")
            from policy_index import policy_index
            for prob, idx in zip(top_probs, top_indices):
                move_uci = policy_index[idx]
                print(f"      {move_uci}: {prob:.4f}")

        print()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
