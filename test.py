"""
Test script to inspect what the RAW dataset contains.
"""

from pathlib import Path
from datasets import load_from_disk
import chess

# Match train.py settings
PROCESSED_DATASET_DIR = "/fs/scratch/PAS2836/lees_stuff/action_value"

print(f"Loading RAW dataset from '{PROCESSED_DATASET_DIR}'...")
raw_dataset = load_from_disk(PROCESSED_DATASET_DIR).to_iterable_dataset()

print("\n" + "="*80)
print("First 10 RAW examples from dataset (as stored on disk):")
print("="*80)

fens_list = []

for i, example in enumerate(raw_dataset):
    if i >= 10:
        break

    print(f"\n{'='*80}")
    print(f"RAW EXAMPLE {i+1}")
    print(f"{'='*80}")

    # Print FEN
    fen = example["fen"]
    fens_list.append(fen)
    print(f"FEN: {fen}")

    # Get legal moves from chess library
    board = chess.Board(fen)
    chess_legal_moves = set(move.uci() for move in board.legal_moves)

    # Print RAW moves and p_win values
    moves = example["moves"]
    p_wins = example["p_win"]
    dataset_moves = set(moves)

    print(f"\nNumber of moves in dataset: {len(moves)}")
    print(f"Number of p_win values: {len(p_wins)}")
    print(f"Number of legal moves from chess library: {len(chess_legal_moves)}")

    # Compare dataset moves with chess library legal moves
    moves_match = dataset_moves == chess_legal_moves
    if moves_match:
        print("✓ Dataset moves MATCH chess library legal moves")
    else:
        print("✗ MISMATCH between dataset and chess library!")
        in_dataset_not_legal = dataset_moves - chess_legal_moves
        in_legal_not_dataset = chess_legal_moves - dataset_moves
        if in_dataset_not_legal:
            print(f"  Moves in dataset but NOT legal: {in_dataset_not_legal}")
        if in_legal_not_dataset:
            print(f"  Legal moves NOT in dataset: {in_legal_not_dataset}")

    # Print all moves with their raw p_win values
    print("\nRAW MOVES AND WIN PROBABILITIES:")
    move_data = list(zip(moves, p_wins))
    # Sort by p_win (best to worst)
    move_data.sort(key=lambda x: x[1], reverse=True)

    for move_uci, p_win in move_data:
        print(f"  {move_uci:6s}: p_win = {p_win:.6f}")

    # For first example, show even more detail
    if i == 0:
        print("\n" + "="*80)
        print("FIRST EXAMPLE - EXTRA DETAIL")
        print("="*80)
        print(f"Best move: {move_data[0][0]} with p_win = {move_data[0][1]:.6f}")
        print(f"Worst move: {move_data[-1][0]} with p_win = {move_data[-1][1]:.6f}")
        print(f"Win probability range: {move_data[-1][1]:.6f} to {move_data[0][1]:.6f}")
        print(f"Difference: {move_data[0][1] - move_data[-1][1]:.6f}")
        print("="*80)

print("\n" + "="*80)
print("SUMMARY - All FENs from first 10 examples:")
print("="*80)
for i, fen in enumerate(fens_list, 1):
    print(f"{i:2d}. {fen}")
print("="*80)
