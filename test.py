"""
Test script to inspect what the RAW dataset contains.
"""

from pathlib import Path
from datasets import load_from_disk
import chess
import numpy as np
from policy_index import policy_index

# Match train.py settings
PROCESSED_DATASET_DIR = "/fs/scratch/PAS3150/lees_stuff/processed_chessfens_consolidated"

print(f"Loading RAW dataset from '{PROCESSED_DATASET_DIR}'...")
raw_dataset = load_from_disk(PROCESSED_DATASET_DIR).to_iterable_dataset()

# Create move index lookup
move_to_idx = {move: idx for idx, move in enumerate(policy_index)}

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

    # Print policy and WDL information
    policy = np.array(example["policy"])
    wdl = np.array(example["wdl"])

    print(f"\nPolicy shape: {policy.shape}")
    print(f"WDL shape: {wdl.shape}")
    print(f"Number of legal moves from chess library: {len(chess_legal_moves)}")

    # Extract moves with policy values > 0
    move_policies = []
    for move_idx, policy_val in enumerate(policy):
        if policy_val > 0:
            move_uci = policy_index[move_idx]
            move_policies.append((move_uci, policy_val))

    print(f"Number of moves with positive policy values: {len(move_policies)}")

    # Sort by policy value (best to worst)
    move_policies.sort(key=lambda x: x[1], reverse=True)

    # Check if policy moves match legal moves
    policy_moves = set(move for move, _ in move_policies)
    if policy_moves == chess_legal_moves:
        print("✓ Policy moves MATCH chess library legal moves")
    else:
        print("✗ MISMATCH between policy and chess library!")
        in_policy_not_legal = policy_moves - chess_legal_moves
        in_legal_not_policy = chess_legal_moves - policy_moves
        if in_policy_not_legal:
            print(f"  Moves in policy but NOT legal: {in_policy_not_legal}")
        if in_legal_not_policy:
            print(f"  Legal moves NOT in policy: {in_legal_not_policy}")

    # Print moves with policy values
    print("\nMOVES AND POLICY VALUES:")
    for move_uci, policy_val in move_policies[:10]:  # Show top 10
        print(f"  {move_uci:6s}: policy = {policy_val:.6f}")
    if len(move_policies) > 10:
        print(f"  ... and {len(move_policies) - 10} more moves")

    # Print WDL information
    if len(wdl) == 3:
        print(f"\nWDL (Win/Draw/Loss): W={wdl[0]:.4f}, D={wdl[1]:.4f}, L={wdl[2]:.4f}")
    elif len(wdl) == 128:
        # 128 bins representing win probability distribution
        print(f"\nWDL: 128-bin value distribution (sum={wdl.sum():.4f})")
        # Find peak bin
        peak_bin = np.argmax(wdl)
        peak_value = peak_bin / 127.0  # Convert bin to win probability
        print(f"  Peak bin: {peak_bin} (win% ≈ {peak_value:.3f})")

    # For first example, show even more detail
    if i == 0:
        print("\n" + "="*80)
        print("FIRST EXAMPLE - EXTRA DETAIL")
        print("="*80)
        if move_policies:
            print(f"Best move: {move_policies[0][0]} with policy = {move_policies[0][1]:.6f}")
            print(f"Worst move: {move_policies[-1][0]} with policy = {move_policies[-1][1]:.6f}")
            print(f"Policy range: {move_policies[-1][1]:.6f} to {move_policies[0][1]:.6f}")
            print(f"Sum of all policy values: {sum(p for _, p in move_policies):.6f}")
        print("="*80)

print("\n" + "="*80)
print("SUMMARY - All FENs from first 10 examples:")
print("="*80)
for i, fen in enumerate(fens_list, 1):
    print(f"{i:2d}. {fen}")
print("="*80)
