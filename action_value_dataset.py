"""
Dataset handler for the action-value dataset created by create_dataset.py.

Uses HuggingFace's Dataset with on-the-fly transformations via set_transform().
This allows parallel processing across dataloader workers without upfront
preprocessing costs.

This dataset format stores:
- fen: FEN string
- moves: list of UCI move strings
- p_win: list of win probabilities for each move

This handler converts to the format expected by the model:
- Tokenizes FEN at runtime
- Converts (move, p_win) pairs to policy distribution using policy_index
- Converts p_win to WDL format
"""

from __future__ import annotations

from pathlib import Path
from functools import partial

import numpy as np
import chess
from datasets import load_from_disk, Dataset

from policy_index import policy_index
from tokenizer import process_fen


EXPECTED_SEQ_LEN = 70
NUM_VALUE_BINS = 128


def create_action_value_dataset(
    dataset_path: str | Path,
    tokenizer,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Load and prepare the action-value dataset for training.

    Returns an HF Dataset that can be passed directly to Trainer.
    All transformations are applied on-the-fly via set_transform(),
    allowing parallel processing across dataloader workers.

    Args:
        dataset_path: Path to the HuggingFace dataset directory
        tokenizer: Tokenizer for encoding FENs
        shuffle: Whether to shuffle the dataset (full random shuffle)
        seed: Random seed for shuffling

    Returns:
        HF Dataset ready for training
    """
    # Build lookup table once (passed to transform function)
    move_to_idx = {move: idx for idx, move in enumerate(policy_index)}
    policy_size = len(policy_index)

    print(f"Loading dataset from {dataset_path}...")

    # Load dataset (uses memory-mapped Arrow files - data stays on disk)
    dataset = load_from_disk(dataset_path)

    print(f"Loaded dataset with {len(dataset):,} examples")

    # Shuffle if requested (full random shuffle, not buffer-limited)
    if shuffle:
        print(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Apply transformation on-the-fly via set_transform
    # This is lazy - transforms happen in dataloader workers during iteration
    # No upfront processing cost, no cache files written
    transform_fn = partial(
        _transform_example,
        tokenizer=tokenizer,
        move_to_idx=move_to_idx,
        policy_size=policy_size,
    )
    dataset.set_transform(transform_fn)

    print(f"Dataset ready with on-the-fly transforms")

    return dataset


def _transform_example(
    examples: dict,
    tokenizer,
    move_to_idx: dict[str, int],
    policy_size: int,
) -> dict:
    """
    Transform a batch of examples to model input format.

    This is a pure function called by HF's set_transform() - no state, no side effects.
    Processes batches for efficiency with dataloader workers.

    Input format (from create_dataset.py):
        - fen: list[str]
        - moves: list[list[str]] (UCI moves)
        - p_win: list[list[float32]] (win probabilities)

    Output format (for model):
        - input_ids: list of tokenized FENs (list of list of ints)
        - policy: list of value vectors over policy_index
            * Illegal moves: -1
            * Legal moves: p_win - max_p_win (best move = 0, others negative)
            * Loss directly measures regret (lost win % vs optimal)
        - wdl: list of smooth distributions over 128 bins (win% from 0.0 to 1.0)
        - true_value: list of scalar win% of best move (for metrics)
        - legal_move_mask: list of binary masks for legal moves
    """
    batch_size = len(examples["fen"])

    # Initialize output lists
    all_input_ids = []
    all_policy = []
    all_wdl = []
    all_true_value = []
    all_legal_move_mask = []

    # Process each example in the batch
    for i in range(batch_size):
        fen = examples["fen"][i]
        moves = examples["moves"][i]
        p_wins = examples["p_win"][i]

        # 1. Tokenize FEN at runtime
        processed = process_fen(fen)
        encoding = tokenizer.encode(processed)
        input_ids = encoding.ids  # Keep as list, convert to tensor in collator

        if len(input_ids) != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Tokenized FEN has length {len(input_ids)}, expected {EXPECTED_SEQ_LEN}. FEN: {fen}"
            )

        # 2. Convert (move, p_win) pairs to policy value vector
        if len(moves) != len(p_wins):
            raise ValueError(
                f"Mismatch: {len(moves)} moves but {len(p_wins)} probabilities"
            )

        # Initialize policy with -1 (marking all moves as illegal)
        policy = np.full(policy_size, -1.0, dtype=np.float32)

        # Fill in p_win values for each legal move
        valid_indices = []
        for move, p_win in zip(moves, p_wins):
            if move in move_to_idx:
                idx = move_to_idx[move]
                policy[idx] = p_win
                valid_indices.append(idx)
            # Skip moves not in policy_index (e.g., promotions to rook/bishop)

        # Normalize so the best move has value 0.0 (subtract max from all legal moves)
        # This makes loss directly correspond to "win% lost" vs optimal play
        # Illegal moves stay at -1
        if valid_indices:
            max_p_win = max(policy[idx] for idx in valid_indices)
            for idx in valid_indices:
                policy[idx] -= max_p_win

        # 3. Create smooth target distribution over 128 bins for win%
        # Use the maximum p_win (best move's win probability)
        best_win_prob = max(p_wins) if p_wins else 0.5

        # Create smooth Gaussian-like distribution centered on best_win_prob
        # Bins represent win% from 0.0 to 1.0
        bin_centers = np.linspace(0, 1, NUM_VALUE_BINS, dtype=np.float32)
        sigma = 0.05  # Smoothing factor (5% std)
        value_dist = np.exp(-0.5 * ((bin_centers - best_win_prob) / sigma) ** 2)
        value_dist = value_dist / value_dist.sum()  # Normalize to sum to 1

        # 4. Compute legal move mask from actual chess board
        board = chess.Board(fen)
        legal_move_mask = np.zeros(policy_size, dtype=np.float32)
        for move in board.legal_moves:
            move_uci = move.uci()
            if move_uci in move_to_idx:
                legal_move_mask[move_to_idx[move_uci]] = 1.0

        # Add to batch outputs
        all_input_ids.append(input_ids)
        all_policy.append(policy.tolist())
        all_wdl.append(value_dist.tolist())
        all_true_value.append(float(best_win_prob))
        all_legal_move_mask.append(legal_move_mask.tolist())

    return {
        "input_ids": all_input_ids,
        "policy": all_policy,
        "wdl": all_wdl,
        "true_value": all_true_value,
        "legal_move_mask": all_legal_move_mask,
    }
