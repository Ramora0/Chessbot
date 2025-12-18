"""
Dataset handler for the action-value dataset created by create_dataset.py.

Uses HuggingFace's native IterableDataset pipeline instead of wrapping in
PyTorch IterableDataset, which avoids worker sharding conflicts.

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
from datasets import load_dataset, IterableDataset as HFIterableDataset

from policy_index import policy_index
from tokenizer import process_fen


EXPECTED_SEQ_LEN = 70
NUM_VALUE_BINS = 128


def create_action_value_dataset(
    dataset_path: str | Path,
    tokenizer,
    shuffle_buffer_size: int = 100_000,
    seed: int = 42,
) -> HFIterableDataset:
    """
    Load and prepare the action-value dataset for training.

    Returns an HF IterableDataset that can be passed directly to Trainer.
    All transformations are applied lazily via .map().

    NOTE: Loads dataset in streaming mode (like MaxLeGrec/ChessFENS).
    Sharding is determined automatically by the number of data files on disk.

    Args:
        dataset_path: Path to the HuggingFace dataset directory
        tokenizer: Tokenizer for encoding FENs
        shuffle_buffer_size: Size of shuffle buffer (0 to disable)
        seed: Random seed for shuffling

    Returns:
        HF IterableDataset ready for training
    """
    # Build lookup table once (passed to map function)
    move_to_idx = {move: idx for idx, move in enumerate(policy_index)}
    policy_size = len(policy_index)

    print(f"Loading dataset from {dataset_path} in streaming mode...")

    # Load as IterableDataset directly from local path
    # This matches how MaxLeGrec/ChessFENS is loaded (streaming from the start)
    import glob
    dataset_dir = Path(dataset_path)

    # Load arrow files directly
    arrow_files = glob.glob(str(dataset_dir / "**" / "*.arrow"), recursive=True)
    if not arrow_files:
        raise ValueError(f"No arrow files found in {dataset_path}")

    print(f"Found {len(arrow_files)} arrow files")
    dataset = load_dataset(
        "arrow",
        data_files=arrow_files,
        split="train",
        streaming=True,
    )

    print(f"Loaded streaming dataset: {type(dataset).__name__}")

    # Shuffle BEFORE worker distribution
    # This ensures each worker gets shuffled data
    if shuffle_buffer_size > 0:
        print(f"Adding shuffle with buffer size {shuffle_buffer_size}")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    else:
        print("Shuffling disabled (shuffle_buffer_size=0)")

    # Apply transformation lazily - HF handles worker sharding correctly
    transform_fn = partial(
        _transform_example,
        tokenizer=tokenizer,
        move_to_idx=move_to_idx,
        policy_size=policy_size,
    )
    dataset = dataset.map(transform_fn)

    return dataset

def _transform_example(
    example: dict,
    tokenizer,
    move_to_idx: dict[str, int],
    policy_size: int,
) -> dict:
    """
    Transform a raw example to model input format.

    This is a pure function called by HF's .map() - no state, no side effects.

    Input format (from create_dataset.py):
        - fen: str
        - moves: list[str] (UCI moves)
        - p_win: list[float32] (win probabilities)

    Output format (for model):
        - input_ids: tokenized FEN (list of ints)
        - policy: value vector over policy_index
            * Illegal moves: -1
            * Legal moves: p_win - max_p_win (best move = 0, others negative)
            * Loss directly measures regret (lost win % vs optimal)
        - wdl: smooth distribution over 128 bins (win% from 0.0 to 1.0)
        - true_value: scalar win% of best move (for metrics)
        - legal_move_mask: binary mask for legal moves
    """
    fen = example["fen"]

    # 1. Tokenize FEN at runtime
    processed = process_fen(fen)
    encoding = tokenizer.encode(processed)
    input_ids = encoding.ids  # Keep as list, convert to tensor in collator

    if len(input_ids) != EXPECTED_SEQ_LEN:
        raise ValueError(
            f"Tokenized FEN has length {len(input_ids)}, expected {EXPECTED_SEQ_LEN}. FEN: {fen}"
        )

    # 2. Convert (move, p_win) pairs to policy value vector
    moves = example["moves"]
    p_wins = example["p_win"]

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

    return {
        "input_ids": input_ids,  # List of token IDs from tokenizer
        "policy": policy.tolist(),  # Lists work better with HF serialization
        "wdl": value_dist.tolist(),
        "true_value": float(best_win_prob),
        "legal_move_mask": legal_move_mask.tolist(),
    }
