"""
Dataset handler for the action-value dataset created by create_dataset.py.

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

from typing import Dict, Iterable, Iterator
from pathlib import Path

import torch
import numpy as np
import chess
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_from_disk, IterableDataset as HFIterableDataset

from policy_index import policy_index
from tokenizer import process_fen


EXPECTED_SEQ_LEN = 70
NUM_VALUE_BINS = 128


class ActionValueDataset(IterableDataset):
    """
    Loads the action-value dataset and converts it to model format at runtime.

    Input format (from create_dataset.py):
        - fen: str
        - moves: list[str] (UCI moves)
        - p_win: list[float32] (win probabilities)

    Output format (for model):
        - input_ids: tokenized FEN
        - policy: value vector over policy_index
            * Illegal moves: -1
            * Legal moves: p_win - max_p_win (best move = 0, others negative)
            * Loss directly measures regret (lost win % vs optimal)
        - wdl: smooth distribution over 128 bins (win% from 0.0 to 1.0)
        - true_value: scalar win% of best move (for metrics)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer,
        streaming: bool = True,
        shuffle_buffer_size: int = 0,
    ) -> None:
        """
        Args:
            dataset_path: Path to the HuggingFace dataset directory
            tokenizer: Tokenizer for encoding FENs
            streaming: Whether to stream the dataset (recommended for large datasets)
            shuffle_buffer_size: Size of the shuffle buffer (only used for streaming datasets)
        """
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.streaming = streaming

        # Load the dataset
        if streaming:
            self.dataset = load_from_disk(str(self.dataset_path)).to_iterable_dataset()
            # Add shuffling to prevent periodic oscillations from sorted shards
            if shuffle_buffer_size > 0:
                self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42)
        else:
            self.dataset = load_from_disk(str(self.dataset_path))

        self.policy_size = len(policy_index)

        # Create move to index mapping for fast lookup
        self.move_to_idx = {move: idx for idx, move in enumerate(policy_index)}

        # TESTING: Track seen FENs to detect duplicates
        self.seen_fens = {}  # fen -> count
        self.total_examples = 0

    @property
    def is_streaming(self) -> bool:
        return self.streaming

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("Streaming dataset has no length")
        return len(self.dataset)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.streaming:
            worker_info = get_worker_info()
            if worker_info is not None:
                shard = self.dataset.shard(worker_info.num_workers, worker_info.id)
                iterator: Iterable[Dict[str, object]] = iter(shard)
            else:
                iterator = iter(self.dataset)
        else:
            iterator = iter(self.dataset)

        for example in iterator:
            try:
                yield self._process_example(example)
            except (ValueError, KeyError) as e:
                # Skip malformed examples
                print(f"Warning: Skipping example due to error: {e}")
                continue

    def _process_example(self, example: Dict[str, object]) -> Dict[str, torch.Tensor]:
        """
        Convert raw example to model input format.

        1. Tokenize FEN
        2. Convert (move, p_win) pairs to policy value vector
           - Illegal moves: -1
           - Legal moves: p_win - max_p_win (so best move = 0)
           - Loss directly measures "lost win %" vs optimal play
        3. Convert p_win to WDL format
        """
        # 1. Tokenize FEN at runtime
        fen = example["fen"]

        # TESTING: Track duplicate FENs
        self.total_examples += 1
        if fen in self.seen_fens:
            self.seen_fens[fen] += 1
            print(f"WARNING: Duplicate FEN detected! '{fen}' seen {self.seen_fens[fen]} times (total examples processed: {self.total_examples})")
        else:
            self.seen_fens[fen] = 1

        processed = process_fen(fen)
        encoding = self.tokenizer.encode(processed)
        input_ids = torch.tensor(encoding.ids, dtype=torch.long)

        if input_ids.shape[0] != EXPECTED_SEQ_LEN:
            raise ValueError(
                f"Tokenized FEN has length {input_ids.shape[0]}, expected {EXPECTED_SEQ_LEN}. FEN: {fen}"
            )

        # 2. Convert (move, p_win) pairs to policy value vector
        moves = example["moves"]
        p_wins = example["p_win"]

        if len(moves) != len(p_wins):
            raise ValueError(
                f"Mismatch: {len(moves)} moves but {len(p_wins)} probabilities"
            )

        # Validate: Check that all legal moves in policy_index have win% values
        # board = chess.Board(fen)
        # legal_moves_uci = {move.uci() for move in board.legal_moves}
        # legal_moves_in_policy = {move for move in legal_moves_uci if move in self.move_to_idx}
        # moves_set = set(moves)

        # missing_moves = legal_moves_in_policy - moves_set
        # if missing_moves:
        #     raise ValueError(
        #         f"Dataset missing win% for legal moves!\n"
        #         f"  FEN: {fen}\n"
        #         f"  Legal moves (in policy_index): {sorted(legal_moves_in_policy)}\n"
        #         f"  Moves in dataset: {sorted(moves_set)}\n"
        #         f"  Missing moves: {sorted(missing_moves)}"
        #     )

        # Initialize policy with -1 (marking all moves as illegal)
        policy = np.full(self.policy_size, -1.0, dtype=np.float32)

        # Fill in p_win values for each legal move
        valid_moves = []
        skipped_moves = []
        for move, p_win in zip(moves, p_wins):
            if move in self.move_to_idx:
                idx = self.move_to_idx[move]
                policy[idx] = p_win
                valid_moves.append(idx)
            else:
                skipped_moves.append(move)
                # Skip moves not in policy_index (e.g., promotions to rook/bishop)

        # Debug: warn if no valid moves found
        if not valid_moves:
            print(f"Warning: No valid moves found for FEN {fen}")
            print(f"  Dataset provided {len(moves)} moves: {moves}")
            print(f"  All skipped (not in policy_index): {skipped_moves}")

        # Normalize so the best move has value 0.0 (subtract max from all legal moves)
        # This makes loss directly correspond to "lost win %" when the bot moves
        # Illegal moves stay at -1
        if valid_moves:
            max_p_win = max(policy[idx] for idx in valid_moves)
            for idx in valid_moves:
                policy[idx] = policy[idx] - max_p_win

        policy = torch.tensor(policy, dtype=torch.float32)

        # 3. Create smooth target distribution over 128 bins for win%
        # Use the maximum p_win (best move's win probability)
        if len(p_wins) > 0:
            max_p_win = max(p_wins)
        else:
            max_p_win = 0.5  # Default to 50% if no moves

        # Create smooth Gaussian-like distribution centered on max_p_win
        # Bins represent win% from 0.0 to 1.0
        bin_centers = np.linspace(0, 1, NUM_VALUE_BINS, dtype=np.float32)

        # Gaussian distribution with small std to create smooth target
        sigma = 0.05  # Smoothing factor (5% std)
        value_dist = np.exp(-0.5 * ((bin_centers - max_p_win) / sigma) ** 2)
        value_dist = value_dist / value_dist.sum()  # Normalize to sum to 1

        wdl = torch.tensor(value_dist, dtype=torch.float32)

        # Also pass the true scalar value for metric computation
        true_value = torch.tensor(max_p_win, dtype=torch.float32)

        # TESTING: Compute legal move mask from actual chess board
        board = chess.Board(fen)
        legal_move_mask = np.zeros(self.policy_size, dtype=np.float32)
        for move in board.legal_moves:
            move_uci = move.uci()
            if move_uci in self.move_to_idx:
                legal_move_mask[self.move_to_idx[move_uci]] = 1.0
        legal_move_mask = torch.tensor(legal_move_mask, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "policy": policy,
            "wdl": wdl,
            "true_value": true_value,
            "legal_move_mask": legal_move_mask,
            "fen": fen,  # TESTING: Include FEN for debugging
        }
