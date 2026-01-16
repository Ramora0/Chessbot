"""
Regret evaluation over positions split by game phase.

Computes expected regret (win% lost vs optimal) across:
- Opening: ≤10 full moves (halfmove count ≤ 20)
- Middlegame: >10 full moves AND both players have ≥13 points of material
- Endgame: both players have <13 points of material

Material values: P=1, N=3, B=3, R=5, Q=9
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

from policy_index import policy_index
from tokenizer import process_fen


# Material values for endgame detection
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Game phase thresholds
OPENING_MAX_FULLMOVES = 10  # Opening ends after 10 full moves
ENDGAME_MATERIAL_THRESHOLD = 13  # Endgame when both sides have <13 points

# Evaluation configuration
NUM_EVAL_POSITIONS = 10_000
EVAL_SEED = 12345  # Fixed seed for deterministic position selection
EVAL_BATCH_SIZE = 512


@dataclass
class PhaseRegretResults:
    """Results for regret evaluation by game phase."""
    opening_regret: float
    opening_count: int
    middlegame_regret: float
    middlegame_count: int
    endgame_regret: float
    endgame_count: int
    overall_regret: float
    total_count: int


def compute_material(board: chess.Board, color: chess.Color) -> int:
    """Compute total material value for a given color (excluding king)."""
    material = 0
    for piece_type, value in PIECE_VALUES.items():
        material += len(board.pieces(piece_type, color)) * value
    return material


def classify_game_phase(fen: str) -> str:
    """
    Classify a position into game phase: 'opening', 'middlegame', or 'endgame'.

    - Opening: ≤10 full moves
    - Endgame: both players have <13 points of material
    - Middlegame: everything else
    """
    board = chess.Board(fen)

    # Parse fullmove number from FEN (6th field)
    fullmove_number = board.fullmove_number

    # Check for opening first (based on move count)
    if fullmove_number <= OPENING_MAX_FULLMOVES:
        return "opening"

    # Check for endgame (based on material)
    white_material = compute_material(board, chess.WHITE)
    black_material = compute_material(board, chess.BLACK)

    if white_material < ENDGAME_MATERIAL_THRESHOLD and black_material < ENDGAME_MATERIAL_THRESHOLD:
        return "endgame"

    # Default to middlegame
    return "middlegame"


class RegretCollator:
    """Collator for regret evaluation that preserves FEN and policy targets."""

    def __init__(self, tokenizer, move_to_idx: Dict[str, int], policy_size: int):
        self.tokenizer = tokenizer
        self.move_to_idx = move_to_idx
        self.policy_size = policy_size

    def __call__(self, batch: List[Dict]) -> Dict[str, object]:
        input_ids_list = []
        policy_list = []
        fen_list = []

        for example in batch:
            fen = example["fen"]
            moves = example["moves"]
            p_wins = example["p_win"]

            # Tokenize FEN
            processed = process_fen(fen)
            encoding = self.tokenizer.encode(processed)
            input_ids = torch.tensor(encoding.ids, dtype=torch.long)

            # Build policy target (same as action_value_dataset.py)
            policy = np.full(self.policy_size, -1.0, dtype=np.float32)
            valid_indices = []

            for move, p_win in zip(moves, p_wins):
                if move in self.move_to_idx:
                    idx = self.move_to_idx[move]
                    policy[idx] = p_win
                    valid_indices.append(idx)

            # Normalize so best move has value 0.0
            if valid_indices:
                max_p_win = max(policy[idx] for idx in valid_indices)
                for idx in valid_indices:
                    policy[idx] -= max_p_win

            input_ids_list.append(input_ids)
            policy_list.append(torch.tensor(policy, dtype=torch.float32))
            fen_list.append(fen)

        return {
            "input_ids": torch.stack(input_ids_list),
            "policy": torch.stack(policy_list),
            "fens": fen_list,
        }


def compute_regret_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    policy_targets: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute regret for a batch of positions.

    Regret = expected win% lost from model's move distribution vs always playing best.

    Since policy targets are encoded as:
    - Best move: 0.0
    - Other legal moves: negative (win% difference from best)
    - Illegal moves: -1.0 (sentinel)

    Regret = -sum(P(move) * target[move]) for legal moves only
    """
    model.eval()

    input_ids = input_ids.to(device)
    policy_targets = policy_targets.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)

        # Use attention policy head if available
        use_old_head = os.getenv('USE_OLD_POLICY_HEAD', '0') == '1'
        if use_old_head:
            logits = outputs.policy_logits
        elif hasattr(outputs, 'attention_policy_logits') and outputs.attention_policy_logits is not None:
            logits = outputs.attention_policy_logits
        else:
            logits = outputs.policy_logits

        # Create legal move mask (policy_target > -1 means legal)
        legal_mask = policy_targets > -0.5  # Small tolerance for floating point

        # Mask illegal moves for softmax
        masked_logits = logits.masked_fill(~legal_mask, float("-inf"))

        # Get probability distribution over legal moves
        probs = torch.softmax(masked_logits, dim=-1)

        # Compute regret: -sum(P(move) * target[move]) for legal moves
        # Since targets are 0 for best move and negative for others,
        # regret = -E[target] = -sum(P * target)
        # This gives positive regret (expected win% lost)
        regret = -torch.sum(probs * policy_targets * legal_mask.float(), dim=-1)

    return regret


def evaluate_regret(
    model: torch.nn.Module,
    dataset_path: str,
    tokenizer,
    num_positions: int = NUM_EVAL_POSITIONS,
    batch_size: int = EVAL_BATCH_SIZE,
    seed: int = EVAL_SEED,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> PhaseRegretResults:
    """
    Evaluate regret over positions split by game phase.

    Args:
        model: The chess model to evaluate
        dataset_path: Path to the HuggingFace dataset
        tokenizer: Tokenizer for encoding FENs
        num_positions: Number of positions to evaluate (default: 10,000)
        batch_size: Batch size for evaluation
        seed: Random seed for deterministic position selection
        device: Device to run evaluation on
        verbose: Whether to show progress bar

    Returns:
        PhaseRegretResults with regret values per game phase
    """
    if device is None:
        device = next(model.parameters()).device

    # Load dataset and select positions deterministically
    print(f"Loading dataset from {dataset_path} for regret evaluation...")
    dataset = load_from_disk(dataset_path)

    # Shuffle with fixed seed and take first num_positions
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(num_positions, len(dataset))))

    print(f"Selected {len(dataset)} positions for regret evaluation")

    # Build move lookup
    move_to_idx = {move: idx for idx, move in enumerate(policy_index)}
    policy_size = len(policy_index)

    # Create dataloader
    collator = RegretCollator(tokenizer, move_to_idx, policy_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    # Accumulate regret by phase
    phase_regrets: Dict[str, List[float]] = {
        "opening": [],
        "middlegame": [],
        "endgame": [],
    }

    model.eval()

    dataloader_iter = tqdm(dataloader, desc="Computing regret") if verbose else dataloader

    for batch in dataloader_iter:
        input_ids = batch["input_ids"]
        policy_targets = batch["policy"]
        fens = batch["fens"]

        # Compute regret for batch
        regrets = compute_regret_batch(model, input_ids, policy_targets, device)
        regrets_cpu = regrets.cpu().numpy()

        # Classify each position and accumulate
        for i, fen in enumerate(fens):
            phase = classify_game_phase(fen)
            phase_regrets[phase].append(float(regrets_cpu[i]))

    # Compute statistics
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    opening_regrets = phase_regrets["opening"]
    middlegame_regrets = phase_regrets["middlegame"]
    endgame_regrets = phase_regrets["endgame"]
    all_regrets = opening_regrets + middlegame_regrets + endgame_regrets

    results = PhaseRegretResults(
        opening_regret=safe_mean(opening_regrets),
        opening_count=len(opening_regrets),
        middlegame_regret=safe_mean(middlegame_regrets),
        middlegame_count=len(middlegame_regrets),
        endgame_regret=safe_mean(endgame_regrets),
        endgame_count=len(endgame_regrets),
        overall_regret=safe_mean(all_regrets),
        total_count=len(all_regrets),
    )

    return results


def print_regret_results(results: PhaseRegretResults) -> None:
    """Print regret evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("REGRET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall regret:    {results.overall_regret * 100:.4f}% (n={results.total_count})")
    print("-" * 60)
    print(f"Opening regret:    {results.opening_regret * 100:.4f}% (n={results.opening_count})")
    print(f"Middlegame regret: {results.middlegame_regret * 100:.4f}% (n={results.middlegame_count})")
    print(f"Endgame regret:    {results.endgame_regret * 100:.4f}% (n={results.endgame_count})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test classification
    test_positions = [
        # Opening position (move 1)
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "opening"),
        # Early opening (move 5)
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "opening"),
        # Middlegame (full material, move 15)
        ("r2qkb1r/pp2pppp/2n2n2/3p1b2/3P1B2/2N2N2/PPP2PPP/R2QKB1R w KQkq - 0 15", "middlegame"),
        # Endgame (low material)
        ("8/5k2/8/8/3K4/8/5P2/8 w - - 0 50", "endgame"),
    ]

    print("Testing game phase classification:")
    for fen, expected in test_positions:
        result = classify_game_phase(fen)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {expected:12s} -> {result:12s}: {fen[:40]}...")
