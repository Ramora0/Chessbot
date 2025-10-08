"""Generate self-play games with Stockfish supervision for online learning."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import ChessPolicyValueModel
from policy_index import policy_index
from tokenizer import create_tokenizer, process_fen


# Stockfish configuration for evaluation
STOCKFISH_PATH = "/usr/games/stockfish"  # Update this to your Stockfish path
ENGINE_TIME_LIMIT = 0.1  # Time limit per move evaluation in seconds

# Self-play configuration
TEMPERATURE = 1.0
TOP_P = 0.9
MAX_MOVES_PER_GAME = 200


def _move_to_policy_index(move: chess.Move) -> int:
    """Convert a chess.Move to a policy index."""
    move_str = move.uci()
    try:
        return policy_index.index(move_str)
    except ValueError:
        return -1


def _get_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """Create a boolean mask for legal moves in the current position."""
    mask = torch.zeros(len(policy_index), dtype=torch.bool)
    for move in board.legal_moves:
        idx = _move_to_policy_index(move)
        if idx >= 0:
            mask[idx] = True
    return mask


def _select_move_with_sampling(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    tokenizer,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> Tuple[Optional[chess.Move], torch.Tensor]:
    """
    Use the model to select a move with temperature sampling and top-p filtering.

    Returns:
        Tuple of (selected_move, input_ids) where input_ids is the tokenized position
    """
    # Process FEN and tokenize
    fen = board.fen()
    processed = process_fen(fen)
    encoding = tokenizer.encode(processed)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
        policy_logits = outputs.policy_logits[0]

    # Mask illegal moves
    legal_mask = _get_legal_moves_mask(board).to(device)
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

    # Apply temperature
    logits_with_temp = masked_logits / temperature

    # Apply top-p (nucleus) sampling
    probs = F.softmax(logits_with_temp, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[0] = False

    # Create a mask for indices to remove
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
    indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True

    # Set removed indices to very low probability
    filtered_probs = probs.masked_fill(indices_to_remove, 0.0)

    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum()

    # Sample from the filtered distribution
    try:
        move_idx = torch.multinomial(filtered_probs, num_samples=1).item()
    except RuntimeError:
        # Fallback to greedy if sampling fails
        move_idx = torch.argmax(filtered_probs).item()

    # Convert index to move
    move_uci = policy_index[move_idx]
    try:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            # Return the move and input_ids (without batch dim)
            return move, input_ids[0]
        else:
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            return (legal_moves[0] if legal_moves else None), input_ids[0]
    except ValueError:
        # If the move is invalid, return a random legal move
        legal_moves = list(board.legal_moves)
        return (legal_moves[0] if legal_moves else None), input_ids[0]


async def _get_stockfish_evals(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    time_limit: float = ENGINE_TIME_LIMIT,
) -> dict[str, float]:
    """
    Get Stockfish centipawn evaluations for all legal moves using MultiPV.

    Returns:
        Dictionary mapping move UCI strings to centipawn evaluations.
        Positive values favor the side to move.
    """
    evals = {}

    # Count legal moves
    num_legal_moves = board.legal_moves.count()

    if num_legal_moves == 0:
        return evals

    # Analyze the position with MultiPV
    info = await engine.analyse(
        board,
        chess.engine.Limit(time=time_limit),
        multipv=num_legal_moves,
    )

    # Extract scores for each move
    for pv_info in info:
        pv = pv_info.get("pv")
        if pv and len(pv) > 0:
            move = pv[0]  # First move in the principal variation
            score = pv_info.get("score")

            if score is not None:
                # Get score from perspective of side to move
                cp_score = score.white().score(mate_score=10000)

                # Adjust for side to move
                if board.turn == chess.BLACK:
                    cp_score = -cp_score if cp_score is not None else None

                if cp_score is not None:
                    evals[move.uci()] = cp_score
                else:
                    # Mate score
                    mate_value = score.white().mate()
                    if mate_value is not None:
                        if board.turn == chess.BLACK:
                            mate_value = -mate_value
                        evals[move.uci()] = 10000 if mate_value > 0 else -10000
                    else:
                        evals[move.uci()] = 0.0
            else:
                evals[move.uci()] = 0.0

    return evals


def _centipawn_to_policy(evals: dict[str, float], temperature: float = 1.0) -> torch.Tensor:
    """
    Convert centipawn evaluations to a policy distribution.

    Uses softmax over centipawn values to create a probability distribution.
    Higher centipawn = higher probability.

    Args:
        evals: Dictionary mapping move UCI to centipawn evaluation
        temperature: Temperature for softmax (default 1.0)

    Returns:
        Tensor of shape (len(policy_index),) with probabilities for each move.
        Illegal moves get -1.
    """
    policy = torch.full((len(policy_index),), -1.0, dtype=torch.float32)

    if not evals:
        return policy

    # Create tensor of evaluations in policy_index order
    move_evals = []
    move_indices = []
    for move_uci, cp in evals.items():
        idx = policy_index.index(move_uci) if move_uci in policy_index else -1
        if idx >= 0:
            move_evals.append(cp)
            move_indices.append(idx)

    if not move_evals:
        return policy

    # Convert to tensor and apply softmax
    eval_tensor = torch.tensor(move_evals, dtype=torch.float32)

    # Normalize centipawns to reasonable range (divide by 100 to convert to pawns)
    # Then apply temperature
    eval_tensor = eval_tensor / (100.0 * temperature)

    # Apply softmax to get probabilities
    probs = F.softmax(eval_tensor, dim=-1)

    # Fill in the policy tensor
    for idx, prob in zip(move_indices, probs):
        policy[idx] = prob.item()

    return policy


@dataclass
class GamePosition:
    """Stores a single position from a game with its Stockfish supervision."""
    input_ids: torch.Tensor  # Tokenized position, shape (seq_len,)
    # Policy from Stockfish evals, shape (len(policy_index),)
    stockfish_policy: torch.Tensor
    move_played: str  # UCI of the move that was played


@dataclass
class GameData:
    """Stores all positions from a single game."""
    positions: List[GamePosition]
    result: str  # 'white_win', 'black_win', 'draw', 'incomplete'
    num_moves: int


async def play_game_with_supervision(
    model: torch.nn.Module,
    stockfish_path: str,
    device: torch.device,
    tokenizer,
    game_id: int,
    max_moves: int = MAX_MOVES_PER_GAME,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    verbose: bool = True,
) -> GameData:
    """
    Play a self-play game while collecting Stockfish evaluations.

    Returns:
        GameData containing all positions with Stockfish supervision.
    """
    model.eval()
    board = chess.Board()

    # Initialize async Stockfish engine
    transport, engine = await chess.engine.popen_uci(stockfish_path)

    positions = []
    move_count = 0

    try:
        with tqdm(desc=f"Game {game_id}", total=max_moves, leave=False, disable=not verbose) as pbar:
            while not board.is_game_over() and move_count < max_moves:
                # Get Stockfish evaluations for all legal moves
                stockfish_evals = await _get_stockfish_evals(
                    engine, board, time_limit=ENGINE_TIME_LIMIT
                )

                # Convert to policy distribution
                stockfish_policy = _centipawn_to_policy(stockfish_evals)

                # Get model's move (with sampling)
                move, input_ids = _select_move_with_sampling(
                    model, board, device, tokenizer, temperature, top_p
                )

                if move is None or move not in board.legal_moves:
                    # Model made an illegal move - end game
                    break

                # Store this position
                position = GamePosition(
                    input_ids=input_ids.cpu(),
                    stockfish_policy=stockfish_policy,
                    move_played=move.uci(),
                )
                positions.append(position)

                # Make the move
                board.push(move)
                move_count += 1
                pbar.update(1)

        # Determine result
        outcome = board.outcome()
        if outcome is None:
            result = "incomplete"
        elif outcome.winner is None:
            result = "draw"
        elif outcome.winner == chess.WHITE:
            result = "white_win"
        else:
            result = "black_win"

        return GameData(
            positions=positions,
            result=result,
            num_moves=move_count,
        )

    finally:
        await engine.quit()


async def generate_games(
    model: torch.nn.Module,
    stockfish_path: str,
    num_games: int,
    device: torch.device,
    tokenizer,
    batch_size: int = 1,  # Number of games to run in parallel
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    verbose: bool = True,
) -> List[GameData]:
    """
    Generate multiple self-play games with Stockfish supervision.

    Args:
        model: The chess model to use for self-play
        stockfish_path: Path to Stockfish executable
        num_games: Number of games to generate
        device: Device to run model on
        tokenizer: Tokenizer for encoding positions
        batch_size: Number of games to run in parallel
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling threshold
        verbose: Whether to print progress

    Returns:
        List of GameData objects containing all positions and supervision
    """
    all_games = []

    if verbose:
        print(
            f"Generating {num_games} self-play games with Stockfish supervision...")
        print(f"  Temperature: {temperature}")
        print(f"  Top-p: {top_p}")
        print(f"  Parallel games: {batch_size}")
        print()

    # Generate games in batches
    for batch_start in range(0, num_games, batch_size):
        batch_end = min(batch_start + batch_size, num_games)
        batch_game_ids = range(batch_start, batch_end)

        # Run games in parallel
        tasks = [
            play_game_with_supervision(
                model=model,
                stockfish_path=stockfish_path,
                device=device,
                tokenizer=tokenizer,
                game_id=game_id,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )
            for game_id in batch_game_ids
        ]

        batch_games = await asyncio.gather(*tasks)
        all_games.extend(batch_games)

        if verbose:
            print(f"Completed {len(all_games)}/{num_games} games")

    if verbose:
        total_positions = sum(len(game.positions) for game in all_games)
        avg_moves = sum(game.num_moves for game in all_games) / len(all_games)
        print(
            f"\nGenerated {total_positions} positions from {len(all_games)} games")
        print(f"Average game length: {avg_moves:.1f} moves")

        # Print result distribution
        results = [game.result for game in all_games]
        white_wins = results.count('white_win')
        black_wins = results.count('black_win')
        draws = results.count('draw')
        incomplete = results.count('incomplete')
        print(
            f"Results: {white_wins}W - {draws}D - {black_wins}B ({incomplete} incomplete)")

    return all_games


def main():
    """Main entry point for generating games."""

    # Configuration
    CHECKPOINT_PATH = "./outputs/checkpoint-90000"
    STOCKFISH_PATH_MAIN = "/users/PAS2836/leedavis/stockfish/src/stockfish"
    NUM_GAMES = 100
    BATCH_SIZE = 4
    TEMPERATURE_MAIN = 1.0
    TOP_P_MAIN = 0.9

    # Load model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = ChessPolicyValueModel.from_pretrained_compiled(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # Create tokenizer
    tokenizer = create_tokenizer()

    # Generate games
    games = asyncio.run(
        generate_games(
            model=model,
            stockfish_path=STOCKFISH_PATH_MAIN,
            num_games=NUM_GAMES,
            device=device,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE_MAIN,
            top_p=TOP_P_MAIN,
        )
    )

    print(f"\nDone! Generated {len(games)} games")


if __name__ == "__main__":
    main()
