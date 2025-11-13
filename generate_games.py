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
ENGINE_TIME_LIMIT = 0.001  # Time limit per move evaluation in seconds

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
        multipv=3,
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


def _evals_to_policy(evals: dict[str, float]) -> torch.Tensor:
    """
    Convert centipawn evaluations to a policy tensor.

    Args:
        evals: Dictionary mapping move UCI to centipawn evaluation

    Returns:
        Tensor of shape (len(policy_index),) with centipawn values for legal moves.
        Illegal moves get -inf.
    """
    policy = torch.full((len(policy_index),),
                        float('-inf'), dtype=torch.float32)

    if not evals:
        return policy

    # Fill in centipawn values for legal moves
    for move_uci, cp in evals.items():
        idx = policy_index.index(move_uci) if move_uci in policy_index else -1
        if idx >= 0:
            policy[idx] = cp

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


@dataclass
class GameState:
    """Manages the state of an ongoing game during batched generation."""
    game_id: int
    board: chess.Board
    positions: List[GamePosition]
    move_count: int
    max_moves: int
    is_complete: bool
    result: Optional[str]

    def __init__(self, game_id: int, max_moves: int = MAX_MOVES_PER_GAME):
        self.game_id = game_id
        self.board = chess.Board()
        self.positions = []
        self.move_count = 0
        self.max_moves = max_moves
        self.is_complete = False
        self.result = None

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.is_complete or self.board.is_game_over() or self.move_count >= self.max_moves

    def get_result(self) -> str:
        """Get the final result of the game."""
        if self.result:
            return self.result

        outcome = self.board.outcome()
        if outcome is None:
            self.result = "incomplete"
        elif outcome.winner is None:
            self.result = "draw"
        elif outcome.winner == chess.WHITE:
            self.result = "white_win"
        else:
            self.result = "black_win"

        return self.result

    def to_game_data(self) -> GameData:
        """Convert to GameData for final output."""
        return GameData(
            positions=self.positions,
            result=self.get_result(),
            num_moves=self.move_count,
        )


def _select_moves_from_model_batch(
    model: torch.nn.Module,
    boards: List[chess.Board],
    device: torch.device,
    tokenizer,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> List[Tuple[Optional[chess.Move], torch.Tensor]]:
    """
    Use the model to select moves for a batch of positions with sampling.

    Returns:
        List of (selected_move, input_ids) tuples.
    """
    if not boards:
        return []

    # Process all FENs and tokenize
    fens = [board.fen() for board in boards]
    processed = [process_fen(fen) for fen in fens]
    encodings = [tokenizer.encode(p) for p in processed]
    input_ids = torch.tensor(
        [enc.ids for enc in encodings], dtype=torch.long, device=device)

    # Get model predictions for entire batch
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
        policy_logits = outputs.policy_logits

    # Process each position in the batch
    results = []
    for i, board in enumerate(boards):
        # Mask illegal moves
        legal_mask = _get_legal_moves_mask(board).to(device)

        # Check if there are any legal moves
        if not legal_mask.any():
            print(f"WARNING: No legal moves for board {i}")
            results.append((None, input_ids[i].cpu()))
            continue

        masked_logits = policy_logits[i].masked_fill(
            ~legal_mask, float("-inf"))

        # Apply temperature
        logits_with_temp = masked_logits / temperature

        # Apply top-p (nucleus) sampling
        probs = F.softmax(logits_with_temp, dim=-1)

        # Check for NaN in probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"WARNING: NaN/Inf in probs for board {i}")
            print(f"  Legal mask sum: {legal_mask.sum()}")
            print(
                f"  Masked logits min/max: {masked_logits[legal_mask].min()}/{masked_logits[legal_mask].max()}")
            print(f"  Probs: {probs}")
            # Fallback to uniform distribution over legal moves
            uniform_probs = torch.zeros_like(probs)
            uniform_probs[legal_mask] = 1.0 / legal_mask.sum()
            probs = uniform_probs
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False

        # Create a mask for indices to remove
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True

        # Set removed indices to very low probability
        filtered_probs = probs.masked_fill(indices_to_remove, 0.0)

        # Renormalize (check for zero sum to avoid NaN)
        prob_sum = filtered_probs.sum()
        if prob_sum > 0:
            filtered_probs = filtered_probs / prob_sum
        else:
            # Fallback: use original probabilities if filtering removed everything
            print(
                f"WARNING: Top-p filtering removed all moves for board {i}, using original probs")
            print(f"  Filtered probs: {filtered_probs}")
            print(f"  Original probs: {probs}")
            filtered_probs = probs / probs.sum()

        # Sample from the filtered distribution
        try:
            move_idx = torch.multinomial(filtered_probs, num_samples=1).item()
        except RuntimeError as e:
            # Fallback to greedy if sampling still fails
            print(
                f"WARNING: Multinomial sampling failed for board {i}: {e}, using greedy")
            print(f"  Problematic probs: {filtered_probs}")
            print(f"  Contains NaN: {torch.isnan(filtered_probs).any()}")
            print(f"  Contains Inf: {torch.isinf(filtered_probs).any()}")
            print(f"  Contains negative: {(filtered_probs < 0).any()}")
            print(f"  Sum: {filtered_probs.sum()}")
            move_idx = torch.argmax(probs).item()

        # Convert index to move
        move_uci = policy_index[move_idx]
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                results.append((move, input_ids[i].cpu()))
            else:
                # Fallback to random legal move
                legal_moves = list(board.legal_moves)
                results.append(
                    (legal_moves[0] if legal_moves else None, input_ids[i].cpu()))
        except ValueError:
            # If the move is invalid, return a random legal move
            legal_moves = list(board.legal_moves)
            results.append(
                (legal_moves[0] if legal_moves else None, input_ids[i].cpu()))

    return results


async def generate_games(
    model: torch.nn.Module,
    stockfish_path: str,
    num_games: int,
    device: torch.device,
    tokenizer,
    batch_size: int = 32,  # Number of games to run in parallel
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    verbose: bool = True,
) -> Tuple[List[GameData], List[chess.Board]]:
    """
    Generate multiple self-play games with Stockfish supervision using batched inference.

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
        Tuple of (completed_games, incomplete_boards)
        - completed_games: List of GameData objects for finished games
        - incomplete_boards: List of board states for games that didn't finish
    """
    model.eval()

    if verbose:
        print(
            f"Generating up to {num_games} self-play games with Stockfish supervision...")
        print(f"  Temperature: {temperature}")
        print(f"  Top-p: {top_p}")
        print(f"  Batch size: {batch_size}")
        print()

    # Initialize engines pool
    async def init_engine():
        transport, engine = await chess.engine.popen_uci(stockfish_path)
        return (transport, engine)

    num_engines = min(batch_size, num_games)

    if verbose:
        print(f"Initializing {num_engines} Stockfish engines...")
    engines = await asyncio.gather(*[init_engine() for _ in range(num_engines)])
    if verbose:
        print("Engines initialized. Starting games...\n")

    # Create game states - use dict instead of list to handle any game_id
    game_states = {i: GameState(i) for i in range(batch_size)}

    # Track which games are using which engines
    game_to_engine = {i: i for i in range(batch_size)}
    pending_game_ids = list(range(batch_size, num_games))

    completed_games = []
    total_moves = 0

    # Timing metrics
    import time
    total_stockfish_time = 0.0
    total_model_time = 0.0
    num_iterations = 0

    try:
        with tqdm(desc="Generating games", unit=" moves", disable=not verbose) as pbar:
            # Track previous iteration's Stockfish results for pipelining
            prev_stockfish_policies = None
            prev_active_games = None

            while game_to_engine:
                # Get active games that aren't over yet
                active_game_ids = list(game_to_engine.keys())
                active_games = [game_states[gid]
                                for gid in active_game_ids if not game_states[gid].is_game_over()]

                # If no active games left, break
                if not active_games:
                    break

                # Launch Stockfish evaluations for current positions (async, non-blocking)
                stockfish_start = time.time()
                stockfish_tasks = [
                    _get_stockfish_evals(
                        engines[game_to_engine[game.game_id]][1],
                        game.board,
                        time_limit=ENGINE_TIME_LIMIT
                    )
                    for game in active_games
                ]
                stockfish_future = asyncio.gather(*stockfish_tasks)

                # While Stockfish runs on CPU, do model inference on GPU for PREVIOUS iteration
                model_start = time.time()
                if prev_stockfish_policies is not None:
                    # Model inference happens synchronously on GPU while Stockfish runs on CPU
                    boards = [game.board for game in prev_active_games]
                    model_results = _select_moves_from_model_batch(
                        model, boards, device, tokenizer, temperature, top_p
                    )
                    total_model_time += time.time() - model_start

                    # Apply moves and update game states from previous iteration
                    for game, (move, input_ids), stockfish_policy in zip(
                        prev_active_games, model_results, prev_stockfish_policies
                    ):
                        if move is None or move not in game.board.legal_moves:
                            # Model made an illegal move - end game
                            game.is_complete = True
                            game.result = "incomplete"
                        else:
                            # Store this position
                            position = GamePosition(
                                input_ids=input_ids,
                                stockfish_policy=stockfish_policy,
                                move_played=move.uci(),
                            )
                            game.positions.append(position)

                            # Make the move
                            game.board.push(move)
                            game.move_count += 1
                            total_moves += 1
                            pbar.update(1)

                    # Check for completed games
                    for game_id in list(game_to_engine.keys()):
                        game = game_states[game_id]
                        if game.is_game_over():
                            # Game is complete
                            completed_games.append(game.to_game_data())

                            # Free up the engine
                            engine_idx = game_to_engine.pop(game_id)

                            # If we have pending games, assign new game to keep batch full
                            if pending_game_ids:
                                new_game_id = pending_game_ids.pop(0)
                                game_states[new_game_id] = GameState(
                                    new_game_id)
                                game_to_engine[new_game_id] = engine_idx
                            # Otherwise, we're winding down - don't refill

                    # Update progress description
                    if verbose and completed_games:
                        pbar.set_description(
                            f"Completed {len(completed_games)}/{num_games} games | Active: {len(game_to_engine)}"
                        )

                # Wait for current Stockfish evaluations to complete
                all_stockfish_evals = await stockfish_future
                total_stockfish_time += time.time() - stockfish_start

                # Convert to policy tensors (centipawn values)
                stockfish_policies = [
                    _evals_to_policy(evals) for evals in all_stockfish_evals
                ]

                # Save for next iteration
                prev_stockfish_policies = stockfish_policies
                prev_active_games = active_games
                num_iterations += 1

            # Handle final iteration (process last batch of Stockfish results)
            if prev_stockfish_policies is not None:
                boards = [game.board for game in prev_active_games]
                model_results = _select_moves_from_model_batch(
                    model, boards, device, tokenizer, temperature, top_p
                )

                for game, (move, input_ids), stockfish_policy in zip(
                    prev_active_games, model_results, prev_stockfish_policies
                ):
                    if move is None or move not in game.board.legal_moves:
                        game.is_complete = True
                        game.result = "incomplete"
                    else:
                        position = GamePosition(
                            input_ids=input_ids,
                            stockfish_policy=stockfish_policy,
                            move_played=move.uci(),
                        )
                        game.positions.append(position)
                        game.board.push(move)
                        game.move_count += 1
                        total_moves += 1

                # Mark completed games from final iteration
                for game in prev_active_games:
                    if game.is_game_over():
                        completed_games.append(game.to_game_data())

    finally:
        # Clean up engines
        for _, engine in engines:
            await engine.quit()

    # Get incomplete boards
    incomplete_boards = [
        game.board for game in game_states.values()
        if game.move_count > 0 and not game.is_complete
    ]

    if verbose:
        print(f"\nCompleted {len(completed_games)} games")
        print(f"Incomplete games: {len(incomplete_boards)}")

        if completed_games:
            total_positions = sum(len(game.positions)
                                  for game in completed_games)
            avg_moves = sum(
                game.num_moves for game in completed_games) / len(completed_games)
            print(f"Total positions: {total_positions}")
            print(f"Average game length: {avg_moves:.1f} moves")

            # Print result distribution
            results = [game.result for game in completed_games]
            white_wins = results.count('white_win')
            black_wins = results.count('black_win')
            draws = results.count('draw')
            incomplete = results.count('incomplete')
            print(
                f"Results: {white_wins}W - {draws}D - {black_wins}B ({incomplete} incomplete)")

        # Print timing metrics
        if num_iterations > 0:
            print(f"\nTiming metrics ({num_iterations} iterations):")
            print(
                f"  Total Stockfish time: {total_stockfish_time:.2f}s ({total_stockfish_time/num_iterations:.3f}s/iter)")
            print(
                f"  Total Model time: {total_model_time:.2f}s ({total_model_time/num_iterations:.3f}s/iter)")
            print(
                f"  Ratio (SF/Model): {total_stockfish_time/total_model_time:.2f}x" if total_model_time > 0 else "  Ratio: N/A")
            if total_stockfish_time > total_model_time:
                print(
                    f"  → Stockfish is the bottleneck ({total_stockfish_time - total_model_time:.2f}s wasted waiting)")
            else:
                print(
                    f"  → Model is the bottleneck ({total_model_time - total_stockfish_time:.2f}s wasted waiting)")

    return completed_games, incomplete_boards


def main():
    """Main entry point for generating games."""

    # Configuration
    CHECKPOINT_PATH = "./outputs/checkpoint-90000"
    STOCKFISH_PATH_MAIN = "/users/PAS2836/leedavis/stockfish/src/stockfish"
    NUM_GAMES = 100
    BATCH_SIZE = 32
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
    games, incomplete_boards = asyncio.run(
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

    print(
        f"\nDone! Generated {len(games)} complete games, {len(incomplete_boards)} incomplete")


if __name__ == "__main__":
    main()
