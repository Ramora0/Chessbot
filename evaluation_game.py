"""Utility to evaluate a chess model by playing games against Stockfish."""

from __future__ import annotations

import asyncio
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from policy_index import policy_index
from tokenizer import process_fen


# Stockfish configuration
STOCKFISH_ELO = 1350
STOCKFISH_SKILL_LEVEL = 0
ENGINE_TIME_LIMIT = 1  # Time limit in seconds for Stockfish moves

# Batching configuration
DEFAULT_BATCH_SIZE = 32  # Number of games to run in parallel

# ELO calculation constants
LOG10 = float(np.log(10.0))
ELO_K = LOG10 / 400.0


def load_puzzle_positions(csv_path: str | Path, num_positions: int) -> List[Tuple[str, bool]]:
    """Load random FEN positions from puzzles.csv.

    Returns pairs of the same position for fairness (model plays both sides).
    Each element is (fen, model_plays_white).
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        all_fens = [row[4] for row in reader]  # FEN is column 5 (index 4)

    # Sample half the positions (we'll play each from both sides)
    num_unique = (num_positions + 1) // 2
    selected_fens = random.sample(all_fens, min(num_unique, len(all_fens)))

    # Create pairs - same position, different colors
    positions = []
    for fen in selected_fens:
        positions.append((fen, True))   # Model plays white
        positions.append((fen, False))  # Model plays black

    # Shuffle the pairs and trim to exact count
    random.shuffle(positions)
    return positions[:num_positions]


def _move_to_policy_index(move: chess.Move) -> int:
    """Convert a chess.Move to a policy index."""
    move_str = move.uci()
    try:
        return policy_index.index(move_str)
    except ValueError:
        # if move_str[-1] == "n":
        return -1
        # print(f"Warning: Move {move_str} not found in policy index.")
        # raise ValueError(f"Move {move_str} not found in policy index.")


def _get_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """Create a boolean mask for legal moves in the current position."""
    mask = torch.zeros(len(policy_index), dtype=torch.bool)
    for move in board.legal_moves:
        idx = _move_to_policy_index(move)
        if idx >= 0:
            mask[idx] = True
    return mask


def _select_moves_from_model_batch(
    model: torch.nn.Module,
    boards: List[chess.Board],
    device: torch.device,
    tokenizer,
) -> List[Optional[chess.Move]]:
    """Use the model to select moves for a batch of positions."""
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
    moves = []
    for i, board in enumerate(boards):
        # Mask illegal moves
        legal_mask = _get_legal_moves_mask(board).to(device)
        masked_logits = policy_logits[i].masked_fill(
            ~legal_mask, float("-inf"))

        # Select move with highest probability
        probs = F.softmax(masked_logits, dim=-1)
        move_idx = torch.argmax(probs).item()

        # Convert index to move
        move_uci = policy_index[move_idx]
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                moves.append(move)
            else:
                # Fallback to random legal move
                legal_moves = list(board.legal_moves)
                moves.append(legal_moves[0] if legal_moves else None)
        except ValueError:
            # If the move is invalid, return a random legal move
            legal_moves = list(board.legal_moves)
            moves.append(legal_moves[0] if legal_moves else None)

    return moves


def _select_move_from_model(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    tokenizer,
) -> Optional[chess.Move]:
    """Use the model to select a move from the current position (single position)."""
    return _select_moves_from_model_batch(model, [board], device, tokenizer)[0]


def play_game(
    model: torch.nn.Module,
    stockfish_path: str,
    device: Optional[torch.device] = None,
    tokenizer=None,
    model_plays_white: bool = True,
    max_moves: int = 200,
) -> Tuple[str, int]:
    """
    Play a single game between the model and Stockfish.

    Returns:
        Tuple of (result, num_moves) where result is 'win', 'draw', or 'loss'
        from the model's perspective.
    """
    if device is None:
        device = next(model.parameters()).device

    if tokenizer is None:
        from tokenizer import create_tokenizer
        tokenizer = create_tokenizer()

    model.eval()
    board = chess.Board()

    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})

    move_count = 0

    try:
        while not board.is_game_over() and move_count < max_moves:
            if (board.turn == chess.WHITE) == model_plays_white:
                # Model's turn
                move = _select_move_from_model(model, board, device, tokenizer)
                if move is None or move not in board.legal_moves:
                    # Model made an illegal move - count as loss
                    return ("loss", move_count)
                board.push(move)
            else:
                # Stockfish's turn
                result = engine.play(
                    board, chess.engine.Limit(time=ENGINE_TIME_LIMIT))
                board.push(result.move)

            move_count += 1

        # Determine result
        outcome = board.outcome()
        if outcome is None:
            # Game reached max moves without conclusion
            return ("draw", move_count)

        if outcome.winner is None:
            return ("draw", move_count)
        elif (outcome.winner == chess.WHITE) == model_plays_white:
            return ("win", move_count)
        else:
            return ("loss", move_count)

    finally:
        engine.quit()


def estimate_elo_from_scores(
    wins: int,
    draws: int,
    losses: int,
    opponent_elo: float = STOCKFISH_ELO,
) -> Tuple[float, float]:
    """
    Estimate ELO rating based on game results against an opponent.

    Uses the standard ELO formula: expected_score = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
    Solves for player_elo given the actual score.

    Returns:
        Tuple of (estimated_elo, standard_error)
    """
    total_games = wins + draws + losses
    if total_games == 0:
        return (float("nan"), float("nan"))

    # Calculate actual score (1 for win, 0.5 for draw, 0 for loss)
    actual_score = (wins + 0.5 * draws) / total_games

    # Clamp score to avoid numerical issues
    actual_score = max(0.001, min(0.999, actual_score))

    # Solve for ELO: actual_score = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
    # Rearranging: 10^((opponent_elo - player_elo) / 400) = (1 - actual_score) / actual_score
    # (opponent_elo - player_elo) / 400 = log10((1 - actual_score) / actual_score)
    # player_elo = opponent_elo - 400 * log10((1 - actual_score) / actual_score)

    estimated_elo = opponent_elo - 400 * \
        math.log10((1 - actual_score) / actual_score)

    # Calculate standard error using binomial approximation
    # SE = 400 / (sqrt(n) * ln(10) * p * (1-p)) where p is expected score at true ELO
    # For simplicity, use actual_score as approximation
    variance_per_game = actual_score * (1 - actual_score)
    if variance_per_game > 0:
        # Standard error in ELO space
        standard_error = 400 / (math.sqrt(total_games)
                                * math.log(10) * math.sqrt(variance_per_game))
    else:
        standard_error = float("inf")

    return (estimated_elo, standard_error)


async def _play_game_async(
    model: torch.nn.Module,
    stockfish_path: str,
    device: torch.device,
    tokenizer,
    model_plays_white: bool,
    game_id: int,
    max_moves: int = 200,
) -> Tuple[str, int, int]:
    """
    Play a single game asynchronously. Returns (result, num_moves, game_id).
    """
    model.eval()
    board = chess.Board()

    # Initialize async Stockfish engine
    transport, engine = await chess.engine.popen_uci(stockfish_path)
    await engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})

    move_count = 0

    try:
        while not board.is_game_over() and move_count < max_moves:
            if (board.turn == chess.WHITE) == model_plays_white:
                # Model's turn - will be batched externally
                # For now, do single inference (batching happens at higher level)
                move = _select_move_from_model(model, board, device, tokenizer)
                if move is None or move not in board.legal_moves:
                    return ("loss", move_count, game_id)
                board.push(move)
            else:
                # Stockfish's turn
                result = await engine.play(board, chess.engine.Limit(time=ENGINE_TIME_LIMIT))
                board.push(result.move)

            move_count += 1

        # Determine result
        outcome = board.outcome()
        if outcome is None:
            return ("draw", move_count, game_id)

        if outcome.winner is None:
            return ("draw", move_count, game_id)
        elif (outcome.winner == chess.WHITE) == model_plays_white:
            return ("win", move_count, game_id)
        else:
            return ("loss", move_count, game_id)

    finally:
        await engine.quit()


class GameState:
    """Manages the state of an ongoing game."""

    def __init__(self, game_id: int, model_plays_white: bool, starting_fen: Optional[str] = None):
        self.game_id = game_id
        self.board = chess.Board(
            starting_fen) if starting_fen else chess.Board()
        self.model_plays_white = model_plays_white
        self.move_count = 0
        self.max_moves = 200
        self.engine = None
        self.is_complete = False
        self.result = None

    def is_model_turn(self) -> bool:
        """Check if it's the model's turn to move."""
        return (self.board.turn == chess.WHITE) == self.model_plays_white

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.is_complete or self.board.is_game_over() or self.move_count >= self.max_moves

    def get_result(self) -> Tuple[str, int]:
        """Get the final result of the game."""
        if self.result:
            return self.result

        outcome = self.board.outcome()
        if outcome is None:
            self.result = ("draw", self.move_count)
        elif outcome.winner is None:
            self.result = ("draw", self.move_count)
        elif (outcome.winner == chess.WHITE) == self.model_plays_white:
            self.result = ("win", self.move_count)
        else:
            self.result = ("loss", self.move_count)

        return self.result


async def play_games_batched(
    model: torch.nn.Module,
    stockfish_path: str,
    num_games: int,
    device: torch.device,
    tokenizer,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stockfish_elo: int = STOCKFISH_ELO,
    verbose: bool = True,
    starting_positions: Optional[List[Tuple[str, bool]]] = None,
) -> Tuple[int, int, int, int]:
    """
    Play multiple games in parallel with batched model inference.

    Args:
        stockfish_elo: ELO rating to configure Stockfish to play at
        starting_positions: Optional list of (fen, model_plays_white) tuples

    Returns:
        Tuple of (wins, draws, losses, total_moves)
    """
    # Create game states with random color assignment or from starting positions
    game_states = []
    for i in range(num_games):
        if starting_positions and i < len(starting_positions):
            fen, model_plays_white = starting_positions[i]
            game_states.append(GameState(i, model_plays_white, fen))
        else:
            game_states.append(
                GameState(i, model_plays_white=random.random() < 0.5))

    # Initialize batch_size engines in parallel (reuse them across games)
    async def init_engine():
        transport, engine = await chess.engine.popen_uci(stockfish_path)
        await engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
        return (transport, engine)

    num_engines = min(batch_size, num_games)

    if verbose:
        with tqdm(total=num_engines, desc="Initializing engines", unit=" engines") as pbar:
            # Initialize engines in batches to update progress
            engines = []
            batch_init_size = 8  # Initialize 8 at a time to show progress
            for i in range(0, num_engines, batch_init_size):
                batch_count = min(batch_init_size, num_engines - i)
                batch_engines = await asyncio.gather(*[init_engine() for _ in range(batch_count)])
                engines.extend(batch_engines)
                pbar.update(batch_count)
        print("All engines initialized. Starting games...\n")
    else:
        engines = await asyncio.gather(*[init_engine() for _ in range(num_engines)])

    # Track which games are currently using which engines
    available_engines = list(range(num_engines))
    game_to_engine = {}  # Maps game_id to engine index
    pending_games = list(range(num_games))

    # Aggregate statistics
    wins = 0
    draws = 0
    losses = 0
    total_moves = 0
    completed_count = 0

    try:
        # Create progress bar for game moves
        with tqdm(desc="Playing games", unit=" moves", leave=False) as pbar:
            # Assign initial batch of games to engines
            for i in range(min(num_engines, len(pending_games))):
                game_id = pending_games.pop(0)
                engine_idx = available_engines.pop(0)
                game_to_engine[game_id] = engine_idx

            # Play all games, reusing engines
            while game_to_engine:
                # Get active games
                active_game_ids = list(game_to_engine.keys())
                active_games = [game_states[gid] for gid in active_game_ids]

                # Find games waiting for model moves
                games_needing_model = [
                    game for game in active_games
                    if not game.is_game_over() and game.is_model_turn()
                ]

                # Batch process model moves
                if games_needing_model:
                    boards = [game.board for game in games_needing_model]
                    moves = _select_moves_from_model_batch(
                        model, boards, device, tokenizer)

                    for game, move in zip(games_needing_model, moves):
                        if move is None or move not in game.board.legal_moves:
                            game.is_complete = True
                            game.result = ("loss", game.move_count)
                        else:
                            game.board.push(move)
                            game.move_count += 1
                            pbar.update(1)

                # Process Stockfish moves in parallel
                games_needing_stockfish = [
                    game for game in active_games
                    if not game.is_game_over() and not game.is_model_turn()
                ]

                if games_needing_stockfish:
                    # Run Stockfish for all games needing opponent moves
                    stockfish_tasks = [
                        engines[game_to_engine[game.game_id]][1].play(
                            game.board, chess.engine.Limit(time=ENGINE_TIME_LIMIT))
                        for game in games_needing_stockfish
                    ]
                    stockfish_results = await asyncio.gather(*stockfish_tasks)

                    # Apply moves to all games
                    for game, result in zip(games_needing_stockfish, stockfish_results):
                        game.board.push(result.move)
                        game.move_count += 1
                        pbar.update(1)

                # Check for completed games and reassign engines
                for game_id in list(game_to_engine.keys()):
                    game = game_states[game_id]
                    if game.is_game_over():
                        # Get result and update stats
                        result_str, num_moves = game.get_result()
                        total_moves += num_moves
                        completed_count += 1

                        if result_str == "win":
                            wins += 1
                        elif result_str == "draw":
                            draws += 1
                        else:
                            losses += 1

                        # Free up the engine
                        engine_idx = game_to_engine.pop(game_id)

                        # Assign a new game if available
                        if pending_games:
                            new_game_id = pending_games.pop(0)
                            game_to_engine[new_game_id] = engine_idx
                        else:
                            # No more games, return engine to pool
                            available_engines.append(engine_idx)

                        # Update progress bar with stats
                        current_score = wins + 0.5 * draws
                        score_pct = (current_score / completed_count *
                                     100) if completed_count > 0 else 0

                        # Calculate rolling window stats (last 50 games)
                        window_size = min(50, completed_count)
                        recent_games = completed_count - window_size
                        pbar.set_description(
                            f"Completed {completed_count}/{num_games} | Score: {score_pct:.1f}% ({wins}W-{draws}D-{losses}L)")

    finally:
        # Clean up engines
        for _, engine in engines:
            await engine.quit()

    return (wins, draws, losses, total_moves)


def evaluate_model_against_stockfish(
    model: torch.nn.Module,
    stockfish_path: str,
    num_games: int = 100,
    tokenizer=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    opponent_elo: int = STOCKFISH_ELO,
    verbose: bool = True,
    puzzle_csv_path: Optional[str | Path] = None,
) -> Tuple[float, float]:
    """
    Evaluate a model by playing multiple games against Stockfish with batched inference.

    Args:
        model: The chess model to evaluate
        stockfish_path: Path to the Stockfish executable
        num_games: Number of games to play (model plays random color each game)
        tokenizer: Tokenizer for processing FEN strings
        batch_size: Number of games to run in parallel
        opponent_elo: ELO rating to configure Stockfish and use for estimation
        verbose: Whether to print progress updates
        puzzle_csv_path: Optional path to puzzles.csv to load starting positions.
                        Games will be played in pairs from each position for fairness.

    Returns:
        Tuple of (estimated_elo, standard_error)
    """
    device = next(model.parameters()).device

    if tokenizer is None:
        from tokenizer import create_tokenizer
        tokenizer = create_tokenizer()

    # Load starting positions if requested
    starting_positions = None
    if puzzle_csv_path:
        if verbose:
            print(f"Loading starting positions from {puzzle_csv_path}...")
        starting_positions = load_puzzle_positions(puzzle_csv_path, num_games)
        if verbose:
            print(
                f"Loaded {len(starting_positions)} positions (each played from both sides)\n")

    if verbose:
        print(
            f"Playing {num_games} games against Stockfish (ELO {opponent_elo})...")
        if starting_positions:
            print(f"  Using puzzle positions from CSV")
        else:
            print(f"  Model plays random color each game")
        print(f"  Batch size: {batch_size} (parallel games)")
        print()

    # Run batched games
    wins, draws, losses, total_moves = asyncio.run(
        play_games_batched(
            model=model,
            stockfish_path=stockfish_path,
            num_games=num_games,
            device=device,
            tokenizer=tokenizer,
            batch_size=batch_size,
            stockfish_elo=opponent_elo,
            verbose=verbose,
            starting_positions=starting_positions,
        )
    )

    # Final ELO calculation
    estimated_elo, standard_error = estimate_elo_from_scores(
        wins, draws, losses, opponent_elo=opponent_elo
    )

    if verbose:
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Games played: {num_games}")
        print(f"Record: {wins}W-{draws}D-{losses}L")
        print(
            f"Score: {wins + 0.5 * draws}/{num_games} ({(wins + 0.5 * draws)/num_games:.1%})")
        print(f"Average game length: {total_moves/num_games:.1f} moves")
        print()
        print(f"Estimated ELO: {estimated_elo:.0f} ± {standard_error:.0f}")
        print(f"Opponent ELO: {opponent_elo}")
        print("=" * 60)

    return (estimated_elo, standard_error)


def main():
    """Test script to evaluate model from checkpoint against Stockfish."""
    import torch
    from model import ChessPolicyValueModel

    # Configuration
    CHECKPOINT_PATH = "./outputs/checkpoint-90000"
    # Adjust path as needed
    STOCKFISH_PATH = "/users/PAS2836/leedavis/stockfish/src/stockfish"
    NUM_GAMES = 400
    BATCH_SIZE = 128
    # TRUE_PERCENT = 1  # 100% random moves for testing
    TRUE_EVAL = 1350

    # Set to a path to load starting positions from puzzles.csv
    # Games will be played in pairs (one from each side) for fairness
    # Set to None to use standard starting position
    PUZZLE_CSV_PATH = "data/puzzles.csv"

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
        puzzle_csv_path=PUZZLE_CSV_PATH,
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Estimated ELO: {estimated_elo:.0f} ± {std_error:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
