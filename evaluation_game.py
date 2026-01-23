"""Utility to evaluate a chess model by playing games against Stockfish."""

from __future__ import annotations

import asyncio
import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from policy_index import policy_index
from tokenizer import process_fen


# Stockfish configuration
STOCKFISH_ELO = 1350
ENGINE_TIME_LIMIT = 0.1  # Time limit in seconds for Stockfish moves
EVAL_TIME_LIMIT = 0.1    # Time limit for full-strength position evaluation

# Win% bucket configuration for conversion tracking
WIN_PERCENT_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                       (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]


def score_to_win_percent(score: chess.engine.Score, model_is_white: bool) -> Optional[float]:
    """Convert a Stockfish score to win percentage from model's perspective.

    Returns None if score is unavailable.
    """
    if score is None:
        return None

    # Get the score from white's perspective
    pov_score = score.white()

    if pov_score.is_mate():
        # Mate in N moves
        mate_in = pov_score.mate()
        if mate_in > 0:
            win_pct = 100.0  # White is winning
        else:
            win_pct = 0.0    # White is losing
    else:
        # Convert centipawns to win probability using standard formula
        # win_prob = 1 / (1 + 10^(-cp/400))
        cp = pov_score.score()
        if cp is None:
            return None
        win_pct = 100.0 / (1.0 + 10.0 ** (-cp / 400.0))

    # Flip if model is black
    if not model_is_white:
        win_pct = 100.0 - win_pct

    return win_pct


def get_bucket_index(win_pct: float) -> int:
    """Get the bucket index for a win percentage."""
    for i, (low, high) in enumerate(WIN_PERCENT_BUCKETS):
        if low <= win_pct < high:
            return i
    # Handle 100% edge case
    return len(WIN_PERCENT_BUCKETS) - 1


class ConversionTracker:
    """Tracks win/draw/loss rates from different Stockfish evaluation buckets."""

    def __init__(self):
        # For each bucket, track (wins, draws, losses)
        self.bucket_outcomes: Dict[int, List[int]] = defaultdict(lambda: [0, 0, 0])
        # Track all evals seen in each game: game_id -> list of bucket indices
        self.game_evals: Dict[int, List[int]] = defaultdict(list)

    def record_eval(self, game_id: int, win_pct: float):
        """Record an evaluation during a game."""
        bucket = get_bucket_index(win_pct)
        self.game_evals[game_id].append(bucket)

    def finalize_game(self, game_id: int, result: str):
        """Record final outcome for all positions seen in a game."""
        result_idx = {"win": 0, "draw": 1, "loss": 2}[result]

        # Attribute outcome to each unique bucket seen in the game
        buckets_seen = set(self.game_evals[game_id])
        for bucket in buckets_seen:
            self.bucket_outcomes[bucket][result_idx] += 1

        # Clean up game data
        del self.game_evals[game_id]

    def get_stats(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Get conversion statistics for each bucket.

        Returns dict mapping bucket range to stats dict with:
            - positions: number of unique game-buckets
            - actual_win_rate: actual win rate from those positions
            - expected_win_rate: midpoint of bucket (what Stockfish predicted)
        """
        stats = {}
        for i, (low, high) in enumerate(WIN_PERCENT_BUCKETS):
            outcomes = self.bucket_outcomes[i]
            total = sum(outcomes)
            if total == 0:
                continue

            wins, draws, losses = outcomes
            actual_win_rate = (wins + 0.5 * draws) / total * 100
            expected_win_rate = (low + high) / 2

            stats[(low, high)] = {
                "positions": total,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "actual_win_rate": actual_win_rate,
                "expected_win_rate": expected_win_rate,
            }

        return stats

    def print_stats(self):
        """Print a formatted table of conversion statistics."""
        stats = self.get_stats()
        if not stats:
            print("No conversion data collected.")
            return

        print("\n" + "=" * 70)
        print("CONVERSION STATISTICS (Stockfish Eval vs Actual Win Rate)")
        print("=" * 70)
        print(f"{'Eval Range':>12} | {'Games':>6} | {'W-D-L':>12} | {'Expected':>8} | {'Actual':>8} | {'Diff':>7}")
        print("-" * 70)

        total_positions = 0
        for (low, high), data in sorted(stats.items()):
            total_positions += data["positions"]
            wdl = f"{data['wins']}-{data['draws']}-{data['losses']}"
            diff = data["actual_win_rate"] - data["expected_win_rate"]
            diff_str = f"{diff:+.1f}%"
            print(f"{low:>5}-{high:<5}% | {data['positions']:>6} | {wdl:>12} | "
                  f"{data['expected_win_rate']:>7.1f}% | {data['actual_win_rate']:>7.1f}% | {diff_str:>7}")

        print("-" * 70)
        print(f"Total game-buckets tracked: {total_positions}")
        print("=" * 70)

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
) -> Tuple[List[Optional[chess.Move]], float]:
    """
    Use the model to select moves for a batch of positions.

    Returns:
        Tuple of (moves, avg_illegality_rate) where illegality_rate is the fraction
        of probability mass on illegal moves averaged across the batch.
    """
    if not boards:
        return [], 0.0

    # Process all FENs and tokenize
    fens = [board.fen() for board in boards]
    processed = [process_fen(fen) for fen in fens]
    encodings = [tokenizer.encode(p) for p in processed]
    input_ids = torch.tensor(
        [enc.ids for enc in encodings], dtype=torch.long, device=device)

    # Get model predictions for entire batch
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)

    # DEFAULT: Prefer attention policy head
    # Can be overridden with USE_OLD_POLICY_HEAD=1 environment variable
    use_old_head = os.getenv('USE_OLD_POLICY_HEAD', '0') == '1'

    if use_old_head:
        policy_logits = outputs.policy_logits
    elif hasattr(outputs, 'attention_policy_logits') and outputs.attention_policy_logits is not None:
        policy_logits = outputs.attention_policy_logits
    else:
        # Fallback to old head if attention head not available
        policy_logits = outputs.policy_logits

    # Process each position in the batch
    moves = []
    illegality_rates = []

    for i, board in enumerate(boards):
        # Mask illegal moves
        legal_mask = _get_legal_moves_mask(board).to(device)

        # Calculate illegality rate before masking
        all_probs = F.softmax(policy_logits[i], dim=-1)
        illegal_mask = ~legal_mask
        illegality_rate = (all_probs * illegal_mask.float()).sum().item()
        illegality_rates.append(illegality_rate)

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

    avg_illegality = sum(illegality_rates) / \
        len(illegality_rates) if illegality_rates else 0.0
    return moves, avg_illegality


def _select_move_from_model(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    tokenizer,
) -> Optional[chess.Move]:
    """Use the model to select a move from the current position (single position)."""
    moves, _ = _select_moves_from_model_batch(
        model, [board], device, tokenizer)
    return moves[0]


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
        self.starting_fen = starting_fen
        self.board = chess.Board(
            starting_fen) if starting_fen else chess.Board()
        self.model_plays_white = model_plays_white
        self.move_count = 0
        self.max_moves = 200
        self.engine = None
        self.is_complete = False
        self.result = None
        self.moves: List[chess.Move] = []  # Track all moves for PGN export

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
    conversion_tracker: Optional[ConversionTracker] = None,
) -> Tuple[int, int, int, int, float, int, List[GameState]]:
    """
    Play multiple games in parallel with batched model inference.

    Args:
        stockfish_elo: ELO rating to configure Stockfish to play at
        starting_positions: Optional list of (fen, model_plays_white) tuples
        conversion_tracker: Optional tracker for eval-to-outcome conversion stats

    Returns:
        Tuple of (wins, draws, losses, total_moves, total_illegality, illegality_count, game_states)
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
    async def init_play_engine():
        transport, engine = await chess.engine.popen_uci(stockfish_path)
        await engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
        return (transport, engine)

    async def init_eval_engine():
        transport, engine = await chess.engine.popen_uci(stockfish_path)
        # Full strength for accurate position evaluation
        return (transport, engine)

    num_engines = min(batch_size, num_games)
    # Use fewer eval engines since analyse() is fast
    num_eval_engines = min(8, num_engines) if conversion_tracker else 0

    if verbose:
        total_engines = num_engines + num_eval_engines
        with tqdm(total=total_engines, desc="Initializing engines", unit=" engines") as pbar:
            # Initialize play engines in batches
            engines = []
            batch_init_size = 8
            for i in range(0, num_engines, batch_init_size):
                batch_count = min(batch_init_size, num_engines - i)
                batch_engines = await asyncio.gather(*[init_play_engine() for _ in range(batch_count)])
                engines.extend(batch_engines)
                pbar.update(batch_count)

            # Initialize eval engines
            eval_engines = []
            if num_eval_engines > 0:
                eval_engines = await asyncio.gather(*[init_eval_engine() for _ in range(num_eval_engines)])
                pbar.update(num_eval_engines)

        print("All engines initialized. Starting games...\n")
    else:
        engines = await asyncio.gather(*[init_play_engine() for _ in range(num_engines)])
        eval_engines = []
        if num_eval_engines > 0:
            eval_engines = await asyncio.gather(*[init_eval_engine() for _ in range(num_eval_engines)])

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
    total_illegality = 0.0
    illegality_count = 0

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
                    moves, avg_illegality = _select_moves_from_model_batch(
                        model, boards, device, tokenizer)

                    # Track illegality
                    total_illegality += avg_illegality * len(boards)
                    illegality_count += len(boards)

                    # Games that will have moves applied (for eval tracking)
                    games_with_moves = []

                    for game, move in zip(games_needing_model, moves):
                        if move is None or move not in game.board.legal_moves:
                            game.is_complete = True
                            game.result = ("loss", game.move_count)
                        else:
                            game.moves.append(move)
                            game.board.push(move)
                            game.move_count += 1
                            pbar.update(1)
                            games_with_moves.append(game)

                    # Evaluate positions after model moves (full-strength Stockfish)
                    if conversion_tracker and eval_engines and games_with_moves:
                        # Process in batches - each engine handles one position per batch
                        num_eval = len(eval_engines)
                        for batch_start in range(0, len(games_with_moves), num_eval):
                            batch_games = games_with_moves[batch_start:batch_start + num_eval]
                            eval_tasks = [
                                eval_engines[i][1].analyse(
                                    game.board,
                                    chess.engine.Limit(time=EVAL_TIME_LIMIT)
                                )
                                for i, game in enumerate(batch_games)
                            ]
                            eval_results = await asyncio.gather(*eval_tasks)

                            for game, info in zip(batch_games, eval_results):
                                score = info.get("score")
                                if score:
                                    win_pct = score_to_win_percent(score, game.model_plays_white)
                                    if win_pct is not None:
                                        conversion_tracker.record_eval(game.game_id, win_pct)

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
                        game.moves.append(result.move)
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

                        # Record conversion stats
                        if conversion_tracker:
                            conversion_tracker.finalize_game(game_id, result_str)

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
        # Clean up play engines
        for _, engine in engines:
            await engine.quit()
        # Clean up eval engines
        for _, engine in eval_engines:
            await engine.quit()

    return (wins, draws, losses, total_moves, total_illegality, illegality_count, game_states)


def export_games_to_pgn(
    game_states: List[GameState],
    pgn_path: str | Path,
    opponent_elo: int,
    verbose: bool = True,
) -> None:
    """
    Export completed games to a PGN file.

    Args:
        game_states: List of completed GameState objects
        pgn_path: Path to write the PGN file
        opponent_elo: ELO rating of Stockfish opponent (for headers)
        verbose: Whether to print progress
    """
    from datetime import datetime

    if verbose:
        print(f"\nExporting {len(game_states)} games to {pgn_path}...")

    with open(pgn_path, 'w') as f:
        for i, game_state in enumerate(game_states):
            game = chess.pgn.Game()

            # Set headers
            game.headers["Event"] = "Model Evaluation"
            game.headers["Site"] = "Local"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i + 1)

            if game_state.model_plays_white:
                game.headers["White"] = "Model"
                game.headers["Black"] = f"Stockfish ({opponent_elo})"
            else:
                game.headers["White"] = f"Stockfish ({opponent_elo})"
                game.headers["Black"] = "Model"

            # Set result
            result_str, _ = game_state.get_result()
            if result_str == "win":
                if game_state.model_plays_white:
                    game.headers["Result"] = "1-0"
                else:
                    game.headers["Result"] = "0-1"
            elif result_str == "loss":
                if game_state.model_plays_white:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1-0"
            else:
                game.headers["Result"] = "1/2-1/2"

            # Set FEN if non-standard starting position
            if game_state.starting_fen:
                game.headers["FEN"] = game_state.starting_fen
                game.headers["SetUp"] = "1"

            # Add moves
            node = game
            board = chess.Board(game_state.starting_fen) if game_state.starting_fen else chess.Board()
            for move in game_state.moves:
                node = node.add_variation(move)

            # Write game to file
            print(game, file=f, end="\n\n")

    if verbose:
        print(f"PGN export complete: {pgn_path}")


def evaluate_model_against_stockfish(
    model: torch.nn.Module,
    stockfish_path: str,
    num_games: int = 100,
    tokenizer=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    opponent_elo: int = STOCKFISH_ELO,
    verbose: bool = True,
    puzzle_csv_path: Optional[str | Path] = None,
    pgn_path: Optional[str | Path] = None,
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
        pgn_path: Optional path to export all games as PGN file.

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

    # Create conversion tracker for eval-to-outcome stats
    conversion_tracker = ConversionTracker()

    # Run batched games
    wins, draws, losses, total_moves, total_illegality, illegality_count, game_states = asyncio.run(
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
            conversion_tracker=conversion_tracker,
        )
    )

    # Export PGN if requested
    if pgn_path:
        export_games_to_pgn(game_states, pgn_path, opponent_elo, verbose)

    # Final ELO calculation
    estimated_elo, standard_error = estimate_elo_from_scores(
        wins, draws, losses, opponent_elo=opponent_elo
    )

    if verbose:
        print("=" * 60)
        print("EVALUATION STATISTICS")
        print("=" * 60)
        print(f"Games played: {num_games}")
        print(f"Record: {wins}W-{draws}D-{losses}L")
        print(
            f"Score: {wins + 0.5 * draws}/{num_games} ({(wins + 0.5 * draws)/num_games:.1%})")
        print(f"Average game length: {total_moves/num_games:.1f} moves")
        if illegality_count > 0:
            avg_illegality = total_illegality / illegality_count
            print(f"Average illegality rate: {avg_illegality:.2%}")
        print()
        print(f"Estimated ELO: {estimated_elo:.0f} ± {standard_error:.0f}")
        print("=" * 60)

        # Print conversion statistics
        conversion_tracker.print_stats()

    return (estimated_elo, standard_error)


def main():
    """Test script to evaluate model from checkpoint against Stockfish."""
    import argparse
    import torch
    from model import ChessPolicyValueModel

    parser = argparse.ArgumentParser(description="Evaluate chess model against Stockfish")
    parser.add_argument("-m", "--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish", default="/users/PAS2836/leedavis/stockfish/src/stockfish",
                        help="Path to Stockfish executable")
    parser.add_argument("-n", "--num-games", type=int, default=400, help="Number of games to play")
    parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size for parallel games")
    parser.add_argument("--elo", type=int, default=1350, help="Stockfish ELO rating")
    parser.add_argument("--puzzles", default=None, help="Path to puzzles.csv for starting positions")
    parser.add_argument("--pgn", default=None, help="Path to export games as PGN file")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")

    # Load model (handles _orig_mod. prefix from torch.compile)
    model = ChessPolicyValueModel.from_pretrained_compiled(args.model)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(
        f"Starting evaluation with {args.num_games} games (batch size: {args.batch_size})...")
    print()

    # Run evaluation
    estimated_elo, std_error = evaluate_model_against_stockfish(
        model=model,
        stockfish_path=args.stockfish,
        num_games=args.num_games,
        batch_size=args.batch_size,
        opponent_elo=args.elo,
        verbose=True,
        puzzle_csv_path=args.puzzles,
        pgn_path=args.pgn,
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Estimated ELO: {estimated_elo:.0f} ± {std_error:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
