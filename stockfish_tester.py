"""Test games between two Stockfish instances with different configurations."""

from __future__ import annotations

import argparse
import asyncio
import math
import random
from typing import Dict, Optional, Tuple

import chess
import chess.engine
from tqdm import tqdm


# Default configurations
DEFAULT_ELO_LIMITED_ELO = 1350
DEFAULT_ELO_LIMITED_TIME = 0.1  # seconds
DEFAULT_NUM_GAMES = 250
DEFAULT_BATCH_SIZE = 8  # Conservative for laptop

# Engine 2 limit options
DEFAULT_LIMIT_TYPE = "depth"  # "time", "depth", or "nodes"
DEFAULT_DEPTH = 1
DEFAULT_NODES = 1000
DEFAULT_TIME_MS = 1  # milliseconds


def estimate_elo_difference(
    wins: int,
    draws: int,
    losses: int,
) -> Tuple[float, float]:
    """
    Estimate ELO difference based on game results.

    Uses the standard ELO formula to compute the rating difference.
    Positive means engine1 is stronger than engine2.

    Returns:
        Tuple of (elo_difference, standard_error)
    """
    total_games = wins + draws + losses
    if total_games == 0:
        return (float("nan"), float("nan"))

    # Calculate actual score (1 for win, 0.5 for draw, 0 for loss)
    actual_score = (wins + 0.5 * draws) / total_games

    # Clamp score to avoid numerical issues
    actual_score = max(0.001, min(0.999, actual_score))

    # Solve for ELO difference:
    # actual_score = 1 / (1 + 10^(-elo_diff / 400))
    # Rearranging: 10^(-elo_diff / 400) = (1 - actual_score) / actual_score
    # -elo_diff / 400 = log10((1 - actual_score) / actual_score)
    # elo_diff = -400 * log10((1 - actual_score) / actual_score)
    elo_diff = -400 * math.log10((1 - actual_score) / actual_score)

    # Calculate standard error using binomial approximation
    variance_per_game = actual_score * (1 - actual_score)
    if variance_per_game > 0:
        standard_error = 400 / (math.sqrt(total_games) * math.log(10) * math.sqrt(variance_per_game))
    else:
        standard_error = float("inf")

    return (elo_diff, standard_error)


class GameState:
    """Manages the state of an ongoing game between two engines."""

    def __init__(self, game_id: int, engine1_plays_white: bool, starting_fen: Optional[str] = None):
        self.game_id = game_id
        self.board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self.engine1_plays_white = engine1_plays_white
        self.move_count = 0
        self.max_moves = 200
        self.is_complete = False
        self.result = None

    def is_engine1_turn(self) -> bool:
        """Check if it's engine1's turn to move."""
        return (self.board.turn == chess.WHITE) == self.engine1_plays_white

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.is_complete or self.board.is_game_over() or self.move_count >= self.max_moves

    def get_result(self) -> Tuple[str, int]:
        """Get the final result from engine1's perspective."""
        if self.result:
            return self.result

        outcome = self.board.outcome()
        if outcome is None:
            self.result = ("draw", self.move_count)
        elif outcome.winner is None:
            self.result = ("draw", self.move_count)
        elif (outcome.winner == chess.WHITE) == self.engine1_plays_white:
            self.result = ("win", self.move_count)
        else:
            self.result = ("loss", self.move_count)

        return self.result


async def play_games(
    stockfish_path: str,
    num_games: int,
    elo_limited_elo: int = DEFAULT_ELO_LIMITED_ELO,
    elo_limited_time: float = DEFAULT_ELO_LIMITED_TIME,
    limit_type: str = DEFAULT_LIMIT_TYPE,
    limit_depth: int = DEFAULT_DEPTH,
    limit_nodes: int = DEFAULT_NODES,
    limit_time_ms: int = DEFAULT_TIME_MS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Play games between ELO-limited Stockfish (engine1) and limited Stockfish (engine2).

    Args:
        stockfish_path: Path to Stockfish executable
        num_games: Number of games to play
        elo_limited_elo: ELO setting for the ELO-limited engine
        elo_limited_time: Time limit in seconds for ELO-limited engine
        limit_type: Type of limit for engine2 ("time", "depth", or "nodes")
        limit_depth: Depth limit for engine2 (if limit_type="depth")
        limit_nodes: Node limit for engine2 (if limit_type="nodes")
        limit_time_ms: Time limit in ms for engine2 (if limit_type="time")
        batch_size: Number of games to run in parallel
        verbose: Whether to print progress

    Returns:
        Tuple of (wins, draws, losses, total_moves) from ELO-limited engine's perspective
    """
    # Create game states with alternating colors for fairness
    # Use a dict to map game_id -> GameState
    game_states: Dict[int, GameState] = {}
    pending_game_ids = []
    for i in range(num_games):
        engine1_plays_white = (i % 2 == 0)
        game_states[i] = GameState(i, engine1_plays_white)
        pending_game_ids.append(i)

    # Shuffle to randomize order
    random.shuffle(pending_game_ids)

    # Initialize engines
    async def init_elo_limited_engine():
        transport, engine = await chess.engine.popen_uci(stockfish_path)
        await engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo_limited_elo})
        return (transport, engine)

    async def init_limited_engine():
        transport, engine = await chess.engine.popen_uci(stockfish_path)
        # No strength limiting - uses depth/time/nodes limit only
        return (transport, engine)

    # Create the limit object for engine2
    if limit_type == "depth":
        engine2_limit = chess.engine.Limit(depth=limit_depth)
    elif limit_type == "nodes":
        engine2_limit = chess.engine.Limit(nodes=limit_nodes)
    else:  # time
        engine2_limit = chess.engine.Limit(time=limit_time_ms / 1000.0)

    num_engine_pairs = min(batch_size, num_games)

    if verbose:
        print(f"Initializing {num_engine_pairs} engine pairs...")

    # Initialize engine pairs
    elo_engines = []
    limited_engines = []

    with tqdm(total=num_engine_pairs * 2, desc="Initializing engines", unit=" engines", disable=not verbose) as pbar:
        batch_init_size = 8
        for i in range(0, num_engine_pairs, batch_init_size):
            batch_count = min(batch_init_size, num_engine_pairs - i)

            # Initialize ELO-limited engines
            batch_elo = await asyncio.gather(*[init_elo_limited_engine() for _ in range(batch_count)])
            elo_engines.extend(batch_elo)
            pbar.update(batch_count)

            # Initialize limited engines (depth/time/nodes)
            batch_limited = await asyncio.gather(*[init_limited_engine() for _ in range(batch_count)])
            limited_engines.extend(batch_limited)
            pbar.update(batch_count)

    if verbose:
        print("All engines initialized. Starting games...\n")

    # Track which games are using which engine pairs
    available_engine_pairs = list(range(num_engine_pairs))
    game_to_engine_pair: Dict[int, int] = {}

    # Statistics
    wins = 0
    draws = 0
    losses = 0
    total_moves = 0
    completed_count = 0

    try:
        with tqdm(desc="Playing games", unit=" moves", leave=False, disable=not verbose) as pbar:
            # Assign initial batch of games to engine pairs
            for i in range(min(num_engine_pairs, len(pending_game_ids))):
                game_id = pending_game_ids.pop(0)
                engine_pair_idx = available_engine_pairs.pop(0)
                game_to_engine_pair[game_id] = engine_pair_idx

            while game_to_engine_pair:
                active_game_ids = list(game_to_engine_pair.keys())
                active_games = [game_states[gid] for gid in active_game_ids]

                # Process engine1 (ELO-limited) moves
                games_needing_engine1 = [
                    game for game in active_games
                    if not game.is_game_over() and game.is_engine1_turn()
                ]

                if games_needing_engine1:
                    tasks = [
                        elo_engines[game_to_engine_pair[game.game_id]][1].play(
                            game.board, chess.engine.Limit(time=elo_limited_time)
                        )
                        for game in games_needing_engine1
                    ]
                    results = await asyncio.gather(*tasks)

                    for game, result in zip(games_needing_engine1, results):
                        game.board.push(result.move)
                        game.move_count += 1
                        pbar.update(1)

                # Process engine2 (limited) moves
                games_needing_engine2 = [
                    game for game in active_games
                    if not game.is_game_over() and not game.is_engine1_turn()
                ]

                if games_needing_engine2:
                    tasks = [
                        limited_engines[game_to_engine_pair[game.game_id]][1].play(
                            game.board, engine2_limit
                        )
                        for game in games_needing_engine2
                    ]
                    results = await asyncio.gather(*tasks)

                    for game, result in zip(games_needing_engine2, results):
                        game.board.push(result.move)
                        game.move_count += 1
                        pbar.update(1)

                # Check for completed games
                for game_id in list(game_to_engine_pair.keys()):
                    game = game_states[game_id]
                    if game.is_game_over():
                        result_str, num_moves = game.get_result()
                        total_moves += num_moves
                        completed_count += 1

                        if result_str == "win":
                            wins += 1
                        elif result_str == "draw":
                            draws += 1
                        else:
                            losses += 1

                        # Free engine pair
                        engine_pair_idx = game_to_engine_pair.pop(game_id)

                        if pending_game_ids:
                            new_game_id = pending_game_ids.pop(0)
                            game_to_engine_pair[new_game_id] = engine_pair_idx
                        else:
                            available_engine_pairs.append(engine_pair_idx)

                        # Update progress
                        score_pct = (wins + 0.5 * draws) / completed_count * 100 if completed_count > 0 else 0
                        pbar.set_description(
                            f"Completed {completed_count}/{num_games} | ELO-limited score: {score_pct:.1f}% ({wins}W-{draws}D-{losses}L)"
                        )

    finally:
        # Clean up engines
        for _, engine in elo_engines:
            await engine.quit()
        for _, engine in limited_engines:
            await engine.quit()

    return (wins, draws, losses, total_moves)


def run_stockfish_test(
    stockfish_path: str,
    num_games: int = DEFAULT_NUM_GAMES,
    elo_limited_elo: int = DEFAULT_ELO_LIMITED_ELO,
    elo_limited_time: float = DEFAULT_ELO_LIMITED_TIME,
    limit_type: str = DEFAULT_LIMIT_TYPE,
    limit_depth: int = DEFAULT_DEPTH,
    limit_nodes: int = DEFAULT_NODES,
    limit_time_ms: int = DEFAULT_TIME_MS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True,
) -> Tuple[int, int, int, float, float]:
    """
    Run Stockfish vs Stockfish test and compute statistics.

    Returns:
        Tuple of (wins, draws, losses, elo_difference, standard_error)
        where wins/draws/losses are from the ELO-limited engine's perspective
    """
    if verbose:
        print("=" * 60)
        print("STOCKFISH VS STOCKFISH TEST")
        print("=" * 60)
        print(f"Engine 1: ELO-limited (UCI_Elo={elo_limited_elo}, time={elo_limited_time}s)")
        if limit_type == "depth":
            print(f"Engine 2: Depth-limited (depth={limit_depth})")
        elif limit_type == "nodes":
            print(f"Engine 2: Node-limited (nodes={limit_nodes})")
        else:
            print(f"Engine 2: Time-limited ({limit_time_ms}ms per move)")
        print(f"Games: {num_games} (batch size: {batch_size})")
        print("=" * 60)
        print()

    wins, draws, losses, total_moves = asyncio.run(
        play_games(
            stockfish_path=stockfish_path,
            num_games=num_games,
            elo_limited_elo=elo_limited_elo,
            elo_limited_time=elo_limited_time,
            limit_type=limit_type,
            limit_depth=limit_depth,
            limit_nodes=limit_nodes,
            limit_time_ms=limit_time_ms,
            batch_size=batch_size,
            verbose=verbose,
        )
    )

    elo_diff, std_error = estimate_elo_difference(wins, draws, losses)

    if verbose:
        print()
        print("=" * 60)
        print("RESULTS (from ELO-limited engine's perspective)")
        print("=" * 60)
        print(f"Games played: {num_games}")
        print(f"Record: {wins}W-{draws}D-{losses}L")
        score = wins + 0.5 * draws
        print(f"Score: {score}/{num_games} ({score/num_games:.1%})")
        print(f"Average game length: {total_moves/num_games:.1f} moves")
        print()
        print(f"ELO difference: {elo_diff:+.0f} ± {std_error:.0f}")
        if elo_diff > 0:
            print(f"  (ELO-limited engine is ~{elo_diff:.0f} ELO stronger)")
        elif elo_diff < 0:
            print(f"  ({limit_type.capitalize()}-limited engine is ~{-elo_diff:.0f} ELO stronger)")
        else:
            print("  (Engines are approximately equal)")
        print("=" * 60)

    return (wins, draws, losses, elo_diff, std_error)


def main():
    parser = argparse.ArgumentParser(
        description="Test games between ELO-limited and depth/time/nodes-limited Stockfish"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="/usr/local/bin/stockfish",
        help="Path to Stockfish executable",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=DEFAULT_NUM_GAMES,
        help=f"Number of games to play (default: {DEFAULT_NUM_GAMES})",
    )
    parser.add_argument(
        "--elo-limited-elo",
        type=int,
        default=DEFAULT_ELO_LIMITED_ELO,
        help=f"ELO setting for ELO-limited engine (default: {DEFAULT_ELO_LIMITED_ELO})",
    )
    parser.add_argument(
        "--elo-limited-time",
        type=float,
        default=DEFAULT_ELO_LIMITED_TIME,
        help=f"Time limit in seconds for ELO-limited engine (default: {DEFAULT_ELO_LIMITED_TIME})",
    )
    # Engine 2 limit type options
    parser.add_argument(
        "--limit-type",
        type=str,
        choices=["time", "depth", "nodes"],
        default=DEFAULT_LIMIT_TYPE,
        help=f"Limit type for engine 2 (default: {DEFAULT_LIMIT_TYPE})",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Depth limit for engine 2 when --limit-type=depth (default: {DEFAULT_DEPTH})",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=DEFAULT_NODES,
        help=f"Node limit for engine 2 when --limit-type=nodes (default: {DEFAULT_NODES})",
    )
    parser.add_argument(
        "--time-ms",
        type=int,
        default=DEFAULT_TIME_MS,
        help=f"Time limit in ms for engine 2 when --limit-type=time (default: {DEFAULT_TIME_MS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of games to run in parallel (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    run_stockfish_test(
        stockfish_path=args.stockfish_path,
        num_games=args.num_games,
        elo_limited_elo=args.elo_limited_elo,
        elo_limited_time=args.elo_limited_time,
        limit_type=args.limit_type,
        limit_depth=args.depth,
        limit_nodes=args.nodes,
        limit_time_ms=args.time_ms,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
