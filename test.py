"""Compare throughput of custom model vs Stockfish."""

import argparse
import random
import time
from typing import List, Tuple

import chess
import chess.engine
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import ChessPolicyValueModel
from policy_index import policy_index
from tokenizer import create_tokenizer, process_fen


def load_test_positions(num_positions: int, min_pieces: int = 10) -> List[str]:
    """
    Generate unique random positions for testing.

    Discards positions with fewer than min_pieces to avoid trivial endgames.
    """
    seen_positions = set()
    positions = []

    pbar = tqdm(total=num_positions, desc="Generating positions", unit=" pos")

    while len(positions) < num_positions:
        board = chess.Board()
        # Play 5-50 random moves for varied game phases
        num_moves = random.randint(5, 50)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))

        # Skip game-over or too-simple positions
        if board.is_game_over():
            continue
        piece_count = len(board.piece_map())
        if piece_count < min_pieces:
            continue

        # Deduplicate by board state (catches transpositions)
        pos_key = (board.board_fen(), board.turn, board.castling_rights, board.ep_square)
        if pos_key in seen_positions:
            continue

        seen_positions.add(pos_key)
        positions.append(board.fen())
        pbar.update(1)

    pbar.close()
    random.shuffle(positions)
    return positions


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


def prepare_model_inputs(
    fens: List[str],
    tokenizer,
    device: torch.device
) -> Tuple[torch.Tensor, List[chess.Board], List[torch.Tensor]]:
    """Prepare all model inputs ahead of time to exclude from timing."""
    boards = [chess.Board(fen) for fen in fens]
    processed = [process_fen(fen) for fen in fens]
    encodings = [tokenizer.encode(p) for p in processed]
    input_ids = torch.tensor(
        [enc.ids for enc in encodings], dtype=torch.long, device=device
    )
    legal_masks = [_get_legal_moves_mask(board).to(device) for board in boards]
    return input_ids, boards, legal_masks


def time_model_inference(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    boards: List[chess.Board],
    legal_masks: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 1,
    show_progress: bool = True,
) -> Tuple[float, List[chess.Move]]:
    """
    Time only the model inference and move selection.

    Returns:
        Tuple of (total_time_ms, selected_moves)
    """
    model.eval()
    moves = []

    num_positions = len(boards)
    total_time_ms = 0.0

    num_batches = (num_positions + batch_size - 1) // batch_size
    pbar = tqdm(
        total=num_positions,
        desc="Model inference",
        unit=" pos",
        disable=not show_progress
    )

    with torch.no_grad():
        for batch_start in range(0, num_positions, batch_size):
            batch_end = min(batch_start + batch_size, num_positions)
            batch_input_ids = input_ids[batch_start:batch_end]
            batch_masks = legal_masks[batch_start:batch_end]
            batch_boards = boards[batch_start:batch_end]

            # Synchronize GPU before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

            start_time = time.perf_counter()

            # Forward pass
            outputs = model(input_ids=batch_input_ids, return_dict=True)

            # Get policy logits
            if hasattr(outputs, 'attention_policy_logits') and outputs.attention_policy_logits is not None:
                policy_logits = outputs.attention_policy_logits
            else:
                policy_logits = outputs.policy_logits

            # Select moves for each position in batch
            for i in range(len(batch_boards)):
                masked_logits = policy_logits[i].masked_fill(
                    ~batch_masks[i], float("-inf")
                )
                probs = F.softmax(masked_logits, dim=-1)
                move_idx = torch.argmax(probs).item()
                move_uci = policy_index[move_idx]
                move = chess.Move.from_uci(move_uci)
                moves.append(move)

            # Synchronize GPU after timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

            end_time = time.perf_counter()
            batch_time_ms = (end_time - start_time) * 1000
            total_time_ms += batch_time_ms

            # Update progress bar
            batch_size_actual = batch_end - batch_start
            pbar.update(batch_size_actual)
            avg_ms = total_time_ms / (batch_start + batch_size_actual)
            pbar.set_postfix({"avg_ms": f"{avg_ms:.2f}"})

    pbar.close()
    return total_time_ms, moves


def time_stockfish_inference(
    stockfish_path: str,
    boards: List[chess.Board],
    nodes: int,
    threads: int = 1,
    show_progress: bool = True,
) -> Tuple[float, List[chess.Move]]:
    """
    Time Stockfish move selection with UCI threads and node limit.

    Uses a single persistent engine instance with configurable threads.
    Each position is searched up to the specified node count.

    Args:
        stockfish_path: Path to Stockfish executable
        boards: List of positions to evaluate
        nodes: Maximum nodes to search per position
        threads: Number of UCI threads for parallel search
        show_progress: Show progress bar

    Returns:
        Tuple of (total_time_ms, selected_moves)
    """
    moves = []
    total_time_ms = 0.0

    # Create engine once (not timed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Configure threads and minimal hash
    engine.configure({
        "Threads": threads,
        "Hash": 64,  # Reasonable hash size for multi-threaded search
    })

    pbar = tqdm(
        boards,
        desc=f"Stockfish ({nodes:,} nodes, {threads} threads)",
        unit=" pos",
        disable=not show_progress
    )

    for i, board in enumerate(pbar):
        # Time ONLY the search
        start_time = time.perf_counter()
        result = engine.play(board, chess.engine.Limit(nodes=nodes))
        end_time = time.perf_counter()

        pos_time_ms = (end_time - start_time) * 1000
        total_time_ms += pos_time_ms
        moves.append(result.move)

        # Update progress bar
        avg_ms = total_time_ms / (i + 1)
        pbar.set_postfix({"avg_ms": f"{avg_ms:.2f}"})

    # Shutdown engine (not timed)
    engine.quit()
    return total_time_ms, moves


def main():
    parser = argparse.ArgumentParser(
        description="Compare throughput of custom model vs Stockfish"
    )
    parser.add_argument(
        "-m", "--model", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--stockfish",
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish executable",
    )
    parser.add_argument(
        "--nodes", type=int, default=10000,
        help="Stockfish node limit per position (default: 10000)"
    )
    parser.add_argument(
        "-n", "--num-positions", type=int, default=2048,
        help="Number of positions to test"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32,
        help="Batch size for model inference (default: 32)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations (excluded from timing)"
    )
    parser.add_argument(
        "--stockfish-threads", type=int, default=4,
        help="UCI Threads for Stockfish parallel search (default: 4). "
             "Stockfish uses multiple threads to search a single position faster."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("THROUGHPUT COMPARISON: Custom Model vs Stockfish")
    print("=" * 70)
    print()

    # Load model
    print(f"Loading model from {args.model}...")
    model = ChessPolicyValueModel.from_pretrained_compiled(args.model)
    # Device priority: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    print()

    # Create tokenizer
    tokenizer = create_tokenizer()

    # Generate test positions
    print(f"Generating {args.num_positions} unique test positions...")
    test_fens = load_test_positions(args.num_positions + args.warmup)
    warmup_fens = test_fens[:args.warmup]
    test_fens = test_fens[args.warmup:]
    print(f"Generated {len(test_fens)} test + {len(warmup_fens)} warmup positions")
    print()

    # Prepare model inputs ahead of time (excluded from timing)
    print("Preparing model inputs...")
    test_input_ids, test_boards, test_legal_masks = prepare_model_inputs(
        test_fens, tokenizer, device
    )
    warmup_input_ids, warmup_boards, warmup_legal_masks = prepare_model_inputs(
        warmup_fens, tokenizer, device
    )
    print("Model inputs prepared")
    print()

    # Prepare Stockfish boards (also excluded from timing)
    stockfish_test_boards = [chess.Board(fen) for fen in test_fens]
    stockfish_warmup_boards = [chess.Board(fen) for fen in warmup_fens]

    # Warmup phase
    print(f"Running warmup ({args.warmup} positions)...")

    # Warmup model
    print("  Warming up model...")
    _, _ = time_model_inference(
        model, warmup_input_ids, warmup_boards, warmup_legal_masks,
        device, batch_size=args.batch_size, show_progress=False
    )

    # Warmup Stockfish
    print("  Warming up Stockfish...")
    _, _ = time_stockfish_inference(
        args.stockfish, stockfish_warmup_boards, args.nodes,
        threads=args.stockfish_threads, show_progress=False
    )

    print("Warmup complete")
    print()

    # Actual timing
    print("=" * 70)
    print("TIMING TEST")
    print("=" * 70)
    print(f"Positions: {len(test_fens)}")
    print(f"Model batch size: {args.batch_size}")
    print(f"Stockfish nodes: {args.nodes:,}")
    print(f"Stockfish threads: {args.stockfish_threads}")
    print()

    # Time model
    print("Timing model inference...")
    model_time_ms, model_moves = time_model_inference(
        model, test_input_ids, test_boards, test_legal_masks,
        device, batch_size=args.batch_size
    )

    # Time Stockfish
    print("Timing Stockfish search...")
    stockfish_time_ms, stockfish_moves = time_stockfish_inference(
        args.stockfish, stockfish_test_boards, args.nodes,
        threads=args.stockfish_threads
    )

    # Calculate statistics
    model_avg_ms = model_time_ms / len(test_fens)
    stockfish_avg_ms = stockfish_time_ms / len(test_fens)

    model_positions_per_sec = len(test_fens) / (model_time_ms / 1000)
    stockfish_positions_per_sec = len(test_fens) / (stockfish_time_ms / 1000)

    speedup = stockfish_time_ms / model_time_ms if model_time_ms > 0 else float('inf')

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Model':>15} {'Stockfish':>15}")
    print("-" * 70)
    print(f"{'Total time (ms)':<30} {model_time_ms:>15.2f} {stockfish_time_ms:>15.2f}")
    print(f"{'Avg time per position (ms)':<30} {model_avg_ms:>15.2f} {stockfish_avg_ms:>15.2f}")
    print(f"{'Positions per second':<30} {model_positions_per_sec:>15.1f} {stockfish_positions_per_sec:>15.1f}")
    print()
    print(f"Model speedup over Stockfish: {speedup:.1f}x")
    print()

    # Check move agreement
    agreements = sum(1 for m, s in zip(model_moves, stockfish_moves) if m == s)
    print(f"Move agreement: {agreements}/{len(test_fens)} ({agreements/len(test_fens)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
