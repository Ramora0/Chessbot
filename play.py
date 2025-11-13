"""Interactive script to play against the chess model with a pygame GUI."""

import torch
import torch.nn.functional as F
import chess
import pygame
from typing import List, Tuple, Optional
import math
import os

from model import ChessPolicyValueModel
from policy_index import policy_index
from tokenizer import create_tokenizer, process_fen


# Constants
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
PIECE_IMAGE_DIR = "./pieces"  # Directory for piece images


def _move_to_policy_index(move: chess.Move) -> int:
    """Convert a chess.Move to a policy index."""
    move_str = move.uci()
    try:
        return policy_index.index(move_str)
    except ValueError:
        return -1


def get_model_move_probabilities(
    model: torch.nn.Module,
    board: chess.Board,
    device: torch.device,
    tokenizer,
) -> Tuple[List[Tuple[chess.Move, float, float]], float]:
    """Get move probabilities for all legal moves from the model.

    Returns:
        Tuple of (move_probabilities, entropy) where entropy is in bits
        move_probabilities is a list of (move, probability, raw_logit) tuples
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

    # Create legal moves mask
    legal_mask = torch.zeros(
        len(policy_index), dtype=torch.bool, device=device)
    legal_moves_list = []
    for move in board.legal_moves:
        idx = _move_to_policy_index(move)
        if idx >= 0:
            legal_mask[idx] = True
            legal_moves_list.append((move, idx))

    # Mask illegal moves and get probabilities
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    probs = F.softmax(masked_logits, dim=-1)

    # Calculate entropy in bits (log base 2)
    entropy = 0.0
    for _, idx in legal_moves_list:
        p = probs[idx].item()
        if p > 0:
            entropy -= p * math.log2(p)

    # Get probabilities and raw logits for all legal moves
    move_probs = []
    for move, idx in legal_moves_list:
        prob = probs[idx].item()
        raw_logit = policy_logits[idx].item()
        move_probs.append((move, prob, raw_logit))

    # Sort by probability descending
    move_probs.sort(key=lambda x: x[1], reverse=True)
    return move_probs, entropy


def display_move_probabilities(move_probs: List[Tuple[chess.Move, float, float]], entropy: float, top_n: int = 10):
    """Display the top N moves and their probabilities in terminal."""
    print("\nModel move probabilities:")
    print("-" * 80)
    for i, (move, prob, logit) in enumerate(move_probs[:top_n]):
        idx = _move_to_policy_index(move)
        print(
            f"{i+1:2d}. {move.uci():6s}  {prob*100:6.2f}%  logit: {logit:7.3f}  (index: {idx})")
    if len(move_probs) > top_n:
        print(f"... and {len(move_probs) - top_n} more moves")
    print(
        f"\nEntropy: {entropy:.3f} bits (max: {math.log2(len(move_probs)):.3f} bits for {len(move_probs)} moves)")


def load_piece_images() -> dict:
    """Load chess piece images. Returns a dict mapping piece symbols to pygame Surfaces."""
    pieces = {}
    piece_names = {
        'P': 'white-pawn', 'N': 'white-knight', 'B': 'white-bishop',
        'R': 'white-rook', 'Q': 'white-queen', 'K': 'white-king',
        'p': 'black-pawn', 'n': 'black-knight', 'b': 'black-bishop',
        'r': 'black-rook', 'q': 'black-queen', 'k': 'black-king'
    }

    # Try to load from piece image directory
    if os.path.exists(PIECE_IMAGE_DIR):
        for symbol, name in piece_names.items():
            img_path = os.path.join(PIECE_IMAGE_DIR, f"{name}.png")
            if os.path.exists(img_path):
                img = pygame.image.load(img_path)
                pieces[symbol] = pygame.transform.scale(
                    img, (SQUARE_SIZE, SQUARE_SIZE))

    # If no images loaded, create simple text-based pieces
    if not pieces:
        print("No piece images found. Using text-based pieces.")
        print(
            f"You can add PNG images to '{PIECE_IMAGE_DIR}/' directory with names like: white-pawn.png, black-king.png, etc.")
        font = pygame.font.Font(None, SQUARE_SIZE)
        for symbol, name in piece_names.items():
            surface = pygame.Surface(
                (SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            text = font.render(symbol, True, (0, 0, 0))
            text_rect = text.get_rect(
                center=(SQUARE_SIZE // 2, SQUARE_SIZE // 2))
            surface.blit(text, text_rect)
            pieces[symbol] = surface

    return pieces


def square_to_coords(square: int, flipped: bool = False) -> Tuple[int, int]:
    """Convert chess square index to pixel coordinates."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    if flipped:
        x = (7 - file) * SQUARE_SIZE
        y = rank * SQUARE_SIZE
    else:
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE

    return x, y


def coords_to_square(x: int, y: int, flipped: bool = False) -> int:
    """Convert pixel coordinates to chess square index."""
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)

    if flipped:
        file = 7 - file
        rank = 7 - rank

    return chess.square(file, rank)


def draw_board(screen: pygame.Surface, board: chess.Board, piece_images: dict,
               selected_square: Optional[int] = None, legal_moves: List[chess.Move] = None,
               flipped: bool = False):
    """Draw the chess board and pieces."""
    # Colors
    LIGHT_SQUARE = (240, 217, 181)
    DARK_SQUARE = (181, 136, 99)
    SELECTED_COLOR = (186, 202, 68)
    LEGAL_MOVE_COLOR = (130, 151, 105, 128)

    # Draw squares
    for square in chess.SQUARES:
        x, y = square_to_coords(square, flipped)
        color = LIGHT_SQUARE if (chess.square_file(
            square) + chess.square_rank(square)) % 2 == 0 else DARK_SQUARE

        # Highlight selected square
        if square == selected_square:
            color = SELECTED_COLOR

        pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

    # Highlight legal move destinations
    if legal_moves and selected_square is not None:
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(surface, LEGAL_MOVE_COLOR,
                           (SQUARE_SIZE // 2, SQUARE_SIZE // 2), SQUARE_SIZE // 6)
        for move in legal_moves:
            if move.from_square == selected_square:
                x, y = square_to_coords(move.to_square, flipped)
                screen.blit(surface, (x, y))

    # Draw pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x, y = square_to_coords(square, flipped)
            screen.blit(piece_images[piece.symbol()], (x, y))


def main():
    # Configuration
    CHECKPOINT_PATH = "./no-mask/checkpoint-35000"  # Adjust path as needed

    print("Loading model from checkpoint...")
    model = ChessPolicyValueModel.from_pretrained_compiled(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    tokenizer = create_tokenizer()

    print(f"Model loaded on {device}")
    print()

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Chess vs AI")
    clock = pygame.time.Clock()

    # Load piece images
    piece_images = load_piece_images()

    # Optional: Start from a specific FEN
    fen_input = input(
        "Enter FEN to start from (or press Enter for starting position): ").strip()
    if fen_input:
        try:
            board = chess.Board(fen_input)
            print(f"Starting from position: {fen_input}")
        except ValueError:
            print("Invalid FEN, starting from default position")
            board = chess.Board()
    else:
        board = chess.Board()

    # Human always plays as white
    human_plays_white = board.turn == chess.WHITE
    board_flipped = not human_plays_white

    print()
    print("=" * 60)
    print("GAME START")
    print("=" * 60)
    print(f"You are playing as {'White' if human_plays_white else 'Black'}")
    print()

    # Game state
    selected_square = None
    dragging_piece = None
    drag_pos = None
    running = True
    game_over = False
    ai_thinking = False

    # If AI plays first, make the first move
    if not ((board.turn == chess.WHITE) == human_plays_white):
        ai_thinking = True

    while running:
        # Handle AI move if it's AI's turn
        if ai_thinking and not game_over:
            print("\nModel is thinking...")
            move_probs, entropy = get_model_move_probabilities(
                model, board, device, tokenizer)
            display_move_probabilities(move_probs, entropy, top_n=10)

            if move_probs:
                best_move, best_prob, best_logit = move_probs[0]
                print(
                    f"\nModel plays: {best_move.uci()} ({best_prob*100:.2f}%)")
                board.push(best_move)
                ai_thinking = False
            else:
                print("Model has no legal moves!")
                game_over = True
                ai_thinking = False

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over and not ai_thinking:
                if (board.turn == chess.WHITE) == human_plays_white:
                    x, y = event.pos
                    square = coords_to_square(x, y, board_flipped)
                    piece = board.piece_at(square)

                    if piece and piece.color == board.turn:
                        selected_square = square
                        dragging_piece = piece
                        drag_pos = event.pos

            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece:
                    drag_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and not game_over and not ai_thinking:
                if dragging_piece and selected_square is not None:
                    x, y = event.pos
                    to_square = coords_to_square(x, y, board_flipped)

                    # Try to make the move
                    move = None
                    for legal_move in board.legal_moves:
                        if legal_move.from_square == selected_square and legal_move.to_square == to_square:
                            # Handle promotion
                            if legal_move.promotion:
                                # Default to queen promotion for simplicity
                                move = chess.Move(
                                    selected_square, to_square, promotion=chess.QUEEN)
                            else:
                                move = legal_move
                            break

                    if move and move in board.legal_moves:
                        board.push(move)
                        print(f"\nYou played: {move.uci()}")
                        ai_thinking = True

                    selected_square = None
                    dragging_piece = None
                    drag_pos = None

        # Check for game over
        if board.is_game_over() and not game_over:
            game_over = True
            print()
            print("=" * 60)
            print("GAME OVER")
            print("=" * 60)

            outcome = board.outcome()
            if outcome:
                if outcome.winner is None:
                    print("Result: Draw")
                elif (outcome.winner == chess.WHITE) == human_plays_white:
                    print("Result: You win!")
                else:
                    print("Result: Model wins!")
                print(f"Termination: {outcome.termination.name}")

        # Drawing
        screen.fill((50, 50, 50))

        # Get legal moves for highlighting
        legal_moves = list(board.legal_moves) if selected_square else None

        # Draw board and pieces
        draw_board(screen, board, piece_images,
                   selected_square, legal_moves, board_flipped)

        # Draw dragging piece
        if dragging_piece and drag_pos:
            piece_surface = piece_images[dragging_piece.symbol()]
            piece_rect = piece_surface.get_rect(center=drag_pos)
            screen.blit(piece_surface, piece_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
