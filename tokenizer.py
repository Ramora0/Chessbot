from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


_EMPTY_SQUARE_TOKEN = "e-p"
_PIECE_TOKENS = {
    "r": "r-p",
    "n": "n-p",
    "b": "b-p",
    "q": "q-p",
    "k": "k-p",
    "p": "p-p",
    "R": "R-p",
    "N": "N-p",
    "B": "B-p",
    "Q": "Q-p",
    "K": "K-p",
    "P": "P-p",
}
_DIGIT_TOKENS = {
    str(i): tuple([_EMPTY_SQUARE_TOKEN] * i)
    for i in range(1, 9)
}
_TURN_TOKENS = {"w": "w-t", "b": "b-t"}
_CASTLING_ORDER = "KQkq"
_CASTLING_TOKENS = {c: f"{c}-c" for c in _CASTLING_ORDER}
_CASTLING_EMPTY = "e-c"
_EN_PASSANT_EMPTY = "e-ep"
_EN_PASSANT_SUFFIX = "-ep"
_THINK_TOKEN = "<THINK>"
_ACT_TOKEN = "<ACT>"


def process_fen(fen: str) -> str:
    """Convert a FEN string into a space-delimited token sequence."""
    parts = fen.split()

    # First part (board): expand digits to 'e-p' tokens, pieces to '<piece>-p'
    board = parts[0]
    board_rows = board.split('/')
    board_processed = []
    for row in board_rows:
        row_tokens = []
        for c in row:
            if c.isdigit():
                row_tokens.extend(['e-p'] * int(c))
            else:
                row_tokens.append(f'{c}-p')
        spaced_line = ' '.join(row_tokens)
        board_processed.append(spaced_line)
    board_result = ' '.join(board_processed)

    # Second part (turn): add -t
    turn = parts[1] + '-t'

    # Third part (castling): standardize to 4 positions (K, Q, k, q)
    castling_raw = parts[2]
    castling_positions = []
    castling_positions.append('K-c' if 'K' in castling_raw else 'e-c')
    castling_positions.append('Q-c' if 'Q' in castling_raw else 'e-c')
    castling_positions.append('k-c' if 'k' in castling_raw else 'e-c')
    castling_positions.append('q-c' if 'q' in castling_raw else 'e-c')
    castling = ' '.join(castling_positions)

    # Fourth part (en passant)
    en_passant = (parts[3] + '-ep') if parts[3] != '-' else 'e-ep'

    # Fifth part (halfmove): add -hm
    # halfmove = parts[4] + '-hm'

    # Sixth part (fullmove): add -fm
    # fullmove = parts[5] + '-fm'

    return f"{board_result} {turn} {castling} {en_passant} {_THINK_TOKEN} {_ACT_TOKEN}"


def process_fen_batch(fens: Iterable[str]) -> List[str]:
    """Process multiple FEN strings ahead of batch tokenization."""
    results: List[str] = []
    append_result = results.append

    piece_tokens = _PIECE_TOKENS
    digit_tokens = _DIGIT_TOKENS
    turn_tokens = _TURN_TOKENS
    castling_tokens = _CASTLING_TOKENS
    castling_order = _CASTLING_ORDER
    castling_empty = _CASTLING_EMPTY
    en_passant_empty = _EN_PASSANT_EMPTY
    en_passant_suffix = _EN_PASSANT_SUFFIX

    for fen in fens:
        board, turn, castling, en_passant, *_ = fen.split()

        board_tokens: List[str] = []
        board_append = board_tokens.append
        board_extend = board_tokens.extend
        for row in board.split('/'):
            for char in row:
                if char in piece_tokens:
                    board_append(piece_tokens[char])
                else:
                    board_extend(digit_tokens[char])
        board_str = " ".join(board_tokens)

        turn_token = turn_tokens[turn]
        castling_str = " ".join(
            castling_tokens[c] if c in castling else castling_empty
            for c in castling_order
        )
        en_passant_token = (
            f"{en_passant}{en_passant_suffix}"
            if en_passant != "-"
            else en_passant_empty
        )

        append_result(
            f"{board_str} {turn_token} {castling_str} {en_passant_token} {_THINK_TOKEN} {_ACT_TOKEN}"
        )

    return results


def create_vocabulary():
    """Create the full vocabulary used for inputs only (no move tokens)."""
    vocab = set()

    # 1. Board pieces with -p (12 pieces + empty)
    pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P', 'e']
    for piece in pieces:
        vocab.add(f"{piece}-p")

    # 2. Turn tokens with -t
    for turn in ['w', 'b']:
        vocab.add(f"{turn}-t")

    # 3. Castling rights with -c and -ep for empty positions
    castling_chars = ['K', 'Q', 'k', 'q']
    for char in castling_chars:
        vocab.add(f"{char}-c")
    vocab.add("e-c")  # Empty castling position

    # 4. En passant squares with -ep (a3-h3, a6-h6) plus empty
    files = 'abcdefgh'
    for file in files:
        vocab.add(f"{file}3-ep")
        vocab.add(f"{file}6-ep")
    vocab.add("e-ep")

    # 5. Halfmove clock (0..800)
    # for i in range(801):
    #     vocab.add(f"{i}-hm")

    # 6. Fullmove number (1..400)
    # for i in range(1, 401):
    #     vocab.add(f"{i}-fm")

    vocab.add(_THINK_TOKEN)
    vocab.add(_ACT_TOKEN)

    return sorted(vocab)


def create_tokenizer():
    """Create a WordLevel tokenizer for the chess input vocabulary only."""
    vocab_list = create_vocabulary()
    vocab_dict = {token: i for i, token in enumerate(vocab_list)}

    # Add essential special tokens only
    special_tokens = ["[PAD]", "[UNK]"]
    for token in special_tokens:
        vocab_dict[token] = len(vocab_dict)

    tokenizer = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    try:
        tokenizer.disable_padding()
    except Exception:
        pass
    return tokenizer
