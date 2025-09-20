"""Generate an evaluation dataset from chess puzzles.

For every puzzle we roll the position forward by the opponent's reply and
record the player-to-move state along with the ground-truth move, ready for
model evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

import chess
from datasets import Dataset

from tokenizer import create_tokenizer, process_fen_batch
from policy_index import policy_index


PUZZLES_CSV = Path("data/puzzles.csv")
OUTPUT_DIR = Path("/fs/scratch/PAS3150/lees_stuff/processed_puzzles_eval")
BATCH_SIZE = 2_048
POLICY_MOVES: List[str] = policy_index
POLICY_SIZE = len(POLICY_MOVES)
MOVE_TO_INDEX = {move: idx for idx, move in enumerate(POLICY_MOVES)}


def chunked(iterable: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(iterable), size):
        yield iterable[i: i + size]


def main() -> None:
    if not PUZZLES_CSV.exists():
        raise FileNotFoundError(f"Puzzle file not found at {PUZZLES_CSV}")

    print(f"Loading puzzles from '{PUZZLES_CSV}'...")
    with PUZZLES_CSV.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        raw_examples: List[dict] = []
        skipped = 0

        for row in reader:
            fen = row.get("FEN")
            moves_str = row.get("Moves", "").strip()
            if not fen or not moves_str:
                skipped += 1
                continue

            try:
                rating = int(row.get("Rating", 0))
            except (TypeError, ValueError):
                rating = 0

            puzzle_id = row.get("PuzzleId", "")
            move_tokens = [token for token in moves_str.split() if token]
            if len(move_tokens) < 2 or len(move_tokens) % 2 != 0:
                skipped += 1
                continue

            board = chess.Board(fen)

            for pair_index in range(0, len(move_tokens), 2):
                opponent_move_uci = move_tokens[pair_index]
                player_move_uci = move_tokens[pair_index + 1]

                try:
                    opponent_move = chess.Move.from_uci(opponent_move_uci)
                except ValueError:
                    try:
                        opponent_move = board.parse_san(opponent_move_uci)
                    except ValueError:
                        skipped += 1
                        break

                if opponent_move not in board.legal_moves:
                    skipped += 1
                    break

                board.push(opponent_move)
                board_state_fen = board.fen()
                legal_moves_list = list(board.legal_moves)
                legal_moves_mask = [0] * POLICY_SIZE
                for legal_move in legal_moves_list:
                    move_uci = legal_move.uci()
                    idx = MOVE_TO_INDEX.get(move_uci)
                    if idx is not None:
                        legal_moves_mask[idx] = 1

                try:
                    player_move = chess.Move.from_uci(player_move_uci)
                except ValueError:
                    try:
                        player_move = board.parse_san(player_move_uci)
                    except ValueError:
                        skipped += 1
                        break

                if player_move not in legal_moves_list:
                    skipped += 1
                    break

                player_move_uci_norm = player_move.uci()
                target_index = MOVE_TO_INDEX.get(player_move_uci_norm)
                if target_index is None:
                    skipped += 1
                    board.push(player_move)
                    continue

                raw_examples.append(
                    {
                        "fen": board_state_fen,
                        "puzzle_id": puzzle_id,
                        "rating": rating,
                        "target_policy_index": target_index,
                        "legal_moves_mask": legal_moves_mask,
                    }
                )

                board.push(player_move)

        print(f"Collected {len(raw_examples)} examples ({skipped} skipped).")

    print("Loading tokenizer...")
    tokenizer = create_tokenizer()

    processed_records: List[dict] = []
    for chunk in chunked(raw_examples, BATCH_SIZE):
        fens = [item["fen"] for item in chunk]
        processed_fens = process_fen_batch(fens)
        encodings = tokenizer.encode_batch(processed_fens)

        for item, encoding in zip(chunk, encodings):
            item.pop("fen", None)
            processed_record = {
                "puzzle_id": item["puzzle_id"],
                "rating": item["rating"],
                "target_policy_index": item["target_policy_index"],
                "legal_moves_mask": item["legal_moves_mask"],
                "input_ids": encoding.ids,
            }
            processed_records.append(processed_record)

    if not processed_records:
        raise RuntimeError("No evaluation examples were generated.")

    print("Building HuggingFace dataset...")
    dataset = Dataset.from_list(processed_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved evaluation dataset to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
