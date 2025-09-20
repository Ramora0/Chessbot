"""Preprocess the ChessFENS dataset into token id sequences."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from tokenizer import create_tokenizer, process_fen_batch


DATASET_NAME = "Maxlegrec/ChessFENS"
DATASET_SPLIT = "train"
OUTPUT_DIR = "/fs/scratch/PAS3150/lees_stuff/processed_chessfens"
BATCH_SIZE = 1_024


def main() -> None:
    print("Loading tokenizer...")
    tokenizer = create_tokenizer()

    print(f"Loading dataset '{DATASET_NAME}' (split: {DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"Dataset loaded with {len(dataset)} samples.")

    print("Preprocessing and tokenizing FEN strings...")

    def preprocess_batch(batch: dict) -> dict:
        processed_fens = process_fen_batch(batch["fen"])
        encodings = tokenizer.encode_batch(processed_fens)
        return {
            "processed_fen": processed_fens,
            "input_ids": [encoding.ids for encoding in encodings],
        }

    processed_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Tokenizing",
    )

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed dataset to '{output_path}'...")
    processed_dataset.save_to_disk(str(output_path))
    print("Processing complete.")


if __name__ == "__main__":
    main()
