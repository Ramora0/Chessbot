"""Preprocess the ChessFENS dataset into token id sequences."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from tokenizer import create_tokenizer, process_fen_batch


DATASET_NAME = "Maxlegrec/ChessFENS"
DATASET_SPLIT = "train"
OUTPUT_DIR = "/fs/scratch/PAS3150/lees_stuff/processed_chessfens"
BATCH_SIZE = 1_024
EXPECTED_SEQ_LEN = 70
# Controls the on-disk shard size when saving the processed dataset.
MAX_SHARD_SIZE = "1500MB"


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

        input_ids = []
        append_ids = input_ids.append
        for encoding in encodings:
            ids = list(encoding.ids)
            if len(ids) != EXPECTED_SEQ_LEN:
                raise ValueError(
                    f"Processed sequence length {len(ids)} does not match expected {EXPECTED_SEQ_LEN}"
                )
            append_ids(ids)

        return {
            "processed_fen": processed_fens,
            "input_ids": input_ids,
            "wdl": batch["wdl"],
        }

    processed_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Tokenizing",
    )

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Saving processed dataset to '{output_path}' with max shard size {MAX_SHARD_SIZE}..."
    )
    processed_dataset.save_to_disk(
        str(output_path),
        max_shard_size=MAX_SHARD_SIZE,
    )
    print("Processing complete.")


if __name__ == "__main__":
    main()
