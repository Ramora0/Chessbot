"""Consolidate an existing ChessFENS dataset into fewer Arrow shards."""

from __future__ import annotations

import shutil
from pathlib import Path

from datasets import load_from_disk


INPUT_DIR = Path("/fs/scratch/PAS3150/lees_stuff/processed_chessfens")
OUTPUT_DIR = Path(
    "/fs/scratch/PAS3150/lees_stuff/processed_chessfens_consolidated")
SHARD_COUNT = 512
OVERWRITE_OUTPUT = False


def main() -> None:
    input_dir = INPUT_DIR.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input dataset directory not found: {input_dir}")

    output_dir = OUTPUT_DIR.expanduser().resolve()
    if output_dir.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Set OVERWRITE_OUTPUT = True to replace it."
            )
        shutil.rmtree(output_dir)

    print(f"Loading dataset from '{input_dir}'...")
    dataset = load_from_disk(str(input_dir))

    print("Flattening indices...")
    dataset = dataset.flatten_indices()

    print(
        f"Saving consolidated dataset to '{output_dir}' with {SHARD_COUNT} shards..."
    )
    dataset.save_to_disk(
        str(output_dir),
        num_shards=SHARD_COUNT
    )
    print("Done.")


if __name__ == "__main__":
    main()
