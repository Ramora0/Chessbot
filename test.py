#!/usr/bin/env python
"""
Check for duplicate FENs in the action value dataset.
"""

from pathlib import Path
from collections import Counter
from datasets import load_from_disk
from tqdm.auto import tqdm

# Which dataset to check
DATASET_PATH = "/fs/scratch/PAS2836/lees_stuff/action_value"
# DATASET_PATH = "/fs/scratch/PAS2836/lees_stuff/action_value_sharded"
# DATASET_PATH = "/fs/scratch/PAS2836/lees_stuff/action_value_1shard"

# How many examples to check (set to None to check all)
MAX_EXAMPLES = None  # Change to e.g., 100_000 for faster testing


def check_duplicates(dataset_path: str, max_examples: int = None):
    """
    Check for duplicate FENs in the dataset.

    Args:
        dataset_path: Path to the HF dataset directory
        max_examples: Maximum number of examples to check (None = all)
    """
    print(f"Loading dataset from: {dataset_path}")

    # Load dataset (non-streaming to get length)
    try:
        dataset = load_from_disk(dataset_path)
        total_examples = len(dataset)
        print(f"Dataset size: {total_examples:,} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying streaming mode...")
        from datasets import load_dataset
        import glob
        dataset_dir = Path(dataset_path)
        arrow_files = glob.glob(str(dataset_dir / "**" / "*.arrow"), recursive=True)
        if not arrow_files:
            raise ValueError(f"No arrow files found in {dataset_path}")
        dataset = load_dataset("arrow", data_files=arrow_files, split="train", streaming=True)
        total_examples = None
        print("Using streaming mode (size unknown)")

    # Track FENs
    fen_counter = Counter()
    examples_checked = 0
    duplicates_found = 0

    # Determine how many to check
    check_limit = max_examples if max_examples is not None else total_examples

    print(f"Checking for duplicates (limit: {check_limit or 'all'})...")
    print("(Will print immediately when duplicates are found)\n")

    # Iterate through dataset
    pbar = tqdm(total=check_limit, desc="Checking FENs", unit="examples")

    for example in dataset:
        fen = example["fen"]
        fen_counter[fen] += 1

        # Print immediately when a duplicate is found
        if fen_counter[fen] == 2:
            duplicates_found += 1
            pbar.write(f"DUPLICATE #{duplicates_found}: {fen} (first seen at some earlier position)")
        elif fen_counter[fen] > 2:
            pbar.write(f"  Additional occurrence of: {fen} (now seen {fen_counter[fen]} times)")

        examples_checked += 1
        pbar.update(1)

        if max_examples is not None and examples_checked >= max_examples:
            break

    pbar.close()

    if duplicates_found == 0:
        print("\n✓ No duplicates found!")

    # Analyze results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total examples checked: {examples_checked:,}")
    print(f"Unique FENs: {len(fen_counter):,}")
    print(f"Duplicate FENs: {sum(1 for count in fen_counter.values() if count > 1):,}")

    # Find most duplicated FENs
    most_common = fen_counter.most_common(10)
    print(f"\nTop 10 most frequent FENs:")
    for fen, count in most_common:
        if count > 1:
            print(f"  {count:4d}x: {fen}")

    # Distribution of duplicates
    duplicate_counts = Counter(count for count in fen_counter.values() if count > 1)
    if duplicate_counts:
        print(f"\nDuplicate distribution:")
        for count, freq in sorted(duplicate_counts.items()):
            print(f"  {freq:,} FENs appear {count} times")

    # Calculate duplicate percentage
    total_duplicates = sum(count - 1 for count in fen_counter.values() if count > 1)
    duplicate_percentage = (total_duplicates / examples_checked) * 100
    print(f"\nDuplicate percentage: {duplicate_percentage:.2f}%")
    print(f"(i.e., {total_duplicates:,} / {examples_checked:,} examples are duplicates)")


if __name__ == "__main__":
    check_duplicates(DATASET_PATH, max_examples=MAX_EXAMPLES)
