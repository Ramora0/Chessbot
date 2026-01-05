"""
Shard the action_value dataset for multi-worker data loading.
"""

from datasets import load_from_disk

DATASET_PATH = "/fs/scratch/PAS2836/lees_stuff/action_value"
OUTPUT_PATH = "/fs/scratch/PAS2836/lees_stuff/action_value_1shard"
NUM_SHARDS = 1  # Single shard to eliminate boundary effects

print(f"Loading dataset from '{DATASET_PATH}'...")
dataset = load_from_disk(DATASET_PATH)

print(f"Dataset size: {len(dataset)} examples")
print(f"Saving to '{OUTPUT_PATH}' with {NUM_SHARDS} shards...")

dataset.save_to_disk(OUTPUT_PATH, num_shards=NUM_SHARDS)

print("Done!")
