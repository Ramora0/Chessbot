"""
Test script to debug the policy loss computation.
"""

from datasets import load_from_disk
from action_value_dataset import ActionValueDataset
from tokenizer import create_tokenizer

DATASET_PATH = "/fs/scratch/PAS2836/lees_stuff/action_value"

# First, look at raw dataset
print("=" * 60)
print("RAW DATASET (before ActionValueDataset processing)")
print("=" * 60)
raw_dataset = load_from_disk(DATASET_PATH).to_iterable_dataset()
example = next(iter(raw_dataset))

fen = example["fen"]
moves = example["moves"]
p_wins = example["p_win"]

print(f"FEN: {fen}")
print(f"\nRaw win% values (sorted):")
move_wins = sorted(zip(moves, p_wins), key=lambda x: x[1], reverse=True)
for move, win_pct in move_wins[:10]:
    print(f"  {move:6s}: {win_pct*100:6.2f}%")
if len(move_wins) > 10:
    print(f"  ... and {len(move_wins) - 10} more moves")

# Now look at processed dataset
print("\n" + "=" * 60)
print("PROCESSED (after ActionValueDataset)")
print("=" * 60)

tokenizer = create_tokenizer()
processed_dataset = ActionValueDataset(
    dataset_path=DATASET_PATH,
    tokenizer=tokenizer,
    streaming=True,
    shuffle_buffer_size=0,
)

processed = next(iter(processed_dataset))
policy = processed["policy"]

print(f"Policy tensor shape: {policy.shape}")
print(f"Policy min: {policy.min().item():.4f}")
print(f"Policy max: {policy.max().item():.4f}")

# Show legal moves (policy >= -0.99, since -1 is illegal)
legal_mask = policy > -0.99
legal_values = policy[legal_mask]
print(f"\nLegal moves count: {legal_mask.sum().item()}")
print(f"Legal move values - min: {legal_values.min().item():.4f}, max: {legal_values.max().item():.4f}")

# The problem: after normalization, best move = 0, others are negative
# So when model.py checks (policy >= 0), only the best move is "legal"!
print(f"\nMoves with policy >= 0: {(policy >= 0).sum().item()}")
print(f"Moves with policy > -0.99: {(policy > -0.99).sum().item()}")
