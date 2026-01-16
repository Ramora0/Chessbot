"""
Analyze the distribution of total piece value across positions in the dataset.

Samples positions from the training dataset and creates histograms of total
material on the board and game phase classification.
"""

import os
import argparse
from collections import Counter

import matplotlib.pyplot as plt
from datasets import load_from_disk
from tqdm import tqdm

# Standard piece values
PIECE_VALUES = {
    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
}

# Maximum possible material (starting position): 2*(8*1 + 2*3 + 2*3 + 2*5 + 1*9) = 78
MAX_MATERIAL = 78

# Endgame threshold: total material < 26 points
ENDGAME_THRESHOLD = 26

# Opening threshold: fullmove number <= 10 (halfmoves <= 20)
OPENING_FULLMOVE_THRESHOLD = 10


def calculate_total_piece_value(fen: str) -> int:
    """Calculate total piece value from a FEN string."""
    board_part = fen.split()[0]

    total = 0
    for char in board_part:
        if char in PIECE_VALUES:
            total += PIECE_VALUES[char]

    return total


def calculate_material_by_side(fen: str) -> tuple[int, int]:
    """Calculate material for each side from a FEN string.

    Returns:
        (white_material, black_material)
    """
    board_part = fen.split()[0]

    white_total = 0
    black_total = 0

    for char in board_part:
        if char.isupper() and char in PIECE_VALUES:
            white_total += PIECE_VALUES[char]
        elif char.islower() and char in PIECE_VALUES:
            black_total += PIECE_VALUES[char]

    return white_total, black_total


def calculate_material_balance(fen: str) -> int:
    """Calculate material balance (white - black) from a FEN string."""
    white_total, black_total = calculate_material_by_side(fen)
    return white_total - black_total


def get_fullmove_number(fen: str) -> int:
    """Extract fullmove number from FEN string."""
    parts = fen.split()
    if len(parts) >= 6:
        return int(parts[5])
    return 0


def classify_position(fen: str) -> str:
    """
    Classify a position as opening, middlegame, or endgame.

    - Endgame: total piece value < 26
    - Opening: fullmove number <= 10 (and not endgame)
    - Middlegame: otherwise
    """
    total_material = calculate_total_piece_value(fen)
    fullmove = get_fullmove_number(fen)

    # Check endgame first (material-based, takes priority)
    if total_material < ENDGAME_THRESHOLD:
        return "endgame"

    # Check opening (move-based)
    if fullmove <= OPENING_FULLMOVE_THRESHOLD:
        return "opening"

    return "middlegame"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze piece value distribution in the training dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.getenv("DATASET_PATH"),
        help="Path to the HuggingFace dataset (default: $DATASET_PATH env var)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of positions to sample (default: 100,000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the plot (default: display interactively)"
    )
    args = parser.parse_args()

    if not args.dataset_path:
        raise ValueError(
            "Dataset path required. Set DATASET_PATH env var or use --dataset-path"
        )

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    total_size = len(dataset)
    print(f"Dataset contains {total_size:,} positions")

    sample_size = min(args.sample_size, total_size)
    print(f"Sampling {sample_size:,} positions...")

    # Sample evenly across the dataset
    step = max(1, total_size // sample_size)
    indices = range(0, total_size, step)[:sample_size]

    total_values = []
    material_balances = []
    phase_classifications = []

    for idx in tqdm(indices, desc="Analyzing positions"):
        fen = dataset[idx]["fen"]
        total_values.append(calculate_total_piece_value(fen))
        material_balances.append(calculate_material_balance(fen))
        phase_classifications.append(classify_position(fen))

    # Create figure with three subplots
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Total piece value distribution
    ax1 = axes[0]
    ax1.hist(total_values, bins=range(0, MAX_MATERIAL + 2), edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Total Piece Value")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Distribution of Total Piece Value\n(n={sample_size:,})")
    ax1.axvline(x=78, color='red', linestyle='--', label='Starting position (78)')

    # Add statistics
    mean_val = sum(total_values) / len(total_values)
    ax1.axvline(x=mean_val, color='green', linestyle='--', label=f'Mean ({mean_val:.1f})')
    ax1.legend()

    # Plot 2: Material balance distribution
    ax2 = axes[1]
    balance_range = max(abs(min(material_balances)), abs(max(material_balances)))
    bins = range(-balance_range - 1, balance_range + 2)
    ax2.hist(material_balances, bins=bins, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel("Material Balance (White - Black)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distribution of Material Balance\n(n={sample_size:,})")
    ax2.axvline(x=0, color='red', linestyle='--', label='Equal material')

    mean_balance = sum(material_balances) / len(material_balances)
    ax2.axvline(x=mean_balance, color='green', linestyle='--', label=f'Mean ({mean_balance:.2f})')
    ax2.legend()

    # Plot 3: Game phase distribution
    ax3 = axes[2]
    phase_counts = Counter(phase_classifications)
    phases = ['opening', 'middlegame', 'endgame']
    counts = [phase_counts[p] for p in phases]
    colors = ['#4CAF50', '#2196F3', '#FF9800']  # green, blue, orange

    bars = ax3.bar(phases, counts, color=colors, edgecolor='black', alpha=0.8)
    ax3.set_xlabel("Game Phase")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Distribution of Game Phases\n(n={sample_size:,})")

    # Add percentage labels on bars
    for bar, count in zip(bars, counts):
        pct = 100 * count / sample_size
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sample_size * 0.01,
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nTotal Piece Value:")
    print(f"  Min: {min(total_values)}")
    print(f"  Max: {max(total_values)}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {sorted(total_values)[len(total_values)//2]}")

    print(f"\nMaterial Balance:")
    print(f"  Min: {min(material_balances)}")
    print(f"  Max: {max(material_balances)}")
    print(f"  Mean: {mean_balance:.2f}")
    print(f"  Positions with equal material: {material_balances.count(0):,} ({100*material_balances.count(0)/len(material_balances):.1f}%)")

    # Count distribution of total values
    value_counts = Counter(total_values)
    print(f"\nMost common total piece values:")
    for val, count in value_counts.most_common(10):
        print(f"  {val}: {count:,} ({100*count/len(total_values):.1f}%)")

    print(f"\nGame Phase Distribution:")
    for phase in phases:
        count = phase_counts[phase]
        pct = 100 * count / sample_size
        print(f"  {phase.capitalize():12s}: {count:,} ({pct:.1f}%)")

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
