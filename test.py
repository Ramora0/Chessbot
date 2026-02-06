"""
Analyze the distribution of winrates near 0 and 1 in the training data.

Computes min(p_win, 1 - p_win) for every move's win probability,
then buckets into log-scale bins to see how many values would be
affected by clipping (e.g., eps=1e-3 vs eps=1e-7 in the softmax target).
"""

import os
import sys
import numpy as np
from collections import Counter
from datasets import load_from_disk


def main():
    dataset_path = os.getenv("DATASET_PATH")
    if dataset_path is None:
        if len(sys.argv) > 1:
            dataset_path = sys.argv[1]
        else:
            print("Usage: python test.py <dataset_path>")
            print("   or: DATASET_PATH=... python test.py")
            sys.exit(1)

    num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    num_examples = min(num_examples, len(dataset))
    print(f"Analyzing {num_examples:,} examples out of {len(dataset):,} total")

    # Log-scale bin edges: 1e-7, 1e-6, ..., 1e-1, 5e-1
    bin_edges = [10**(-i) for i in range(7, 0, -1)] + [0.5]
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        bin_labels.append(f"[{bin_edges[i]:.0e}, {bin_edges[i+1]:.0e})")
    bin_labels.append(f"[{bin_edges[-1]:.0e}, inf)")  # the 0.5 bucket (values at 0.5 exactly)

    counts = Counter()
    total_moves = 0
    exactly_zero = 0
    exactly_one = 0

    for i in range(num_examples):
        p_wins = dataset[i]["p_win"]
        for p in p_wins:
            total_moves += 1
            if p == 0.0:
                exactly_zero += 1
            if p == 1.0:
                exactly_one += 1

            dist = min(p, 1.0 - p)

            # Find which bin this falls into
            placed = False
            for j, edge in enumerate(bin_edges):
                if dist < edge:
                    counts[j] = counts.get(j, 0) + 1
                    placed = True
                    break
            if not placed:
                counts[len(bin_edges)] = counts.get(len(bin_edges), 0) + 1

        if (i + 1) % 10_000 == 0:
            print(f"  processed {i+1:,} examples...")

    print(f"\n{'='*60}")
    print(f"Winrate distance from 0 or 1: min(p, 1-p)")
    print(f"Total moves analyzed: {total_moves:,}")
    print(f"Exactly 0.0: {exactly_zero:,} ({100*exactly_zero/total_moves:.4f}%)")
    print(f"Exactly 1.0: {exactly_one:,} ({100*exactly_one/total_moves:.4f}%)")
    print(f"{'='*60}\n")

    print(f"{'Bin':<22} {'Count':>12} {'%':>10} {'Cumulative %':>14}")
    print("-" * 60)

    cumulative = 0
    for j in range(len(bin_edges)):
        c = counts.get(j, 0)
        cumulative += c
        pct = 100 * c / total_moves
        cum_pct = 100 * cumulative / total_moves
        if j < len(bin_labels):
            label = bin_labels[j]
        else:
            label = "overflow"
        print(f"{label:<22} {c:>12,} {pct:>9.4f}% {cum_pct:>13.4f}%")

    # Also handle the overflow bin
    c = counts.get(len(bin_edges), 0)
    if c > 0:
        cumulative += c
        pct = 100 * c / total_moves
        cum_pct = 100 * cumulative / total_moves
        print(f"{'[5e-1, inf)':<22} {c:>12,} {pct:>9.4f}% {cum_pct:>13.4f}%")

    # Summary of what clipping thresholds would affect
    print(f"\n{'='*60}")
    print("Impact of clipping thresholds on softmax targets:")
    print(f"{'='*60}")
    for thresh_exp in [3, 4, 5, 6, 7]:
        thresh = 10 ** (-thresh_exp)
        affected = sum(counts.get(j, 0) for j in range(len(bin_edges)) if j < len(bin_edges) and bin_edges[j] <= thresh)
        # More precise: count moves with dist < thresh
        # The bin structure means bin j holds values in [bin_edges[j-1], bin_edges[j])
        # Bin 0 holds [0, 1e-7), bin 1 holds [1e-7, 1e-6), etc.
        # Values < thresh means all bins where the upper edge <= thresh
        affected = 0
        for j in range(len(bin_edges)):
            if bin_edges[j] <= thresh:
                affected += counts.get(j, 0)
            else:
                break
        pct = 100 * affected / total_moves
        print(f"  eps=1e-{thresh_exp}: {affected:>12,} moves clipped ({pct:.4f}%)")


if __name__ == "__main__":
    main()
