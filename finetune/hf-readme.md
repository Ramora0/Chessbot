---
license: cc-by-4.0
tags:
- chess
- stockfish
pretty_name: ChessBench Action-Values + Mate-in-N
size_categories:
- 100M<n<1B
---

# ChessBench Action-Values + Mate-in-N

This dataset is derived from the action-value data released with ["Grandmaster-Level Chess Without Search"](https://arxiv.org/abs/2402.04494) (DeepMind). It provides:

1. A reorganization of the original action-value format into a **per-position** structure: each FEN maps to a list of all legal moves with per-move win probabilities (as provided in the upstream release).
2. A **mate-in-N augmentation**: for moves with win probability >= 99.99% or <= 0.01%, Stockfish mate search adds a `mate` depth field per move when a forced mate is detected.

## Use Cases

- Training a policy/value model with richer per-position action structure
- Studying calibration of win probability vs. mate depth
- Training models to predict mate imminence signals beyond saturated win rates

## Source Dataset

- **Upstream**: DeepMind ["searchless chess" / ChessBench](https://github.com/google-deepmind/searchless_chess) action-value release
- **Paper**: ["Grandmaster-Level Chess Without Search"](https://arxiv.org/abs/2402.04494) (Ruoss et al., 2024)
- **Upstream license**: Some portions are CC0 (Lichess), remainder is CC BY 4.0

## Schema

Each row contains:

| Field | Type | Description |
|-------|------|-------------|
| `fen` | `str` | Chess position in FEN notation |
| `moves` | `List[str]` | All legal moves in UCI format (e.g., `["e2e4", "d2d4", ...]`) |
| `p_win` | `List[float]` | Win probability for side-to-move per move, in `[0.0, 1.0]` (unchanged from source) |
| `mate` | `List[int]` | Mate depth per move (new field, see below) |

All three list fields (`moves`, `p_win`, `mate`) share the same length for each row.

## Mate Field Definition

The `mate` field encodes forced mate depth in **full moves** (not plies), from the perspective of the **side to move**:

| Value | Meaning |
|-------|---------|
| `mate > 0` | Playing this move leads to mating the opponent in N moves |
| `mate < 0` | Playing this move leads to getting mated in N moves |
| `mate = 0` | No forced mate detected, or move was not analyzed |

Only moves meeting the extreme win probability threshold are analyzed for mate depth. All other moves have `mate = 0`.

### Analysis Thresholds

- Moves with `p_win >= 0.9999` are analyzed as potential winning mates
- Moves with `p_win <= 0.0001` are analyzed as potential losing mates
- All other moves are not analyzed (`mate = 0`)

## How Mate-in-N Was Computed

Mate depth labels were generated using Stockfish with depth 16; we found this found forced mates for ~98.5% of moves with winrates of 0% or 100%.

### Analysis Procedure

For each move meeting the win probability threshold:

1. The move is applied to the board position
2. If the resulting position is immediate checkmate, `mate = 1` (or `-1` if a losing move)
3. Otherwise, Stockfish analyzes the resulting position to depth 16
4. If Stockfish reports a mate score, the mate depth (in full moves) is recorded
5. If no mate score is found within the depth limit, `mate = 0`

**Note**: `mate = 0` does not imply no forced mate exists -- only that none was found within the search depth limit.

## Limitations / Known Issues

- Mate-in-N labels depend on Stockfish search depth; deeper searches may find mates that depth 16 misses
- Upstream win probabilities near 0% or 100% may reflect Stockfish evaluation saturation rather than true forced mates, which is why not all analyzed moves yield confirmed mates
- Legal move generation and UCI formatting must match python-chess rules for underpromotions, castling rights, and en passant

## License

This dataset is licensed under **CC BY 4.0**.

### Attribution & Changes

- **Upstream attribution**: DeepMind "searchless chess" action-value release
- **Changes**: Reorganization from per-move records into per-position legal-move lists, plus mate-in-N augmentation via Stockfish analysis

## Citation

If you use this dataset, please cite the upstream DeepMind paper:

```bibtex
@article{ruoss2024grandmaster,
  title={Grandmaster-Level Chess Without Search},
  author={Ruoss, Anian and Del{\'e}tang, Gr{\'e}goire and Medapati, Sourabh and Grau-Moya, Jordi and Wenliang, Li Ke and Catt, Elliot and Reid, John and Genewein, Tim},
  journal={arXiv preprint arXiv:2402.04494},
  year={2024}
}
```
