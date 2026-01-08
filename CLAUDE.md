# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A transformer-based chess engine that plays **without search**, predicting optimal moves through learned pattern recognition. Unlike traditional engines (AlphaZero, Stockfish), this achieves true zero-search prediction with a single forward pass.

## Data Source and Format

### Training Data Origin

The model is trained on **Google DeepMind's "Grandmaster-Level Chess Without Search" dataset** (searchless_chess), containing:
- **500,000,000+ chess positions**
- **Stockfish win% evaluations for every legal move** in each position
- Multiple difficulty levels and game phases

This is fundamentally different from typical chess datasets that only label the best move. Here, every legal move gets a win probability, enabling the model to learn fine-grained move quality distinctions.

### Dataset Schema

```python
{
    "fen": str,              # Chess position in FEN notation
    "moves": List[str],      # Legal moves in UCI format (e.g., ["e2e4", "d2d4", ...])
    "p_win": List[float],    # Win probability [0.0-1.0] for each move
}
```

After preprocessing (action_value_dataset.py), the data is transformed to:

```python
{
    "input_ids": List[int],     # Tokenized position (70 tokens)
    "policy": Tensor[1858],     # Normalized win% per move: best=0.0, worse<0, illegal=-1.0
    "wdl": Tensor[128],         # Win probability distribution over 128 bins
    "true_value": float,        # Absolute win% of position after best move
}
```

**Critical encoding detail**: The `policy` tensor uses **relative win% encoding**:
- Best move in position: `0.0` (no regret)
- Suboptimal legal moves: negative values (e.g., `-0.03` = loses 3% win probability vs best)
- Illegal moves: `-1.0` (sentinel value)

This encoding enables the expected regret loss formulation.

## Architecture Overview

### Model Hierarchy

```
ChessPolicyValueModel (model.py)
├── LlamaModel (transformer backbone)
│   ├── Embedding layer (vocab → hidden_size)
│   ├── Positional embeddings (70 learned positions)
│   └── 20 transformer layers (default)
├── MultiTaskAttentionPooling
│   ├── Shared K/V projections
│   └── Task-specific queries & output heads:
│       ├── Policy head (1858 moves)
│       ├── WDL head (128 bins)
│       └── Illegality head (1858 moves)
└── LM head (masked token prediction)
```

### Multi-Task Learning Design

The model simultaneously optimizes **5 distinct objectives**:

1. **Policy Loss (softmax-based)**: Expected regret from not choosing best move
2. **Move Win% Loss (sigmoid-based)**: Absolute win% prediction for all legal moves
3. **WDL Loss**: Position value distribution over 128 bins (Huber loss)
4. **Illegality Loss**: Binary classification of legal vs illegal moves
5. **Masked Token Loss**: Language modeling objective on board tokens

**Key insight**: The policy head logits serve **dual purposes** - both softmax policy distribution and sigmoid win% predictions. This architectural sharing forces the model to learn a rich move representation.

Loss weights are centralized in `loss_weights.py` with careful tuning:
- Policy: 10.0 (dominant signal)
- Move win%: 0.7 (ranking all moves)
- WDL: 30.0 (position understanding)
- Illegality: 50.0 (legal move detection)
- Masked token: 1.0 (representation learning)

### Position Tokenization (tokenizer.py)

Chess positions (FEN) are tokenized into exactly **70 tokens**:

**Board representation (64 tokens)**:
- One token per square (a8→h8, a7→h7, ..., a1→h1)
- Piece tokens: `R-p`, `N-p`, `B-p`, `Q-p`, `K-p`, `P-p` (white uppercase)
- Piece tokens: `r-p`, `n-p`, `b-p`, `q-p`, `k-p`, `p-p` (black lowercase)
- Empty square: `e-p`

**Metadata (6 tokens)**:
- Token 64: Turn (`w-t` or `b-t`)
- Tokens 65-68: Castling rights (`K-c`, `Q-c`, `k-c`, `q-c` or `e-c` for empty)
- Token 69: En passant square (`a3-ep` through `h6-ep` or `e-ep` for none)

**Positional embeddings**: The model learns 70 position-specific embeddings added to token embeddings, allowing instant recognition of "knight on f3" vs "knight on g1" even though both use token `N-p`.

### Move Policy Indexing (policy_index.py)

All possible chess moves are mapped to **1858 indices**:
- Every (from_square, to_square) pair for all 64 squares
- Includes queen, rook, bishop, and knight underpromotions
- Index 0 = `a1h8`, index 1 = `a1a8`, etc.

The policy head outputs 1858 logits, one per move. During inference:
1. Apply legal move mask from python-chess
2. Softmax over legal moves only
3. Sample or take argmax

### Multi-Task Attention Pooling (model.py:22-88)

Novel pooling mechanism that computes all task outputs efficiently:

1. **Shared computation**: Single K/V projection across all tasks
2. **Task-specific queries**: Each task (policy, WDL, illegality) has a learnable query vector
3. **Attention pooling**: Each query attends to all 70 position tokens
4. **Output projection**: Task-specific linear layers produce final logits

This is more efficient than separate pooling layers while maintaining task-specific feature extraction.

## Data Flow Pipeline

### Training Pipeline (train.py)

```
HuggingFace Dataset (sharded Arrow files)
    ↓ (streaming with shuffle)
ChessPolicyCollator (data.py)
    ↓ (batch creation, optional masking)
{input_ids, policy, wdl, true_value} tensors
    ↓
ChessPolicyValueModel.forward()
    ├→ Embed tokens (vocab → hidden_size)
    ├→ Add positional embeddings (70 positions)
    ├→ Transformer (20 layers)
    ├→ Multi-task attention pooling
    │   ├→ Policy logits [batch, 1858]
    │   ├→ WDL logits [batch, 128]
    │   └→ Illegality logits [batch, 1858]
    └→ LM head (if masking enabled) [batch, seq_len, vocab_size]
    ↓
5 loss components → weighted sum → backward
```

### Inference Pipeline (play.py, evaluation_puzzle.py)

```
FEN string
    ↓ (tokenizer.process_fen)
Space-delimited tokens
    ↓ (tokenizer.encode)
input_ids tensor
    ↓ (model.forward)
policy_logits [1858]
    ↓ (apply legal move mask)
legal_logits subset
    ↓ (softmax → sample/argmax)
UCI move string
```

## Key Implementation Details

### Why Two Losses on Policy Head?

The policy head logits are used for **both**:
1. **Softmax loss**: Encourages picking the best move (cross-entropy on move distribution)
2. **Sigmoid loss**: Encourages correct win% for ALL moves (BCE on individual move values)

This dual supervision prevents the model from only learning "best move" while ignoring the quality ranking of alternatives - critical for producing meaningful policy distributions.

### WDL Head: Why 128 Bins?

Rather than directly regressing win%, the WDL head predicts a distribution over 128 bins [0.0, 0.01, 0.02, ..., 1.0]. This:
- Allows the model to express uncertainty (multimodal distributions)
- Provides richer gradients than regression
- Uses Huber loss on expected values (smooth, distance-aware)

### Illegality Head: Not Redundant

The illegality head is the **only** head that receives explicit training signal on illegal moves:
- **Policy head (softmax)**: Illegal moves get -1e9 logits but still participate in softmax normalization
- **Move winrate head (sigmoid)**: Illegal moves are completely masked out (line 322-323) - NO training signal
- **Illegality head**: Learns explicit binary classification (legal=1, illegal=0) with BCE loss

This means the illegality head is the only component that can learn to predict legality from position features alone, without relying on the implicit "these moves have bad values" signal.

### Masked Token Prediction

The model optionally masks 5% of board tokens to predict them (like BERT). This:
- Prevents the model from "forgetting" piece positions after attention pooling
- Maintains compatibility with downstream LLM integration
- Improves board state representations
- Only applied to maskable positions (board + castling, not turn/en passant)

Controlled by `MASKED_TOKEN_LOSS_WEIGHT` in `loss_weights.py`.

## File Relationships

**Core architecture**:
- `model.py`: Model definition, multi-task pooling, loss computation
- `tokenizer.py`: FEN → tokens conversion
- `policy_index.py`: Move → index mapping (1858 moves)
- `loss_weights.py`: Centralized loss weight configuration

**Data handling**:
- `data.py`: ChessPolicyCollator for batching
- `action_value_dataset.py`: Convert raw data to training format
- `data/create_dataset.py`: Build HuggingFace dataset from raw files
- `data/shard_data.py`: Partition large datasets for parallel processing

**Training & evaluation**:
- `train.py`: Main training loop with dynamic batch scaling
- `evaluation_puzzle.py`: Tactical puzzle solving, ELO estimation
- `evaluation_game.py`: Full game evaluation vs Stockfish
- `grpo_trainer.py`: Group Relative Policy Optimization (RL fine-tuning)
- `grpo_model.py`: Model modifications for GRPO

**Inference & interaction**:
- `play.py`: PyGame GUI for human play
- `generate_games.py`: Batch game generation for analysis

## Critical Architectural Decisions

1. **No causal masking**: Unlike GPT, this model uses bidirectional attention since chess positions don't have temporal causality (model.py:229-231)

2. **Learned positional embeddings**: Rather than sinusoidal encodings, 70 position-specific embeddings are learned (model.py:138)

3. **Shared policy logits**: Same logits used for softmax policy loss AND sigmoid win% loss - forces unified representation (model.py:264-333)

4. **Relative value encoding**: Training targets are win% differences from best move, not absolute values. This makes the policy loss formulation cleaner (model.py:286-304)

5. **Streaming dataset**: Uses HuggingFace datasets in streaming mode to handle 500M+ positions without loading into RAM (train.py)

## Typical Modification Patterns

**Changing model size**: Edit config in `train.py`:
```python
config = LlamaConfig(
    hidden_size=768,         # Model dimension
    num_hidden_layers=20,    # Depth
    num_attention_heads=8,   # Attention heads
)
```

**Adjusting loss weights**: Edit `loss_weights.py` with ratios `target_loss / observed_loss` to equalize loss scales.

**Adding new prediction head**:
1. Add to `task_output_dims` in `MultiTaskAttentionPooling` (model.py:143-149)
2. Add loss computation in `forward()` (model.py:233-498)
3. Add to `ChessPolicyValueOutput` dataclass (model.py:98-118)

**Dataset changes**: Modify `ChessPolicyCollator` (data.py) to handle new fields, update `action_value_dataset.py` for preprocessing.
