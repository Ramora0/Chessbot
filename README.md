# Chessbot: Search-Free Chess AI with Transformer Architecture

A transformer-based chess engine that plays at strong levels **without any search**, predicting optimal moves instantly through learned pattern recognition. Built on multi-task learning principles, this project serves as a foundational chess encoder for various downstream applications including position evaluation, multimodal LLMs, and chess understanding.

## Overview

This project implements a **searchless chess policy network** inspired by Google DeepMind's [Grandmaster-Level Chess Without Search](https://arxiv.org/html/2402.04494v1) research. Unlike traditional chess engines that rely on tree search algorithms (e.g., AlphaZero, Stockfish), this model uses a single forward pass through a transformer to directly predict move probabilities—trained against Stockfish's policy evaluations.

**Important Technical Distinction**: While the referenced paper's best model actually performs depth-1 search (requiring one forward pass per legal move), this implementation achieves true zero-search prediction with **a single forward pass total**, making it significantly more efficient for real-time inference.

### Key Features

- **Zero-Search Architecture**: Instant move prediction with no MCTS or minimax search required
- **Multi-Task Learning**: Simultaneously predicts move policy, win-draw-loss probabilities, and move-level win rates
- **Transformer-Based**: LLaMA-style architecture with custom attention pooling for chess-specific tasks
- **Advanced Training**: Supports both supervised learning and Group Relative Policy Optimization (GRPO) for fine-tuning
- **Efficient Dataset Pipeline**: Handles large-scale chess position datasets with sharding and streaming
- **Comprehensive Evaluation**: Built-in puzzle solving and ELO estimation during training
- **Interactive Play**: PyGame-based GUI for playing against the trained model

### Technical Highlights

**Architecture:**

- Custom tokenization scheme for chess positions (FEN notation)
- Multi-task attention pooling with shared K/V projections
- Separate prediction heads for policy (1958 moves), WDL (3-way), and move evaluation

**Training Infrastructure:**

- Mixed-precision training with gradient accumulation
- Streaming dataset support for massive corpora
- Checkpoint resumption and automatic experiment tracking (W&B)
- Dynamic batch scaling and learning rate scheduling
- Continuous ELO evaluation on puzzle datasets

**Engineering:**

- Modular codebase with clear separation of concerns
- Efficient data processing pipelines for multi-TB datasets
- Support for distributed training on HPC clusters
- Production-ready inference with batching and optimization

## Project Status

This is an ongoing research and development project aimed at building a comprehensive chess encoder for multiple downstream tasks including gameplay, position analysis, and integration with multimodal language models.

---

## Training Setup Guide

Complete setup guide for training the chess transformer from scratch.

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/Ramora0/Chessbot.git
cd Chessbot
pip install -r requirements.txt

# Train a model (configure train.py with your dataset path first)
python train.py

# Play against your trained model
python play.py
```

## Project Structure

```
├── model.py                    # Core transformer architecture with multi-task heads
├── train.py                    # Main training script with distributed support
├── grpo_trainer.py            # GRPO (reinforcement learning) fine-tuning
├── tokenizer.py               # Chess position tokenization (FEN to tokens)
├── policy_index.py            # Move indexing for policy predictions
├── data.py                    # Dataset utilities and collators
├── action_value_dataset.py    # Action-value dataset creation
├── evaluation_puzzle.py       # Puzzle solving evaluation and ELO estimation
├── evaluation_game.py         # Full game evaluation utilities
├── play.py                    # Interactive GUI for playing against the model
├── loss_weights.py            # Multi-task loss weight configuration
└── data/                      # Dataset processing scripts
    ├── create_dataset.py      # Convert raw data to HuggingFace format
    ├── shard_data.py          # Shard large datasets for parallel processing
    └── get_data.py            # Data fetching utilities
```

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with ~24GB VRAM (or reduce batch size)
- **Storage**: ~200GB for dataset + checkpoints
- **RAM**: 32GB+ recommended
- **OS**: Linux (tested on HPC clusters)

### Software Requirements

```bash
# Python 3.9+
python --version

# CUDA (for GPU support)
nvidia-smi
```

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Chessbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- `torch` - PyTorch
- `transformers` - Hugging Face transformers
- `datasets` - Hugging Face datasets
- `chess` - Chess library
- `wandb` - Experiment tracking
- `tokenizers`, `numpy`, `tqdm`, `matplotlib`, `pygame`

### 3. Set Up Weights & Biases (Optional but Recommended)

```bash
wandb login
```

Follow the prompts to authenticate with your W&B account.

## Preparing the Dataset

The training requires a dataset in HuggingFace format with chess positions and move evaluations.

### Option A: Use Existing Dataset

If you already have a processed dataset:

1. Place it in an accessible directory
2. Note the full path (e.g., `/path/to/action_value_sharded`)
3. Skip to "Configure Training"

### Option B: Build Dataset from Scratch

If you have raw `.bag` files from the searchless_chess project:

#### Step 1: Partition Raw Data into Shards

```bash
cd data
# Edit shard_data.py to set:
# - BAG_DIR: path to your .bag files
# - PARTITION_DIR: where to write shards
# - N_SHARDS: number of shards (default: 512)
python shard_data.py
```

This creates binary shard files (e.g., `shard-0000.bin`, `shard-0001.bin`, ...).

#### Step 2: Convert to HuggingFace Dataset

```bash
# Edit create_dataset.py to set:
# - PARTITION_DIR: path to shards from step 1
# - OUT_DIR: output directory for HF dataset
python create_dataset.py
```

This creates a HuggingFace dataset with schema:

```python
{
    "fen": str,              # Chess position in FEN notation
    "moves": List[str],      # List of moves in UCI format
    "p_win": List[float],    # Win probability for each move
}
```

#### Step 3: Reshard for Training (Optional)

```bash
cd ..
# Edit shard_av.py to set:
# - DATASET_PATH: path to dataset from step 2
# - OUTPUT_PATH: final output path
# - NUM_SHARDS: desired number of shards
python shard_av.py
```

### Dataset Requirements

Your final dataset should:

- Be in HuggingFace Dataset format (Arrow or Parquet files)
- Contain fields: `fen`, `moves`, `p_win`
- Be sharded for efficient parallel loading
- Have positions with move evaluations (win probabilities)

## Configure Training

Edit `train.py` and set these required parameters:

```python
# Where to save model checkpoints
OUTPUT_DIR = "outputs"

# Path to your prepared dataset
DATASET_PATH = "/path/to/your/action_value_sharded"

# Optional: Resume from a checkpoint
RESUME_FROM_CHECKPOINT = None  # or "./outputs/checkpoint-10000"

# Training hyperparameters (optional - defaults are good)
DROPOUT = 0.1
MAX_SEQ_LENGTH = 256
```

### Key Settings to Adjust:

**Batch Size** (in `train()` function):

```python
per_device_batch_size = 1024  # Reduce if out of memory (try 512, 256, 128)
```

**Model Size** (in `train()` function):

```python
config = LlamaConfig(
    hidden_size=768,         # Model dimension
    num_hidden_layers=20,    # Number of transformer layers
    num_attention_heads=8,   # Attention heads
    # Reduce these for smaller/faster models
)
```

**Evaluation** (optional):

```python
ELO_EVAL_STEPS = 4000  # How often to run puzzle evaluation (0 to disable)
EVAL_BATCH_SIZE = 4096  # Batch size for evaluation
```

## Run Training

### Start Training

```bash
python train.py
```

The script will:

1. Load the dataset in streaming mode
2. Initialize or load the model
3. Train with automatic checkpointing
4. Log metrics to Weights & Biases
5. Periodically evaluate on chess puzzles (if `data/puzzles.csv` exists)

### Resume from Checkpoint

If training stops, resume by editing `train.py`:

```python
RESUME_FROM_CHECKPOINT = "./outputs/checkpoint-45000"
```

Then run:

```bash
python train.py
```

## Model Architecture Details

### Transformer Backbone

- **Base Model**: LLaMA-style decoder-only transformer
- **Default Configuration**: 768 hidden dimensions, 20 layers, 8 attention heads
- **Context Length**: 256 tokens (72 for board representation)
- **Vocabulary**: Custom chess tokenizer for FEN positions

### Multi-Task Prediction Heads

1. **Policy Head**: Predicts probability distribution over 1,858 legal chess moves
2. **Win-Draw-Loss (WDL) Head**: Predicts game outcome (3-way classification)
3. **Move Win Rate Head**: Predicts win probability for each candidate move
4. **Illegality Head**: Learns to distinguish legal from illegal moves

### Training Methodology

- **Supervised Learning**: Train against Stockfish policy evaluations
- **Multi-Task Loss**: Weighted combination of policy, WDL, and value losses
- **GRPO Fine-Tuning**: Optional reinforcement learning phase for policy optimization
- **Curriculum Learning**: Dynamic batch scaling with learning rate adjustment

## Evaluation & Testing

```bash
# Evaluate model on puzzle dataset
python evaluation_puzzle.py

# Run evaluation games against Stockfish
python evaluation_game.py

# Interactive testing
python play.py
```

The evaluation suite includes:

- **Puzzle Solving**: Tests tactical understanding on chess puzzles
- **ELO Estimation**: Approximates playing strength through puzzle performance
- **Game Simulation**: Plays full games for comprehensive evaluation

## Future Directions

This project is designed as a foundational chess encoder for multiple downstream applications:

- **Multimodal Integration**: Embedding chess positions in LLM context
- **Position Understanding**: Fine-tuning for chess commentary and analysis
- **Transfer Learning**: Using learned representations for chess-related NLP tasks
- **Advanced RL**: Exploring self-play and more sophisticated policy optimization
- **Hybrid Systems**: Combining learned intuition with lightweight search

## Citations

This work is inspired by:

```bibtex
@article{ruoss2024grandmaster,
  title={Grandmaster-Level Chess Without Search},
  author={Ruoss, Anian and others},
  journal={arXiv preprint arXiv:2402.04494},
  year={2024},
  url={https://arxiv.org/html/2402.04494v1}
}
```

## License

This project is available for educational and research purposes.

---

**Note**: This is an active research project. Model architectures, training procedures, and performance characteristics are subject to ongoing experimentation and refinement.
