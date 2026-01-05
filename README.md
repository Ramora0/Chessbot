# Chessbot Training Setup

Complete setup guide for training a chess transformer from scratch.

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
NUM_THINKING_TOKENS = 0
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
