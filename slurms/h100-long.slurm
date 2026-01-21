#!/bin/bash
#SBATCH --job-name=chessformer
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

# Optional: Set RUN_NAME environment variable before submitting
# export RUN_NAME="my-experiment"
: ${RUN_NAME:=""}

# Copy code to local storage FIRST (quick, locks in code version)
CODE_LOCAL="$TMPDIR/Chessbot"
echo "================================================"
echo "Copying code to local storage..."
echo "================================================"

# Copy current code state to local node
mkdir -p "$CODE_LOCAL"
cp -r "$HOME/personal/Chessbot/"* "$CODE_LOCAL/" 2>/dev/null || true
cp -r "$HOME/personal/Chessbot/".??* "$CODE_LOCAL/" 2>/dev/null || true

echo "Code copy complete."

# Original dataset location on shared storage
DATASET_SOURCE="/fs/scratch/PAS2836/lees_stuff/action_value"

# Local copy in node's temporary storage
DATASET_LOCAL="$TMPDIR/action_value"

echo "================================================"
echo "Copying dataset to local storage..."
echo "Source: $DATASET_SOURCE"
echo "Destination: $DATASET_LOCAL"
echo "================================================"

# Copy dataset to local node storage
mkdir -p "$DATASET_LOCAL"

time cp -r "$DATASET_SOURCE/"* "$DATASET_LOCAL/"

echo "Dataset copy complete. Size:"
du -sh "$DATASET_LOCAL"

# Go to local code copy
cd "$CODE_LOCAL"

# Your python env
source .h100/bin/activate

# Run training with local dataset path
export DATASET_PATH="$DATASET_LOCAL"
if [ -n "$RUN_NAME" ]; then
    python train.py --run-name "$RUN_NAME"
else
    python train.py
fi

echo "================================================"
echo "Training complete. Cleaning up local data..."
echo "================================================"
rm -rf "$DATASET_LOCAL"
rm -rf "$CODE_LOCAL"