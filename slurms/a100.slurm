#!/bin/bash
#SBATCH --job-name=chessformer
#SBATCH --partition=nextgen
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

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

# Go to your code
cd "$HOME/personal/Chessbot"

# Your python env
source .a100/bin/activate

# Run training with local dataset path
export DATASET_PATH="$DATASET_LOCAL"
python train.py

echo "================================================"
echo "Training complete. Cleaning up local data..."
echo "================================================"
rm -rf "$DATASET_LOCAL"
