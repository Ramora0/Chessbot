#!/bin/bash
#SBATCH --job-name=chessformer
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
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
time rsync -a --info=progress2 "$DATASET_SOURCE" "$DATASET_LOCAL"

echo "Dataset copy complete. Size:"
du -sh "$DATASET_LOCAL"

# Go to your code
cd "$HOME/personal/chessformer"

# Your python env
source .h100/bin/activate

# Run training with local dataset path
export DATASET_PATH="$DATASET_LOCAL"
python train.py

echo "================================================"
echo "Training complete. Cleaning up local data..."
echo "================================================"
rm -rf "$DATASET_LOCAL"