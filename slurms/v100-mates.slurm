#!/bin/bash
#SBATCH --job-name=gen-mates
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=128G
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

cd "$HOME/personal/Chessbot"

# Copy dataset to local storage
DATASET_SOURCE="/fs/scratch/PAS2836/lees_stuff/searchless_mates_hf_fixed"
DATASET_LOCAL="$TMPDIR/action_value"

echo "================================================"
echo "Copying dataset to local storage..."
echo "Source: $DATASET_SOURCE"
echo "Destination: $DATASET_LOCAL"
echo "================================================"

mkdir -p "$DATASET_LOCAL"
time cp -r "$DATASET_SOURCE/"* "$DATASET_LOCAL/"

echo "Dataset copy complete. Size:"
du -sh "$DATASET_LOCAL"

source .v100/bin/activate

export DATASET_PATH="$DATASET_LOCAL"
python finetune/generate_mates.py

echo "================================================"
echo "Generation complete. Cleaning up local data..."
echo "================================================"
rm -rf "$DATASET_LOCAL"