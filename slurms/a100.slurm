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

# CLI arguments (passed after -- in sbatch command)
# Usage: sbatch slurms/a100.slurm [branch] [run_name]
# Example: sbatch slurms/a100.slurm feature-branch my-experiment
BRANCH="${1:-master}"
RUN_NAME="${2:-}"

# Each job gets unique directory (SLURM_JOB_ID ensures no collisions)
CODE_LOCAL="$TMPDIR/Chessbot-$SLURM_JOB_ID"

echo "================================================"
echo "Cloning branch '$BRANCH' to local storage..."
echo "Job ID: $SLURM_JOB_ID"
echo "================================================"

# Clone specific branch directly into isolated temp directory
# Uses --single-branch for efficiency (only fetches needed branch)
git clone --branch "$BRANCH" --single-branch --depth 1 \
    "$HOME/personal/Chessbot" "$CODE_LOCAL"

echo "Clone complete. Branch: $(cd "$CODE_LOCAL" && git branch --show-current)"
echo "Commit: $(cd "$CODE_LOCAL" && git rev-parse --short HEAD)"

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
source .a100/bin/activate

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
