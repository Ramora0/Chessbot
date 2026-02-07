sbatch slurms/a100.slurm master wide-shallow --ffn-dim 2048 --depth 12
sbatch slurms/a100.slurm master more-heads --heads 24
sbatch slurms/a100.slurm master beta2 --beta2 0.98