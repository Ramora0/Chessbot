sbatch slurms/a100.slurm master wide-shallow --ffn-dim 1536 --depth 14
sbatch slurms/a100.slurm master more-heads --heads 16
sbatch slurms/a100.slurm master beta2 --beta2 0.98
sbatch slurms/a100.slurm master no-reg --weight-decay 0 --dropout 0

sbatch --time=12:00:00 slurms/h100.slurm master large --hidden-dim 1024 --ffn-dim 2048 --depth 14 --heads 16