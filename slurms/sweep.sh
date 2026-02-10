sbatch slurms/a100.slurm dynamic-temp dynamic-temp --grad-accum 2
sbatch slurms/a100.slurm master only-softmax --mate-logit-bonus 10 --grad-accum 2
sbatch slurms/a100.slurm master mates-4p --mate-bucket 0.04 --grad-accum 2