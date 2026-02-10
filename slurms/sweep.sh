sbatch slurms/h100.slurm dynamic-temp dynamic-temp
sbatch slurms/h100.slurm master only-softmax --mate-logit-bonus 10
sbatch slurms/h100.slurm master mates-4p --mate-bucket 0.04