#!/bin/bash
#SBATCH --job-name=reconcile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=job.%j.reconcile.out

# Compare the new user table against the existing viz_data baseline.
# Output: analysis_data/reconciliation_2020.json
#
# Usage:
#   sbatch slurm/reconcile_baseline.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.reconcile_baseline run --year=2020
