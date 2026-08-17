#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# Compare the new user table against the existing viz_data baseline.
# Output: analysis_data/reconciliation_2020.json
#
# Usage:
#   sbatch slurm/reconcile_baseline.slurm

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion

python -m gender_domain.reconcile_baseline run --year=2020
