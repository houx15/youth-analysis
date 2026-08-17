#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# Reconcile the analysis sample: scan every daily file in cleaned_weibo_cov/{year}
# reading only user_id + gender, then report unique users, gender composition, and
# users whose gender is inconsistent across days, against the design doc's
# 225,339 (173,503 F / 51,836 M).
# Output: analysis_data/schema_report/users_{year}.json
#
# Usage:
#   sbatch slurm/inspect_cov_users.slurm

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion

python inspect_cov_schema.py users 2020
