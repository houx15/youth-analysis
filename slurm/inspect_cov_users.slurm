#!/bin/bash
#SBATCH --job-name=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --output=job.%j.users.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Reconcile the analysis sample: scan every daily file in cleaned_weibo_cov/{year}
# reading only user_id + gender, then report unique users, gender composition, and
# users whose gender is inconsistent across days, against the design doc's
# 225,339 (173,503 F / 51,836 M).
# Output: analysis_data/schema_report/users_{year}.json
#
# Usage:
#   sbatch slurm/inspect_cov_users.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python inspect_cov_schema.py users 2020
