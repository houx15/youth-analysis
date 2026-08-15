#!/bin/bash
#SBATCH --job-name=userCD
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --output=job.%j.userCD.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Tables C and D: aggregate the post and event shards into the user master table
# and the user-month panel. Loads all shards at once, hence the larger memory ask.
# Output: analysis_data/user_domain_2020.parquet, user_month_domain_2020.parquet
#
# Usage:
#   sbatch slurm/build_user_tables.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.build_user_tables build --year=2020
