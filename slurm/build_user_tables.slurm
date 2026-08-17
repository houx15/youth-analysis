#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# Tables C and D: aggregate the post and event shards into the user master table
# and the user-month panel. Loads all shards at once, hence the larger memory ask.
# Output: analysis_data/user_domain_2020.parquet, user_month_domain_2020.parquet
#
# Usage:
#   sbatch slurm/build_user_tables.slurm

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion

python -m gender_domain.build_user_tables build --year=2020
